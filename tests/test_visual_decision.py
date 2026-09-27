"""Visual gradients, candidate sensitivity, causal replay, and frozen caching."""
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')
import torch
from torch import nn
from playmodel.learning.visual_decision import VisualDecisionContext
from playmodel.laya.forward import ChoiceForward


class VisualDecisionTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        torch.manual_seed(5)

    def test_zero_migration_then_visual_and_temporal_learning_and_reload(self):
        model = VisualDecisionContext(16)
        frames = torch.randint(0, 256, (1, 3, 3, 96, 96), dtype=torch.uint8)
        candidates = torch.randn(1, 4, 16, requires_grad=True)
        self.assertEqual(model(frames, candidates).abs().sum().item(), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for index in range(2):
            optimizer.zero_grad()
            loss = -model(frames, candidates).log_softmax(-1)[0, 0]
            loss.backward()
            if index == 0:
                self.assertEqual(model.encoder[0].weight.grad.abs().sum().item(), 0)
            else:
                self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.temporal.weight_ih_l0.grad.abs().sum().item(), 0)
                self.assertGreater(candidates.grad.abs().sum().item(), 0)
            optimizer.step()
        for prefix in ('encoder.', 'temporal.', 'project.'):
            self.assertTrue(any(not torch.equal(before[key], value)
                                for key, value in model.state_dict().items() if key.startswith(prefix)))
        actual = model(frames, candidates).detach()
        self.assertFalse(torch.equal(actual, model(torch.zeros_like(frames), candidates)))
        self.assertGreater(actual.std().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            restored = VisualDecisionContext(16)
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(actual, restored(frames, candidates), rtol=0, atol=0)

    def test_cache_never_caches_trainable_head_or_retains_gradients(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(20, 16)
                self.calls = 0
            def forward(self, input_ids, attention_mask):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        model = nn.Module()
        model.encoder = Encoder()
        model.encoder.requires_grad_(False)
        model.type_emb = nn.Embedding(3, 16)
        model.head = None
        model.scorer = nn.Linear(16, 1)
        model.eval()
        engine = ChoiceForward(model, max_bytes=256)
        tensors = {'input_ids': torch.tensor([[1, 2, 3, 4]]), 'attention_mask': torch.ones(1, 4),
                   'qtype': torch.tensor([0]), 'marker_pos': torch.tensor([[1, 3]]),
                   'marker_mask': torch.tensor([[True, True]])}
        first = engine(tensors, key='same')
        first.sum().backward()
        with torch.no_grad():
            model.scorer.weight.add_(.1)
        second = engine(tensors, key='same')
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(model.encoder.calls, 1)
        self.assertIsNone(model.encoder.embedding.weight.grad)
        self.assertTrue(all(not value.requires_grad for value in engine.cache.values()))
        engine(tensors, key='other')
        self.assertEqual(list(engine.cache), ['other'])
        self.assertLessEqual(engine.bytes, engine.max_bytes)

    def test_window_rejects_changed_source_and_future_observation(self):
        import hashlib
        from playmodel.laya.visual import validate_window, tensor_window
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'visual.rgb'
            raw = bytes(3 * 96 * 96)
            path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            row = {'path': str(path), 'sha256': sha, 'source_path': str(path),
                   'source_sha256': sha, 'observed_at_ns': 10, 'available_at_ns': 20,
                   'transform': 'bgra_to_rgb_area96_round_uint8_v1'}
            evidence = {'frame_sha256': sha, 'observed_at_ns': 10, 'available_at_ns': 20}
            validate_window([row], evidence)
            self.assertEqual(tuple(tensor_window([row], 'cpu').shape), (1, 1, 3, 96, 96))
            with self.assertRaises(ValueError):
                validate_window([{**row, 'available_at_ns': 21}], evidence)
            with self.assertRaises(ValueError):
                validate_window([row, row], evidence)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_window([row], evidence)


if __name__ == '__main__':
    unittest.main()
