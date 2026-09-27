"""Human preferences never become reward labels or silently modify a live policy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

from playmodel.laya.preferences import (default_preferences, load_preferences,
    preference_hash, validate_preferences, checkpoint_preferences)
from playmodel.laya.worker import Learner


class PreferenceTests(unittest.TestCase):
    def test_checkpoint_report_preferences_must_match_or_both_be_legacy(self):
        self.assertEqual(checkpoint_preferences({'report': {}}), default_preferences())
        value = default_preferences()
        value.update(revision=1, reason='saved preference')
        fields = {'preferences': value, 'preference_hash': preference_hash(value)}
        bundle = {**deepcopy(fields), 'report': deepcopy(fields)}
        self.assertEqual(checkpoint_preferences(bundle), value)
        broken = deepcopy(bundle)
        broken['report']['preferences']['values']['retreat'] = .5
        with self.assertRaises(ValueError):
            checkpoint_preferences(broken)
        broken = deepcopy(bundle)
        broken['report'].pop('preference_hash')
        with self.assertRaises(ValueError):
            checkpoint_preferences(broken)
        with self.assertRaises(ValueError):
            checkpoint_preferences({**fields, 'report': {}})

    def test_default_is_deterministic_and_zero(self):
        with tempfile.TemporaryDirectory() as root:
            document = load_preferences(Path(root) / 'missing.json')
        self.assertEqual(document, default_preferences())
        self.assertTrue(all(v == 0 for v in document['values'].values()))
        self.assertEqual(preference_hash(document), preference_hash(default_preferences()))

    def test_bad_values_unknown_actions_and_empty_saved_reason_rejected(self):
        for bad in (True, float('nan'), float('inf'), -2.01, 2.01, '1'):
            doc = default_preferences()
            doc['values']['retreat'] = bad
            with self.assertRaises(ValueError):
                validate_preferences(doc)
        doc = default_preferences()
        doc['values']['shoot'] = 1
        with self.assertRaises(ValueError):
            validate_preferences(doc)
        doc = default_preferences()
        doc['revision'] = 1
        with self.assertRaises(ValueError):
            validate_preferences(doc)

    def test_reason_preserved_and_hash_changes(self):
        doc = default_preferences()
        doc.update(revision=1, reason='  사용자 원문\n후퇴를 선호  ')
        doc['values']['retreat'] = 2
        value = validate_preferences(doc)
        self.assertEqual(value['reason'], doc['reason'])
        self.assertNotEqual(preference_hash(value), preference_hash(default_preferences()))

    def test_apply_only_empty_boundary_and_preserves_immutable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = Learner.__new__(Learner)
            learner.output, learner.base_hash = Path(directory), 'base'
            learner.graph_hash = 'synthetic-graph'
            learner.head_hash = lambda: 'unchanged-head'
            learner.preference_resume = 'synthetic_test'
            learner.preferences = default_preferences()
            learner.preference_hash = preference_hash(learner.preferences)
            learner.version = learner._version()
            initial = learner.version
            learner.pending, learner.accepted = {'d': {}}, []
            doc = default_preferences()
            doc.update(revision=1, reason='prefer spacing')
            doc['values']['hold_distance'] = .5
            with self.assertRaises(ValueError):
                learner.configure_preferences(doc)
            learner.pending, learner.accepted = {}, [{}]
            with self.assertRaises(ValueError):
                learner.configure_preferences(doc)
            learner.accepted = []
            result = learner.configure_preferences(doc)
            self.assertNotEqual(result['behavior_version'], initial)
            self.assertEqual(learner.head_hash(), 'unchanged-head')
            stored = json.loads(next(Path(directory).glob('preferences-applied-*.json')).read_text())
            self.assertEqual(stored['preferences']['reason'], doc['reason'])
            self.assertFalse(stored['reward_source'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch unavailable')
    def test_float32_logit_bias_actor_and_training_replay_match(self):
        import torch
        learner = Learner.__new__(Learner)
        learner.torch = torch
        logits = torch.tensor([.2, -.3], requires_grad=True, dtype=torch.float32)
        learner.logits = lambda item: logits
        with torch.no_grad():
            actor, raw = learner.probs({}, [1., -1.], return_raw=True)
            chosen_logp = actor[0].log()
        replay = learner.probs({}, [1., -1.])
        self.assertTrue(torch.equal(actor, replay))
        self.assertTrue(torch.equal(raw, torch.softmax(logits, -1)))
        self.assertGreater(float(actor[0]), float(raw[0]))
        self.assertAlmostEqual(float(replay[0].log()), float(chosen_logp), places=7)
        (-replay[0].log()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(learner.probs({}, [0., 0.]), raw))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch unavailable')
    def test_head_summary_excludes_encoder_and_measures_actual_delta(self):
        import torch
        learner = Learner.__new__(Learner)
        learner.torch = torch
        learner.model = torch.nn.Module()
        learner.model.encoder = torch.nn.Linear(2, 2, bias=False)
        learner.model.head = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            learner.model.head.weight.copy_(torch.tensor([[3., 4.]]))
        before = {'head.weight': torch.tensor([[0., 0.]])}
        result = learner.head_summary(before)
        self.assertEqual(result['parameter_count'], 2)
        self.assertEqual(result['trainable_parameter_count'], 2)
        self.assertEqual(result['head_l2_norm'], 5.)
        self.assertEqual(result['head_delta_l2_norm'], 5.)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch unavailable')
    def test_cuda_head_summary_with_cpu_snapshot(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA runtime required; separately exercised in .venv-laya')
        learner = Learner.__new__(Learner)
        learner.torch = torch
        learner.model = torch.nn.Module()
        learner.model.head = torch.nn.Linear(2, 1, bias=False, device='cuda')
        with torch.no_grad():
            learner.model.head.weight.copy_(torch.tensor([[3., 4.]], device='cuda'))
        before = {'head.weight': torch.zeros((1, 2), device='cpu')}
        result = learner.head_summary(before)
        self.assertEqual(result['head_l2_norm'], 5.)
        self.assertEqual(result['head_delta_l2_norm'], 5.)
        self.assertEqual(before['head.weight'].device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
