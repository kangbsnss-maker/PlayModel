"""Context expansion preserves old behavior without claiming new learning."""
from dataclasses import replace
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch is not installed")

import torch

from playmodel.learning.recurrent_ppo import (
    ModelConfig, RecurrentActorCritic, RolloutBatch, load_checkpoint,
    migrate_build_state_checkpoint, ppo_update, save_checkpoint,
)


class CheckpointMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source, self.target = self.directory / "source.pt", self.directory / "expanded.pt"
        torch.manual_seed(241)
        self.original = RecurrentActorCritic(ModelConfig(hidden_size=32, visual_size=32,
                                                         candidate_hidden_size=16)).eval()
        self.source_metadata = {"full_run_manifest": "old-training/manifest.json",
                                "ppo": {"optimizer_steps": 44}, "training_performed": True}
        save_checkpoint(self.original, self.source, self.source_metadata)
        self.original_bytes = self.source.read_bytes()

    def migrate(self):
        report = migrate_build_state_checkpoint(self.source, self.target)
        model, metadata = load_checkpoint(self.target)
        return report, model, metadata

    def inputs(self):
        return (torch.rand(2, 3, 96, 96), torch.rand(2, 16) * 2 - 1,
                torch.tensor([0, 3]), torch.rand(2, 6, 16) * 2 - 1,
                torch.tensor([[True] * 9, [True] * 6 + [False] * 3]),
                torch.rand(2, 32) * 2 - 1)

    def test_exact_weight_copy_phase_relocation_and_immutable_lineage(self):
        report, model, metadata = self.migrate()
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(report["source_sha256"], hashlib.sha256(self.original_bytes).hexdigest())
        self.assertEqual(report["context_schema"], "brotato-observed-build-v2")
        self.assertEqual(model.config.context_dim, 64)
        self.assertEqual(ModelConfig().context_dim, 16)
        self.assertNotEqual(model.policy_version(), self.original.policy_version())
        self.assertFalse(metadata["training_performed"])
        self.assertFalse(metadata["warm_start"]["additional_features_trained"])
        self.assertFalse(metadata["warm_start"]["old_rollouts_reusable_for_ppo"])
        self.assertNotIn("full_run_manifest", metadata)
        self.assertNotIn("ppo", metadata)
        self.assertEqual(metadata["lineage"]["source_metadata"], self.source_metadata)
        before, after = self.original.state_dict(), model.state_dict()
        for name in before:
            if name != "memory.weight_ih":
                self.assertTrue(torch.equal(before[name], after[name]), name)
        visual = self.original.config.visual_size
        self.assertTrue(torch.equal(before['memory.weight_ih'][:, :visual + 16],
                                    after['memory.weight_ih'][:, :visual + 16]))
        self.assertEqual(int(torch.count_nonzero(after['memory.weight_ih'][:, visual + 16:visual + 64])), 0)
        self.assertTrue(torch.equal(before['memory.weight_ih'][:, visual + 16:],
                                    after['memory.weight_ih'][:, visual + 64:]))
        restored, old_metadata = load_checkpoint(self.source)
        self.assertEqual(restored.policy_version(), self.original.policy_version())
        self.assertEqual(old_metadata, self.source_metadata)

    def test_old_outputs_preserved_with_zero_or_observed_additional_inputs(self):
        _, model, _ = self.migrate()
        images, context, phase, candidates, mask, hidden = self.inputs()
        with torch.no_grad():
            expected = self.original.step(images, context, phase, candidates, mask, hidden)
            for extra in (torch.zeros(2, 48), torch.rand(2, 48) * 2 - 1):
                actual = model.step(images, torch.cat((context, extra), -1), phase, candidates, mask, hidden)
                for field in ('logits', 'value', 'next_hidden'):
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=1e-6, rtol=1e-5)

    def test_added_feature_columns_receive_gradient_and_can_change_policy(self):
        _, model, _ = self.migrate()
        images, context, phase, candidates, mask, hidden = self.inputs()
        known = torch.ones(2, 48)
        expanded = torch.cat((context, known), -1)
        output = model.step(images, expanded, phase, candidates, mask, hidden)
        loss = -output.logits.log_softmax(-1)[:, 0].sum() + output.value.square().sum()
        loss.backward()
        visual = model.config.visual_size
        added = model.memory.weight_ih.grad[:, visual + 16:visual + 64]
        self.assertTrue(torch.isfinite(added).all())
        self.assertGreater(float(added.abs().sum()), 0)
        # A synthetic gradient step is a test only, never checkpoint training evidence.
        with torch.no_grad():
            model.memory.weight_ih[:, visual + 16:visual + 64].add_(added, alpha=-.1)
            present = model.step(images, expanded, phase, candidates, mask, hidden)
            absent = model.step(images, torch.cat((context, torch.zeros_like(known)), -1), phase, candidates, mask, hidden)
        self.assertGreater(float((present.probabilities - absent.probabilities).abs().max()), 0)

    def test_old_rollout_cannot_train_migrated_policy_even_if_zero_padded(self):
        _, model, _ = self.migrate()
        images, context, phase, candidates, mask, hidden = self.inputs()
        with torch.no_grad():
            output = self.original.step(images, context, phase, candidates, mask, hidden)
            action = output.logits.argmax(-1)
            logs = output.logits.log_softmax(-1).gather(-1, action[:, None]).squeeze(-1)
        batch = RolloutBatch(images[None], context[None], phase[None], candidates[None], mask[None],
            action[None], logs[None], output.value[None], torch.ones(1, 2), torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 2), hidden, self.original.policy_version(), 'legacy-synthetic-rollout')
        version = model.policy_version()
        for old in (batch, replace(batch, context=torch.cat((batch.context, torch.zeros(1, 2, 48)), -1))):
            with self.assertRaisesRegex(ValueError, 'off-policy'):
                ppo_update(model, old)
        self.assertEqual(model.policy_version(), version)

    def test_unsupported_context_and_existing_target_refuse_without_overwrite(self):
        for context in (4, 32, 64):
            wrong = self.directory / f'context-{context}.pt'
            save_checkpoint(RecurrentActorCritic(replace(self.original.config, context_dim=context)), wrong)
            with self.assertRaisesRegex(ValueError, 'context16'):
                migrate_build_state_checkpoint(wrong, self.target)
            self.assertFalse(self.target.exists())
        with self.assertRaises(FileExistsError):
            migrate_build_state_checkpoint(self.source, self.source)
        self.target.write_bytes(b'preserved-existing-target')
        with self.assertRaises(FileExistsError):
            migrate_build_state_checkpoint(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b'preserved-existing-target')
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_cli_creates_explicit_target_and_reports_existing_target_failure(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/migrate_build_state_model.py'
        spec = importlib.util.spec_from_file_location('migration_cli', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for expected in (0, 1):
            captured = io.StringIO()
            with redirect_stdout(captured):
                self.assertEqual(module.main([str(self.source), str(self.target)]), expected)
            report = json.loads(captured.getvalue())
            self.assertFalse(report['training_performed'])
            self.assertEqual(report['status'], 'warm_start_created' if expected == 0 else 'migration_failed')
        self.assertEqual(self.source.read_bytes(), self.original_bytes)


if __name__ == '__main__':
    unittest.main()
