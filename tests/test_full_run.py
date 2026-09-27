"""Synthetic full-run joining tests; no game input or gameplay claims."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch unavailable")
import torch

from playmodel.learning.full_run import FullRunRecorder, load_full_run
from playmodel.learning.runtime_contract import (
    RUNTIME_CONTRACT, LEGACY_RUNTIME_CONTRACT, LEGACY_PHASE_SCHEMA, contract_fields,
)
from playmodel.learning.recurrent_ppo import (
    ModelConfig, PPOConfig, RecurrentActorCritic, _advantage_targets, ppo_update,
)


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(33)
        self.model = RecurrentActorCritic(ModelConfig(context_dim=4, candidate_dim=3,
            hidden_size=32, visual_size=32, candidate_hidden_size=16))
        self.recorder = FullRunRecorder(self.model, "synthetic-whole-run", session_ids=("synthetic-session",))

    def test_currency_derivation_binds_source_and_derived_files(self):
        from playmodel.learning.full_run import _currency_sources
        frame = self.root/'frame.png'
        frame.write_bytes(b'original frame')
        derived = self.root/'currency-roi.png'
        derived.write_bytes(b'derived repeated digits')
        evidence = {'source_path': str(frame),
                    'source_png_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
                    'evidence_files': [[str(derived), hashlib.sha256(derived.read_bytes()).hexdigest()]]}
        (self.root/'currency-roi.json').write_text(json.dumps(evidence), encoding='utf-8')
        self.assertEqual(len(_currency_sources(frame)), 2)
        derived.write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'derivation changed'):
            _currency_sources(frame)
        frame.write_bytes(b'another frame')
        with self.assertRaisesRegex(ValueError, 'another source'):
            _currency_sources(frame)

    def source(self, name, observed):
        path = self.root / name
        path.write_bytes(name.encode())
        return {"frame_ref": str(path), "frame_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "observed_at_ns": observed, "available_at_ns": observed + 1,
                "verified_at_ns": observed + 2, "verified": True,
                "independent_of_policy": True, "verifier_id": "synthetic-verifier"}

    def decision(self, phase, index, count=3):
        recorder = self.recorder
        data = {"images": torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8),
                "context": torch.zeros(1, 4), "phase": torch.tensor([phase]),
                "candidates": torch.rand(1, count, 3),
                "legal_mask": torch.tensor([[True] * (9 if phase == 0 else count) +
                                              [False] * (max(9, count) - (9 if phase == 0 else count))]),
                "hidden_before": recorder.hidden.clone(), "reset": torch.tensor([not recorder.records])}
        with torch.no_grad():
            output = recorder.model.step(*(data[key] for key in
                ("images", "context", "phase", "candidates", "legal_mask")),
                hidden=data["hidden_before"], reset=data["reset"])
            action, logp = output.sample(generator=torch.Generator().manual_seed(index))
        data.update(next_hidden=output.next_hidden, actions=action,
                    old_log_probs=logp, old_values=output.value)
        observed = (index + 1) * 1_000_000_000
        evidence = self.source(f"frame-{index}", observed)
        evidence.update(decided_at_ns=observed + 3, sent_at_ns=observed + 4,
                        actual_action=int(action.item()), action_origin="policy", transmitted=True,
                        acknowledged=None, game_application_verified=phase != 0)
        return data, evidence

    def collect(self):
        for index, phase in enumerate((0, 1, 3, 0)):
            data, evidence = self.decision(phase, index, count=1 if phase == 0 else 3)
            self.recorder.append_decision(data, evidence=evidence)

    def finish(self, *, kind="death", extra=None):
        end = self.source("ending", 10_000_000_000)
        end["kind"] = kind
        return self.recorder.finish(self.root / "frozen", kind=kind, evidence=end,
            chunk_steps=2, burn_in=1, **(extra or {}))

    def test_menu_combat_seams_and_full_scalar_gae(self):
        self.collect()
        report = self.finish()
        chunks, flat, indices, manifest = load_full_run(report["manifest_path"])
        self.assertEqual(flat.phase[:, 0].tolist(), [0, 1, 3, 0])
        self.assertEqual(flat.terminated[:, 0].tolist(), [False, False, False, True])
        self.assertFalse(flat.truncated.any())
        self.assertTrue(torch.equal(flat.next_values[:-1], flat.old_values[1:]))
        self.assertEqual(manifest["phase_counts"], {"0": 2, "1": 1, "3": 1})
        self.assertFalse(manifest["full_run_complete"])  # starting a recorder is not proof of a new run
        config = PPOConfig(epochs=1, minibatch_sequences=2, burn_in=1, gamma=1, gae_lambda=1)
        learning = chunks.valid.clone()
        learning[:1] = False
        _, returns, details = _advantage_targets(self.recorder.model, chunks, config, learning, flat, indices)
        self.assertTrue(torch.allclose(returns[learning], torch.full((4,), -1.), atol=1e-6))
        self.assertEqual(details["gae_scope"], "full_chronological_trajectory")
        updated = ppo_update(self.recorder.model, chunks, config, trajectory=flat, transition_indices=indices)
        self.assertEqual(updated["optimized_transitions"], 4)
        self.assertEqual(updated["gae_scope"], "full_chronological_trajectory")

    def test_legacy_frozen_rollout_is_readable_but_cannot_train_new_runtime(self):
        self.collect()
        report = self.finish()
        path = Path(report['manifest_path'])
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['phase_schema'] = LEGACY_PHASE_SCHEMA
        manifest.pop('runtime_contract')
        # Build a legacy-format synthetic fixture only; no real evidence changes.
        for filename in ('flat.pt', 'rollout.pt'):
            payload = torch.load(path.parent / filename, weights_only=True)
            payload['batch'].pop('runtime_contract')
            torch.save(payload, path.parent / filename)
        for row in manifest['files']:
            row['sha256'] = hashlib.sha256((path.parent / row['path']).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest), encoding='utf-8')
        chunks, _, _, legacy = load_full_run(path, require_training=False)
        self.assertEqual(chunks.runtime_contract, LEGACY_RUNTIME_CONTRACT)
        self.assertNotIn('runtime_contract', legacy)
        with self.assertRaisesRegex(ValueError, 'legacy runtime'):
            load_full_run(path)
        with self.assertRaisesRegex(ValueError, 'incompatible runtime'):
            ppo_update(self.recorder.model, chunks, PPOConfig(burn_in=1))
        with self.assertRaisesRegex(ValueError, 'required scope'):
            load_full_run(path, require_training=False, expected_runtime_contract=RUNTIME_CONTRACT)

    def test_source_index_duplication_and_chunk_tampering_rejected(self):
        self.collect()
        report = self.finish()
        chunks, flat, indices, _ = load_full_run(report["manifest_path"])
        config = PPOConfig(epochs=1, burn_in=1)
        invalid_indices = indices.clone()
        invalid_indices[1, 1] = 0
        with self.assertRaisesRegex(ValueError, "exactly once"):
            ppo_update(self.recorder.model, chunks, config, trajectory=flat, transition_indices=invalid_indices)
        rewards = chunks.rewards.clone()
        rewards[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "differs from full trajectory"):
            ppo_update(self.recorder.model, replace(chunks, rewards=rewards), config,
                       trajectory=flat, transition_indices=indices)

    def test_hidden_discontinuity_and_unknown_menu_application_rejected(self):
        data, evidence = self.decision(0, 0)
        self.recorder.append_decision(data, evidence=evidence)
        data, evidence = self.decision(1, 1)
        evidence["game_application_verified"] = False
        with self.assertRaisesRegex(ValueError, "unknown menu"):
            self.recorder.append_decision(data, evidence=evidence)
        evidence["game_application_verified"] = True
        data["hidden_before"] = torch.zeros_like(data["hidden_before"])
        with self.assertRaisesRegex(ValueError, "hidden-state seam"):
            self.recorder.append_decision(data, evidence=evidence)
        self.assertEqual(len(self.recorder.records), 1)

    def test_full_run_manifest_binds_original_sources(self):
        self.collect()
        report = self.finish()
        (self.root / "frame-0").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "evidence digest mismatch"):
            load_full_run(report["manifest_path"])

    def test_unknown_bootstrap_and_guard_failure_are_not_training_runs(self):
        self.collect()
        self.recorder.invalidate("synthetic human takeover")
        report = self.finish(kind="truncated")
        self.assertFalse(report["training_eligible"])
        self.assertIn("true final-observation bootstrap missing", report["rejection_reasons"])
        with self.assertRaisesRegex(ValueError, "not eligible"):
            load_full_run(report["manifest_path"])
        _, flat, _, _ = load_full_run(report["manifest_path"], require_training=False)
        self.assertTrue(flat.truncated[-1, 0])

    def test_abort_preserves_unfinished_records_without_training_rollout(self):
        self.collect()
        report = self.recorder.abort(self.root / "aborted", "no fresh final observation")
        self.assertFalse(report["training_eligible"])
        self.assertTrue((self.root / "aborted" / "partial-records.pt").exists())
        self.assertFalse((self.root / "aborted" / "rollout.pt").exists())
        with self.assertRaisesRegex(ValueError, "not eligible"):
            load_full_run(report["manifest_path"])

    def test_cycle_collects_trains_then_evaluates_both_without_promoting(self):
        specification = importlib.util.spec_from_file_location("cycle_test_module",
            Path(__file__).resolve().parents[1] / "scripts/run_recurrent_cycle.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        calls = []
        def collect(checkpoint, *, split, tag):
            calls.append(("collect", checkpoint, split))
            return {"run_id": tag, "session_ids": [tag + "-session"],
                    **contract_fields(), 'runtime_source_hashes': {},
                    "full_run_complete": True, "training_eligible": split == "train",
                    "verified_wave_clears": 2, "setup_conditions": {"character_slot": 1}}
        def train(checkpoint, training):
            calls.append(("train", checkpoint, training["run_id"]))
            return {"checkpoint": "candidate.pt", "final_kl_within_target": True}
        result = module.run_pipeline("source.pt", collect_run=collect, train_candidate=train)
        self.assertEqual(calls, [("collect", "source.pt", "train"),
                                ("train", "source.pt", "training"),
                                ("collect", "source.pt", "evaluation"),
                                ("collect", "candidate.pt", "evaluation")])
        self.assertEqual(result["status"], "comparison_recorded")
        self.assertFalse(result["deployment_approved"])
        self.assertFalse(result["performance_improvement_verified"])
        rejected = module.run_pipeline("source.pt", collect_run=lambda *a, **k: {
            **contract_fields(),
            "run_id": "incomplete", "training_eligible": True, "full_run_complete": False},
            train_candidate=lambda *a: self.fail("incomplete full run must not enter training"))
        self.assertEqual(rejected["status"], "training_run_incomplete_or_rejected")

    def test_local_status_heartbeats_do_not_claim_progress_or_call_agents(self):
        specification = importlib.util.spec_from_file_location("cycle_status_test_module",
            Path(__file__).resolve().parents[1] / "scripts/run_recurrent_cycle.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        status = module.LocalStatus(self.root / "status", interval_seconds=60)
        self.addCleanup(status.close)
        status.update(status="running", phase="combat")
        progress = status.state["last_progress_monotonic_ns"]
        status.update(progress=False)
        self.assertEqual(progress, status.state["last_progress_monotonic_ns"])
        self.assertFalse(status.state["automatic_agent_call"])
        status.fault("synthetic fault")
        import json
        request = json.loads((self.root / "status/help-request.json").read_text())
        self.assertFalse(request["automatic_agent_call"])
        self.assertFalse(request["restart_attempted"])
        self.assertTrue(status.state["needs_agent_help"])

    def test_continuous_mode_honors_local_stop_and_prints_small_summary(self):
        import contextlib
        import io
        import json
        specification = importlib.util.spec_from_file_location("cycle_continuous_test_module",
            Path(__file__).resolve().parents[1] / "scripts/run_recurrent_cycle.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        class FakeCycle:
            def __init__(self, **kwargs):
                self.directory = Path(kwargs['output'])
                self.runtime_source_hashes = module._runtime_contract(kwargs['root'])
                self.stop_file = Path(kwargs["output"]) / "synthetic-stop"
            def collect_run(self, checkpoint, *, split, tag):
                if tag == "evaluation-0-candidate":
                    self.stop_file.write_text("synthetic user stop")
                return {"run_id": tag, "session_ids": [tag], "full_run_complete": True,
                        **contract_fields(), 'runtime_source_hashes': self.runtime_source_hashes,
                        "training_eligible": split == "train", "verified_wave_clears": 1}
            def train_candidate(self, checkpoint, training):
                candidate = self.directory / 'synthetic-candidate.pt'
                candidate.write_bytes(b'not loaded by fake collector')
                return {"checkpoint": str(candidate), "final_kl_within_target": True}
        module.LocalCycle = FakeCycle
        checkpoint = self.root / "synthetic-checkpoint"
        checkpoint.write_bytes(b"not loaded by fake orchestration")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = module.main([str(checkpoint), "--continuous", "--device", "cpu",
                                "--output", str(self.root / "continuous")])
        summary = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "user_stopped")
        self.assertEqual(summary["completed_comparisons"], 1)
        self.assertFalse(summary["automatic_agent_call"])
        self.assertFalse(summary["help_requested"])
        self.assertLess(len(output.getvalue()), 1500)


class ResumeOrchestrationTests(unittest.TestCase):
    """Crash boundaries and mocked collectors; never create live game input."""
    def setUp(self):
        from unittest.mock import Mock
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        specification = importlib.util.spec_from_file_location('cycle_resume_test_module',
            Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py')
        self.module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.module)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'source frozen for orchestration fixture')
        self.candidate = self.root / 'candidate.pt'
        self.candidate.write_bytes(b'candidate frozen for orchestration fixture')
        self.training = {'run_id': 'training-run', 'session_ids': ['training-session'],
                         **contract_fields(), 'runtime_source_hashes': {},
                         'split': 'train', 'full_run_complete': True, 'training_eligible': True,
                         'setup_conditions': {'character_slot': 1}, 'verified_wave_clears': 2}
        self.train = Mock(return_value={'checkpoint': str(self.candidate), 'final_kl_within_target': True})

    def evaluation(self, tag, *, suffix='fresh', complete=True):
        return {'run_id': tag + '-' + suffix, 'session_ids': [tag + '-' + suffix + '-session'],
                **contract_fields(), 'runtime_source_hashes': {},
                'split': 'evaluation', 'full_run_complete': complete, 'training_eligible': False,
                'setup_conditions': {'character_slot': 1}, 'verified_wave_clears': 2 if complete else 999}

    def test_crash_after_training_resumes_evaluation_without_retraining(self):
        journal = self.module.CycleState(self.root / 'state', {'checkpoint': str(self.source)})
        def collect(checkpoint, *, split, tag):
            if split == 'train':
                return self.training
            raise RuntimeError('simulated process interruption')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.module.run_pipeline(self.source, collect_run=collect, train_candidate=self.train,
                on_progress=lambda report: journal.commit(pipeline=report))
        restored = self.module.CycleState(journal.directory)
        pending = restored.value['pipeline']['pending']
        self.assertEqual(pending['role'], 'source')
        calls, checkpoints = [], []
        def resumed(checkpoint, *, split, tag):
            self.assertEqual(split, 'evaluation')
            calls.append(tag)
            return self.evaluation(tag)
        result = self.module.run_pipeline(self.source, collect_run=resumed,
            train_candidate=lambda *args: self.fail('completed training must not run twice'),
            resume_report=restored.value['pipeline'], on_progress=checkpoints.append)
        self.assertEqual(result['status'], 'comparison_recorded')
        self.assertEqual(calls, ['evaluation-0-source', 'evaluation-0-candidate'])
        self.assertEqual(checkpoints[0]['pending']['operation_id'], pending['operation_id'])
        self.train.assert_called_once()

    def test_resumed_failure_reports_current_cause_not_old_recovery_error(self):
        resumed = {'source_checkpoint': str(self.source), 'training': self.training,
                   'candidate': self.train.return_value, 'evaluations': [],
                   'error': 'Old recovery failed'}
        result = self.module.run_pipeline(self.source,
            collect_run=lambda checkpoint, split, tag: {
                **self.evaluation(tag, complete=False), 'error': 'policy_deadline'},
            train_candidate=lambda *args: self.fail('must reuse saved candidate'),
            resume_report=resumed)
        self.assertEqual(result['status'], 'evaluation_run_incomplete')
        self.assertEqual(result['error'], 'policy_deadline')
        self.assertEqual(resumed['error'], 'Old recovery failed')

    def test_incomplete_evaluation_is_preserved_excluded_and_replaced(self):
        first = self.module.run_pipeline(self.source,
            collect_run=lambda checkpoint, split, tag: self.training if split == 'train'
                else self.evaluation(tag, suffix='failed', complete=False), train_candidate=self.train)
        self.assertEqual(first['status'], 'evaluation_run_incomplete')
        result = self.module.run_pipeline(self.source,
            collect_run=lambda checkpoint, split, tag: self.evaluation(tag),
            train_candidate=lambda *args: self.fail('evaluation data must not update weights'), resume_report=first)
        self.assertEqual(len(result['excluded_evaluations']), 1)
        self.assertEqual(result['excluded_evaluations'][0]['verified_wave_clears'], 999)
        self.assertEqual(result['mean_verified_wave_clears'], {'source': 2., 'candidate': 2.})

    def test_saved_candidate_hash_change_blocks_resume(self):
        journal = self.module.CycleState(self.root / 'state', {'checkpoint': str(self.source),
            'pipeline': {'source_checkpoint': str(self.source),
                         'candidate': {'checkpoint': str(self.candidate)}}})
        journal.commit(pipeline={**journal.value['pipeline'], 'pipeline_phase': 'collect_evaluation'})
        self.assertEqual(len(list((journal.directory / 'state-history').glob('*.json'))), 2)
        self.candidate.write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'persisted cycle file changed'):
            self.module.CycleState(journal.directory)

    def test_finished_collection_is_recovered_after_intent_only_crash(self):
        from unittest.mock import patch, Mock
        coordinator = object.__new__(self.module.LocalCycle)
        coordinator.output, coordinator.operation_id = self.root, 'persisted-intent'
        coordinator.root = Path(__file__).resolve().parents[1]
        coordinator.runtime_source_hashes = self.module._runtime_contract(coordinator.root)
        coordinator.evaluation_scope_id = 'current-scope'
        directory = self.root / 'training-finished'
        directory.mkdir()
        self.module._save_json(directory / 'run-operation.json', {
            **contract_fields(), 'runtime_source_hashes': coordinator.runtime_source_hashes,
            'evaluation_scope_id': coordinator.evaluation_scope_id,
            'operation_id': 'persisted-intent', 'checkpoint': str(self.source), 'split': 'train', 'partial': False})
        recovered = {**self.training, 'checkpoint': str(self.source), 'manifest_path': 'fixture-manifest',
                     'runtime_source_hashes': coordinator.runtime_source_hashes,
                     'evaluation_scope_id': coordinator.evaluation_scope_id}
        self.module._save_json(directory / 'cycle-run.json', recovered)
        coordinator._new_run = Mock(side_effect=AssertionError('completed run must not restart'))
        with patch('playmodel.learning.full_run.load_full_run', return_value=(None, None, None, {})) as check:
            self.assertEqual(coordinator.collect_run(self.source, split='train', tag='training'), recovered)
        check.assert_called_once_with('fixture-manifest', require_training=False,
                                      expected_runtime_contract=RUNTIME_CONTRACT)
        coordinator._new_run.assert_not_called()

    def test_partial_recovery_cannot_be_training_or_skip_failed_ending(self):
        from unittest.mock import Mock, patch
        coordinator = object.__new__(self.module.LocalCycle)
        with self.assertRaisesRegex(ValueError, 'partial recovery'):
            coordinator.collect_run(self.source, split='train', tag='recovery', partial=True)
        coordinator.stop_file = self.root / 'STOP'
        coordinator.recover_active_run = True
        for scene in ('level_up', 'loot'):
            with self.subTest(scene=scene):
                directory = self.root / scene
                directory.mkdir()
                coordinator._current_scene = Mock(return_value=scene)
                coordinator.collect_run = Mock(return_value={'recovery_completed': False, 'training_eligible': False})
                with patch('playmodel.games.brotato.setup_run.prepare_next') as setup:
                    with self.assertRaisesRegex(OSError, 'verified normal ending'):
                        coordinator._new_run(directory, self.source)
                    setup.assert_not_called()
                coordinator.collect_run.assert_called_once_with(self.source, split='evaluation', tag='partial-recovery', partial=True)

    def test_startup_combat_uses_original_capture_without_a_second_downsample(self):
        from unittest.mock import Mock, patch
        coordinator = object.__new__(self.module.LocalCycle)
        coordinator.stop_file = self.root / 'STOP'
        coordinator.executable, coordinator.ocr_script = self.root / 'unused.exe', self.root / 'unused.ps1'
        original_pixels = b'original capture pixels retained by MenuCapture'
        shot = {'session_directory': str(self.root), 'capture_started_at_ns': 1234}
        with patch('playmodel.games.brotato.menu_capture.MenuCapture') as capture_factory, \
             patch('playmodel.games.brotato.ocr.MenuOcr') as ocr_factory, \
             patch('playmodel.games.brotato.vision.BrotatoVision') as vision_factory, \
             patch('playmodel.games.brotato.capture.read_diagnostic_png') as decoder:
            capture_factory.return_value.__enter__.return_value.read.return_value = (shot, original_pixels, 1920, 1080)
            ocr_factory.return_value.__enter__.return_value.read.return_value = {'lines': []}
            vision_factory.return_value.observe.return_value = Mock(combat_likely=True, player=(.5, .47))
            self.assertEqual(coordinator._current_scene(self.root / 'startup'), 'combat')
            vision_factory.return_value.observe.assert_called_once_with(original_pixels, 1920, 1080, observed_at_ns=1234)
            decoder.assert_not_called()

    def test_old_ocr_fault_never_hides_new_input_failure(self):
        report = {'training': {'error': 'old OCR fault', 'stop_category': 'runtime_error'}}
        self.assertTrue(self.module._recoverable_report(report))
        for reason in ('sink_deadline', 'input_guard', 'human takeover', 'identity changed', 'release failed'):
            self.assertFalse(self.module._recoverable_report({**report, 'error': reason}))


if __name__ == "__main__":
    unittest.main()
