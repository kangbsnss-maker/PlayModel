"""Online actor handoff against fake capture/input; no game or external service."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')
import torch
import test_neural_runtime as fixtures
from playmodel.games.brotato.online_runtime import OnlineActorSession
from playmodel.learning.full_run import FullRunRecorder
from playmodel.learning.online_ppo import _load_fragment
from playmodel.learning.recurrent_ppo import load_checkpoint, save_checkpoint


class OnlineRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.fixture = fixtures.NeuralRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.root, self.model = self.fixture.root, self.fixture.model
        self.checkpoint = self.root / 'source.pt'
        save_checkpoint(self.model, self.checkpoint)
        self.recorder = FullRunRecorder(self.model, 'physical-run', split='train')
        self.manager = OnlineActorSession(self.recorder, checkpoint=self.checkpoint,
            root=Path(__file__).resolve().parents[1], output=self.root / 'online', chunk_actions=4)
        self.addCleanup(self.manager.close)
        self.job_count = 0

    def fake_job(self, manifest, **kwargs):
        owner = self
        self.job_count += 1
        number = self.job_count
        source, _ = load_checkpoint(Path(manifest).parent / 'behavior-policy.pt')
        source_version = source.policy_version()
        candidate = deepcopy(source)
        with torch.no_grad():
            next(candidate.parameters()).add_(.001)
        checkpoint = self.root / f'candidate-{number}.pt'
        save_checkpoint(candidate, checkpoint)
        report = dict(checkpoint=str(checkpoint), candidate_version=candidate.policy_version(),
                      source_version=source_version, optimizer_steps=1, final_kl_within_target=True)

        class Job:
            def describe(self):
                return {'worker_pid': 0, 'status': 'running'}

            def poll(self):
                return report

            def load_candidate(self, current_version, **kwargs):
                owner.assertEqual(current_version, source_version)
                return candidate, report

            def close(self, **kwargs):
                pass

        return Job()

    def add_menu_prelude(self):
        inputs = (torch.zeros(1, 3, 96, 96, dtype=torch.uint8), torch.zeros(1, 16),
            torch.tensor([1]), torch.zeros(1, 2, self.model.config.candidate_dim),
            torch.tensor([[True, True, False, False, False, False, False, False, False]]))
        before = self.recorder.hidden
        with torch.no_grad():
            result = self.model.step(*inputs, hidden=before, reset=torch.tensor([True]))
            action, logp = result.sample()
        proof = self.fixture.evidence('menu')
        tensors = dict(zip(('images', 'context', 'phase', 'candidates', 'legal_mask'), inputs))
        tensors.update(hidden_before=before, next_hidden=result.next_hidden, actions=action,
            old_log_probs=logp, old_values=result.value, reset=torch.tensor([True]))
        evidence = dict(frame_ref=proof.frame_ref, frame_sha256=proof.frame_sha256,
            observed_at_ns=proof.observed_at_ns, available_at_ns=proof.available_at_ns,
            decided_at_ns=proof.available_at_ns, sent_at_ns=proof.available_at_ns,
            action_origin='policy', transmitted=True, actual_action=int(action),
            game_application_verified=True)
        self.recorder.append_decision(tensors, evidence=evidence)

    def test_actual_midcombat_adoption_keeps_versioned_fragments_and_menu_prelude(self):
        self.add_menu_prelude()
        with patch('playmodel.learning.online_ppo.start_online_job', side_effect=self.fake_job):
            report = self.fixture.run_trial(steps=50, online_session=self.manager,
                initial_hidden=self.recorder.hidden, reset_first=False)
        self.assertEqual(report['reason'], 'step_limit', report)
        self.assertTrue(report['online_collection_eligible'], report)
        self.assertFalse(report['rollout_eligible'])
        self.assertIsNone(report['flat_rollout_path'])
        snapshot = self.manager.snapshot()
        self.assertGreater(len(snapshot['adoptions']), 0, snapshot)
        rows = [json.loads(line) for line in (Path(report['session_directory'])/'actions.jsonl').read_text().splitlines()]
        self.assertGreater(len(set(row['behavior_version'] for row in rows)), 1)
        for index, row in enumerate(rows):
            if index and row['actor_segment'] != rows[index-1]['actor_segment']:
                self.assertTrue(row['reset'])
                self.assertEqual(set(row['hidden_before']), {0.})
                self.assertGreater(row['observed_at_ns'], rows[index-1]['sent_at_ns'])
                adoption = next(item for item in snapshot['adoptions'] if item['segment'] == row['actor_segment'])
                self.assertEqual(adoption['applied_at_ns'], row['sent_at_ns'])
        for fragment in snapshot['fragments']:
            _load_fragment(fragment['manifest_path'])
        first = json.loads(Path(snapshot['fragments'][0]['manifest_path']).read_text())
        self.assertEqual(first['phase_counts'], {'0': 4, '1': 1})
        self.assertFalse(self.manager.recorder.closed)
        self.assertTrue(torch.equal(self.manager.recorder.hidden, torch.zeros_like(self.manager.recorder.hidden)))
        journal = json.loads(Path(snapshot['applied_checkpoint_path']).read_text())
        self.assertEqual(journal['applied_at_ns'], journal['action_evidence']['sent_at_ns'])

    def test_guard_tail_is_preserved_without_learning_or_weight_commit(self):
        self.fixture.state.stop = True
        with patch('playmodel.learning.online_ppo.start_online_job') as launch:
            report = self.fixture.run_trial(steps=100, online_session=self.manager)
        self.assertEqual(report['reason'], 'F8', report)
        self.assertFalse(report['online_collection_eligible'])
        launch.assert_not_called()
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot['fragments'], [])
        self.assertEqual(snapshot['adoptions'], [])
        self.assertGreater(sum(row['actions'] for row in snapshot['excluded_tails']), 0)
        self.assertEqual(snapshot['checkpoint'], str(self.checkpoint.resolve()))

    def test_evaluation_cannot_use_online_actor(self):
        with self.assertRaisesRegex(ValueError, 'training split'):
            self.fixture.run_trial(split='evaluation', online_session=self.manager)
        self.assertEqual(self.fixture.state.count, 0)

    def test_unsealed_wave_tail_has_only_verified_reward(self):
        self.manager.chunk_actions = 64
        self.fixture.state.menu = True
        evidence = []
        def terminal():
            if self.fixture.state.count < 2:
                return None
            if not evidence:
                evidence.append(self.fixture.evidence('wave_clear'))
            return evidence[0]
        with patch('playmodel.learning.online_ppo.start_online_job', side_effect=self.fake_job):
            report = self.fixture.run_trial(steps=100, terminal=terminal, online_session=self.manager)
        self.assertEqual(report['reason'], 'terminal_wave_clear', report)
        self.assertTrue(report['online_collection_eligible'], report)
        fragment = self.manager.snapshot()['fragments'][0]
        _, _, _, flat, _, header = _load_fragment(fragment['manifest_path'])
        self.assertEqual(float(flat.rewards.sum()), 1.)
        self.assertFalse(header['full_run_complete'])

    def test_expired_candidate_proposal_never_adopts_or_learns(self):
        candidate = deepcopy(self.model)
        with torch.no_grad():
            next(candidate.parameters()).add_(.001)
        version = candidate.policy_version()
        original_step = candidate.step
        def slow(*args, **kwargs):
            time.sleep(.12)
            return original_step(*args, **kwargs)
        candidate.step = slow
        self.manager._ready = dict(model=candidate, version=version,
            source_version=self.model.policy_version(), checkpoint=str(self.root/'unsent.pt'),
            hidden=candidate.initial_hidden(1), token=1, reason='unit_staged_candidate')
        with patch('playmodel.learning.online_ppo.start_online_job') as launch:
            report = self.fixture.run_trial(steps=4, policy_budget_ms=40, online_session=self.manager)
        self.assertEqual(report['steps'], 0, report)
        self.assertFalse(report['online_collection_eligible'])
        self.assertEqual(self.manager.snapshot()['adoptions'], [])
        self.assertEqual(self.manager.checkpoint, str(self.checkpoint.resolve()))
        launch.assert_not_called()
