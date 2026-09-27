"""Bounded post-release recovery; no game, keyboard, or desktop capture."""
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
from playmodel.games.brotato import neural_runtime as runtime
from playmodel.games.brotato.pilot import PilotConfig
from playmodel.learning.recurrent_ppo import RecurrentActorCritic


class SchedulingRecoveryTests(unittest.TestCase):
    def setUp(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        self.fixture = fixtures.NeuralRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root

    def report(self, name, *, reason='held_observation_expired', release_late=False):
        directory = self.root / name
        directory.mkdir()
        events = [dict(kind='authority', reason=reason), dict(kind='release', reason=reason,
            receipt={'transmitted': True, 'acknowledged': None}, send_finished_at_ns=30,
            deadline_ns=20 if release_late else 40)]
        (directory/'control-events.json').write_text(json.dumps(events))
        (directory/'input-attempts.json').write_text('[]')
        report = dict(session_directory=str(directory), reason='controller_guard', error='guard',
            controller_guard_reason=reason, recorder_complete=True, worker_stopped=True,
            cleanup_errors=[], steps=1, rollout_eligible=False, target_identity={'hwnd':123})
        (directory/'report.json').write_text(json.dumps(report))
        return report

    def test_only_two_consecutive_restarts_and_no_modified_original_reports(self):
        reports = [self.report(str(index)) for index in range(3)]
        with patch.object(runtime, '_run_neural_trial_once', side_effect=reports) as collect:
            report = runtime.run_neural_trial(self.root/'game', self.root, model=self.fixture.model,
                combat_entry=self.fixture.evidence('combat'), scheduling_recovery=True, recovery_only=True)
        self.assertEqual(collect.call_count, 3)
        self.assertEqual(len(report['scheduling_recoveries']), 2)
        self.assertFalse(report['rollout_eligible'])
        self.assertFalse(report['evaluation_score_eligible'])
        for item in reports:
            self.assertNotIn('scheduling_recoveries', json.loads((Path(item['session_directory'])/'report.json').read_text()))
        second = collect.call_args_list[1].kwargs
        self.assertTrue(second['reset_first'])
        self.assertFalse(second['initial_hidden'].any())
        self.assertLess(second['config'].max_seconds, 60)

    def test_stop_transport_and_late_release_never_restart(self):
        variants = [('late', 'held_observation_expired', True), ('transport', 'sink_deadline', False),
                    ('human', 'human_intervention', False), ('f8', 'F8', False)]
        for name, reason, late in variants:
            with self.subTest(name=name):
                original = self.report(name, reason=reason, release_late=late)
                with patch.object(runtime, '_run_neural_trial_once', return_value=original) as collect:
                    result = runtime.run_neural_trial(self.root/'game', self.root, model=self.fixture.model,
                        combat_entry=self.fixture.evidence('combat'), scheduling_recovery=True, recovery_only=True)
                self.assertEqual(collect.call_count, 1)
                self.assertEqual(result['scheduling_recoveries'], [])
        original = self.report('stopped')
        stop = self.root/'STOP'
        stop.touch()
        with patch.object(runtime, '_run_neural_trial_once', return_value=original) as collect:
            runtime.run_neural_trial(self.root/'game', self.root, model=self.fixture.model,
                combat_entry=self.fixture.evidence('combat'), scheduling_recovery=True,
                recovery_only=True, stop_file=stop)
        self.assertEqual(collect.call_count, 1)

    def test_actual_guard_releases_then_fresh_pair_resets_hidden_without_training_gap(self):
        self.fixture.state.menu = True
        evidence, calls = [], [0]
        def terminal():
            if self.fixture.state.count < 2:
                return None
            if not evidence:
                evidence.append(self.fixture.evidence('wave_clear'))
            return evidence[0]
        original = RecurrentActorCritic.step
        def slow(model, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                time.sleep(.12)
            return original(model, *args, **kwargs)
        config = PilotConfig(max_seconds=4, max_steps=20, max_frame_age_ms=80,
            input_watchdog_ms=200, policy_budget_ms=200, startup_seconds=1, terminal_wait_seconds=1, tick_ms=2)
        with patch.object(RecurrentActorCritic, 'step', slow):
            result = runtime.run_neural_trial(self.root/'game', self.root/'trials', model=self.fixture.model,
                combat_entry=self.fixture.evidence('combat'), config=config, terminal_observer=terminal,
                scheduling_recovery=True, recovery_only=True, **self.fixture.factories())
        self.assertEqual(result['reason'], 'terminal_wave_clear', result)
        self.assertTrue(result['recovery_completed'], result)
        self.assertEqual(len(result['scheduling_recoveries']), 1, result)
        self.assertFalse(result['rollout_eligible'])
        directory = Path(result['session_directory'])
        proof = json.loads((directory/'scheduling-reacquisition.json').read_text())
        self.assertEqual(len(proof['observations']), 2)
        self.assertGreater(proof['observations'][0]['observed_at_ns'], proof['released_at_ns'])
        self.assertGreater(proof['observations'][1]['observed_at_ns'], proof['observations'][0]['available_at_ns'])
        first = json.loads((directory/'actions.jsonl').read_text().splitlines()[0])
        self.assertTrue(first['reset'])
        self.assertEqual(set(first['hidden_before']), {0.})

    def test_fixed_scored_run_cannot_silently_enable_recovery(self):
        with self.assertRaisesRegex(ValueError, 'fixed scored'):
            runtime.run_neural_trial(self.root/'game', self.root, model=self.fixture.model,
                combat_entry=self.fixture.evidence('combat'), scheduling_recovery=True, split='evaluation')
