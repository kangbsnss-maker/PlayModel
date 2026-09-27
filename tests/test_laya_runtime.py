"""Laya choices reuse input guards while remaining outside CNN PPO."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch is not installed')

import test_neural_menu_controller as menu_tests
import test_neural_runtime as combat_tests
import test_scheduling_recovery as recovery_tests
from playmodel.games.brotato.laya_menu import CNN_EXCLUSION, LayaMenuController, LayaMenuDecision
from playmodel.games.brotato.neural_choices import MAX_AGE_NS
from playmodel.games.brotato.neural_menu_controller import NeuralMenuError


class FakeClient:
    def __init__(self, owner):
        self.owner = owner
        self.choices, self.accepted, self.abandoned = [], [], []
        self.latency = 0
        self.illegal = False

    def choose(self, state, options, evidence):
        self.choices.append((state, options, evidence))
        self.owner.now += self.latency
        return {'decision_id': 'laya-choice-' + str(len(self.choices)),
                'action_id': 'invalid' if self.illegal else next(iter(options)),
                'behavior_version': 'laya-test-1', 'decided_at_ns': self.owner.now,
                'distribution': {key: 1 / len(options) for key in options}}

    def accept(self, decision_id, proof):
        self.accepted.append((decision_id, proof))
        return {'accepted': True}

    def abandon(self, reason):
        self.abandoned.append(reason)


class LayaRuntimeTests(unittest.TestCase):
    setUpClass = menu_tests.NeuralMenuControllerTests.__dict__['setUpClass']
    tearDownClass = menu_tests.NeuralMenuControllerTests.__dict__['tearDownClass']
    observation = menu_tests.NeuralMenuControllerTests.observation
    send = menu_tests.NeuralMenuControllerTests.send

    def setUp(self):
        menu_tests.NeuralMenuControllerTests.setUp(self)
        self.client = FakeClient(self)
        self.controller = LayaMenuController(self.recorder, client=self.client,
            output_directory=self.root / 'laya-macros', clock=lambda: self.now)

    def propose_loot(self):
        self.controller.handle(*self.observation(1, scene='loot'))
        return self.controller.handle(*self.observation(2, scene='loot'))

    def test_choice_is_separate_type_and_invalidates_cnn_without_closing(self):
        self.assertEqual(self.propose_loot().status, 'navigate')
        decision = self.controller.pending_decision
        self.assertIsInstance(decision, LayaMenuDecision)
        self.assertFalse(hasattr(decision, 'tensors'))
        self.assertFalse(hasattr(decision, 'old_value'))
        self.assertIn(CNN_EXCLUSION, self.recorder.rejection_reasons)
        self.assertFalse(self.recorder.closed)
        self.assertEqual(len(self.recorder.records), 0)
        self.assertEqual(self.client.choices[0][2]['run_id'], self.recorder.run_id)

    def test_accepted_game_application_goes_only_to_laya(self):
        self.propose_loot()
        decision = self.controller.pending_decision
        args = self.observation(3, scene='loot', target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        with patch.object(self.recorder, 'append_macro', side_effect=AssertionError('CNN PPO contaminated')):
            self.assertEqual(self.controller.handle(*self.observation(4, color=110)).status, 'wait')
            accepted = self.controller.handle(*self.observation(5, color=110))
        self.assertEqual(accepted.status, 'accepted')
        self.assertEqual(len(self.client.accepted), 1)
        identity, proof = self.client.accepted[0]
        self.assertEqual(identity, decision.decision_id)
        self.assertTrue(proof['accepted'])
        self.assertTrue(proof['successful_transport_reported'])
        self.assertEqual(proof['actual_target'], decision.target)
        self.assertEqual(len(proof['after_frames']), 2)
        self.assertEqual(len(self.recorder.records), 0)

    def test_slow_laya_reobserves_without_resampling_or_old_authorization(self):
        self.client.latency = MAX_AGE_NS + 1
        self.assertEqual(self.propose_loot().status, 'wait')
        decision = self.controller.pending_decision
        self.assertIsNotNone(decision)
        self.client.latency = 0
        # New capture after inference, same candidate, then valid fresh Enter.
        args = self.observation(20, scene='loot', target=decision.target)
        self.assertEqual(self.controller.handle(*args).status, 'navigate')
        self.send(args)
        self.assertEqual(len(self.client.choices), 1)

    def test_changed_candidate_cannot_execute_old_choice(self):
        self.propose_loot()
        decision = self.controller.pending_decision
        args = self.observation(3, scene='loot', target=decision.target, loot_title='Different item')
        self.assertEqual(self.controller.handle(*args).status, 'wait')
        with self.assertRaisesRegex(NeuralMenuError, 'Candidate changed'):
            self.controller.authorize_enter(decision.decision_id, decision.target, *args)
        self.assertTrue(self.client.abandoned)
        self.assertEqual(self.client.accepted, [])

    def test_illegal_backend_action_fails_closed(self):
        self.client.illegal = True
        with self.assertRaisesRegex(NeuralMenuError, 'unavailable action'):
            self.propose_loot()
        self.assertEqual(self.client.accepted, [])

    def test_combat_gap_discards_laya_choices_and_pending_authorization(self):
        self.propose_loot()
        self.assertIsNotNone(self.controller.pending_decision)
        self.controller.discard_interrupted_outcome()
        self.assertIsNone(self.controller.pending_decision)
        self.assertIsNone(self.controller._prepared_observation)
        self.assertEqual(self.client.abandoned, ['released_combat_gap_outcome_excluded'])

    def test_unconfirmed_application_never_enters_learning(self):
        self.propose_loot()
        decision = self.controller.pending_decision
        args = self.observation(3, scene='loot', target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        self.controller.max_result_observations = 2
        self.controller.handle(*self.observation(4, scene='loot'))
        self.controller.handle(*self.observation(5, scene='loot'))
        with self.assertRaisesRegex(NeuralMenuError, 'remained unverified'):
            self.controller.handle(*self.observation(6, scene='loot'))
        self.assertEqual(self.client.accepted, [])
        self.assertTrue(self.client.abandoned)


class LayaOrchestrationTests(unittest.TestCase):
    def test_fixed_combat_separate_split_terminal_callback_and_boundary_discard(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_laya_learning.py'
        spec = importlib.util.spec_from_file_location('laya_cli_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'combat.pt'
            checkpoint.write_bytes(b'fixed combat source')
            args = SimpleNamespace(checkpoint=checkpoint, output=root / 'output', model_dir=root / 'model',
                laya_checkpoint=None, device='cpu', seed=0, character_slot=1, weapon='SMG',
                max_run_seconds=100, recover_active_run=True, continuous=False, cycles=1)
            client = Mock()
            client.output = root / 'laya'
            client.output.mkdir()
            client.ready = {'behavior_version': 'laya-1'}
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=False)
            client.finish.return_value = {'status': 'updated', 'behavior_version': 'laya-2'}
            client.abandon.return_value = {'status': 'abandoned'}
            client.configure_preferences.return_value = {'status': 'configured', 'behavior_version': 'laya-pref'}
            status = Mock()
            constructed, collected = [], []
            def make_cycle(**kwargs):
                constructed.append(kwargs)
                def collect(source, **options):
                    collected.append((source, options))
                    kwargs['terminal_callback']('wave_clear', {'path': 'proof', 'sha256': 'proofhash'})
                    return {'full_run_complete': True, 'training_eligible': False}
                return SimpleNamespace(stop_file=root / 'STOP', collect_run=collect)
            fake_cycle = SimpleNamespace(LocalCycle=make_cycle, LocalStatus=Mock(return_value=status))
            with patch.dict(sys.modules, {'run_recurrent_cycle': fake_cycle}), \
                    patch('playmodel.laya.client.LayaClient', return_value=client), \
                    patch('playmodel.obs.ensure_obs_running'), \
                    patch('playmodel.laya.combat_economy.CombatEconomy') as economy:
                economy.return_value.finish.return_value = {'revision': 1}
                self.assertEqual(module.run_laya(args, root), 0)
                economy.return_value.finish.assert_called_once()
            self.assertEqual(checkpoint.read_bytes(), b'fixed combat source')
            self.assertEqual(collected[0][1]['split'], 'train')
            self.assertNotIn('online_factory', constructed[0])
            self.assertTrue(callable(constructed[0]['menu_factory']))
            self.assertTrue(callable(constructed[0]['tactical_factory']))
            client.finish.assert_called_once_with('wave_clear', {'path': 'proof', 'sha256': 'proofhash'})
            client.abandon.assert_called_once_with('run_boundary_or_interruption')
            state_path = next((root / 'output').glob('*/laya-state.json'))
            state = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertFalse(state['cnn_online_ppo'])
            self.assertFalse(state['reports'][0]['evaluation_score_eligible'])
            self.assertEqual(len(state['updates']), 1)
            status.close.assert_called_once()


class MixedControlRolloutTests(unittest.TestCase):
    setUpClass = combat_tests.NeuralRuntimeTests.__dict__['setUpClass']
    tearDownClass = combat_tests.NeuralRuntimeTests.__dict__['tearDownClass']
    setUp = combat_tests.NeuralRuntimeTests.setUp
    tearDown = combat_tests.NeuralRuntimeTests.tearDown
    evidence = combat_tests.NeuralRuntimeTests.evidence
    factories = combat_tests.NeuralRuntimeTests.factories
    run_trial = combat_tests.NeuralRuntimeTests.run_trial

    def zero_boundary(self, kind, *, invalid=None, cleanup_error=False):
        from dataclasses import replace
        from playmodel.games.brotato import neural_runtime
        from playmodel.games.brotato.pilot import PilotConfig
        factories = self.factories()
        class MenuVision:
            def observe(self, *args, **kwargs):
                return SimpleNamespace(combat_likely=False)
        factories['vision_factory'] = MenuVision
        if cleanup_error:
            base = factories['stream_factory']
            class BadClose(base):
                def close(self):
                    raise OSError('fixture capture cleanup failed')
            factories['stream_factory'] = BadClose
        cache = []
        def terminal():
            if not cache:
                proof = self.evidence(kind)
                if invalid == 'independent':
                    proof = replace(proof, independent_of_policy=False)
                elif invalid == 'hash':
                    proof = replace(proof, frame_sha256='0' * 64)
                cache.append(proof)
            return cache[0]
        return neural_runtime.run_neural_trial(self.root / 'fake.exe', self.root / 'trials',
            model=self.model, combat_entry=self.evidence('combat'), terminal_observer=terminal,
            scheduling_recovery=True, recovery_only=True, split='evaluation',
            config=PilotConfig(max_seconds=4, max_steps=4, max_frame_age_ms=1000,
                               input_watchdog_ms=1000, policy_budget_ms=500,
                               terminal_wait_seconds=1, startup_seconds=1, tick_ms=2), **factories)

    def test_zero_input_verified_clear_advances_recovery_without_training(self):
        result = self.zero_boundary('wave_clear')
        self.assertEqual(result['steps'], 0)
        self.assertTrue(result['verified_terminal_boundary'], result)
        self.assertTrue(result['recovery_completed'], result)
        self.assertFalse(result['rollout_eligible'])
        self.assertFalse(result['cnn_training_eligible'])
        self.assertIsNone(result['flat_rollout_path'])

    def test_zero_input_verified_death_advances_recovery_without_training(self):
        result = self.zero_boundary('death')
        self.assertEqual(result['steps'], 0)
        self.assertTrue(result['verified_terminal_boundary'], result)
        self.assertTrue(result['recovery_completed'], result)
        self.assertFalse(result['rollout_eligible'])

    def test_zero_input_unverified_terminal_never_advances(self):
        result = self.zero_boundary('death', invalid='independent')
        self.assertFalse(result['verified_terminal_boundary'])
        self.assertFalse(result['recovery_completed'])

    def test_zero_input_changed_terminal_hash_never_advances(self):
        result = self.zero_boundary('death', invalid='hash')
        self.assertFalse(result['verified_terminal_boundary'])
        self.assertFalse(result['recovery_completed'])

    def test_zero_input_cleanup_failure_never_advances(self):
        result = self.zero_boundary('death', cleanup_error=True)
        self.assertTrue(result['cleanup_errors'])
        self.assertFalse(result['verified_terminal_boundary'])
        self.assertFalse(result['recovery_completed'])

    def test_zero_input_late_release_proof_never_advances(self):
        from dataclasses import replace
        from playmodel.games.brotato import neural_runtime
        original = neural_runtime.RealtimeController.events
        def late(controller):
            return [replace(event, deadline_ns=event.send_finished_at_ns)
                    if event.kind == 'release' else event for event in original(controller)]
        with patch.object(neural_runtime.RealtimeController, 'events', late):
            result = self.zero_boundary('death')
        self.assertFalse(result['verified_terminal_boundary'])
        self.assertFalse(result['recovery_completed'])

    def test_train_split_mixed_control_blocks_ppo_but_keeps_audit(self):
        from playmodel.games.brotato import neural_runtime
        from playmodel.learning.full_run import FullRunRecorder
        original = neural_runtime._run_neural_trial_once
        def collect(*args, **kwargs):
            return original(*args, **kwargs, mixed_control=True)
        with patch.object(neural_runtime, '_run_neural_trial_once', side_effect=collect):
            report = self.run_trial(split='train')
        self.assertTrue(report['rollout_eligible'], report)
        self.assertTrue(report['mixed_control'])
        self.assertFalse(report['cnn_training_eligible'])
        with self.assertRaisesRegex(ValueError, 'excluded from CNN PPO'):
            neural_runtime.load_rollout(report['flat_rollout_path'])
        audit = neural_runtime.load_rollout(report['flat_rollout_path'], allow_mixed_control_audit=True)
        self.assertEqual(audit.split, 'train')
        recorder = FullRunRecorder(self.model, 'mixed-control-test', split='train')
        recorder.append_combat_report(report)
        self.assertIn(CNN_EXCLUSION, recorder.rejection_reasons)
        manifest = recorder.abort(self.root / 'whole-run', 'audit only')
        self.assertFalse(manifest['training_eligible'])
        self.assertGreater(len(recorder.records), 0)


class CaptureShutdownTests(unittest.TestCase):
    def spawn_stream(self, source):
        from playmodel.games.brotato.stream import CaptureStream
        original = subprocess.Popen
        def spawn(args, **kwargs):
            self.assertEqual(kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return original([sys.executable, '-u', '-c', source], **kwargs)
        with patch('playmodel.games.brotato.stream.subprocess.Popen', side_effect=spawn):
            return CaptureStream(Path('unused.exe'))

    def test_intentional_shutdown_does_not_create_capture_failure(self):
        stream = self.spawn_stream('import sys; sys.stdin.buffer.read()')
        stream.close()
        self.assertIsNone(stream.error)
        self.assertFalse(stream._reader.is_alive())

    def test_unexpected_exit_retains_error_through_close(self):
        stream = self.spawn_stream('import sys; sys.stderr.write("unexpected fixture failure"); sys.exit(2)')
        stream._reader.join(3)
        try:
            self.assertIn('Capture worker ended', stream.error)
            self.assertIn('unexpected fixture failure', stream.error)
            before = stream.error
        finally:
            stream.close()
        self.assertEqual(stream.error, before)

    def test_prior_capture_error_survives_intentional_shutdown(self):
        stream = self.spawn_stream('import sys; sys.stdin.buffer.read()')
        stream.error = 'prior capture failure'
        stream.close()
        self.assertEqual(stream.error, 'prior capture failure')


class MixedControlRecoveryTests(unittest.TestCase):
    def test_existing_release_reacquisition_and_retry_limits_apply_to_mixed_run(self):
        from playmodel.games.brotato import neural_runtime
        fixture = recovery_tests.SchedulingRecoveryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = neural_runtime.run_neural_trial
        def mixed(*args, **kwargs):
            kwargs.update(mixed_control=True, recovery_only=False, split='train')
            return original(*args, **kwargs)
        with patch.object(neural_runtime, 'run_neural_trial', side_effect=mixed):
            fixture.test_actual_guard_releases_then_fresh_pair_resets_hidden_without_training_gap()

    def test_coordinator_aborts_old_recorder_resets_gru_and_skips_gap_reward(self):
        self._coordinator_gap(followup=False)

    def test_recovered_clear_excludes_reward_then_next_normal_wave_can_learn(self):
        self._coordinator_gap(followup=True)

    def _coordinator_gap(self, *, followup):
        from playmodel.games.brotato.pilot import PilotConfig
        from playmodel.learning.recurrent_ppo import RecurrentActorCritic
        path = Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py'
        spec = importlib.util.spec_from_file_location('laya_gap_cycle_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            terminal_dir = root / 'combat'
            terminal_dir.mkdir()
            kind = 'wave_clear' if followup else 'death'
            (terminal_dir / 'terminal.json').write_text(json.dumps({'kind': kind}))
            next_dir = root / 'next-combat'
            next_dir.mkdir()
            (next_dir / 'terminal.json').write_text(json.dumps({'kind': 'death'}))
            (next_dir / 'report.json').write_text(json.dumps({'reason': 'terminal_death'}))
            cycle = module.LocalCycle.__new__(module.LocalCycle)
            cycle.root, cycle.output = root, root / 'runs'
            cycle.runtime_source_hashes = {}
            cycle.operation_id = cycle.evaluation_scope_id = cycle.status = None
            cycle.max_run_seconds, cycle.seed = 30, 0
            cycle.executable, cycle.stop_file, cycle.ocr_script = root / 'game', root / 'STOP', root / 'ocr'
            cycle._new_run = Mock(return_value=({}, None))
            cycle.terminal_callback = Mock(return_value={'status': 'updated'})
            client, originals = Mock(), []
            def factory(recorder, **kwargs):
                originals.append(recorder)
                recorder.hidden.fill_(1)
                return LayaMenuController(recorder, client=client, **kwargs)
            cycle.menu_factory = factory
            result = {'session_directory': str(terminal_dir), 'reason': 'terminal_' + kind,
                      'terminal_kind': kind, 'steps': 1, 'total_actual_steps': 5, 'error': None, 'recovery_completed': True,
                      'attempt_reports': ['old-report', 'recovered-report'],
                      'scheduling_recoveries': [{'guard': 'held_observation_expired'}], 'rollout_eligible': False}
            normal = {'session_directory': str(next_dir), 'reason': 'terminal_death',
                      'terminal_kind': 'death', 'steps': 0, 'error': None, 'scheduling_recoveries': [],
                      'rollout_eligible': True, 'flat_rollout_path': 'unused-audit'}
            def session(executable, directory, **kwargs):
                pilot = kwargs['combat_runner'](executable, directory, config=PilotConfig())
                self.assertEqual(pilot['status'], 'neural_rollout_ready')
                new = kwargs['neural_menu'].recorder
                self.assertIsNot(new, originals[0])
                self.assertEqual(new.split, 'train')
                self.assertFalse(new.hidden.any())
                self.assertTrue(originals[0].closed)
                cycle.terminal_callback.assert_not_called()
                if followup:
                    kwargs['combat_runner'](executable, directory, config=PilotConfig())
                return {'session_directory': str(directory), 'reason': 'run_finished'}
            with patch.object(module, '_runtime_contract', return_value={}), \
                    patch('playmodel.learning.recurrent_ppo.load_checkpoint', return_value=(RecurrentActorCritic(), {})), \
                    patch('playmodel.games.brotato.neural_runtime.run_neural_trial', side_effect=[result, normal]) as trial, \
                    patch('playmodel.learning.full_run.FullRunRecorder.append_combat_report'), \
                    patch('playmodel.games.brotato.session.run_session', side_effect=session):
                report = cycle.collect_run(root / 'source.pt', split='train', tag='laya-test')
            self.assertIsNone(report['error'])
            self.assertTrue(report['full_run_complete'])
            self.assertTrue(trial.call_args.kwargs['mixed_control'])
            self.assertTrue(trial.call_args.kwargs['scheduling_recovery'])
            self.assertFalse(trial.call_args.kwargs['recovery_only'])
            if followup:
                cycle.terminal_callback.assert_called_once()
                self.assertEqual(cycle.terminal_callback.call_args.args[0], 'death')
                self.assertTrue(trial.call_args.kwargs['reset_first'])
                self.assertFalse(trial.call_args.kwargs['initial_hidden'].any())
            else:
                cycle.terminal_callback.assert_not_called()
            client.abandon.assert_called_once_with('released_combat_gap_outcome_excluded')
            expected = [{'status': 'excluded', 'reason': 'released_combat_gap'}]
            self.assertEqual(report['choice_updates'], expected + ([{'status': 'updated'}] if followup else []))
            self.assertEqual(report['steps'], 5)
            self.assertEqual(report['combat_attempt_reports'], ['old-report', 'recovered-report'])


if __name__ == '__main__':
    unittest.main()
