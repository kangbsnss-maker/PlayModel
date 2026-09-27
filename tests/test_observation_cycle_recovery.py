"""Coordinator resets audit memory across observation gaps without inventing rewards."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')

from playmodel.games.brotato.laya_menu import LayaMenuController
from playmodel.games.brotato.pilot import PilotConfig
from playmodel.learning.recurrent_ppo import RecurrentActorCritic, ModelConfig


class ObservationCycleRecoveryTests(unittest.TestCase):
    def exercise(self, mode):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py'
        spec = importlib.util.spec_from_file_location('observation_cycle_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            proof = {'schema': 'playmodel.released-observation-transition.v1',
                'previous_session': str(root / 'gap'), 'released_at_ns': 123,
                'target_identity': {'hwnd': 1}, 'training_eligible': False, 'terminal_reward_verified': False}
            gap = {'session_directory': str(root / 'gap'), 'reason': 'screen_changed',
                   'observation_transition': proof, 'steps': 2, 'terminal_kind': None, 'error': None}
            def terminal(name, kind):
                directory = root / name
                directory.mkdir()
                (directory / 'terminal.json').write_text(json.dumps({'kind': kind}))
                (directory / 'report.json').write_text(json.dumps({'reason': 'terminal_' + kind}))
                return {'session_directory': str(directory), 'reason': 'terminal_' + kind,
                    'terminal_kind': kind, 'steps': 0, 'error': None, 'scheduling_recoveries': [],
                    'verified_terminal_boundary': True, 'rollout_eligible': True, 'flat_rollout_path': 'audit'}
            clear, death = terminal('clear', 'wave_clear'), terminal('death', 'death')
            trial_results = [gap, clear, death] if mode == 'combat_resume' else [gap]
            calls = []
            def session(executable, directory, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    result = kwargs['combat_runner'](executable, directory, config=PilotConfig())
                    self.assertEqual(result['status'], 'observation_wait')
                    fresh = kwargs['neural_menu'].recorder
                    self.assertIsNot(fresh, originals[0])
                    self.assertTrue(originals[0].closed)
                    self.assertFalse(fresh.hidden.any())
                    self.assertEqual(fresh.run_id, originals[0].run_id)
                    self.assertEqual(fresh.split, 'train')
                    cycle.terminal_callback.assert_not_called()
                    if mode == 'combat_resume':
                        kwargs['observation_gap_callback']({'reason': 'observation_resumed', 'scene': 'combat'})
                        result = kwargs['combat_runner'](executable, directory, config=PilotConfig())
                        self.assertEqual(result['status'], 'neural_rollout_ready')
                        cycle.terminal_callback.assert_not_called()
                        kwargs['combat_runner'](executable, directory, config=PilotConfig())
                        return {'session_directory': str(directory), 'reason': 'run_finished'}
                    if mode == 'stop':
                        cycle.stop_file.touch()
                    reason = {'pause': 'awaiting_game_resume', 'stop': 'F8', 'error': 'capture_error',
                              'untyped': 'awaiting_game_resume'}[mode]
                    return {'session_directory': str(directory), 'reason': reason,
                        'observation_wait_segment': mode != 'untyped', 'observation_wait_seconds': 1.25,
                        'run_context': {'observation_wait': {'reason': 'pause'}}}
                self.assertEqual(mode, 'pause')
                self.assertIn('observation_wait', kwargs['run_context'])
                return {'session_directory': str(directory), 'reason': 'F8'}
            model = RecurrentActorCritic(ModelConfig(hidden_size=16, visual_size=16, candidate_hidden_size=8))
            with patch.object(module, '_runtime_contract', return_value={}), \
                    patch('playmodel.learning.recurrent_ppo.load_checkpoint', return_value=(model, {})), \
                    patch('playmodel.games.brotato.neural_runtime.run_neural_trial', side_effect=trial_results), \
                    patch('playmodel.games.brotato.neural_runtime.safe_observation_transition', return_value=proof) as checked, \
                    patch('playmodel.learning.full_run.FullRunRecorder.append_combat_report'), \
                    patch('playmodel.games.brotato.session.run_session', side_effect=session):
                result = cycle.collect_run(root / 'source.pt', split='train', tag='laya-test')
            checked.assert_called_once()
            self.assertEqual(result['observation_gaps'][0]['reason'], 'screen_changed')
            self.assertGreaterEqual(client.abandon.call_count, 1)
            if mode == 'combat_resume':
                cycle.terminal_callback.assert_called_once()
                self.assertEqual(cycle.terminal_callback.call_args.args[0], 'death')
                self.assertIsNone(result['error'])
                self.assertTrue(result['full_run_complete'])
            else:
                cycle.terminal_callback.assert_not_called()
                self.assertEqual(len(calls), 2 if mode == 'pause' else 1)
                self.assertEqual(result['observation_wait_seconds'], 1.25)

    def test_gap_resets_memory_and_excludes_resumed_wave_but_next_wave_learns(self):
        self.exercise('combat_resume')

    def test_typed_pause_wait_continues_session_without_reward(self):
        self.exercise('pause')

    def test_stop_error_and_untyped_wait_do_not_retry(self):
        for mode in ('stop', 'error', 'untyped'):
            with self.subTest(mode=mode):
                self.exercise(mode)


if __name__ == '__main__':
    unittest.main()
