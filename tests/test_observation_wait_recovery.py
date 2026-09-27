"""Released observation gaps never authorize pause keys or manufacture rewards."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')

from playmodel.games.brotato import session
from playmodel.games.brotato.vision import VisionObservation
from playmodel.games.brotato.neural_menu_controller import MenuDirective
from test_brotato_session_recovery import ocr


class ObservationWaitTests(unittest.TestCase):
    def run_sequence(self, scenes, *, pending=False, stop=False, identity=False,
                     release_error=False, budget=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            controller, capture, reader, neural = Mock(), Mock(), Mock(), Mock()
            controller.hwnd = 1
            controller._context.key_state.return_value = 0
            controller._context.foreground.return_value = 0
            neural.pending_decision = object() if pending else None
            neural.awaiting_application = False
            neural.recorder.model.config.context_dim = 16
            neural.handle.return_value = MenuDirective('unhandled', None, None, 'fixture')
            neural.abort.side_effect = ValueError('unresolved menu decision')
            if release_error:
                controller.release.side_effect = OSError('release failed')
            statuses, gaps, reads, texts = [], [], [], {}
            pixels = bytes((20, 20, 20, 0)) * 1920 * 1080
            clock = [100.0]
            def read(output, check, **kwargs):
                if budget and reads:
                    clock[0] = 111.0
                    check()
                index = len(reads)
                scene = scenes[index]
                path = output / str(index)
                path.mkdir(parents=True)
                raw = ('frame-' + str(index)).encode()
                (path / 'frame.png').write_bytes(raw)
                now = time.perf_counter_ns()
                shot = {'session_directory': str(path), 'hwnd': 1,
                        'pid': 3 if identity and index else 2, 'executable': 'game.exe',
                        'frame_sha256': hashlib.sha256(raw).hexdigest(),
                        'capture_started_at_ns': now, 'available_at_ns': now}
                texts[str(path / 'frame.png')] = ocr(scene)
                reads.append(shot)
                if stop and len(reads) == 1:
                    (root / 'STOP').touch()
                return shot, pixels, 1920, 1080
            capture.read.side_effect = read
            reader.read.side_effect = lambda path: texts[str(path)]
            combat = Mock(return_value={'session_directory': str(root / 'combat'),
                'status': 'neural_rollout_ready', 'reason': 'terminal_wave_clear', 'training_performed': False})
            vision = Mock()
            vision.observe.return_value = VisionObservation(player=(.5, .5), combat_likely=True)
            cleanup_capture = Mock(return_value={'session_directory': str(root)})
            for name, value in (('MenuCapture', Mock(return_value=capture)),
                                ('MenuOcr', Mock(return_value=reader)),
                                ('BackgroundController', Mock(return_value=controller)),
                                ('BrotatoVision', Mock(return_value=vision)),
                                ('capture_session', cleanup_capture),
                                ('inspect_installation', Mock(return_value={'installations': []}))):
                stack.enter_context(patch.object(session, name, value))
            if budget:
                stack.enter_context(patch.object(session.time, 'perf_counter', side_effect=lambda: clock[0]))
            report = session.run_session.__wrapped__(root / 'game.exe', root / 'sessions',
                waves=1, seconds=10, stop_file=root / 'STOP', ocr_script=root / 'ocr',
                combat_runner=combat, neural_menu=neural, observation_recovery=True,
                observation_gap_callback=gaps.append, observation_status_callback=statuses.append)
            actions = json.loads((Path(report['session_directory']) / 'menu-actions.json').read_text())
            self.assertEqual(actions, [])
            controller.tap_menu.assert_not_called()
            cleanup_capture.assert_not_called()
            return report, statuses, gaps, reads, combat, neural

    def test_pause_to_two_fresh_combat_frames_resumes_without_escape(self):
        report, statuses, gaps, reads, combat, neural = self.run_sequence(['pause', 'pause', 'unknown', 'unknown'])
        self.assertEqual(report['reason'], 'wave_limit')
        self.assertEqual(len(reads), 4)
        combat.assert_called_once()
        self.assertEqual([row['reason'] for row in gaps], ['awaiting_game_resume', 'observation_resumed'])
        self.assertEqual(gaps[-1]['scene'], 'combat')
        self.assertFalse(gaps[-1]['gap_training_eligible'])
        self.assertIn('awaiting_game_resume', statuses)
        self.assertEqual(statuses[-1], 'combat')
        self.assertNotIn('observation_wait', report['run_context'])
        self.assertGreater(report['observation_wait_seconds'], 0)

    def test_pause_to_two_verified_result_menu_frames_never_enters_combat(self):
        report, _, gaps, reads, combat, neural = self.run_sequence(['pause', 'result', 'result'])
        self.assertEqual(report['reason'], 'run_finished')
        self.assertEqual(len(reads), 3)
        self.assertEqual(gaps[-1]['scene'], 'result')
        combat.assert_not_called()
        neural.handle.assert_called_once()

    def test_pending_menu_is_not_discarded_to_force_resume(self):
        report, _, gaps, _, combat, neural = self.run_sequence(['pause'], pending=True)
        self.assertIn('unresolved menu decision', report['reason'])
        self.assertFalse(report['observation_wait_segment'])
        self.assertEqual(gaps, [])
        neural.abort.assert_called_once()
        combat.assert_not_called()

    def test_stop_remains_hard_failure(self):
        report, _, _, _, combat, _ = self.run_sequence(['pause'], stop=True)
        self.assertIn('Session stopped', report['reason'])
        self.assertFalse(report['observation_wait_segment'])
        combat.assert_not_called()

    def test_identity_change_remains_hard_failure(self):
        report, _, _, _, combat, _ = self.run_sequence(['pause', 'unknown'], identity=True)
        self.assertIn('identity changed', report['reason'])
        self.assertFalse(report['observation_wait_segment'])
        combat.assert_not_called()

    def test_release_failure_is_not_a_wait_segment(self):
        report, _, gaps, _, combat, _ = self.run_sequence(['pause'], release_error=True)
        self.assertIn('release failed', report['reason'])
        self.assertTrue(report['release_error'])
        self.assertEqual(gaps, [])
        combat.assert_not_called()

    def test_wait_time_limit_is_typed_nonterminal_segment(self):
        report, _, _, reads, combat, _ = self.run_sequence(['pause'], budget=True)
        self.assertEqual(report['reason'], 'awaiting_game_resume')
        self.assertTrue(report['observation_wait_segment'])
        self.assertEqual(report['observation_wait_seconds'], 11)
        self.assertEqual(report['run_context']['observation_wait']['target_identity']['pid'], 2)
        self.assertEqual(len(reads), 1)
        combat.assert_not_called()


class ResumeGateTests(unittest.TestCase):
    def test_stale_duplicate_unordered_and_unknown_frames_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {'hwnd': 1, 'pid': 2, 'executable': 'game.exe'}
            gate = session.ObservationResumeGate(identity, 10)
            def shot(index, observed, available):
                folder = root / str(index)
                folder.mkdir(exist_ok=True)
                source = folder / 'frame.png'
                source.write_bytes(b'frame')
                return {**identity, 'session_directory': str(folder), 'capture_started_at_ns': observed,
                        'available_at_ns': available, 'frame_sha256': hashlib.sha256(b'frame').hexdigest()}
            first = shot(1, 20, 25)
            self.assertIsNone(gate.observe(first, scene='combat', available_at_ns=30, now_ns=35))
            self.assertIsNone(gate.observe(first, scene='combat', available_at_ns=30, now_ns=35))
            self.assertIsNone(gate.observe(shot(2, 28, 31), scene='combat', available_at_ns=32, now_ns=35))
            self.assertIsNone(gate.observe(shot(3, 36, 37), scene='unknown', available_at_ns=38, now_ns=39))
            self.assertIsNone(gate.observe(shot(4, 40, 41), scene='combat', available_at_ns=42, now_ns=43))
            proof = gate.observe(shot(5, 44, 45), scene='combat', available_at_ns=46, now_ns=47)
            self.assertEqual(len(proof['observations']), 2)
            self.assertFalse(proof['gap_training_eligible'])
            self.assertIsNone(gate.observe(shot(6, 50, 51), scene='combat', available_at_ns=52, now_ns=600_000_000))

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'frame.png').write_bytes(b'changed')
            target = {'hwnd': 1, 'pid': 2, 'executable': 'game.exe'}
            gate = session.ObservationResumeGate(target, 10)
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                gate.observe({**target, 'session_directory': str(root), 'frame_sha256': '0' * 64,
                    'capture_started_at_ns': 20, 'available_at_ns': 21},
                    scene='combat', available_at_ns=22, now_ns=23)


if __name__ == '__main__':
    unittest.main()
