"""No game input: exercise recovery routing through the real session loop."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch unavailable for neural menu recovery")

from playmodel.games.brotato import session
from playmodel.games.brotato.menu import BUTTONS, ENGLISH_ANCHORS, ENGLISH_OCR_VARIANTS
from playmodel.games.brotato.neural_menu_controller import MenuDirective


def frame(*rects):
    image = bytearray(bytes((20, 20, 20, 0)) * 1920 * 1080)
    for left, top, right, bottom in rects:
        row = bytes((200, 200, 200, 0)) * (right - left)
        for y in range(top, bottom):
            start = (y * 1920 + left) * 4
            image[start:start + len(row)] = row
    return bytes(image)


def ocr(scene):
    anchors = ENGLISH_OCR_VARIANTS['loot'] if scene == 'loot' else ENGLISH_ANCHORS.get(scene, ())
    now = time.perf_counter_ns()
    return {'text': ' '.join(token for token, _ in anchors),
            'lines': [{'words': [{'text': token, 'x': r[0] + 5, 'y': r[1] + 5,
                                  'width': 120, 'height': 30}]} for token, r in anchors],
            'available_at_ns': now, 'processing_started_at_ns': now,
            'recognition_path': 'offline_fixture'}


class RecoverySessionTests(unittest.TestCase):
    def run_fixture(self, *, partial, ambiguous=False, original_combat_frame=None):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            stat = frame((1115,633,1470,635), (1115,666,1470,668))
            if ambiguous:
                stat = frame(BUTTONS['loot']['take'], BUTTONS['loot']['recycle'],
                             (1115,633,1470,635), (1115,666,1470,668))
            frames = iter([(original_combat_frame, 'unknown')] if original_combat_frame is not None else
                          [(stat, 'loot'), (frame(BUTTONS['loot']['take']), 'loot'),
                           (frame(BUTTONS['loot']['recycle']), 'loot'), (frame(), 'result')])
            reports, capture_count = {}, []
            capture, reader, controller, neural = Mock(), Mock(), Mock(), Mock()
            controller.hwnd = 1
            controller._context.key_state.return_value = 0
            neural.recorder.model.config.context_dim = 16
            neural.handle.return_value = MenuDirective('unhandled', None, None, 'not_a_supported_learned_choice')

            def capture_read(output, **kwargs):
                pixels, scene = next(frames)
                index = len(capture_count)
                directory = output / str(index)
                directory.mkdir(parents=True)
                source = directory / 'frame.png'
                reports[str(source)] = ocr(scene)
                capture_count.append(index)
                now = time.perf_counter_ns()
                return ({'session_directory': str(directory), 'hwnd': 1,
                         'frame_sha256': str(index) * 64,
                         'capture_started_at_ns': now, 'available_at_ns': now}, pixels, 1920, 1080)

            capture.read.side_effect = capture_read
            reader.read.side_effect = lambda source: reports[str(source)]
            combat = (Mock(return_value={'status': 'aborted', 'reason': 'offline_combat_boundary_reached',
                                         'training_performed': False}) if original_combat_frame is not None else
                      Mock(side_effect=AssertionError('unexpected combat')))
            for name, value in (('MenuCapture', Mock(return_value=capture)),
                                ('MenuOcr', Mock(return_value=reader)),
                                ('BackgroundController', Mock(return_value=controller)),
                                ('capture_session', Mock(side_effect=OSError('offline test'))),
                                ('inspect_installation', Mock(return_value={'installations': []}))):
                stack.enter_context(patch.object(session, name, value))
            report = session.run_session.__wrapped__(root / 'game.exe', root / 'sessions', waves=1,
                seconds=10, stop_file=root / 'STOP', ocr_script=root / 'unused.ps1',
                edit=False, run_context={'partial_recovery': partial},
                combat_runner=combat, neural_menu=neural)
            if original_combat_frame is not None:
                self.assertIs(combat.call_args.kwargs['terminal_ocr_reader'], reader)
            actions = json.loads((Path(report['session_directory']) / 'menu-actions.json').read_text(encoding='utf-8'))
            return report, actions, controller, neural, capture_count

    def test_partial_recovery_reobserves_focus_and_records_rule_without_training_label(self):
        report, actions, controller, neural, captures = self.run_fixture(partial=True)
        self.assertEqual(report['reason'], 'run_finished')
        self.assertEqual([call.args[0] for call in controller.tap_menu.call_args_list], ['left', 'down', 'enter'])
        self.assertEqual(len(captures), 4)
        self.assertEqual([a['selected'] for a in actions], ['stat_633', 'take', 'recycle'])
        self.assertTrue(all(a['policy_origin'] == 'bootstrap_rule' and a['learned_choice'] is False
                            and a['recovery_only'] is True and a['training_eligible'] is False for a in actions))
        neural.authorize_enter.assert_not_called()
        neural.mark_sent.assert_not_called()
        neural.recorder.sample_choice.assert_not_called()

    def test_normal_run_keeps_unsupported_loot_fail_closed(self):
        report, actions, controller, neural, captures = self.run_fixture(partial=False)
        self.assertIn('Neural menu scene unsupported', report['reason'])
        self.assertEqual(actions, [])
        controller.tap_menu.assert_not_called()
        self.assertEqual(len(captures), 1)

    def test_ambiguous_buttons_do_not_fall_back_to_stat_focus(self):
        report, actions, controller, neural, captures = self.run_fixture(partial=True, ambiguous=True)
        self.assertIn('Menu selection is ambiguous', report['reason'])
        self.assertEqual(actions, [])
        controller.tap_menu.assert_not_called()

    def test_failure_log_binds_inner_error_to_source_and_scene(self):
        with patch('playmodel.execution_log.exception') as exception_log, patch('playmodel.execution_log.event') as event_log:
            report, actions, controller, neural, captures = self.run_fixture(partial=False)
        exception_log.assert_called_once()
        self.assertEqual(exception_log.call_args.args[0], 'game_session_failed')
        self.assertIn('Neural menu scene unsupported', str(exception_log.call_args.args[1]))
        event_log.assert_called_once()
        self.assertEqual(event_log.call_args.args[0], 'game_session_failure_context')
        context = event_log.call_args.kwargs
        self.assertEqual(context['scene'], 'loot')
        self.assertTrue(context['frame_path'].endswith('frame.png'))
        self.assertEqual(context['error'], report['reason'])

    def test_actual_departure_frames_keep_original_sampling_and_enter_combat(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        from playmodel.games.brotato.vision import BrotatoVision
        root = (Path(__file__).resolve().parents[1] /
                'artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/'
                'partial-recovery-e21c0140420a4c51a64d0e020ce957a2/'
                'segments/20260927T082700Z-a80eae59/menus')
        names = ('20260927T082721Z-c617dc74', '20260927T082721Z-8ffb624d',
                 '20260927T082721Z-37c9db0d', '20260927T082721Z-eb5c689e')
        if not all((root / name / 'frame.png').is_file() for name in names):
            self.skipTest('private failed departure evidence absent')
        for name in names:
            with self.subTest(frame=name):
                source = root / name / 'frame.png'
                reduced, rw, rh = read_diagnostic_png(source, stride=6)
                self.assertEqual(BrotatoVision().observe(reduced, rw, rh).status, 'player_ambiguous')
                original, width, height = read_diagnostic_png(source)
                vision = BrotatoVision().observe(original, width, height)
                self.assertTrue(vision.combat_likely)
                self.assertIsNotNone(vision.player)
                self.assertTrue(.48 < vision.player[0] < .52 and .45 < vision.player[1] < .50)
                # Drive the production menu->combat path with this original
                # frame. A mock runner proves entry without any game input.
                report, actions, controller, neural, captures = self.run_fixture(
                    partial=True, original_combat_frame=original)
                self.assertEqual(report['reason'], 'offline_combat_boundary_reached')
                self.assertEqual(actions, [])
                controller.tap_menu.assert_not_called()
                neural.combat_entry.assert_called_once()


if __name__ == '__main__':
    unittest.main()
