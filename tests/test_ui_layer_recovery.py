from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from playmodel.games.brotato.ui_layers import UiLayers
from playmodel.laya.safety_watch import watch_ui, user_stop


class UiLayerRecoveryTests(unittest.TestCase):
    def test_overlay_does_not_relabel_old_scene_as_current_or_grant_input(self):
        tracker = UiLayers()
        tracker.observe('shop', frame_ref='a', observed_at_ns=1)
        for name in ('pause', 'unknown'):
            layer = tracker.observe(name, frame_ref='b', observed_at_ns=2, pending_action=True)
            self.assertEqual(layer['base_scene'], 'shop')
            self.assertFalse(layer['base_scene_is_current'])
            self.assertFalse(layer['input_authorized'])
            self.assertTrue(layer['pending_action'])
        layer = tracker.observe('loot', frame_ref='c', observed_at_ns=3)
        self.assertEqual(layer['base_scene'], 'loot')
        self.assertTrue(layer['base_scene_is_current'])

    def test_transport_fault_keeps_observing_without_any_input_until_user_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'frame.png').write_bytes(b'capture fixture')
            capture, ocr, status = Mock(), Mock(), Mock()
            capture.read.return_value = ({'session_directory': str(root), 'capture_started_at_ns': 5}, b'', 1920, 1080)
            ocr.read.return_value = {}
            calls = [0]
            def stopped():
                calls[0] += 1
                return calls[0] > 7
            with patch('playmodel.games.brotato.menu.classify_scene', return_value=SimpleNamespace(scene='shop')):
                result = watch_ui(root / 'game.exe', root, stop_file=root / 'STOP', status=status,
                    incident={'category': 'partial_transport', 'release_state': 'unconfirmed'},
                    capture_factory=lambda: capture, ocr_factory=lambda: ocr,
                    stopped=stopped, sleep=lambda seconds: None)
            self.assertEqual(result, 0)
            observed = status.update.call_args_list[0].kwargs
            self.assertEqual(observed['phase'], 'waiting_safety')
            self.assertFalse(observed['automatic_rearm_allowed'])
            self.assertFalse(observed['input_authorized'])
            self.assertEqual(status.update.call_args.kwargs['status'], 'user_stopped')
            capture.close.assert_called()
            ocr.close.assert_called()

    def test_user_stop_is_never_turned_into_recovery(self):
        self.assertTrue(user_stop('OSError: Stopped by F8'))
        self.assertTrue(user_stop('human_intervention'))
        self.assertFalse(user_stop('capture timed out'))

    def test_f8_pressed_and_released_inside_ocr_remains_latched(self):
        import ctypes
        import threading
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, ocr, status = Mock(), Mock(), Mock()
            capture.read.return_value = ({'session_directory': str(root), 'capture_started_at_ns': 5}, b'', 1920, 1080)
            pressed, sampled = [False], threading.Event()
            def key_state(key):
                if pressed[0]:
                    sampled.set()
                    return 0x8000
                return 0
            def slow_ocr(source):
                pressed[0] = True
                self.assertTrue(sampled.wait(1))
                pressed[0] = False
                time.sleep(.02)
                return {}
            ocr.read.side_effect = slow_ocr
            native = SimpleNamespace(user32=SimpleNamespace(GetAsyncKeyState=key_state))
            with patch.object(ctypes, 'windll', native, create=True):
                self.assertEqual(watch_ui(root/'game.exe', root, stop_file=root/'STOP', status=status,
                    incident={}, capture_factory=lambda:capture, ocr_factory=lambda:ocr), 0)
            self.assertTrue((root/'STOP').exists())
            self.assertEqual(status.update.call_args.kwargs['status'], 'user_stopped')
