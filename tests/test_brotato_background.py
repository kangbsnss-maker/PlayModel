import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from playmodel.games.brotato.background import BackgroundController
from playmodel.games.brotato.interaction import WindowController


class BackgroundInputTests(unittest.TestCase):
    def control(self):
        control = object.__new__(BackgroundController)
        control.held = set()
        control.check = Mock()
        control._key = Mock()
        return control

    def test_direction_change_releases_only_owned_keys(self):
        control = self.control()
        control.set_movement({0x57, 0x44})
        control._key.reset_mock()
        control.set_movement({0x44, 0x53})
        self.assertEqual(control._key.call_args_list[0].args, (0x57, True))
        self.assertEqual(control._key.call_args_list[1].args, (0x53, False))
        self.assertEqual(control.held, {0x44, 0x53})

    def test_release_attempts_every_owned_key_after_one_failure(self):
        control = self.control()
        control.held = {0x41, 0x57}
        control._key.side_effect = [OSError("queue failed"), None]
        with self.assertRaises(OSError):
            control.release()
        self.assertEqual(control._key.call_count, 2)
        self.assertEqual(len(control.held), 1)

    def test_foreground_and_mouse_paths_are_retired(self):
        old = object.__new__(WindowController)
        for method in ("activate", "hover", "click", "set_movement", "release"):
            with self.subTest(method=method), self.assertRaisesRegex(OSError, "retired"):
                getattr(old, method)()
        background = self.control()
        for method in ("hover", "click"):
            with self.subTest(method=method), self.assertRaisesRegex(OSError, "unverified"):
                getattr(background, method)(1, 2, (100, 100))
        background._key.assert_not_called()

    def test_stopped_controller_sends_no_movement(self):
        control = self.control()
        control.check.side_effect = OSError("Stopped by F8")
        with self.assertRaises(OSError):
            control.set_movement({0x44})
        control._key.assert_not_called()

    def test_identical_os_image_name_does_not_repeat_filesystem_resolution(self):
        control = object.__new__(WindowController)
        with patch.object(Path, 'resolve', side_effect=[Path('/first/game.exe'), Path('/second/game.exe')]) as resolve:
            first = control._canonical_image_path('reported-image-one')
            self.assertEqual(control._canonical_image_path('reported-image-one'), first)
            self.assertEqual(resolve.call_count, 1)
            self.assertNotEqual(control._canonical_image_path('reported-image-two'), first)
            self.assertEqual(resolve.call_count, 2)

    def test_slow_identity_check_posts_no_keys_after_original_deadline(self):
        control = self.control()
        control.held = {0x57}
        with patch('playmodel.games.brotato.background.time.perf_counter_ns', side_effect=[10, 30]):
            with self.assertRaisesRegex(OSError, 'identity check'):
                control.set_movement_before({0x44}, deadline_ns=25)
        control._key.assert_not_called()
        self.assertEqual(control.held, {0x57})
        self.assertEqual(control.last_movement_timing['posted_keys'], 0)

    def test_key_mapping_delay_is_rechecked_before_native_post(self):
        control = object.__new__(BackgroundController)
        control._scan, control._message = Mock(return_value=17), Mock()
        with patch('playmodel.games.brotato.background.time.perf_counter_ns', return_value=30):
            with self.assertRaisesRegex(OSError, 'before key post'):
                control._key(0x44, False, deadline_ns=25)
        control._message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
