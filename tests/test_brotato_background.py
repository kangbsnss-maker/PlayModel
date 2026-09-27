import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from playmodel.games.brotato.background import (
    BackgroundController, MINIMUM_POST_BUDGET_NS, PrePostMovementDeadline,
)
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
            with self.assertRaisesRegex(PrePostMovementDeadline, 'identity check') as raised:
                control.set_movement_before({0x44}, deadline_ns=25)
        control._key.assert_not_called()
        self.assertEqual(control.held, {0x57})
        self.assertEqual(control.last_movement_timing['posted_keys'], 0)
        self.assertTrue(raised.exception.timing['identity_check_passed'])
        self.assertFalse(raised.exception.timing['native_post_attempted'])
        self.assertEqual(raised.exception.timing['attempted_posts'], 0)

    def native_control(self):
        control = self.control()
        del control._key
        control._scan, control._message = Mock(return_value=17), Mock()
        return control

    def test_first_key_mapping_expiry_is_definitely_unsent_with_held_keys_preserved(self):
        control = self.native_control()
        control.held = {0x57}
        with patch('playmodel.games.brotato.background.time.perf_counter_ns', side_effect=[10, 15, 30]):
            with self.assertRaises(PrePostMovementDeadline):
                control.set_movement_before({0x44}, deadline_ns=25)
        control._message.assert_not_called()
        self.assertEqual(control.held, {0x57})
        self.assertIsNone(control.last_movement_timing['posts_started_at_ns'])
        control.release()
        self.assertEqual(control.held, set())

    def test_native_post_error_and_partial_posts_never_use_unsent_type(self):
        control = self.native_control()
        control._message.side_effect = OSError('native post failed')
        with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                   side_effect=[10_000_000, 11_000_000, 12_000_000, 13_000_000]):
            with self.assertRaises(OSError) as raised:
                control.set_movement_before({0x44}, deadline_ns=25_000_000)
        self.assertNotIsInstance(raised.exception, PrePostMovementDeadline)
        self.assertTrue(control.last_movement_timing['native_post_attempted'])
        self.assertEqual(control.last_movement_timing['attempted_posts'], 1)
        self.assertEqual(control.last_movement_timing['posted_keys'], 0)
        control = self.native_control()
        control.held = {0x57}
        with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                   side_effect=[10_000_000, 11_000_000, 12_000_000, 30_000_000]):
            with self.assertRaises(OSError) as raised:
                control.set_movement_before({0x44}, deadline_ns=25_000_000)
        self.assertNotIsInstance(raised.exception, PrePostMovementDeadline)
        self.assertEqual(control.last_movement_timing['posted_keys'], 1)
        self.assertNotIn('cancellation_reason', control.last_movement_timing)

    def test_post_bookkeeping_reuses_final_deadline_check_clock(self):
        control = self.native_control()
        # Fourth timestamp belongs to after posting; no new clock call occurs
        # between the final freshness check and the native call.
        with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                   side_effect=[10_000_000, 11_000_000, 12_000_000, 30_000_000]) as clock:
            control._message.side_effect = lambda *args: self.assertEqual(clock.call_count, 3)
            control.set_movement_before({0x44}, deadline_ns=25_000_000)
        self.assertEqual(control.last_movement_timing['posts_started_at_ns'], 12_000_000)

    def test_first_post_reserve_boundary_uses_original_deadline(self):
        self.assertEqual(MINIMUM_POST_BUDGET_NS, 2_000_000)
        deadline = 25_000_000
        for remaining in (1, 1_999_999, 2_000_000, 2_000_001):
            with self.subTest(remaining=remaining):
                control = self.native_control()
                checked = deadline - remaining
                with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                           side_effect=[10_000_000, 11_000_000, checked, deadline - 1]) as clock:
                    if remaining < MINIMUM_POST_BUDGET_NS:
                        with self.assertRaises(PrePostMovementDeadline) as raised:
                            control.set_movement_before({0x44}, deadline_ns=deadline)
                        proof = raised.exception.timing
                        self.assertEqual(proof['cancellation_reason'], 'post_budget_insufficient')
                        self.assertEqual(proof['minimum_post_budget_ns'], MINIMUM_POST_BUDGET_NS)
                        self.assertEqual(proof['deadline_checked_at_ns'], checked)
                        self.assertEqual(proof['deadline_ns'], deadline)
                        self.assertEqual(proof['call_id'], 1)
                        self.assertTrue(proof['identity_check_passed'])
                        self.assertFalse(proof['native_post_attempted'])
                        self.assertEqual(proof['attempted_posts'], 0)
                        self.assertEqual(proof['posted_keys'], 0)
                        self.assertIsNone(proof['posts_started_at_ns'])
                        self.assertIsNone(proof['posts_finished_at_ns'])
                        self.assertEqual(clock.call_count, 3)
                        control._message.assert_not_called()
                        self.assertEqual(control.held, set())
                    else:
                        control._message.side_effect = lambda *args: self.assertEqual(clock.call_count, 3)
                        control.set_movement_before({0x44}, deadline_ns=deadline)
                        control._message.assert_called_once()
                        self.assertEqual(control.held, {0x44})

    def test_slow_identity_reserve_preserves_held_keys_until_normal_release(self):
        control = self.native_control()
        control.held = {0x57}
        # Reproduce the observed 24.8459ms identity stage under the unchanged 25ms budget.
        with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                   side_effect=[0, 24_845_900, 24_846_000]) as clock:
            with self.assertRaises(PrePostMovementDeadline) as raised:
                control.set_movement_before({0x44}, deadline_ns=25_000_000)
        self.assertEqual(clock.call_count, 3)
        control._message.assert_not_called()
        self.assertEqual(control.held, {0x57})
        self.assertEqual(raised.exception.timing['cancellation_reason'], 'post_budget_insufficient')
        self.assertEqual(control.last_movement_timing['attempted_posts'], 0)
        control.release()
        self.assertEqual(control.held, set())
        control._message.assert_called_once()
        self.assertEqual(control._message.call_args.args[:2], (0x0101, 0x57))

    def test_reserve_is_not_reapplied_after_first_native_post(self):
        control = self.native_control()
        control.held = {0x57}
        with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                   side_effect=[10_000_000, 11_000_000, 12_000_000, 24_000_000, 24_100_000]):
            control.set_movement_before({0x44}, deadline_ns=25_000_000)
        self.assertEqual(control._message.call_count, 2)
        self.assertEqual(control.last_movement_timing['attempted_posts'], 2)
        self.assertEqual(control.last_movement_timing['posted_keys'], 2)
        self.assertEqual(control.held, {0x44})

    def test_unchanged_held_state_needs_no_first_post_reserve(self):
        for keys in (set(), {0x57}):
            with self.subTest(keys=keys):
                control = self.native_control()
                control.held = keys.copy()
                with patch('playmodel.games.brotato.background.time.perf_counter_ns',
                           side_effect=[0, 24_845_900, 24_846_000]) as clock:
                    control.set_movement_before(keys, deadline_ns=25_000_000)
                self.assertEqual(clock.call_count, 3)
                control._scan.assert_not_called()
                control._message.assert_not_called()
                self.assertEqual(control.held, keys)
                self.assertFalse(control.last_movement_timing['native_post_attempted'])
                self.assertEqual(control.last_movement_timing['posted_keys'], 0)

    def test_identity_f8_or_minimized_failure_never_use_unsent_type(self):
        for reason in ('Stopped by F8', 'Game identity changed or game was minimized'):
            control = self.control()
            control.check.side_effect = OSError(reason)
            with self.assertRaises(OSError) as raised:
                control.set_movement_before({0x44}, deadline_ns=25)
            self.assertNotIsInstance(raised.exception, PrePostMovementDeadline)
            self.assertFalse(control.last_movement_timing['identity_check_passed'])

    def test_key_mapping_delay_is_rechecked_before_native_post(self):
        control = object.__new__(BackgroundController)
        control._scan, control._message = Mock(return_value=17), Mock()
        with patch('playmodel.games.brotato.background.time.perf_counter_ns', return_value=30):
            with self.assertRaisesRegex(OSError, 'before key post'):
                control._key(0x44, False, deadline_ns=25)
        control._message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
