import unittest

from playmodel.games.brotato.menu_transition import MenuTransitionGate, MenuTransitionTimeout


class MenuTransitionTests(unittest.TestCase):
    old = 'a' * 64
    new = 'b' * 64

    def test_identical_fresh_renders_never_repeat_enter(self):
        gate = MenuTransitionGate(timeout_ns=1000, max_observations=8)
        gate.record(self.old, 'enter', posted_at_ns=100)
        for now in (110, 160, 220):
            self.assertFalse(gate.allow(self.old, captured_at_ns=now, now_ns=now + 1))
        self.assertIsNotNone(gate.pending)
        self.assertEqual(gate.observations, 3)
        self.assertTrue(gate.allow(self.new, captured_at_ns=250, now_ns=251))
        self.assertIsNone(gate.pending)
        self.assertIn('requires_normal_verification', gate.last_reason)

    def test_arrows_have_no_confirmation_wait_even_with_same_sha(self):
        gate = MenuTransitionGate(clock=lambda: 100)
        for key in ('left', 'right', 'up', 'down'):
            self.assertTrue(gate.allow(self.old))
            gate.record(self.old, key)
            self.assertIsNone(gate.pending)

    def test_changed_but_pre_input_capture_does_not_acknowledge_enter(self):
        gate = MenuTransitionGate(timeout_ns=1000)
        gate.record(self.old, 'enter', posted_at_ns=100)
        self.assertFalse(gate.allow(self.new, captured_at_ns=90, now_ns=200))
        self.assertFalse(gate.allow(self.new, captured_at_ns=100, now_ns=210))
        self.assertTrue(gate.allow(self.new, captured_at_ns=211, now_ns=212))

    def test_elapsed_timeout_latches_and_does_not_allow_retry(self):
        gate = MenuTransitionGate(timeout_ns=50)
        gate.record(self.old, 'enter', posted_at_ns=100)
        with self.assertRaises(MenuTransitionTimeout):
            gate.allow(self.old, captured_at_ns=150, now_ns=150)
        self.assertIsNotNone(gate.pending)
        with self.assertRaises(MenuTransitionTimeout):
            gate.allow(self.new, captured_at_ns=151, now_ns=151)
        with self.assertRaises(RuntimeError):
            gate.record(self.old, 'enter', posted_at_ns=152)

    def test_observation_budget_bounds_a_frozen_clock(self):
        gate = MenuTransitionGate(timeout_ns=1000, max_observations=2)
        gate.record(self.old, 'enter', posted_at_ns=100)
        self.assertFalse(gate.allow(self.old, now_ns=101))
        self.assertFalse(gate.allow(self.old, now_ns=101))
        with self.assertRaises(MenuTransitionTimeout):
            gate.allow(self.old, now_ns=101)

    def test_verified_boundary_reset_releases_pending(self):
        gate = MenuTransitionGate(clock=lambda: 100)
        gate.record(self.old, 'enter')
        gate.reset()
        self.assertTrue(gate.allow(self.old))

    def test_invalid_clock_never_clears_pending(self):
        gate = MenuTransitionGate()
        gate.record(self.old, 'enter', posted_at_ns=100)
        for captured, now in ((110, 105), (90, 99)):
            with self.assertRaises(ValueError):
                gate.allow(self.new, captured_at_ns=captured, now_ns=now)
        self.assertIsNotNone(gate.pending)


if __name__ == '__main__':
    unittest.main()
