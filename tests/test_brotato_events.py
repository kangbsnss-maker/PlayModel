"""Evidence-bound HUD transition rules, without OCR or live-game side effects."""
from dataclasses import replace
import unittest

from playmodel.games.brotato.events import HUDEventTracker, VerifiedHUD


def snapshot(at=100, **changes):
    return replace(VerifiedHUD("run1", (1920, 1080), at, at + 5, "combat", True,
                               "reviewed-numeric-hud-v1", f"frames/{at}.png", "a" * 64,
                               hp=15, max_hp=15, xp_progress=0.2, level=1, currency=30), **changes)


class HUDEventTests(unittest.TestCase):
    def setUp(self):
        self.tracker = HUDEventTracker(max_gap_ns=100)
        self.tracker.observe(snapshot())

    def kinds(self, result):
        return [(event.kind, event.delta) for event in result.events]

    def test_loss_heal_xp_currency_preserve_both_sources_and_unknown_cause(self):
        result = self.tracker.observe(snapshot(110, hp=12, xp_progress=0.5, currency=33))
        self.assertEqual(self.kinds(result), [("hp_loss", -3), ("xp_progress", 0.3), ("currency_delta", 3)])
        self.assertTrue(all(event.cause == "unknown" for event in result.events))
        self.assertTrue(all(event.before.evidence_ref == "frames/100.png" for event in result.events))
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(120, hp=14, xp_progress=0.5, currency=33))),
                         [("heal", 2)])

    def test_max_hp_change_does_not_become_damage_or_healing(self):
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(110, hp=12, max_hp=12))), [])
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(120, hp=18, max_hp=20))), [])

    def test_currency_gain_is_not_xp_gain_or_item_identity(self):
        result = self.tracker.observe(snapshot(110, currency=33))
        self.assertEqual(self.kinds(result), [("currency_delta", 3)])
        result = self.tracker.observe(snapshot(120, currency=25))
        self.assertEqual(self.kinds(result), [("currency_delta", -8)])
        self.assertEqual(result.events[0].cause, "unknown")

    def test_level_wrap_does_not_invent_total_xp_or_loss(self):
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(110, level=2, xp_progress=0.1))),
                         [("level_up", 1)])
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(120, level=4, xp_progress=0.0))),
                         [("level_up", 2)])

    def test_xp_requires_known_same_level_and_ignores_unexplained_decrease(self):
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(110, xp_progress=0.1))), [])
        self.assertEqual(self.kinds(self.tracker.observe(snapshot(120, xp_progress=0.8, level=None))), [])

    def test_hp_zero_does_not_mean_death_verified_terminal_does_once(self):
        result = self.tracker.observe(snapshot(110, hp=0))
        self.assertEqual(self.kinds(result), [("hp_loss", -15)])
        result = self.tracker.observe(snapshot(120, phase="death", hp=None, max_hp=None))
        self.assertEqual(self.kinds(result), [("death", None)])
        self.assertEqual(self.tracker.observe(snapshot(130, phase="death")).events, ())

    def test_unknown_unverified_pause_menu_break_chain(self):
        for changes in ({"verified": False}, {"phase": "unknown"}, {"phase": "paused"}, {"phase": "menu"}):
            with self.subTest(changes=changes):
                self.tracker.observe(snapshot(100))
                self.assertEqual(self.tracker.observe(snapshot(110, **changes)).status, "baseline_broken")
                self.assertEqual(self.tracker.observe(snapshot(120, hp=3)).events, ())

    def test_episode_size_and_verifier_changes_start_new_baseline(self):
        for changes in ({"episode_id": "run2"}, {"resolution": (1280, 720)}, {"verifier_id": "new-calibration"}):
            with self.subTest(changes=changes):
                tracker = HUDEventTracker(max_gap_ns=100)
                tracker.observe(snapshot())
                result = tracker.observe(snapshot(110, hp=1, **changes))
                self.assertEqual(result.status, "context_changed")
                self.assertEqual(result.events, ())

    def test_long_gap_and_out_of_order_do_not_bridge_changes(self):
        self.assertEqual(self.tracker.observe(snapshot(201, hp=2)).status, "gap_exceeded")
        self.assertEqual(self.tracker.observe(snapshot(201, hp=1)).status, "out_of_order")
        self.assertEqual(self.tracker.observe(snapshot(202, hp=15)).events, ())
        self.assertEqual(self.tracker.observe(snapshot(203, available_at_ns=204)).status, "out_of_order")

    def test_level_regression_possible_reset_and_explicit_reset_break_chain(self):
        self.assertEqual(self.tracker.observe(snapshot(110, level=0, hp=1)).status, "possible_reset")
        self.assertEqual(self.tracker.observe(snapshot(120, hp=10)).events, ())
        self.tracker.reset()
        self.assertEqual(self.tracker.observe(snapshot(130, hp=1)).events, ())

    def test_unknown_numeric_values_are_not_zero_filled(self):
        self.assertEqual(self.tracker.observe(snapshot(110, hp=None, max_hp=None, xp_progress=None,
                                                       level=None, currency=None)).events, ())
        self.assertEqual(self.tracker.observe(snapshot(120)).events, ())

    def test_malformed_numeric_time_and_evidence_contracts_rejected(self):
        for changes in ({"hp": float("nan")}, {"hp": True}, {"hp": -1}, {"hp": 16},
                        {"max_hp": 0}, {"xp_progress": 1.1}, {"level": 1.5}, {"currency": -1},
                        {"observed_at_ns": True}, {"available_at_ns": 99}, {"verified": 1}, {"clock_domain": "wall_time"},
                        {"evidence_ref": ""}, {"evidence_sha256": "missing"}, {"resolution": [1920, 1080]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                snapshot(**changes)
        with self.assertRaises(ValueError):
            HUDEventTracker(max_gap_ns=True)


if __name__ == "__main__":
    unittest.main()
