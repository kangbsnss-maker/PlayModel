"""Synthetic geometry only; no actual enemy/projectile detection claims."""
from dataclasses import replace
import time
import unittest

from playmodel.games.brotato.tactical_state import TacticalPlanner, execute_tactic, fallback_movement
from playmodel.games.brotato.vision import VisualCandidate, VisionObservation
from playmodel.games.brotato.state_features import STATS


class TacticalStateTests(unittest.TestCase):
    def setUp(self):
        self.now = time.perf_counter_ns()
        self.planner = TacticalPlanner(clock=lambda: self.now)

    def view(self, *, player=(.5, .5), hazards=(), **kwargs):
        return VisionObservation(player=player, player_confidence=.65, combat_likely=True,
            status='heuristic_observation', observed_at_ns=self.now - 1_000_000,
            hazards=tuple(VisualCandidate(x, y, radius, .2, 'unknown', 'synthetic_unverified')
                          for x, y, radius in hazards), **kwargs)

    def observe(self, *, build=None, **kwargs):
        vision = self.view(**kwargs)
        return self.planner.observe(vision, observed_at_ns=vision.observed_at_ns,
            available_at_ns=self.now, build_state=build)

    def step(self, seconds=.1, **kwargs):
        self.now += int(seconds * 1e9)
        return self.observe(**kwargs)

    def test_target_choice_changes_position_goal_and_expires_when_target_disappears(self):
        situation = self.observe(hazards=((.2, .5, .02), (.8, .5, .02)))
        targets = [key for key in situation['options'] if '@' in key]
        self.assertEqual(len(targets), 2)
        actions = [execute_tactic(key, situation, now_ns=self.now) for key in targets]
        self.assertNotEqual(actions[0], actions[1])
        old_signature = situation['signature']
        fresh = self.step(hazards=((.2, .5, .02), (.8, .5, .02)))
        self.assertEqual(fresh['signature'], old_signature)
        missing = self.step(hazards=((.2, .5, .02),))
        self.assertEqual(execute_tactic(targets[1], missing, now_ns=self.now), 0)

    def test_invalid_player_confidence_and_pickups_are_not_action_geometry(self):
        for confidence in (float('inf'), float('nan'), 1.1, 0):
            vision = replace(self.view(), player_confidence=confidence)
            planner = TacticalPlanner(clock=lambda: self.now)
            situation = planner.observe(vision, observed_at_ns=vision.observed_at_ns, available_at_ns=self.now)
            self.assertFalse(situation['world']['valid'])
        for x, confidence in ((float('nan'), .3), (1.1, .3), (.5, float('inf'))):
            planner = TacticalPlanner(clock=lambda: self.now)
            vision = self.view(pickups=(VisualCandidate(x, .5, .02, confidence, 'unknown', 'synthetic'),))
            situation = planner.observe(vision, observed_at_ns=vision.observed_at_ns, available_at_ns=self.now)
            self.assertNotIn('collect', situation['options'])
            self.assertEqual(situation['world']['pickups'], [])

    def test_approaching_crossing_and_collision_are_unverified_hypotheses(self):
        first = self.observe(hazards=((.75, .5, .02),))
        self.assertIsNone(first['world']['tracks'][0]['velocity'])
        second = self.step(hazards=((.68, .5, .02),))
        track = second['world']['tracks'][0]
        self.assertIn('approaching', track['pattern_hypotheses'])
        self.assertIn('crossing', track['pattern_hypotheses'])
        self.assertIn('fast_small_candidate', track['pattern_hypotheses'])
        self.assertGreater(track['collision_time_hypothesis_seconds'], 0)
        self.assertIsNone(track['ttc'])
        self.assertEqual(track['identity'], 'unknown')
        self.assertEqual(track['owner'], 'unknown')
        self.assertFalse(track['association_verified'])
        self.assertIn('avoid_crossing', second['options'])
        self.assertNotIn('approach', second['options'])
        self.assertNotEqual(first['signature'], second['signature'])

    def test_camera_translation_cancels_in_relative_motion(self):
        self.observe(player=(.4, .5), hazards=((.6, .5, .02),))
        result = self.step(player=(.44, .52), hazards=((.64, .52, .02),),
                           camera_status='shift_candidate', camera_shift=(.04, .02), camera_confidence=.4)
        velocity = result['world']['tracks'][0]['velocity']
        self.assertAlmostEqual(velocity[0], 0)
        self.assertAlmostEqual(velocity[1], 0)
        self.assertEqual(result['world']['camera_status'], 'shift_candidate')

    def test_turning_hypothesis_requires_prior_velocity(self):
        self.observe(hazards=((.7, .5, .04),))
        self.step(hazards=((.68, .5, .04),))
        result = self.step(hazards=((.7, .5, .04),))
        self.assertIn('turning', result['world']['tracks'][0]['pattern_hypotheses'])

    def test_ambiguous_crossing_association_does_not_invent_velocity(self):
        self.observe(hazards=((.6, .48, .02), (.6, .52, .02)))
        result = self.step(hazards=((.6, .5, .02),))
        self.assertIsNone(result['world']['tracks'][0]['velocity'])

    def test_dropout_teleport_and_long_gap_reset_tracking(self):
        self.observe(hazards=((.6, .5, .02),))
        missing = self.step(player=None)
        self.assertFalse(missing['world']['valid'])
        fresh = self.step(hazards=((.6, .5, .02),))
        self.assertIsNone(fresh['world']['tracks'][0]['velocity'])
        teleported = self.step(player=(.1, .1), hazards=((.2, .1, .02),))
        self.assertEqual(teleported['world']['reset_reason'], 'player_teleport_or_bad_association')
        self.assertIsNone(teleported['world']['tracks'][0]['velocity'])
        gap = self.step(.4, player=(.1, .1), hazards=((.2, .1, .02),))
        self.assertIsNone(gap['world']['tracks'][0]['velocity'])

    def test_future_stale_or_reordered_observation_cannot_produce_actions(self):
        good = self.observe(hazards=((.7, .5, .02),))
        vision = self.view()
        for at, available in ((self.now + 1, self.now + 1),
                              (self.now - 300_000_000, self.now),
                              (vision.observed_at_ns, self.now)):
            bad = self.planner.observe(replace(vision, observed_at_ns=at),
                    observed_at_ns=at, available_at_ns=available)
            self.assertFalse(bad['world']['valid'])
            self.assertEqual(bad['options'], {})
            self.assertEqual(fallback_movement(bad, now_ns=self.now), 0)
        self.assertEqual(execute_tactic('retreat', good, now_ns=self.now + 300_000_000), 0)
        self.assertEqual(execute_tactic('invented', good, now_ns=self.now), 0)

    def test_same_tactic_uses_moving_geometry_without_coordinate_signature_churn(self):
        left = self.observe(hazards=((.7, .5, .04),))
        # A fresh planner isolates semantic signature from transient track events.
        other = TacticalPlanner(clock=lambda: self.now)
        right_view = self.view(hazards=((.3, .5, .04),))
        right = other.observe(right_view, observed_at_ns=right_view.observed_at_ns, available_at_ns=self.now)
        self.assertEqual(left['signature'], right['signature'])
        self.assertEqual(execute_tactic('retreat', left, now_ns=self.now), 7)
        self.assertEqual(execute_tactic('retreat', right, now_ns=self.now), 3)
        self.assertEqual(execute_tactic('retreat', left, now_ns=self.now), 7)

    def build(self, stats, *, future=False):
        observed = self.now
        class Build:
            def features(self, when, *, available_at_ns):
                masks = [float(not future and name in stats) for name in STATS]
                return [0.] * 16 + masks + [0.] * 16
            def snapshot(self):
                return {'stats': stats, 'stats_status': 'two_independent_ocr_agreement', 'confidence': None,
                        'stats_sources': [{'observed_at_ns': observed - 10_000_000,
                                           'available_at_ns': observed + (1 if future else -1_000_000)}]}
        return Build()

    def test_observed_stats_change_spacing_and_candidates_but_never_true_range_or_hp(self):
        melee = self.observe(hazards=((.61, .5, .02),),
                             build=self.build({'melee_damage': 20, 'ranged_damage': 0, 'range': 999}))
        self.planner.reset()
        ranged = self.observe(hazards=((.61, .5, .02),),
                              build=self.build({'melee_damage': 0, 'ranged_damage': 20, 'speed': -5,
                                                'armor': -2, 'dodge': 1, 'hp_regeneration': 2, 'attack_speed': -1}))
        self.assertIn('approach', melee['options'])
        self.assertNotIn('approach', ranged['options'])
        self.assertNotEqual(melee['signature'], ranged['signature'])
        self.assertIsNone(ranged['world']['numeric_hp'])
        self.assertIsNone(ranged['world']['weapon_range_screen_units'])
        self.assertIsNone(ranged['world']['build']['confidence'])
        self.assertGreater(ranged['world']['build']['age_seconds'], 0)

    def test_future_build_stats_are_not_exposed_from_snapshot(self):
        result = self.observe(build=self.build({'melee_damage': 99, 'ranged_damage': 1}, future=True))
        self.assertEqual(result['world']['stats'], {})
        self.assertEqual(result['state']['stats'], {})
        self.assertEqual(result['world']['parameters']['profile'], 'unknown_or_balanced')

    def test_empty_scene_has_explicit_geometric_fallback_and_bounded_options(self):
        result = self.observe()
        self.assertEqual(list(result['options']), ['move_open'])
        self.assertEqual(fallback_movement(result, now_ns=self.now), 3)
        crowded = self.step(hazards=tuple((.6 + i * .001, .5, .01) for i in range(60)))
        self.assertLessEqual(len(crowded['options']), 8)
        self.assertEqual(len(crowded['world']['tracks']), 60)
        self.assertLessEqual(len(crowded['state']['risks']), 2)


if __name__ == '__main__':
    unittest.main()
