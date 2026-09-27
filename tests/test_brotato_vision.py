"""Synthetic contract tests; these do not measure real-game detector accuracy."""
from __future__ import annotations

import math
from pathlib import Path
import random
import struct
import unittest
import zlib

from playmodel.games.brotato.vision import (
    BrotatoVision, VisualCandidate, plan_navigation,
)


class Scene:
    width, height = 320, 180

    def __init__(self, *, hp=1.0, player=True, hud=True, alpha=255):
        self.pixels = bytearray((64, 78, 87, alpha)) * (self.width * self.height)
        self.alpha = alpha
        if hud:
            self.rect(4, 4, 57, 13, (0, 0, 0))
            self.rect(5, 5, 56, 11, (60, 60, 60))
            self.rect(5, 5, 5 + round(51 * hp), 11, (192, 0, 0))
            self.rect(4, 13, 57, 21, (0, 0, 0))
            self.rect(5, 14, 56, 20, (62, 62, 62))
            # Green HUD coin, deliberately larger than a field pickup.
            self.rect(7, 24, 12, 30, (55, 181, 64))
        if player:
            self.rect(156, 88, 164, 98, (250, 250, 250))

    def rect(self, x1, y1, x2, y2, rgb):
        r, g, b = rgb
        pixel = bytes((b, g, r, self.alpha))
        for y in range(y1, y2):
            self.pixels[(y * self.width + x1) * 4:(y * self.width + x2) * 4] = pixel * (x2 - x1)

    def observe(self, tracker=None, **kwargs):
        return (tracker or BrotatoVision()).observe(
            bytes(self.pixels), self.width, self.height, **kwargs)


class PerceptionTests(unittest.TestCase):
    def test_hud_coin_is_not_a_field_pickup(self):
        scene = Scene()
        scene.rect(220, 106, 224, 110, (45, 185, 52))
        result = scene.observe()
        self.assertTrue(result.navigation[2], result)
        self.assertEqual(len(result.pickups), 1)
        self.assertGreater(result.pickups[0].x, 0.6)
        self.assertGreater(result.navigation[0], 0)

    def test_unknown_player_never_proposes_movement(self):
        result = Scene(player=False).observe()
        self.assertEqual(result.status, "player_unknown")
        self.assertEqual(result.navigation, (0.0, 0.0, False))
        self.assertIsNone(result.player)

    def test_initial_multiple_similar_white_objects_are_ambiguous(self):
        scene = Scene()
        scene.rect(200, 88, 208, 98, (250, 250, 250))
        result = scene.observe()
        self.assertEqual(result.status, "player_ambiguous")
        self.assertFalse(result.navigation[2])

    def test_previous_player_position_resolves_same_size_distractor(self):
        tracker = BrotatoVision()
        initial = Scene().observe(tracker, observed_at_ns=1_000_000_000)
        scene = Scene(player=False)
        scene.rect(160, 90, 168, 100, (250, 250, 250))
        scene.rect(240, 90, 248, 100, (250, 250, 250))
        result = scene.observe(tracker, observed_at_ns=1_050_000_000)
        self.assertTrue(result.navigation[2])
        self.assertLess(result.player[0], 0.55)
        self.assertGreater(result.player_confidence, initial.player_confidence)

    def test_large_temporal_jump_rejected_then_reacquired_after_three_misses(self):
        tracker = BrotatoVision()
        Scene().observe(tracker)
        scene = Scene(player=False)
        scene.rect(252, 140, 260, 150, (250, 250, 250))
        for _ in range(3):
            self.assertFalse(scene.observe(tracker).navigation[2])
        self.assertTrue(scene.observe(tracker).navigation[2])

    def test_duplicate_and_reversed_timestamps_cannot_reuse_detection(self):
        tracker = BrotatoVision()
        scene = Scene()
        self.assertTrue(scene.observe(tracker, observed_at_ns=100).navigation[2])
        for timestamp in (100, 99):
            result = scene.observe(tracker, observed_at_ns=timestamp)
            self.assertEqual(result.status, "non_increasing_frame_time")
            self.assertFalse(result.navigation[2])

    def test_half_health_is_only_a_color_fill_approximation(self):
        result = Scene(hp=0.5).observe()
        self.assertAlmostEqual(result.hp_fill_fraction, 0.5, delta=0.06)
        self.assertIn("hp_color_fraction_not_ocr", result.reasons)
        self.assertFalse(hasattr(result, "health_points"))

    def test_empty_red_bar_does_not_claim_death_or_zero_hp(self):
        result = Scene(hp=0).observe()
        self.assertFalse(result.combat_likely)
        self.assertIsNone(result.hp_fill_fraction)
        self.assertFalse(result.navigation[2])

    def test_white_damage_flash_keeps_combat_but_health_unknown(self):
        scene = Scene(hp=0)
        scene.rect(5, 5, 18, 11, (255, 255, 255))
        result = scene.observe()
        self.assertTrue(result.combat_likely, result)
        self.assertIsNone(result.hp_fill_fraction)
        self.assertIn('hp_flash_fill_unknown', result.reasons)

    def test_white_flash_still_requires_hp_border_and_xp_strip(self):
        for missing in ('border', 'xp'):
            with self.subTest(missing=missing):
                scene = Scene(hp=0)
                scene.rect(5, 5, 18, 11, (255, 255, 255))
                if missing == 'border':
                    scene.rect(4, 4, 57, 5, (220, 220, 220))
                    scene.rect(4, 11, 57, 14, (220, 220, 220))
                    scene.rect(4, 20, 57, 21, (220, 220, 220))
                else:
                    scene.rect(5, 14, 56, 20, (220, 0, 0))
                result = scene.observe()
                self.assertFalse(result.combat_likely, result)
                self.assertIsNone(result.hp_fill_fraction)

    def test_white_hp_digits_or_unanchored_patch_do_not_imply_combat(self):
        for region in ((25, 6, 37, 9), (15, 5, 25, 11), (5, 6, 18, 9)):
            with self.subTest(region=region):
                scene = Scene(hp=0)
                scene.rect(*region, (255, 255, 255))
                self.assertFalse(scene.observe().combat_likely)

    def test_white_flash_does_not_bypass_menu_overlay(self):
        scene = Scene(hp=0)
        scene.rect(5, 5, 18, 11, (255, 255, 255))
        scene.rect(88, 38, 248, 145, (22, 22, 22))
        result = scene.observe()
        self.assertFalse(result.combat_likely)
        self.assertEqual(result.status, 'possible_menu_overlay')

    def test_paused_or_missing_hud_has_no_navigation(self):
        paused = Scene().observe(paused=True)
        menu = Scene(hud=False).observe()
        self.assertEqual(paused.status, "paused")
        for result in (paused, menu):
            self.assertFalse(result.combat_likely)
            self.assertFalse(result.navigation[2])

    def test_dark_overlay_rejected_even_with_intact_hud(self):
        scene = Scene()
        scene.rect(88, 38, 248, 145, (22, 22, 22))
        self.assertEqual(scene.observe().status, "possible_menu_overlay")

    def test_gdi_unused_alpha_and_real_transparency_have_distinct_contracts(self):
        scene = Scene(alpha=0)
        self.assertTrue(scene.observe(alpha_mode="ignore").navigation[2])
        result = scene.observe(alpha_mode="straight")
        self.assertFalse(result.navigation[2])
        self.assertIsNone(result.player)

    def test_malformed_buffers_dimensions_and_flags_are_invalid(self):
        tracker = BrotatoVision()
        scene = Scene()
        for width, height in ((float("nan"), 180), (320, float("inf")),
                              (True, 180), (0, 180), (320, 0), (10, 10)):
            with self.subTest(width=width, height=height):
                self.assertEqual(tracker.observe(bytes(scene.pixels), width, height).status,
                                 "invalid_frame")
        for pixels in (b"", bytes(scene.pixels[:-4]), [0] * 320 * 180 * 4):
            self.assertEqual(tracker.observe(pixels, 320, 180).status, "invalid_frame")
        for kwargs in ({"paused": "false"}, {"alpha_mode": "guess"},
                       {"observed_at_ns": float("nan")}, {"observed_at_ns": True}):
            self.assertEqual(scene.observe(**kwargs).status, "invalid_frame")

    def test_detected_contrast_has_unknown_identity_and_low_confidence(self):
        scene = Scene()
        scene.rect(184, 90, 190, 98, (210, 55, 35))
        result = scene.observe()
        self.assertTrue(result.hazards)
        self.assertTrue(all(item.label == "unknown" and item.confidence <= 0.2
                            for item in result.hazards))
        self.assertLess(result.navigation[0], 0)

    def test_reset_removes_previous_episode_position(self):
        tracker = BrotatoVision()
        Scene().observe(tracker, observed_at_ns=100)
        tracker.reset()
        scene = Scene(player=False)
        scene.rect(252, 140, 260, 150, (250, 250, 250))
        self.assertTrue(scene.observe(tracker, observed_at_ns=1).navigation[2])

    def test_black_face_features_can_split_white_player_into_adjacent_pieces(self):
        scene = Scene(player=False)
        scene.rect(156, 86, 164, 92, (250, 250, 250))
        scene.rect(156, 94, 164, 98, (250, 250, 250))
        result = scene.observe()
        self.assertTrue(result.navigation[2], result)
        self.assertAlmostEqual(result.player[0], 0.5, delta=0.02)

    def test_small_white_damage_text_is_not_hazard(self):
        scene = Scene()
        scene.rect(180, 82, 184, 86, (250, 250, 250))
        result = scene.observe()
        self.assertTrue(result.navigation[2])
        self.assertEqual(result.hazards, ())


class CameraTests(unittest.TestCase):
    @staticmethod
    def textured_scene(shift_x=0):
        scene = Scene(player=False)
        source = random.Random(8)
        gray = [source.randrange(50, 108) for _ in range(160 * 90)]
        for gy in range(20, 84):
            for gx in range(4, 155):
                value = gray[gy * 160 + gx - shift_x]
                scene.rect(gx * 2, gy * 2, gx * 2 + 2, gy * 2 + 2, (value,) * 3)
        scene.rect(156, 88, 164, 98, (250, 250, 250))
        return scene

    def test_uniform_background_has_unknown_camera_motion(self):
        tracker = BrotatoVision()
        Scene().observe(tracker)
        result = Scene().observe(tracker)
        self.assertEqual(result.camera_status, "unknown")
        self.assertIsNone(result.camera_shift)

    def test_common_translation_resets_track_but_preserves_current_relative_navigation(self):
        tracker = BrotatoVision()
        self.textured_scene().observe(tracker)
        result = self.textured_scene(1).observe(tracker)
        self.assertEqual(result.status, "heuristic_observation", result)
        self.assertEqual(result.camera_status, "shift_candidate")
        self.assertAlmostEqual(result.camera_shift[0], 1 / 160)
        self.assertEqual(result.camera_shift[1], 0)
        self.assertTrue(result.navigation[2])
        self.assertEqual(result.player_confidence, 0.45)
        self.assertTrue(self.textured_scene(1).observe(tracker).navigation[2])

    def test_unchanged_texture_reports_only_no_shift_detected(self):
        tracker = BrotatoVision()
        scene = self.textured_scene()
        scene.observe(tracker)
        result = scene.observe(tracker)
        self.assertEqual(result.camera_status, "no_shift_detected")
        self.assertTrue(result.navigation[2])


class LocalCaptureRegressionTests(unittest.TestCase):
    """Optional local evidence. The copyrighted source frames stay Git-ignored.

    Expected face/field regions were visually checked by the developer, not
    generated by this detector. These are calibration examples, not holdout data.
    """
    @staticmethod
    def capture(relative, tracker=None):
        path = Path(__file__).resolve().parents[1] / relative / "frame.png"
        if not path.is_file():
            raise unittest.SkipTest("Local calibration capture not present")
        data = path.read_bytes()
        offset, compressed = 8, []
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
            offset += 12 + length
            if kind == b"IHDR":
                width, height, depth, color, *_ = struct.unpack(">IIBBBBB", payload)
            if kind == b"IDAT":
                compressed.append(payload)
        assert (depth, color) == (8, 2)
        rows = zlib.decompress(b"".join(compressed))
        row_stride = width * 3 + 1
        assert all(rows[y * row_stride] == 0 for y in range(height))
        sampled = bytearray()
        for y in range(0, height, 6):
            for x in range(0, width, 6):
                index = y * row_stride + 1 + x * 3
                r, g, b = rows[index:index + 3]
                sampled.extend((b, g, r, 0))
        return (tracker or BrotatoVision()).observe(bytes(sampled), (width + 5) // 6, (height + 5) // 6)

    def test_initial_parachute_highlight_does_not_override_face(self):
        result = self.capture("data/raw/brotato-observations/20260926T162542Z-c39a7b5c")
        self.assertTrue(result.navigation[2], result)
        self.assertAlmostEqual(result.player[0], 847 / 1920, delta=0.02)
        self.assertAlmostEqual(result.player[1], 140 / 1080, delta=0.02)
        self.assertEqual(result.pickups, ())

    def test_gray_arena_faces_with_damage_text_and_enemy_candidates(self):
        examples = (("20260926T163842Z-0eee9928", 1200, 380, 0, 0),
                    ("20260926T163843Z-3e4938bc", 1720, 520, 1, 2),
                    ("20260926T163845Z-69c220cd", 1720, 720, 1, 4),
                    ("20260926T163846Z-debfb709", 1200, 720, 1, 0))
        for name, x, y, pickups, hazards in examples:
            with self.subTest(capture=name):
                result = self.capture("data/raw/brotato-movement-probe/" + name)
                self.assertTrue(result.navigation[2], result)
                self.assertAlmostEqual(result.player[0], x / 1920, delta=0.025)
                self.assertAlmostEqual(result.player[1], y / 1080, delta=0.035)
                self.assertGreaterEqual(len(result.pickups), pickups)
                self.assertGreaterEqual(len(result.hazards), hazards)

    def test_actual_shop_images_do_not_propose_combat_movement(self):
        for name in ("20260926T163710Z-ea89b534", "20260926T163712Z-eb470bdc"):
            with self.subTest(capture=name):
                result = self.capture("data/raw/brotato-movement-probe/" + name)
                self.assertFalse(result.combat_likely)
                self.assertFalse(result.navigation[2])

    def test_level_up_text_cannot_bootstrap_player_identity(self):
        relative = ("artifacts/brotato-pilot/20260926T164806Z-71aa3af0/ocr/"
                    "20260926T164808Z-8bf57c95")
        cold = self.capture(relative)
        self.assertEqual(cold.status, "player_ambiguous")
        self.assertFalse(cold.navigation[2])
        # The last saved raw frame independently supplies the recent face
        # location; it does not turn the level-up glyph into a player label.
        raw = (Path(__file__).resolve().parents[1] / "artifacts/brotato-pilot/"
               "20260926T164806Z-71aa3af0/frames/000026.bgra")
        if not raw.is_file():
            raise unittest.SkipTest("Local prior raw frame not present")
        tracker = BrotatoVision()
        previous = tracker.observe(raw.read_bytes(), 320, 180)
        self.assertTrue(previous.navigation[2])
        tracked = self.capture(relative, tracker)
        self.assertTrue(tracked.navigation[2], tracked)
        self.assertAlmostEqual(tracked.player[0], 950 / 1920, delta=0.025)
        self.assertAlmostEqual(tracked.player[1], 650 / 1080, delta=0.025)

    def test_wave_clear_and_loot_overlays_with_hp_hud_are_not_combat(self):
        root = "artifacts/brotato-pilot/20260926T164806Z-71aa3af0/ocr/"
        for name in ("20260926T164812Z-10186ae9", "20260926T164814Z-0863e826"):
            with self.subTest(capture=name):
                result = self.capture(root + name)
                self.assertFalse(result.combat_likely)
                self.assertFalse(result.navigation[2])

    def test_actual_white_hp_flash_and_following_shop_are_distinct(self):
        root = ('artifacts/recurrent-recovery/20260927T145646/'
                'training-25da2a887fba4a8b84f89f9106478ca5/segments/'
                '20260927T055725Z-4f3d03b1/pilots/'
                'neural-5835b15382e84203ae0c49ab56c1cf46/ocr/')
        flash = self.capture(root + '20260927T055837Z-60b15b57')
        self.assertTrue(flash.combat_likely, flash)
        self.assertIsNone(flash.hp_fill_fraction)
        for name in ('20260927T055825Z-a12aac18', '20260927T055839Z-e3ccd66e'):
            with self.subTest(frame=name):
                self.assertTrue(self.capture(root + name).combat_likely)
        shop = self.capture(root + '20260927T055842Z-9008d982')
        self.assertFalse(shop.combat_likely)


class NavigationTests(unittest.TestCase):
    @staticmethod
    def candidate(x, y, confidence=0.2):
        return VisualCandidate(x, y, 0.01, confidence, "unknown", "synthetic_test")

    def test_nearer_threat_overrides_pickup_in_same_direction(self):
        result = plan_navigation((0.5, 0.5), (self.candidate(0.7, 0.5, 0.4),),
                                 (self.candidate(0.55, 0.5),))
        self.assertTrue(result[2])
        self.assertLess(result[0], -0.9)

    def test_nearer_threat_wins_opposing_distant_candidate(self):
        result = plan_navigation((0.5, 0.5), hazards=(self.candidate(0.52, 0.5),
                                                    self.candidate(0.43, 0.5)))
        self.assertLess(result[0], 0)

    def test_screen_edge_moves_inward_before_outward_pickup(self):
        result = plan_navigation((0.01, 0.5), (self.candidate(0.001, 0.5, 0.4),))
        self.assertGreater(result[0], 0)

    def test_invalid_or_unknown_navigation_inputs(self):
        for player in (None, (float("nan"), 0), (0, float("inf")), (True, 0), (2, 0), (0,)):
            self.assertEqual(plan_navigation(player), (0.0, 0.0, False))
        for aspect in (float("nan"), 0, -1, True):
            self.assertFalse(plan_navigation((0.5, 0.5), aspect_ratio=aspect)[2])
        invalid = self.candidate(float("nan"), 0.5)
        self.assertFalse(plan_navigation((0.5, 0.5), (invalid,))[2])

    def test_vectors_are_normalized_and_finite(self):
        for player in ((0.5, 0.5), (0, 0), (0.4, 0.5), (1, 1)):
            dx, dy, valid = plan_navigation(player)
            self.assertTrue(valid)
            self.assertTrue(math.isfinite(dx) and math.isfinite(dy))
            self.assertAlmostEqual(math.hypot(dx, dy), 1)

    def test_exact_overlap_without_escape_direction_is_invalid(self):
        self.assertFalse(plan_navigation((0.5, 0.5), hazards=(self.candidate(0.5, 0.5),))[2])


if __name__ == "__main__":
    unittest.main()
