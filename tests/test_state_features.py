"""Local build evidence and strict unknown masks; no live game input."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from playmodel.games.brotato.shop_learning import ShopObservation, ShopOffer, VerifiedShop
from playmodel.games.brotato.state_features import (
    BoundedBuildState, STATS, STAT_LABELS, STATS_ROI, SCHEMA, make_stats_observation, parse_stats,
)
from playmodel.games.brotato.stats_roi_ocr import enlarged_stats_crop, read_stats_roi
from playmodel.games.brotato.capture import _png


def record(index, rows=("Max HP 16", "% Speed -5", "% Damage 0"), **changes):
    at = 1_000_000_000 + index * 100_000_000
    return {"frame_ref": f"frame-{index}.png", "source_png_sha256": "a" * 64,
            "source_pixels_sha256": "b" * 64, "game_build_id": "23429717", "language": "en",
            "observed_at_ns": at, "ocr_started_at_ns": at + 1_000_000,
            "available_at_ns": at + 2_000_000, "scene": "shop", "raw_text": rows,
            "recognition_path": "persistent_local_ocr", **changes}


def selection_evidence():
    return {"frame_ref": "weapon-selection.png", "frame_sha256": "c" * 64,
            "observed_at_ns": 900_000_000, "available_at_ns": 910_000_000,
            "verified": True, "independent_of_policy": True}


class BuildFeatureTests(unittest.TestCase):
    def test_signed_values_percent_and_zero_are_distinct_from_unknown(self):
        state = BoundedBuildState("run")
        self.assertTrue(state.observe_stats_pair(record(1), record(2)))
        features = state.features(1_300_000_000)
        self.assertEqual(len(features), 48)
        self.assertLess(features[STATS.index("speed")], 0)
        self.assertEqual(features[STATS.index("damage")], 0)
        self.assertEqual(features[16 + STATS.index("damage")], 1)
        self.assertEqual(features[16 + STATS.index("armor")], 0)
        self.assertEqual(features[40], 1)
        self.assertGreater(features[41], 0)
        self.assertEqual(features[-1], 1)
        self.assertTrue(all(-1 <= value <= 1 for value in features))

    def test_all_sixteen_literal_labels_and_numeric_rejections(self):
        result = parse_stats([f"{label}{index + 1}" for index, label in enumerate(STAT_LABELS)])
        self.assertTrue(all(value is not None for value in result.values()))
        for text in ("MaxHP1O", "MaxHP01", "MaxHP--1", "MaxHP-1", "MaxHP0", "MaxHP1/2",
                     "Speed5", "%SpeedO", "%Speed5%", "%Speed+", "Armor1,000"):
            self.assertFalse(any(value is not None for value in parse_stats([text]).values()), text)
        self.assertEqual(parse_stats(["％ Life Steal 2.5"])["life_steal"], 2.5)
        self.assertIsNone(parse_stats(["Armor5", "Armor5"])["armor"])

    def test_actual_stored_shop_ocr_strings_parse_without_numeric_repair(self):
        # Literal rows from the preserved 20260927T060630Z-ac62e853 crop, scale2.
        rows = ("MaxHP16", "HPRegeneration0", "%LifeSteal0", "%Damage0", "MeleeDamage2",
                "RangedDamage0", "EIementaIDamage0", "%AttackSpeed0", "%Critchance0",
                "Engineering0", "Range0", "Armor0", "%D0dge0", "%Speed5", "Luck0", "Harvesting9")
        state = BoundedBuildState("run")
        self.assertTrue(state.observe_stats_pair(record(1, rows), record(2, rows)))
        self.assertEqual(sum(state.features(1_300_000_000)[16:32]), 16)
        self.assertEqual(state.snapshot()["stats"]["max_hp"], 16)
        self.assertEqual(state.snapshot()["stats"]["harvesting"], 9)
        self.assertIsNone(state.snapshot()["confidence"])
        self.assertFalse(state.snapshot()["current_hp_known"])

    def test_disagreement_unknown_does_not_retain_prior_value(self):
        state = BoundedBuildState("run")
        state.observe_stats_pair(record(1), record(2))
        state.observe_stats_pair(record(3, ("MaxHP16",)), record(4, ("MaxHP18",)))
        self.assertEqual(sum(state.features(1_500_000_000)[16:32]), 0)
        self.assertIsNone(state.snapshot()["stats"]["max_hp"])

    def test_future_observation_and_late_availability_never_enter_policy(self):
        state = BoundedBuildState("run")
        state.observe_stats_pair(record(1), record(2))
        unknown = (0.0,) * 47 + (1.0,)
        self.assertEqual(state.features(1_190_000_000), unknown)
        self.assertEqual(state.features(1_200_000_000), unknown)
        self.assertGreater(sum(state.features(1_200_000_000, available_at_ns=1_202_000_000)[16:32]), 0)

    def test_cached_foreign_and_out_of_order_pairs_are_unknown(self):
        bad = [record(2, recognition_path="exact_image_cache"), record(2, game_build_id="other"),
               record(2, language="zh"), record(2, source_png_sha256=None),
               record(2, ocr_started_at_ns=1), record(2, frame_ref="frame-1.png")]
        for second in bad:
            state = BoundedBuildState("run")
            self.assertFalse(state.observe_stats_pair(record(1), second))
            self.assertEqual(sum(state.features(1_400_000_000)[16:32]), 0)

    def test_source_snapshot_immutable_and_invalidation_no_rewards(self):
        state = BoundedBuildState("run")
        left, right = record(1), record(2)
        state.observe_stats_pair(left, right)
        right["raw_text"] = ["MaxHP999"]
        snapshot = state.snapshot()
        self.assertEqual(snapshot["schema"], SCHEMA)
        self.assertEqual(snapshot["stats_sources"][-1]["source_png_sha256"], "a" * 64)
        snapshot["stats"]["max_hp"] = 999
        self.assertEqual(state.snapshot()["stats"]["max_hp"], 16)
        state.invalidate_stats("upgrade_applied_waiting_for_observation")
        self.assertEqual(sum(state.features(1_400_000_000)[16:32]), 0)
        self.assertNotIn("reward", state.snapshot())
        json.dumps(state.snapshot(), allow_nan=False)

    def test_verified_weapon_seed_and_unknown_total(self):
        state = BoundedBuildState("run")
        self.assertFalse(state.seed_weapon("Pistol", {**selection_evidence(), "verified": False}))
        self.assertTrue(state.seed_weapon("Pistol", selection_evidence()))
        features = state.features(1_000_000_000)
        self.assertTrue(any(features[32:40]))
        self.assertEqual(features[42], 1)
        self.assertEqual(features[44], 0)  # inventory coverage unknown
        self.assertEqual(features[45], 0)  # total inventory not guessed
        self.assertFalse(state.seed_weapon("SMG", selection_evidence()))
        self.assertEqual(state.snapshot()["weapons"][0]["tier"], None)

    def test_level_up_uses_its_own_calibrated_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "frame.png"
            source.write_bytes(b"preserved test source")
            def word(text, x, y):
                return {"words": [{"text": text, "x": x, "y": y, "width": 80, "height": 20}]}
            raw = {"lines": [word("MaxHP16", 1585, 403), word("+9Armor", 50, 450)],
                   "processing_started_at_ns": 110, "recognition_path": "persistent_local_ocr"}
            obs = make_stats_observation(raw, frame_ref=str(source), pixels_sha256="b" * 64,
                observed_at_ns=100, available_at_ns=120, scene="level_up")
            self.assertEqual(obs["raw_text"], ("MaxHP16",))
            self.assertEqual(obs["source_png_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_same_bound_pair_does_not_erase_richer_roi_stats(self):
        state = BoundedBuildState("run")
        state.observe_stats_pair(record(1), record(2))
        self.assertTrue(state.observe_stats_pair(record(1, ("MaxHP",)), record(2, ("MaxHP",))))
        self.assertEqual(state.snapshot()["stats"]["max_hp"], 16)


class InventoryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def shop(self, index, *, count=1, currency=80, purchased=False, inventory_hash="1"):
        rows = []
        for offset in (0, 1):
            source = Path(self.tmp.name) / f"{index + offset}.png"
            source.write_bytes(f"source {index + offset}".encode())
            at = 1_000_000_000 + (index + offset) * 100_000_000
            offers = tuple(ShopOffer(slot, "Laser Gun" if slot == 0 else f"Other{slot}", "Gun", "weapon",
                                    ("Damage:40",), 10, f"weapon:{slot}", not purchased if slot == 0 else True,
                                    "e" * 64) for slot in range(4))
            rows.append(ShopObservation(str(source), "b" * 64, "23429717", at, at + 2_000_000,
                at + 1_000_000, currency, 1, count, 6, offers, ("MaxHP16",), "d" * 64,
                inventory_hash * 64))
        return VerifiedShop(*rows, True, True)

    def test_verified_purchase_updates_names_and_unexplained_inventory_clears_them(self):
        state = BoundedBuildState("run")
        state.seed_weapon("Pistol", selection_evidence())
        before = self.shop(1)
        state.observe_shop_pair(before)
        self.assertEqual(state.features(1_300_000_000)[44], 1)
        after = self.shop(3, count=2, currency=70, purchased=True, inventory_hash="2")
        self.assertTrue(state.apply_verified_purchase(before, after, 0))
        self.assertEqual([w["name"] for w in state.snapshot()["weapons"]], ["Pistol", "Laser Gun"])
        self.assertEqual(state.features(1_500_000_000)[44], 1)
        state.observe_shop_pair(self.shop(5, count=2, inventory_hash="3"))
        self.assertFalse(state.snapshot()["weapons"])
        self.assertEqual(state.features(1_700_000_000)[44], 0)
        self.assertGreater(state.features(1_700_000_000)[46], 0)

    def test_unverified_purchase_not_added_and_partial_inventory_explicit(self):
        state = BoundedBuildState("run")
        state.seed_weapon("Pistol", selection_evidence())
        before = self.shop(1, count=2)
        state.observe_shop_pair(before)
        self.assertEqual(state.features(1_300_000_000)[44], .5)
        wrong_spend = self.shop(3, count=3, currency=79, purchased=True, inventory_hash="2")
        self.assertFalse(state.apply_verified_purchase(before, wrong_spend, 0))
        self.assertEqual(len(state.snapshot()["weapons"]), 1)


class StatsRoiTests(unittest.TestCase):
    def test_enlargement_exactly_matches_original_pixel_replication(self):
        # Distinct channel/pixel values catch byte-, channel- and row-order bugs.
        pixels = bytes(range(256)) * (1920 * 1080 * 4 // 256)
        for scene in ("shop", "level_up"):
            x0, y0, x1, y1 = STATS_ROI[scene]
            for scale in (1, 2, 3):
                expected = bytearray()
                for y in range(y0, y1):
                    row = pixels[(y * 1920 + x0) * 4:(y * 1920 + x1) * 4]
                    expected.extend(b"".join(row[i:i + 4] * scale for i in range(0, len(row), 4)) * scale)
                actual, _, width, height = enlarged_stats_crop(pixels, scene, scale=scale)
                self.assertEqual(actual, _png(width, height, bytes(expected), compression_level=1))

    def test_crop_calibration_and_malformed_source(self):
        with self.assertRaises(ValueError):
            enlarged_stats_crop(b"x", "shop")
        with self.assertRaises(ValueError):
            enlarged_stats_crop(bytes(1920 * 1080 * 4), "combat")

    def test_reader_cache_disabled_and_original_derivations_bound(self):
        class Reader:
            cache_seconds = 2.0
            def read(self, path):
                self.seen_cache = self.cache_seconds
                at = time.perf_counter_ns()
                return {"processing_started_at_ns": at, "available_at_ns": at,
                        "recognition_path": "persistent_local_ocr", "language": "zh-Hans-CN",
                        "lines": [{"words": [{"text": "MaxHP16", "x": 0, "y": 0, "width": 180, "height": 40}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            pixels = bytes((10, 20, 30, 255)) * 1920 * 1080
            path = Path(tmp) / "frame.png"
            path.write_bytes(_png(1920, 1080, pixels))
            reader = Reader()
            result = read_stats_roi(reader, path, pixels=pixels, observed_at_ns=time.perf_counter_ns(),
                                    frame_id=str(path), game_build_id="23429717", scene="shop")
            self.assertEqual(reader.seen_cache, 0)
            self.assertEqual(reader.cache_seconds, 2)
            self.assertEqual(result["raw_text"], ["MaxHP16"])
            self.assertEqual(result["source_png_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(result["evidence_files"]), 3)
            for proof in result["evidence_files"]:
                self.assertEqual(hashlib.sha256(Path(proof["path"]).read_bytes()).hexdigest(), proof["sha256"])
            with self.assertRaises(FileExistsError):
                read_stats_roi(reader, path, pixels=pixels, observed_at_ns=time.perf_counter_ns(),
                               frame_id=str(path), game_build_id="23429717", scene="shop")


if __name__ == "__main__":
    unittest.main()
