"""Synthetic contracts plus optional local capture regressions; no live input."""
from dataclasses import replace
from pathlib import Path
import struct
import unittest
import zlib

from playmodel.games.brotato.scroll import (
    ACCESSIBILITY_1080, MenuScrollTracker, ScrollbarCalibration, detect_scrollbar,
)


CALIBRATION = ScrollbarCalibration("test.menu", 40, 120, (30, 10, 35, 110), "synthetic-v1")


def frame(thumb=(10, 60), *, extra=None):
    pixels = bytearray(bytes((15, 15, 15, 0)) * 40 * 120)
    for y in range(10, 110):
        for x in range(30, 35):
            light = thumb is not None and thumb[0] <= y < thumb[1]
            light = light or extra is not None and extra[0] <= y < extra[1]
            value = 192 if light else 30
            start = (y * 40 + x) * 4
            pixels[start:start + 4] = bytes((value, value, value, 0))
    return bytes(pixels)


def observe(thumb=(10, 60), at=1, **kwargs):
    return detect_scrollbar(frame(thumb), 40, 120, scene_id="test.menu", observed_at_ns=at,
                            calibration=CALIBRATION, **kwargs)


def own_capture_bgra(path):
    """Decode only this project's RGB/filter-zero diagnostic PNG, for fixtures."""
    png = path.read_bytes()
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", png[16:29])
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("unsupported local fixture")
    offset, parts = 8, []
    while offset < len(png):
        length = struct.unpack(">I", png[offset:offset + 4])[0]
        if png[offset + 4:offset + 8] == b"IDAT":
            parts.append(png[offset + 8:offset + 8 + length])
        offset += length + 12
    raw = zlib.decompress(b"".join(parts))
    stride = width * 3 + 1
    if len(raw) != stride * height or any(raw[y * stride] for y in range(height)):
        raise ValueError("unsupported local fixture rows")
    pixels = bytearray(width * height * 4)
    for y in range(height):
        row = raw[y * stride + 1:(y + 1) * stride]
        start, end = y * width * 4, (y + 1) * width * 4
        pixels[start:end:4], pixels[start + 1:end:4], pixels[start + 2:end:4] = row[2::3], row[1::3], row[0::3]
    return bytes(pixels), width, height


class ScrollGeometryTests(unittest.TestCase):
    def test_top_middle_bottom_and_remaining_content(self):
        for thumb, edge, up, down, position in (
                ((10, 60), "top", False, True, 0),
                ((35, 85), "middle", True, True, 0.5),
                ((60, 110), "bottom", True, False, 1)):
            with self.subTest(edge=edge):
                result = observe(thumb)
                self.assertEqual(result.status, "observed")
                self.assertEqual((result.edge, result.can_scroll_up, result.can_scroll_down), (edge, up, down))
                self.assertEqual(result.position, position)
                self.assertEqual(len(result.frame_sha256), 64)

    def test_near_edge_is_unknown_not_a_claim_of_list_end(self):
        result = observe((55, 105))
        self.assertEqual(result.edge, "near_bottom")
        self.assertIsNone(result.can_scroll_down)
        result = observe((15, 65))
        self.assertEqual(result.edge, "near_top")
        self.assertIsNone(result.can_scroll_up)

    def test_wrong_scene_dimensions_timing_and_mutable_pixels_rejected(self):
        for changes in ({"scene_id": "combat"}, {"width": True}, {"width": 80, "height": 60},
                        {"observed_at_ns": -1}, {"observed_at_ns": True},
                        {"bgra": bytearray(frame())}, {"bgra": b""}):
            arguments = dict(bgra=frame(), width=40, height=120, scene_id="test.menu", observed_at_ns=1,
                             calibration=CALIBRATION)
            arguments.update(changes)
            with self.subTest(changes=changes.keys()):
                result = detect_scrollbar(**arguments)
                self.assertEqual(result.status, "unknown")
                self.assertIsNone(result.can_scroll_down)

    def test_blank_full_height_multiple_tiny_and_colored_tracks_are_unknown(self):
        colored = bytearray(frame())
        colored[(30 * 40 + 30) * 4:(30 * 40 + 35) * 4] = bytes((0, 255, 0, 0)) * 5
        for pixels in (frame(None), frame((10, 110)), frame((10, 20)),
                       frame((10, 40), extra=(70, 100)), bytes(colored)):
            with self.subTest(size=len(pixels)):
                result = detect_scrollbar(pixels, 40, 120, scene_id="test.menu", observed_at_ns=1,
                                          calibration=CALIBRATION)
                self.assertEqual(result.status, "unknown")

    def test_invalid_calibration_rejected(self):
        for changes in ({"track": (30, 0, 41, 110)}, {"track": [30, 10, 35, 110]},
                        {"width": True}, {"edge_tolerance_px": 6}, {"scene_id": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(CALIBRATION, **changes)


class ScrollTrackingTests(unittest.TestCase):
    def test_movement_and_two_observation_boundary_confirmation(self):
        tracker = MenuScrollTracker()
        tracker.observe(observe(), menu_instance="open1", requested_direction="down")
        moved = tracker.observe(observe((35, 85), 2), menu_instance="open1", requested_direction="down")
        self.assertEqual(moved.motion, "down")
        self.assertFalse(moved.stop_navigation)
        edge = tracker.observe(observe((60, 110), 3), menu_instance="open1", requested_direction="down")
        self.assertTrue(edge.stop_navigation)
        self.assertIsNone(edge.boundary_confirmed)
        confirmed = tracker.observe(observe((60, 110), 4), menu_instance="open1", requested_direction="down")
        self.assertEqual(confirmed.boundary_confirmed, "bottom")

    def test_unchanged_middle_stops_without_claiming_end_or_applied_input(self):
        tracker = MenuScrollTracker()
        for at in range(1, 5):
            result = tracker.observe(observe((35, 85), at), menu_instance="open1", requested_direction="down")
        self.assertEqual(result.reason, "stalled_unknown")
        self.assertTrue(result.stop_navigation)
        self.assertIsNone(result.boundary_confirmed)

    def test_duplicates_do_not_confirm_endpoint_or_increase_stall_count(self):
        tracker = MenuScrollTracker()
        for _ in range(4):
            result = tracker.observe(observe((60, 110)), menu_instance="open1", requested_direction="down")
        self.assertEqual(result.reason, "duplicate_or_out_of_order")
        self.assertIsNone(result.boundary_confirmed)
        self.assertEqual(result.no_scroll_count, 0)

    def test_reentered_menu_and_unknown_frame_break_evidence_chain(self):
        tracker = MenuScrollTracker()
        tracker.observe(observe(), menu_instance="open1")
        result = tracker.observe(observe(at=2), menu_instance="open2")
        self.assertIsNone(result.boundary_confirmed)
        unknown = replace(observe(at=3), status="unknown", thumb=None)
        self.assertTrue(tracker.observe(unknown, menu_instance="open2").stop_navigation)
        self.assertIsNone(tracker.observe(observe(at=4), menu_instance="open2").boundary_confirmed)

    def test_near_edge_opposite_motion_and_extent_changes_stop(self):
        tracker = MenuScrollTracker()
        result = tracker.observe(observe((55, 105)), menu_instance="open1", requested_direction="down")
        self.assertEqual(result.reason, "near_boundary_unconfirmed")
        result = tracker.observe(observe((35, 85), 2), menu_instance="open1", requested_direction="down")
        self.assertEqual(result.reason, "opposite_motion")
        result = tracker.observe(observe((35, 90), 3), menu_instance="open1", requested_direction="down")
        self.assertEqual(result.reason, "content_extent_changed")

    def test_direction_change_and_no_request_do_not_accumulate_stalls(self):
        tracker = MenuScrollTracker()
        for at, direction in enumerate(("down", "up", None, "down", "up"), 1):
            result = tracker.observe(observe((35, 85), at), menu_instance="open1", requested_direction=direction)
            self.assertEqual(result.no_scroll_count, 0)


class LocalScrollCaptureTests(unittest.TestCase):
    def test_accessibility_top_to_near_bottom_without_shipping_game_images(self):
        root = Path(__file__).resolve().parents[1] / "data/raw/brotato-observations"
        paths = [root / name / "frame.png" for name in
                 ("20260926T163600Z-5160759b", "20260926T163627Z-d4b1026e")]
        if not all(path.is_file() for path in paths):
            self.skipTest("local private captures are intentionally excluded from Git")
        observations = []
        for at, path in enumerate(paths, 1):
            pixels, width, height = own_capture_bgra(path)
            observations.append(detect_scrollbar(pixels, width, height, observed_at_ns=at,
                                                 scene_id=ACCESSIBILITY_1080.scene_id))
        self.assertEqual(observations[0].thumb, (223, 748))
        self.assertEqual(observations[0].edge, "top")
        self.assertEqual(observations[1].thumb, (450, 975))
        self.assertEqual(observations[1].edge, "near_bottom")
        self.assertIsNone(observations[1].can_scroll_down)
        tracker = MenuScrollTracker()
        tracker.observe(observations[0], menu_instance="accessibility1", requested_direction="down")
        result = tracker.observe(observations[1], menu_instance="accessibility1", requested_direction="down")
        self.assertEqual(result.motion, "down")
        self.assertTrue(result.stop_navigation)
        self.assertIsNone(result.boundary_confirmed)


if __name__ == "__main__":
    unittest.main()
