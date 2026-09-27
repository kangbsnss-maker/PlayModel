"""Currency fallback provenance, literal parsing and abstention contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from playmodel.games.brotato.capture import _png, read_diagnostic_png
from playmodel.games.brotato.shop_currency_ocr import (
    GAME_BUILD, CurrencyObservation, bound_currency, parse_repeated_currency,
    read_shop_currency, repeated_currency_crop,
)
from playmodel.games.brotato.shop_learning import observe_shop


def report(value="1", *, merged=False):
    words = ([dict(text=value * 3, x=10, y=10, width=160, height=40)] if merged else
             [dict(text=value, x=i * 60 + 10, y=10, width=40, height=40) for i in range(3)])
    return dict(lines=[dict(words=words)], text="".join(w["text"] for w in words),
                recognition_path="persistent_local_ocr")


def pixels(alpha=0):
    data = bytearray(bytes((40, 40, 40, alpha)) * 1920 * 1080)
    # Synthetic bright glyph-shaped region. Fake OCR supplies a test observation,
    # not a trained template or production digit classifier.
    for y in range(45, 85):
        start = (y * 1920 + 850) * 4
        data[start:start + 12 * 4] = bytes((250, 250, 250, alpha)) * 12
    return bytes(data)


class FakeReader:
    def __init__(self, *, cached=False):
        self.cached = cached
        self.calls = 0

    def read(self, image):
        self.calls += 1
        _, width, height = read_diagnostic_png(image)
        tile = width // 3
        started = time.perf_counter_ns()
        return dict(lines=[dict(words=[dict(text="1", x=i * tile + 10, y=10,
                                           width=tile - 20, height=height - 20)
                                        for i in range(3)])], text="1 1 1",
                    processing_started_at_ns=started, available_at_ns=time.perf_counter_ns(),
                    recognition_path="exact_image_cache" if self.cached else "persistent_local_ocr")


class CurrencyRoiTests(unittest.TestCase):
    def test_literal_repeated_tokens_and_merged_numbers(self):
        for value in ("0", "1", "29", "102", "99999"):
            self.assertEqual(parse_repeated_currency(report(value), 60, 60), int(value))
            self.assertEqual(parse_repeated_currency(report(value, merged=True), 60, 60), int(value))

    def test_disagreement_missing_copy_and_non_digits_abstain(self):
        for text in ("l", "O", "1.", "-1", "+1", "01", ""):
            self.assertIsNone(parse_repeated_currency(report(text), 60, 60))
        changed = report()
        changed["lines"][0]["words"][1]["text"] = "2"
        changed["text"] = "121"
        self.assertIsNone(parse_repeated_currency(changed, 60, 60))
        short = report()
        short["lines"][0]["words"].pop()
        short["text"] = "11"
        self.assertIsNone(parse_repeated_currency(short, 60, 60))
        self.assertIsNone(parse_repeated_currency(report("1111", merged=True) | {"text": "1111"}, 60, 60))

    def test_numeric_text_without_coordinate_correspondence_abstains(self):
        for key, value in (("x", 10), ("y", 80), ("width", float("nan"))):
            bad = deepcopy(report())
            bad["lines"][0]["words"][1][key] = value
            self.assertIsNone(parse_repeated_currency(bad, 60, 60))
        self.assertIsNone(parse_repeated_currency(report() | {"recognition_path": "exact_image_cache"}, 60, 60))
        self.assertIsNone(parse_repeated_currency(report() | {"cache_source": "older"}, 60, 60))
        self.assertIsNone(parse_repeated_currency(report() | {"text": "111 bonus"}, 60, 60))

    def test_observed_leading_digit_loss_is_rejected_without_majority_vote(self):
        # Actual failed crop: the first 35 was recognized as 5. Never fill the
        # missing digit from the purchase price or the other two copies.
        failed = dict(text="5 35 35", recognition_path="persistent_local_ocr", lines=[dict(words=[
            dict(text="5", x=41, y=15, width=31, height=42),
            dict(text="35", x=89, y=7, width=64, height=46),
            dict(text="35", x=170, y=0, width=64, height=45)])])
        self.assertIsNone(parse_repeated_currency(failed, 81, 61))
        # Fresh OCR of the same pixels with 20px source padding. All three
        # independently positioned tokens are literal 35 under the same parser.
        padded = dict(text="35 35 35", recognition_path="persistent_local_ocr", lines=[dict(words=[
            dict(text="35", x=19, y=24, width=63, height=44),
            dict(text="35", x=120, y=18, width=63, height=44),
            dict(text="35", x=221, y=12, width=62, height=44)])])
        self.assertEqual(parse_repeated_currency(padded, 101, 81), 35)

    def test_private_failed_balance_crop_preserves_wider_source_padding(self):
        path = (Path(__file__).resolve().parents[1] /
            "artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/"
            "partial-recovery-42b051f90195406b8592c6fff70e9222/segments/"
            "20260927T070617Z-0a5ef04a/menus/20260927T070627Z-bba54dc7/frame.png")
        if not path.is_file():
            self.skipTest("private failed currency observation absent")
        frame, width, height = read_diagnostic_png(path)
        self.assertEqual((width, height), (1920, 1080))
        _, crop, tile_width, tile_height = repeated_currency_crop(frame)
        self.assertEqual(crop, (816, 26, 917, 107))
        self.assertEqual((tile_width, tile_height), (101, 81))

    def test_copy_preserves_original_rgb_and_never_invents_blank_digit(self):
        original = pixels()
        generated, crop, width, height = repeated_currency_crop(original)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "crop.png"
            image.write_bytes(generated)
            decoded, actual_width, actual_height = read_diagnostic_png(image)
        self.assertEqual((actual_width, actual_height), (width * 3, height))
        left, top, right, bottom = crop
        for row, y in enumerate(range(top, bottom)):
            expected = original[(y * 1920 + left) * 4:(y * 1920 + right) * 4] * 3
            actual = decoded[row * actual_width * 4:(row + 1) * actual_width * 4]
            self.assertEqual(actual, expected)
        blank = bytes((40, 40, 40, 0)) * 1920 * 1080
        self.assertIsNone(repeated_currency_crop(blank))

    def make_read(self, directory, *, alpha=0, cached=False):
        image = directory / "frame.png"
        frame = pixels(alpha=alpha)
        image.write_bytes(_png(1920, 1080, frame))
        observed = time.perf_counter_ns()
        result = read_shop_currency(FakeReader(cached=cached), image, pixels=frame,
                                    observed_at_ns=observed, frame_id=str(image.resolve()),
                                    game_build_id=GAME_BUILD)
        kwargs = dict(pixels=frame, frame_id=str(image.resolve()), game_build_id=GAME_BUILD,
                      observed_at_ns=observed, available_at_ns=result.available_at_ns)
        return result, kwargs

    def test_evidence_is_source_bound_alpha_agnostic_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, args = self.make_read(Path(tmp), alpha=255)
            self.assertEqual(bound_currency(result, **args), 1)
            self.assertEqual(bound_currency(json.loads(json.dumps(asdict(result))), **args), 1)
            self.assertIsNone(result.confidence)
            self.assertFalse(result.verified)
            self.assertEqual(len(result.evidence_files), 2)
            for path, digest in result.evidence_files:
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            for key, wrong in (("frame_id", "other-frame"), ("game_build_id", "other-build"),
                               ("observed_at_ns", args["observed_at_ns"] - 1),
                               ("available_at_ns", result.available_at_ns - 1)):
                self.assertIsNone(bound_currency(result, **(args | {key: wrong})))
            self.assertIsNone(bound_currency(replace(result, currency=9), **args))
            self.assertIsNone(bound_currency(asdict(result) | {"available_at_ns": "late"}, **args))
            with self.assertRaises(FileExistsError):
                read_shop_currency(FakeReader(), Path(result.source_path), pixels=args["pixels"],
                                   observed_at_ns=args["observed_at_ns"], frame_id=args["frame_id"], game_build_id=GAME_BUILD)

    def test_modified_derived_or_source_file_is_rejected(self):
        for filename in ("currency-roi.png", "currency-roi-ocr.json", "frame.png"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                result, args = self.make_read(Path(tmp))
                target = Path(tmp) / filename
                target.write_bytes(target.read_bytes() + b"changed")
                self.assertIsNone(bound_currency(result, **args))

    def test_cached_recognition_remains_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, args = self.make_read(Path(tmp), cached=True)
            self.assertIsNone(result.currency)
            self.assertIsNone(bound_currency(result, **args))

    def test_mismatched_pixels_do_not_produce_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            original = pixels()
            image.write_bytes(_png(1920, 1080, original))
            wrong = bytearray(original)
            wrong[0] ^= 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_shop_currency(FakeReader(), image, pixels=bytes(wrong), observed_at_ns=time.perf_counter_ns(),
                                   frame_id="one", game_build_id=GAME_BUILD)

    def test_shop_fallback_is_observation_and_conflicts_abstain(self):
        from test_shop_learning import report as shop_report, frame as shop_frame
        with tempfile.TemporaryDirectory() as tmp:
            # Use the complete synthetic shop so scene/offer guards remain active.
            frame = bytearray(shop_frame())
            source = pixels()
            for y in range(25, 110):
                start, end = (y * 1920 + 760) * 4, (y * 1920 + 1050) * 4
                frame[start:end] = source[start:end]
            frame = bytes(frame)
            image = Path(tmp) / "frame.png"
            image.write_bytes(_png(1920, 1080, frame))
            observed = time.perf_counter_ns()
            raw = shop_report(currency=1)
            raw["processing_started_at_ns"] = observed
            fallback = read_shop_currency(FakeReader(), image, pixels=frame, observed_at_ns=observed,
                                          frame_id="one", game_build_id=GAME_BUILD)
            args = dict(pixels=frame, observed_at_ns=observed, available_at_ns=fallback.available_at_ns,
                        frame_id="one", game_build_id=GAME_BUILD, currency_observation=asdict(fallback))
            parsed = observe_shop(raw, **args)
            self.assertEqual(parsed.currency, 1)
            self.assertEqual(parsed.currency_evidence["frame_id"], "one")
            raw["lines"][2]["words"][0]["text"] = "9"
            self.assertIsNone(observe_shop(raw, **args))
            raw["lines"][2]["words"][0]["text"] = "00"
            # Invalid canonical spelling cannot contradict a bound ROI reading.
            self.assertEqual(observe_shop(raw, **args).currency, 1)
            raw["lines"].pop(2)
            parsed = observe_shop(raw, **args)
            self.assertEqual(parsed.currency, 1)


if __name__ == "__main__":
    unittest.main()
