"""Local OCR fallback for isolated shop currency digits.

WinRT sometimes omits a solitary digit. Three copies of the same tightly cropped
number provide text-line context. They are one observation, never three votes.
Only literal, geometrically corresponding, identical numeric readings survive;
prices, previous currency and purchase actions are not inputs to this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from .capture import _png, read_diagnostic_png


WIDTH, HEIGHT = 1920, 1080
GAME_BUILD = "23429717"
CURRENCY_RECT = (760, 25, 1050, 110)
COPIES = 3
# A 10px margin made WinRT clip the leading digit of the first repeated "35"
# (20260927T070627Z-bba54dc7). Preserve 20px of original surrounding pixels so
# line-angle estimation has room; all three literal readings must still agree.
PADDING = 20


@dataclass(frozen=True)
class CurrencyObservation:
    frame_id: str
    frame_sha256: str
    source_png_sha256: str
    source_path: str
    game_build_id: str
    observed_at_ns: int
    processing_started_at_ns: int
    available_at_ns: int
    currency: int | None
    reason: str
    crop_rect: tuple[int, int, int, int] | None
    copies: int
    tile_width: int
    tile_height: int
    derived_png_sha256: str | None
    report_path: str
    raw_text: str
    recognition_path: str
    evidence_files: tuple[tuple[str, str], ...] = ()
    confidence: None = None
    verified: bool = False
    clock_domain: str = "perf_counter_ns_same_host"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repeated_currency_crop(pixels: bytes) -> tuple[bytes, tuple[int, int, int, int], int, int] | None:
    """Copy original pixels unchanged; bright pixels locate, never label, digits."""
    if not isinstance(pixels, bytes) or len(pixels) != WIDTH * HEIGHT * 4:
        return None
    left, top, right, bottom = CURRENCY_RECT
    ink = []
    for y in range(top, bottom):
        for x in range(left, right):
            offset = (y * WIDTH + x) * 4
            if min(pixels[offset:offset + 3]) > 190:
                ink.append((x, y))
    if not 20 <= len(ink) <= 6000:
        return None
    min_x, max_x = min(p[0] for p in ink), max(p[0] for p in ink)
    min_y, max_y = min(p[1] for p in ink), max(p[1] for p in ink)
    # A clipped number or unrelated bright panel is not a recoverable reading.
    if (min_x <= left or max_x >= right - 1 or min_y <= top or max_y >= bottom - 1
            or not 8 <= max_y - min_y + 1 <= 65 or max_x - min_x + 1 < 3):
        return None
    crop = (max(left, min_x - PADDING), max(top, min_y - PADDING),
            min(right, max_x + PADDING + 1), min(bottom, max_y + PADDING + 1))
    x0, y0, x1, y1 = crop
    tile_width, tile_height = x1 - x0, y1 - y0
    repeated = bytearray()
    for y in range(y0, y1):
        row = pixels[(y * WIDTH + x0) * 4:(y * WIDTH + x1) * 4]
        repeated.extend(row * COPIES)
    return _png(tile_width * COPIES, tile_height, bytes(repeated), compression_level=1), crop, tile_width, tile_height


def parse_repeated_currency(report: dict, tile_width: int, tile_height: int) -> int | None:
    """Require each copy exactly once and reject nonnumeric or misplaced tokens."""
    if (report.get("recognition_path") != "persistent_local_ocr" or "cache_source" in report
            or type(tile_width) is not int or type(tile_height) is not int
            or not 3 <= tile_width <= CURRENCY_RECT[2] - CURRENCY_RECT[0]
            or not 8 <= tile_height <= CURRENCY_RECT[3] - CURRENCY_RECT[1]):
        return None
    lines = report.get("lines")
    if not isinstance(lines, list) or len(lines) != 1:
        return None
    words = lines[0].get("words")
    if not isinstance(words, list) or not words:
        return None
    digits, spans = [], []
    for word in sorted(words, key=lambda w: w.get("x", -1)):
        text = word.get("text")
        if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,15}", text):
            return None
        coords = [word.get(key) for key in ("x", "y", "width", "height")]
        if any(type(v) not in (int, float) for v in coords):
            return None
        x, y, width, height = coords
        if not (0 <= x < x + width <= tile_width * COPIES and 0 <= y < y + height <= tile_height):
            return None
        covered = [i for i in range(COPIES)
                   if x < (i + 1) * tile_width - 3 and x + width > i * tile_width + 3]
        if not covered or covered != list(range(covered[0], covered[-1] + 1)):
            return None
        # A token must include the center of every tile it claims to read.
        if any(not x <= (i + .5) * tile_width <= x + width for i in covered):
            return None
        if len(text) % len(covered):
            return None
        size = len(text) // len(covered)
        pieces = [text[i:i + size] for i in range(0, len(text), size)]
        if not 1 <= size <= 5 or len(set(pieces)) != 1:
            return None
        digits.extend(pieces)
        spans.extend(covered)
    if spans != list(range(COPIES)) or len(set(digits)) != 1:
        return None
    value = digits[0]
    if len(value) > 1 and value.startswith("0"):
        return None
    # The raw report cannot hide extra punctuation or unaccounted tokens.
    if "".join(str(report.get("text", "")).split()) != "".join(digits):
        return None
    return int(value)


def read_shop_currency(reader, image_path: Path, *, pixels: bytes, observed_at_ns: int,
                       frame_id: str, game_build_id: str) -> CurrencyObservation:
    """Save derivation evidence beside an immutable capture; no input injection."""
    image_path = Path(image_path)
    started = time.perf_counter_ns()
    if (game_build_id != GAME_BUILD or not isinstance(frame_id, str) or not frame_id.strip()
            or type(observed_at_ns) is not int or not 0 < observed_at_ns <= started
            or not isinstance(pixels, bytes) or len(pixels) != WIDTH * HEIGHT * 4):
        raise ValueError("Unsupported shop currency observation")
    evidence_path = image_path.with_name("currency-roi.json")
    derived_path = image_path.with_name("currency-roi.png")
    raw_path = image_path.with_name("currency-roi-ocr.json")
    source_pixels, source_width, source_height = read_diagnostic_png(image_path)
    if ((source_width, source_height) != (WIDTH, HEIGHT)
            or any(source_pixels[channel::4] != pixels[channel::4] for channel in range(3))):
        raise ValueError("Currency source PNG does not match bound pixels")
    evidence_files = []
    transformed = repeated_currency_crop(pixels)
    crop, tile_width, tile_height, derived_hash = None, 0, 0, None
    raw, currency, reason = {}, None, "number_pixels_unreadable"
    if transformed is not None:
        derived, crop, tile_width, tile_height = transformed
        with derived_path.open("xb") as stream:
            stream.write(derived)
        derived_hash = _digest(derived)
        raw = reader.read(derived_path)
        with raw_path.open("x", encoding="utf-8") as stream:
            json.dump(raw, stream, ensure_ascii=False, indent=2, allow_nan=False)
        evidence_files = [(str(derived_path.resolve()), derived_hash),
                          (str(raw_path.resolve()), _digest(raw_path.read_bytes()))]
        currency = parse_repeated_currency(raw, tile_width, tile_height)
        reason = "literal_repeated_roi_agreement" if currency is not None else "repeated_roi_unreadable"
    result = CurrencyObservation(
        frame_id, _digest(pixels), _digest(image_path.read_bytes()), str(image_path.resolve()),
        game_build_id, observed_at_ns, started, time.perf_counter_ns(), currency, reason, crop,
        COPIES, tile_width, tile_height, derived_hash, str(raw_path.resolve()),
        str(raw.get("text", "")), str(raw.get("recognition_path", "not_run")), tuple(evidence_files))
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def bound_currency(observation: CurrencyObservation | dict | None, *, pixels: bytes, frame_id: str,
                   game_build_id: str, observed_at_ns: int, available_at_ns: int) -> int | None:
    """The fallback may supplement only its own frame, within the caller's clock."""
    if isinstance(observation, dict):
        try:
            observation = CurrencyObservation(**observation)
        except (TypeError, ValueError):
            return None
    if (not isinstance(observation, CurrencyObservation)
            or observation.frame_id != frame_id or observation.frame_sha256 != _digest(pixels)
            or observation.game_build_id != game_build_id or game_build_id != GAME_BUILD
            or observation.observed_at_ns != observed_at_ns
            or any(type(v) is not int for v in (observed_at_ns, available_at_ns,
                   observation.processing_started_at_ns, observation.available_at_ns))
            or not observed_at_ns <= observation.processing_started_at_ns <= observation.available_at_ns <= available_at_ns
            or observation.recognition_path != "persistent_local_ocr"
            or observation.reason != "literal_repeated_roi_agreement"
            or observation.clock_domain != "perf_counter_ns_same_host"
            or type(observation.currency) is not int or not 0 <= observation.currency <= 99999):
        return None
    try:
        source = Path(observation.source_path)
        expected_files = (source.with_name("currency-roi.png"), source.with_name("currency-roi-ocr.json"))
        files = observation.evidence_files
        if (len(files) != 2 or _digest(source.read_bytes()) != observation.source_png_sha256
                or any(Path(pair[0]).resolve() != path.resolve() or _digest(path.read_bytes()) != pair[1]
                       for pair, path in zip(files, expected_files))
                or files[0][1] != observation.derived_png_sha256):
            return None
        raw = json.loads(expected_files[1].read_text(encoding="utf-8"))
        if (parse_repeated_currency(raw, observation.tile_width, observation.tile_height) != observation.currency
                or not observation.processing_started_at_ns <= raw["processing_started_at_ns"] <= raw["available_at_ns"] <= observation.available_at_ns):
            return None
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None
    return observation.currency
