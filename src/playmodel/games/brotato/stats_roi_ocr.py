"""Calibrated local primary-stat OCR, preserving every derivation.

Nearest-neighbor enlargement changes no digit strokes. The OCR result is still
an observation, never a reward label. No capture, game input or network access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .capture import _png, read_diagnostic_png
from .ocr import rows_in_region
from .state_features import CLOCK, GAME_BUILD, PARSER_VERSION, STATS_ROI

WIDTH, HEIGHT, SCALE = 1920, 1080, 2


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def enlarged_stats_crop(pixels: bytes, scene: str, *, scale: int = SCALE):
    """Return (PNG, ROI, width, height), preserving source pixels exactly."""
    if (not isinstance(pixels, bytes) or len(pixels) != WIDTH * HEIGHT * 4
            or scene not in STATS_ROI or type(scale) is not int or scale not in (1, 2, 3)):
        raise ValueError("uncalibrated stats crop")
    x0, y0, x1, y1 = roi = STATS_ROI[scene]
    result = bytearray()
    for y in range(y0, y1):
        row = pixels[(y * WIDTH + x0) * 4:(y * WIDTH + x1) * 4]
        # Copy each byte channel at its repeated pixel offset. Slice assignment
        # runs in C and produces exactly the prior nearest-neighbor bytes, with
        # eight operations per row at scale2 instead of 340 Python pixel calls.
        enlarged = bytearray(len(row) * scale)
        for channel in range(4):
            values = row[channel::4]
            for repeat in range(scale):
                enlarged[channel + repeat * 4::scale * 4] = values
        result.extend(enlarged * scale)
    width, height = (x1 - x0) * scale, (y1 - y0) * scale
    return _png(width, height, bytes(result), compression_level=1), roi, width, height


def read_stats_roi(reader, image_path: Path, *, pixels: bytes, observed_at_ns: int,
                   frame_id: str, game_build_id: str, scene: str) -> dict:
    """Read once via the existing hidden WinRT worker; never reuse its cache.

    Return the complete observation consumed by BoundedBuildState. Calls must
    precede policy sampling. Two *different source frames* are needed for state
    agreement. The caller applies its observation-age guard after this call.
    """
    image_path = Path(image_path).resolve()
    started = time.perf_counter_ns()
    if (game_build_id != GAME_BUILD or scene not in STATS_ROI
            or not isinstance(frame_id, str) or Path(frame_id).resolve() != image_path
            or type(observed_at_ns) is not int or not 0 < observed_at_ns <= started
            or not isinstance(pixels, bytes) or len(pixels) != WIDTH * HEIGHT * 4):
        raise ValueError("unsupported stats observation")
    source_bytes = image_path.read_bytes()
    source_sha = _sha(source_bytes)
    source_pixels, width, height = read_diagnostic_png(image_path)
    # Capture PNGs can omit alpha; compare the three source color channels.
    if ((width, height) != (WIDTH, HEIGHT)
            or any(source_pixels[channel::4] != pixels[channel::4] for channel in range(3))):
        raise ValueError("stats source PNG does not match bound pixels")
    derived, roi, width, height = enlarged_stats_crop(pixels, scene)
    derived_path = image_path.with_name("stats-roi.png")
    raw_path = image_path.with_name("stats-roi-ocr.json")
    evidence_path = image_path.with_name("stats-roi.json")
    with derived_path.open("xb") as stream:
        stream.write(derived)
    old_cache = getattr(reader, "cache_seconds", None)
    if old_cache is not None:
        reader.cache_seconds = 0.0
    try:
        raw = reader.read(derived_path)
    finally:
        if old_cache is not None:
            reader.cache_seconds = old_cache
    if (raw.get("recognition_path") != "persistent_local_ocr" or "cache_source" in raw
            or type(raw.get("processing_started_at_ns")) is not int
            or type(raw.get("available_at_ns")) is not int
            or not started <= raw["processing_started_at_ns"] <= raw["available_at_ns"] <= time.perf_counter_ns()):
        raise ValueError("fresh independently executed stats OCR required")
    with raw_path.open("x", encoding="utf-8") as stream:
        json.dump(raw, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if _sha(image_path.read_bytes()) != source_sha:
        raise ValueError("stats source changed during OCR")
    evidence_files = [{"path": str(derived_path), "sha256": _sha(derived)},
                      {"path": str(raw_path), "sha256": _sha(raw_path.read_bytes())}]
    result = {"frame_ref": str(image_path), "frame_id": frame_id, "source_png_sha256": source_sha,
              "source_pixels_sha256": _sha(pixels), "game_build_id": game_build_id,
              "observed_at_ns": observed_at_ns, "processing_started_at_ns": started,
              "ocr_started_at_ns": raw["processing_started_at_ns"], "available_at_ns": time.perf_counter_ns(),
              "scene": scene, "language": "en", "ocr_language": raw.get("language"),
              "raw_text": rows_in_region(raw, (0, 0, width, height)), "roi": roi, "scale": SCALE,
              "derived_size": (width, height), "recognition_path": raw["recognition_path"],
              "confidence": None, "verified": False, "clock_domain": CLOCK,
              "extractor_version": PARSER_VERSION, "evidence_files": evidence_files,
              "report_path": str(evidence_path)}
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    # Include the immutable derivation report without a circular self-hash.
    result["evidence_files"] = [*evidence_files, {"path": str(evidence_path), "sha256": _sha(evidence_path.read_bytes())}]
    result["available_at_ns"] = time.perf_counter_ns()
    return result
