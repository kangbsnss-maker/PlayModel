"""Explicit menu calibration: local OCR, verified foreground, one pointer action.

This developer tool records observations. It does not train a gameplay model.
"""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

from pathlib import Path
import argparse
import json
import sys
import time
import subprocess

from playmodel.games.brotato.capture import capture_session, region_digest
from playmodel.games.brotato.installation import inspect_installation
from playmodel.games.brotato.interaction import WindowController
from playmodel.games.brotato.ocr import read_menu, rows_in_region

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", required=True, help="Exact normalized title to verify before input")
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--click", action="store_true")
    args = parser.parse_args()
    installs = [i for i in inspect_installation()["installations"] if i["status"] == "files_present"]
    if len(installs) != 1:
        raise OSError("Exactly one Brotato installation required")
    exe = Path(installs[0]["path"]) / "Brotato.exe"
    before = capture_session(exe, ROOT / "data/raw/brotato-observations")
    frame = Path(before["session_directory"]) / "frame.png"
    ocr = read_menu(frame, ROOT / "scripts/windows_ocr.ps1")
    width, height = before["width"], before["height"]
    title_rows = rows_in_region(ocr, (int(width * .2), 0, int(width * .8), int(height * .15)))
    if args.expect not in title_rows:
        raise OSError(f"Expected title not found: {args.expect!r}; observed {title_rows!r}")
    fresh = capture_session(exe, ROOT / "data/raw/brotato-observations")
    regions = [(int(width * .4), int(height * .065), int(width * .6), int(height * .14)),
               (max(0, args.x - 25), max(0, args.y - 25), min(width, args.x + 25), min(height, args.y + 25))]
    fresh_frame = Path(fresh["session_directory"]) / "frame.png"
    if (any(region_digest(frame, region) != region_digest(fresh_frame, region) for region in regions)
            or fresh["hwnd"] != before["hwnd"]
            or time.perf_counter_ns() - fresh["capture_started_at_ns"] > 500_000_000):
        raise OSError("Menu changed during OCR or observation expired; no input sent")
    controller = WindowController(before["hwnd"], exe)
    controller.activate()
    live_before = capture_session(exe, ROOT / "data/raw/brotato-observations", backend="visible_client")
    if any(region_digest(frame, region) != region_digest(Path(live_before["session_directory"]) / "frame.png", region)
           for region in regions):
        raise OSError("Visible menu differs from diagnostic image; no input sent")
    command = [sys.executable, "-m", "playmodel.games.brotato.interaction", "--hwnd", str(before["hwnd"]),
               "--exe", str(exe), "--x", str(args.x), "--y", str(args.y), "--width", str(width), "--height", str(height),
               "--deadline", str(time.perf_counter_ns() + 500_000_000)]
    if args.click:
        command.append("--click")
    action = subprocess.run(command, capture_output=True, text=True, timeout=3,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if action.returncode:
        raise OSError(action.stderr.strip() or "Calibration action failed")
    action_at = time.perf_counter_ns()
    time.sleep(1.0)  # Diagnostic tooltip settle; never used in the combat loop.
    after = capture_session(exe, ROOT / "data/raw/brotato-observations", backend="visible_client")
    session = Path(after["session_directory"])
    ocr = read_menu(session / "frame.png", ROOT / "scripts/windows_ocr.ps1")
    record = {
        "actor": "developer_calibration", "action": "click" if args.click else "hover",
        "x": args.x, "y": args.y, "action_at_ns": action_at,
        "before_session": before["session_id"], "after_session": after["session_id"],
        "expected_title": args.expect, "usable_for_training": False,
        "reason": "Calibration observations have no verified policy labels or outcomes",
        "ocr": ocr,
    }
    (session / "menu-probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"session": str(session), "action": record["action"],
                      "rows": rows_in_region(ocr, (0, 0, width, int(height * .66)))}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
