"""One background tooltip/click capability test, preserving raw observations."""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
import json
from pathlib import Path
import time

from playmodel.games.brotato.background import BackgroundController
from playmodel.games.brotato.capture import capture_session, region_digest
from playmodel.games.brotato.installation import inspect_installation
from playmodel.games.brotato.ocr import read_menu, rows_in_region

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--expect", default="角色选择")
    parser.add_argument("--click", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    installations = [i for i in inspect_installation()["installations"] if i["status"] == "files_present"]
    if len(installations) != 1:
        raise OSError("Expected one Brotato installation")
    exe = Path(installations[0]["path"]) / "Brotato.exe"
    root = ROOT / "data/raw/brotato-observations"
    before = capture_session(exe, root)
    before_path = Path(before["session_directory"]) / "frame.png"
    ocr = read_menu(before_path, ROOT / "scripts/windows_ocr.ps1")
    width, height = before["width"], before["height"]
    titles = rows_in_region(ocr, (int(width * .3), 0, int(width * .7), int(height * .15)))
    if args.expect not in titles:
        raise OSError(f"Expected screen {args.expect!r}, observed {titles!r}; no input posted")
    fresh = capture_session(exe, root)
    regions = [(int(width * .4), int(height * .065), int(width * .6), int(height * .14)),
               (max(0, args.x - 25), max(0, args.y - 25), min(width, args.x + 25), min(height, args.y + 25))]
    if (fresh["hwnd"] != before["hwnd"] or time.perf_counter_ns() - fresh["capture_started_at_ns"] > 500_000_000
            or any(region_digest(before_path, region) != region_digest(Path(fresh["session_directory"]) / "frame.png", region)
                   for region in regions)):
        raise OSError("Menu changed or capture expired; no input posted")
    control = BackgroundController(before["hwnd"], exe)
    desktop_before = control.desktop_state()
    (control.click if args.click else control.hover)(args.x, args.y, (width, height))
    posted = time.perf_counter_ns()
    time.sleep(.3)
    after = capture_session(exe, root)
    session = Path(after["session_directory"])
    ocr = read_menu(session / "frame.png", ROOT / "scripts/windows_ocr.ps1")
    report = {"backend": "postmessage_window_only", "action": "click" if args.click else "hover",
              "target": [args.x, args.y], "before_session": before["session_id"], "after_session": after["session_id"],
              "posted_at_ns": posted, "desktop_before": desktop_before, "desktop_after": control.desktop_state(),
              "elapsed_seconds": time.perf_counter() - started, "game_applied": None,
              "knowledge_verified": False, "training_performed": False,
              "character_rows": rows_in_region(ocr, (210, 180, 975, 680)), "ocr": ocr}
    (session / "background-probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ocr"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
