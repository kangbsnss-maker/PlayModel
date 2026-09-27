"""Bounded HWND-only menu navigation. Does not touch global cursor or focus."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

from pathlib import Path
import argparse
import json
import time

from playmodel.games.brotato.background import BackgroundController
from playmodel.games.brotato.capture import capture_session
from playmodel.games.brotato.installation import inspect_installation
from playmodel.games.brotato.ocr import read_menu, rows_in_region

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("key", choices=("left", "right", "up", "down", "enter", "escape", "tab"))
parser.add_argument("--expect-text", required=True)
parser.add_argument("--count", type=int, default=1)
args = parser.parse_args()
if not 1 <= args.count <= 20 or (args.key in ("enter", "escape") and args.count != 1):
    raise ValueError("Only directional navigation may repeat, at most 20 times")
install = next(i for i in inspect_installation()["installations"] if i["status"] == "files_present")
exe = Path(install["path"]) / "Brotato.exe"
root = ROOT / "data/raw/brotato-observations"
started = time.perf_counter()
before = capture_session(exe, root)
ocr = read_menu(Path(before["session_directory"]) / "frame.png", ROOT / "scripts/windows_ocr.ps1")
rows = rows_in_region(ocr, (0, 0, before["width"], before["height"]))
if not any(args.expect_text in row for row in rows):
    raise OSError("Expected menu text not found; no key posted")
if time.perf_counter_ns() - before["capture_started_at_ns"] > 2_000_000_000:
    raise OSError("Menu observation expired; no key posted")
control = BackgroundController(before["hwnd"], exe)
desktop_before = control.desktop_state()
for index in range(args.count):
    control.tap_menu(args.key)
    if index + 1 < args.count:
        time.sleep(.075)
posted = time.perf_counter_ns()
time.sleep(.2)
# A second capture checks a newly processed frame; first may be retained render.
capture_session(exe, root)
time.sleep(.2)
after = capture_session(exe, root)
session = Path(after["session_directory"])
ocr = read_menu(session / "frame.png", ROOT / "scripts/windows_ocr.ps1")
record = {"action": "menu_" + args.key, "backend": "postmessage_window_only", "posted_at_ns": posted,
          "count": args.count,
          "before_session": before["session_id"], "after_session": after["session_id"],
          "desktop_before": desktop_before, "desktop_after": control.desktop_state(),
          "elapsed_seconds": time.perf_counter() - started, "game_applied": None, "training_performed": False,
          "rows": rows_in_region(ocr, (0, 0, before["width"], before["height"])), "ocr": ocr}
(session / "background-key.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in record.items() if k != "ocr"}, ensure_ascii=True))
