"""Preparation and observation commands. No game input or training is performed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playmodel import __version__
from playmodel.doctor import inspect_environment
from playmodel.profiles import ProfileError, inspect_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PlayModel project preparation tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect this machine without installing anything")
    doctor.add_argument("--storage", type=Path, default=Path.cwd())
    brotato = commands.add_parser("brotato-inspect", help="Read-only discovery of an existing Steam Brotato installation")
    brotato.add_argument("--steam-root", type=Path, help="Steam directory; auto-detect current Windows user by default")
    capture = commands.add_parser("brotato-capture", help="Capture one Brotato client frame; no input or training")
    capture.add_argument("--steam-root", type=Path)
    capture.add_argument("--output", type=Path, default=Path("data/raw/brotato-observations"))
    profile = commands.add_parser("validate-profile", help="Validate a game profile configuration")
    profile.add_argument("path", type=Path)
    profile.add_argument("--allow-draft", action="store_true", help="Accept structurally valid drafts")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = inspect_environment(args.storage)
            code = 0 if report["python_supported"] else 2
        elif args.command in ("brotato-inspect", "brotato-capture"):
            from playmodel.games.brotato.installation import inspect_installation

            report = inspect_installation(args.steam_root)
            code = 0 if report["status"] == "files_present" else 2
            if args.command == "brotato-capture" and code == 0:
                from playmodel.games.brotato.capture import capture_session

                installations = [item for item in report["installations"] if item["status"] == "files_present"]
                if len(installations) != 1:
                    raise OSError("Multiple Brotato installations; select a single Steam root")
                report = capture_session(Path(installations[0]["path"]) / "Brotato.exe", args.output)
        else:
            report = inspect_profile(args.path)
            code = 0 if report["configuration_complete"] or args.allow_draft else 2
    except (OSError, ProfileError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return code
