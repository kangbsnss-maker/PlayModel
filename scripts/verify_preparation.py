"""Verify the preparation repository without installing ML or accessing a game."""

from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_sources(root: Path) -> dict:
    manifest = json.loads((root / "docs/research/source-manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported research manifest version")
    pages = manifest["pages"]
    require(len(pages) == manifest["source_count"] == 29, "Research must cover all 29 supplied pages")
    require(sorted(page["page"] for page in pages) == list(range(1, 30)), "Missing or duplicate research pages")
    require(len({page["filename"] for page in pages}) == 29, "Duplicate source filenames")
    cleaned = (root / "docs/source-cleanup.json").exists()
    for page in pages:
        name = page["filename"]
        require(Path(name).name == name and "/" not in name and "\\" not in name, "Unsafe source filename")
        require(re.fullmatch(r"[0-9a-f]{64}", page["sha256"]) is not None, "Invalid source hash")
        require(page["width"] > 0 and page["height"] > 0 and page["bytes"] > 0, "Invalid source size")
        require(bool(page["title"].strip()), "Missing research title")
        note, anchor = page["notes"].split("#", 1)
        note_path = (root / note).resolve()
        require(note_path.is_relative_to(root.resolve()), "Research notes outside repository")
        require(anchor == f"page-{page['page']:02d}", "Incorrect page anchor")
        text = note_path.read_text(encoding="utf-8")
        require(f"## {anchor}\n" in text, f"Missing durable note for {name}")
        body = text.split(f"## {anchor}\n", 1)[1].split("\n## page-", 1)[0]
        require(len(body.strip()) >= 150, f"Research note too short to preserve {name}")
        original = root / name
        if not cleaned:
            require(original.is_file(), f"Source missing before verified cleanup: {name}")
            raw = original.read_bytes()
            require(len(raw) == page["bytes"], f"Source size changed: {name}")
            require(hashlib.sha256(raw).hexdigest() == page["sha256"], f"Source hash mismatch: {name}")
            require(len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR", f"Invalid PNG header: {name}")
            require(struct.unpack(">II", raw[16:24]) == (page["width"], page["height"]), f"Source dimensions changed: {name}")
    listed = {page["filename"] for page in pages}
    require(all(path.name in listed for path in root.glob("분석 출력*.png")), "Unlisted source image")
    if cleaned:
        verify_cleanup(root, manifest)
    return manifest


def verify_cleanup(root: Path, manifest: dict) -> None:
    for page in manifest["pages"]:
        require(not (root / page["filename"]).exists(), f"Source image still present: {page['filename']}")
    receipt = json.loads((root / "docs/source-cleanup.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "completed", "Source cleanup not completed")
    expected = {page["filename"]: page["sha256"] for page in manifest["pages"]}
    actual = {page["filename"]: page["sha256"] for page in receipt["deleted_sources"]}
    require(len(receipt["deleted_sources"]) == len(expected) and actual == expected, "Cleanup receipt mismatch")
    require(receipt["pre_delete_verification"] == "PREPARATION_VERIFIED", "Missing pre-delete verification")


def verify_links(root: Path) -> None:
    files = [root / "README.md", root / "AGENTS.md"] + list((root / "docs").rglob("*.md"))
    count = 0
    for path in files:
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(target.split("#", 1)[0].strip("<>"))
            resolved = (path.parent / relative).resolve()
            require(resolved.is_relative_to(root.resolve()), f"Nonportable document link in {path.name}: {relative}")
            require(resolved.exists(), f"Broken document link in {path.name}: {relative}")
            count += 1
    print(f"Local document links verified: {count}")


def verify_ignore_rules(root: Path) -> None:
    ignored = [
        "data/raw/session.json", "models/candidate.pt", "media/captures/session.mp4",
        "artifacts/doctor.json", "configs/local/first-game.json", ".env", "credentials.json",
        "game.iso", "game.bk2", ".venv/local-file", "분석 출력 1.png",
    ]
    visible = ["src/playmodel/cli.py", "README.md", "data/README.md", "models/README.md", "media/README.md",
               "artifacts/README.md", "configs/local/README.md", "configs/profiles/first-game.template.json"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(ignored + visible) + "\0").encode("utf-8"), capture_output=True, cwd=root,
        timeout=15, check=False,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    require(result.returncode == 0, f"Git ignore check failed: {result.stderr.strip()}")
    observed = set(result.stdout.decode("utf-8").rstrip("\0").split("\0"))
    require(observed == set(ignored), f"Git ignore boundary mismatch: {observed.symmetric_difference(ignored)}")
    print(f"Git ignore boundary verified: {len(ignored)} excluded, {len(visible)} visible controls")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-cleanup", action="store_true", help="Verify source preservation and completed deletion only")
    args = parser.parse_args()
    try:
        manifest = verify_sources(ROOT)
        if args.check_cleanup:
            verify_cleanup(ROOT, manifest)
            print("CLEANUP_VERIFIED")
            return 0
        required = [
            "README.md", "AGENTS.md", "docs/architecture.md", "docs/roadmap.md", "docs/development.md",
            "docs/github-readiness.md", "docs/research/page-notes.md", ".github/workflows/checks.yml",
        ]
        for name in required:
            require((ROOT / name).is_file(), f"Missing project artifact: {name}")
        with (ROOT / "pyproject.toml").open("rb") as source:
            metadata = tomllib.load(source)
        require(metadata["project"]["name"] == "playmodel", "Unexpected package name")
        verify_links(ROOT)
        verify_ignore_rules(ROOT)
        sys.path.insert(0, str(ROOT / "src"))
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        require(result.wasSuccessful(), "Preparation behavior tests failed")
        if (ROOT / "docs/source-cleanup.json").exists():
            verify_cleanup(ROOT, manifest)
        print(f"Research source pages verified: {len(manifest['pages'])}")
        print("PREPARATION_VERIFIED")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"VERIFICATION_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
