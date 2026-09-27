"""Discover an existing Steam installation without reading player saves."""

from __future__ import annotations

import os
import re
from pathlib import Path

APP_ID = "1942280"
_TOKENS = re.compile(r'\s+|//[^\n]*|"((?:\\.|[^"\\])*)"|([{}])')


def parse_vdf(text: str) -> dict:
    """Parse quoted Valve KeyValues objects; reject ambiguous/malformed input."""
    tokens: list[tuple[str, str]] = []
    cursor = 0
    for match in _TOKENS.finditer(text.lstrip("\ufeff")):
        if match.start() != cursor:
            raise ValueError("Unsupported or malformed Steam KeyValues content")
        cursor = match.end()
        if match.group(1) is not None:
            tokens.append(("text", re.sub(r'\\(["\\])', r'\1', match.group(1))))
        elif match.group(2):
            tokens.append((match.group(2), match.group(2)))
    if cursor != len(text.lstrip("\ufeff")):
        raise ValueError("Malformed Steam KeyValues trailing content")
    index = 0

    def object_body(nested: bool) -> dict:
        nonlocal index
        result = {}
        while index < len(tokens):
            kind, key = tokens[index]
            index += 1
            if kind == "}":
                if nested:
                    return result
                raise ValueError("Unexpected closing brace")
            if kind != "text" or index >= len(tokens):
                raise ValueError("Missing Steam KeyValues key/value")
            if key in result:
                raise ValueError("Duplicate Steam KeyValues key")
            kind, value = tokens[index]
            index += 1
            if kind == "{":
                value = object_body(True)
            elif kind != "text":
                raise ValueError("Invalid Steam KeyValues value")
            result[key] = value
        if nested:
            raise ValueError("Unclosed Steam KeyValues object")
        return result

    return object_body(False)


def _default_steam_root() -> Path | None:
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _ = winreg.QueryValueEx(key, "SteamPath")
            return Path(value)
    except OSError:
        return None


def inspect_installation(steam_root: Path | None = None) -> dict:
    root = steam_root if steam_root is not None else _default_steam_root()
    report = {
        "game": "Brotato", "steam_app_id": APP_ID, "status": "not_found",
        "installations": [], "warnings": [], "runtime_ready": False,
        "capabilities": {
            "installation_probe": True, "screen_recognition": False,
            "game_input": False, "autonomous_play": False, "trained_policy": False,
        },
        "unverified": ["game_display_version", "enabled_dlc", "enabled_mods", "unlocks", "language", "input_bindings", "auto_aim"],
    }
    if root is None or not root.is_dir():
        report["warnings"].append("Steam root not found; specify --steam-root if needed.")
        return report
    libraries = [root.resolve()]
    library_file = root / "steamapps/libraryfolders.vdf"
    if library_file.is_file():
        try:
            parsed = parse_vdf(library_file.read_text(encoding="utf-8"))
            library_entries = parsed.get("libraryfolders", {})
            if not isinstance(library_entries, dict):
                raise ValueError("Invalid Steam library list")
            for key, entry in library_entries.items():
                if not key.isdigit():
                    continue
                value = entry.get("path") if isinstance(entry, dict) else entry
                if isinstance(value, str):
                    path = Path(value)
                    if not path.is_absolute():
                        raise ValueError("Steam library path must be absolute")
                    if path.resolve() not in libraries:
                        libraries.append(path.resolve())
        except (OSError, UnicodeError, ValueError) as error:
            report["warnings"].append(f"Library discovery incomplete: {error}")
    for library in libraries:
        manifest = library / f"steamapps/appmanifest_{APP_ID}.acf"
        if not manifest.is_file():
            continue
        try:
            app = parse_vdf(manifest.read_text(encoding="utf-8")).get("AppState")
            if not isinstance(app, dict) or app.get("appid") != APP_ID:
                raise ValueError("Manifest does not identify Brotato")
            folder = app.get("installdir")
            if not isinstance(folder, str) or not folder or any(c in folder for c in '/\\:') or folder in (".", ".."):
                raise ValueError("Invalid install folder in manifest")
            common = (library / "steamapps/common").resolve()
            game = (common / folder).resolve()
            if not game.is_relative_to(common):
                raise ValueError("Game path escapes Steam common directory")
            exe, pack = game / "Brotato.exe", game / "Brotato.pck"
            ready_files = exe.is_file() and pack.is_file()
            report["installations"].append({
                "path": str(game), "steam_build_id": app.get("buildid"),
                "steam_state_flags": app.get("StateFlags"),
                "executable_found": exe.is_file(), "resource_pack_found": pack.is_file(),
                "status": "files_present" if ready_files else "incomplete_files",
            })
        except (OSError, UnicodeError, ValueError) as error:
            report["warnings"].append(f"Brotato manifest could not be inspected: {error}")
    if any(item["status"] == "files_present" for item in report["installations"]):
        report["status"] = "files_present"
    elif report["installations"]:
        report["status"] = "incomplete_files"
    return report
