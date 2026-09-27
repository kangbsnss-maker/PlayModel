"""Strict draft configuration validation, deliberately separate from runtime readiness."""

from __future__ import annotations

import json
import math
from pathlib import Path


class ProfileError(ValueError):
    """A profile does not meet the preparation format contract."""


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProfileError(f"{label}: expected exactly keys {sorted(keys)}")
    return value


def _string(value: object, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{label}: expected nonempty string" + (" or null" if optional else ""))


def _integer(value: object, label: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ProfileError(f"{label}: expected integer in [{lower}, {upper}]")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def inspect_profile(path: Path) -> dict:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ProfileError(f"Invalid UTF-8 JSON: {error}") from error
    _object(profile, {"format_version", "profile_id", "game", "adapter", "observation", "action", "control", "evaluation"}, "profile")
    if type(profile["format_version"]) is not int or profile["format_version"] != 1:
        raise ProfileError("format_version: expected 1")
    _string(profile["profile_id"], "profile_id")
    game = _object(profile["game"], {"title", "genre", "platform", "task"}, "game")
    for key, value in game.items():
        _string(value, f"game.{key}", optional=True)
    if game["platform"] not in (None, "pc", "emulator", "console", "synthetic"):
        raise ProfileError("game.platform: unsupported platform")
    adapter = _object(profile["adapter"], {"capture", "input", "reset"}, "adapter")
    for key, value in adapter.items():
        _string(value, f"adapter.{key}", optional=True)
    observation = _object(profile["observation"], {"width", "height", "policy_hz", "clock", "preprocess_version"}, "observation")
    for key in ("width", "height"):
        _integer(observation[key], f"observation.{key}", 16, 4096)
    rate = observation["policy_hz"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 240:
        raise ProfileError("observation.policy_hz: expected finite number in (0, 240]")
    if observation["clock"] != "monotonic_ns":
        raise ProfileError("observation.clock: expected monotonic_ns")
    _string(observation["preprocess_version"], "observation.preprocess_version")
    action = _object(profile["action"], {"spec_version", "buttons", "axes"}, "action")
    _string(action["spec_version"], "action.spec_version")
    for key in ("buttons", "axes"):
        values = action[key]
        if not isinstance(values, list) or len(values) > 64:
            raise ProfileError(f"action.{key}: expected list with at most 64 entries")
        for value in values:
            _string(value, f"action.{key} entry")
        if len(values) != len(set(values)):
            raise ProfileError(f"action.{key}: duplicate entries")
    if set(action["buttons"]) & set(action["axes"]):
        raise ProfileError("action: button and axis names must be distinct")
    control = _object(profile["control"], {"start_disarmed", "human_priority", "neutral_on_fault", "watchdog_timeout_ms"}, "control")
    for key in ("start_disarmed", "human_priority", "neutral_on_fault"):
        if control[key] is not True:
            raise ProfileError(f"control.{key}: must be true")
    _integer(control["watchdog_timeout_ms"], "control.watchdog_timeout_ms", 1, 5000)
    evaluation = _object(profile["evaluation"], {"protocol_id", "success_rule", "reset_rule", "holdout_unit"}, "evaluation")
    for key in ("protocol_id", "success_rule", "reset_rule"):
        _string(evaluation[key], f"evaluation.{key}", optional=True)
    if evaluation["holdout_unit"] != "session":
        raise ProfileError("evaluation.holdout_unit: must be session to prevent adjacent-frame leakage")
    unresolved = [f"game.{key}" for key, value in game.items() if value is None]
    unresolved += [f"adapter.{key}" for key, value in adapter.items() if value is None]
    unresolved += [f"evaluation.{key}" for key in ("protocol_id", "success_rule", "reset_rule") if evaluation[key] is None]
    if not action["buttons"] and not action["axes"]:
        unresolved.append("action.buttons_or_axes")
    return {
        "profile_id": profile["profile_id"],
        "structurally_valid": True,
        "configuration_complete": not unresolved,
        "unresolved": unresolved,
        "runtime_ready": False,
        "limitations": ["Configuration validation does not validate hardware, adapters or a trained model."],
    }
