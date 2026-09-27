"""Behavior checks for preparation boundaries and unsafe configuration rejection."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from playmodel.cli import main
from playmodel.doctor import inspect_environment
from playmodel.profiles import ProfileError, inspect_profile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/profiles/first-game.template.json"


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profile.json"

    def write(self, value=None):
        self.path.write_text(json.dumps(self.profile if value is None else value), encoding="utf-8")
        return self.path

    def test_draft_is_structurally_valid_but_never_ready(self):
        report = inspect_profile(self.write())
        self.assertTrue(report["structurally_valid"])
        self.assertFalse(report["configuration_complete"])
        self.assertFalse(report["runtime_ready"])
        self.assertIn("game.title", report["unresolved"])

    def test_draft_requires_explicit_cli_flag(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["validate-profile", str(self.write())]), 2)
            self.assertEqual(main(["validate-profile", str(self.path), "--allow-draft"]), 0)

    def test_completed_config_does_not_claim_game_compatibility(self):
        self.profile["game"] = {"title": "test fixture", "genre": "puzzle", "platform": "synthetic", "task": "reach goal"}
        self.profile["adapter"] = {"capture": "fixture", "input": "fixture", "reset": "fixture"}
        self.profile["action"]["buttons"] = ["left", "right"]
        self.profile["evaluation"].update(protocol_id="fixture-v1", success_rule="goal event", reset_rule="fixed seed")
        report = inspect_profile(self.write())
        self.assertTrue(report["configuration_complete"])
        self.assertFalse(report["runtime_ready"])

    def test_fail_closed_on_fault_behavior_disabled(self):
        for key in ("start_disarmed", "human_priority", "neutral_on_fault"):
            for bad in (False, 1, "true", None):
                with self.subTest(key=key, value=bad):
                    value = copy.deepcopy(self.profile)
                    value["control"][key] = bad
                    with self.assertRaises(ProfileError):
                        inspect_profile(self.write(value))

    def test_allow_draft_does_not_bypass_invalid_configuration(self):
        self.profile["control"]["human_priority"] = False
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["validate-profile", str(self.write()), "--allow-draft"]), 1)

    def test_invalid_numeric_values_rejected(self):
        cases = [("policy_hz", v) for v in (True, 0, -1, float("nan"), float("inf"), 241)]
        cases += [("width", v) for v in (True, 0, 8192, 320.5)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                profile = copy.deepcopy(self.profile)
                profile["observation"][key] = value
                with self.assertRaises(ProfileError):
                    inspect_profile(self.write(profile))

    def test_frame_level_holdout_rejected(self):
        self.profile["evaluation"]["holdout_unit"] = "frame"
        with self.assertRaises(ProfileError):
            inspect_profile(self.write())

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"format_version": 1, "format_version": 2}', encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "Duplicate JSON key"):
            inspect_profile(self.path)

    def test_unknown_fields_rejected(self):
        self.profile["hidden_option"] = True
        with self.assertRaises(ProfileError):
            inspect_profile(self.write())

    def test_duplicate_actions_rejected(self):
        self.profile["action"]["buttons"] = ["jump", "jump"]
        with self.assertRaises(ProfileError):
            inspect_profile(self.write())

    def test_bad_file_reports_error_without_traceback(self):
        self.path.write_bytes(b"\xff")
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(main(["validate-profile", str(self.path)]), 1)
        self.assertIn("Invalid UTF-8 JSON", stream.getvalue())


class DoctorTests(unittest.TestCase):
    def test_no_gpu_tool_is_a_reported_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("playmodel.doctor.shutil.which", return_value=None):
                report = inspect_environment(Path(directory))
        self.assertEqual(report["gpu"]["status"], "unavailable")
        self.assertFalse(report["runtime_ready"])

    def test_gpu_query_timeout_does_not_hang_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("playmodel.doctor.shutil.which", return_value="test-command"):
                with patch("playmodel.doctor.subprocess.run", side_effect=subprocess.TimeoutExpired("test-command", 10)):
                    report = inspect_environment(Path(directory))
        self.assertEqual(report["gpu"]["status"], "query_failed")

    def test_gpu_detection_does_not_claim_cuda_compatibility(self):
        result = subprocess.CompletedProcess([], 0, "Test GPU, 6144, 560.94\n", "")
        with tempfile.TemporaryDirectory() as directory:
            with patch("playmodel.doctor.shutil.which", return_value="test-command"):
                with patch("playmodel.doctor.subprocess.run", return_value=result):
                    report = inspect_environment(Path(directory))
        self.assertEqual(report["gpu"]["devices"][0]["memory_mib"], "6144")
        self.assertFalse(report["runtime_ready"])


if __name__ == "__main__":
    unittest.main()
