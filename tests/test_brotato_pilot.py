"""Fake game integration; this suite never captures a desktop or sends OS input."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest

from playmodel.control import ControlLimits, Observation, RealtimeController
from playmodel.learning import ALL_ACTIONS_MASK, NEUTRAL_ONLY_MASK, LinearMovementPolicy, StateEvidence, extract_features
from playmodel.games.brotato.pilot import (
    ActionPacket, ArtifactWriter, MovementSink, PilotConfig, SafetyMonitor,
    TerminalRules, _frame_ok, load_combat_entry, run_pilot,
)
from playmodel.games.brotato.stream import Frame
from playmodel.games.brotato.vision import VisionObservation


class FakeStream:
    def __init__(self, executable=None, **kwargs):
        self.error = None
        self.sequence = 0
        self.lock = threading.Lock()
        self.backend = kwargs.get("backend", "printwindow")
        self.closed = False

    def latest(self, **kwargs):
        with self.lock:
            now = time.perf_counter_ns()
            self.sequence += 1
            return Frame(self.sequence, {"backend": "win32_printwindow_client", "clock": "perf_counter_ns_same_host",
                                        "capture_started_at_ns": now - 100_000, "capture_finished_at_ns": now - 50_000,
                                        "sample_width": 80, "sample_height": 45, "hwnd": 7},
                         bytes((30, 60, 90, 255)) * (80 * 45), now)

    def close(self):
        self.closed = True


class FakeBackground:
    def __init__(self, hwnd=7, executable=None):
        self.held = set()
        self.sent = []
        self.releases = 0
        self.stop_f8 = False

    def check(self):
        if self.stop_f8:
            raise OSError("Stopped by F8")

    def set_movement(self, keys):
        self.check()
        if not keys <= {0x57, 0x41, 0x53, 0x44}:
            raise AssertionError("not movement keys")
        self.held = set(keys)
        self.sent.append(set(keys))

    def release(self):
        self.held.clear()
        self.releases += 1


class FakeVision:
    def __init__(self, background, *, stop_after=3):
        self.background, self.stop_after = background, stop_after
        self.menu = threading.Event()

    def observe(self, *args, **kwargs):
        if len(self.background.sent) >= self.stop_after:
            self.menu.set()
            return VisionObservation(status="possible_menu_overlay")
        return VisionObservation(navigation=(1, 0, True), player=(0.5, 0.5), player_confidence=0.5,
                                 combat_likely=True, status="heuristic_observation",
                                 observed_at_ns=kwargs["observed_at_ns"])


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = PilotConfig(max_seconds=0.5, max_steps=50, startup_seconds=0.1,
                                  terminal_wait_seconds=0.2, tick_ms=1)
        self.background = FakeBackground()
        self.stream = FakeStream()
        self.vision = FakeVision(self.background)
        self.policy = LinearMovementPolicy.initialized_for_collection(seed=42)

    def tearDown(self):
        self.temp.cleanup()

    def run_fake(self, **kwargs):
        return run_pilot(Path("Brotato.exe"), self.root / "sessions", config=kwargs.pop("config", self.config),
                         stream_factory=lambda *args, **options: self._stream(options),
                         background_factory=lambda *args: self.background,
                         vision_factory=lambda: self.vision, policy=self.policy, **kwargs)

    def _stream(self, options):
        self.assertEqual(options["backend"], "printwindow")
        return self.stream

    def evidence(self, kind, *, at=None):
        source = self.root / f"{kind}.png"
        if not source.exists():
            source.write_bytes(b"synthetic evidence; not a screenshot")
        now = time.perf_counter_ns() if at is None else at
        return StateEvidence(kind, str(source), hashlib.sha256(source.read_bytes()).hexdigest(),
                             now, now, now, "developer_verified", "fixture-review-only", True, True)

    def test_unknown_menu_releases_and_never_learns_without_terminal(self):
        report = self.run_fake()
        self.assertEqual(report["reason"], "screen_changed")
        self.assertEqual(report["status"], "aborted")
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["actions_posted"], 3)
        self.assertEqual(report["actions_recorded"], 3)
        self.assertFalse(self.background.held)
        self.assertTrue(self.stream.closed)
        session = Path(report["session_directory"])
        lines = [json.loads(line) for line in (session / "actions.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 3)
        for record in lines:
            self.assertFalse(record["game_application_verified"])
            self.assertTrue((session / record["step"]["frame_ref"]).is_file())
        self.assertFalse((session / "candidate-policy.json").exists())
        phase = json.loads((session / report["phase_frame"]).read_text())
        last_sent_sequence = lines[-1]["step"]["sequence"]
        self.assertGreater(phase["sequence"], last_sent_sequence)
        self.assertEqual(phase["reason"], "possible_menu_overlay")
        self.assertEqual(phase["vision"]["status"], "possible_menu_overlay")
        self.assertFalse(phase["used_for_training"])
        raw = (session / phase["raw_frame"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), phase["raw_sha256"])
        self.assertEqual(raw, bytes((30, 60, 90, 255)) * (80 * 45))
        self.assertTrue((session / phase["preview"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertLessEqual(phase["metadata"]["capture_started_at_ns"], phase["available_at_ns"])
        self.assertLessEqual(phase["available_at_ns"], phase["rejected_at_ns"])
        manifest = json.loads((session / "manifest.json").read_text())
        recorded_files = {item["path"] for item in manifest["files"]}
        self.assertTrue({"phase-frame.json", "phase-frame.bgra", "phase-frame.png"} <= recorded_files)

    def test_verified_complete_fake_episode_updates_and_reloads_candidate(self):
        entry = self.evidence("combat", at=time.perf_counter_ns() - 1_000_000)

        def terminal_observer():
            self.assertTrue(self.vision.menu.wait(1))
            return self.evidence("wave_clear")

        report = self.run_fake(config=replace(self.config, train=True), combat_entry=entry,
                               terminal_observer=terminal_observer)
        self.assertTrue(report["training_performed"], report)
        self.assertEqual(report["status"], "candidate_saved_pending_evaluation")
        session = Path(report["session_directory"])
        update = json.loads((session / "learning-update.json").read_text())
        self.assertTrue(update["weights_changed"])
        self.assertEqual(update["terminal_origin"], "developer_verified")
        self.assertFalse(update["performance_improvement_verified"])
        self.assertFalse(self.background.held)

    def test_slow_ocr_does_not_block_actions_and_truncation_never_learns(self):
        release = threading.Event()
        entered = threading.Event()
        self.vision = FakeVision(self.background, stop_after=999)

        def slow_ocr():
            entered.set()
            release.wait(2)
            return None

        try:
            report = self.run_fake(config=replace(self.config, max_steps=3), terminal_observer=slow_ocr)
            self.assertTrue(entered.is_set())
            self.assertFalse(release.is_set())
            self.assertEqual(report["actions_posted"], 3)
            self.assertEqual(report["reason"], "step_limit")
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["ocr_worker_stopped"])
            self.assertFalse(self.background.held)
        finally:
            release.set()

    def test_initial_unknown_never_sends_movement(self):
        self.vision = FakeVision(self.background, stop_after=0)
        report = self.run_fake()
        self.assertEqual(report["actions_posted"], 0)
        self.assertFalse(report["training_performed"])
        self.assertGreater(self.background.releases, 0)
        self.assertTrue((Path(report["session_directory"]) / "phase-frame.png").is_file())

    def test_transient_combat_unknown_records_neutral_and_reacquires(self):
        class TransientVision(FakeVision):
            calls = 0

            def observe(vision, *args, **kwargs):
                vision.calls += 1
                if vision.calls in (2, 3):
                    return VisionObservation(status="player_unknown", combat_likely=True,
                                             observed_at_ns=kwargs["observed_at_ns"])
                return super(TransientVision, vision).observe(*args, **kwargs)

        self.vision = TransientVision(self.background, stop_after=4)
        report = self.run_fake()
        session = Path(report["session_directory"])
        steps = [json.loads(line)["step"] for line in (session / "actions.jsonl").read_text().splitlines()]
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0]["decision"]["allowed_actions"], list(ALL_ACTIONS_MASK))
        for step in steps[1:3]:
            self.assertEqual(step["actual_action_index"], 0)
            self.assertEqual(step["decision"]["allowed_actions"], list(NEUTRAL_ONLY_MASK))
            self.assertEqual(step["decision"]["log_probability"], 0)
            self.assertEqual(step["decision"]["features"][153:156], [0, 0, 0])
        self.assertEqual(steps[3]["decision"]["features"][144:153], [1] + [0] * 8)
        self.assertEqual(steps[3]["decision"]["allowed_actions"], list(ALL_ACTIONS_MASK))
        self.assertEqual(self.background.sent[1:3], [set(), set()])
        self.assertEqual(report["reason"], "screen_changed")

    def test_sustained_combat_unknown_is_fatal_not_late_terminal_eligible(self):
        class UnknownVision(FakeVision):
            def observe(vision, *args, **kwargs):
                return VisionObservation(status="player_unknown", combat_likely=True,
                                         observed_at_ns=kwargs["observed_at_ns"])

        self.vision = UnknownVision(self.background)
        report = self.run_fake(config=replace(self.config, transient_unknown_seconds=0.02))
        self.assertEqual(report["reason"], "perception_timeout")
        self.assertTrue(report["guard_failure"])
        self.assertFalse(report["training_performed"])
        self.assertGreater(report["actions_recorded"], 0)
        self.assertTrue(all(not keys for keys in self.background.sent))

    def test_missing_training_evidence_fails_before_capture_or_input(self):
        with self.assertRaises(ValueError):
            self.run_fake(config=replace(self.config, train=True))
        self.assertEqual(self.background.sent, [])
        self.assertFalse((self.root / "sessions").exists())

    def test_preexisting_stop_file_prevents_input(self):
        stop = self.root / "STOP"
        stop.touch()
        report = self.run_fake(stop_file=stop)
        self.assertEqual(report["reason"], "stop_file")
        self.assertEqual(self.background.sent, [])
        self.assertFalse(report["training_performed"])

    def test_independent_watchdog_releases_while_policy_is_blocked(self):
        directory = self.root / "watchdog"
        directory.mkdir()
        writer = ArtifactWriter(directory)
        sink = MovementSink(self.background, writer, 7)
        entered, unblock = threading.Event(), threading.Event()

        def blocked_policy(observation, deadline):
            entered.set()
            unblock.wait(1)
            return None

        controller = RealtimeController(blocked_policy, sink, ControlLimits(500_000_000, 400_000_000,
                                                                          25_000_000, 300_000_000))
        generation = controller.arm()
        frame = self.stream.latest()
        controller.publish(Observation(frame.sequence, generation, frame.metadata["capture_started_at_ns"],
                                       frame.available_at_ns, frame))
        controller.tick()
        self.assertTrue(entered.wait(1))
        self.background.held = {0x44}
        stop = self.root / "watchdog-stop"
        monitor = SafetyMonitor(controller, sink, self.stream, stop, self.config, time.perf_counter_ns())
        try:
            stop.touch()
            deadline = time.perf_counter() + 1
            while monitor.reason is None and time.perf_counter() < deadline:
                time.sleep(0.001)
            self.assertEqual(monitor.reason, "stop_file")
            self.assertFalse(self.background.held)
            self.assertFalse(unblock.is_set())
        finally:
            monitor.close()
            unblock.set()
            controller.close(timeout=0.2)
            writer.close()

    def test_f8_guard_aborts_without_sampling_or_input(self):
        self.background.stop_f8 = True
        report = self.run_fake()
        self.assertEqual(report["reason"], "F8")
        self.assertEqual(report["actions_posted"], 0)
        self.assertFalse(self.background.held)

    def test_frame_backend_identity_and_clock_contract(self):
        frame = self.stream.latest()
        self.assertTrue(_frame_ok(frame, now=time.perf_counter_ns(), max_age_ms=250, hwnd=7))
        for changes in ({"backend": "win32_visible_client_bitblt"}, {"clock": "monotonic_ns_same_host"},
                        {"hwnd": 8}, {"capture_started_at_ns": time.perf_counter_ns() + 1_000_000_000}):
            bad = replace(frame, metadata={**frame.metadata, **changes})
            self.assertFalse(_frame_ok(bad, now=time.perf_counter_ns(), max_age_ms=250, hwnd=7))

    def test_terminal_rules_require_calibration_and_two_unambiguous_anchors(self):
        document = {"schema": "brotato-pilot-terminal-rules-v1", "verified_against_game_build": True,
                    "calibration_ref": "two-verified-fixture-screens", "game_build_id": "fixture",
                    "rules": [{"kind": "wave_clear", "all_text": ["SHOP", "NEXT WAVE"]},
                              {"kind": "death", "all_text": ["DEAD", "RETRY"]}]}
        rules = TerminalRules(document)
        self.assertIsNone(rules.classify("SHOP"))
        self.assertEqual(rules.classify("SHOP\nnext wave"), "wave_clear")
        self.assertIsNone(rules.classify("SHOP NEXT WAVE DEAD RETRY"))
        with self.assertRaises(ValueError):
            TerminalRules({**document, "verified_against_game_build": False})
        with self.assertRaises(ValueError):
            TerminalRules({**document, "rules": [{"kind": "wave_clear", "all_text": ["SHOP"]}]})

    def test_terminal_roi_rejects_same_words_in_wrong_positions(self):
        rules = TerminalRules({"schema": "brotato-pilot-terminal-rules-v1",
                               "verified_against_game_build": True, "calibration_ref": "fixture",
                               "game_build_id": "fixture", "rules": [{"kind": "wave_clear",
                               "all_text": ["SHOP", "NEXT"],
                               "regions": {"SHOP": [0, 0, 0.5, 0.3], "NEXT": [0.5, 0.7, 1, 1]}}]})
        lines = [{"words": [{"text": "SHOP", "x": 10, "y": 10, "width": 10, "height": 10},
                            {"text": "NEXT", "x": 70, "y": 80, "width": 10, "height": 10}]}]
        self.assertEqual(rules.classify("SHOP NEXT", lines=lines, image_size=(100, 100)), "wave_clear")
        self.assertIsNone(rules.classify("SHOP NEXT"))
        lines[0]["words"][0]["y"] = 80
        self.assertIsNone(rules.classify("SHOP NEXT", lines=lines, image_size=(100, 100)))

    def test_guard_failure_cannot_be_overridden_by_late_terminal(self):
        sent = threading.Event()

        class SlowBackground(FakeBackground):
            def set_movement(background, keys):
                super(SlowBackground, background).set_movement(keys)
                time.sleep(0.04)  # Exceeds the real 25 ms sink budget.
                sent.set()

        self.background = SlowBackground()
        self.vision = FakeVision(self.background, stop_after=999)
        entry = self.evidence("combat", at=time.perf_counter_ns() - 1_000_000)

        def terminal_observer():
            sent.wait(1)
            time.sleep(0.002)
            return self.evidence("wave_clear")

        report = self.run_fake(config=replace(self.config, train=True), combat_entry=entry,
                               terminal_observer=terminal_observer)
        self.assertEqual(report["reason"], "controller_guard")
        self.assertTrue(report["guard_failure"])
        self.assertEqual(report["status"], "aborted")
        self.assertFalse(report["training_performed"])
        self.assertFalse(self.background.held)

    def test_combat_entry_requires_preserved_matching_source(self):
        from dataclasses import asdict
        evidence = self.evidence("combat")
        path = self.root / "entry.json"
        path.write_text(json.dumps(asdict(evidence)), encoding="utf-8")
        self.assertEqual(load_combat_entry(path), evidence)
        Path(evidence.frame_ref).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_combat_entry(path)


if __name__ == "__main__":
    unittest.main()
