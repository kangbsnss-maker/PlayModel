"""Bounded HWND-only combat pilot. Importing this module never sends input.

Fast path: latest PrintWindow frame -> local vision/policy worker -> owned WASD.
OCR, artifact writes and learning stay outside the control actor. A separate
safety thread requests release on F8, stop-file, missing frames or expired lease.
This thread does not guarantee release after a whole-process/OS crash.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import threading
import time
import unicodedata
import uuid
from typing import Any, Callable

from playmodel.control import ControlLimits, Observation, RealtimeController, SendReceipt
from playmodel.learning import (
    NEUTRAL_ONLY_MASK, EpisodeRecord, LinearMovementPolicy, MovementDecision, MovementStep, StateEvidence,
    extract_features, load_checkpoint, reinforce_update, save_checkpoint,
)
from .background import BackgroundController
from .capture import _png, capture_session
from .ocr import read_menu, rows_in_region
from .stream import CaptureStream, Frame
from .vision import BrotatoVision, VisionObservation
from .motion import MovementStabilizer


CLOCK = "perf_counter_ns_same_host"


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PilotConfig:
    max_seconds: float = 60.0
    max_steps: int = 1200
    fps: float = 20.0
    stride: int = 6
    max_frame_age_ms: float = 250.0
    policy_budget_ms: float = 100.0
    input_watchdog_ms: float = 200.0
    tick_ms: float = 5.0
    startup_seconds: float = 5.0
    terminal_wait_seconds: float = 18.0
    transient_unknown_seconds: float = 0.5
    seed: int = 20260927
    train: bool = False
    defer_training: bool = False
    movement_hold_ms: float = 0

    def __post_init__(self) -> None:
        bounded = {"max_seconds": (0, 600), "fps": (0, 60), "max_frame_age_ms": (0, 1000),
                   "policy_budget_ms": (0, 500), "input_watchdog_ms": (0, 1000),
                   "tick_ms": (0, 20), "startup_seconds": (0, 30), "terminal_wait_seconds": (-1, 30),
                   "transient_unknown_seconds": (0, 0.5)}
        for name, (low, high) in bounded.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"invalid bounded pilot setting: {name}")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 12000:
            raise ValueError("max_steps must be in [1,12000]")
        if type(self.stride) is not int or not 1 <= self.stride <= 16:
            raise ValueError("stride must be in [1,16]")
        if (self.terminal_wait_seconds < 0 or type(self.seed) is not int or self.seed < 0
                or type(self.train) is not bool or type(self.defer_training) is not bool):
            raise ValueError("invalid seed or training flag")
        if type(self.movement_hold_ms) not in (int,float) or not 0 <= self.movement_hold_ms <= 300:
            raise ValueError('Invalid movement hold time')


@dataclass(frozen=True)
class ActionPacket:
    frame: Frame
    decision: MovementDecision
    vision: VisionObservation
    decided_at_ns: int


class ArtifactWriter:
    """Bounded nonblocking producer; slow disk causes a stop, not silent data loss."""

    def __init__(self, directory: Path, capacity: int = 32):
        self.directory = directory
        (directory / "frames").mkdir()
        self.pending: queue.Queue[tuple[MovementStep, ActionPacket]] = queue.Queue(capacity)
        self.steps: list[MovementStep] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="brotato-artifacts", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            with (self.directory / "actions.jsonl").open("x", encoding="utf-8") as ledger:
                while not self._stop.is_set() or not self.pending.empty():
                    try:
                        step, packet = self.pending.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    frame_path = self.directory / step.frame_ref
                    with frame_path.open("xb") as stream:
                        stream.write(packet.frame.pixels)
                    _json(frame_path.with_suffix(".json"), {
                        "metadata": packet.frame.metadata,
                        "available_at_ns": packet.frame.available_at_ns,
                        "vision": asdict(packet.vision), "vision_is_ground_truth": False,
                    })
                    ledger.write(json.dumps({"step": asdict(step), "transport": "hwnd_postmessage",
                                             "game_application_verified": False}, allow_nan=False) + "\n")
                    ledger.flush()
                    self.steps.append(step)
                    self.pending.task_done()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self, timeout: float = 3) -> bool:
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive() and self.error is None


class MovementSink:
    """Only movement and owned release; no menu, activation or cursor capabilities."""

    def __init__(self, background: BackgroundController, writer: ArtifactWriter, hwnd: int,
                 *, clock: Callable[[], int] = time.perf_counter_ns):
        self.background, self.writer, self.hwnd, self.clock = background, writer, hwnd, clock
        self.previous_action = 0
        self.sent_count = 0
        self.last_sent_at_ns: int | None = None
        self.release_failed = False

    def send(self, action: ActionPacket, *, generation: int, observation_sequence: int,
             deadline_ns: int) -> SendReceipt:
        if not isinstance(action, ActionPacket):
            raise ValueError("pilot requires an ActionPacket")
        if (self.writer.error or self.writer.pending.full()
                or action.frame.metadata["hwnd"] != self.hwnd
                or action.frame.sequence != observation_sequence):
            raise OSError("artifact backpressure or changed frame identity")
        if self.clock() >= deadline_ns:
            raise OSError("movement deadline expired")
        dx, dy = action.decision.movement
        keys = set()
        if dx:
            keys.add(0x44 if dx > 0 else 0x41)
        if dy:
            keys.add(0x53 if dy > 0 else 0x57)
        # BackgroundController.check verifies identity/minimize/F8 immediately before posting.
        self.background.set_movement(keys)
        sent = self.clock()
        self.sent_count += 1
        self.previous_action = action.decision.action_index
        self.last_sent_at_ns = sent
        step = MovementStep(
            observation_sequence, generation, f"frames/{self.sent_count:06d}.bgra",
            hashlib.sha256(action.frame.pixels).hexdigest(),
            action.frame.metadata["capture_started_at_ns"], action.frame.available_at_ns,
            action.decided_at_ns, sent, action.decision, action.decision.action_index, True,
            f"actions.jsonl#{self.sent_count}", acknowledged=None,
        )
        self.writer.pending.put_nowait((step, action))
        return SendReceipt(True, None)  # Posting is not game acknowledgement/application.

    def release(self, *, generation: int, reason: str, deadline_ns: int) -> SendReceipt:
        try:
            self.background.release()
        except Exception:
            self.release_failed = True
            raise
        self.previous_action = 0
        self.last_sent_at_ns = None
        return SendReceipt(True, None)


class TerminalRules:
    """Explicitly calibrated OCR text rules; no guessed default win/death vocabulary."""

    def __init__(self, document: dict):
        if (document.get("schema") != "brotato-pilot-terminal-rules-v1"
                or document.get("verified_against_game_build") is not True
                or not document.get("calibration_ref") or not document.get("game_build_id")):
            raise ValueError("terminal rules need game-build calibration evidence")
        rules = document.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("terminal rules are missing")
        for rule in rules:
            if (not isinstance(rule, dict) or rule.get("kind") not in ("wave_clear", "death")
                    or not isinstance(rule.get("all_text"), list) or len(rule["all_text"]) < 2
                    or any(type(text) is not str or not text.strip() for text in rule["all_text"])
                    or not isinstance(rule.get("forbidden_text", []), list)
                    or any(type(text) is not str or not text.strip() for text in rule.get("forbidden_text", []))):
                raise ValueError("each terminal rule needs at least two calibrated text anchors")
            regions = rule.get("regions", {})
            if not isinstance(regions, dict) or any(token not in rule["all_text"] for token in regions):
                raise ValueError("terminal regions must map configured text anchors")
            for region in regions.values():
                if (not isinstance(region, list) or len(region) != 4
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in region)
                        or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)):
                    raise ValueError("terminal regions must be normalized [left,top,right,bottom]")
        self.document = document
        self.version = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(unicodedata.normalize("NFKC", text).lower().split())

    def classify(self, text: str, *, lines: list | None = None,
                 image_size: tuple[int, int] | None = None) -> str | None:
        observed = self._normalize(text)

        def region_matches(rule):
            for token, (left, top, right, bottom) in rule.get("regions", {}).items():
                if lines is None or image_size is None:
                    return False
                width, height = image_size
                if width <= 0 or height <= 0:
                    return False
                try:
                    words = [word for line in lines for word in line["words"]
                             if left <= (word["x"] + word["width"] / 2) / width <= right
                             and top <= (word["y"] + word["height"] / 2) / height <= bottom]
                    region_text = self._normalize("".join(rows_in_region(
                        {"lines": [{"words": words}]}, (0, 0, width, height))))
                except (KeyError, TypeError):
                    return False
                if self._normalize(token) not in region_text:
                    return False
            return True

        matches = {rule["kind"] for rule in self.document["rules"]
                   if all(self._normalize(token) in observed for token in rule["all_text"])
                   and not any(self._normalize(token) in observed for token in rule.get("forbidden_text", []))
                   and region_matches(rule)}
        return matches.pop() if len(matches) == 1 else None


class LocalTerminalObserver:
    """Full-resolution local OCR on the slow path, independent of movement policy."""

    def __init__(self, executable: Path, directory: Path, script: Path, rules: TerminalRules):
        self.executable, self.directory, self.script, self.rules = executable, directory, script, rules
        self.previous: tuple[str, int] | None = None
        from .menu_capture import MenuCapture
        from .ocr import MenuOcr
        # Slow observations need fresh independent frames, not a fresh Python
        # and PowerShell process on every observation. Keep their CPU load low.
        self.capture = MenuCapture(executable, fps=4)
        self.reader = MenuOcr(script, cache_seconds=0)

    def __call__(self) -> StateEvidence | None:
        capture, _, _, _ = self.capture.read(self.directory / "ocr", timeout=4)
        source = Path(capture["session_directory"])
        raw = self.reader.read(source / "frame.png")
        # OCR helper's internal monotonic timestamps are diagnostic only. These use runtime QPC.
        available = time.perf_counter_ns()
        _json(source / "ocr.json", {"raw": raw, "runtime_available_at_ns": available, "clock": CLOCK})
        kind = self.rules.classify(raw.get("text", ""), lines=raw.get("lines"),
                                   image_size=(capture["width"], capture["height"]))
        observed = capture["capture_started_at_ns"]
        previous, self.previous = self.previous, (kind, observed) if kind else None
        if (kind is None or previous is None or previous[0] != kind
                or observed - previous[1] < 200_000_000):
            return None
        return StateEvidence(kind, str((source / "frame.png").resolve()), capture["frame_sha256"],
                             observed, available, available, "local_detector",
                             f"calibrated_local_ocr:{self.rules.version}", True, True)

    def close(self):
        try:
            self.reader.close()
        finally:
            self.capture.close()


class SlowTerminalWorker:
    def __init__(self, observer: Callable[[], StateEvidence | None]):
        self.observer = observer
        self.latest: StateEvidence | None = None
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="brotato-terminal-ocr", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    evidence = self.observer()
                    if evidence is not None:
                        self.latest = evidence
                except Exception as error:
                    self.error = f"{type(error).__name__}: {error}"
                self.stop.wait(0.5)
        finally:
            close = getattr(self.observer, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception as error:
                    self.error = f"observer_close:{type(error).__name__}: {error}"

    def close(self) -> bool:
        self.stop.set()
        self.thread.join(0.05)
        return not self.thread.is_alive()


class SafetyMonitor:
    """Independent of capture, vision/policy, OCR and disk writer work."""

    def __init__(self, controller: RealtimeController, sink: MovementSink, stream: CaptureStream,
                 stop_file: Path, config: PilotConfig, started_ns: int):
        self.controller, self.sink, self.stream = controller, sink, stream
        self.stop_file, self.config, self.started_ns = stop_file, config, started_ns
        self.reason: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="brotato-safety", daemon=True)
        self.thread.start()

    def trip(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            self.controller.disarm(reason)

    def _run(self) -> None:
        while not self.stop.is_set() and self.reason is None:
            try:
                if self.stop_file.exists():
                    self.trip("stop_file")
                elif time.perf_counter_ns() - self.started_ns >= self.config.max_seconds * 1e9:
                    self.trip("time_limit")
                elif self.stream.error:
                    self.trip("capture_error")
                elif self.stream.latest(max_age_ms=self.config.max_frame_age_ms) is None:
                    self.trip("stale_capture")
                elif self.sink.writer.error:
                    self.trip("artifact_error")
                elif (self.sink.last_sent_at_ns is not None and time.perf_counter_ns() - self.sink.last_sent_at_ns
                      >= self.config.input_watchdog_ms * 1e6):
                    self.trip("independent_input_watchdog")
                else:
                    self.sink.background.check()  # F8/identity/minimize, no input.
                    context = getattr(self.sink.background, "_context", None)
                    if (context is not None and context.foreground() == self.sink.hwnd
                            and any(context.key_state(key) & 0x8000 for key in (0x57, 0x41, 0x53, 0x44))):
                        self.trip("human_intervention")
            except Exception as error:
                self.trip("F8" if "F8" in str(error) else "input_guard_error")
            self.stop.wait(0.005)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(0.1)


def load_combat_entry(path: Path) -> StateEvidence:
    evidence = StateEvidence(**json.loads(path.read_text(encoding="utf-8")))
    if (evidence.kind != "combat" or evidence.verified is not True
            or evidence.independent_of_policy is not True or evidence.clock_domain != CLOCK
            or evidence.origin not in ("developer_verified", "local_detector")):
        raise ValueError("combat entry needs independent verified frame evidence")
    source = Path(evidence.frame_ref)
    if not source.is_absolute():
        source = path.parent / source
    if _sha(source) != evidence.frame_sha256:
        raise ValueError("combat entry source digest mismatch")
    return StateEvidence(**{**asdict(evidence), "frame_ref": str(source.resolve())})


def _frame_ok(frame: Frame, *, now: int, max_age_ms: float, hwnd: int | None = None) -> bool:
    metadata = frame.metadata
    try:
        return (metadata["backend"] == "win32_printwindow_client" and metadata["clock"] == CLOCK
                and (hwnd is None or metadata["hwnd"] == hwnd)
                and 0 <= metadata["capture_started_at_ns"] <= metadata["capture_finished_at_ns"]
                <= frame.available_at_ns <= now
                and now - metadata["capture_started_at_ns"] < max_age_ms * 1e6
                and len(frame.pixels) == metadata["sample_width"] * metadata["sample_height"] * 4)
    except (KeyError, TypeError):
        return False


def run_pilot(executable: Path, output_root: Path, *, config: PilotConfig = PilotConfig(),
              stop_file: Path | None = None, policy: LinearMovementPolicy | None = None,
              combat_entry: StateEvidence | None = None, terminal_rules: TerminalRules | None = None,
              ocr_script: Path | None = None,
              stream_factory: Callable[..., Any] = CaptureStream,
              background_factory: Callable[..., Any] = BackgroundController,
              vision_factory: Callable[[], Any] = BrotatoVision,
              terminal_observer: Callable[[], StateEvidence | None] | None = None,
              scene_callback: Callable[[str], None] | None = None) -> dict:
    """Run one bounded combat attempt; real side effects happen only on this call.

    No menu/retry inputs are available. --train additionally requires independently
    verified combat entry and calibrated terminal OCR. Failed/truncated attempts
    preserve their evidence and always report training_performed=False.
    """
    if config.train and (combat_entry is None or (terminal_rules is None and terminal_observer is None)):
        raise ValueError("training needs verified combat entry and independent terminal observer/rules")
    session = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    stop_file = stop_file or session / "STOP"
    policy = policy or LinearMovementPolicy.initialized_for_collection(seed=config.seed)
    save_checkpoint(policy, session / "initial-policy.json")
    _json(session / "capture-config.json", {**asdict(config), "backend": "printwindow", "clock": CLOCK,
                                           "input": "background_hwnd_wasd", "stop_file": str(stop_file.resolve())})
    if combat_entry:
        _json(session / "combat-entry.json", asdict(combat_entry))
    if terminal_rules:
        _json(session / "terminal-rules.json", terminal_rules.document)
    writer = ArtifactWriter(session)
    stream = controller = safety = terminal_worker = None
    reason, terminal, sink = "startup_failed", None, None
    phase_ended = threading.Event()
    phase_reason = [None]
    transient_started_ns: list[int | None] = [None]
    # Keep the exact rejected frame, not a later screenshot of a changed screen.
    # This is only a reference assignment on the policy path; encoding/I/O follows release.
    phase_observation: list[tuple[Frame, VisionObservation, int] | None] = [None]
    worker_stopped, recorder_complete, ocr_stopped = True, False, True
    cleanup_errors = []
    generation, caught_error = 0, None
    try:
        stream = stream_factory(executable, fps=config.fps, stride=config.stride, backend="printwindow")
        deadline = time.perf_counter() + config.startup_seconds
        first = None
        while first is None and time.perf_counter() < deadline:
            if stop_file.exists():
                reason = "stop_file"
                break
            if stream.error:
                raise OSError(stream.error)
            first = stream.latest(max_age_ms=config.max_frame_age_ms)
            if first is None:
                time.sleep(0.005)
        if first is None:
            raise OSError("No fresh startup frame")
        if not _frame_ok(first, now=time.perf_counter_ns(), max_age_ms=config.max_frame_age_ms):
            raise OSError("Unsupported/stale startup frame")
        if config.train and combat_entry is not None:
            age = first.metadata["capture_started_at_ns"] - combat_entry.observed_at_ns
            if not 0 <= age <= 60_000_000_000:
                raise ValueError("combat-entry evidence must precede startup by at most 60 seconds")
        hwnd = first.metadata["hwnd"]
        background = background_factory(hwnd, executable)
        background.release()
        sink = MovementSink(background, writer, hwnd)
        vision, rng = vision_factory(), random.Random(config.seed + 1)
        stabilizer = MovementStabilizer(config.movement_hold_ms)

        def local_policy(observation: Observation, policy_deadline: int) -> ActionPacket | None:
            frame = observation.payload
            state = vision.observe(frame.pixels, frame.metadata["sample_width"], frame.metadata["sample_height"],
                                   observed_at_ns=observation.observed_at_ns, alpha_mode="ignore")
            mask = None
            if (state.status != "heuristic_observation" or not state.combat_likely
                    or state.player is None or not state.navigation[2]):
                # A still-visible combat HUD with an unknown player permits only a
                # recorded, deterministic neutral action. This is part of the
                # fixed conditional policy, not an unrecorded recovery override.
                transient = (state.combat_likely and state.status in (
                    "player_unknown", "player_ambiguous", "camera_shift_candidate",
                    "heuristic_observation", "too_many_candidates")) or (
                    state.status == 'combat_hud_not_found' and sink.sent_count > 0)
                if transient:
                    if transient_started_ns[0] is None:
                        transient_started_ns[0] = observation.observed_at_ns
                    elapsed = observation.observed_at_ns - transient_started_ns[0]
                    if elapsed < config.transient_unknown_seconds * 1e9:
                        mask = NEUTRAL_ONLY_MASK
                    else:
                        phase_reason[0] = (state.status if state.status == 'combat_hud_not_found'
                                           else f"perception_timeout:{state.status}")
                else:
                    phase_reason[0] = state.status
                if mask is None:
                    if scene_callback:
                        scene_callback("unknown")
                    phase_observation[0] = (frame, state, time.perf_counter_ns())
                    phase_ended.set()
                    return None
            else:
                transient_started_ns[0] = None
            if scene_callback:
                scene_callback("combat" if mask is None else "unknown")
            features = extract_features(frame.pixels, frame.metadata["sample_width"], frame.metadata["sample_height"],
                                        sink.previous_action, navigation=None if mask else state.navigation)
            if mask is None and config.movement_hold_ms:
                mask = stabilizer.mask(state, sink.previous_action, observation.observed_at_ns)
            decision = policy.sample(features, rng=rng, mask=mask)
            return ActionPacket(frame, decision, state, time.perf_counter_ns())

        limits = ControlLimits(int(config.max_frame_age_ms * 1e6), int(config.policy_budget_ms * 1e6),
                               25_000_000, int(config.input_watchdog_ms * 1e6), config.max_steps * 3 + 32)
        controller = RealtimeController(local_policy, sink, limits)
        generation = controller.arm()
        started = time.perf_counter_ns()
        safety = SafetyMonitor(controller, sink, stream, stop_file, config, started)
        if terminal_observer is None and terminal_rules is not None:
            if ocr_script is None:
                raise ValueError("local OCR script is required with terminal rules")
            terminal_observer = LocalTerminalObserver(executable, session, ocr_script, terminal_rules)
        if terminal_observer is not None:
            terminal_worker = SlowTerminalWorker(terminal_observer)
        last_sequence = -1
        reason = "time_limit"
        while True:
            if safety.reason:
                reason = safety.reason
                break
            if not controller.armed:
                reason = "controller_guard"
                break
            if phase_ended.is_set():
                reason = "perception_timeout" if str(phase_reason[0]).startswith("perception_timeout:") else "screen_changed"
                controller.disarm(reason)
                break
            if terminal_worker and terminal_worker.latest:
                terminal = terminal_worker.latest
                if terminal.observed_at_ns >= started:
                    reason = f"terminal_{terminal.kind}"
                    controller.disarm(reason)
                    break
                terminal = None
            if sink.sent_count >= config.max_steps:
                reason = "step_limit"
                controller.disarm(reason)
                break
            frame = stream.latest(max_age_ms=config.max_frame_age_ms)
            if frame is not None and frame.sequence != last_sequence:
                if not _frame_ok(frame, now=time.perf_counter_ns(), max_age_ms=config.max_frame_age_ms, hwnd=hwnd):
                    reason = "frame_contract_error"
                    controller.disarm(reason)
                    break
                controller.publish(Observation(frame.sequence, generation, frame.metadata["capture_started_at_ns"],
                                               frame.available_at_ns, frame))
                last_sequence = frame.sequence
            controller.tick()
            time.sleep(config.tick_ms / 1000)
        controller.disarm(reason)
        # No movement resumes while slow OCR distinguishes a terminal screen from unknown UI.
        if reason == "screen_changed" and terminal_worker:
            wait_until = time.perf_counter() + config.terminal_wait_seconds
            while time.perf_counter() < wait_until and not stop_file.exists():
                if safety.reason:
                    reason = safety.reason
                    break
                candidate = terminal_worker.latest
                if (candidate is not None and sink.last_sent_at_ns is None
                        and candidate.observed_at_ns >= started):
                    terminal, reason = candidate, f"terminal_{candidate.kind}"
                    break
                time.sleep(0.01)
    except Exception as error:
        caught_error = f"{type(error).__name__}: {error}"
        if reason not in ("stop_file",):
            reason = "runtime_error"
    finally:
        if scene_callback:
            scene_callback("unknown")
        if safety is not None:
            safety.close()
        if controller is not None:
            worker_stopped = controller.close(timeout=0.1)
        elif sink is not None:
            sink.release(generation=0, reason="startup_failure", deadline_ns=time.perf_counter_ns() + 25_000_000)
        if terminal_worker is not None:
            ocr_stopped = terminal_worker.close()
        if stream is not None:
            try:
                stream.close()
            except Exception as error:
                cleanup_errors.append(f"capture_close:{type(error).__name__}:{error}")
        recorder_complete = writer.close()
    if safety is not None and safety.reason is not None:
        reason = safety.reason
    if terminal is not None:
        _json(session / "terminal.json", asdict(terminal))
    phase_frame_record = None
    if phase_observation[0] is not None:
        frame, state, rejected_at = phase_observation[0]
        raw_path, png_path = session / "phase-frame.bgra", session / "phase-frame.png"
        with raw_path.open("xb") as destination:
            destination.write(frame.pixels)
        with png_path.open("xb") as destination:
            destination.write(_png(frame.metadata["sample_width"], frame.metadata["sample_height"], frame.pixels))
        phase_frame_record = {
            "sequence": frame.sequence, "metadata": frame.metadata,
            "available_at_ns": frame.available_at_ns, "rejected_at_ns": rejected_at,
            "vision": asdict(state), "reason": state.status, "clock": CLOCK,
            "raw_frame": raw_path.name, "raw_sha256": _sha(raw_path),
            "preview": png_path.name, "preview_sha256": _sha(png_path),
            "purpose": "phase_stop_evidence", "used_for_training": False,
        }
        _json(session / "phase-frame.json", phase_frame_record)
    control_events = controller.events() if controller is not None else ()
    if controller is not None:
        _json(session / "control-events.json", [asdict(event) for event in control_events])
    normal_authority = {"ai_granted", "screen_changed", "terminal_wave_clear", "terminal_death", "closed"}
    guard_failure = any((event.kind == "authority" and event.reason not in normal_authority)
                        or (event.kind == "dispatch" and event.reason != "transmitted")
                        or (event.kind == "release" and event.reason.startswith("release_"))
                        for event in control_events)
    originals = ["initial-policy.json", "capture-config.json", "actions.jsonl"]
    originals += [name for name in ("combat-entry.json", "terminal.json", "terminal-rules.json",
                                    "phase-frame.bgra", "phase-frame.png", "phase-frame.json")
                  if (session / name).exists()]
    manifest = {"schema": "brotato-pilot-manifest-v1", "episode_id": session.name,
                "files": [{"path": name, "sha256": _sha(session / name)} for name in originals
                          if (session / name).exists()],
                "frames": [{"path": step.frame_ref, "sha256": step.frame_sha256} for step in writer.steps],
                "steps": len(writer.steps), "recorder_complete": recorder_complete}
    _json(session / "manifest.json", manifest)
    report = {"session_directory": str(session.resolve()), "status": "aborted", "reason": reason,
              "training_performed": False, "policy_version": policy.version, "baseline_origin": policy.baseline_origin,
              "actions_posted": sink.sent_count if sink else 0, "actions_recorded": len(writer.steps),
              "game_application_verified": False, "runtime_ready": False, "screen_status": phase_reason[0],
              "phase_frame": "phase-frame.json" if phase_frame_record else None,
              "error": caught_error, "artifact_error": writer.error,
              "cleanup_errors": cleanup_errors, "guard_failure": guard_failure,
              "policy_worker_stopped": worker_stopped, "ocr_worker_stopped": ocr_stopped,
              "release_failed": sink.release_failed if sink else False,
              "terminal_origin": terminal.origin if terminal else None,
              "ocr_error": terminal_worker.error if terminal_worker else None,
              "stop_file": str(stop_file.resolve()), "safety": "thread_watchdog_not_process_crash_guarantee"}
    changes = sum(a.actual_action_index != b.actual_action_index for a,b in zip(writer.steps,writer.steps[1:]))
    seconds_observed = ((writer.steps[-1].sent_at_ns-writer.steps[0].sent_at_ns)/1e9 if len(writer.steps)>1 else 0.)
    report['movement_metrics'] = {'direction_changes':changes,'seconds_observed':seconds_observed,
                                  'changes_per_second':changes/seconds_observed if seconds_observed else None,
                                  'hold_ms':config.movement_hold_ms,'method':'recorded_conditional_action_mask'}
    complete = (reason in ("terminal_wave_clear", "terminal_death") and recorder_complete
                and sink is not None and not sink.release_failed and worker_stopped and not guard_failure
                and not cleanup_errors
                and len(writer.steps) == sink.sent_count and bool(writer.steps))
    if complete:
        report["status"] = "terminal_observed"
    if config.train and complete:
        episode = EpisodeRecord(session.name, policy.version, "manifest.json", _sha(session / "manifest.json"),
                                "capture-config.json", generation, 0, combat_entry, tuple(writer.steps), terminal)
        _json(session / "episode.json", asdict(episode))
        if config.defer_training:
            report.update(status='training_data_ready', training_deferred=True)
            _json(session / 'report.json', report)
            return report
        try:
            update = reinforce_update(policy, episode)
            _json(session / "learning-update.json", update.report)
            if update.report["weights_changed"]:
                candidate = save_checkpoint(update.policy, session / "candidate-policy.json")
                if load_checkpoint(candidate).version != update.policy.version:
                    raise ValueError("candidate reload mismatch")
                report.update(training_performed=True, candidate_policy_version=update.policy.version,
                              status="candidate_saved_pending_evaluation")
            else:
                report.update(status="terminal_observed_no_parameter_change", training_skip="zero_gradient")
        except (OSError, ValueError) as error:
            report["training_rejection"] = str(error)
    _json(session / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded local Brotato combat pilot; HWND WASD only, F8/stop-file stops")
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/brotato-pilot"))
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--combat-entry", type=Path)
    parser.add_argument("--terminal-rules", type=Path)
    parser.add_argument("--ocr-script", type=Path, default=Path("scripts/windows_ocr.ps1"))
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    if args.exe.name.lower() != "brotato.exe" or not args.exe.is_file():
        parser.error("--exe must identify the installed Brotato.exe")
    config = PilotConfig(max_seconds=args.seconds, max_steps=args.max_steps, fps=args.fps,
                         stride=args.stride, seed=args.seed, train=args.train)
    rules = TerminalRules(json.loads(args.terminal_rules.read_text(encoding="utf-8"))) if args.terminal_rules else None
    entry = load_combat_entry(args.combat_entry) if args.combat_entry else None
    report = run_pilot(args.exe, args.output, config=config, stop_file=args.stop_file,
                       policy=load_checkpoint(args.checkpoint) if args.checkpoint else None,
                       combat_entry=entry, terminal_rules=rules, ocr_script=args.ocr_script)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["status"] != "aborted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
