"""Latest-only local control with cancellable generations and measured deadlines.

One policy job may run at a time. OCR, training, capture and persistence must run
elsewhere. A slow policy cannot hold the dispatch lock or accumulate a backlog.
Python cannot forcibly interrupt callbacks: sinks MUST be bounded and cooperative;
the eventual OS adapter additionally needs an independently tested release path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
import time
from typing import Any, Callable, Protocol


def _natural(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")


@dataclass(frozen=True)
class Observation:
    """Same-host QPC timestamps; payload must stay immutable while in use.

    observed_at_ns is capture/sample time, never OCR completion time. It does not
    assert that the game rendered a new frame. available_at_ns is when this
    particular payload became usable. A generation identifies current authority
    and scene/episode; sequence increases within that generation.
    """

    sequence: int
    generation: int
    observed_at_ns: int
    available_at_ns: int
    payload: Any
    clock_domain: str = "perf_counter_ns_same_host"

    def __post_init__(self) -> None:
        for name in ("sequence", "generation", "observed_at_ns", "available_at_ns"):
            _natural(getattr(self, name), name)
        if self.observed_at_ns > self.available_at_ns:
            raise ValueError("observed_at_ns must not exceed available_at_ns")
        if not isinstance(self.clock_domain, str) or not self.clock_domain:
            raise ValueError("clock_domain must be a nonempty string")


@dataclass(frozen=True)
class SlotResult:
    accepted: bool
    reason: str


class LatestObservationSlot:
    """A replace-only slot; a reader never consumes another reader's observation."""

    def __init__(self, max_age_ns: int, *, generation: int = 0,
                 clock: Callable[[], int] = time.perf_counter_ns,
                 clock_domain: str = "perf_counter_ns_same_host") -> None:
        _natural(max_age_ns, "max_age_ns", positive=True)
        _natural(generation, "generation")
        self._max_age_ns = max_age_ns
        self._clock = clock
        if not isinstance(clock_domain, str) or not clock_domain:
            raise ValueError("clock_domain must be a nonempty string")
        self.clock_domain = clock_domain
        self._generation = generation
        self._latest: Observation | None = None
        self._lock = Lock()

    def reset(self, generation: int) -> None:
        _natural(generation, "generation")
        with self._lock:
            if generation <= self._generation:
                raise ValueError("generation must increase")
            self._generation = generation
            self._latest = None

    def publish(self, observation: Observation) -> SlotResult:
        if not isinstance(observation, Observation):
            raise TypeError("expected Observation")
        with self._lock:
            now = self._clock()
            if observation.clock_domain != self.clock_domain:
                return SlotResult(False, "clock_domain_mismatch")
            if observation.generation != self._generation:
                return SlotResult(False, "generation_changed")
            if observation.available_at_ns > now:
                return SlotResult(False, "future_observation")
            if now - observation.observed_at_ns >= self._max_age_ns:
                return SlotResult(False, "stale_observation")
            if self._latest is not None:
                if observation.sequence <= self._latest.sequence:
                    return SlotResult(False, "duplicate_or_out_of_order")
                if observation.observed_at_ns < self._latest.observed_at_ns:
                    return SlotResult(False, "capture_time_regressed")
            self._latest = observation
            return SlotResult(True, "published")

    def latest(self) -> Observation | None:
        """Return a reference in O(1); each consumer must check freshness at use."""
        with self._lock:
            return self._latest


@dataclass(frozen=True)
class ControlLimits:
    """Explicit proposed budgets, not measured game safety guarantees."""

    observation_age_ns: int
    policy_budget_ns: int
    sink_budget_ns: int
    input_watchdog_ns: int
    event_capacity: int = 256
    policy_deadline_retries: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _natural(getattr(self, name), name, positive=name != "policy_deadline_retries")
        if self.policy_deadline_retries > 2:
            raise ValueError("at most two consecutive policy deadline retries")


@dataclass(frozen=True)
class SendReceipt:
    """Sink reports transmission; acknowledgement is not proof of game effect."""

    transmitted: bool
    acknowledged: bool | None = None

    def __post_init__(self) -> None:
        if type(self.transmitted) is not bool:
            raise ValueError("transmitted must be boolean")
        if self.acknowledged is not None and type(self.acknowledged) is not bool:
            raise ValueError("acknowledged must be boolean or None")
        if self.acknowledged is True and not self.transmitted:
            raise ValueError("unsent input cannot be acknowledged")


class InputSink(Protocol):
    """Exclusive writer, nonblocking I/O; never call the controller recursively.

    Send checks the deadline and live window/focus immediately before OS input.
    Release clears ONLY this sink's AI-owned inputs, preserving human input.
    Methods return within their supplied deadline or raise. They must not wait
    on OCR, training, disk/network, or policy locks. An OS adapter must separately
    verify this contract; passing these unit tests does not verify physical stop.
    """

    def send(self, action: Any, *, generation: int, observation_sequence: int,
             deadline_ns: int) -> SendReceipt: ...

    def release(self, *, generation: int, reason: str, deadline_ns: int) -> SendReceipt: ...


@dataclass(frozen=True)
class ControlEvent:
    kind: str
    reason: str
    generation: int
    sequence: int | None
    at_ns: int
    observed_at_ns: int | None = None
    available_at_ns: int | None = None
    policy_started_at_ns: int | None = None
    policy_finished_at_ns: int | None = None
    deadline_ns: int | None = None
    send_started_at_ns: int | None = None
    send_finished_at_ns: int | None = None
    sink_deadline_ns: int | None = None
    receipt: SendReceipt | None = None
    # Only a subsequent game observation can establish application.
    game_applied: None = None
    clock_domain: str = "perf_counter_ns_same_host"
    guard_snapshot: dict[str, int | None] | None = None


@dataclass(frozen=True)
class _Job:
    observation: Observation
    deadline_ns: int


@dataclass(frozen=True)
class _Result:
    job: _Job
    action: Any
    started_at_ns: int
    finished_at_ns: int
    error: str | None


class _PolicyWorker:
    """One daemon worker, one result; never enqueue a second job while occupied."""

    def __init__(self, policy: Callable[[Observation, int], Any], clock: Callable[[], int]) -> None:
        self._policy, self._clock = policy, clock
        self._condition = Condition()
        self._job: _Job | None = None
        self._result: _Result | None = None
        self._occupied = False
        self._closed = False
        self._thread = Thread(target=self._run, name="playmodel-policy", daemon=True)
        self._thread.start()

    def submit(self, job: _Job) -> bool:
        with self._condition:
            if self._closed or self._occupied:
                return False
            self._occupied = True
            self._job = job
            self._condition.notify()
            return True

    def poll(self) -> _Result | None:
        with self._condition:
            result = self._result
            if result is not None:
                self._result = None
                self._occupied = False
            return result

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._job is not None)
                if self._closed:
                    return
                job, self._job = self._job, None
            assert job is not None
            started = self._clock()
            action, error = None, None
            try:
                if started >= job.deadline_ns:
                    error = "policy_deadline"
                else:
                    action = self._policy(job.observation, job.deadline_ns)
            except Exception as exc:
                # Record the type; exception text can contain private payloads.
                error = f"policy_error:{type(exc).__name__}"
            result = _Result(job, action, started, self._clock(), error)
            with self._condition:
                if self._closed:
                    return
                self._result = result

    def close(self, timeout: float) -> bool:
        with self._condition:
            self._closed = True
            self._job = None
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()


class RealtimeController:
    """Pluggable local scheduler, disarmed by default; it does not drive a game.

    tick() performs no policy/OCR/training work and never waits for policy output.
    Use one periodic run() actor or invoke tick() from a dedicated runtime thread.
    All sink calls and authority transitions share one gate. A transition cancels
    pending work before releasing the gate; an already executing sink operation
    must finish first. Deadline guarantees depend on the sink and OS scheduler.
    """

    def __init__(self, policy: Callable[[Observation, int], Any], sink: InputSink,
                 limits: ControlLimits, *, clock: Callable[[], int] = time.perf_counter_ns,
                 clock_domain: str = "perf_counter_ns_same_host") -> None:
        self._sink, self._limits, self._clock = sink, limits, clock
        self._gate = Lock()
        self._generation = 0
        self._armed = False
        self._closed = False
        self._slot = LatestObservationSlot(limits.observation_age_ns, clock=clock, clock_domain=clock_domain)
        self._worker = _PolicyWorker(policy, clock)
        self._inflight: _Job | None = None
        self._last_submitted: tuple[int, int] | None = None
        self._last_sent_at_ns: int | None = None
        self._held_observation_deadline_ns: int | None = None
        self._disarm_reason: str | None = None
        self._retry_count = 0
        self._retry_deadline_ns: int | None = None
        self._retry_source: Observation | None = None
        self._expired_job: _Job | None = None
        self._events: deque[ControlEvent] = deque(maxlen=limits.event_capacity)

    @property
    def generation(self) -> int:
        with self._gate:
            return self._generation

    @property
    def armed(self) -> bool:
        with self._gate:
            return self._armed

    @property
    def disarm_reason(self) -> str | None:
        with self._gate:
            return self._disarm_reason

    @property
    def awaiting_expired_policy(self) -> bool:
        """A cancelled worker still owns the sole slot; newer retry frames may replace it."""
        with self._gate:
            return self._armed and self._expired_job is not None

    def events(self) -> tuple[ControlEvent, ...]:
        """Bounded diagnostic ring; not a durable session/action ledger."""
        with self._gate:
            return tuple(self._events)

    def publish(self, observation: Observation) -> SlotResult:
        with self._gate:
            if self._closed:
                return SlotResult(False, "closed")
            if (self._retry_source is not None and observation.generation == self._generation
                    and (observation.sequence <= self._retry_source.sequence
                         or observation.observed_at_ns <= self._retry_source.observed_at_ns
                         or (self._last_sent_at_ns is not None
                             and observation.observed_at_ns <= self._last_sent_at_ns))):
                return SlotResult(False, "pre_retry_observation")
            return self._slot.publish(observation)

    def latest(self) -> Observation | None:
        """Independent OCR/logging consumers may sample without taking frames away."""
        return self._slot.latest()

    def _advance(self) -> None:
        self._generation += 1
        self._slot.reset(self._generation)
        self._last_submitted = None

    def arm(self) -> int:
        """Explicit authority grant; caller verifies focus/scene/adapter readiness."""
        with self._gate:
            if self._closed:
                raise RuntimeError("controller is closed")
            if self._armed:
                raise RuntimeError("already armed; use invalidate for a scene boundary")
            if self._inflight is not None:
                raise RuntimeError("policy still occupied; tick to collect it before rearming")
            self._advance()
            self._armed = True
            self._disarm_reason = None
            self._retry_count = 0
            self._retry_deadline_ns = self._retry_source = self._expired_job = None
            self._events.append(ControlEvent("authority", "ai_granted", self._generation, None, self._clock(),
                                             clock_domain=self._slot.clock_domain))
            return self._generation

    def _release(self, reason: str) -> bool:
        started = self._clock()
        deadline = started + self._limits.sink_budget_ns
        receipt, error = None, None
        try:
            receipt = self._sink.release(generation=self._generation, reason=reason, deadline_ns=deadline)
            if not isinstance(receipt, SendReceipt) or not receipt.transmitted:
                error = "release_not_transmitted"
            elif receipt.acknowledged is False:
                error = "release_not_acknowledged"
        except Exception as exc:
            error = f"release_error:{type(exc).__name__}"
        finished = self._clock()
        if finished >= deadline:
            error = "release_deadline"
        self._last_sent_at_ns = None
        self._held_observation_deadline_ns = None
        self._events.append(ControlEvent("release", error or reason, self._generation, None, finished,
                                         deadline_ns=deadline, send_started_at_ns=started,
                                         send_finished_at_ns=finished, receipt=receipt,
                                         clock_domain=self._slot.clock_domain))
        return error is None

    def _disarm(self, reason: str) -> None:
        latest = self._slot.latest()
        pending = self._inflight
        snapshot = {
            "latest_sequence": latest.sequence if latest else None,
            "latest_observed_at_ns": latest.observed_at_ns if latest else None,
            "latest_available_at_ns": latest.available_at_ns if latest else None,
            "pending_sequence": pending.observation.sequence if pending else None,
            "pending_observed_at_ns": pending.observation.observed_at_ns if pending else None,
            "pending_deadline_ns": pending.deadline_ns if pending else None,
            "last_sent_at_ns": self._last_sent_at_ns,
            "held_observation_deadline_ns": self._held_observation_deadline_ns,
        }
        self._armed = False
        self._disarm_reason = reason
        self._advance()
        self._events.append(ControlEvent("authority", reason, self._generation, None, self._clock(),
                                         clock_domain=self._slot.clock_domain, guard_snapshot=snapshot))
        self._release(reason)

    def disarm(self, reason: str = "human_takeover") -> None:
        """Cancel old decisions and request AI-owned input release, even while policy hangs."""
        with self._gate:
            if not self._closed:
                self._disarm(reason)

    def invalidate(self, reason: str = "scene_changed") -> int:
        """Scene/episode boundary: revoke all observations and outstanding decisions."""
        with self._gate:
            if self._closed:
                raise RuntimeError("controller is closed")
            self._advance()
            self._events.append(ControlEvent("authority", reason, self._generation, None, self._clock(),
                                             clock_domain=self._slot.clock_domain))
            if not self._release(reason):
                self._armed = False
            return self._generation

    def _reject_reason(self, observation: Observation, deadline: int, now: int) -> str | None:
        if observation.generation != self._generation or not self._armed:
            return "authority_changed"
        if observation.available_at_ns > now:
            return "future_observation"
        if now - observation.observed_at_ns >= self._limits.observation_age_ns:
            return "stale_observation"
        if now >= deadline:
            return "policy_deadline"
        return None

    def _event(self, kind: str, reason: str, result: _Result, **kwargs: Any) -> None:
        obs = result.job.observation
        self._events.append(ControlEvent(kind, reason, obs.generation, obs.sequence, self._clock(),
                                         obs.observed_at_ns, obs.available_at_ns, result.started_at_ns,
                                         result.finished_at_ns, result.job.deadline_ns,
                                         clock_domain=obs.clock_domain, **kwargs))

    def _policy_expired(self, job: _Job, result: _Result | None = None) -> None:
        """Discard unsent work; only retry inside the *unchanged* held-input lease.

        This does not release, synthesize input, extend an input deadline or
        commit policy state. A slow/hung worker remains the sole occupied job.
        Default controllers retain immediate disarm behavior.
        """
        obs = job.observation
        self._events.append(ControlEvent(
            "expired", "policy_deadline", obs.generation, obs.sequence, self._clock(),
            observed_at_ns=obs.observed_at_ns, available_at_ns=obs.available_at_ns,
            policy_started_at_ns=result.started_at_ns if result else None,
            policy_finished_at_ns=result.finished_at_ns if result else None,
            deadline_ns=job.deadline_ns, clock_domain=obs.clock_domain))
        now = self._clock()
        if self._retry_count >= self._limits.policy_deadline_retries:
            if result is not None:
                self._event("rejected", "policy_deadline", result)
            self._disarm("policy_deadline_retry_exhausted" if self._retry_count else "policy_deadline")
            return
        if self._last_sent_at_ns is None or self._held_observation_deadline_ns is None:
            self._disarm("policy_deadline")
            return
        lease = min(self._held_observation_deadline_ns,
                    self._last_sent_at_ns + self._limits.input_watchdog_ns)
        if now >= lease:
            self._disarm("policy_deadline")
            return
        self._retry_count += 1
        self._retry_deadline_ns = min(lease, now + self._limits.policy_budget_ns)
        self._retry_source = obs
        self._expired_job = job if result is None else None
        self._advance()
        self._events.append(ControlEvent(
            "retry", "policy_deadline_fresh_frame", self._generation, obs.sequence, now,
            observed_at_ns=obs.observed_at_ns, available_at_ns=obs.available_at_ns,
            deadline_ns=self._retry_deadline_ns, clock_domain=obs.clock_domain))

    def _dispatch(self, result: _Result) -> None:
        observation = result.job.observation
        if result.job is self._expired_job:
            self._expired_job = None
            self._event("discarded", "policy_deadline", result)
            if self._armed and result.error not in (None, "policy_deadline"):
                self._disarm(result.error)
            return
        reason = self._reject_reason(observation, result.job.deadline_ns, self._clock())
        if reason != "authority_changed" and result.error not in (None, "policy_deadline"):
            self._event("rejected", result.error, result)
            self._disarm(result.error)
            return
        if reason is not None:
            if reason == "policy_deadline":
                self._policy_expired(result.job, result)
                return
            self._event("rejected", reason, result)
            if reason != "authority_changed":
                self._disarm(reason)
            return
        if result.error is not None:
            if result.error == "policy_deadline":
                self._policy_expired(result.job, result)
                return
            self._event("rejected", result.error, result)
            self._disarm(result.error)
            return
        if result.action is None:
            self._event("rejected", "policy_abstained", result)
            if not self._release("policy_abstained"):
                self._armed = False
                self._advance()
            return
        self._event("proposed", "policy_result", result)
        # This final check is adjacent to the sink call, under the authority gate.
        started = self._clock()
        reason = self._reject_reason(observation, result.job.deadline_ns, started)
        if reason is not None:
            if reason == "policy_deadline":
                self._policy_expired(result.job, result)
                return
            self._event("rejected", reason, result)
            self._disarm(reason)
            return
        deadline = min(result.job.deadline_ns, started + self._limits.sink_budget_ns)
        receipt, error = None, None
        try:
            receipt = self._sink.send(result.action, generation=observation.generation,
                                      observation_sequence=observation.sequence, deadline_ns=deadline)
            if not isinstance(receipt, SendReceipt) or not receipt.transmitted:
                error = "input_not_transmitted"
            elif receipt.acknowledged is False:
                error = "input_not_acknowledged"
        except Exception as exc:
            error = f"input_error:{type(exc).__name__}"
        finished = self._clock()
        if finished >= deadline:
            error = "sink_deadline"
        self._event("dispatch", error or "transmitted", result, send_started_at_ns=started,
                    send_finished_at_ns=finished, sink_deadline_ns=deadline, receipt=receipt)
        if error is not None:
            self._disarm(error)
        else:
            if self._retry_count:
                self._event("retry_completed", "fresh_action_transmitted", result)
            self._retry_count = 0
            self._retry_deadline_ns = self._retry_source = None
            self._last_sent_at_ns = finished
            self._held_observation_deadline_ns = observation.observed_at_ns + self._limits.observation_age_ns

    def tick(self) -> None:
        """One bounded scheduling pass, subject to the InputSink contract."""
        with self._gate:
            if self._closed:
                return
            now = self._clock()
            if (self._armed and self._last_sent_at_ns is not None
                    and now - self._last_sent_at_ns >= self._limits.input_watchdog_ns):
                self._disarm("input_watchdog")
            if (self._armed and self._held_observation_deadline_ns is not None
                    and now >= self._held_observation_deadline_ns):
                self._disarm("held_observation_expired")
            if (self._armed and self._retry_deadline_ns is not None
                    and now >= self._retry_deadline_ns):
                self._disarm("policy_retry_timeout")
            result = self._worker.poll()
            if result is not None:
                self._inflight = None
                self._dispatch(result)
            if not self._armed:
                return
            now = self._clock()
            if self._inflight is not None:
                if (self._inflight.observation.generation == self._generation
                        and now >= self._inflight.deadline_ns):
                    self._policy_expired(self._inflight)
                return
            observation = self._slot.latest()
            if observation is None:
                return
            key = observation.generation, observation.sequence
            if key == self._last_submitted:
                return
            deadline = min(observation.observed_at_ns + self._limits.observation_age_ns,
                           now + self._limits.policy_budget_ns)
            if self._retry_deadline_ns is not None:
                deadline = min(deadline, self._retry_deadline_ns)
            reason = self._reject_reason(observation, deadline, now)
            if reason is not None:
                self._disarm(reason)
                return
            job = _Job(observation, deadline)
            if self._worker.submit(job):
                self._inflight = job
                self._last_submitted = key

    def run(self, stop: Event, *, interval_ns: int) -> None:
        """Dedicated actor loop. Missing a cadence skips ticks instead of catching up.

        Human takeover should call disarm directly, without waiting for this loop.
        Setting stop requests disarm at the next loop wake, not an OS emergency stop.
        """
        _natural(interval_ns, "interval_ns", positive=True)
        try:
            while not stop.is_set():
                self.tick()
                if self._closed:
                    return
                stop.wait(interval_ns / 1_000_000_000)
        finally:
            self.disarm("actor_stopped")

    def close(self, *, timeout: float = 0.05) -> bool:
        """Release and stop worker. False means a callback is still running as daemon.

        Timeout bounds the worker join only; InputSink's own deadline contract is
        still required. Late callback results can never transmit after close.
        """
        if type(timeout) not in (int, float) or not 0 <= timeout <= 1:
            raise ValueError("timeout must be between 0 and 1 seconds")
        with self._gate:
            if not self._closed:
                self._disarm("closed")
                self._closed = True
        return self._worker.close(timeout)
