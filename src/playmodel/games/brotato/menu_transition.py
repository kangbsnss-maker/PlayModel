"""Wait for a new rendering after Enter without slowing directional navigation.

This is a duplicate-confirmation gate, not an application verifier. A changed
image merely allows the normal scene/OCR/target checks to run again. The caller
keeps its stop checks and capture loop active; this module never sleeps or sends
input. Timeout never authorizes another Enter on the unchanged image.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


class MenuTransitionTimeout(OSError):
    """No fresh changed rendering arrived within the bounded observation wait."""


@dataclass(frozen=True)
class PendingConfirmation:
    frame_sha256: str
    posted_at_ns: int


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError('a lowercase frame SHA256 is required')


class MenuTransitionGate:
    def __init__(self, *, timeout_ns: int = 6_000_000_000, max_observations: int = 40,
                 clock: Callable[[], int] = time.perf_counter_ns):
        if type(timeout_ns) is not int or timeout_ns <= 0:
            raise ValueError('timeout_ns must be a positive integer')
        if type(max_observations) is not int or max_observations <= 0:
            raise ValueError('max_observations must be a positive integer')
        self.timeout_ns, self.max_observations, self.clock = timeout_ns, max_observations, clock
        self.pending: PendingConfirmation | None = None
        self.observations = 0
        self._timed_out = False
        self.last_reason = 'no_pending_confirmation'

    def record(self, frame_sha256: str, key: str, *, posted_at_ns: int | None = None) -> None:
        """Call immediately AFTER a successful input post, with its source SHA.

        Arrows add no waiting. Recording any second input while Enter remains
        pending is an integration error; the caller must consult allow first.
        """
        _sha(frame_sha256)
        if type(key) is not str or not key:
            raise ValueError('input key must be a nonempty string')
        if self.pending is not None:
            raise RuntimeError('input posted before the pending Enter rendering changed')
        if key != 'enter':
            return
        posted_at_ns = self.clock() if posted_at_ns is None else posted_at_ns
        if type(posted_at_ns) is not int or posted_at_ns < 0:
            raise ValueError('posted_at_ns must be a nonnegative integer')
        self.pending = PendingConfirmation(frame_sha256, posted_at_ns)
        self.observations = 0
        self._timed_out = False
        self.last_reason = 'awaiting_changed_rendering'

    def allow(self, frame_sha256: str, *, captured_at_ns: int | None = None,
              now_ns: int | None = None) -> bool:
        """False asks the caller to capture again, without posting another input.

        Supply capture time to reject cached observations predating Enter. A
        timeout stays latched until a separately verified boundary is reset;
        it does not silently clear the pending confirmation or permit retry.
        """
        _sha(frame_sha256)
        if captured_at_ns is not None and (type(captured_at_ns) is not int or captured_at_ns < 0):
            raise ValueError('captured_at_ns must be a nonnegative integer')
        now = self.clock() if now_ns is None else now_ns
        if type(now) is not int or now < 0:
            raise ValueError('now_ns must be a nonnegative integer')
        if self.pending is None:
            self.last_reason = 'no_pending_confirmation'
            return True
        if now < self.pending.posted_at_ns or (captured_at_ns is not None and captured_at_ns > now):
            raise ValueError('capture/confirmation clock order is invalid')
        if self._timed_out or now - self.pending.posted_at_ns >= self.timeout_ns or self.observations >= self.max_observations:
            self._timed_out = True
            self.last_reason = 'menu_transition_timeout'
            raise MenuTransitionTimeout('Menu Enter transition timed out; repeated confirmation suppressed')
        self.observations += 1
        if captured_at_ns is not None and captured_at_ns <= self.pending.posted_at_ns:
            self.last_reason = 'capture_predates_confirmation'
            return False
        if frame_sha256 == self.pending.frame_sha256:
            self.last_reason = 'unchanged_rendering_after_confirmation'
            return False
        self.pending = None
        self.observations = 0
        self._timed_out = False
        self.last_reason = 'changed_rendering_requires_normal_verification'
        return True

    def reset(self) -> None:
        """Caller-established phase/authority boundary only, never a retry timer."""
        self.pending = None
        self.observations = 0
        self._timed_out = False
        self.last_reason = 'caller_verified_boundary'
