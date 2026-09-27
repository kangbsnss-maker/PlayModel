"""Verified HUD transitions only; no pixels, OCR, input or reward calculation.

Callers must independently validate numbers/state against preserved evidence.
Current heuristic vision HP fractions do not satisfy this contract. Values are
observations, not claims about enemy causes or game mechanics learned by a model.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VerifiedHUD:
    episode_id: str
    resolution: tuple[int, int]
    observed_at_ns: int
    available_at_ns: int
    phase: str  # combat, death, paused, menu, unknown
    verified: bool
    verifier_id: str
    evidence_ref: str
    evidence_sha256: str
    hp: float | None = None
    max_hp: float | None = None
    xp_progress: float | None = None  # fraction within the displayed level
    level: int | None = None
    currency: int | None = None
    clock_domain: str = "perf_counter_ns_same_host"

    def __post_init__(self):
        for value in (self.episode_id, self.verifier_id, self.evidence_ref):
            if type(value) is not str or not value.strip():
                raise ValueError("episode/verifier/evidence IDs are required")
        if (type(self.evidence_sha256) is not str or len(self.evidence_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.evidence_sha256)):
            raise ValueError("evidence needs a lowercase SHA256")
        if (type(self.resolution) is not tuple or len(self.resolution) != 2
                or any(type(v) is not int or v < 1 for v in self.resolution)):
            raise ValueError("resolution must contain positive integers")
        if (any(type(v) is not int or v < 0 for v in (self.observed_at_ns, self.available_at_ns))
                or self.observed_at_ns > self.available_at_ns):
            raise ValueError("invalid observation/availability timestamps")
        if self.clock_domain != "perf_counter_ns_same_host":
            raise ValueError("unsupported HUD source clock")
        if type(self.verified) is not bool or self.phase not in ("combat", "death", "paused", "menu", "unknown"):
            raise ValueError("invalid verification/state")
        for value in (self.hp, self.max_hp, self.xp_progress):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("HUD numbers must be finite and nonnegative or None")
        if self.max_hp is not None and self.max_hp <= 0:
            raise ValueError("max_hp must be positive")
        if self.hp is not None and self.max_hp is not None and self.hp > self.max_hp:
            raise ValueError("HP above declared maximum is unverified")
        if self.xp_progress is not None and self.xp_progress > 1:
            raise ValueError("xp_progress must be in [0,1]")
        for value in (self.level, self.currency):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("level/currency must be nonnegative integers or None")


@dataclass(frozen=True)
class HUDEvent:
    kind: str
    delta: float | int | None
    before: VerifiedHUD
    after: VerifiedHUD
    cause: str = "unknown"


@dataclass(frozen=True)
class HUDTransition:
    events: tuple[HUDEvent, ...]
    status: str


class HUDEventTracker:
    """No automatic rewards. max_gap_ns is an explicit caller calibration.

    A delayed older OCR result clears the baseline instead of bridging through
    the uncertainty. A new valid episode/size starts a fresh baseline. Death
    requires its own verified phase observation, never HP=0 or an absent HUD.
    """

    def __init__(self, *, max_gap_ns: int):
        if type(max_gap_ns) is not int or max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be a positive integer")
        self.max_gap_ns = max_gap_ns
        self._previous: VerifiedHUD | None = None

    def reset(self):
        self._previous = None

    def observe(self, current: VerifiedHUD) -> HUDTransition:
        if not isinstance(current, VerifiedHUD):
            raise ValueError("expected VerifiedHUD")
        previous = self._previous
        self._previous = None
        if not current.verified or current.phase not in ("combat", "death"):
            return HUDTransition((), "baseline_broken")
        if previous is None:
            self._previous = current if current.phase == "combat" else None
            return HUDTransition((), "baseline_started" if current.phase == "combat" else "unpaired_terminal")
        if (current.episode_id != previous.episode_id or current.resolution != previous.resolution
                or current.verifier_id != previous.verifier_id):
            self._previous = current if current.phase == "combat" else None
            return HUDTransition((), "context_changed")
        if (current.observed_at_ns <= previous.observed_at_ns
                or current.available_at_ns < previous.available_at_ns):
            return HUDTransition((), "out_of_order")
        if current.observed_at_ns - previous.observed_at_ns > self.max_gap_ns:
            self._previous = current if current.phase == "combat" else None
            return HUDTransition((), "gap_exceeded")
        if current.phase == "death":
            return HUDTransition((HUDEvent("death", None, previous, current),), "verified_terminal")
        if current.level is not None and previous.level is not None and current.level < previous.level:
            return HUDTransition((), "possible_reset")
        self._previous = current
        events = []

        def add(kind, delta):
            events.append(HUDEvent(kind, delta, previous, current))

        if (previous.max_hp is not None and previous.max_hp == current.max_hp
                and previous.hp is not None and current.hp is not None):
            delta = current.hp - previous.hp
            if delta:
                add("hp_loss" if delta < 0 else "heal", delta)
        if previous.level is not None and current.level is not None:
            if current.level > previous.level:
                add("level_up", current.level - previous.level)
            elif (previous.xp_progress is not None and current.xp_progress is not None
                  and current.xp_progress > previous.xp_progress):
                add("xp_progress", current.xp_progress - previous.xp_progress)
        if previous.currency is not None and current.currency is not None:
            delta = current.currency - previous.currency
            if delta:
                add("currency_delta", delta)
        return HUDTransition(tuple(events), "observed")
