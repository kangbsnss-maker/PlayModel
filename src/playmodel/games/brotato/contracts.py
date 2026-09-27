"""Pure proposal validation for future adapters. This module sends no inputs.

Observation assertions must eventually come from a verified live adapter. Passing
this validator alone proves neither the screen identity nor correct gameplay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping as RuntimeMapping
from enum import Enum
from typing import Mapping


class Screen(str, Enum):
    UNKNOWN = "unknown"
    TITLE = "title"
    CHARACTER = "character"
    WEAPON = "weapon"
    MAP = "map"
    MODE = "mode"
    DIFFICULTY = "difficulty"
    COMBAT = "combat"
    LOOT = "loot"
    LEVEL_UP = "level_up"
    SHOP = "shop"
    RESULT = "result"
    PAUSE = "pause"


PHASE_ACTIONS = {
    Screen.TITLE: frozenset({"start_run"}),
    Screen.CHARACTER: frozenset({"select_character"}),
    Screen.WEAPON: frozenset({"select_weapon"}),
    Screen.MAP: frozenset({"select_map"}),
    Screen.MODE: frozenset({"select_mode"}),
    Screen.DIFFICULTY: frozenset({"select_difficulty"}),
    Screen.COMBAT: frozenset({"move", "aim", "pause"}),
    Screen.LOOT: frozenset({"take_loot", "recycle_loot"}),
    Screen.LEVEL_UP: frozenset({"select_upgrade", "reroll_upgrade"}),
    Screen.SHOP: frozenset({"buy", "lock", "unlock", "reroll_shop", "combine", "recycle_weapon", "next_wave"}),
    Screen.RESULT: frozenset({"retry_run", "return_to_title"}),
    Screen.PAUSE: frozenset({"resume"}),
}
TARGET_ACTIONS = frozenset({
    "select_character", "select_weapon", "select_map", "select_mode", "select_difficulty",
    "take_loot", "recycle_loot", "select_upgrade", "reroll_upgrade", "buy", "lock", "unlock",
    "reroll_shop", "combine", "recycle_weapon",
})
PAID_ACTIONS = frozenset({"buy", "reroll_shop", "reroll_upgrade"})


@dataclass(frozen=True)
class ScreenSnapshot:
    screen: Screen
    sequence: int
    control_epoch: int
    observed_at_ns: int
    available_at_ns: int
    verified: bool
    foreground: bool
    # An observed action mask; game/DLC/character capabilities cannot be guessed.
    legal_actions: frozenset[str]
    targets: Mapping[str, frozenset[str]] = field(default_factory=dict)
    materials: int | None = None
    costs: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    action: str
    observation_sequence: int
    control_epoch: int
    expires_at_ns: int
    target: str | None = None
    movement: tuple[float, float] | None = None


@dataclass(frozen=True)
class Validation:
    allowed: bool
    reason: str


def validate_proposal(snapshot: ScreenSnapshot, proposal: Proposal, now_ns: int, max_age_ns: int) -> Validation:
    """Reject proposals with stale ownership, state, targets or unknown payment."""
    import math

    if proposal.action == "stop":
        return Validation(True, "internal_stop_only")
    timestamps = (now_ns, max_age_ns, snapshot.observed_at_ns, snapshot.available_at_ns, proposal.expires_at_ns,
                  snapshot.sequence, snapshot.control_epoch, proposal.observation_sequence, proposal.control_epoch)
    if any(type(value) is not int or value < 0 for value in timestamps) or max_age_ns == 0:
        return Validation(False, "invalid_clock_or_sequence")
    if (not isinstance(snapshot.screen, Screen) or type(proposal.action) is not str
            or not isinstance(snapshot.legal_actions, frozenset)
            or any(type(action) is not str for action in snapshot.legal_actions)
            or not isinstance(snapshot.targets, RuntimeMapping)
            or not isinstance(snapshot.costs, RuntimeMapping)):
        return Validation(False, "invalid_state_contract")
    if any(type(action) is not str or not isinstance(targets, frozenset)
           or any(type(target) is not str for target in targets)
           for action, targets in snapshot.targets.items()):
        return Validation(False, "invalid_state_contract")
    if proposal.target is not None and type(proposal.target) is not str:
        return Validation(False, "invalid_state_contract")
    if snapshot.verified is not True or snapshot.foreground is not True or snapshot.screen == Screen.UNKNOWN:
        return Validation(False, "unverified_or_unfocused_screen")
    if (not snapshot.observed_at_ns <= snapshot.available_at_ns <= now_ns
            or now_ns - snapshot.observed_at_ns > max_age_ns):
        return Validation(False, "stale_or_future_observation")
    if now_ns >= proposal.expires_at_ns:
        return Validation(False, "proposal_expired")
    if proposal.control_epoch != snapshot.control_epoch:
        return Validation(False, "authority_changed")
    if proposal.observation_sequence != snapshot.sequence:
        return Validation(False, "observation_changed")
    if proposal.action not in PHASE_ACTIONS.get(snapshot.screen, ()) or proposal.action not in snapshot.legal_actions:
        return Validation(False, "action_not_available")
    if proposal.action in TARGET_ACTIONS:
        if proposal.target is None or proposal.target not in snapshot.targets.get(proposal.action, ()):
            return Validation(False, "target_not_available")
    elif proposal.target is not None:
        return Validation(False, "unexpected_target")
    if proposal.action in PAID_ACTIONS:
        cost = snapshot.costs.get(proposal.target)
        if type(cost) is not int or cost < 0 or type(snapshot.materials) is not int or snapshot.materials < cost:
            return Validation(False, "unknown_or_unaffordable_cost")
    if proposal.action in ("move", "aim"):
        vector = proposal.movement
        if not isinstance(vector, tuple) or len(vector) != 2:
            return Validation(False, "invalid_vector")
        if any(type(v) not in (float, int) or not math.isfinite(v) or not -1 <= v <= 1 for v in vector):
            return Validation(False, "invalid_vector")
    elif proposal.movement is not None:
        return Validation(False, "unexpected_vector")
    return Validation(True, "contract_valid_not_sent")
