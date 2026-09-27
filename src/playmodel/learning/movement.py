"""Small local REINFORCE pilot; no capture, OS input, OCR, or network dependency.

One immutable linear-softmax policy per rollout. Terminal return is +1 for an
independently verified wave clear, -1 for verified death; gamma=1, baseline=0.
Self-generated actions are policy-gradient samples, never BC target labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any


ACTION_NAMES = ("neutral", "up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left")
ALL_ACTIONS_MASK = (True,) * len(ACTION_NAMES)
NEUTRAL_ONLY_MASK = (True,) + (False,) * (len(ACTION_NAMES) - 1)
# Screen coordinates: positive y is down. The game/input adapter owns diagonal speed handling.
MOVEMENTS = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
GRID_WIDTH, GRID_HEIGHT = 8, 6
RGB_FEATURES = GRID_WIDTH * GRID_HEIGHT * 3
NAVIGATION_OFFSET = RGB_FEATURES + len(ACTION_NAMES)
FEATURE_COUNT = NAVIGATION_OFFSET + 3 + 1
CLOCK_DOMAIN = "perf_counter_ns_same_host"
SCHEMA = "playmodel.linear-movement.v3"
MASK_CONTRACT = "predecision_legal_actions_v1"


def _integer(value: Any, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _action(value: int) -> None:
    _integer(value, "action_index")
    if value >= len(ACTION_NAMES):
        raise ValueError("action_index outside movement space")


def _mask(mask: tuple[bool, ...] | None) -> tuple[bool, ...]:
    if mask is None:
        return ALL_ACTIONS_MASK
    if (type(mask) is not tuple or len(mask) != len(ACTION_NAMES)
            or any(type(value) is not bool for value in mask) or not any(mask)):
        raise ValueError("action mask requires nine booleans and at least one allowed action")
    return mask


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _digest(value: str, name: str) -> None:
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _features(features: tuple[float, ...]) -> int:
    if not isinstance(features, tuple) or len(features) != FEATURE_COUNT:
        raise ValueError(f"features must be an immutable {FEATURE_COUNT}-tuple")
    for value in features:
        _finite(value, "feature")
        if not -1 <= value <= 1:
            raise ValueError("feature outside [-1, 1]")
    previous = features[RGB_FEATURES:NAVIGATION_OFFSET]
    if any(value not in (0, 1) for value in previous) or sum(previous) != 1 or features[-1] != 1:
        raise ValueError("features require previous-action one-hot and bias=1")
    dx, dy, valid = features[NAVIGATION_OFFSET:NAVIGATION_OFFSET + 3]
    if valid not in (0, 1) or (valid == 0 and (dx != 0 or dy != 0)):
        raise ValueError("navigation requires binary validity and zero vector when invalid")
    return previous.index(1)


def extract_features(bgra: bytes, width: int, height: int, previous_action: int, *,
                     navigation: tuple[float, float, bool | int] | None = None) -> tuple[float, ...]:
    """Sample RGB at the centers of an 8x6 grid; packed top-down BGRA input.

    The live capture proposal is 320x180. Any dimensions >=8x6 are accepted for
    diagnostics; all episode frames must use one declared capture configuration.
    RGB is mapped to [-1,1], then actual previous action one-hot, navigation,
    and bias. Navigation is (goal_dx, goal_dy, valid), already normalized to
    [-1,1], with positive y down. Its source must be this frame or an explicitly
    logged, already available observation, never a future detector/OCR result.
    None or valid=False produces (0,0,0), permitting untargeted exploration.
    """
    _integer(width, "width", minimum=GRID_WIDTH)
    _integer(height, "height", minimum=GRID_HEIGHT)
    _action(previous_action)
    if not isinstance(bgra, bytes) or len(bgra) != width * height * 4:
        raise ValueError("expected immutable packed width*height*4 BGRA bytes")
    dx, dy, valid = 0.0, 0.0, False
    if navigation is not None:
        if not isinstance(navigation, tuple) or len(navigation) != 3:
            raise ValueError("navigation must be (normalized_dx, normalized_dy, valid)")
        dx, dy, valid = navigation
        _finite(dx, "navigation dx")
        _finite(dy, "navigation dy")
        if not -1 <= dx <= 1 or not -1 <= dy <= 1:
            raise ValueError("navigation dx/dy must be in [-1,1]")
        if type(valid) not in (bool, int) or valid not in (False, True):
            raise ValueError("navigation validity must be boolean or integer 0/1")
        if not valid:
            dx, dy = 0.0, 0.0
    features = []
    for grid_y in range(GRID_HEIGHT):
        y = ((2 * grid_y + 1) * height) // (2 * GRID_HEIGHT)
        for grid_x in range(GRID_WIDTH):
            x = ((2 * grid_x + 1) * width) // (2 * GRID_WIDTH)
            offset = (y * width + x) * 4
            features.extend((bgra[offset + channel] / 127.5 - 1 for channel in (2, 1, 0)))
    features.extend(1.0 if index == previous_action else 0.0 for index in range(len(ACTION_NAMES)))
    features.extend((float(dx), float(dy), float(valid)))
    features.append(1.0)
    return tuple(features)


@dataclass(frozen=True)
class MovementDecision:
    action_index: int
    movement: tuple[int, int]
    probabilities: tuple[float, ...]
    log_probability: float
    policy_version: str
    features: tuple[float, ...]
    allowed_actions: tuple[bool, ...] = ALL_ACTIONS_MASK


@dataclass(frozen=True)
class LinearMovementPolicy:
    weights: tuple[tuple[float, ...], ...]
    updates: int = 0
    parent_version: str | None = None
    baseline_origin: str = "custom"
    navigation_prior_strength: float = 0.0
    version: str = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.updates, "updates")
        if self.parent_version is not None:
            _digest(self.parent_version, "parent_version")
        if self.baseline_origin not in ("random", "geometry_prior", "custom"):
            raise ValueError("unsupported baseline origin")
        _finite(self.navigation_prior_strength, "navigation_prior_strength")
        if not 0 <= self.navigation_prior_strength <= 8:
            raise ValueError("navigation_prior_strength must be in [0,8]")
        if (self.baseline_origin == "geometry_prior") != (self.navigation_prior_strength > 0):
            raise ValueError("geometry prior requires an explicit positive initial strength")
        if not isinstance(self.weights, tuple) or len(self.weights) != len(ACTION_NAMES):
            raise ValueError("weights require one immutable row per action")
        for row in self.weights:
            if not isinstance(row, tuple) or len(row) != FEATURE_COUNT:
                raise ValueError("invalid immutable weight row")
            for value in row:
                _finite(value, "weight")
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "version", hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    @classmethod
    def random(cls, *, seed: int = 0, scale: float = 0.01) -> LinearMovementPolicy:
        _integer(seed, "seed")
        _finite(scale, "scale")
        if not 0 < scale <= 0.1:
            raise ValueError("initial weight scale must be in (0, 0.1]")
        rng = random.Random(seed)
        return cls(tuple(tuple(rng.uniform(-scale, scale) for _ in range(FEATURE_COUNT))
                         for _ in ACTION_NAMES), baseline_origin="random")

    @classmethod
    def initialized_for_collection(cls, *, seed: int = 0, scale: float = 0.01,
                                   navigation_strength: float = 6.0) -> LinearMovementPolicy:
        """Handcoded geometric parameter prior, not a claim of learned collection.

        Initial navigation logit = strength * unit_movement dot goal_direction.
        Diagonal vectors are normalized so they do not win merely for being longer.
        No target gives zero navigation contribution and leaves seeded exploration.
        Subsequent REINFORCE updates train these same weights with no action override.
        """
        _finite(navigation_strength, "navigation_strength")
        if not 0 < navigation_strength <= 8:
            raise ValueError("navigation_strength must be in (0,8]")
        seed_policy = cls.random(seed=seed, scale=scale)
        weights = [list(row) for row in seed_policy.weights]
        for row, (dx, dy) in zip(weights, MOVEMENTS):
            norm = math.hypot(dx, dy)
            row[NAVIGATION_OFFSET] = navigation_strength * dx / norm if norm else 0.0
            row[NAVIGATION_OFFSET + 1] = navigation_strength * dy / norm if norm else 0.0
            row[NAVIGATION_OFFSET + 2] = 0.0
        return cls(tuple(tuple(row) for row in weights), baseline_origin="geometry_prior",
                   navigation_prior_strength=float(navigation_strength))

    def _payload(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "grid": [GRID_WIDTH, GRID_HEIGHT], "feature_count": FEATURE_COUNT,
                "actions": list(ACTION_NAMES), "weights": self.weights, "updates": self.updates,
                "parent_version": self.parent_version, "baseline_origin": self.baseline_origin,
                "navigation_prior_strength": self.navigation_prior_strength,
                "navigation_features": ["goal_dx", "goal_dy", "valid"],
                "action_mask_contract": MASK_CONTRACT}

    def probabilities(self, features: tuple[float, ...], *,
                      mask: tuple[bool, ...] | None = None) -> tuple[float, ...]:
        _features(features)
        allowed = _mask(mask)
        try:
            logits = tuple(math.fsum(weight * value for weight, value in zip(row, features))
                           if permitted else -math.inf for row, permitted in zip(self.weights, allowed))
        except (OverflowError, ValueError) as exc:
            raise ValueError("nonfinite policy logits") from exc
        if not all(math.isfinite(value) for value, permitted in zip(logits, allowed) if permitted):
            raise ValueError("nonfinite policy logits")
        maximum = max(logits)
        exponentials = tuple(math.exp(value - maximum) for value in logits)
        total = math.fsum(exponentials)
        return tuple(value / total for value in exponentials)

    def sample(self, features: tuple[float, ...], *, rng: random.Random,
               mask: tuple[bool, ...] | None = None) -> MovementDecision:
        allowed = _mask(mask)
        probabilities = self.probabilities(features, mask=allowed)
        threshold, cumulative = rng.random(), 0.0
        chosen = max(index for index, probability in enumerate(probabilities) if probability > 0)
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                chosen = index
                break
        return MovementDecision(chosen, MOVEMENTS[chosen], probabilities,
                                math.log(probabilities[chosen]), self.version, features, allowed)


@dataclass(frozen=True)
class StateEvidence:
    kind: str  # combat, wave_clear, death
    frame_ref: str
    frame_sha256: str
    observed_at_ns: int
    available_at_ns: int
    verified_at_ns: int
    origin: str  # developer_verified or local_detector
    verifier_id: str
    verified: bool
    independent_of_policy: bool
    clock_domain: str = CLOCK_DOMAIN


@dataclass(frozen=True)
class MovementStep:
    sequence: int
    generation: int
    frame_ref: str
    frame_sha256: str
    observed_at_ns: int
    available_at_ns: int
    decided_at_ns: int
    sent_at_ns: int
    decision: MovementDecision
    actual_action_index: int
    transmitted: bool
    transmission_ref: str
    acknowledged: bool | None = None
    action_origin: str = "policy"
    clock_domain: str = CLOCK_DOMAIN


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    policy_version: str
    manifest_ref: str
    manifest_sha256: str
    capture_config_ref: str
    control_generation: int
    initial_action_index: int
    combat_entry: StateEvidence | None
    steps: tuple[MovementStep, ...]
    terminal: StateEvidence | None
    truncated: bool = False
    human_intervention: bool = False
    guard_failure: bool = False


def _evidence(evidence: StateEvidence | None, allowed: tuple[str, ...], policy_version: str) -> None:
    if not isinstance(evidence, StateEvidence) or evidence.kind not in allowed:
        raise ValueError(f"missing verified state evidence: {allowed}")
    if evidence.verified is not True or evidence.independent_of_policy is not True:
        raise ValueError("state evidence must be verified independently of policy")
    if evidence.origin not in ("developer_verified", "local_detector"):
        raise ValueError("unsupported verifier origin")
    _text(evidence.verifier_id, "verifier_id")
    if evidence.verifier_id == policy_version:
        raise ValueError("policy cannot verify its own outcome")
    _text(evidence.frame_ref, "evidence frame_ref")
    _digest(evidence.frame_sha256, "evidence frame_sha256")
    for name in ("observed_at_ns", "available_at_ns", "verified_at_ns"):
        _integer(getattr(evidence, name), name)
    if not evidence.observed_at_ns <= evidence.available_at_ns <= evidence.verified_at_ns:
        raise ValueError("invalid evidence time order")
    if evidence.clock_domain != CLOCK_DOMAIN:
        raise ValueError("evidence clock domain mismatch")


def validate_episode(policy: LinearMovementPolicy, episode: EpisodeRecord) -> float:
    """Validate on-policy terminal learning eligibility; return the scalar return.

    This checks the supplied ledger, not pixels or the truth of external labels.
    Capture/terminal/transmission evidence must be preserved and independently
    reviewed by the runtime/integrator. No failed/truncated episode is relabelled.
    """
    if not isinstance(episode, EpisodeRecord):
        raise ValueError("expected EpisodeRecord")
    for name in ("episode_id", "manifest_ref", "capture_config_ref"):
        _text(getattr(episode, name), name)
    _digest(episode.manifest_sha256, "manifest_sha256")
    _integer(episode.control_generation, "control_generation")
    _action(episode.initial_action_index)
    if episode.policy_version != policy.version:
        raise ValueError("episode policy version mismatch; off-policy replay is not supported")
    for name in ("truncated", "human_intervention", "guard_failure"):
        if getattr(episode, name) is not False:
            raise ValueError(f"episode ineligible: {name}")
    _evidence(episode.combat_entry, ("combat",), policy.version)
    _evidence(episode.terminal, ("wave_clear", "death"), policy.version)
    if not isinstance(episode.steps, tuple) or not episode.steps:
        raise ValueError("episode needs immutable nonempty transmitted steps")
    assert episode.combat_entry is not None and episode.terminal is not None
    if episode.terminal.observed_at_ns <= episode.combat_entry.observed_at_ns:
        raise ValueError("terminal must follow combat entry")
    previous_action = episode.initial_action_index
    previous_sequence, previous_observed, previous_sent = -1, episode.combat_entry.observed_at_ns, -1
    seen_transmissions: set[str] = set()
    for step in episode.steps:
        if not isinstance(step, MovementStep) or not isinstance(step.decision, MovementDecision):
            raise ValueError("invalid movement step")
        for name in ("sequence", "generation", "observed_at_ns", "available_at_ns", "decided_at_ns", "sent_at_ns"):
            _integer(getattr(step, name), name)
        if step.sequence <= previous_sequence or step.generation != episode.control_generation:
            raise ValueError("repeated sequence or changed authority generation")
        if not (previous_observed <= step.observed_at_ns <= step.available_at_ns
                <= step.decided_at_ns <= step.sent_at_ns < episode.terminal.observed_at_ns):
            raise ValueError("invalid frame/availability/decision/transmission/terminal order")
        if previous_sent > step.decided_at_ns:
            raise ValueError("next decision predates previous actual transmission")
        _text(step.frame_ref, "step frame_ref")
        _digest(step.frame_sha256, "step frame_sha256")
        _text(step.transmission_ref, "transmission_ref")
        if step.transmission_ref in seen_transmissions:
            raise ValueError("duplicate transmission evidence")
        if step.clock_domain != CLOCK_DOMAIN:
            raise ValueError("step clock domain mismatch")
        if step.transmitted is not True or step.action_origin != "policy":
            raise ValueError("only actually transmitted policy actions are eligible")
        if step.acknowledged is not None and step.acknowledged is not True:
            raise ValueError("negative or malformed transmission acknowledgement")
        decision = step.decision
        _action(decision.action_index)
        _action(step.actual_action_index)
        if decision.policy_version != policy.version:
            raise ValueError("step policy version mismatch")
        if (step.actual_action_index != decision.action_index
                or type(decision.movement) is not tuple
                or any(type(value) is not int for value in decision.movement)
                or decision.movement != MOVEMENTS[decision.action_index]):
            raise ValueError("sampled and actual transmitted action mismatch")
        if _features(decision.features) != previous_action:
            raise ValueError("previous-action feature does not match actual transmission history")
        if decision.allowed_actions is None:
            raise ValueError("decision must record its action mask")
        allowed = _mask(decision.allowed_actions)
        if not allowed[decision.action_index]:
            raise ValueError("selected action is forbidden by its recorded mask")
        expected = policy.probabilities(decision.features, mask=allowed)
        if not isinstance(decision.probabilities, tuple) or len(decision.probabilities) != len(ACTION_NAMES):
            raise ValueError("missing immutable sampling probabilities")
        for actual, probability in zip(decision.probabilities, expected):
            _finite(actual, "sampling probability")
            if not 0 <= actual <= 1 or not math.isclose(actual, probability, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("sampling distribution does not match frozen policy")
        if not math.isclose(math.fsum(decision.probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("sampling probabilities must sum to one")
        selected = expected[decision.action_index]
        _finite(decision.log_probability, "selected log probability")
        if selected <= 0 or not math.isclose(decision.log_probability, math.log(selected), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("selected log probability mismatch")
        previous_action = step.actual_action_index
        previous_sequence, previous_observed, previous_sent = step.sequence, step.observed_at_ns, step.sent_at_ns
        seen_transmissions.add(step.transmission_ref)
    return 1.0 if episode.terminal.kind == "wave_clear" else -1.0


@dataclass(frozen=True)
class UpdateResult:
    policy: LinearMovementPolicy
    report: dict[str, Any]


def reinforce_update(policy: LinearMovementPolicy, episode: EpisodeRecord, *, learning_rate: float = 0.01,
                     gradient_clip: float = 1.0) -> UpdateResult:
    """One terminal-return update on a new immutable policy; old policy stays fixed.

    gradient = R * sum_t (one_hot(action_t) - probabilities_t) outer features_t.
    No per-episode reward centering: constant terminal R must not become zero.
    Global L2 clipping is applied before gradient ascent. No BC loss is involved.
    """
    _finite(learning_rate, "learning_rate")
    _finite(gradient_clip, "gradient_clip")
    if not 0 < learning_rate <= 1 or not 0 < gradient_clip <= 1_000:
        raise ValueError("learning_rate must be in (0,1], gradient_clip in (0,1000]")
    reward = validate_episode(policy, episode)
    gradient = [[0.0] * FEATURE_COUNT for _ in ACTION_NAMES]
    for step in episode.steps:
        decision = step.decision
        for index, probability in enumerate(decision.probabilities):
            multiplier = reward * ((1.0 if index == decision.action_index else 0.0) - probability)
            row = gradient[index]
            for feature_index, value in enumerate(decision.features):
                row[feature_index] += multiplier * value
    flattened = tuple(value for row in gradient for value in row)
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError("nonfinite policy gradient")
    norm = math.hypot(*flattened)
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    scale = min(1.0, gradient_clip / norm) if norm else 1.0
    weights = tuple(tuple(old + learning_rate * scale * delta for old, delta in zip(row, gradient_row))
                    for row, gradient_row in zip(policy.weights, gradient))
    candidate = LinearMovementPolicy(weights, policy.updates + 1, policy.version,
                                     policy.baseline_origin, policy.navigation_prior_strength)
    terminal = episode.terminal
    assert terminal is not None
    return UpdateResult(candidate, {
        "algorithm": "terminal_reinforce", "gamma": 1.0, "baseline": 0.0,
        "episode_id": episode.episode_id, "manifest_ref": episode.manifest_ref,
        "manifest_sha256": episode.manifest_sha256, "source_policy_version": policy.version,
        "candidate_policy_version": candidate.version, "steps": len(episode.steps), "return": reward,
        "baseline_origin": policy.baseline_origin, "navigation_prior_strength": policy.navigation_prior_strength,
        "action_mask_contract": MASK_CONTRACT,
        "neutral_only_steps": sum(step.decision.allowed_actions == NEUTRAL_ONLY_MASK for step in episode.steps),
        "terminal_kind": terminal.kind, "terminal_origin": terminal.origin,
        "terminal_verifier_id": terminal.verifier_id, "learning_rate": learning_rate,
        "gradient_norm_before_clip": norm, "gradient_norm_after_clip": norm * scale,
        "gradient_scale": scale, "weights_changed": weights != policy.weights,
        "promotion_approved": False, "performance_improvement_verified": False,
    })


def save_checkpoint(policy: LinearMovementPolicy, path: str | Path) -> Path:
    """Write a new checkpoint atomically without replacing any existing artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {**policy._payload(), "policy_version": policy.version}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".movement-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes a complete file and fails if target exists.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return target


def load_checkpoint(path: str | Path) -> LinearMovementPolicy:
    target = Path(path)
    if target.stat().st_size > 1_000_000:
        raise ValueError("checkpoint exceeds small-pilot size limit")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if (payload["schema"] != SCHEMA or payload["grid"] != [GRID_WIDTH, GRID_HEIGHT]
                or payload["feature_count"] != FEATURE_COUNT or payload["actions"] != list(ACTION_NAMES)
                or payload["navigation_features"] != ["goal_dx", "goal_dy", "valid"]
                or payload["action_mask_contract"] != MASK_CONTRACT):
            raise ValueError("incompatible movement checkpoint schema")
        policy = LinearMovementPolicy(tuple(tuple(row) for row in payload["weights"]),
                                      payload["updates"], payload["parent_version"],
                                      payload["baseline_origin"], payload["navigation_prior_strength"])
        if payload["policy_version"] != policy.version:
            raise ValueError("checkpoint policy digest mismatch")
        return policy
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed movement checkpoint") from exc
