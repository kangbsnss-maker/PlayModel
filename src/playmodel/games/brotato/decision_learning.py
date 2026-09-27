"""Local high-level choice RL baseline, independent of capture and input sending.

This is an experimental comparison baseline, not a final game-understanding model.
Candidate semantics are observations, not correct-action labels. A shared hashed
linear actor includes state/candidate interactions; a separate value baseline is
trained from full-run returns. Verified selection acceptance does not establish
the selected item's numeric effects. No tree rewards or inferred win labels exist.
All timestamps use perf_counter_ns on the same host. The caller owns verification
of image contents: hashes/timing here validate provenance, not OCR correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

SCHEMA = "playmodel.choice-rl.v1"
ACTOR_SIZE = 512
VALUE_SIZE = 128
CLOCK_DOMAIN = "perf_counter_ns_same_host"
REWARD_RULE = "verified_waves_n_over_n_plus20_plus_terminal_v1"
Features = tuple[tuple[str, float], ...]


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{name} requires nonempty text <=2048 characters")


def _number(value: Any, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} requires a finite number")
    return float(value)


def _integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} requires a nonnegative integer")


def _digest(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")


def _features(values: Mapping[str, float] | Sequence[Sequence[Any]]) -> Features:
    entries = list(values.items()) if isinstance(values, Mapping) else list(values)
    result: dict[str, float] = {}
    if len(entries) > 64:
        raise ValueError("at most 64 named features are supported")
    for key, value in entries:
        _text(key, "feature name")
        value = _number(value, "feature")
        if key in result or not -1 <= value <= 1:
            raise ValueError("features require unique names and values in [-1,1]")
        result[key] = value
    return tuple(sorted(result.items()))


def _hashed(entries: Sequence[tuple[str, float]], size: int) -> tuple[float, ...]:
    vector = [0.0] * size
    for key, value in entries:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % size
        vector[bucket] += value * (1 if digest[4] & 1 else -1)
    # Fixed scale, independent of other offered candidates: permutation invariant.
    return tuple(value / 8 for value in vector)


@dataclass(frozen=True)
class ChoiceCandidate:
    candidate_id: str
    action: str
    features: Features | Mapping[str, float]
    legal: bool = True
    semantic_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _text(self.action, "action")
        if type(self.legal) is not bool:
            raise ValueError("legal must be boolean")
        if self.semantic_id is not None:
            _text(self.semantic_id, "semantic_id")
        object.__setattr__(self, "features", _features(self.features))


def _actor_features(state: Features, candidate: ChoiceCandidate) -> tuple[float, ...]:
    # candidate_id and UI position are deliberately absent. Action is the stable
    # operation (buy/upgrade/skip), never a per-slot command such as buy:3.
    base = [("action=" + candidate.action, 1.0)]
    if candidate.semantic_id is not None:
        base.append(("identity=" + candidate.semantic_id, 1.0))
    base.extend(("candidate=" + key, value) for key, value in candidate.features)
    entries = list(base)
    entries.extend(("state=" + skey + "|" + ckey, sval * cval)
                   for skey, sval in state for ckey, cval in base)
    return _hashed(entries, ACTOR_SIZE)


def _value_features(state: Features) -> tuple[float, ...]:
    return _hashed([("bias", 1.0), *(("state=" + k, v) for k, v in state)], VALUE_SIZE)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class ChoiceDecision:
    state: Features
    candidates: tuple[ChoiceCandidate, ...]
    selected_id: str
    action: str
    probabilities: tuple[float, ...]
    log_probability: float
    value_estimate: float
    policy_version: str


@dataclass(frozen=True)
class ChoicePolicy:
    actor_weights: tuple[float, ...]
    value_weights: tuple[float, ...]
    updates: int = 0
    parent_version: str | None = None
    trained_run_ids: tuple[str, ...] = ()
    training_manifest_sha256: str | None = None
    version: str = field(init=False)

    def __post_init__(self) -> None:
        for weights, size in ((self.actor_weights, ACTOR_SIZE), (self.value_weights, VALUE_SIZE)):
            if not isinstance(weights, tuple) or len(weights) != size:
                raise ValueError("invalid immutable weight dimensions")
            for weight in weights:
                if abs(_number(weight, "weight")) > 100:
                    raise ValueError("weight outside supported bounds")
        _integer(self.updates, "updates")
        if self.parent_version is not None:
            _digest(self.parent_version)
        if self.training_manifest_sha256 is not None:
            _digest(self.training_manifest_sha256)
        if not isinstance(self.trained_run_ids, tuple) or len(set(self.trained_run_ids)) != len(self.trained_run_ids):
            raise ValueError("trained run IDs must be unique immutable values")
        for run_id in self.trained_run_ids:
            _text(run_id, "run_id")
        object.__setattr__(self, "version", _hash(_json(self._payload())))

    def _payload(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "actor_weights": self.actor_weights,
                "value_weights": self.value_weights, "updates": self.updates,
                "parent_version": self.parent_version, "trained_run_ids": self.trained_run_ids,
                "training_manifest_sha256": self.training_manifest_sha256,
                "reward_rule": REWARD_RULE, "deployment_status": "unapproved_candidate"}

    @classmethod
    def initial(cls, seed: int = 0) -> ChoicePolicy:
        rng = random.Random(seed)
        return cls(tuple(rng.uniform(-0.01, 0.01) for _ in range(ACTOR_SIZE)), (0.0,) * VALUE_SIZE)

    def distribution(self, state: Mapping[str, float] | Features,
                     candidates: Sequence[ChoiceCandidate]) -> tuple[float, ...]:
        state = _features(state)
        candidates = tuple(candidates)
        if not candidates or len(candidates) > 64 or len({c.candidate_id for c in candidates}) != len(candidates):
            raise ValueError("require 1..64 candidates with unique IDs")
        if not any(c.legal for c in candidates):
            raise ValueError("no legal choices; caller must abstain")
        logits = [_dot(self.actor_weights, _actor_features(state, c)) if c.legal else -math.inf
                  for c in candidates]
        maximum = max(logits)
        masses = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in logits]
        total = math.fsum(masses)
        return tuple(mass / total for mass in masses)

    def sample(self, state: Mapping[str, float] | Features,
               candidates: Sequence[ChoiceCandidate], rng: random.Random) -> ChoiceDecision:
        state = _features(state)
        candidates = tuple(candidates)
        probabilities = self.distribution(state, candidates)
        # Sampling also does not depend on the input list's UI ordering.
        ordered = sorted(range(len(candidates)), key=lambda i: candidates[i].candidate_id)
        draw = rng.random()
        selected = next(i for i in reversed(ordered) if probabilities[i] > 0)
        cumulative = 0.0
        for index in ordered:
            cumulative += probabilities[index]
            if probabilities[index] > 0 and draw < cumulative:
                selected = index
                break
        candidate = candidates[selected]
        return ChoiceDecision(state, candidates, candidate.candidate_id, candidate.action,
                              probabilities, math.log(probabilities[selected]),
                              _dot(self.value_weights, _value_features(state)), self.version)

    def save(self, path: Path | str) -> None:
        _write_new(Path(path), {**self._payload(), "version": self.version})

    @classmethod
    def load(cls, path: Path | str) -> ChoicePolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload.get("schema") != SCHEMA or payload.get("reward_rule") != REWARD_RULE
                or payload.get("deployment_status") != "unapproved_candidate"):
            raise ValueError("unsupported choice checkpoint")
        policy = cls(tuple(payload["actor_weights"]), tuple(payload["value_weights"]),
                     payload["updates"], payload["parent_version"], tuple(payload["trained_run_ids"]),
                     payload["training_manifest_sha256"])
        if payload.get("version") != policy.version:
            raise ValueError("checkpoint version/hash mismatch")
        return policy


@dataclass(frozen=True)
class Evidence:
    frame_ref: str
    frame_sha256: str
    observed_at_ns: int
    available_at_ns: int
    verified_at_ns: int
    verifier_id: str
    verified: bool = False
    independent_of_policy: bool = False
    origin: str = "local_verifier"


@dataclass(frozen=True)
class AppliedEvidence:
    kind: str
    before: Evidence
    after: Evidence
    accepted: bool
    effect_verified: bool = False
    currency_before: int | None = None
    currency_after: int | None = None
    cost: int | None = None
    ownership_changed: bool = False


@dataclass(frozen=True)
class ChoiceStep:
    decision_id: str
    decision: ChoiceDecision
    observation: Evidence
    decided_at_ns: int
    sent_at_ns: int
    actual_candidate_id: str
    transmission_ref: str
    applied: AppliedEvidence | None = None
    action_origin: str = "policy"
    acknowledged: bool = False


@dataclass(frozen=True)
class RunOutcome:
    kind: str  # death, objective_reached, or truncated
    evidence: Evidence | None
    wave_clears: tuple[tuple[int, Evidence], ...] = ()
    objective_id: str | None = None


@dataclass(frozen=True)
class DecisionRun:
    run_id: str
    split: str
    movement_policy_version: str
    choice_policy_version: str
    steps: tuple[ChoiceStep, ...]
    outcome: RunOutcome
    session_ids: tuple[str, ...]
    clock_domain: str = CLOCK_DOMAIN
    human_intervention: bool = False
    guard_failure: bool = False


def _check_evidence(evidence: Evidence, *, verify_files: bool = False) -> None:
    _text(evidence.frame_ref, "frame_ref")
    _digest(evidence.frame_sha256)
    _text(evidence.verifier_id, "verifier_id")
    for value in (evidence.observed_at_ns, evidence.available_at_ns, evidence.verified_at_ns):
        _integer(value, "evidence timestamp")
    if not evidence.observed_at_ns <= evidence.available_at_ns <= evidence.verified_at_ns:
        raise ValueError("invalid evidence timing")
    if evidence.verified is not True or evidence.independent_of_policy is not True:
        raise ValueError("independent verified evidence required")
    if evidence.origin not in {"local_verifier", "developer_review", "human_review"}:
        raise ValueError("raw OCR or policy output is not verified evidence")
    if verify_files:
        path = Path(evidence.frame_ref)
        if not path.is_absolute() or not path.is_file() or _hash(path.read_bytes()) != evidence.frame_sha256:
            raise ValueError("evidence source missing or modified; absolute file references required")


def _run_evidence(run: DecisionRun) -> list[Evidence]:
    result = []
    for step in run.steps:
        result.append(step.observation)
        if step.applied is not None:
            result.extend((step.applied.before, step.applied.after))
    if run.outcome.evidence is not None:
        result.append(run.outcome.evidence)
    result.extend(e for _, e in run.outcome.wave_clears)
    return result


def _validate_decision(policy: ChoicePolicy, decision: ChoiceDecision) -> int:
    if decision.policy_version != policy.version:
        raise ValueError("off-policy or mid-run policy replacement")
    expected = policy.distribution(decision.state, decision.candidates)
    if len(expected) != len(decision.probabilities) or any(
            not math.isclose(_number(p, "probability"), e, rel_tol=1e-9, abs_tol=1e-12)
            for p, e in zip(decision.probabilities, expected)):
        raise ValueError("logged behavior probabilities differ from source policy")
    indices = [i for i, c in enumerate(decision.candidates) if c.candidate_id == decision.selected_id]
    if len(indices) != 1:
        raise ValueError("selected candidate missing")
    index = indices[0]
    if expected[index] <= 0 or decision.action != decision.candidates[index].action:
        raise ValueError("illegal or mismatched selected action")
    if not math.isclose(_number(decision.log_probability, "log_probability"), math.log(expected[index]), abs_tol=1e-9):
        raise ValueError("behavior log probability mismatch")
    if not math.isclose(_number(decision.value_estimate, "value_estimate"),
                        _dot(policy.value_weights, _value_features(decision.state)), abs_tol=1e-9):
        raise ValueError("value baseline mismatch")
    return index


def _return(outcome: RunOutcome) -> float:
    if outcome.kind not in {"death", "objective_reached"} or outcome.evidence is None:
        raise ValueError("verified full-run terminal required; truncated runs cannot train")
    _check_evidence(outcome.evidence)
    if outcome.kind == "objective_reached":
        _text(outcome.objective_id, "independently verified objective_id")
    waves: set[int] = set()
    frames: set[str] = set()
    previous_wave = 0
    previous_time = 0
    for wave, evidence in outcome.wave_clears:
        _integer(wave, "wave")
        _check_evidence(evidence)
        if wave <= previous_wave or evidence.frame_sha256 in frames:
            raise ValueError("wave rewards require unique increasing waves and distinct evidence")
        if not previous_time <= evidence.observed_at_ns <= outcome.evidence.observed_at_ns:
            raise ValueError("wave evidence out of terminal chronology")
        waves.add(wave)
        frames.add(evidence.frame_sha256)
        previous_wave, previous_time = wave, evidence.observed_at_ns
    # Explicit experimental shaping rule, NOT per-item causal effect or win proof.
    # Monotonic even beyond wave 20, including endless play; 20 is an explicit
    # shaping scale, not a claim that every supported objective ends on wave 20.
    return len(waves) / (len(waves) + 20) + (1.0 if outcome.kind == "objective_reached" else -1.0)


def validate_run(policy: ChoicePolicy, run: DecisionRun, *, for_training: bool = True,
                 verify_files: bool = False, max_age_ms: float = 500) -> dict[str, Any]:
    _text(run.run_id, "run_id")
    _text(run.movement_policy_version, "movement_policy_version")
    if run.split not in {"train", "evaluation"} or (for_training and run.split != "train"):
        raise ValueError("evaluation runs cannot train")
    if run.choice_policy_version != policy.version or run.run_id in policy.trained_run_ids:
        raise ValueError("wrong behavior policy or duplicate run")
    if run.clock_domain != CLOCK_DOMAIN or run.human_intervention or run.guard_failure:
        raise ValueError("unsupported clock, human takeover, or guard failure")
    if (not run.session_ids or len(set(run.session_ids)) != len(run.session_ids)
            or not isinstance(run.steps, tuple)):
        raise ValueError("require session provenance and immutable steps")
    for session in run.session_ids:
        _text(session, "session_id")
    max_age = _number(max_age_ms, "max_age_ms") * 1_000_000
    if not 0 < max_age <= 500_000_000:
        raise ValueError("freshness limit must be in (0,500] ms")
    reward = _return(run.outcome)
    eligible: list[int] = []
    excluded: dict[str, str] = {}
    ids: set[str] = set()
    last_sent = -1
    for index, step in enumerate(run.steps):
        _text(step.decision_id, "decision_id")
        if step.decision_id in ids:
            raise ValueError("duplicate decision IDs")
        ids.add(step.decision_id)
        _validate_decision(policy, step.decision)
        _check_evidence(step.observation)
        _integer(step.decided_at_ns, "decided_at_ns")
        _integer(step.sent_at_ns, "sent_at_ns")
        if not (step.observation.verified_at_ns <= step.decided_at_ns <= step.sent_at_ns
                < run.outcome.evidence.observed_at_ns and step.sent_at_ns > last_sent):
            raise ValueError("future observation, nonchronological send, or nonterminal run")
        last_sent = step.sent_at_ns
        if step.sent_at_ns - step.observation.observed_at_ns > max_age:
            raise ValueError("stale observation at input sending")
        if step.action_origin != "policy" or step.actual_candidate_id != step.decision.selected_id:
            raise ValueError("policy proposal differs from actual action or actor")
        _text(step.transmission_ref, "transmission_ref")
        if step.acknowledged is not True:
            excluded[step.decision_id] = "transmission_unacknowledged"
            continue
        applied = step.applied
        if applied is None or applied.accepted is not True:
            excluded[step.decision_id] = "application_unknown"
            continue
        _check_evidence(applied.before)
        _check_evidence(applied.after)
        next_send = run.steps[index + 1].sent_at_ns if index + 1 < len(run.steps) else run.outcome.evidence.observed_at_ns
        if (applied.before != step.observation or applied.after.observed_at_ns <= step.sent_at_ns
                or applied.after.observed_at_ns >= next_send
                or applied.after.verified_at_ns > run.outcome.evidence.verified_at_ns
                or applied.before.frame_sha256 == applied.after.frame_sha256):
            raise ValueError("application requires changed post-send evidence before the next action/terminal")
        kinds = {"upgrade_selected": "upgrade", "weapon_selected": "weapon",
                 "purchase_applied": "buy", "shop_departed": "skip"}
        if kinds.get(applied.kind) != step.decision.action:
            raise ValueError("unsupported or action-mismatched application evidence")
        if applied.kind == "purchase_applied":
            for value in (applied.currency_before, applied.currency_after, applied.cost):
                _integer(value, "purchase amount")
            if (applied.currency_before - applied.currency_after != applied.cost
                    or applied.ownership_changed is not True):
                excluded[step.decision_id] = "purchase_delta_or_ownership_unverified"
                continue
        eligible.append(index)
    if verify_files:
        # Preserve even pending/unverified originals; validity is only required
        # above for evidence that justifies a training sample.
        for evidence in _run_evidence(run):
            path = Path(evidence.frame_ref)
            _digest(evidence.frame_sha256)
            if not path.is_absolute() or not path.is_file() or _hash(path.read_bytes()) != evidence.frame_sha256:
                raise ValueError("evidence source missing or modified")
    return {"eligible_indices": eligible, "excluded": excluded, "return": reward,
            "reward_rule": REWARD_RULE, "split": run.split,
            "credit_limit": "run-level association; movement/environment confound item effects"}


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json(payload) + b"\n")


def save_run_manifest(run: DecisionRun, directory: Path | str) -> Path:
    """Freeze raw trajectory and bind it to session IDs. Never overwrite files.

    This also saves pending/truncated runs for audit; training validates them
    separately. No evidence flags or labels are inferred by this serializer.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = directory / "run.json"
    _write_new(record, {"schema": SCHEMA, "run": asdict(run)})
    manifest = directory / "manifest.json"
    _write_new(manifest, {"schema": SCHEMA, "run_file": "run.json",
                         "run_sha256": _hash(record.read_bytes()), "run_id": run.run_id,
                         "split": run.split, "session_ids": run.session_ids})
    return manifest


def _decision_from_dict(data: dict[str, Any]) -> ChoiceDecision:
    return ChoiceDecision(_features(data["state"]), tuple(ChoiceCandidate(**c) for c in data["candidates"]),
                          data["selected_id"], data["action"], tuple(data["probabilities"]),
                          data["log_probability"], data["value_estimate"], data["policy_version"])


def load_run_manifest(path: Path | str) -> DecisionRun:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("run_file") != "run.json":
        raise ValueError("invalid immutable run manifest")
    raw = (path.parent / "run.json").read_bytes()
    if _hash(raw) != manifest.get("run_sha256"):
        raise ValueError("trajectory hash mismatch")
    record = json.loads(raw)
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported trajectory schema")
    data = record["run"]
    steps = []
    for raw_step in data["steps"]:
        item = dict(raw_step)
        item["decision"] = _decision_from_dict(item["decision"])
        item["observation"] = Evidence(**item["observation"])
        if item["applied"] is not None:
            applied = dict(item["applied"])
            applied["before"] = Evidence(**applied["before"])
            applied["after"] = Evidence(**applied["after"])
            item["applied"] = AppliedEvidence(**applied)
        steps.append(ChoiceStep(**item))
    outcome = dict(data["outcome"])
    if outcome["evidence"] is not None:
        outcome["evidence"] = Evidence(**outcome["evidence"])
    outcome["wave_clears"] = tuple((w, Evidence(**e)) for w, e in outcome["wave_clears"])
    run = DecisionRun(**{**data, "steps": tuple(steps), "outcome": RunOutcome(**outcome),
                         "session_ids": tuple(data["session_ids"])})
    if any(manifest.get(key) != value for key, value in (
            ("run_id", run.run_id), ("split", run.split), ("session_ids", list(run.session_ids)))):
        raise ValueError("manifest provenance differs from trajectory")
    return run


def train_run(manifest_path: Path | str, source_checkpoint: Path | str,
              output: Path | str, *, learning_rate: float = 0.05,
              value_learning_rate: float = 0.1, max_kl: float = 0.02) -> dict[str, Any]:
    """One on-policy REINFORCE update, not behavior cloning or repeated epochs.

    Writes a new *unapproved* candidate only after numeric/KL guards pass. These
    guards are not independent performance evaluation or deployment approval.
    """
    for name, value in (("learning_rate", learning_rate), ("value_learning_rate", value_learning_rate), ("max_kl", max_kl)):
        if not 0 < _number(value, name) <= 1:
            raise ValueError(f"{name} must be in (0,1]")
    policy = ChoicePolicy.load(source_checkpoint)
    run = load_run_manifest(manifest_path)
    report = validate_run(policy, run, verify_files=True)
    eligible = report["eligible_indices"]
    if not eligible:
        raise ValueError("no verified-applied choices; candidate not written")
    actor_gradient = [0.0] * ACTOR_SIZE
    value_gradient = [0.0] * VALUE_SIZE
    for index in eligible:
        decision = run.steps[index].decision
        selected = _validate_decision(policy, decision)
        advantage = report["return"] - decision.value_estimate
        vectors = [_actor_features(decision.state, c) for c in decision.candidates]
        for feature in range(ACTOR_SIZE):
            expectation = math.fsum(p * v[feature] for p, v in zip(decision.probabilities, vectors))
            actor_gradient[feature] += advantage * (vectors[selected][feature] - expectation) / len(eligible)
        for feature, value in enumerate(_value_features(decision.state)):
            value_gradient[feature] += advantage * value / len(eligible)
    # Clipping is an explicit numerical guard, not an improvement certificate.
    norm = math.sqrt(math.fsum(v * v for v in actor_gradient))
    actor_scale = min(1.0, 1 / norm) if norm else 1.0
    value_norm = math.sqrt(math.fsum(v * v for v in value_gradient))
    value_scale = min(1.0, 1 / value_norm) if value_norm else 1.0
    candidate = ChoicePolicy(
        tuple(w + learning_rate * actor_scale * g for w, g in zip(policy.actor_weights, actor_gradient)),
        tuple(w + value_learning_rate * value_scale * g for w, g in zip(policy.value_weights, value_gradient)),
        policy.updates + 1, policy.version, (*policy.trained_run_ids, run.run_id),
        _hash(Path(manifest_path).read_bytes()))
    divergences = []
    for index in eligible:
        decision = run.steps[index].decision
        after = candidate.distribution(decision.state, decision.candidates)
        divergences.append(math.fsum(p * math.log(p / q) for p, q in zip(decision.probabilities, after) if p > 0))
    measured_kl = max(divergences)
    if measured_kl > max_kl:
        raise ValueError("candidate exceeds maximum per-observation KL; not saved")
    candidate.save(output)
    return {**report, "run_id": run.run_id, "source_version": policy.version,
            "candidate_version": candidate.version, "candidate_path": str(output),
            "gradient_norm": norm, "maximum_kl": measured_kl,
            "updated_choices": len(eligible), "deployment_status": "unapproved_candidate",
            "evaluation_status": "independent_full_run_evaluation_required",
            "algorithm": "candidate_conditioned_REINFORCE_with_learned_value_baseline"}


def verify_split_separation(train_manifests: Sequence[Path | str],
                            evaluation_manifests: Sequence[Path | str]) -> dict[str, Any]:
    """Reject shared physical-run IDs, session IDs, or source image hashes.

    Session/run IDs are supplied provenance, not proof of separate game seeds.
    Comparison protocols must additionally pin movement policy and game settings.
    """
    groups = [list(map(load_run_manifest, paths)) for paths in (train_manifests, evaluation_manifests)]
    if not all(groups):
        raise ValueError("both training and evaluation manifests required")
    keys = []
    for runs, split in zip(groups, ("train", "evaluation")):
        if any(run.split != split for run in runs):
            raise ValueError("manifest split mismatch")
        run_ids = [r.run_id for r in runs]
        sessions = [s for r in runs for s in r.session_ids]
        if len(set(run_ids)) != len(run_ids) or len(set(sessions)) != len(sessions):
            raise ValueError("duplicate run/session grouping within split")
        keys.append((set(run_ids), set(sessions), {e.frame_sha256 for r in runs for e in _run_evidence(r)}))
    if any(left & right for left, right in zip(*keys)):
        raise ValueError("train/evaluation run, session, or image leakage")
    return {"independent_by_manifest": True, "training_runs": len(groups[0]),
            "evaluation_runs": len(groups[1]), "performance_improvement_proven": False}
