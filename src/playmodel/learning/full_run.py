"""Join verified menu macros and combat under one frozen recurrent policy.

This module sends no input. Navigation is execution of a sampled macro, not a
new policy decision. A bootstrap-only image never commits recurrent memory.
Scalar GAE is computed over the chronological trajectory by recurrent_ppo;
images are optimized in short overlapping sequences with explicit source IDs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .recurrent_ppo import RecurrentActorCritic, RolloutBatch, save_checkpoint
from .runtime_contract import (PHASE_SCHEMA, RUNTIME_CONTRACT, LEGACY_RUNTIME_CONTRACT,
                               contract_fields, contract_identity, require_current_contract)

SCHEMA = "playmodel.full-recurrent-run.v1"
CLOCK = "perf_counter_ns_same_host"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _proof(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"missing original evidence: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _currency_sources(frame):
    """Bind supplementary local OCR to the immutable frame used for a macro."""
    frame = Path(frame).resolve()
    evidence_path = frame.with_name('currency-roi.json')
    if not evidence_path.exists():
        return []
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if (Path(evidence['source_path']).resolve() != frame
            or evidence['source_png_sha256'] != _sha(frame)):
        raise ValueError('currency OCR belongs to another source frame')
    sources = [_proof(evidence_path)]
    for filename, expected in evidence['evidence_files']:
        path = Path(filename).resolve()
        if path.parent != frame.parent:
            raise ValueError('currency derivation left the source frame directory')
        proof = _proof(path)
        if proof['sha256'] != expected:
            raise ValueError('currency OCR derivation changed')
        sources.append(proof)
    return sources


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)


def _save(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def chunk_full_trajectory(batch: RolloutBatch, initial_states: torch.Tensor, *,
                          chunk_steps=32, burn_in=8):
    """Return (batch, original-transition indices); each loss step occurs once."""
    if (type(chunk_steps) is not int or type(burn_in) is not int or chunk_steps < 1
            or burn_in < 0 or chunk_steps + burn_in > 64 or batch.valid.shape[1] != 1
            or not batch.valid.all()
            or initial_states.shape != (batch.valid.shape[0], 1, batch.initial_hidden.shape[1])):
        raise ValueError("require one chronological sequence and <=64-step chunks")
    total = batch.valid.shape[0]
    count = math.ceil(total / chunk_steps)
    length = chunk_steps + burn_in
    result = {}
    indices = torch.full((length, count), -1, dtype=torch.long)
    for name, value in batch.__dict__.items():
        if not isinstance(value, torch.Tensor):
            result[name] = value
        elif name == "initial_hidden":
            result[name] = torch.zeros(count, value.shape[1], dtype=value.dtype)
        else:
            result[name] = torch.zeros(length, count, *value.shape[2:], dtype=value.dtype)
    for column, start in enumerate(range(0, total, chunk_steps)):
        source = max(0, start - burn_in)
        target = burn_in - (start - source)
        stop = min(total, start + chunk_steps)
        result["initial_hidden"][column] = initial_states[source, 0]
        indices[target:target + stop - source, column] = torch.arange(source, stop)
        for name, value in batch.__dict__.items():
            if isinstance(value, torch.Tensor) and name != "initial_hidden":
                result[name][target:target + stop - source, column] = value[source:stop, 0]
    return RolloutBatch(**result), indices


def build_source_proofs(snapshot: dict) -> list[dict]:
    """Bind original and derived observation files without turning OCR into truth."""
    proofs = {}
    def visit(value):
        if isinstance(value, dict):
            source = value.get('frame_ref')
            expected = value.get('source_png_sha256', value.get('frame_sha256'))
            if source and expected:
                proof = _proof(source)
                if proof['sha256'] != expected:
                    raise ValueError('Observed build source modified')
                proofs[proof['path']] = proof
            if value.get('path') and value.get('sha256'):
                proof = _proof(value['path'])
                if proof['sha256'] != value['sha256']:
                    raise ValueError('Observed build derived evidence modified')
                proofs[proof['path']] = proof
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
    visit(snapshot)
    return list(proofs.values())


class FullRunRecorder:
    def __init__(self, model: RecurrentActorCritic, run_id: str, *, split="train",
                 session_ids=(), start_evidence=None, game_build_id="23429717"):
        if not isinstance(run_id, str) or not run_id or split not in ("train", "evaluation", "validation", "test"):
            raise ValueError("run ID and fixed dataset split required")
        self.model = deepcopy(model).to("cpu").eval()
        self.behavior_version = self.model.policy_version()
        self.run_id, self.split = run_id, split
        self.session_ids = list(session_ids)
        self.start_evidence = deepcopy(start_evidence)
        if self.start_evidence is not None:
            start = self.start_evidence
            proof = _proof(start["frame_ref"])
            if (start.get("verified") is not True or start.get("independent_of_policy") is not True
                    or proof["sha256"] != start.get("frame_sha256")
                    or type(start.get("observed_at_ns")) is not int):
                raise ValueError("new-run origin requires independent source-backed evidence")
        self.game_build_id = game_build_id
        self.build_state = None
        if self.model.config.context_dim == 64:
            from playmodel.games.brotato.state_features import BoundedBuildState
            self.build_state = BoundedBuildState(run_id, game_build_id=game_build_id)
        self.hidden = self.model.initial_hidden(1)
        self.records = []
        self.rejection_reasons = []
        self.source_manifests = []
        self.closed = False

    def invalidate(self, reason: str):
        """Preserve diagnostics but prevent training after an untracked action."""
        if not reason:
            raise ValueError("rejection reason required")
        self.rejection_reasons.append(str(reason))

    def abort(self, directory, reason: str):
        """Preserve partial records when no valid final bootstrap is available."""
        self._open()
        self.invalidate(reason)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        save_checkpoint(self.model, directory / "behavior-policy.pt", {"run_id": self.run_id, "split": self.split})
        _save(directory / "partial-records.pt", {"records": self.records, "last_committed_hidden": self.hidden})
        _json(directory / "events.json", [{k: v for k, v in r.items() if k != "tensors"} for r in self.records])
        manifest = {"schema": SCHEMA, **contract_fields(), "run_id": self.run_id,
                    "split": self.split, "behavior_version": self.behavior_version,
                    "training_eligible": False, "full_run_complete": False, "ending": "aborted",
                    "rejection_reasons": self.rejection_reasons, "steps": len(self.records),
                    "sources": [r["source"] for r in self.records],
                    "files": [{"path": p.name, "sha256": _sha(p)} for p in sorted(directory.iterdir())]}
        _json(directory / "manifest.json", manifest)
        self.closed = True
        return {**manifest, "directory": str(directory), "manifest_path": str(directory / "manifest.json")}

    def _open(self):
        if self.closed or self.model.policy_version() != self.behavior_version:
            raise ValueError("recorder closed or behavior model changed mid-run")

    def append_decision(self, tensors: dict, *, evidence: dict, reward=0.0,
                        reward_evidence=None):
        """Append detached inputs/action/value/logp/hidden after actual acceptance.

        ``tensors`` is the FrozenMacroDecision.tensors() contract. Combat uses the
        same fields. Menu actions require verified application; movement transport
        may have unknown game application, never promoted into an effect label.
        """
        self._open()
        record = self._prepare_decision(tensors, evidence=evidence, reward=reward,
            reward_evidence=reward_evidence, expected_hidden=self.hidden,
            previous=self.records[-1] if self.records else None, first=not self.records)
        self._validate_policy_records([record])
        self._open()
        self._commit_records([record])

    def _prepare_decision(self, tensors, *, evidence, reward, reward_evidence,
                          expected_hidden, previous, first):
        """Check and stage one record without changing this recorder's history."""
        required = ("images", "context", "phase", "candidates", "legal_mask", "hidden_before",
                    "next_hidden", "actions", "old_log_probs", "old_values", "reset")
        data = {key: tensors[key].detach().cpu().clone() for key in required}
        evidence = deepcopy(evidence)
        if evidence.get("clock_domain", CLOCK) != CLOCK:
            raise ValueError("mixed timestamp clocks")
        observed, available, decided, sent = [evidence[key] for key in
                                             ("observed_at_ns", "available_at_ns", "decided_at_ns", "sent_at_ns")]
        if (any(type(x) is not int or x < 0 for x in (observed, available, decided, sent))
                or not observed <= available <= decided <= sent):
            raise ValueError("future observation or invalid action timing")
        if previous is not None and observed <= previous["evidence"]["sent_at_ns"]:
            raise ValueError("next decision observation predates the previous action")
        if first and self.start_evidence and observed < self.start_evidence["observed_at_ns"]:
            raise ValueError("decision precedes the verified new-run origin")
        action = int(data["actions"].item())
        phase = int(data["phase"].item())
        if (evidence.get("action_origin") != "policy" or evidence.get("transmitted") is not True
                or evidence.get("actual_action") != action or evidence.get("acknowledged") is False):
            raise ValueError("policy proposal differs from actual transport")
        if phase != 0 and evidence.get("game_application_verified") is not True:
            raise ValueError("pending or unknown menu application cannot enter PPO")
        source = _proof(evidence["frame_ref"])
        if source["sha256"] != evidence["frame_sha256"]:
            raise ValueError("original observation modified")
        if bool(data["reset"].item()) != first:
            raise ValueError("only the beginning of this run resets recurrent memory")
        if (data["hidden_before"].shape != expected_hidden.shape
                or not torch.allclose(data["hidden_before"], expected_hidden, atol=2e-5, rtol=2e-5)):
            raise ValueError("menu/combat hidden-state seam mismatch")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("finite reward required")
        if reward:
            if (not reward_evidence or reward_evidence.get("verified") is not True
                    or reward_evidence.get("independent_of_policy") is not True):
                raise ValueError("nonzero rewards require independent verified event provenance")
            reward_source = _proof(reward_evidence["frame_ref"])
            if reward_source["sha256"] != reward_evidence["frame_sha256"]:
                raise ValueError("reward original modified")
        return {"tensors": data, "evidence": evidence, "source": source,
                "reward": float(reward), "reward_evidence": deepcopy(reward_evidence),
                "next_value": None, "elapsed_seconds": None}

    def _validate_policy_records(self, records):
        """Recompute each logged decision, batching only independent saved states.

        Each row retains its own original recurrent input; this never advances
        memory using a batched neighbour or substitutes a bootstrap state.
        """
        keys = ("images", "context", "phase", "candidates", "legal_mask")
        with torch.no_grad():
            for start in range(0, len(records), 32):
                part = [row["tensors"] for row in records[start:start + 32]]
                data = {key: torch.cat([row[key] for row in part], dim=0) for key in
                        (*keys, "hidden_before", "reset", "actions", "old_log_probs", "old_values", "next_hidden")}
                output = self.model.step(*(data[key] for key in keys),
                    hidden=data["hidden_before"], reset=data["reset"])
                actions = data["actions"]
                if ((actions < 0).any() or (actions >= output.logits.shape[-1]).any()
                        or not data["legal_mask"].gather(-1, actions[:, None]).all()):
                    raise ValueError("actual action was illegal")
                logp = output.logits.log_softmax(-1).gather(-1, actions[:, None]).squeeze(-1)
                for expected, actual in ((logp, data["old_log_probs"]),
                                         (output.value, data["old_values"]),
                                         (output.next_hidden, data["next_hidden"])):
                    if not torch.allclose(expected, actual, atol=2e-5, rtol=2e-5):
                        raise ValueError("logged policy quantities do not match frozen behavior")

    def _commit_records(self, records):
        """Commit a completely verified group, including its prior-record seam."""
        if not records:
            raise ValueError("empty decision group")
        combined = list(self.records)
        if combined:
            # Only this shallow record changes; its frozen tensors/evidence do
            # not. Failed verification never alters the existing final seam.
            combined[-1] = dict(combined[-1])
        for record in records:
            if combined:
                previous = combined[-1]
                previous["next_value"] = float(record["tensors"]["old_values"].item())
                previous["elapsed_seconds"] = (record["evidence"]["observed_at_ns"]
                                                - previous["evidence"]["decided_at_ns"]) / 1e9
            combined.append(record)
        hidden = records[-1]["tensors"]["next_hidden"].clone()
        self.records, self.hidden = combined, hidden

    def append_macro(self, decision, application, *, frame_path=None):
        self._open()
        if (decision.behavior_version != self.behavior_version or application.accepted is not True
                or application.decision_id != decision.decision_id or application.actual_target != decision.target):
            raise ValueError("unknown, mismatched, or off-policy macro application")
        observation = decision.observation
        path = Path(frame_path or observation.frame_id).resolve()
        if Path(observation.frame_id).resolve() != path:
            raise ValueError("macro source does not match observation identity")
        proof = _proof(path)
        after_sources = [_proof(name) for name in application.after_frame_ids]
        auxiliary_sources = [proof for name in (path, *application.after_frame_ids)
                             for proof in _currency_sources(name)]
        build_snapshot = json.loads(decision.build_state_json) if decision.build_state_json else None
        if build_snapshot is not None:
            auxiliary_sources.extend(build_source_proofs(build_snapshot))
        if (not after_sources or application.next_observed_at_ns is None
                or not application.sent_at_ns < application.next_observed_at_ns <= application.verified_at_ns):
            raise ValueError("macro requires post-send application evidence")
        evidence = {"frame_ref": str(path), "frame_sha256": proof["sha256"],
            "source_pixels_sha256": observation.source_pixels_sha256,
            "observed_at_ns": observation.observed_at_ns, "available_at_ns": observation.available_at_ns,
            "decided_at_ns": decision.decided_at_ns, "sent_at_ns": application.sent_at_ns,
            "actual_action": decision.action_index, "action_origin": "policy", "transmitted": True,
            "acknowledged": None, "game_application_verified": True, "clock_domain": CLOCK,
            "decision_id": decision.decision_id, "macro_application": asdict(application),
            "candidate_observations": [asdict(c) for c in observation.candidates],
            "after_sources": after_sources, "auxiliary_sources": auxiliary_sources,
            "feature_effects_verified": application.feature_effects_verified,
            "observed_build_state": build_snapshot}
        self.append_decision(decision.tensors(), evidence=evidence)

    def append_combat_report(self, report: dict):
        """Read the collector's hash-checked flat records, never its chunk seams."""
        from playmodel.games.brotato.neural_runtime import load_rollout
        started = time.perf_counter()
        self._open()
        directory = Path(report["session_directory"]).resolve()
        actual = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if actual.get("rollout_eligible") is not True or not actual.get("flat_rollout_path"):
            self.invalidate("combat collector rejected this segment")
            raise ValueError("combat segment is not eligible")
        batch = load_rollout(actual["flat_rollout_path"])
        if (batch.runtime_contract != RUNTIME_CONTRACT or batch.behavior_version != self.behavior_version
                or batch.split != self.split or not batch.valid.all() or batch.valid.shape[1] != 1):
            raise ValueError("incompatible combat segment")
        states = torch.load(actual["initial_states_path"], weights_only=True, map_location="cpu")["initial_states"]
        evidence = [json.loads(line) for line in (directory / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(evidence) != batch.valid.shape[0] or states.shape != (len(evidence), 1, self.hidden.shape[1]):
            raise ValueError("combat action/state records are incomplete")
        terminal = json.loads((directory / "terminal.json").read_text(encoding="utf-8")) if (directory / "terminal.json").exists() else None
        loaded = time.perf_counter()
        pending = []
        expected_hidden = self.hidden
        previous = self.records[-1] if self.records else None
        for index, row in enumerate(evidence):
            data = {key: getattr(batch, key)[index].clone() for key in
                    ("images", "context", "phase", "candidates", "legal_mask", "actions", "old_log_probs", "old_values", "reset")}
            data["hidden_before"] = states[index]
            data["next_hidden"] = states[index + 1] if index + 1 < len(evidence) else torch.tensor(actual["final_hidden"])
            row["frame_ref"] = str((directory / row["frame_ref"]).resolve())
            record = self._prepare_decision(data, evidence=row, reward=float(batch.rewards[index, 0]),
                reward_evidence=terminal, expected_hidden=expected_hidden, previous=previous,
                first=not self.records and index == 0)
            pending.append(record)
            expected_hidden, previous = record["tensors"]["next_hidden"], record
        prepared = time.perf_counter()
        self._validate_policy_records(pending)
        validated = time.perf_counter()
        sources = [_proof(directory / "manifest.json"),
                   *json.loads((directory / 'manifest.json').read_text(encoding='utf-8')).get('sources', [])]
        # One pre/post hash of the owned model replaces a hash per transition.
        # A mutation during loading/validation rejects the entire staged group.
        self._open()
        self._commit_records(pending)
        self.session_ids.append(directory.name)
        self.source_manifests.extend(sources)
        finished = time.perf_counter()
        from playmodel.execution_log import event
        event('full_run_combat_append', run_id=self.run_id, session_directory=str(directory),
              transitions=len(pending), batch_size=32, seconds={
                  'load_and_initial_model_check': loaded - started,
                  'evidence_and_seam_validation': prepared - loaded,
                  'batched_policy_validation': validated - prepared,
                  'final_model_check_and_commit': finished - validated,
                  'total': finished - started})

    def finish(self, directory, *, kind: str, evidence: dict, next_value: float | None = None,
               chunk_steps=32, burn_in=8):
        """Freeze a death-completed run or explicitly truncated partial trajectory.

        Human takeover/unknown actions must first call invalidate(). A truncation
        bootstrap is evaluated on the true final observation without committing
        its hidden state. A death is the only full-run terminal in this adapter.
        """
        self._open()
        if kind not in ("death", "truncated", "aborted") or not self.records:
            raise ValueError("nonempty run and supported ending required")
        evidence = deepcopy(evidence)
        source = _proof(evidence["frame_ref"])
        if source["sha256"] != evidence["frame_sha256"]:
            raise ValueError("ending evidence changed")
        last = self.records[-1]
        end_time = evidence["observed_at_ns"]
        if type(end_time) is not int or end_time <= last["evidence"]["sent_at_ns"]:
            raise ValueError("ending observation predates last input")
        if kind == "death":
            if (evidence.get("kind") != "death" or evidence.get("verified") is not True
                    or evidence.get("independent_of_policy") is not True):
                raise ValueError("independently verified death required")
            last["reward"] = -1.0
            last["reward_evidence"] = evidence
            next_value = 0.0
        elif next_value is None or not math.isfinite(next_value):
            self.invalidate("true final-observation bootstrap missing")
            next_value = 0.0
        if kind == "aborted":
            self.invalidate("run aborted")
        last["next_value"] = float(next_value)
        last["elapsed_seconds"] = (end_time - last["evidence"]["decided_at_ns"]) / 1e9
        batch = self._flat(kind)
        initial_states = torch.stack([r["tensors"]["hidden_before"] for r in self.records])
        chunks, indices = chunk_full_trajectory(batch, initial_states, chunk_steps=chunk_steps, burn_in=burn_in)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        save_checkpoint(self.model, directory / "behavior-policy.pt", {"run_id": self.run_id, "split": self.split})
        _save(directory / "flat.pt", {"batch": batch.__dict__, "initial_states": initial_states})
        _save(directory / "rollout.pt", {"batch": chunks.__dict__, "transition_indices": indices})
        ledger = [{key: value for key, value in row.items() if key != "tensors"} for row in self.records]
        _json(directory / "events.json", ledger)
        external = [r["source"] for r in self.records] + self.source_manifests + [source]
        if self.start_evidence:
            external.append(_proof(self.start_evidence["frame_ref"]))
        for row in self.records:
            external.extend(row["evidence"].get("after_sources", []))
            external.extend(row["evidence"].get("auxiliary_sources", []))
            if row["reward_evidence"]:
                external.append(_proof(row["reward_evidence"]["frame_ref"]))
        external = list({r["path"]: r for r in external}.values())
        for proof in external:
            if _sha(proof["path"]) != proof["sha256"]:
                raise ValueError("original source changed before manifest freeze")
        manifest = {"schema": SCHEMA, **contract_fields(),
            "run_id": self.run_id, "split": self.split, "behavior_version": self.behavior_version,
            "game_build_id": self.game_build_id, "session_ids": list(dict.fromkeys(self.session_ids)),
            "start_evidence": self.start_evidence, "ending": kind, "ending_evidence": evidence,
            "full_run_complete": kind == "death" and bool(self.start_evidence and self.start_evidence.get("verified") is True),
            "training_eligible": not self.rejection_reasons and self.split == "train",
            "rejection_reasons": self.rejection_reasons, "phase_counts": {
                str(phase): sum(int(r["tensors"]["phase"].item()) == phase for r in self.records)
                for phase in sorted({int(r["tensors"]["phase"].item()) for r in self.records})},
            "steps": len(self.records), "chunk_steps": chunk_steps, "burn_in": burn_in,
            "gae_scope": "full_chronological_scalar_targets_short_visual_sequences",
            "sources": external, "files": [{"path": p.name, "sha256": _sha(p)} for p in sorted(directory.iterdir())],
            "deployment_status": "unapproved_experiment", "game_improvement_proven": False}
        _json(directory / "manifest.json", manifest)
        self.closed = True
        return {**manifest, "manifest_path": str(directory / "manifest.json"), "directory": str(directory)}

    def _flat(self, kind):
        count = max(r["tensors"]["candidates"].shape[1] for r in self.records)
        width = max(9, count)
        columns = {key: [] for key in ("images", "context", "phase", "candidates", "legal_mask", "actions", "old_log_probs", "old_values", "reset")}
        for row in self.records:
            for key in columns:
                value = row["tensors"][key]
                if key == "candidates":
                    value = F.pad(value, (0, 0, 0, count - value.shape[1]))
                elif key == "legal_mask":
                    value = F.pad(value, (0, width - value.shape[1]))
                columns[key].append(value)
        result = {key: torch.stack(value) for key, value in columns.items()}
        length = len(self.records)
        result.update(rewards=torch.tensor([[r["reward"]] for r in self.records]),
            next_values=torch.tensor([[r["next_value"]] for r in self.records]),
            elapsed_seconds=torch.tensor([[r["elapsed_seconds"]] for r in self.records]),
            terminated=torch.zeros(length, 1, dtype=torch.bool), truncated=torch.zeros(length, 1, dtype=torch.bool),
            valid=torch.ones(length, 1, dtype=torch.bool),
            initial_hidden=self.records[0]["tensors"]["hidden_before"], behavior_version=self.behavior_version,
            rollout_id=self.run_id, split=self.split, runtime_contract=RUNTIME_CONTRACT)
        result["terminated"][-1, 0] = kind == "death"
        result["truncated"][-1, 0] = kind != "death"
        return RolloutBatch(**result)


def load_full_run(path, *, device="cpu", require_training=True, expected_runtime_contract=None):
    """Return (chunk_batch, chronological_batch, source_indices, manifest)."""
    path = Path(path).resolve()
    manifest_path = path if path.is_file() else path / "manifest.json"
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or not manifest.get("phase_schema"):
        raise ValueError("unsupported full-run manifest")
    runtime_contract, _ = contract_identity(manifest)
    if expected_runtime_contract is not None and runtime_contract != expected_runtime_contract:
        raise ValueError("full-run runtime contract differs from required scope")
    if require_training and manifest.get("training_eligible") is not True:
        raise ValueError("run is not eligible for PPO training")
    if require_training:
        require_current_contract(manifest)
    for row in manifest["files"]:
        source = (directory / row["path"]).resolve()
        if not source.is_relative_to(directory) or _sha(source) != row["sha256"]:
            raise ValueError("frozen run file digest mismatch")
    for row in manifest["sources"]:
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError("original external evidence digest mismatch")
    flat = torch.load(directory / "flat.pt", map_location="cpu", weights_only=True)
    chunks = torch.load(directory / "rollout.pt", map_location="cpu", weights_only=True)
    def read_batch(data):
        # Dataclass defaults are for new in-memory batches, never legacy files.
        return RolloutBatch(**{**data, "runtime_contract": data.get("runtime_contract", LEGACY_RUNTIME_CONTRACT)})
    batch, trajectory = read_batch(chunks["batch"]), read_batch(flat["batch"])
    if any(b.behavior_version != manifest["behavior_version"] or b.rollout_id != manifest["run_id"]
           or b.split != manifest["split"] or b.runtime_contract != runtime_contract for b in (batch, trajectory)):
        raise ValueError("run identity/version/split mismatch")
    return batch.to(device), trajectory, chunks["transition_indices"].to(device), manifest
