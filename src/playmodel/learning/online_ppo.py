"""Strict same-version PPO fragments; no capture, input, or actor mutation.

All public operations except describe() perform file/model work and belong on a
collector/coordinator thread, never the realtime input gate. Actor handoff is an
explicit caller operation after load_candidate(), with a fresh observation and
reset recurrent state. Existing full-run training eligibility is not weakened.
"""
from __future__ import annotations

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module("playmodel.learning.online_ppo")

import argparse
from dataclasses import asdict
import io
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
import uuid

from playmodel.atomic_io import atomic_json
from .recurrent_training_worker import (
    TrainingJob, TrainingJobError, TrainingCancelled, _Busy, _exclusive, _occupied,
    _sha, _digest, _read, _settings, CANCEL_GRACE_SECONDS, STARTUP_GRACE_SECONDS,
)
from .runtime_contract import RUNTIME_CONTRACT, contract_fields, require_current_contract

SCHEMA = "playmodel.online-ppo-job.v1"
BOOTSTRAP_SCHEMA = "playmodel.online-ppo-bootstrap.v1"
MAX_JOB_SECONDS = 300
COLLECT_STEPS = 64  # Initial proposal, not an empirically optimal setting.
CODE_FILES = (
    "src/playmodel/learning/online_ppo.py", "src/playmodel/learning/recurrent_training_worker.py",
    "src/playmodel/learning/recurrent_ppo.py", "src/playmodel/learning/full_run.py",
    "src/playmodel/learning/runtime_contract.py", "src/playmodel/atomic_io.py",
)


def _save_tensor(path, payload):
    import torch
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    return path


def save_online_bootstrap(path, inputs, hidden, *, behavior_version,
                          observed_at_ns, available_at_ns):
    """Preserve a value-only observation without sampling or committing hidden."""
    if (len(inputs) != 5 or not isinstance(behavior_version, str) or not behavior_version
            or type(observed_at_ns) is not int or type(available_at_ns) is not int
            or not 0 <= observed_at_ns <= available_at_ns):
        raise ValueError("invalid causal bootstrap contract")
    path = _save_tensor(path, dict(schema=BOOTSTRAP_SCHEMA, **contract_fields(),
        inputs=tuple(item.detach().cpu().clone() for item in inputs), hidden=hidden.detach().cpu().clone(),
        behavior_version=behavior_version, observed_at_ns=observed_at_ns, available_at_ns=available_at_ns))
    return {"bootstrap_ref": str(path), "bootstrap_sha256": _sha(path)}


def _header(path):
    path = Path(path).resolve()
    header = _read(path)
    require_current_contract(header)
    if (header.get("split") != "train" or header.get("training_eligible") is not True
            or header.get("rejection_reasons") or header.get("ending") not in ("truncated", "death")
            or not isinstance(header.get("steps"), int) or not 1 <= header["steps"] <= 4096
            or not header.get("run_id") or not header.get("session_ids")):
        raise ValueError("online PPO requires a verified train fragment with episode/session identity")
    names = {item["path"] for item in header.get("files", [])}
    if not {"behavior-policy.pt", "flat.pt", "rollout.pt", "events.json"}.issubset(names):
        raise ValueError("online fragment lacks frozen policy/trajectory/evidence")
    if header["ending"] == "truncated":
        ending = header.get("ending_evidence", {})
        if not ending.get("bootstrap_ref") or not ending.get("bootstrap_sha256"):
            raise ValueError("truncated online fragment needs recorded bootstrap inputs")
    return path, header


def _load_fragment(path, *, device="cpu"):
    """Reuse full-run hashes and PPO checks, adding recorded bootstrap/evidence."""
    import torch
    from .full_run import load_full_run
    from .recurrent_ppo import load_checkpoint, recurrent_outputs

    path, header = _header(path)
    chunks, flat, indices, manifest = load_full_run(path, device=device,
                                                 expected_runtime_contract=RUNTIME_CONTRACT)
    model, metadata = load_checkpoint(path.parent / "behavior-policy.pt", device=device)
    version = model.policy_version()
    if version != manifest["behavior_version"] or flat.behavior_version != version:
        raise ValueError("fragment/source policy mismatch")
    events = _read(path.parent / "events.json")
    if len(events) != flat.valid.shape[0] or len(events) != header["steps"]:
        raise ValueError("fragment evidence/transition count mismatch")
    ending = manifest["ending_evidence"]
    for index, row in enumerate(events):
        evidence = row["evidence"]
        observed, available, decided, sent = (evidence[k] for k in
            ("observed_at_ns", "available_at_ns", "decided_at_ns", "sent_at_ns"))
        if (any(type(x) is not int for x in (observed, available, decided, sent))
                or not 0 <= observed <= available <= decided <= sent
                or evidence.get("action_origin") != "policy" or evidence.get("transmitted") is not True
                or evidence.get("acknowledged") is False
                or evidence.get("actual_action") != int(flat.actions[index, 0])
                or (int(flat.phase[index, 0]) != 0 and evidence.get("game_application_verified") is not True)
                or _sha(evidence["frame_ref"]) != evidence["frame_sha256"]):
            raise ValueError("invalid actual-action evidence in online fragment")
        next_time = events[index + 1]["evidence"]["observed_at_ns"] if index + 1 < len(events) else ending["observed_at_ns"]
        if (next_time <= sent or not math.isclose(float(flat.elapsed_seconds[index, 0]),
                (next_time - decided) / 1e9, rel_tol=2e-5, abs_tol=2e-5)):
            raise ValueError("online transition crosses missing/incorrect causal observation")
        reward = float(flat.rewards[index, 0])
        if reward != float(row["reward"]) or reward not in (-1., 0., 1.):
            raise ValueError("unsupported online reward")
        if reward:
            proof = row.get("reward_evidence") or {}
            if (proof.get("verified") is not True or proof.get("independent_of_policy") is not True
                    or proof.get("kind") != ("death" if reward == -1 else "wave_clear")
                    or _sha(proof["frame_ref"]) != proof["frame_sha256"]):
                raise ValueError("online reward lacks independent event proof")
    if not bool(flat.reset[0, 0]) or bool(flat.reset[1:].any()):
        raise ValueError("online version fragment must start with exactly one recurrent reset")
    with torch.no_grad():
        _, _, states = recurrent_outputs(model, flat)
    logged = torch.load(path.parent / "flat.pt", map_location="cpu", weights_only=True)["initial_states"].to(device)
    if (logged.shape != states.shape or not torch.allclose(logged[0], flat.initial_hidden, atol=2e-5, rtol=2e-5)
            or not torch.allclose(logged[1:], states[:-1], atol=2e-5, rtol=2e-5)):
        raise ValueError("online recurrent hidden seam mismatch")
    if manifest["ending"] == "death":
        if (ending.get("kind") != "death" or ending.get("verified") is not True
                or ending.get("independent_of_policy") is not True
                or not bool(flat.terminated[-1, 0]) or float(flat.next_values[-1, 0]) != 0.):
            raise ValueError("online death terminal/bootstrap mismatch")
    else:
        reference = Path(ending["bootstrap_ref"])
        if _sha(reference) != ending["bootstrap_sha256"]:
            raise ValueError("online bootstrap evidence changed")
        bootstrap = torch.load(reference, map_location="cpu", weights_only=True)
        require_current_contract(bootstrap)
        if (bootstrap.get("schema") != BOOTSTRAP_SCHEMA or bootstrap.get("behavior_version") != version
                or bootstrap.get("observed_at_ns") != ending["observed_at_ns"]
                or bootstrap.get("available_at_ns") != ending.get("available_at_ns")
                or not bootstrap["observed_at_ns"] <= bootstrap["available_at_ns"]
                or not torch.allclose(bootstrap["hidden"].to(device), states[-1], atol=2e-5, rtol=2e-5)
                or not bool(flat.truncated[-1, 0])):
            raise ValueError("online bootstrap source/time/hidden mismatch")
        with torch.no_grad():
            predicted = model.step(*(x.to(device) for x in bootstrap["inputs"]), hidden=bootstrap["hidden"].to(device)).value
        if not torch.allclose(predicted, flat.next_values[-1], atol=2e-5, rtol=2e-5):
            raise ValueError("online bootstrap value differs from frozen policy")
    return model, metadata, chunks, flat, indices, manifest


def _request(manifest_path, *, root, output, device, seed, stop_file):
    path, header = _header(manifest_path)
    root = Path(root).resolve()
    source = path.parent / "behavior-policy.pt"
    body = dict(schema=SCHEMA, manifest_path=str(path), manifest_sha256=_sha(path),
        checkpoint=str(source), source_sha256=_sha(source), source_version=header["behavior_version"],
        training={"manifest_path": str(path), "run_id": header["run_id"]},
        root=str(root), output=str(Path(output).resolve()), device=str(device), seed=seed,
        ppo_config=_settings(header["burn_in"], seed), max_seconds=MAX_JOB_SECONDS,
        stop_file=str(Path(stop_file).resolve()) if stop_file else None,
        code_files={name: _sha(root / name) for name in CODE_FILES})
    identity = {k: body[k] for k in ("schema", "source_sha256", "manifest_sha256", "manifest_path")}
    return {**body, "job_id": _digest(identity), "request_sha256": _digest(body)}


def _verify(request, *, code=True):
    body = {k: v for k, v in request.items() if k not in ("job_id", "request_sha256")}
    identity = {k: request[k] for k in ("schema", "source_sha256", "manifest_sha256", "manifest_path")}
    if (request.get("schema") != SCHEMA or _digest(body) != request.get("request_sha256")
            or _digest(identity) != request.get("job_id")):
        raise ValueError("online request identity changed")
    if _sha(request["checkpoint"]) != request["source_sha256"] or _sha(request["manifest_path"]) != request["manifest_sha256"]:
        raise ValueError("online source or manifest changed")
    manifest_path = Path(request["manifest_path"])
    header = _read(manifest_path)
    for proof in header["files"]:
        original = (manifest_path.parent / proof["path"]).resolve()
        if not original.is_relative_to(manifest_path.parent) or _sha(original) != proof["sha256"]:
            raise ValueError("online frozen fragment file changed")
    for proof in header["sources"]:
        if _sha(proof["path"]) != proof["sha256"]:
            raise ValueError("online original evidence changed")
    ending = header.get("ending_evidence", {})
    if ending.get("bootstrap_ref") and _sha(ending["bootstrap_ref"]) != ending.get("bootstrap_sha256"):
        raise ValueError("online bootstrap evidence changed")
    if code:
        if set(request["code_files"]) != set(CODE_FILES):
            raise ValueError("online worker code scope changed")
        for name, digest in request["code_files"].items():
            if _sha(Path(request["root"]) / name) != digest:
                raise ValueError(f"online worker code changed: {name}")


def _candidate(path, request):
    from .recurrent_ppo import load_checkpoint
    _verify(request, code=False)
    model, metadata = load_checkpoint(path, device="cpu")
    require_current_contract(metadata)
    report = metadata.get("ppo", {})
    if (metadata.get("online_request_sha256") != request["request_sha256"]
            or metadata.get("source_checkpoint_sha256") != request["source_sha256"]
            or metadata.get("online_fragment_manifest_sha256") != request["manifest_sha256"]
            or metadata.get("ppo_config") != request["ppo_config"]
            or report.get("source_version") != request["source_version"]
            or report.get("candidate_version") != model.policy_version()):
        raise ValueError("online candidate provenance mismatch")
    return model, {**report, "checkpoint": str(Path(path).resolve()), "checkpoint_reload_verified": True}


class OnlineJob(TrainingJob):
    def _result(self):
        path = self.directory / "result.json"
        if not path.exists():
            return None
        try:
            result = _read(path)
            if (result.get("request_sha256") != self.request["request_sha256"]
                    or _sha(result["checkpoint"]) != result["checkpoint_sha256"]):
                raise ValueError("online result/checkpoint changed")
            _, report = _candidate(result["checkpoint"], self.request)
            return {**report, "training_job": self.describe()}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as error:
            raise TrainingJobError(f"{error}; job={self.directory}; log={self.directory / 'worker.log'}") from error

    def poll(self):
        """Non-waiting result check; still disk/model work, never call in input gate."""
        if self.request.get("stop_file") and Path(self.request["stop_file"]).exists():
            raise TrainingCancelled(f"User stop; job={self.directory}")
        result = self._result()
        if result is not None:
            return result
        state = self.describe()
        if state.get("status") in ("failed", "cancelled", "timed_out"):
            raise TrainingJobError(f"Online worker {state['status']}: {state.get('error')}; log={state['worker_log']}")
        if self.process is not None and self.process.poll() is not None and not _occupied(self.directory / "optimizer.lock"):
            raise TrainingJobError(f"Online worker exited without result; log={state['worker_log']}")
        return None

    def load_candidate(self, current_version, *, device="cpu"):
        if current_version != self.request["source_version"]:
            raise TrainingJobError("stale online candidate: actor source version changed; preserve and exclude")
        report = self.poll()
        if report is None:
            return None
        if (type(report.get("optimizer_steps")) is not int or report["optimizer_steps"] < 1
                or report.get("final_kl_within_target") is not True
                or report.get("candidate_version") == current_version):
            raise TrainingJobError("online candidate failed update/KL gate; actor must keep frozen weights")
        model, _ = _candidate(report["checkpoint"], self.request)
        return model.to(device).eval(), report


def start_online_job(manifest_path, *, root, output, device="cpu", seed=0, stop_file=None):
    request = _request(manifest_path, root=root, output=output, device=device, seed=seed, stop_file=stop_file)
    directory = Path(output).resolve() / "online-jobs" / request["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive(directory / "launch.lock"):
        path = directory / "request.json"
        if path.exists():
            old = _read(path)
            if old != request:
                ignored = {"code_files", "request_sha256"}
                if ((directory / "result.json").exists()
                        and {k:v for k,v in old.items() if k not in ignored} == {k:v for k,v in request.items() if k not in ignored}):
                    return OnlineJob(directory, old)
                raise TrainingJobError("online job settings/code changed; fresh version fragment required")
        else:
            atomic_json(path, request, durable=True)
        job = OnlineJob(directory, request)
        state = job.describe()
        if ((directory / "result.json").exists() or _occupied(directory / "optimizer.lock")
                or state.get("status") == "launching" and time.time() - state.get("updated_at", 0) < STARTUP_GRACE_SECONDS):
            return job
        if stop_file and Path(stop_file).exists():
            atomic_json(directory / "state.json", {"status":"cancelled", "error":"user stop before launch"}, durable=True)
            return job
        attempt = int(state.get("attempt", 0)) + 1
        if (directory / "state.json").exists():
            atomic_json(directory / f"attempt-{attempt-1:04d}-state.json", state, durable=True)
        if (directory / "cancel.json").exists():
            os.replace(directory / "cancel.json", directory / f"attempt-{attempt-1:04d}-cancel.json")
        atomic_json(directory / "state.json", {"status":"launching", "attempt":attempt, "updated_at":time.time()}, durable=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env["PYTHONPATH"] = str(Path(root).resolve() / "src") + os.pathsep + env.get("PYTHONPATH", "")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        try:
            with (directory / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen([sys.executable, "-m", "playmodel.learning.online_ppo", "--job", str(directory)],
                    cwd=str(Path(root).resolve()), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    creationflags=flags, close_fds=True)
        except OSError as error:
            atomic_json(directory / "state.json", {"status":"failed", "error":str(error)}, durable=True)
            raise TrainingJobError(f"Online launch failed; log={directory / 'worker.log'}") from error
        return OnlineJob(directory, request, process)


def _train(request, directory, update):
    import torch
    from .recurrent_ppo import PPOConfig, ppo_update, save_checkpoint
    update(phase="online_evidence")
    model, metadata, chunks, flat, indices, manifest = _load_fragment(request["manifest_path"], device=request["device"])
    for path in sorted(directory.glob("candidate-*.pt")):
        try:
            _, report = _candidate(path, request)
            return report
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, TypeError, pickle.UnpicklingError):
            continue
    progress = {"steps":0, "started":None, "finished":None}
    class ObservedAdam(torch.optim.Adam):
        def step(self, closure=None):
            update(phase="online_optimization", optimizer_steps=progress["steps"])
            if progress["started"] is None:
                progress["started"] = time.perf_counter_ns()
            result = super().step(closure)
            progress["finished"] = time.perf_counter_ns()
            progress["steps"] += 1
            return result
    config = PPOConfig(**request["ppo_config"])
    started = time.perf_counter_ns()
    optimizer = ObservedAdam(model.parameters(), lr=config.learning_rate)
    report = ppo_update(model, chunks, config, optimizer=optimizer, trajectory=flat, transition_indices=indices)
    update(phase="online_save", optimizer_steps=progress["steps"])
    _verify(request)
    report.update(worker_pid=os.getpid(), optimization_started_at_ns=progress["started"],
        optimization_finished_at_ns=progress["finished"], training_elapsed_seconds=(time.perf_counter_ns()-started)/1e9,
        online_fragment=True, full_run_complete=False, phase_counts=manifest["phase_counts"],
        hidden_on_actor_apply="explicit_zero_reset", evaluation_training=False)
    path = directory / ("candidate-" + uuid.uuid4().hex + ".pt")
    save_checkpoint(model, path, {**metadata, **contract_fields(), "ppo":report, "ppo_config":asdict(config),
        "online_request_sha256":request["request_sha256"], "source_checkpoint_sha256":request["source_sha256"],
        "online_fragment_manifest":request["manifest_path"], "online_fragment_manifest_sha256":request["manifest_sha256"],
        "deployment_approved":False})
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
    return _candidate(path, request)[1]


def _run_job(directory):
    directory = Path(directory).resolve()
    request = _read(directory / "request.json")
    try:
        lock = _exclusive(directory / "optimizer.lock")
        lock.__enter__()
    except _Busy:
        return 0
    done, state_lock = threading.Event(), threading.RLock()
    started = time.monotonic()
    state = {"status":"running", "attempt":_read(directory / "state.json").get("attempt", 1), "worker_pid":os.getpid()}
    last = [0., None]
    def save(status=None, **values):
        with state_lock:
            state.update(values, updated_at=time.time())
            if status is not None:
                state["status"] = status
            atomic_json(directory / "state.json", state, durable=True)
    def stopped():
        return (directory / "cancel.json").exists() or bool(request.get("stop_file") and Path(request["stop_file"]).exists())
    def update(**values):
        if stopped():
            raise TrainingCancelled("user stop requested")
        if time.monotonic()-started >= request["max_seconds"]:
            raise TimeoutError("bounded online PPO job expired")
        if values.get("phase") != last[1] or time.monotonic()-last[0] >= 1:
            save(**values)
            last[:] = time.monotonic(), values.get("phase")
    def watchdog():
        while not done.wait(.1):
            timeout = time.monotonic()-started >= request["max_seconds"]
            if stopped() or timeout:
                if done.wait(CANCEL_GRACE_SECONDS):
                    return
                try:
                    save("timed_out" if timeout else "cancelled", error="online worker cancellation grace expired")
                finally:
                    os._exit(3 if timeout else 2)
    watcher = threading.Thread(target=watchdog, daemon=True)
    try:
        _verify(request)
        if OnlineJob(directory, request)._result() is not None:
            return 0
        save()
        watcher.start()
        update(phase="online_imports")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if os.name != "nt":
            os.nice(5)
        report = _train(request, directory, update)
        update(phase="online_result")
        _verify(request)
        atomic_json(directory / "result.json", dict(schema=SCHEMA, request_sha256=request["request_sha256"],
            checkpoint=report["checkpoint"], checkpoint_sha256=_sha(report["checkpoint"])), durable=True)
        save("completed", checkpoint=report["checkpoint"], finished_at_ns=time.perf_counter_ns())
        return 0
    except BaseException as error:
        save("cancelled" if isinstance(error, TrainingCancelled) else "timed_out" if isinstance(error, TimeoutError) else "failed",
             error=f"{type(error).__name__}: {error}")
        raise
    finally:
        done.set()
        if watcher.is_alive():
            watcher.join(.2)
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    raise SystemExit(_run_job(parser.parse_args().job))
