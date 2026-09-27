"""Durable, hidden PPO jobs. No capture, game input, network, or deployment.

The coordinator may detach. An OS lock, rather than a recorded PID, owns the
optimizer. A completed candidate is recovered before any new optimizer runs.
"""
from __future__ import annotations

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module("playmodel.learning.recurrent_training_worker")

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from playmodel.atomic_io import atomic_json
from .runtime_contract import RUNTIME_CONTRACT, PHASE_SCHEMA, contract_fields, require_current_contract

SCHEMA = "playmodel.recurrent-training-job.v1"
MAX_JOB_SECONDS = 1800
CANCEL_GRACE_SECONDS = 5
STARTUP_GRACE_SECONDS = 30
CODE_FILES = (
    "src/playmodel/learning/recurrent_training_worker.py",
    "src/playmodel/learning/recurrent_ppo.py", "src/playmodel/learning/full_run.py",
    "src/playmodel/learning/runtime_contract.py", "src/playmodel/atomic_io.py",
    "src/playmodel/execution_log.py",
)


class TrainingJobError(RuntimeError):
    pass


class TrainingCancelled(TrainingJobError):
    pass


class _Busy(TrainingJobError):
    pass


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def _exclusive(path):
    """Kernel ownership is released even if its process exits abruptly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise _Busy(f"Training job already owned: {path}") from error
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _occupied(path):
    try:
        with _exclusive(path):
            return False
    except _Busy:
        return True


def _settings(burn_in, seed):
    # Explicitly pin every optimizer/objective setting, including former defaults.
    return dict(epochs=4, minibatch_sequences=2, learning_rate=3e-4, clip_ratio=.2,
                value_clip=.2, value_coefficient=.5, entropy_coefficient=.01,
                max_grad_norm=.5, target_kl=.02, gamma=.997, gae_lambda=.95,
                discount_time_unit_seconds=1.0, burn_in=burn_in,
                normalize_advantages=True, seed=seed)


def _training_manifest(training):
    if any(training.get(key) is not True for key in ("full_run_complete", "training_eligible")) or training.get("split") != "train":
        raise ValueError("only a verified complete training run may update weights")
    require_current_contract(training)
    path = Path(training["manifest_path"]).resolve()
    manifest = _read(path)
    require_current_contract(manifest)
    if (manifest.get("split") != "train" or manifest.get("training_eligible") is not True
            or manifest.get("full_run_complete") is not True
            or (training.get("run_id") is not None and training["run_id"] != manifest.get("run_id"))):
        raise ValueError("training report differs from complete frozen training manifest")
    return path, manifest


def _request(checkpoint, training, *, root, output, device, seed, stop_file, wait_for_combat=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    checkpoint = Path(checkpoint).resolve()
    manifest_path, manifest = _training_manifest(training)
    normalized = {key: manifest[key] for key in ("run_id", "split", "training_eligible", "full_run_complete")}
    normalized.update(contract_fields(), manifest_path=str(manifest_path))
    body = dict(schema=SCHEMA, checkpoint=str(checkpoint), source_sha256=_sha(checkpoint),
                training=normalized, manifest_sha256=_sha(manifest_path), root=str(root), output=str(output),
                device=str(device), seed=seed, ppo_config=_settings(manifest["burn_in"], seed),
                code_files={name: _sha(root / name) for name in CODE_FILES},
                stop_file=str(Path(stop_file).resolve()) if stop_file else None,
                wait_for_combat=bool(wait_for_combat),
                max_seconds=MAX_JOB_SECONDS)
    # Settings/code changes conflict with the existing job, rather than allowing
    # another optimizer for the same frozen source and data in this output root.
    identity = {key: body[key] for key in ("schema", "source_sha256", "manifest_sha256")}
    identity["manifest_path"] = str(manifest_path)
    return {**body, "job_id": _digest(identity), "request_sha256": _digest(body)}


def _verify_request(request, *, verify_code=True):
    body = {key: value for key, value in request.items() if key not in ("job_id", "request_sha256")}
    if request.get("schema") != SCHEMA or _digest(body) != request.get("request_sha256"):
        raise ValueError("training request changed")
    identity = {key: request[key] for key in ("schema", "source_sha256", "manifest_sha256")}
    identity["manifest_path"] = request["training"]["manifest_path"]
    if request.get("job_id") != _digest(identity):
        raise ValueError("training request identity/code scope changed")
    if _sha(request["checkpoint"]) != request["source_sha256"]:
        raise ValueError("queued source checkpoint changed")
    if _sha(request["training"]["manifest_path"]) != request["manifest_sha256"]:
        raise ValueError("queued training manifest changed")
    if verify_code:
        if set(request["code_files"]) != set(CODE_FILES):
            raise ValueError("training request code scope changed")
        for name, expected in request["code_files"].items():
            if _sha(Path(request["root"]) / name) != expected:
                raise ValueError(f"queued training code changed: {name}")


def _notify(callback, **details):
    if callback is not None:
        callback(**details)


def train_candidate_once(checkpoint, training, *, output, device, seed, status_callback=None,
                         optimization_gate=None):
    """Same guarded PPO update as the synchronous cycle; never deploy weights."""
    import torch
    from .full_run import load_full_run
    from .recurrent_ppo import load_checkpoint, save_checkpoint, ppo_update, PPOConfig

    started = time.perf_counter_ns()
    manifest_path, header = _training_manifest(training)
    source_sha, manifest_sha = _sha(checkpoint), _sha(manifest_path)
    model, metadata = load_checkpoint(checkpoint, device=device)
    source_version = model.policy_version()
    if header.get("behavior_version") != source_version:
        raise ValueError("training manifest differs from source policy")
    _notify(status_callback, phase="training_evidence", training_manifest=str(manifest_path))
    chunks, flat, indices, manifest = load_full_run(manifest_path, device=device,
                                                 expected_runtime_contract=RUNTIME_CONTRACT)
    config = PPOConfig(**_settings(manifest["burn_in"], seed))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for target in sorted(output.glob("candidate-*/candidate.pt")):
        _notify(status_callback, phase="training_recovery", checkpoint=str(target))
        try:
            restored, saved = load_checkpoint(target, device="cpu")
            report = saved.get("ppo", {})
            if (Path(saved.get("full_run_manifest", "")).resolve() != manifest_path
                    or saved.get("runtime_contract") != RUNTIME_CONTRACT or saved.get("phase_schema") != PHASE_SCHEMA
                    or report.get("source_version") != source_version
                    or report.get("candidate_version") != restored.policy_version()
                    or (saved.get("full_run_manifest_sha256") is not None and saved["full_run_manifest_sha256"] != manifest_sha)
                    or (saved.get("source_checkpoint_sha256") is not None and saved["source_checkpoint_sha256"] != source_sha)
                    or (saved.get("ppo_config") is not None and saved["ppo_config"] != asdict(config))):
                continue
        except (OSError, ValueError, RuntimeError, EOFError, TypeError, KeyError, pickle.UnpicklingError):
            continue  # Preserve incomplete/mismatching originals; never overwrite.
        result = dict(report, checkpoint=str(target), checkpoint_reload_verified=True,
                      recovered_completed_update=True)
        if _sha(checkpoint) != source_sha or _sha(manifest_path) != manifest_sha:
            raise ValueError("training originals changed during candidate recovery")
        atomic_json(target.parent / "training-report.json", result, durable=True)
        return result

    progress = {"started": None, "finished": None, "steps": 0}

    class ObservedAdam(torch.optim.Adam):
        def step(self, closure=None):
            _notify(status_callback, phase="training", optimizer_steps=progress["steps"])
            if progress["started"] is None:
                progress["started"] = time.perf_counter_ns()
            result = super().step(closure)
            progress["finished"] = time.perf_counter_ns()
            progress["steps"] += 1
            return result

    gate = optimization_gate() if optimization_gate is not None else None
    _notify(status_callback, phase="training", training_manifest=str(manifest_path))
    optimizer = ObservedAdam(model.parameters(), lr=config.learning_rate)
    report = ppo_update(model, chunks, config, optimizer=optimizer, trajectory=flat, transition_indices=indices)
    _notify(status_callback, phase="training_save", optimizer_steps=progress["steps"])
    if _sha(checkpoint) != source_sha or _sha(manifest_path) != manifest_sha:
        raise ValueError("training originals changed during optimization")
    report.update(worker_pid=os.getpid(), optimization_started_at_ns=progress["started"],
                  optimization_finished_at_ns=progress["finished"],
                  training_elapsed_seconds=(time.perf_counter_ns() - started) / 1e9)
    if gate is not None:
        report["combat_gate"] = gate
    directory = output / ("candidate-" + uuid.uuid4().hex)
    directory.mkdir(exist_ok=False)
    target = directory / "candidate.pt"
    save_checkpoint(model, target, {**metadata, "full_run_manifest": str(manifest_path),
        "full_run_manifest_sha256": manifest_sha, "source_checkpoint_sha256": source_sha,
        **contract_fields(), "ppo_config": asdict(config), "ppo": report, "deployment_approved": False})
    with target.open("r+b") as stream:
        os.fsync(stream.fileno())
    restored, _ = load_checkpoint(target, device="cpu")
    if restored.policy_version() != model.policy_version():
        raise ValueError("new candidate reload differs from trained weights")
    report.update(checkpoint=str(target), checkpoint_reload_verified=True)
    atomic_json(directory / "training-report.json", report, durable=True)
    return report


class TrainingJob:
    def __init__(self, directory, request, process=None):
        self.directory, self.request, self.process = Path(directory), request, process

    def describe(self):
        state = _read(self.directory / "state.json") if (self.directory / "state.json").exists() else {"status": "pending"}
        return {**state, "job_id": self.request["job_id"], "job_directory": str(self.directory),
                "request_path": str(self.directory / "request.json"),
                "status_path": str(self.directory / "state.json"),
                "result_path": str(self.directory / "result.json"),
                "worker_log": str(self.directory / "worker.log"), "asynchronous": True}

    def release_for_combat(self, *, sent_at_ns, run_id, session_id):
        """Called only by the source evaluation's first actual transport receipt."""
        released = time.perf_counter_ns()
        if (type(sent_at_ns) is not int or not 0 < sent_at_ns <= released
                or not isinstance(run_id, str) or not run_id.strip()
                or not isinstance(session_id, str) or not session_id.strip()
                or run_id == self.request["training"]["run_id"]):
            raise ValueError("combat gate requires a current actual send in a distinct evaluation run")
        payload = dict(request_sha256=self.request["request_sha256"], sent_at_ns=sent_at_ns,
                       run_id=run_id, session_id=session_id, released_at_ns=released)
        with _exclusive(self.directory / "combat-gate.lock"):
            path = self.directory / "combat-gate.json"
            if path.exists():
                existing = _read(path)
                if any(existing.get(key) != payload[key] for key in ("request_sha256", "sent_at_ns", "run_id", "session_id")):
                    raise TrainingJobError(f"Combat gate already frozen for another actual send: {self.directory}")
                return existing
            atomic_json(path, payload, durable=True)
        return payload

    def _result(self):
        path = self.directory / "result.json"
        if not path.exists():
            return None
        envelope = _read(path)
        if envelope.get("request_sha256") != self.request["request_sha256"]:
            raise TrainingJobError(f"Worker result/request mismatch: {self.directory}")
        result = envelope["candidate"]
        if _sha(result["checkpoint"]) != envelope["checkpoint_sha256"]:
            raise TrainingJobError(f"Saved worker candidate changed: {self.directory}")
        # Results are consumed after source evaluation. Validate originals and the
        # checkpoint itself, not merely an editable JSON result envelope.
        try:
            _verify_request(self.request, verify_code=False)
            from .recurrent_ppo import load_checkpoint
            source, _ = load_checkpoint(self.request["checkpoint"], device="cpu")
            candidate, metadata = load_checkpoint(result["checkpoint"], device="cpu")
            saved_report = metadata.get("ppo", {})
            if (Path(metadata.get("full_run_manifest", "")).resolve()
                    != Path(self.request["training"]["manifest_path"])
                    or metadata.get("runtime_contract") != RUNTIME_CONTRACT
                    or metadata.get("phase_schema") != PHASE_SCHEMA
                    or saved_report.get("source_version") != source.policy_version()
                    or saved_report.get("candidate_version") != candidate.policy_version()
                    or result.get("source_version") != saved_report.get("source_version")
                    or result.get("candidate_version") != saved_report.get("candidate_version")
                    or any(result.get(key) != value for key, value in saved_report.items())
                    or metadata.get("source_checkpoint_sha256", self.request["source_sha256"]) != self.request["source_sha256"]
                    or metadata.get("full_run_manifest_sha256", self.request["manifest_sha256"]) != self.request["manifest_sha256"]
                    or metadata.get("ppo_config", self.request["ppo_config"]) != self.request["ppo_config"]):
                raise ValueError("candidate provenance differs from frozen training job")
        except (OSError, ValueError, RuntimeError, EOFError, TypeError, KeyError, pickle.UnpicklingError) as error:
            raise TrainingJobError(f"Invalid worker result: {error}; job={self.directory}; log={self.directory / 'worker.log'}") from error
        return dict(result, training_job=self.describe())

    def wait(self, timeout=None):
        deadline = time.monotonic() + (self.request["max_seconds"] + STARTUP_GRACE_SECONDS + 10 if timeout is None else timeout)
        while True:
            if self.request.get("stop_file") and Path(self.request["stop_file"]).exists():
                self.close(cancel=True)
                raise TrainingCancelled(f"User stop requested; job={self.directory}; log={self.directory / 'worker.log'}")
            result = self._result()
            if result is not None:
                return result
            state = self.describe()
            if state.get("status") in ("failed", "cancelled", "timed_out"):
                raise TrainingJobError(f"Training worker {state['status']}: {state.get('error')}; job={self.directory}; log={state['worker_log']}")
            if self.process is not None and self.process.poll() is not None:
                if _occupied(self.directory / "optimizer.lock"):
                    self.process = None  # A startup-race loser may attach to the owner.
                else:
                    raise TrainingJobError(f"Training worker exited {self.process.returncode} without a result; job={self.directory}; log={state['worker_log']}")
            if time.monotonic() >= deadline:
                from playmodel.execution_log import event
                event("recurrent_training_wait_timeout", job_directory=str(self.directory), error="bounded wait expired")
                raise TrainingJobError(f"Training wait timed out; job={self.directory}; log={state['worker_log']}")
            time.sleep(.1)

    def close(self, cancel=False):
        # Detach by default. Never terminate a recorded PID or an unrelated process.
        if not cancel or (self.directory / "result.json").exists():
            return
        atomic_json(self.directory / "cancel.json", {"requested_at": time.time(), "reason": "coordinator_cancel"}, durable=True)
        if self.process is not None:
            try:
                self.process.wait(timeout=CANCEL_GRACE_SECONDS + 3)
            except subprocess.TimeoutExpired:
                # Popen retains the original process handle; PID reuse is irrelevant.
                self.process.terminate()
                self.process.wait(timeout=3)


def start_training_job(checkpoint, training, *, root, output, device, seed, stop_file=None, wait_for_combat=False):
    request = _request(checkpoint, training, root=root, output=output, device=device, seed=seed,
                       stop_file=stop_file, wait_for_combat=wait_for_combat)
    directory = Path(output).resolve() / "training-jobs" / request["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive(directory / "launch.lock"):
        request_path = directory / "request.json"
        if request_path.exists():
            previous_request = _read(request_path)
            if previous_request != request:
                ancestry_fields = {"code_files", "request_sha256"}
                stable_old = {k: v for k, v in previous_request.items() if k not in ancestry_fields}
                stable_new = {k: v for k, v in request.items() if k not in ancestry_fields}
                if (directory / "result.json").exists() and stable_old == stable_new:
                    # The optimizer already finished under its recorded code.
                    # Consumption validates weights/data; it never re-optimizes.
                    return TrainingJob(directory, previous_request)
                raise TrainingJobError(f"Frozen training job settings/code changed; fresh training required: {directory}")
        else:
            atomic_json(request_path, request, durable=True)
        job = TrainingJob(directory, request)
        state = job.describe()
        if ((directory / "result.json").exists() or _occupied(directory / "optimizer.lock")
                or (state.get("status") == "launching" and time.time() - state.get("updated_at", 0) < STARTUP_GRACE_SECONDS)):
            return job
        if stop_file and Path(stop_file).exists():
            atomic_json(directory / "state.json", {"status": "cancelled", "error": "user stop before launch", "updated_at": time.time()}, durable=True)
            return job
        # Calling start again explicitly resumes a released job. Retain the
        # previous attempt; the child recovers any fully saved candidate first.
        attempt = int(state.get("attempt", 0)) + 1
        if (directory / "state.json").exists():
            atomic_json(directory / f"attempt-{attempt - 1:04d}-state.json", state, durable=True)
        if (directory / "cancel.json").exists():
            os.replace(directory / "cancel.json", directory / f"attempt-{attempt - 1:04d}-cancel.json")
        if (directory / "combat-gate.json").exists():
            os.replace(directory / "combat-gate.json", directory / f"attempt-{attempt - 1:04d}-combat-gate.json")
        atomic_json(directory / "state.json", {"status": "launching", "attempt": attempt, "updated_at": time.time()}, durable=True)
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        environment["PYTHONPATH"] = str(Path(root).resolve() / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        try:
            with (directory / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen([sys.executable, "-m", "playmodel.learning.recurrent_training_worker",
                    "--job", str(directory)], cwd=str(Path(root).resolve()), env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags, close_fds=True)
        except OSError as error:
            atomic_json(directory / "state.json", {"status": "failed", "error": f"launch:{error}", "updated_at": time.time()}, durable=True)
            raise TrainingJobError(f"Worker launch failed; job={directory}; log={directory / 'worker.log'}") from error
        return TrainingJob(directory, request, process)


def _run_job(directory):
    directory = Path(directory).resolve()
    request = _read(directory / "request.json")
    try:
        lock = _exclusive(directory / "optimizer.lock")
        lock.__enter__()
    except _Busy:
        return 0  # Another child owns this exact optimizer job.
    finished = threading.Event()
    state_lock = threading.RLock()
    started = time.monotonic()
    last_write, phase = [0.], [None]
    previous_state = _read(directory / "state.json") if (directory / "state.json").exists() else {}
    state = dict(status="running", attempt=previous_state.get("attempt", 1), worker_pid=os.getpid(),
                 started_at_ns=time.perf_counter_ns(), updated_at=time.time())

    def save_state(status=None, **details):
        with state_lock:
            state.update(details, updated_at=time.time())
            if status is not None:
                state["status"] = status
            atomic_json(directory / "state.json", state, durable=True)

    def cancelled():
        return ((directory / "cancel.json").exists()
                or bool(request.get("stop_file") and Path(request["stop_file"]).exists()))

    def update(**details):
        if cancelled():
            raise TrainingCancelled("user/coordinator stop requested")
        if time.monotonic() - started >= request["max_seconds"]:
            raise TimeoutError("bounded training job deadline exceeded")
        if details.get("phase") != phase[0] or time.monotonic() - last_write[0] >= 1:
            save_state(**details)
            phase[0], last_write[0] = details.get("phase"), time.monotonic()

    def watchdog():
        while not finished.wait(.1):
            timed_out = time.monotonic() - started >= request["max_seconds"]
            if cancelled() or timed_out:
                if finished.wait(CANCEL_GRACE_SECONDS):
                    return
                try:
                    save_state("timed_out" if timed_out else "cancelled", error="worker stopped after cancellation grace")
                finally:
                    # Only this worker exits; OS-owned lock releases automatically.
                    os._exit(3 if timed_out else 2)

    def optimization_gate():
        if not request.get("wait_for_combat"):
            return None
        path = directory / "combat-gate.json"
        while True:
            update(phase="waiting_for_combat")
            if path.exists():
                gate = _read(path)
                if (gate.get("request_sha256") != request["request_sha256"]
                        or type(gate.get("sent_at_ns")) is not int
                        or type(gate.get("released_at_ns")) is not int
                        or not 0 < gate["sent_at_ns"] <= gate["released_at_ns"] <= time.perf_counter_ns()
                        or not isinstance(gate.get("run_id"), str) or not gate["run_id"].strip()
                        or gate["run_id"] == request["training"]["run_id"]
                        or not isinstance(gate.get("session_id"), str) or not gate["session_id"].strip()):
                    raise ValueError("combat gate differs from frozen job or actual-send contract")
                return gate
            time.sleep(.1)

    watcher = threading.Thread(target=watchdog, daemon=True, name="ppo-job-watchdog")
    try:
        _verify_request(request)
        existing = TrainingJob(directory, request)._result()
        if existing is not None:
            return 0
        save_state()
        watcher.start()
        update(phase="training_imports")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if os.name != "nt":
            os.nice(5)
        candidate = train_candidate_once(request["checkpoint"], request["training"],
            output=request["output"], device=request["device"], seed=request["seed"], status_callback=update,
            optimization_gate=optimization_gate)
        _verify_request(request)
        update(phase="training_result")
        envelope = {"schema": SCHEMA, "request_sha256": request["request_sha256"],
                    "checkpoint_sha256": _sha(candidate["checkpoint"]), "candidate": candidate}
        atomic_json(directory / "result.json", envelope, durable=True)
        save_state("completed", checkpoint=candidate["checkpoint"], finished_at_ns=time.perf_counter_ns())
        return 0
    except BaseException as error:
        status = "cancelled" if isinstance(error, TrainingCancelled) else "timed_out" if isinstance(error, TimeoutError) else "failed"
        save_state(status, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        finished.set()
        if watcher.is_alive():
            watcher.join(.2)
        lock.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    return _run_job(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
