"""Local neural full-run collection, separate PPO, and fresh-run comparison.

Character/initial weapon/highest-difficulty setup remains the verified bootstrap
controller. Movement, upgrade and supported shop decisions use one frozen neural
policy and shared recurrent state. No model is automatically approved/deployed.
Only an explicit CLI call starts game input; importing this file does not.
"""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from playmodel.learning.runtime_contract import (RUNTIME_CONTRACT, PHASE_SCHEMA,
    contract_fields, contract_identity, require_current_contract)


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _compact_cycle(report, path):
    def run_fields(run):
        return {key: run.get(key) for key in ("run_id", "directory", "manifest_path", "split", "role",
            "training_eligible", "full_run_complete", "steps", "phase_counts", "verified_wave_clears",
            "stop_category", "error", "runtime_contract", "phase_schema", "evaluation_scope_id")}
    return {"status": report["status"], "report_path": str(path),
            "source_checkpoint": report.get("source_checkpoint"),
            "candidate_checkpoint": report.get("candidate", {}).get("checkpoint"),
            "training": run_fields(report.get("training", {})),
            "evaluations": [run_fields(run) for run in report.get("evaluations", [])],
            "mean_verified_wave_clears": report.get("mean_verified_wave_clears"),
            "error": report.get("error"), "deployment_approved": False,
            "runtime_contract": report.get("runtime_contract"), "phase_schema": report.get("phase_schema"),
            "evaluation_scope_id": report.get("evaluation_scope_id"),
            "training_execution": report.get("training_execution"),
            "training_overlap": report.get("training_overlap")}


def _bind_evaluation_scope(report, runtime_source_hashes):
    """Restart only comparisons on semantic/code changes; retain frozen ancestry."""
    result = deepcopy(report)
    changed = (result.get('runtime_contract') != RUNTIME_CONTRACT
               or result.get('phase_schema') != PHASE_SCHEMA
               or result.get('runtime_source_hashes') != runtime_source_hashes)
    if changed:
        if result.get('evaluations') or result.get('pending') or result.get('evaluation_scope_id'):
            result.setdefault('excluded_evaluation_scopes', []).append({
                'evaluation_scope_id': result.get('evaluation_scope_id'),
                'runtime_contract': result.get('runtime_contract'),
                'phase_schema': result.get('phase_schema'),
                'runtime_source_hashes': result.get('runtime_source_hashes'),
                'evaluations': deepcopy(result.get('evaluations', [])),
                'pending': deepcopy(result.get('pending')),
                'reason': 'runtime_or_phase_contract_changed'})
        result.setdefault('excluded_evaluations', []).extend(result.get('evaluations', []))
        result.update(evaluations=[], pending=None, pipeline_phase='evaluation_contract_changed',
                      evaluation_scope_id=uuid.uuid4().hex)
    result.update(**contract_fields(), runtime_source_hashes=deepcopy(runtime_source_hashes))
    result.setdefault('evaluation_scope_id', uuid.uuid4().hex)
    return result


class LocalStatus:
    """A process heartbeat is not a claim that gameplay is making progress."""
    def __init__(self, directory, interval_seconds=5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._heartbeat_error = None
        self.online_provider = None
        self.interval = interval_seconds
        self.state = {"process_id": os.getpid(), "status": "starting", "phase": "initializing",
                      "automatic_agent_call": False, "needs_agent_help": False,
                      "deployment_approved": False, "last_progress_monotonic_ns": time.perf_counter_ns()}
        self.update()
        self.thread = threading.Thread(target=self._heartbeat, name="local-cycle-status", daemon=True)
        self.thread.start()

    def update(self, *, progress=True, **values):
        with self.lock:
            if progress and self._heartbeat_error is not None:
                error, self._heartbeat_error = self._heartbeat_error, None
                raise error
            tracked = ('phase', 'status', 'run_id', 'error')
            previous = tuple(self.state.get(key) for key in tracked)
            state = {**self.state, **values}
            training_status = (state.get('training_job') or {}).get('status_path')
            if training_status:
                try:
                    worker = json.loads(Path(training_status).read_text(encoding='utf-8'))
                    state['background_training'] = {key: worker.get(key) for key in (
                        'status', 'phase', 'worker_pid', 'optimizer_steps', 'updated_at', 'error')}
                except (OSError, ValueError):
                    # The child may still be creating its first status file.
                    # Job/result verification remains at the episode boundary.
                    pass
            if progress:
                state["last_progress_monotonic_ns"] = time.perf_counter_ns()
            state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
            state["heartbeat_monotonic_ns"] = time.perf_counter_ns()
            from playmodel.atomic_io import atomic_json
            atomic_json(self.directory / 'status.json', state)
            self.state = state
            if previous != tuple(state.get(key) for key in tracked):
                from playmodel.execution_log import event
                event('runtime_status', status_path=str(self.directory / 'status.json'), **state)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                online = self.online_provider() if self.online_provider else None
                self.update(progress=False, **({'online_learning': online} if online is not None else {}))
            except (OSError, ValueError, RuntimeError) as error:
                # A persistent background write error remains visible even if
                # its cause clears before the next foreground progress update.
                with self.lock:
                    self._heartbeat_error = error
                from playmodel.execution_log import event
                event('runtime_status_write_failed', status_path=str(self.directory / 'status.json'),
                      error=f'{type(error).__name__}: {error}', process_id=os.getpid())
                return

    def fault(self, message, **details):
        self.update(status="fault_paused", phase="stopped", needs_agent_help=True,
                    error=str(message), stop_category="runtime_error", **details)
        target = self.directory / "help-request.json"
        history = self.directory / 'help-requests'
        history.mkdir(exist_ok=True)
        if target.exists():
            # Preserve legacy requests too; the latest pointer must describe
            # this failure instead of the first failure in a resumed cycle.
            (history / f'previous-{uuid.uuid4().hex}.json').write_bytes(target.read_bytes())
        request = {"reason": str(message), "details": details,
            "created_at": datetime.now(timezone.utc).isoformat(), "process_id": os.getpid(),
            "automatic_agent_call": False, "agent_action_required": "inspect diagnostics, fix, test, explicitly restart",
            "status_path": str(self.directory / "status.json"), "restart_attempted": False}
        _save_json(history / f'fault-{uuid.uuid4().hex}.json', request)
        _atomic_json(target, request)

    def close(self):
        self.stop.set()
        self.thread.join(2)


def run_pipeline(source_checkpoint, *, collect_run, train_candidate, evaluation_runs=1,
                 resume_report=None, on_progress=None, runtime_source_hashes=None,
                 start_training=None):
    """Persist phase boundaries; reuse frozen training, never train evaluation.

    Interrupted/incomplete evaluation runs are retained separately and replaced
    with fresh physical runs. A callback persists intent before each side effect
    and its result immediately afterwards.
    """
    if type(evaluation_runs) is not int or not 1 <= evaluation_runs <= 5:
        raise ValueError("evaluation_runs must be 1..5 per policy")
    result = deepcopy(resume_report) if resume_report else {
        "source_checkpoint": str(source_checkpoint), "evaluations": [],
        "excluded_evaluations": [], "deployment_approved": False,
        "performance_improvement_verified": False}
    result = _bind_evaluation_scope(result, runtime_source_hashes if runtime_source_hashes is not None
                                     else result.get('runtime_source_hashes', {}))
    if Path(result['source_checkpoint']).resolve() != Path(source_checkpoint).resolve():
        raise ValueError('resume source checkpoint differs from frozen cycle')
    if result.get('evaluation_runs', evaluation_runs) != evaluation_runs:
        raise ValueError('evaluation protocol cannot change inside a resumed cycle')
    result['evaluation_runs'] = evaluation_runs
    # A resumed attempt has its own outcome. Previous failure reports remain
    # archived in the cycle journal, not attached to an unrelated new failure.
    result.pop('error', None)

    def progress(phase, **pending):
        if phase in ('collect_training', 'collect_evaluation'):
            pending.update(**contract_fields(), evaluation_scope_id=result['evaluation_scope_id'])
        previous_phase, previous_pending = result.get('pipeline_phase'), result.get('pending') or {}
        if phase in ('collect_training', 'collect_evaluation'):
            same = previous_phase == phase and all(previous_pending.get(key) == value for key, value in pending.items())
            pending['operation_id'] = previous_pending.get('operation_id') if same else uuid.uuid4().hex
            pending['operation_id'] = pending['operation_id'] or uuid.uuid4().hex
        result['pipeline_phase'] = phase
        result['pending'] = pending or None
        if on_progress:
            on_progress(deepcopy(result))

    training = result.get('training')
    if not training or not training.get('training_eligible') or not training.get('full_run_complete'):
        if training:
            result.setdefault('excluded_training_runs', []).append(training)
        progress('collect_training', split='train', tag='training', checkpoint=str(source_checkpoint))
        training = collect_run(source_checkpoint, split="train", tag="training")
        require_current_contract(training)
        result['training'] = training
        progress('training_collected')
    if not training.get("training_eligible") or not training.get("full_run_complete"):
        result.update(status="training_run_incomplete_or_rejected",
                      error=training.get('error') or 'Training run incomplete or rejected')
        progress('stopped')
        return result
    seen_runs = {training["run_id"]}
    seen_sessions = set(training.get("session_ids", []))
    retained = []
    for row in result.get('evaluations', []):
        if (row.get('full_run_complete') and row.get('setup_conditions') == training.get('setup_conditions')
                and contract_identity(row) == (RUNTIME_CONTRACT, PHASE_SCHEMA)
                and row.get('evaluation_scope_id') == result['evaluation_scope_id']
                and row.get('runtime_source_hashes') == result['runtime_source_hashes']):
            retained.append(row)
        else:
            result.setdefault('excluded_evaluations', []).append(row)
    result['evaluations'] = retained
    for row in retained + result.get('excluded_evaluations', []):
        if row['run_id'] in seen_runs or set(row.get('session_ids', [])) & seen_sessions:
            raise ValueError('training/evaluation physical run or session overlap')
        seen_runs.add(row['run_id'])
        seen_sessions.update(row.get('session_ids', []))

    def collect_evaluation(index, role, checkpoint):
        progress('collect_evaluation', split='evaluation', tag=f'evaluation-{index}-{role}',
                 checkpoint=str(checkpoint), role=role, evaluation_index=index)
        begin = time.perf_counter_ns()
        evaluation = collect_run(checkpoint, split="evaluation", tag=f"evaluation-{index}-{role}")
        end = time.perf_counter_ns()
        require_current_contract(evaluation)
        if evaluation.get('runtime_source_hashes') != result['runtime_source_hashes']:
            raise ValueError('evaluation runtime sources differ from frozen comparison scope')
        if evaluation.get('evaluation_scope_id') not in (None, result['evaluation_scope_id']):
            raise ValueError('evaluation belongs to another comparison scope')
        sessions = set(evaluation.get("session_ids", []))
        if evaluation["run_id"] in seen_runs or sessions & seen_sessions:
            raise ValueError("training/evaluation physical run or session overlap")
        seen_runs.add(evaluation["run_id"])
        seen_sessions.update(sessions)
        result["evaluations"].append({**evaluation, "role": role, 'evaluation_index': index,
            'evaluation_scope_id': result['evaluation_scope_id'],
            'collection_started_at_ns': begin, 'collection_finished_at_ns': end})
        progress('evaluation_collected')
        if evaluation.get("setup_conditions") != training.get("setup_conditions"):
            result.update(status="evaluation_setup_mismatch")
            progress('stopped')
            return False
        if not evaluation.get("full_run_complete"):
            result.update(status="evaluation_run_incomplete",
                          error=evaluation.get('error') or 'Evaluation run incomplete')
            progress('stopped')
            return False
        return True

    candidate = result.get('candidate')
    if candidate is None:
        # Old training remains provenance for an already saved candidate. It is
        # never silently replayed into the newly interpreted choice policy.
        require_current_contract(training)
        pending_source = (start_training is not None and result.get('pipeline_phase') == 'collect_evaluation'
                          and (result.get('pending') or {}).get('role') == 'source'
                          and (result.get('pending') or {}).get('evaluation_index', 0) == 0)
        if not pending_source:
            progress('train_candidate', training_manifest=training.get('manifest_path'),
                     checkpoint=str(source_checkpoint))
        if start_training is None:
            candidate = train_candidate(source_checkpoint, training)
            result['training_execution'] = 'synchronous'
        else:
            job = start_training(source_checkpoint, training)
            result['training_job'] = job.describe()
            result['training_execution'] = 'separate_local_process'
            try:
                if pending_source:
                    # Preserve the collector operation ID across a crash after
                    # physical collection but before its result was journaled.
                    if on_progress:
                        on_progress(deepcopy(result))
                else:
                    progress('training_worker_started', training_job=result['training_job'])
                if not any(row.get('role') == 'source' and row.get('evaluation_index', 0) == 0
                           for row in retained):
                    if not collect_evaluation(0, 'source', source_checkpoint):
                        return result
                # A candidate never enters an active run. Wait only after this
                # fixed-source evaluation has ended and released game input.
                progress('await_training_worker', training_job=result['training_job'])
                candidate = job.wait()
            finally:
                # A finite job may finish after a collection failure; its
                # durable result is recovered without another update on resume.
                job.close()
        result["candidate"] = candidate
        progress('candidate_saved')
    if candidate.get("final_kl_within_target") is not True:
        result.update(status="candidate_failed_numerical_gate")
        progress('stopped')
        return result
    for index in range(evaluation_runs):
        # Alternate ordering across repetitions; this is not controlled game RNG.
        policies = [("source", source_checkpoint), ("candidate", candidate["checkpoint"])]
        if index % 2:
            policies.reverse()
        for role, checkpoint in policies:
            if any(row.get('role') == role and row.get('evaluation_index', 0) == index for row in retained):
                continue
            if not collect_evaluation(index, role, checkpoint):
                return result
    optimization_start = candidate.get('optimization_started_at_ns')
    optimization_end = candidate.get('optimization_finished_at_ns')
    if isinstance(optimization_start, int) and isinstance(optimization_end, int):
        sources = [row for row in result['evaluations'] if row['role'] == 'source']
        def overlap(begin, end):
            return max(0, min(end, optimization_end) - max(begin, optimization_start)) / 1e9
        result['training_overlap'] = {
            'source_collection_seconds': sum(overlap(row.get('collection_started_at_ns', 0),
                row.get('collection_finished_at_ns', 0)) for row in sources),
            'actual_movement_seconds': sum(overlap(begin, end) for row in sources
                for begin, end in row.get('combat_intervals_ns', [])),
            'scope': 'intersection_with_optimizer_interval; movement_uses_actual_transmission_bounds'}
    means = {}
    for role in ("source", "candidate"):
        scores = [row["verified_wave_clears"] for row in result["evaluations"] if row["role"] == role]
        means[role] = sum(scores) / len(scores)
    result.update(status="comparison_recorded", mean_verified_wave_clears=means,
                  game_rng_seed_controlled=False,
                  comparison_limit="few uncontrolled full runs; no promotion or causal item-effect claim")
    progress('complete')
    return result


class LocalCycle:
    def __init__(self, *, root, output, character_slot, weapon, max_run_seconds,
                 device="cuda", seed=0, status=None, recover_active_run=False,
                 menu_factory=None, terminal_callback=None, tactical_factory=None):
        import torch
        from playmodel.games.brotato.installation import inspect_installation
        self.torch = torch
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.character_slot, self.weapon = character_slot, weapon
        self.max_run_seconds, self.device, self.seed = max_run_seconds, device, seed
        self.status = status
        self.recover_active_run = recover_active_run
        self.menu_factory = menu_factory
        self.terminal_callback = terminal_callback
        self.tactical_factory = tactical_factory
        self.operation_id = None
        self.evaluation_scope_id = None
        self._training_job = None
        self.runtime_source_hashes = _runtime_contract(self.root)
        self.stop_file = self.root / "artifacts/BROTATO_STOP"
        self.ocr_script = self.root / "scripts/windows_ocr.ps1"
        installation = next((item for item in inspect_installation()["installations"]
                             if item["status"] == "files_present"), None)
        if installation is None:
            raise OSError("Brotato installation not found")
        self.executable = Path(installation["path"]) / "Brotato.exe"

    def _current_scene(self, directory):
        from playmodel.games.brotato.menu_capture import MenuCapture
        from playmodel.games.brotato.ocr import MenuOcr
        from playmodel.games.brotato.menu import classify_scene
        from playmodel.games.brotato.vision import BrotatoVision
        from playmodel.games.brotato.setup_run import recognize_main_menu
        with MenuCapture(self.executable) as capture, MenuOcr(self.ocr_script) as ocr:
            for _ in range(3):
                if self.stop_file.exists():
                    raise OSError('User stop file exists')
                shot, pixels, width, height = capture.read(directory)
                source = Path(shot['session_directory']) / 'frame.png'
                raw = ocr.read(source)
                _save_json(source.parent / 'startup-ocr.json', raw)
                scene = classify_scene(raw, width=width, height=height).scene
                if recognize_main_menu(ocr,source,raw,pixels,width,height):
                    return 'main_menu'
                if scene != 'unknown':
                    return scene
                from playmodel.games.brotato.menu import rows_in_region
                header=''.join(rows_in_region(raw,(500,60,1450,160))).casefold()
                if 'characterselection' in header:
                    return 'character_selection'
                back=''.join(rows_in_region(raw,(20,20,300,100))).casefold()
                weapon_text=''.join(rows_in_region(raw,(1180,180,1550,460))).casefold()
                if 'back' in back and 'damage' in weapon_text and 'range' in weapon_text:
                    return 'weapon_selection'
                # Use the original capture; BrotatoVision bounds its own grid.
                # A second downsample can create artificial player ambiguity.
                vision = BrotatoVision().observe(pixels, width, height,
                                                 observed_at_ns=shot['capture_started_at_ns'])
                if vision.combat_likely and vision.player is not None:
                    return 'combat'
                from playmodel.execution_log import event
                event('startup_scene_unrecognized', frame_path=str(source), scene=scene,
                      vision_status=vision.status, combat_likely=vision.combat_likely,
                      player=vision.player)
        raise OSError('Unrecognized startup screen after three observations; no action')

    def _new_run(self, directory, checkpoint):
        from playmodel.games.brotato.session import run_session
        from playmodel.games.brotato.setup_run import prepare_next
        if self.stop_file.exists():
            raise OSError("User stop file exists")
        scene = self._current_scene(directory / 'initial-state')
        if scene in ('combat', 'level_up', 'shop', 'pause', 'loot') and self.recover_active_run:
            recovery = self.collect_run(checkpoint, split='evaluation', tag='partial-recovery', partial=True)
            _save_json(directory / 'active-run-recovery.json', recovery)
            if not recovery.get('recovery_completed'):
                raise OSError('Active run recovery did not reach a verified normal ending')
            scene = self._current_scene(directory / 'after-recovery')
        if scene == "death":
            def no_combat(*args, **kwargs):
                raise OSError("Unexpected combat while acknowledging the old death screen")
            acknowledged = run_session(self.executable, directory / "old-result", waves=1,
                seconds=30, stop_file=self.stop_file, ocr_script=self.ocr_script,
                record=True, edit=False, combat_runner=no_combat)
            if acknowledged["reason"] != "run_finished":
                raise OSError("Old death did not reach a verified result screen")
        elif scene not in ("result", "difficulty", "main_menu", "character_selection", "weapon_selection"):
            raise OSError("A new cycle requires a result/death screen; the active run is preserved")
        setup = prepare_next(self.executable, root=self.root, character_slot=self.character_slot,
                               weapon=self.weapon, record=True,
                               **({'concept': self.campaign_concept} if getattr(self, 'campaign_concept', None) else {}))
        context = setup.get("context", {})
        if (setup.get("error") or not context.get("setup_complete")
                or not all(context.get(key) for key in ("character_source", "weapon_source", "difficulty_menu_source"))):
            raise OSError("Fresh character/weapon/difficulty setup was not fully verified")
        frame = Path(context["difficulty_menu_source"]).resolve()
        metadata = json.loads((frame.parent / "observation.json").read_text(encoding="utf-8"))
        start = {"kind": "new_run_setup", "frame_ref": str(frame),
                 "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                 "observed_at_ns": metadata["capture_started_at_ns"],
                 "available_at_ns": metadata.get("available_at_ns", metadata["capture_started_at_ns"]),
                 "verified_at_ns": time.perf_counter_ns(), "verified": True,
                 "independent_of_policy": True, "origin": "local_verifier",
                 "verifier_id": "fresh_result_character_weapon_difficulty_transition_v1",
                 "setup_directory": setup["directory"], "selection_policy": "bootstrap_rules_not_learned"}
        _save_json(directory / "setup-evidence.json", {"setup": setup, "start": start})
        return context, start

    def _training_combat_callback(self, *, run_id, split, tag, partial):
        job = getattr(self, '_training_job', None)
        if (job is None or getattr(self, '_training_gate_released', False) or partial
                or split != 'evaluation' or tag != 'evaluation-0-source'):
            return None

        def release_training(record):
            job.release_for_combat(sent_at_ns=record['sent_at_ns'], run_id=run_id,
                                   session_id=record['session_id'])
            self._training_gate_released = True
        return release_training

    def collect_run(self, checkpoint, *, split, tag, partial=False):
        if partial and split != 'evaluation':
            raise ValueError('partial recovery is excluded from training and evaluation scores')
        if split == 'train':
            self._training_job = None
        from playmodel.learning.recurrent_ppo import load_checkpoint
        from playmodel.learning.full_run import FullRunRecorder
        from playmodel.games.brotato.neural_runtime import run_neural_trial, safe_observation_transition
        from playmodel.games.brotato.neural_runtime import _safe_recovery_release
        from playmodel.games.brotato.neural_menu_controller import NeuralMenuController
        from playmodel.games.brotato.session import run_session
        operation_id = self.operation_id if not partial else None
        if _runtime_contract(self.root) != self.runtime_source_hashes:
            raise ValueError('runtime sources changed after collector startup')
        if operation_id:
            for marker in self.output.glob(f'{tag}-*/run-operation.json'):
                operation = json.loads(marker.read_text(encoding='utf-8'))
                report_path = marker.parent / 'cycle-run.json'
                if operation.get('operation_id') != operation_id or not report_path.is_file():
                    continue
                recovered = json.loads(report_path.read_text(encoding='utf-8'))
                require_current_contract(operation)
                require_current_contract(recovered)
                if (operation.get('partial') is not False or operation.get('split') != split
                        or operation.get('runtime_source_hashes') != self.runtime_source_hashes
                        or recovered.get('runtime_source_hashes') != self.runtime_source_hashes
                        or operation.get('evaluation_scope_id') != self.evaluation_scope_id
                        or recovered.get('evaluation_scope_id') != self.evaluation_scope_id
                        or Path(operation['checkpoint']).resolve() != Path(checkpoint).resolve()
                        or recovered['split'] != split or recovered.get('recovery_only')
                        or Path(recovered['checkpoint']).resolve() != Path(checkpoint).resolve()):
                    raise ValueError('completed collection differs from persisted operation')
                if recovered.get('manifest_path') and recovered.get('full_run_complete'):
                    from playmodel.learning.full_run import load_full_run
                    load_full_run(recovered['manifest_path'], require_training=False,
                                  expected_runtime_contract=RUNTIME_CONTRACT)
                return recovered
        run_id = f"{tag}-{uuid.uuid4().hex}"
        directory = self.output / run_id
        directory.mkdir(parents=True, exist_ok=False)
        _save_json(directory / 'run-operation.json', {'operation_id': operation_id,
                   'checkpoint': str(Path(checkpoint).resolve()), 'split': split, 'partial': partial,
                   **contract_fields(), 'runtime_source_hashes': self.runtime_source_hashes,
                   'evaluation_scope_id': self.evaluation_scope_id})
        if self.status:
            self.status.update(status="running", phase="new_run_setup", run_id=run_id, split=split,
                               checkpoint=str(checkpoint), run_directory=str(directory))
        model, _ = load_checkpoint(checkpoint, device="cpu")
        context, start = ({'partial_recovery': True}, None) if partial else self._new_run(directory, checkpoint)
        if self.status:
            # Nested partial recovery may have replaced the displayed run. The
            # fresh recorder below belongs to this verified new-run boundary.
            self.status.update(status="running", phase="new_run_setup", run_id=run_id, split=split,
                               checkpoint=str(checkpoint), run_directory=str(directory),
                               recorded_transitions=0, last_combat_outcome=None)
        recorder = FullRunRecorder(model, run_id, split=split, start_evidence=start)
        if recorder.build_state is not None and not partial and context.get('weapon_source'):
            weapon_source = Path(context['weapon_source']).resolve()
            weapon_observation = json.loads((weapon_source.parent / 'observation.json').read_text(encoding='utf-8'))
            proof_time = time.perf_counter_ns()
            for weapon_name in context.get('weapons', []):
                recorder.build_state.seed_weapon(weapon_name, {
                    'verified': True, 'independent_of_policy': True,
                    'frame_ref': str(weapon_source),
                    'frame_sha256': hashlib.sha256(weapon_source.read_bytes()).hexdigest(),
                    'observed_at_ns': weapon_observation['capture_started_at_ns'],
                    'available_at_ns': proof_time, 'verified_at_ns': proof_time,
                    'origin': 'verified_initial_weapon_setup'})
        choice_backend = getattr(self, 'menu_factory', None) if not partial else None
        tactical_factory = getattr(self, 'tactical_factory', None) if not partial else None
        if tactical_factory and not choice_backend:
            raise ValueError('tactical combat requires the separate local choice menu controller')
        if (choice_backend or tactical_factory) and getattr(self, 'online_factory', None):
            raise ValueError('External menu learning cannot share online CNN PPO')
        menus = (choice_backend or NeuralMenuController)(
            recorder, output_directory=directory / "macro-actions", seed=self.seed)
        tactical_actor = tactical_factory(recorder, output_directory=directory / 'tactics', seed=self.seed) if tactical_factory else None
        def collection_errors():
            from playmodel.games.brotato.laya_menu import CNN_EXCLUSION
            return [reason for reason in recorder.rejection_reasons
                    if not (choice_backend and reason in (CNN_EXCLUSION, 'mixed_control_excluded_from_cnn_ppo'))]
        online = None
        if split == 'train' and not partial and getattr(self, 'online_factory', None):
            online = self.online_factory(recorder, checkpoint=checkpoint, root=self.root,
                output=directory / 'online', device=self.device, seed=self.seed, stop_file=self.stop_file)

            def online_progress():
                snapshot = {**online.snapshot(), 'run_id': run_id}
                callback = getattr(self, 'online_progress', None)
                if callback:
                    callback(snapshot)
                return snapshot

            if self.status:
                self.status.online_provider = online_progress
        trials, segments = [], []
        death_evidence = None
        started = time.perf_counter()
        observation_wait_seconds = 0.0
        observation_gaps = []
        exclude_next_terminal = False

        def observation_gap(proof):
            nonlocal recorder, exclude_next_terminal
            if proof['reason'] == 'observation_resumed':
                # A verified menu pair starts a new decision interval. Combat
                # resumed within a wave still has an unobserved action gap.
                exclude_next_terminal = proof['scene'] == 'combat'
                observation_gaps.append(proof)
                return
            if menus.pending_decision is not None or menus.awaiting_application:
                menus.abort('Observation recovery encountered an unresolved menu decision')
            if choice_backend:
                menus.discard_interrupted_outcome()
            previous_build = deepcopy(recorder.build_state)
            recorder.abort(directory / ('observation-history-' + uuid.uuid4().hex),
                           'released observation gap excluded from learning and evaluation')
            recorder = FullRunRecorder(model, run_id, split=split)
            recorder.build_state = previous_build
            if choice_backend:
                from playmodel.games.brotato.laya_menu import CNN_EXCLUSION
                recorder.invalidate(CNN_EXCLUSION)
            menus.recorder = recorder
            exclude_next_terminal = True
            observation_gaps.append(proof)

        def observation_status(phase):
            if self.status:
                self.status.update(status='running', phase=phase,
                                   recorded_transitions=len(recorder.records))

        def combat_runner(executable, output_root, *, policy=None, config, **kwargs):
            nonlocal death_evidence, recorder, exclude_next_terminal
            kwargs.pop("vision_factory", None)
            first_action = self._training_combat_callback(run_id=run_id, split=split, tag=tag, partial=partial)
            if first_action is not None:
                kwargs['first_action_callback'] = first_action
            if online is not None:
                kwargs['online_session'] = online
            if choice_backend:
                kwargs['mixed_control'] = True
            if tactical_actor is not None:
                kwargs['combat_actor'] = tactical_actor
            if online is not None or partial or choice_backend:
                kwargs.update(scheduling_recovery=True, recovery_only=partial)
            if self.status:
                self.status.update(phase="combat", recorded_transitions=len(recorder.records))
            result = run_neural_trial(executable, output_root, model=recorder.model,
                initial_hidden=recorder.hidden, reset_first=not recorder.records,
                build_state=recorder.build_state,
                config=replace(config, train=False, defer_training=False, seed=self.seed),
                split=split, chunk_steps=32, burn_in=8, **kwargs)
            trials.append(result)
            result["training_performed"] = False
            if choice_backend and not result.get('observation_transition'):
                released = _safe_recovery_release(result)
                if released is not None and not self.stop_file.exists():
                    # Exhausted timed retries become an input-free UI boundary.
                    # The next scene still needs two fresh observations; this
                    # interrupted interval never supplies an outcome reward.
                    observation_gap({'reason': 'released_controller_guard', **released})
                    result.update(status='observation_wait', observation_transition={**released, 'blocked_scene': 'combat'},
                        choice_learning={'status': 'excluded', 'reason': 'released_controller_guard'})
                    observation_status('waiting_observation')
                    return result
            if (choice_backend or partial) and result.get('observation_transition'):
                transition = safe_observation_transition(result)
                if transition is None or transition != result['observation_transition']:
                    raise ValueError('Released observation transition evidence changed')
                observation_gap({'reason': 'screen_changed', **transition})
                result.update(status='observation_wait',
                    choice_learning={'status': 'excluded', 'reason': 'unverified_observation_boundary'})
                observation_status('waiting_observation')
                return result
            if online is not None:
                recorder = online.recorder
                if menus.pending_decision is not None or menus.awaiting_application:
                    raise ValueError('online model cannot change across an unresolved menu decision')
                menus.recorder = recorder
                if result.get('terminal_kind') == 'death':
                    death_evidence = json.loads((Path(result['session_directory']) / 'terminal.json').read_text(encoding='utf-8'))
                accepted = (result.get('reason') in ('terminal_wave_clear', 'terminal_death')
                            and result.get('online_collection_eligible') is True and not result.get('error'))
                if not accepted:
                    recorder.invalidate('online combat rejected: ' + str(result.get('reason')))
                result['status'] = 'neural_rollout_ready' if accepted else 'aborted'
                if self.status:
                    self.status.update(phase='menu', online_learning=online_progress(),
                                       last_combat_outcome=result.get('terminal_kind'))
                return result
            observation_gap_terminal = exclude_next_terminal
            mixed_gap = bool(choice_backend and (result.get('scheduling_recoveries') or observation_gap_terminal))
            if mixed_gap:
                if menus.pending_decision is not None or menus.awaiting_application:
                    menus.abort('Combat recovery encountered an unresolved menu decision')
                menus.discard_interrupted_outcome()
                result['choice_learning'] = {'status': 'excluded', 'reason': 'released_combat_gap'}
            if (partial and result.get('recovery_only')) or mixed_gap or observation_gap_terminal:
                accepted = ((result.get('recovery_completed') is True or
                             (observation_gap_terminal and result.get('verified_terminal_boundary') is True))
                            and result.get('reason') in ('terminal_wave_clear', 'terminal_death')
                            and not result.get('error'))
                if accepted:
                    if menus.pending_decision is not None or menus.awaiting_application:
                        raise ValueError('recovery cannot reset unresolved menu memory')
                    previous_build = deepcopy(recorder.build_state)
                    recorder.abort(directory / ('recovery-history-' + uuid.uuid4().hex),
                                   'interrupted recovery history excluded from learning and evaluation')
                    recorder = FullRunRecorder(model, run_id, split=split)
                    recorder.build_state = previous_build
                    if mixed_gap:
                        from playmodel.games.brotato.laya_menu import CNN_EXCLUSION
                        recorder.invalidate(CNN_EXCLUSION)
                    menus.recorder = recorder
                    exclude_next_terminal = False
                    if result.get('terminal_kind') == 'death':
                        death_evidence = json.loads((Path(result['session_directory']) / 'terminal.json').read_text(encoding='utf-8'))
                else:
                    recorder.invalidate('partial recovery combat rejected: ' + str(result.get('reason')))
                result['status'] = 'neural_rollout_ready' if accepted else 'aborted'
                return result
            if tactical_actor is not None:
                if result.get('verified_terminal_boundary') is not True or result.get('error'):
                    recorder.invalidate('tactical combat rejected: ' + str(result.get('reason')))
                    result['status'] = 'aborted'
                    return result
                # Only provenance enters the full-run audit. There is no CNN
                # probability, value, hidden tensor, or fabricated PPO step.
                _save_json(directory / ('tactical-segment-' + uuid.uuid4().hex + '.json'), {
                    'report_path': str(Path(result['session_directory']) / 'report.json'),
                    'report_sha256': _file_sha(Path(result['session_directory']) / 'report.json'),
                    'cnn_training_eligible': False, 'game_application_verified': False})
            elif not result.get("rollout_eligible") or not result.get("flat_rollout_path"):
                recorder.invalidate("combat segment rejected: " + str(result.get("reason")))
                result["status"] = "aborted"
                return result
            else:
                recorder.append_combat_report(result)
                count = int(result['steps'])
                accepted = recorder.records[-count:]
                if accepted:
                    result['actual_movement_interval_ns'] = [accepted[0]['evidence']['sent_at_ns'],
                                                            accepted[-1]['evidence']['sent_at_ns']]
            if self.status:
                self.status.update(phase="menu", recorded_transitions=len(recorder.records),
                                   last_combat_outcome=result.get("terminal_kind"))
            if result.get("terminal_kind") == "death":
                death_evidence = json.loads((Path(result["session_directory"]) / "terminal.json").read_text(encoding="utf-8"))
            if choice_backend and result.get('reason') in ('terminal_wave_clear', 'terminal_death'):
                callback = getattr(self, 'terminal_callback', None)
                if callback is not None:
                    terminal_path = Path(result['session_directory']) / 'terminal.json'
                    terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
                    result['choice_learning'] = callback(result['terminal_kind'], {
                        **terminal, 'run_id': run_id,
                        'path': str(terminal_path.resolve()), 'sha256': _file_sha(terminal_path),
                        'terminal_path': str(terminal_path.resolve()),
                        'terminal_sha256': _file_sha(terminal_path),
                        'report_path': str((terminal_path.parent / 'report.json').resolve()),
                        'report_sha256': _file_sha(terminal_path.parent / 'report.json')})
            result["status"] = "neural_rollout_ready"
            return result

        failure = None
        verified_result = False
        stop_category = "completed"
        try:
            while (time.perf_counter() - started - observation_wait_seconds < self.max_run_seconds
                   and death_evidence is None):
                remaining = self.max_run_seconds - (time.perf_counter() - started - observation_wait_seconds)
                if remaining < 10 or self.stop_file.exists():
                    break
                report = run_session(self.executable, directory / "segments", waves=10,
                    seconds=min(600, remaining), stop_file=self.stop_file, ocr_script=self.ocr_script,
                    record=True, edit=False, run_context=context, combat_runner=combat_runner,
                      neural_menu=menus, observation_recovery=bool(choice_backend or partial),
                      navigation_ledger=self.root / 'artifacts/local-learning/ui-navigation.jsonl',
                      navigation_learning=split == 'train',
                    observation_gap_callback=observation_gap if choice_backend or partial else None,
                    observation_status_callback=observation_status)
                segments.append(report["session_directory"])
                context = report.get("run_context", context)
                wait_duration = report.get('observation_wait_seconds', 0.0)
                if (type(wait_duration) not in (float, int) or not math.isfinite(wait_duration)
                        or wait_duration < 0):
                    raise ValueError('Invalid input-free observation duration')
                observation_wait_seconds += wait_duration
                if report.get('release_error') or (report.get('recording') or {}).get('error'):
                    raise OSError('Session input release or recording completion failed')
                if death_evidence is not None:
                    break
                if ((partial or (choice_backend and observation_gaps))
                        and report['reason'] == 'run_finished' and context.get('result_source')
                        and not collection_errors() and not report.get('release_error')
                        and not report.get('error')):
                    verified_result = True
                    break
                waiting_segment = (report.get('observation_wait_segment') is True
                    and report['reason'] in ('awaiting_game_resume', 'waiting_observation')
                    and context.get('observation_wait') and (choice_backend or partial))
                if collection_errors() or (report["reason"] not in ("wave_limit", "segment_limit", "menu_limit")
                                           and not waiting_segment):
                    failure = "session stopped: " + str(report["reason"])
                    break
            if online is not None:
                online.close()
                frozen = {'schema': 'playmodel.online-run.v1', 'training_eligible': False,
                          'full_run_complete': bool(death_evidence is not None and not recorder.rejection_reasons),
                          'online_learning': online_progress(), 'steps': sum(row.get('steps', 0) for row in trials),
                          'note': 'versioned fragments train independently; aggregate is not a fixed-policy PPO trajectory'}
                if not frozen['full_run_complete']:
                    stop_category = 'user_stop' if self.stop_file.exists() else 'runtime_error'
            elif choice_backend:
                complete = bool((death_evidence is not None or verified_result) and not failure and not collection_errors())
                frozen = recorder.abort(directory / 'trajectory', 'separate Laya choices excluded from CNN PPO')
                audit_recorded_steps = frozen.get('steps', 0)
                frozen.update(schema='playmodel.laya-run.v1', full_run_complete=complete,
                              training_eligible=False, evaluation_score_eligible=False,
                              cnn_training_eligible=False, audit_recorded_steps=audit_recorded_steps,
                              steps=sum(r.get('total_actual_steps', r.get('steps', 0)) for r in trials),
                              combat_attempt_reports=[path for r in trials for path in r.get('attempt_reports', [])],
                              menu_choice_backend='local_laya',
                              combat_actor='local_laya_tactics' if tactical_actor is not None else 'frozen_cnn_gru',
                              choice_updates=[r['choice_learning'] for r in trials if 'choice_learning' in r])
                if not complete:
                    stop_category = 'user_stop' if self.stop_file.exists() else 'runtime_error' if failure else 'time_budget'
            elif partial and ((death_evidence is not None and not recorder.rejection_reasons) or verified_result):
                verified_result = True
                frozen = recorder.abort(directory / 'trajectory', 'partial recovery reached a verified ending; excluded history')
            elif death_evidence is not None and not recorder.rejection_reasons:
                frozen = recorder.finish(directory / "trajectory", kind="death", evidence=death_evidence)
            elif partial and verified_result:
                frozen = recorder.abort(directory / 'trajectory', 'partial history ended at verified result; no death label inferred')
            else:
                stop_category = ("user_stop" if self.stop_file.exists() else
                                 "runtime_error" if failure or recorder.rejection_reasons else "time_budget")
                frozen = recorder.abort(directory / "trajectory", failure or "bounded run ended without verified death")
        except Exception as error:
            from playmodel.execution_log import exception
            exception('recurrent_collection_failed', error)
            failure = f"{type(error).__name__}: {error}"
            stop_category = "user_stop" if self.stop_file.exists() else "runtime_error"
            if online is not None:
                online.close()
                frozen = {'schema': 'playmodel.online-run.v1', 'training_eligible': False,
                          'full_run_complete': False, 'online_learning': online_progress()}
            else:
                frozen = recorder.abort(directory / "trajectory-aborted", failure) if not recorder.closed else {}
        finally:
            if online is not None and self.status:
                self.status.online_provider = None
        summary = {**frozen, "run_id": run_id, "split": split, "checkpoint": str(Path(checkpoint).resolve()),
            **contract_fields(), "runtime_source_hashes": self.runtime_source_hashes,
            "evaluation_scope_id": self.evaluation_scope_id,
            "session_ids": sorted(set(segments + frozen.get("session_ids", []))),
            "setup_policy": "bootstrap_character_weapon_difficulty", "error": failure,
            "stop_category": stop_category,
            "verified_wave_clears": sum(row.get("terminal_kind") == "wave_clear" for row in trials),
            "setup_conditions": {key: context.get(key) for key in
                ("character", "character_slot", "weapons", "difficulty", "endless_verified",
                 "concept", "requested_weapon", "observed_weapon_names", "weapon_rotation_match")},
            "elapsed_seconds": time.perf_counter() - started,
            "observation_wait_seconds": observation_wait_seconds,
            "observation_gaps": observation_gaps,
            "scheduling_recovery_count": sum(len(row.get('scheduling_recoveries') or []) for row in trials),
            "completion_evidence": ({'kind': 'observed_result_after_quarantined_gap',
                'frame_ref': context.get('result_source'), 'frame_sha256': context.get('result_sha256'),
                'reward_assigned': False} if verified_result else None),
            "combat_intervals_ns": [row['actual_movement_interval_ns'] for row in trials
                                    if row.get('actual_movement_interval_ns')],
            "contact_projectile_damage_attribution": "not_implemented",
            "stat_effect_verification": "unknown_unless_separately_verified",
            "agent_or_external_api_in_runtime": False, "deployment_approved": False}
        if partial:
            summary.update(recovery_only=True, full_run_complete=False, training_eligible=False,
                           recovery_completed=bool((death_evidence is not None and not failure
                                                    and not recorder.rejection_reasons) or verified_result),
                           evaluation_score_eligible=False)
        _save_json(directory / "cycle-run.json", summary)
        return summary

    def train_candidate(self, checkpoint, training):
        """Compatibility path; the normal CLI uses the separate worker below."""
        from playmodel.learning.recurrent_training_worker import train_candidate_once
        return train_candidate_once(checkpoint, training, output=self.output, device=self.device,
            seed=self.seed, status_callback=self.status.update if self.status else None)

    def start_training(self, checkpoint, training):
        from playmodel.learning.recurrent_training_worker import start_training_job
        job = start_training_job(checkpoint, training, root=self.root, output=self.output,
            device=self.device, seed=self.seed, stop_file=self.stop_file, wait_for_combat=True)
        self._training_job = job
        self._training_gate_released = (job.directory / 'combat-gate.json').exists()
        if self.status:
            self.status.update(phase='training_worker_started', training_job=job.describe(),
                               training_manifest=training['manifest_path'])
        return job


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    from playmodel.atomic_io import atomic_json
    atomic_json(path, value, durable=True)


class CycleState:
    """One worker owns this journal. Completed evidence files remain immutable."""
    schema = 'playmodel.local-cycle-state.v1'

    def __init__(self, directory, initial=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'session-state.json'
        if self.path.exists():
            self.value = json.loads(self.path.read_text(encoding='utf-8'))
            if self.value.get('schema') != self.schema:
                raise ValueError('unsupported local cycle state')
            for filename, expected in self.value.get('file_bindings', {}).items():
                if _file_sha(filename) != expected:
                    raise ValueError('persisted cycle file changed: ' + filename)
        elif initial is not None:
            self.value = {'schema': self.schema, 'revision': 0, 'reports': [],
                          'cycle_index': 0, 'pipeline': None, 'file_bindings': {}, **initial}
            self.commit()
        else:
            raise ValueError('resumable state does not exist')

    def commit(self, **updates):
        value = {**self.value, **deepcopy(updates)}
        report = value.get('pipeline') or {}
        paths = [value.get('checkpoint'), report.get('source_checkpoint'),
                 (report.get('candidate') or {}).get('checkpoint'),
                 (report.get('training') or {}).get('manifest_path')]
        paths.extend(row.get('manifest_path') for row in report.get('evaluations', []))
        bindings = dict(value.get('file_bindings', {}))
        for filename in filter(None, paths):
            path = Path(filename).resolve()
            if str(path) not in bindings:
                bindings[str(path)] = _file_sha(path)
        value.update(file_bindings=bindings, revision=value['revision'] + 1,
                     updated_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid())
        history = self.directory / 'state-history'
        history.mkdir(exist_ok=True)
        _save_json(history / f"{value['revision']:08d}-{uuid.uuid4().hex[:8]}.json", value)
        _atomic_json(self.path, value)
        self.value = value


def _resume_summary(path):
    """Import a frozen cycle; earlier evaluations become an excluded old round."""
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'summary.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if 'training' not in report and report.get('cycles'):
        last = report['cycles'][-1]
        report = json.loads(Path(last['report_path']).read_text(encoding='utf-8'))
    training, candidate = report.get('training') or {}, report.get('candidate') or {}
    if not (training.get('split') == 'train' and training.get('training_eligible')
            and training.get('full_run_complete') and candidate.get('final_kl_within_target') is True):
        raise ValueError('resume summary requires completed eligible training and a saved valid candidate')
    from playmodel.learning.full_run import load_full_run
    from playmodel.learning.recurrent_ppo import load_checkpoint
    _, _, _, manifest = load_full_run(training['manifest_path'], require_training=False)
    source, _ = load_checkpoint(report['source_checkpoint'], device='cpu')
    _, metadata = load_checkpoint(candidate['checkpoint'], device='cpu')
    if (manifest['behavior_version'] != source.policy_version()
            or metadata.get('full_run_manifest') != training['manifest_path']
            or metadata.get('ppo', {}).get('final_kl_within_target') is not True):
        raise ValueError('saved source/candidate does not belong to frozen training')
    report.setdefault('excluded_evaluations', []).extend(report.pop('evaluations', []))
    report.update(evaluations=[], status='evaluation_pending', pending=None,
                  resume_summary=str(path), reused_frozen_training_for_fresh_comparison=True)
    return report


def _recoverable_report(report):
    """Only bounded observation failures may start a freshly verified recovery."""
    last = (report.get('evaluations') or [report.get('training', {})])[-1]
    reason = ' '.join(str(error) for error in (last.get('error'), report.get('error')) if error).casefold()
    if last.get('stop_category') == 'user_stop':
        return False
    unsafe = ('sink_deadline', 'input_guard', 'f8', 'human', 'release', 'identity',
              'recording', 'digest', 'tamper', 'overlap', 'numerical')
    if any(word in reason for word in unsafe):
        return False
    return any(word in reason for word in ('screen_changed', 'unrecognized', 'ocr',
                                           'stale_capture', 'capture_error', 'stale observation'))


def _runtime_contract(root):
    root = Path(root)
    directory = root / 'src/playmodel/games/brotato'
    hashes = {name: _file_sha(directory / name) for name in (
        'neural_runtime.py', 'vision.py', 'menu.py', 'menu_focus.py', 'neural_navigation.py', 'session.py',
        'neural_choices.py', 'neural_menu_controller.py', 'state_features.py',
        'stats_roi_ocr.py', 'shop_learning.py', 'shop_currency_ocr.py',
        'pilot.py', 'menu_capture.py', 'ocr.py', 'capture.py', 'stream.py',
          'background.py', 'interaction.py', 'setup_run.py', 'ui_layers.py', 'combat_experience.py')}
    for filename in ('src/playmodel/control/realtime.py',
                       'src/playmodel/learning/full_run.py', 'src/playmodel/learning/recurrent_ppo.py',
                         'src/playmodel/learning/visual_decision.py',
                         'src/playmodel/learning/ui_navigation.py',
                         'src/playmodel/learning/outcome_values.py',
                     'src/playmodel/learning/runtime_contract.py', 'scripts/run_recurrent_cycle.py'):
        hashes[filename] = _file_sha(root / filename)
    worker = 'src/playmodel/learning/recurrent_training_worker.py'
    hashes[worker] = _file_sha(root / worker)
    for filename in ('src/playmodel/learning/online_ppo.py',
                     'src/playmodel/games/brotato/online_runtime.py', 'scripts/run_online_learning.py',
                       'src/playmodel/games/brotato/laya_menu.py', 'scripts/run_laya_learning.py',
                       'src/playmodel/games/brotato/tactical_runtime.py',
                       'src/playmodel/games/brotato/tactical_state.py'):
        if (root / filename).is_file():
            hashes[filename] = _file_sha(root / filename)
    for filename in sorted((root / 'src/playmodel/laya').glob('*.py')):
        hashes[str(filename.relative_to(root))] = _file_sha(filename)
    return hashes


def _validate_resume_pipeline(report):
    """Reusing a score requires its unchanged trajectory and original evidence."""
    if not report:
        return
    from playmodel.learning.full_run import load_full_run
    from playmodel.learning.recurrent_ppo import load_checkpoint
    source, _ = load_checkpoint(report['source_checkpoint'], device='cpu')
    versions = {'source': source.policy_version()}
    training = report.get('training') or {}
    if training.get('full_run_complete') and training.get('training_eligible'):
        _, _, _, manifest = load_full_run(training['manifest_path'], require_training=False)
        if (manifest['run_id'] != training['run_id'] or manifest['split'] != 'train'
                or manifest['behavior_version'] != versions['source']):
            raise ValueError('resumed training does not match its source or run')
    candidate = report.get('candidate')
    if candidate:
        model, metadata = load_checkpoint(candidate['checkpoint'], device='cpu')
        if metadata.get('full_run_manifest') != training.get('manifest_path'):
            raise ValueError('resumed candidate belongs to different training')
        versions['candidate'] = model.policy_version()
    elif training.get('full_run_complete') and training.get('training_eligible'):
        require_current_contract(training)
    for row in report.get('evaluations', []):
        if not row.get('full_run_complete'):
            continue
        require_current_contract(row)
        if (row.get('evaluation_scope_id') != report.get('evaluation_scope_id')
                or row.get('runtime_source_hashes') != report.get('runtime_source_hashes')):
            raise ValueError('resumed evaluation belongs to a different runtime scope')
        _, _, _, manifest = load_full_run(row['manifest_path'], require_training=False,
                                        expected_runtime_contract=RUNTIME_CONTRACT)
        if (manifest['run_id'] != row['run_id'] or manifest['split'] != 'evaluation'
                or manifest['behavior_version'] != versions.get(row['role'])):
            raise ValueError('resumed evaluation does not match its role or run')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs='?')
    parser.add_argument("--output", type=Path, default=Path("artifacts/recurrent-cycles"))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--continuous", action="store_true",
                        help="repeat local training/comparison until stop, budget pause, or fault; iterate unapproved experimental candidates")
    parser.add_argument("--evaluation-runs", type=int, default=1, help="fresh full runs per source/candidate")
    parser.add_argument("--character-slot", type=int, default=1)
    parser.add_argument("--weapon", default="SMG")
    parser.add_argument("--max-run-seconds", type=int, default=1800)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-experimental-candidates", action="store_true",
                        help="use a compared candidate in the next experimental cycle; never approves it")
    parser.add_argument('--resume', type=Path, help='existing session directory or session-state.json')
    parser.add_argument('--resume-latest', action='store_true', help='resume latest state under --output, if present')
    parser.add_argument('--resume-summary', type=Path, help='reuse frozen training/candidate for a fresh evaluation round')
    parser.add_argument('--recover-active-run', action='store_true',
                        help='finish a positively recognized active run as excluded partial recovery')
    parser.add_argument('--max-recovery-attempts', type=int, default=2)
    parser.add_argument('--online-updates', action='store_true',
                        help='learn from verified live fragments and apply candidates at action boundaries')
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= 5 or not 1 <= args.evaluation_runs <= 5:
        parser.error("cycles and evaluation-runs must be 1..5")
    if not 1 <= args.character_slot <= 50 or not 30 <= args.max_run_seconds <= 7200:
        parser.error("character-slot must be 1..50 and max-run-seconds 30..7200")
    if not 0 <= args.max_recovery_attempts <= 3:
        parser.error('max-recovery-attempts must be 0..3')
    from playmodel.instance import session_lock
    root = Path(__file__).resolve().parents[1]
    try:
        with session_lock(root / 'artifacts/local-learning/worker.lock'):
            if args.online_updates:
                from run_online_learning import run_online
                return run_online(args, root)
            return _run_main(args, parser, root)
    except OSError as error:
        from playmodel.execution_log import exception
        exception('recurrent_worker_start_failed', error)
        print(json.dumps({'status': 'worker_start_failed', 'error': str(error),
                          'automatic_agent_call': False, 'deployment_approved': False}))
        return 1


def _run_main(args, parser, root):
    resume = args.resume
    if resume is None and args.resume_latest:
        states = list(args.output.resolve().glob('*/session-state.json'))
        if states:
            resume = max(states, key=lambda path: path.stat().st_mtime_ns)
    output = (resume.parent if resume and resume.is_file() else resume) if resume else (
        args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=bool(resume))
    status = LocalStatus(output)
    try:
        import torch
        torch.set_num_threads(2)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; select --device cpu")
        if resume:
            state = CycleState(output)
            settings = state.value.get('settings', {})
            if any(settings.get(key) != getattr(args, key) for key in ('character_slot', 'weapon', 'evaluation_runs')):
                raise ValueError('resume character/weapon/evaluation settings differ from saved session')
            source_hashes = _runtime_contract(root)
            pipeline = state.value.get('pipeline')
            if pipeline:
                pipeline = _bind_evaluation_scope(pipeline, source_hashes)
            if (pipeline != state.value.get('pipeline')
                    or state.value.get('runtime_source_hashes') != source_hashes
                    or state.value.get('runtime_contract') != RUNTIME_CONTRACT):
                state.commit(pipeline=pipeline, **contract_fields(), runtime_source_hashes=source_hashes)
            _validate_resume_pipeline(state.value.get('pipeline'))
        else:
            imported = _resume_summary(args.resume_summary) if args.resume_summary else None
            checkpoint = Path(imported['source_checkpoint']) if imported else args.checkpoint
            if checkpoint is None or not checkpoint.is_file():
                raise ValueError('checkpoint does not exist')
            source_hashes = _runtime_contract(root)
            if imported:
                imported = _bind_evaluation_scope(imported, source_hashes)
            state = CycleState(output, {'checkpoint': str(checkpoint.resolve()), 'pipeline': imported,
                'settings': {key: getattr(args, key) for key in ('character_slot', 'weapon', 'evaluation_runs')},
                'recovery_attempts': 0, **contract_fields(), 'runtime_source_hashes': source_hashes})
        coordinator = LocalCycle(root=root, output=output, character_slot=args.character_slot,
                                 weapon=args.weapon, max_run_seconds=args.max_run_seconds,
                                 device=args.device, seed=args.seed, status=status,
                                 recover_active_run=args.recover_active_run)
    except Exception as error:
        from playmodel.execution_log import exception
        exception('recurrent_worker_setup_failed', error)
        status.fault(f"{type(error).__name__}: {error}")
        status.close()
        return 1
    checkpoint = Path(state.value['checkpoint'])
    reports = state.value['reports']
    cycle = state.value['cycle_index']
    session_cycles = 0
    consecutive_rejections = 0
    exit_code = 0

    def collect(checkpoint, *, split, tag):
        pipeline = state.value.get('pipeline') or {}
        coordinator.operation_id = (pipeline.get('pending') or {}).get('operation_id')
        coordinator.evaluation_scope_id = pipeline.get('evaluation_scope_id')
        return coordinator.collect_run(checkpoint, split=split, tag=tag)

    while args.continuous or session_cycles < args.cycles:
        if coordinator.stop_file.exists():
            status.update(status="user_stopped", phase="stopped", stop_category="user_stop")
            break
        status.update(status="running", phase="cycle_start", cycle=cycle,
                      continuous=args.continuous, experimental_checkpoint=str(checkpoint))
        try:
            report = run_pipeline(checkpoint, collect_run=collect,
                                  train_candidate=coordinator.train_candidate, evaluation_runs=args.evaluation_runs,
                                  start_training=getattr(coordinator, 'start_training', None),
                                  resume_report=state.value.get('pipeline'),
                                  runtime_source_hashes=state.value['runtime_source_hashes'],
                                  on_progress=lambda current: state.commit(pipeline=current, checkpoint=str(checkpoint)))
        except Exception as error:
            from playmodel.execution_log import exception
            exception('recurrent_cycle_failed', error)
            report = {**(state.value.get('pipeline') or {}), "status": "cycle_failed",
                      "error": f"{type(error).__name__}: {error}",
                      "source_checkpoint": str(checkpoint), "deployment_approved": False}
            state.commit(pipeline=report)
        cycle_path = output / f"cycle-{cycle:02d}-{uuid.uuid4().hex[:8]}.json"
        _save_json(cycle_path, report)
        reports.append(_compact_cycle(report, cycle_path))
        if report['status'] != 'comparison_recorded':
            state.commit(reports=reports, pipeline=report)
        if (report['status'] != 'comparison_recorded' and args.recover_active_run
                and not coordinator.stop_file.exists() and _recoverable_report(report)
                and state.value.get('recovery_attempts', 0) < args.max_recovery_attempts):
            attempts = state.value.get('recovery_attempts', 0) + 1
            state.commit(recovery_attempts=attempts)
            status.update(phase='bounded_observation_recovery', recovery_attempts=attempts)
            continue
        if report["status"] == "candidate_failed_numerical_gate" and args.continuous:
            consecutive_rejections += 1
            if consecutive_rejections < 3:
                status.update(status="running", phase="candidate_rejected",
                              consecutive_rejections=consecutive_rejections,
                              experimental_checkpoint=str(checkpoint))
                cycle += 1
                session_cycles += 1
                state.commit(pipeline=None, cycle_index=cycle, recovery_attempts=0)
                continue
            status.fault("Three consecutive candidates failed numerical gates; no automatic retry")
            exit_code = 1
            break
        if report["status"] == "candidate_failed_numerical_gate":
            status.update(status="candidate_rejected", phase="stopped",
                          stop_category="numerical_rejection", needs_agent_help=False,
                          experimental_checkpoint=str(checkpoint))
            break
        if report["status"] != "comparison_recorded":
            last = (report.get("evaluations") or [report.get("training", {})])[-1]
            category = "user_stop" if coordinator.stop_file.exists() else last.get("stop_category", "runtime_error")
            if category == "time_budget":
                status.update(status="budget_paused", phase="stopped", stop_category=category,
                              needs_agent_help=False, automatic_agent_call=False)
            elif category == "user_stop":
                status.update(status="user_stopped", phase="stopped", stop_category=category)
            else:
                status.fault(report.get("error") or report["status"], cycle_report=str(cycle_path),
                             resume_state=str(state.path))
                exit_code = 1
            break
        consecutive_rejections = 0
        if args.continuous or args.continue_experimental_candidates:
            checkpoint = Path(report["candidate"]["checkpoint"])
        cycle += 1
        session_cycles += 1
        state.commit(reports=reports, pipeline=None, checkpoint=str(checkpoint), cycle_index=cycle, recovery_attempts=0)
        status.update(status="running", phase="comparison_complete", completed_cycles=cycle,
                      experimental_checkpoint=str(checkpoint), deployment_approved=False)
    else:
        status.update(status="budget_complete", phase="stopped", stop_category="cycle_budget")
    summary = {"directory": str(output), "completed_comparisons": sum(r["status"] == "comparison_recorded" for r in reports),
               "requested_cycles": None if args.continuous else args.cycles, "continuous": args.continuous,
               "cycles": reports, "status": status.state["status"],
               "deployment_approved": False, "performance_improvement_verified": False}
    _atomic_json(output / "summary.json", summary)
    print(json.dumps({"directory": str(output), "status": summary["status"],
                      "completed_comparisons": summary["completed_comparisons"],
                      "status_path": str(output / "status.json"),
                      "summary_path": str(output / "summary.json"),
                      "help_requested": (output / "help-request.json").exists(),
                      "automatic_agent_call": False, "deployment_approved": False},
                     ensure_ascii=False, indent=2))
    status.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
