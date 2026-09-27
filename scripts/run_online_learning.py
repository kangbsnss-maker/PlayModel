"""Continuous local actor/learner orchestration; invoked under the launcher lock."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from playmodel.atomic_io import atomic_json
from playmodel.execution_log import event, exception

SCHEMA = 'playmodel.online-state.v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class OnlineState:
    def __init__(self, directory, initial=None):
        self.path = Path(directory) / 'online-state.json'
        self.lock = threading.RLock()
        self.value = read(self.path) if self.path.exists() else {'schema': SCHEMA, **(initial or {})}
        if self.value.get('schema') != SCHEMA:
            raise ValueError('not an online learning state')
        if self.value.get('checkpoint_sha256') != digest(self.value['checkpoint']):
            raise ValueError('online checkpoint changed since journal commit')
        self.commit()

    def commit(self, **values):
        with self.lock:
            candidate = {**self.value, **values, 'updated_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(self.path, candidate, durable=True)
            self.value = candidate

    def applied(self, snapshot):
        with self.lock:
            if snapshot.get('run_id') != self.value.get('active_run_id'):
                return
            count = len(snapshot.get('adoptions', []))
            if count < self.value.get('active_adoption_count', 0):
                return
            checkpoint = snapshot.get('checkpoint')
            if checkpoint and Path(checkpoint).resolve() != Path(self.value['checkpoint']).resolve():
                self.commit(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest(checkpoint),
                            active_adoption_count=count, last_online_snapshot=snapshot)


def seed_checkpoint(args):
    """Continue the last validated experimental weights without rewriting old trials."""
    if args.resume_summary:
        raise ValueError('online mode resumes its own state; legacy comparison summaries stay in fixed-policy mode')
    if args.resume_latest:
        states = sorted(args.output.resolve().glob('*/session-state.json'),
                        key=lambda path: path.stat().st_mtime_ns, reverse=True)
        if states:
            state = read(states[0])
            candidate = (state.get('pipeline') or {}).get('candidate') or {}
            if candidate.get('final_kl_within_target') is True and candidate.get('checkpoint_reload_verified') is True:
                return Path(candidate['checkpoint']).resolve(), str(states[0])
            if state.get('checkpoint'):
                return Path(state['checkpoint']).resolve(), str(states[0])
    if args.checkpoint is None:
        raise ValueError('initial checkpoint required')
    return args.checkpoint.resolve(), 'explicit_checkpoint'


def recover_applied_checkpoint(state, directory):
    """Recover a durable actual-send adoption newer than the heartbeat journal."""
    from playmodel.learning.recurrent_ppo import load_checkpoint
    run_id = state.value.get('active_run_id')
    if not run_id:
        return
    path = Path(directory) / run_id / 'online' / 'applied-checkpoint.json'
    if not path.exists():
        return
    proof = read(path)
    checkpoint = Path(proof['checkpoint']).resolve()
    if checkpoint == Path(state.value['checkpoint']).resolve():
        return
    action = proof.get('action_evidence', {})
    sent = proof.get('applied_at_ns')
    if (type(sent) is not int or action.get('sent_at_ns') != sent
            or not 0 <= action.get('observed_at_ns', sent) < sent
            or action.get('actual_action') != action.get('proposed_action')
            or digest(action['frame_ref']) != action.get('frame_sha256')):
        raise ValueError('online adoption lacks matching actual-send evidence')
    model, _ = load_checkpoint(checkpoint, device='cpu')
    if model.policy_version() != proof.get('candidate_version'):
        raise ValueError('applied checkpoint differs from actual-send policy version')
    state.commit(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                 recovered_adoption=proof)


def run_online(args, root):
    from run_recurrent_cycle import LocalCycle, LocalStatus, _runtime_contract
    from playmodel.games.brotato.online_runtime import OnlineActorSession
    from playmodel.learning.recurrent_ppo import load_checkpoint
    import torch

    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select --device cpu')
    resume = args.resume
    if resume is None and args.resume_latest:
        choices = list(args.output.resolve().glob('*/online-state.json'))
        resume = max(choices, key=lambda path: path.stat().st_mtime_ns) if choices else None
    directory = (resume.parent if resume.is_file() else resume) if resume else (
        args.output.resolve() / ('online-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True, exist_ok=bool(resume))
    status = LocalStatus(directory)
    try:
        settings = {name: getattr(args, name) for name in ('character_slot', 'weapon', 'evaluation_runs')}
        if resume:
            state = OnlineState(directory)
            if state.value.get('settings') != settings:
                raise ValueError('online resume settings differ from saved session')
            recover_applied_checkpoint(state, directory)
        else:
            checkpoint, origin = seed_checkpoint(args)
            model, _ = load_checkpoint(checkpoint, device='cpu')
            if model.config.context_dim != 64:
                raise ValueError('online Brotato mode requires the context64 checkpoint')
            state = OnlineState(directory, {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'seed_origin': origin, 'settings': settings, 'completed_runs': 0, 'completed_comparisons': 0,
                'reports': [], 'pending': None, 'deployment_approved': False})
        coordinator = LocalCycle(root=root, output=directory, character_slot=args.character_slot,
            weapon=args.weapon, max_run_seconds=args.max_run_seconds, device=args.device, seed=args.seed,
            status=status, recover_active_run=args.recover_active_run)
        def online_factory(recorder, **kwargs):
            state.commit(active_run_id=recorder.run_id, active_adoption_count=0)
            return OnlineActorSession(recorder, **kwargs)
        coordinator.online_factory = online_factory
        coordinator.online_progress = state.applied
        source_hashes = _runtime_contract(root)
        pending = state.value.get('pending')
        if pending and pending.get('runtime_source_hashes') != source_hashes:
            state.commit(pending=None, interrupted_operations=state.value.get('interrupted_operations', []) + [pending])
        elif pending and pending.get('kind') == 'online_training':
            # Live training may have adopted several policies. Preserve its
            # operation and begin a fresh verified run from the committed model.
            state.commit(pending=None, interrupted_operations=state.value.get('interrupted_operations', []) + [pending])
        count = 0
        while args.continuous or count < args.cycles:
            if coordinator.stop_file.exists():
                status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                break
            pending = state.value.get('pending')
            if pending is None:
                checkpoint = state.value['checkpoint']
                pending = {'kind': 'online_training', 'operation_id': uuid.uuid4().hex,
                           'checkpoint': checkpoint, 'runtime_source_hashes': source_hashes}
                state.commit(pending=pending)
            status.update(status='running', phase='cycle_start', continuous=args.continuous,
                          completed_cycles=state.value['completed_comparisons'],
                          completed_online_runs=state.value['completed_runs'], mode='online_updates')
            if pending['kind'] == 'online_training':
                coordinator.operation_id = pending['operation_id']
                coordinator.evaluation_scope_id = pending['operation_id']
                # A resumed live run starts with the last actually applied model.
                report = coordinator.collect_run(Path(state.value['checkpoint']), split='train', tag='training')
                snapshot = report.get('online_learning') or {}
                state.applied(snapshot)
                event('online_run_finished', run_id=report.get('run_id'), complete=report.get('full_run_complete'),
                      checkpoint=state.value['checkpoint'], online_learning=snapshot)
                state.commit(reports=state.value['reports'] + [report])
                if not report.get('full_run_complete'):
                    if coordinator.stop_file.exists():
                        status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                    else:
                        status.fault(report.get('error') or 'Online run stopped before a verified ending')
                    return 1 if not coordinator.stop_file.exists() else 0
                pending = {'kind': 'fixed_evaluation', 'operation_id': uuid.uuid4().hex,
                           'source': pending['checkpoint'], 'candidate': state.value['checkpoint'],
                           'runtime_source_hashes': source_hashes, 'evaluations': []}
                state.commit(pending=pending, completed_runs=state.value['completed_runs'] + 1)
            # Evaluation weights stay fixed. No evaluation data goes to the online worker.
            for index in range(args.evaluation_runs):
                for role in ('source', 'candidate'):
                    if any(row['role'] == role and row['index'] == index for row in pending['evaluations']):
                        continue
                    attempt_key = f'{index}-{role}'
                    attempts = pending.setdefault('attempt_operations', {})
                    coordinator.operation_id = attempts.setdefault(attempt_key, uuid.uuid4().hex)
                    state.commit(pending=pending)
                    coordinator.evaluation_scope_id = pending['operation_id']
                    report = coordinator.collect_run(Path(pending[role]), split='evaluation', tag=f'evaluation-{index}-{role}')
                    pending['evaluations'].append({'role': role, 'index': index, 'report': report})
                    state.commit(pending=pending)
                    if not report.get('full_run_complete'):
                        # Keep the failed evidence, but require a new completed run on resume.
                        pending['evaluations'].pop()
                        pending['attempt_operations'].pop(attempt_key, None)
                        state.commit(pending=pending, reports=state.value['reports'] + [report])
                        if coordinator.stop_file.exists():
                            status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                            return 0
                        status.fault(report.get('error') or 'Fixed evaluation stopped')
                        return 1
            state.commit(pending=None, completed_comparisons=state.value['completed_comparisons'] + 1,
                         reports=state.value['reports'] + [pending])
            count += 1
            status.update(status='running', phase='comparison_complete',
                          completed_cycles=state.value['completed_comparisons'])
        else:
            status.update(status='budget_complete', phase='stopped', stop_category='cycle_budget')
        return 0
    except Exception as error:
        exception('online_learning_failed', error)
        status.fault(f'{type(error).__name__}: {error}')
        return 1
    finally:
        status.online_provider = None
        status.close()
