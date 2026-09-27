"""Durable, local, single-process learning queue. No game input or network calls."""
from __future__ import annotations

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module('playmodel.learning.worker')


import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from .movement import (EpisodeRecord, MovementDecision, MovementStep, StateEvidence,
                       load_checkpoint, reinforce_update, save_checkpoint, validate_episode)
from playmodel.instance import session_lock

ALGORITHM = 'terminal-reinforce-background-v1'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


@contextmanager
def database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, episode_directory TEXT NOT NULL, episode_sha TEXT NOT NULL,
        style_sha TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
        updated REAL NOT NULL, result TEXT, error TEXT)''')
    try:
        with db:
            yield db
    finally:
        db.close()


def enqueue(path: Path, episode_directory: Path, *, style_sha: str) -> str:
    directory = episode_directory.resolve()
    episode_sha = digest(directory / 'episode.json')
    job_id = hashlib.sha256((ALGORITHM+episode_sha+style_sha).encode()).hexdigest()
    with database(path) as db:
        db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                   (job_id, str(directory), episode_sha, style_sha, 'pending', time.time(), time.time(), None, None))
    return job_id


def child_file(directory: Path, relative: str) -> Path:
    target = (directory / relative).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError('Episode artifact escapes its directory')
    return target


def load_episode(directory: Path, expected_sha: str):
    if digest(directory / 'episode.json') != expected_sha:
        raise ValueError('Queued episode was modified')
    raw = json.loads((directory / 'episode.json').read_text(encoding='utf-8'))
    manifest_path = child_file(directory, raw['manifest_ref'])
    if digest(manifest_path) != raw['manifest_sha256']:
        raise ValueError('Episode manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['recorder_complete'] or manifest['steps'] != len(raw['steps']):
        raise ValueError('Incomplete action recorder')
    for record in manifest['files'] + manifest['frames']:
        if digest(child_file(directory, record['path'])) != record['sha256']:
            raise ValueError('Source artifact hash mismatch')
    steps = []
    frame_hashes = {record['path']:record['sha256'] for record in manifest['frames']}
    ledger = [json.loads(line)['step'] for line in (directory/'actions.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if ledger != raw['steps']:
        raise ValueError('Episode steps differ from the preserved transmission ledger')
    for item in raw['steps']:
        if frame_hashes.get(item['frame_ref']) != item['frame_sha256']:
            raise ValueError('Step frame differs from manifest')
        item = dict(item)
        decision = dict(item.pop('decision'))
        for field in ('movement', 'probabilities', 'features', 'allowed_actions'):
            decision[field] = tuple(decision[field])
        steps.append(MovementStep(**item, decision=MovementDecision(**decision)))
    raw['steps'] = tuple(steps)
    for name in ('combat_entry', 'terminal'):
        raw[name] = StateEvidence(**raw[name]) if raw[name] else None
    episode = EpisodeRecord(**raw)
    policy = load_checkpoint(directory / 'initial-policy.json')
    validate_episode(policy, episode)
    return policy, episode


def evaluate_structure(source, candidate, episode) -> dict:
    """Numerical/drift checks only. These are NOT held-out gameplay evaluation."""
    divergences = []
    for step in episode.steps[::max(1, len(episode.steps)//64)]:
        before = source.probabilities(step.decision.features, mask=step.decision.allowed_actions)
        after = candidate.probabilities(step.decision.features, mask=step.decision.allowed_actions)
        if any(not math.isfinite(value) or value < 0 for value in after):
            raise ValueError('Invalid candidate probabilities')
        divergences.append(sum(p*math.log(p/q) for p,q in zip(before,after) if p > 0))
    maximum = max(divergences)
    if maximum > .05:
        raise ValueError('Candidate policy drift exceeds the configured safety bound')
    return {'numerical_checks_passed': True, 'max_kl_on_training_observations': maximum,
            'held_out_gameplay_evaluated': False, 'promotion_approved': False,
            'performance_improvement_verified': False,
            'eligible_for_experimental_training_continuation': True}


def process_next(queue: Path) -> dict | None:
    with database(queue) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT id,episode_directory,episode_sha,style_sha FROM jobs WHERE state='pending' ORDER BY created LIMIT 1").fetchone()
        if row is None:
            return None
        job_id, location, episode_sha, style_sha = row
        db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?", (time.time(),job_id))
    directory = Path(location)
    try:
        source, episode = load_episode(directory, episode_sha)
        output = queue.parent / 'results' / job_id
        output.mkdir(parents=True, exist_ok=True)
        # Training is deterministic from the frozen on-policy source. Never apply a past
        # episode's policy gradient to the latest policy, and never repeat epochs as if on-policy.
        update = reinforce_update(source, episode)
        evaluation = evaluate_structure(source, update.policy, episode)
        checkpoint = output / 'candidate-policy.json'
        if checkpoint.exists():
            if load_checkpoint(checkpoint).version != update.policy.version:
                raise ValueError('Conflicting recovered candidate checkpoint')
        else:
            save_checkpoint(update.policy, checkpoint)
        if load_checkpoint(checkpoint).version != update.policy.version:
            raise ValueError('Candidate reload mismatch')
        result = {'job_id':job_id, 'algorithm':ALGORITHM, 'episode_directory':str(directory),
                  'style_sha256':style_sha, 'checkpoint':str(checkpoint.resolve()),
                  'source_policy_version':source.version, 'candidate_policy_version':update.policy.version,
                  'training':update.report, 'evaluation':evaluation, 'worker_pid':os.getpid(),
                  'execution_kind':'reproduction_check' if (directory/'candidate-policy.json').exists() else 'new_episode_update'}
        (output / 'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        with database(queue) as db:
            db.execute("UPDATE jobs SET state='done',updated=?,result=? WHERE id=?", (time.time(),json.dumps(result),job_id))
        return result
    except Exception as error:
        with database(queue) as db:
            db.execute("UPDATE jobs SET state='rejected',updated=?,error=? WHERE id=?", (time.time(),f'{type(error).__name__}: {error}',job_id))
        return {'job_id':job_id, 'rejected':True, 'error':str(error)}


def ready_candidate(queue: Path, source_version: str, style_sha: str) -> dict | None:
    with database(queue) as db:
        rows = db.execute("SELECT result FROM jobs WHERE state='done' AND style_sha=? ORDER BY updated DESC", (style_sha,)).fetchall()
    for (raw,) in rows:
        result = json.loads(raw)
        if (result['source_policy_version'] == source_version
                and result['evaluation']['eligible_for_experimental_training_continuation']):
            candidate = load_checkpoint(result['checkpoint'])
            if candidate.version == result['candidate_policy_version'] and candidate.version != source_version:
                return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, default=Path('artifacts/learning/queue.sqlite3'))
    parser.add_argument('--stop-file', type=Path, default=Path('artifacts/LEARNING_STOP'))
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
            raise OSError('Cannot lower background learner priority')
    with session_lock(args.queue.with_suffix('.lock')):
        # Only one worker holds this OS lock. Interrupted work can be deterministically retried.
        with database(args.queue) as db:
            db.execute("UPDATE jobs SET state='pending' WHERE state='running'")
        while not args.stop_file.exists():
            result = process_next(args.queue)
            status = {'pid':os.getpid(),'updated_at':time.time(),'state':'idle' if result is None else 'processed',
                      'last_job':result.get('job_id') if result else None,'external_model_calls':0}
            (args.queue.parent/'worker-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            if args.once:
                break
            time.sleep(2 if result is None else .1)


if __name__ == '__main__':
    main()
