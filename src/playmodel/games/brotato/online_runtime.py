"""Versioned local actor/learner bridge; no input transport or external service.

The input thread only stages immutable model references and commits a reference
after actual transmission. Evidence validation, serialization, PPO launch and
candidate loading belong to this bridge's background thread.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import torch

from playmodel.atomic_io import atomic_json
from playmodel.learning.full_run import FullRunRecorder, build_source_proofs


class OnlineActorSession:
    def __init__(self, recorder, *, checkpoint, root, output, device='cpu', seed=0,
                 stop_file=None, chunk_actions=64):
        if recorder.split != 'train' or type(chunk_actions) is not int or chunk_actions < 2:
            raise ValueError('online updates require training data and at least two actual actions')
        self.recorder = recorder
        self.checkpoint = str(Path(checkpoint).resolve())
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.device, self.seed, self.stop_file = device, seed, stop_file
        self.chunk_actions = chunk_actions
        self._actor_model = recorder.model
        self._actor_version = recorder.behavior_version
        self._token = 0
        self._recorder_token = 0
        self._ready = None
        self._job = None
        self._job_done = False
        self._submitted_token = None
        self._buffer = []
        self._work = queue.Queue(256)
        self._stop = threading.Event()
        self._error = None
        self._build_state = deepcopy(recorder.build_state)
        self._build_snapshot = self._build_state.snapshot() if self._build_state else None
        self._build_proofs = build_source_proofs(self._build_snapshot) if self._build_snapshot else []
        self.fragments, self.jobs, self.adoptions, self.excluded_tails = [], [], [], []
        self.memory_resets = []
        self._last_poll = 0.
        self._thread = threading.Thread(target=self._run, daemon=True, name='online-actor-evidence')
        self._thread.start()

    def begin_combat(self, model, build_state):
        self._check()
        if model.policy_version() != self._actor_version or self.recorder.closed:
            raise ValueError('online menu/combat policy version differs')
        self._build_state = deepcopy(build_state)
        self._build_snapshot = self._build_state.snapshot() if self._build_state else None
        self._build_proofs = build_source_proofs(self._build_snapshot) if self._build_snapshot else []
        return self._actor_model

    def inference_state(self, hidden, reset):
        """Called by the sole policy worker. No file access, loading or mutation."""
        self._check()
        ready = self._ready
        if ready is not None and ready['source_version'] == self._actor_version:
            return ready['model'], ready['hidden'], True, ready['version'], ready['token']
        return self._actor_model, hidden, reset, self._actor_version, self._token

    def validate_dispatch(self, packet):
        self._check()
        if packet.actor_segment == self._token and packet.behavior_version == self._actor_version:
            return
        ready = self._ready
        if (ready is None or packet.actor_segment != ready['token']
                or packet.behavior_version != ready['version']
                or ready['source_version'] != self._actor_version or not packet.reset):
            raise ValueError('online proposal no longer belongs to the staged policy')

    def commit(self, packet):
        """Reference-only commit after transport; never called for stale proposals."""
        if packet.actor_segment == self._token:
            return
        ready = self._ready
        self._actor_model, self._actor_version = ready['model'], ready['version']
        self._token, self.checkpoint = ready['token'], ready['checkpoint']
        self.adoptions.append({'source_version': ready['source_version'], 'candidate_version': ready['version'],
            'checkpoint': self.checkpoint, 'segment': self._token, 'sent_at_ns': packet.sent_at_ns,
            'applied_at_ns': packet.sent_at_ns,
            'observation_sequence': packet.frame.sequence,
            'observed_at_ns': packet.frame.metadata['capture_started_at_ns'],
            'hidden_reset': True, 'reason': ready['reason'], 'deployment_approved': False})
        self._ready = None

    def record_action(self, packet, record, directory):
        """Recorder-thread callback, after frame and actual action ledger writes."""
        self._check()
        row = dict(record, frame_ref=str((Path(directory) / record['frame_ref']).resolve()))
        row['auxiliary_sources'] = self._build_proofs
        row['observed_build_state'] = self._build_snapshot
        self._work.put_nowait(('action', packet, row, str(directory)))

    def _check(self):
        if self._error:
            raise RuntimeError('online evidence worker failed: ' + self._error)

    def _new_recorder(self, model, token):
        old = self.recorder
        self.recorder = FullRunRecorder(model, old.run_id, split='train', game_build_id=old.game_build_id)
        self.recorder.build_state = deepcopy(self._build_state)
        self._recorder_token = token

    def _append(self, items):
        if not items:
            return
        recorder = self.recorder
        recorder._open()
        staged = []
        hidden, previous = recorder.hidden, recorder.records[-1] if recorder.records else None
        for packet, evidence, directory in items:
            if packet.behavior_version != recorder.behavior_version:
                raise ValueError('online fragment mixes behavior policies')
            tensors = dict(zip(('images', 'context', 'phase', 'candidates', 'legal_mask'), packet.inputs))
            tensors.update(hidden_before=packet.hidden_before, next_hidden=packet.hidden_after,
                actions=torch.tensor([packet.action], dtype=torch.long),
                old_log_probs=torch.tensor([packet.log_probability]), old_values=torch.tensor([packet.value]),
                reset=torch.tensor([packet.reset], dtype=torch.bool))
            row = recorder._prepare_decision(tensors, evidence=evidence, reward=0., reward_evidence=None,
                expected_hidden=hidden, previous=previous, first=not recorder.records and not staged)
            staged.append(row)
            hidden, previous = row['tensors']['next_hidden'], row
        recorder._validate_policy_records(staged)
        recorder._open()
        recorder._commit_records(staged)
        recorder.session_ids.extend(Path(item[2]).name for item in items)

    def _ending(self, frame, *, kind='truncated'):
        directory = self.output / 'bootstrap-sources'
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + '.bgra')
        path.write_bytes(frame.pixels)
        return {'kind': kind, 'frame_ref': str(path.resolve()),
            'frame_sha256': hashlib.sha256(frame.pixels).hexdigest(),
            'observed_at_ns': frame.metadata['capture_started_at_ns'], 'available_at_ns': frame.available_at_ns,
            'clock_domain': 'perf_counter_ns_same_host'}

    def _freeze(self, frame, inputs, hidden, value, *, terminal=None):
        from playmodel.learning.online_ppo import save_online_bootstrap, start_online_job
        recorder = self.recorder
        kind = 'death' if terminal is not None and terminal.kind == 'death' else 'truncated'
        if terminal is not None:
            from dataclasses import asdict
            evidence = asdict(terminal)
            if terminal.kind == 'wave_clear':
                recorder.records[-1]['reward'] = 1.
                recorder.records[-1]['reward_evidence'] = evidence
        else:
            evidence = self._ending(frame)
        if kind != 'death':
            proof_path = self.output / 'bootstrap-sources' / (uuid.uuid4().hex + '.pt')
            proof_path.parent.mkdir(exist_ok=True)
            evidence.update(save_online_bootstrap(proof_path, inputs, hidden,
                behavior_version=recorder.behavior_version, observed_at_ns=evidence['observed_at_ns'],
                available_at_ns=evidence['available_at_ns']))
        fragment = recorder.finish(self.output / ('fragment-' + uuid.uuid4().hex), kind=kind,
                                   evidence=evidence, next_value=value)
        summary = {key: fragment[key] for key in ('manifest_path', 'behavior_version', 'steps', 'ending',
                                                'training_eligible', 'full_run_complete')}
        summary.update(segment=self._recorder_token, physical_run_id=recorder.run_id)
        self.fragments.append(summary)
        if (self._job is None or self._job_done) and self._ready is None:
            self._job = start_online_job(fragment['manifest_path'], root=self.root, output=self.output,
                device=self.device, seed=self.seed, stop_file=self.stop_file)
            self._job_done = False
            self._submitted_token = self._recorder_token
            self.jobs.append({'manifest_path': fragment['manifest_path'], 'status': 'running',
                              'source_version': recorder.behavior_version,
                              **self._job.describe()})
        else:
            summary['training_submitted'] = False
            summary['reason'] = 'learner_running_or_candidate_waiting_for_actual_handoff'

    def _exclude_tail(self, reason):
        if self._buffer or (not self.recorder.closed and self.recorder.records):
            target = self.output / ('excluded-tail-' + uuid.uuid4().hex + '.json')
            value = {'reason': reason, 'training_eligible': False, 'segment': self._recorder_token,
                     'behavior_version': self.recorder.behavior_version,
                     'menu_records': [{k: v for k, v in row.items() if k != 'tensors'}
                                      for row in self.recorder.records] if not self.recorder.closed else [],
                     'actions': [item[1] for item in self._buffer]}
            atomic_json(target, value, durable=True)
            self.excluded_tails.append({'path': str(target), 'reason': reason,
                'actions': len(self._buffer), 'behavior_version': self.recorder.behavior_version})
        self._buffer = []

    def _poll(self):
        if self._job is None or self._job_done or time.monotonic() - self._last_poll < .2:
            return
        self._last_poll = time.monotonic()
        from playmodel.learning.recurrent_training_worker import TrainingCancelled
        try:
            report = self._job.poll()
        except TrainingCancelled:
            self._job_done = True
            self.jobs[-1].update(status='cancelled')
            return
        if report is None:
            return
        self._job_done = True
        self.jobs[-1].update(status='completed', optimizer_steps=report.get('optimizer_steps'),
                            candidate_version=report.get('candidate_version'), checkpoint=report.get('checkpoint'),
                            optimization_started_at_ns=report.get('optimization_started_at_ns'),
                            optimization_finished_at_ns=report.get('optimization_finished_at_ns'))
        if (report.get('final_kl_within_target') is not True
                or type(report.get('optimizer_steps')) is not int or report['optimizer_steps'] < 1
                or report.get('candidate_version') == self._actor_version):
            model, version, checkpoint = self._actor_model, self._actor_version, self.checkpoint
            reason = 'numerical_candidate_rejected_memory_reset'
        else:
            model, checked = self._job.load_candidate(current_version=self._actor_version, device='cpu')
            version, checkpoint = model.policy_version(), checked['checkpoint']
            reason = 'verified_online_ppo_candidate'
        self._ready = {'model': model, 'version': version, 'source_version': self._actor_version,
            'checkpoint': checkpoint, 'hidden': model.initial_hidden(1), 'token': self._token + 1,
            'reason': reason}

    def _end(self, *, frame, phase, terminal, eligible):
        from .neural_runtime import _inputs
        if self._buffer and not self.recorder.closed and eligible and frame is not None:
            self._append(self._buffer)
            self._buffer = []
            inputs = _inputs(self.recorder.model, frame, int(self.recorder.records[-1]['tensors']['actions'].item()),
                             phase=phase, build_state=self._build_state)
            hidden = self.recorder.hidden
            with torch.no_grad():
                value = float(self.recorder.model.step(*inputs, hidden=hidden).value.item())
            self._freeze(frame, inputs, hidden, value, terminal=terminal)
        else:
            self._exclude_tail('collector_guard_or_stop' if not eligible else 'old_version_tail_after_frozen_update')
        self._new_recorder(self._actor_model, self._token)
        self.memory_resets.append({'at_ns': time.perf_counter_ns(), 'segment': self._token,
                                   'reason': 'combat_menu_fragment_boundary', 'behavior_version': self._actor_version})

    def _run(self):
        try:
            while not self._stop.is_set() or not self._work.empty():
                try:
                    work = self._work.get(timeout=.02)
                except queue.Empty:
                    self._poll()
                    continue
                try:
                    if work[0] == 'action':
                        _, packet, row, directory = work
                        if packet.actor_segment != self._recorder_token:
                            self._exclude_tail('source_version_superseded_after_actual_candidate_send')
                            self._new_recorder(packet.behavior_model, packet.actor_segment)
                            adoption = next(row for row in reversed(self.adoptions)
                                            if row['segment'] == packet.actor_segment)
                            atomic_json(self.output / 'applied-checkpoint.json',
                                        {**adoption, 'action_evidence': row}, durable=True)
                        self._buffer.append((packet, row, directory))
                        if (not self.recorder.closed and len(self._buffer) > self.chunk_actions
                                and (self._job is None or self._job_done)):
                            chosen, bootstrap = self._buffer[:self.chunk_actions], self._buffer[self.chunk_actions][0]
                            self._append(chosen)
                            self._buffer = self._buffer[self.chunk_actions:]
                            self._freeze(bootstrap.frame, bootstrap.inputs, bootstrap.hidden_before, bootstrap.value)
                    elif work[0] == 'end':
                        _, details, completed = work
                        try:
                            self._end(**details)
                        finally:
                            completed.set()
                finally:
                    self._work.task_done()
                self._poll()
        except Exception as error:
            self._error = f'{type(error).__name__}: {error}'
            from playmodel.execution_log import exception
            exception('online_actor_evidence_failed', error)
            atomic_json(self.output / 'failure.json', {'error': self._error}, durable=True)

    def end_combat(self, *, frame, phase, terminal, eligible):
        self._check()
        completed = threading.Event()
        self._work.put(('end', dict(frame=frame, phase=phase, terminal=terminal, eligible=eligible), completed))
        while not completed.wait(.05):
            self._check()
        self._check()

    def snapshot(self):
        return {'checkpoint': self.checkpoint, 'behavior_version': self._actor_version,
            'applied_checkpoint_path': str(self.output / 'applied-checkpoint.json'),
            'status': 'failed' if self._error else 'candidate_ready' if self._ready else
                      'learning' if self._job is not None and not self._job_done else 'collecting',
            'error': self._error, 'fragments': deepcopy(self.fragments), 'jobs': deepcopy(self.jobs),
            'adoptions': deepcopy(self.adoptions), 'excluded_tails': deepcopy(self.excluded_tails),
            'memory_resets': deepcopy(self.memory_resets), 'deployment_approved': False}

    def close(self):
        self._stop.set()
        self._thread.join(10)
        if self._thread.is_alive():
            raise RuntimeError('online evidence worker did not finish bounded cleanup')
        if self._buffer or (not self.recorder.closed and self.recorder.records):
            self._exclude_tail('online_session_closed')
        if self._job is not None:
            self._job.close(cancel=not self._job_done)
            if not self._job_done:
                self.jobs[-1].update(status='closed_unapplied_job_preserved')
        atomic_json(self.output / 'online-session.json', self.snapshot(), durable=True)
