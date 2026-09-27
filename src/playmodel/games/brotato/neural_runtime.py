"""Bounded, on-policy neural movement collection; no training or menu input.

This adapter records successful HWND transmissions, not proof that the game
applied movement. Only independently verified state evidence supplies sparse
rewards. A wave boundary truncates a movement trial; it is not a completed run.
Importing this module does not capture a screen or send input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid

import torch
from torch.nn import functional as F

from playmodel.control import ControlLimits, Observation, RealtimeController, SendReceipt
from playmodel.learning import StateEvidence
from playmodel.learning.recurrent_ppo import RecurrentActorCritic, RolloutBatch, save_checkpoint
from playmodel.learning.runtime_contract import (RUNTIME_CONTRACT, PHASE_SCHEMA,
                                                 LEGACY_RUNTIME_CONTRACT, LEGACY_PHASE_SCHEMA)
from .background import BackgroundController
from .capture import _png, read_diagnostic_png
from .menu import classify_scene
from .pilot import (CLOCK, LocalTerminalObserver, PilotConfig, SafetyMonitor, SlowTerminalWorker,
                    TerminalRules, _frame_ok, _json, _sha)
from .stream import CaptureStream, Frame
from .vision import BrotatoVision

SCHEMA = 'playmodel.neural-movement-rollout.v1'
MOVEMENT = ((0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))


def image_tensor(frame: Frame) -> torch.Tensor:
    """Preserved BGRA sample -> RGB uint8 [1,3,96,96]; never future pixels."""
    width, height = frame.metadata['sample_width'], frame.metadata['sample_height']
    if len(frame.pixels) != width * height * 4 or width < 1 or height < 1:
        raise ValueError('invalid raw frame size')
    raw = torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8).reshape(height, width, 4)
    rgb = raw[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(rgb.float(), size=(96, 96), mode='area').round().to(torch.uint8)


def _inputs(model, frame, previous_action, *, phase=0, build_state=None):
    device = next(model.parameters()).device
    context = torch.zeros(1, model.config.context_dim, device=device)
    # Known previous ACTUAL transmission only. Remaining dimensions are reserved
    # unknown fields; no invented HP, inventory, enemy identity, or tree reward.
    if model.config.context_dim < 9:
        raise ValueError('neural movement context requires nine previous-action slots')
    context[0, previous_action] = 1
    if model.config.context_dim == 64:
        if build_state is None:
            raise ValueError('context64 requires observed build state')
        context[0, 16:] = torch.tensor(build_state.features(
            frame.metadata['capture_started_at_ns'], available_at_ns=frame.available_at_ns),
            dtype=context.dtype, device=device)
    candidates = torch.zeros(1, 1, model.config.candidate_dim, device=device)
    legal = torch.ones(1, 9, dtype=torch.bool, device=device)
    if phase:
        legal[:, 1:] = False  # Value-only unknown-menu state, never sampled/sent.
    return (image_tensor(frame).to(device), context,
            torch.tensor([phase], dtype=torch.long, device=device), candidates, legal)


def _evidence(evidence: StateEvidence, kinds, *, rules=None):
    if (not isinstance(evidence, StateEvidence) or evidence.kind not in kinds
            or evidence.verified is not True or evidence.independent_of_policy is not True
            or evidence.clock_domain != CLOCK or not evidence.verifier_id
            or evidence.origin not in ('local_detector', 'developer_verified')
            or not 0 <= evidence.observed_at_ns <= evidence.available_at_ns <= evidence.verified_at_ns
            or evidence.verified_at_ns > time.perf_counter_ns()):
        raise ValueError('independent state evidence required')
    if _sha(Path(evidence.frame_ref)) != evidence.frame_sha256:
        raise ValueError('state evidence source digest mismatch')
    if rules is not None and evidence.verifier_id != f'calibrated_local_ocr:{rules.version}':
        raise ValueError('terminal evidence is not from the configured calibrated rules')


@dataclass
class NeuralAction:
    frame: Frame
    inputs: tuple
    hidden_before: torch.Tensor
    hidden_after: torch.Tensor
    reset: bool
    action: int
    log_probability: float
    value: float
    probabilities: list
    decided_at_ns: int
    sent_at_ns: int | None = None
    generation: int | None = None
    transport_started_at_ns: int | None = None
    behavior_version: str | None = None
    actor_segment: int | None = None
    behavior_model: object | None = field(default=None, repr=False)
    legal_mask: tuple[bool, ...] = field(init=False, repr=False)
    recorded_hidden_before: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        # Materialize native snapshots on the policy worker, within its budget.
        # Neither the input gate nor the recorder may enter PyTorch (including
        # CPU tensor indexing, scalar conversion, or device synchronization).
        mask = self.inputs[4].tolist()
        if (len(mask) != 1 or len(mask[0]) != len(MOVEMENT)
                or any(type(value) is not bool for value in mask[0])):
            raise ValueError('neural action requires a boolean movement mask')
        self.legal_mask = tuple(mask[0])
        self.recorded_hidden_before = tuple(self.hidden_before.tolist()[0])


class _Writer:
    def __init__(self, directory, first_action_callback=None, action_callback=None):
        self.directory = directory
        self.first_action_callback = first_action_callback
        self.action_callback = action_callback
        (directory / 'frames').mkdir()
        self.pending = queue.Queue(32)
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='neural-rollout-writer')
        self.thread.start()

    def _run(self):
        try:
            with (self.directory / 'actions.jsonl').open('x', encoding='utf-8') as stream:
                while not self.stop.is_set() or not self.pending.empty():
                    try:
                        packet = self.pending.get(timeout=.02)
                    except queue.Empty:
                        continue
                    record = _action_record(packet)
                    path = self.directory / record['frame_ref']
                    with path.open('xb') as output:
                        output.write(packet.frame.pixels)
                    _json(path.with_suffix('.json'), packet.frame.metadata)
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    if self.action_callback is not None:
                        self.action_callback(packet, record, self.directory)
                    if self.first_action_callback is not None:
                        callback, self.first_action_callback = self.first_action_callback, None
                        # This is the recorder thread, after a real transmission
                        # and its evidence write. Never block the input gate on
                        # background-training coordination or filesystem I/O.
                        callback({**record, 'session_id': self.directory.name})
                    self.pending.task_done()
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'

    def close(self):
        self.stop.set()
        self.thread.join(5)
        return not self.thread.is_alive() and self.error is None


def _action_record(packet):
    return {'sequence': packet.frame.sequence, 'generation': packet.generation,
            'frame_ref': f'frames/{packet.frame.sequence:08d}.bgra',
            'frame_sha256': hashlib.sha256(packet.frame.pixels).hexdigest(),
            'observed_at_ns': packet.frame.metadata['capture_started_at_ns'],
            'available_at_ns': packet.frame.available_at_ns,
            'decided_at_ns': packet.decided_at_ns, 'sent_at_ns': packet.sent_at_ns,
            'transport_started_at_ns': packet.transport_started_at_ns,
            'transport_finished_at_ns': packet.sent_at_ns,
            'proposed_action': packet.action, 'actual_action': packet.action,
            'probabilities': packet.probabilities, 'old_log_probability': packet.log_probability,
            'old_value': packet.value, 'legal_mask': packet.legal_mask,
            'hidden_before': packet.recorded_hidden_before, 'reset': packet.reset,
            'action_origin': 'policy', 'transmitted': True, 'acknowledged': None,
            'game_application_verified': False, 'clock_domain': CLOCK,
            'behavior_version': packet.behavior_version, 'actor_segment': packet.actor_segment}


class NeuralMovementSink:
    """Single writer: sends the sampled action unchanged, then commits memory."""
    def __init__(self, background, writer, hwnd, hidden, *, reset_first=True, online_session=None):
        self.background, self.writer, self.hwnd = background, writer, hwnd
        self.hidden = hidden
        self.reset_first = reset_first
        self.packets = []
        self.transport_attempts = []
        self.sent_count = 0
        self.previous_action = 0
        self.last_sent_at_ns = None
        self.release_failed = False
        self.online_session = online_session

    def send(self, action, *, generation, observation_sequence, deadline_ns):
        if (not isinstance(action, NeuralAction) or action.frame.sequence != observation_sequence
                or action.frame.metadata['hwnd'] != self.hwnd or self.writer.error
                or self.writer.pending.full() or type(action.action) is not int
                or not 0 <= action.action < 9 or not action.legal_mask[action.action]):
            raise OSError('invalid, stale, illegal, or unrecordable neural action')
        dx, dy = MOVEMENT[action.action]
        keys = set()
        if dx:
            keys.add(0x44 if dx > 0 else 0x41)
        if dy:
            keys.add(0x53 if dy > 0 else 0x57)
        # Timestamp only the background adapter call as transport. The realtime
        # controller still bounds the ENTIRE sink call with its original budget.
        # A failed call may have posted some keys; preserve that uncertainty.
        attempt = {'sequence': observation_sequence, 'generation': generation,
                   'action': action.action, 'deadline_ns': deadline_ns,
                   'transport_started_at_ns': None, 'transport_finished_at_ns': None,
                   'transmitted': None, 'acknowledged': None,
                   'game_application_verified': False, 'error': None, 'clock_domain': CLOCK}
        started = time.perf_counter_ns()
        if self.online_session is not None:
            self.online_session.validate_dispatch(action)
            started = time.perf_counter_ns()
        if started >= deadline_ns:
            raise OSError('neural movement deadline expired')
        attempt['transport_started_at_ns'] = started
        self.transport_attempts.append(attempt)
        try:
            bounded = getattr(self.background, 'set_movement_before', None)
            if bounded is not None:
                bounded(keys, deadline_ns=deadline_ns)
            else:
                self.background.set_movement(keys)
        except Exception as error:
            attempt['transport_finished_at_ns'] = time.perf_counter_ns()
            attempt['error'] = type(error).__name__
            raise
        finally:
            stages = getattr(self.background, 'last_movement_timing', None)
            if isinstance(stages, dict):
                attempt['background_stages'] = dict(stages)
        sent = time.perf_counter_ns()
        attempt['transport_finished_at_ns'], attempt['transmitted'] = sent, True
        action.transport_started_at_ns, action.sent_at_ns, action.generation = started, sent, generation
        self.packets.append(action)
        self.sent_count += 1
        self.previous_action = action.action
        self.last_sent_at_ns = action.sent_at_ns
        self.hidden = action.hidden_after
        if self.online_session is not None:
            self.online_session.commit(action)
        # Publish only after all native transmission state is committed. The
        # recorder receives a complete packet and never mutates policy state.
        self.writer.pending.put_nowait(action)
        return SendReceipt(True, None)

    def release(self, *, generation, reason, deadline_ns):
        try:
            self.background.release()
        except Exception:
            self.release_failed = True
            raise
        # A release is a safety event, not an additional PPO action.
        self.last_sent_at_ns = None
        return SendReceipt(True, None)


def chunk_rollout(batch: RolloutBatch, *, initial_states: torch.Tensor,
                  chunk_steps=32, burn_in=8) -> RolloutBatch:
    """Bounded BPTT: disjoint suffixes, actual overlapping burn-in observations.

    GAE traces stop at chunk edges and bootstrap stored next values. This is a
    truncated approximation, not an assertion of exact full-run GAE.
    """
    if (type(chunk_steps) is not int or type(burn_in) is not int or chunk_steps < 1
            or burn_in < 0 or chunk_steps + burn_in > 64 or batch.valid.shape[1] != 1
            or not batch.valid.all() or initial_states.shape != (batch.valid.shape[0], 1, batch.initial_hidden.shape[1])):
        raise ValueError('require one complete sequence, logged states, and <=64-step chunks')
    total = batch.valid.shape[0]
    count = (total + chunk_steps - 1) // chunk_steps
    length = chunk_steps + burn_in
    result = {}
    for name, value in batch.__dict__.items():
        if not isinstance(value, torch.Tensor):
            result[name] = value
        elif name == 'initial_hidden':
            result[name] = torch.zeros(count, value.shape[1], dtype=value.dtype, device=value.device)
        else:
            result[name] = torch.zeros(length, count, *value.shape[2:], dtype=value.dtype, device=value.device)
    for column, start in enumerate(range(0, total, chunk_steps)):
        source = max(0, start - burn_in)
        target = burn_in - (start - source)
        stop = min(total, start + chunk_steps)
        result['initial_hidden'][column] = initial_states[source, 0]
        for name, value in batch.__dict__.items():
            if isinstance(value, torch.Tensor) and name != 'initial_hidden':
                result[name][target:target + stop - source, column] = value[source:stop, 0]
    return RolloutBatch(**result)


def _batch(model, packets, final_frame, *, terminal, final_phase, rollout_id, split, chunk_steps, burn_in, build_state=None):
    if not packets or final_frame.metadata['capture_started_at_ns'] <= packets[-1].sent_at_ns:
        raise ValueError('a true post-action final observation is required')
    with torch.no_grad():
        final_inputs = _inputs(model, final_frame, packets[-1].action, phase=final_phase, build_state=build_state)
        final_output = model.step(*final_inputs, hidden=packets[-1].hidden_after.to(next(model.parameters()).device))
    values = [p.value for p in packets]
    next_values = values[1:] + [float(final_output.value.item())]
    next_times = [p.frame.metadata['capture_started_at_ns'] for p in packets[1:]] + [final_frame.metadata['capture_started_at_ns']]
    for packet, at in zip(packets, next_times):
        if at <= packet.sent_at_ns:
            raise ValueError('next observation predates the actual action')
    count = len(packets)
    tensor = lambda values, dtype=torch.float32: torch.tensor(values, dtype=dtype).reshape(count, 1)
    rewards = [0.] * count
    terminated, truncated = [False] * count, [False] * count
    if terminal and terminal.kind == 'death':
        rewards[-1], terminated[-1], next_values[-1] = -1., True, 0.
    else:
        truncated[-1] = True
        if terminal and terminal.kind == 'wave_clear':
            rewards[-1] = 1.
    batch = RolloutBatch(
        images=torch.stack([p.inputs[0] for p in packets]), context=torch.stack([p.inputs[1] for p in packets]),
        phase=torch.stack([p.inputs[2] for p in packets]), candidates=torch.stack([p.inputs[3] for p in packets]),
        legal_mask=torch.stack([p.inputs[4] for p in packets]), actions=tensor([p.action for p in packets], torch.long),
        old_log_probs=tensor([p.log_probability for p in packets]), old_values=tensor(values),
        rewards=tensor(rewards), next_values=tensor(next_values), terminated=tensor(terminated, torch.bool),
        truncated=tensor(truncated, torch.bool), valid=torch.ones(count, 1, dtype=torch.bool),
        reset=tensor([p.reset for p in packets], torch.bool),
        elapsed_seconds=tensor([(at-p.decided_at_ns)/1e9 for p, at in zip(packets, next_times)]),
        initial_hidden=packets[0].hidden_before, behavior_version=model.policy_version(), rollout_id=rollout_id,
        split=split, runtime_contract=RUNTIME_CONTRACT)
    states = torch.stack([p.hidden_before for p in packets])
    return chunk_rollout(batch, initial_states=states, chunk_steps=chunk_steps, burn_in=burn_in), batch, states


def load_rollout(path: Path | str, *, device='cpu') -> RolloutBatch:
    """Verify the frozen local evidence manifest before deserializing tensors."""
    path = Path(path).resolve()
    manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or manifest.get('recorder_complete') is not True:
        raise ValueError('incomplete neural rollout manifest')
    files = manifest.get('files', [])
    if not any(row['path'] == path.name for row in files):
        raise ValueError('rollout absent from manifest')
    for row in files:
        source = (path.parent / row['path']).resolve()
        if not source.is_relative_to(path.parent) or _sha(source) != row['sha256']:
            raise ValueError('neural rollout evidence digest mismatch')
    for proof in manifest.get('sources', []):
        if _sha(Path(proof['path'])) != proof['sha256']:
            raise ValueError('observed build source digest mismatch')
    entry = json.loads((path.parent / 'combat-entry.json').read_text(encoding='utf-8'))
    if _sha(path.parent / 'combat-entry-source') != entry['frame_sha256']:
        raise ValueError('combat entry copied evidence mismatch')
    if (path.parent / 'terminal.json').exists():
        terminal = json.loads((path.parent / 'terminal.json').read_text(encoding='utf-8'))
        if _sha(path.parent / 'terminal-source.png') != terminal['frame_sha256']:
            raise ValueError('terminal copied evidence mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA:
        raise ValueError('unsupported neural rollout schema')
    contract = manifest.get('runtime_contract', LEGACY_RUNTIME_CONTRACT)
    phase_schema = manifest.get('phase_schema', LEGACY_PHASE_SCHEMA)
    if (contract, phase_schema) not in ((RUNTIME_CONTRACT, PHASE_SCHEMA),
                                       (LEGACY_RUNTIME_CONTRACT, LEGACY_PHASE_SCHEMA)):
        raise ValueError('unsupported neural rollout runtime contract')
    values = dict(payload['batch'])
    values.setdefault('runtime_contract', LEGACY_RUNTIME_CONTRACT)
    batch = RolloutBatch(**values)
    if batch.runtime_contract != contract:
        raise ValueError('rollout runtime contract differs from manifest')
    if batch.behavior_version != manifest['behavior_version']:
        raise ValueError('rollout behavior version differs from manifest')
    return batch.to(device)


def _safe_recovery_release(report):
    """Only completed, released scheduling guards may open a separate trial."""
    if (report.get('reason') != 'controller_guard'
            or report.get('controller_guard_reason') not in ('held_observation_expired', 'input_watchdog')
            or report.get('recorder_complete') is not True or report.get('worker_stopped') is not True
            or report.get('cleanup_errors') or report.get('capture_error') or report.get('safety_reason')):
        return None
    directory = Path(report['session_directory'])
    events = json.loads((directory / 'control-events.json').read_text(encoding='utf-8'))
    attempts = json.loads((directory / 'input-attempts.json').read_text(encoding='utf-8'))
    if any(row.get('transmitted') is not True or row.get('error') for row in attempts):
        return None
    guard = next((index for index, row in enumerate(events) if row['kind'] == 'authority'
                  and row['reason'] == report['controller_guard_reason']), None)
    if guard is None:
        return None
    releases = [row for row in events[guard:] if row['kind'] == 'release']
    if not releases or any(not isinstance(row.get('receipt'), dict)
            or row['receipt'].get('transmitted') is not True
            or row['receipt'].get('acknowledged') is False
            or 'error' in row['reason'] or row['reason'].startswith('release_')
            or not isinstance(row.get('send_finished_at_ns'), int)
            or row['send_finished_at_ns'] >= row['deadline_ns'] for row in releases):
        return None
    return {'released_at_ns': max(row['send_finished_at_ns'] for row in releases),
            'target_identity': report.get('target_identity'),
            'previous_session': str(directory), 'guard': report['controller_guard_reason']}


def _reacquire_combat(stream, background, vision, *, recovery, config, stop_file, directory):
    """Two independent post-release observations; this helper sends no input."""
    target = recovery.get('target_identity')
    if not target or not target.get('hwnd'):
        raise ValueError('recovery lacks original target identity')
    sources, previous = [], None
    until = time.perf_counter() + min(2., config.startup_seconds)
    while len(sources) < 2 and time.perf_counter() < until:
        if stop_file.exists() or stream.error:
            raise OSError('stop or capture failure during released recovery')
        background.check()
        context = getattr(background, '_context', None)
        if (context is not None and context.foreground() == target['hwnd']
                and any(context.key_state(key) & 0x8000 for key in (0x57, 0x41, 0x53, 0x44))):
            raise OSError('human intervention during released recovery')
        frame = stream.latest(max_age_ms=config.max_frame_age_ms)
        if frame is None:
            time.sleep(.005)
            continue
        if (not _frame_ok(frame, now=time.perf_counter_ns(), max_age_ms=config.max_frame_age_ms,
                          hwnd=target['hwnd']) or any(frame.metadata.get(key) != value
                for key, value in target.items() if value is not None)):
            raise ValueError('recovery target/frame changed')
        observed = frame.metadata['capture_started_at_ns']
        if observed <= recovery['released_at_ns'] or (previous is not None
                and (frame.sequence <= previous.sequence or observed <= previous.available_at_ns)):
            time.sleep(.005)
            continue
        state = vision.observe(frame.pixels, frame.metadata['sample_width'], frame.metadata['sample_height'],
                               observed_at_ns=observed, alpha_mode='ignore')
        if not state.combat_likely:
            raise ValueError('released recovery requires two confirmed combat observations')
        path = directory / f'recovery-combat-{len(sources)}.png'
        path.write_bytes(_png(frame.metadata['sample_width'], frame.metadata['sample_height'], frame.pixels))
        sources.append({'frame_ref': str(path.resolve()), 'frame_sha256': _sha(path),
                        'observed_at_ns': observed, 'available_at_ns': frame.available_at_ns,
                        'sequence': frame.sequence})
        previous = frame
    if len(sources) != 2:
        raise ValueError('released recovery fresh observation pair unavailable')
    proof = {**recovery, 'observations': sources, 'verified_at_ns': time.perf_counter_ns(),
             'hidden_reset': True, 'gap_training_eligible': False}
    _json(directory / 'scheduling-reacquisition.json', proof)
    last = sources[-1]
    return previous, StateEvidence('combat', last['frame_ref'], last['frame_sha256'],
        last['observed_at_ns'], last['available_at_ns'], proof['verified_at_ns'],
        'local_detector', 'two_fresh_visual_combat_observations_v1', True, True)


def run_neural_trial(executable: Path, output_root: Path, *, model: RecurrentActorCritic,
                     combat_entry: StateEvidence, config: PilotConfig = PilotConfig(),
                     scheduling_recovery=False, recovery_only=False, **kwargs) -> dict:
    """Optional bounded, released restarts; original attempt artifacts stay immutable."""
    if type(scheduling_recovery) is not bool or type(recovery_only) is not bool:
        raise ValueError('recovery flags must be boolean')
    online = kwargs.get('online_session')
    if scheduling_recovery and online is None and not recovery_only:
        raise ValueError('fixed scored/training runs cannot silently recover across a gap')
    started, attempts, recoveries, consecutive, total = time.perf_counter(), [], [], 0, 0
    active_model, active_kwargs, current = model, dict(kwargs), config
    while True:
        report = _run_neural_trial_once(executable, output_root, model=active_model,
            combat_entry=combat_entry, config=current, **active_kwargs)
        attempts.append(report)
        total += int(report.get('steps', 0))
        if not scheduling_recovery:
            return report
        proof = _safe_recovery_release(report)
        stop = kwargs.get('stop_file')
        if (proof is None or (stop and Path(stop).exists())
                or (Path(report['session_directory']) / 'STOP').exists()):
            break
        # A single successful post must not turn a persistent stall into an
        # unbounded retry loop. One full collection chunk establishes progress.
        if report.get('steps', 0) >= 64:
            consecutive = 0
        remaining = config.max_seconds - (time.perf_counter() - started)
        if consecutive >= 2 or remaining <= 0 or total >= config.max_steps:
            break
        consecutive += 1
        recoveries.append({**proof, 'attempt': len(attempts), 'consecutive_retry': consecutive,
                           'previous_steps': report.get('steps', 0)})
        if online is not None:
            active_model = online.recorder.model
            active_kwargs['build_state'] = online.recorder.build_state
        active_kwargs.update(initial_hidden=active_model.initial_hidden(1), reset_first=True,
                             first_action_callback=None, _recovery=proof)
        current = replace(config, max_seconds=remaining, max_steps=config.max_steps-total)
    result = dict(report)
    result.update(scheduling_recoveries=recoveries,
        attempt_reports=[str(Path(item['session_directory']) / 'report.json') for item in attempts],
        total_actual_steps=total, recovery_only=recovery_only, evaluation_score_eligible=not recoveries and not recovery_only)
    if recovery_only or (recoveries and online is None):
        result.update(rollout_eligible=False, rollout_path=None, flat_rollout_path=None, initial_states_path=None,
            recovery_completed=bool(report.get('reason') in ('terminal_wave_clear', 'terminal_death')
                                    and report.get('rollout_eligible') is True and not report.get('error')))
    _json(Path(report['session_directory']) / 'scheduling-recovery-summary.json', result)
    return result


def _run_neural_trial_once(executable: Path, output_root: Path, *, model: RecurrentActorCritic,
                     combat_entry: StateEvidence, config: PilotConfig = PilotConfig(), stop_file=None,
                     terminal_rules: TerminalRules | None = None, ocr_script=None, terminal_observer=None,
                     stream_factory=CaptureStream, background_factory=BackgroundController,
                     vision_factory=BrotatoVision, chunk_steps=32, burn_in=8, split='train',
                     initial_hidden=None, reset_first=None, scene_callback=None, build_state=None,
                     first_action_callback=None, terminal_ocr_reader=None, online_session=None, _recovery=None) -> dict:
    """Explicit live collection entry point; performs no PPO update or menu action.

    An optional hidden state carries history from a caller-owned menu path, but
    this artifact still proves only the bounded movement trial. The caller must
    keep the same checkpoint through its own larger run. Guard/human/failed
    transport trials are retained for diagnosis and excluded from PPO.
    """
    if config.train or config.defer_training or split not in ('train', 'validation', 'test', 'evaluation'):
        raise ValueError('collection is separate from training; use a fixed split')
    if online_session is not None and split != 'train':
        raise ValueError('online actor updates require the training split')
    if not 1 <= chunk_steps <= 64 or not 0 <= burn_in < 64 or chunk_steps + burn_in > 64:
        raise ValueError('bounded recurrent chunk settings required')
    if terminal_observer is None and (terminal_rules is None or ocr_script is None):
        raise ValueError('independent calibrated terminal observer required')
    _evidence(combat_entry, ('combat',))
    # A private copy guarantees that an external trainer cannot change behavior
    # weights while this collector owns input. It is never promoted here.
    behavior = (online_session.begin_combat(model, build_state) if online_session is not None
                else deepcopy(model).eval())
    build_state = deepcopy(build_state)
    if behavior.config.context_dim == 64 and build_state is None:
        raise ValueError('context64 collection requires observed build state')
    device = next(behavior.parameters()).device
    hidden = behavior.initial_hidden(1) if initial_hidden is None else initial_hidden.detach().clone().to(device)
    if hidden.shape != (1, behavior.config.hidden_size) or not torch.isfinite(hidden).all():
        raise ValueError('invalid initial recurrent state')
    reset_first = initial_hidden is None if reset_first is None else reset_first
    if type(reset_first) is not bool:
        raise ValueError('reset_first must be boolean')
    session = Path(output_root) / ('neural-' + uuid.uuid4().hex)
    session.mkdir(parents=True, exist_ok=False)
    build_sources = []
    if build_state is not None:
        from playmodel.learning.full_run import build_source_proofs
        build_snapshot = build_state.snapshot()
        build_sources = build_source_proofs(build_snapshot)
        _json(session / 'build-state.json', build_snapshot)
    stop_file = Path(stop_file) if stop_file else session / 'STOP'
    save_checkpoint(behavior, session / 'behavior-policy.pt', {'scope': 'bounded_movement_trial', 'split': split})
    _json(session / 'capture-config.json', {**asdict(config), 'chunk_steps': chunk_steps, 'burn_in': burn_in,
          'split': split, 'clock': CLOCK, 'context_schema': ('brotato-observed-build-v2'
              if behavior.config.context_dim == 64 else 'actual_previous_action_onehot9_remaining_unknown'),
          'initial_hidden_carried': initial_hidden is not None})
    _json(session / 'combat-entry.json', asdict(combat_entry))
    with (session / 'combat-entry-source').open('xb') as output:
        output.write(Path(combat_entry.frame_ref).read_bytes())
    if terminal_rules:
        _json(session / 'terminal-rules.json', terminal_rules.document)
    writer = _Writer(session, first_action_callback=first_action_callback,
                     action_callback=online_session.record_action if online_session is not None else None)
    stream = controller = safety = terminal_worker = sink = None
    terminal = final_frame = None
    first = None
    final_phase = 0
    reason, caught = 'startup_failed', None
    controller_guard_reason = None
    phase_frame = [None]
    phase_rejection = [None]
    phase_ended = threading.Event()
    policy_timings = []
    last_policy_frame = [None]
    worker_stopped, recorder_complete = True, False
    cleanup_errors = []
    try:
        stream = stream_factory(executable, fps=config.fps, stride=config.stride, backend='printwindow')
        startup = time.perf_counter() + config.startup_seconds
        first = None
        while first is None and time.perf_counter() < startup:
            if stop_file.exists() or stream.error:
                raise OSError('stop requested or capture failed during startup')
            first = stream.latest(max_age_ms=config.max_frame_age_ms)
            if first is None:
                time.sleep(.005)
        if first is None or not _frame_ok(first, now=time.perf_counter_ns(), max_age_ms=config.max_frame_age_ms):
            raise ValueError('fresh startup frame required')
        if _recovery is None and not 0 <= first.metadata['capture_started_at_ns'] - combat_entry.observed_at_ns <= 60_000_000_000:
            raise ValueError('combat entry must precede startup by at most 60 seconds')
        hwnd = first.metadata['hwnd']
        background = background_factory(hwnd, executable)
        background.release()
        sink = NeuralMovementSink(background, writer, hwnd, hidden, reset_first=reset_first,
                                  online_session=online_session)
        vision = vision_factory()
        if _recovery is not None:
            first, combat_entry = _reacquire_combat(stream, background, vision, recovery=_recovery,
                config=config, stop_file=stop_file, directory=session)
        rng = torch.Generator(device=device).manual_seed(config.seed)
        # Complete lazy kernel setup before acquiring timed input authority.
        with torch.no_grad():
            warm = behavior.step(*_inputs(behavior, first, 0, build_state=build_state), hidden=hidden)
            warm.value.cpu().tolist()

        def policy(observation, deadline):
            frame = observation.payload
            last_policy_frame[0] = frame
            timing = {'sequence': observation.sequence, 'observed_at_ns': observation.observed_at_ns,
                      'available_at_ns': observation.available_at_ns, 'deadline_ns': deadline,
                      'started_at_ns': time.perf_counter_ns(), 'stage': 'vision', 'outcome': 'running'}
            policy_timings.append(timing)  # Memory only; no logging/filesystem in this callback.
            try:
                state = vision.observe(frame.pixels, frame.metadata['sample_width'], frame.metadata['sample_height'],
                                       observed_at_ns=observation.observed_at_ns, alpha_mode='ignore')
                timing['vision_finished_at_ns'] = time.perf_counter_ns()
                if not state.combat_likely:
                    phase_frame[0] = frame
                    phase_rejection[0] = {
                        'status': getattr(state, 'status', 'unknown'), 'combat_likely': False,
                        'hp_fill_fraction': getattr(state, 'hp_fill_fraction', None),
                        'extractor_version': getattr(state, 'extractor_version', None),
                        'rejected_at_ns': time.perf_counter_ns(),
                    }
                    timing['outcome'] = 'phase_rejected'
                    phase_ended.set()
                    return None
                timing['stage'] = 'scene_callback'
                if scene_callback:
                    scene_callback('combat')
                timing['callback_finished_at_ns'] = time.perf_counter_ns()
                timing['stage'] = 'input_tensors'
                reset = sink.sent_count == 0 and sink.reset_first
                chosen, version, segment = behavior, None, None
                before = sink.hidden
                if online_session is not None:
                    chosen, before, reset, version, segment = online_session.inference_state(before, reset)
                inputs = _inputs(chosen, frame, sink.previous_action, build_state=build_state)
                before = before.detach().clone()
                timing['inputs_finished_at_ns'] = time.perf_counter_ns()
                timing['stage'] = 'model_and_sample'
                with torch.no_grad():
                    output = chosen.step(*inputs, hidden=before, reset=torch.tensor([reset], device=device))
                    action, logp = output.sample(generator=rng)
                timing['model_finished_at_ns'] = time.perf_counter_ns()
                timing['stage'] = 'packet'
                packet = NeuralAction(frame, tuple(item.detach().cpu() for item in inputs), before.cpu(),
                                      output.next_hidden.detach(), reset, int(action.item()), float(logp.item()),
                                      float(output.value.item()), output.probabilities[0].cpu().tolist(), 0)
                packet.behavior_version, packet.actor_segment = version, segment
                packet.behavior_model = chosen if online_session is not None else None
                packet.decided_at_ns = time.perf_counter_ns()
                timing['outcome'] = 'proposal_ready'
                return packet
            except Exception as error:
                timing['outcome'] = f'exception:{type(error).__name__}'
                raise
            finally:
                timing['finished_at_ns'] = time.perf_counter_ns()

        controller = RealtimeController(policy, sink, ControlLimits(
            int(config.max_frame_age_ms * 1e6), int(config.policy_budget_ms * 1e6), 25_000_000,
            int(config.input_watchdog_ms * 1e6), config.max_steps * 8 + 64,
            policy_deadline_retries=1))
        if _recovery is not None:
            if stop_file.exists() or stream.error:
                raise OSError('stop or capture failure before recovered authority')
            background.check()
        generation = controller.arm()
        started = time.perf_counter_ns()
        safety = SafetyMonitor(controller, sink, stream, stop_file, config, started)
        observer = terminal_observer or LocalTerminalObserver(executable, session, Path(ocr_script),
                                                              terminal_rules, reader=terminal_ocr_reader)
        terminal_worker = SlowTerminalWorker(observer)
        last_sequence = -1
        pending_sent_count = None
        requested_after_send = None
        while True:
            if safety.reason:
                reason = safety.reason
                break
            if phase_ended.is_set():
                reason = 'screen_changed'
                final_frame = phase_frame[0]
                controller.disarm(reason)
                break
            if not controller.armed:
                reason = 'controller_guard'
                controller_guard_reason = controller.disarm_reason
                caught = f'controller_guard:{controller_guard_reason or "unknown"}'
                break
            candidate = terminal_worker.latest
            if candidate is not None and candidate.observed_at_ns >= started:
                terminal, reason = candidate, f'terminal_{candidate.kind}'
                break
            if sink.sent_count >= config.max_steps:
                reason = 'step_limit'
                break
            frame = stream.latest(max_age_ms=config.max_frame_age_ms)
            current_generation = controller.generation
            if current_generation != generation:
                # Only the controller can preserve authority across a retry.
                # The expired proposal transmitted nothing: clear the publish
                # gate, not recurrent memory or the last actual action/time.
                # publish() also requires a newer capture than the failed job.
                generation = current_generation
                pending_sent_count = None
            if pending_sent_count is not None and sink.sent_count > pending_sent_count:
                pending_sent_count = None
            # RealtimeController may dispatch and submit in the same tick. Do
            # not prepublish a frame captured before that pending dispatch;
            # otherwise the next observation would precede the previous action.
            # While the expired callback drains, replacement is safe. Retain
            # the pending count even on iterations with no newer frame: tick
            # may drain the old callback and submit that saved frame together.
            if ((pending_sent_count is None or controller.awaiting_expired_policy)
                    and frame is not None and frame.sequence > last_sequence):
                if not _frame_ok(frame, now=time.perf_counter_ns(), max_age_ms=config.max_frame_age_ms, hwnd=hwnd):
                    raise ValueError('fresh frame contract violated')
                if sink.last_sent_at_ns is None or frame.metadata['capture_started_at_ns'] > sink.last_sent_at_ns:
                    accepted = controller.publish(Observation(frame.sequence, generation, frame.metadata['capture_started_at_ns'],
                                                               frame.available_at_ns, frame))
                    if accepted.accepted:
                        last_sequence = frame.sequence
                        pending_sent_count = sink.sent_count
            controller.tick()
            if sink.last_sent_at_ns is not None and sink.last_sent_at_ns != requested_after_send:
                request = getattr(stream, 'request_fresh', None)
                if request is not None:
                    request(sink.last_sent_at_ns)
                requested_after_send = sink.last_sent_at_ns
            time.sleep(config.tick_ms / 1000)
        # Obtain a real post-action observation before normal bounded release.
        if reason == 'step_limit' and sink.packets:
            until = time.perf_counter() + min(.15, config.input_watchdog_ms / 2000)
            while time.perf_counter() < until and not safety.reason:
                candidate = stream.latest(max_age_ms=config.max_frame_age_ms)
                if candidate and candidate.metadata['capture_started_at_ns'] > sink.packets[-1].sent_at_ns:
                    final_frame = candidate
                    break
                time.sleep(.002)
        elif reason == 'time_limit':
            final_frame = stream.latest(max_age_ms=config.max_frame_age_ms)
        controller.disarm(reason)
        if reason == 'screen_changed':
            until = time.perf_counter() + config.terminal_wait_seconds
            while time.perf_counter() < until and not safety.reason and not stop_file.exists():
                candidate = terminal_worker.latest
                if candidate is not None and candidate.observed_at_ns >= started:
                    terminal, reason = candidate, f'terminal_{candidate.kind}'
                    break
                time.sleep(.01)
        if terminal is not None:
            _evidence(terminal, ('death', 'wave_clear'), rules=terminal_rules)
            pixels, width, height = read_diagnostic_png(Path(terminal.frame_ref))
            if terminal.kind == 'wave_clear':
                raw_ocr_path = Path(terminal.frame_ref).parent / 'ocr.json'
                raw_ocr = json.loads(raw_ocr_path.read_text(encoding='utf-8'))
                scene = classify_scene(raw_ocr['raw'], width=width, height=height).scene
                # V2 uses the same candidate-scoring phase for upgrades and
                # verified loot choices; this is bound in the runtime contract.
                final_phase = {'level_up': 1, 'loot': 1, 'shop': 3}.get(scene)
                if final_phase is None:
                    raise ValueError('wave boundary menu phase is unknown; cannot bootstrap')
                _json(session / 'terminal-ocr.json', raw_ocr)
            final_frame = Frame(last_sequence + 1, {**first.metadata, 'sample_width': width, 'sample_height': height,
                'capture_started_at_ns': terminal.observed_at_ns, 'capture_finished_at_ns': terminal.observed_at_ns},
                pixels, terminal.available_at_ns)
    except Exception as error:
        reason, caught = 'runtime_error', f'{type(error).__name__}: {error}'
    finally:
        if scene_callback:
            scene_callback('unknown')
        if safety:
            safety.close()
            if safety.reason and safety.reason != 'time_limit':
                reason = safety.reason
        if controller:
            worker_stopped = controller.close(timeout=.1)
        elif sink:
            sink.release(generation=0, reason='startup_failure', deadline_ns=time.perf_counter_ns() + 25_000_000)
        if terminal_worker:
            terminal_worker.close()
        if stream:
            try:
                stream.close()
            except Exception as error:
                cleanup_errors.append(f'capture_close:{type(error).__name__}')
        recorder_complete = writer.close()
    rollout_path = flat_rollout_path = initial_states_path = None
    last_hidden = sink.hidden.detach().cpu() if sink else hidden.detach().cpu()
    events = [asdict(event) for event in controller.events()] if controller else []
    policy_retries = [event for event in events if event['kind'] == 'retry']
    completed_retries = [event for event in events if event['kind'] == 'retry_completed']
    _json(session / 'control-events.json', events)
    _json(session / 'input-attempts.json', sink.transport_attempts if sink else [])
    # A still-running callback remains an incomplete timing record. It is not
    # marked transmitted or committed, and the frozen copy cannot be rewritten.
    timing_snapshot = [dict(row) for row in policy_timings]
    _json(session / 'policy-timings.json', {'clock_domain': CLOCK, 'records': timing_snapshot,
                                         'worker_stopped': worker_stopped,
                                         'scope': 'policy_compute_not_transmission_or_game_application'})
    guard_frame_path = None
    if controller_guard_reason and last_policy_frame[0] is not None:
        guard_frame = last_policy_frame[0]
        guard_frame_path = session / 'guard-frame.bgra'
        guard_frame_path.write_bytes(guard_frame.pixels)
        _json(session / 'guard-frame.json', {'sequence': guard_frame.sequence, 'metadata': guard_frame.metadata,
            'available_at_ns': guard_frame.available_at_ns, 'frame_ref': guard_frame_path.name,
            'frame_sha256': hashlib.sha256(guard_frame.pixels).hexdigest(),
            'scope': 'latest_started_policy_observation_not_an_action_or_training_label'})
    if phase_frame[0] is not None:
        # Preserve the exact rejected observation, including failed/unknown
        # trials. A later terminal screenshot cannot explain this decision.
        rejected = phase_frame[0]
        (session / 'phase-rejection.bgra').write_bytes(rejected.pixels)
        (session / 'phase-rejection.png').write_bytes(_png(
            rejected.metadata['sample_width'], rejected.metadata['sample_height'], rejected.pixels))
        _json(session / 'phase-rejection.json', {
            'sequence': rejected.sequence, 'metadata': rejected.metadata,
            'available_at_ns': rejected.available_at_ns,
            'frame_ref': 'phase-rejection.bgra',
            'frame_sha256': hashlib.sha256(rejected.pixels).hexdigest(),
            'decision': 'abstain', 'vision': phase_rejection[0], 'clock_domain': CLOCK,
        })
    if terminal:
        _json(session / 'terminal.json', asdict(terminal))
        with (session / 'terminal-source.png').open('xb') as output:
            output.write(Path(terminal.frame_ref).read_bytes())
    eligible = (reason in ('terminal_death', 'terminal_wave_clear', 'step_limit', 'time_limit')
                and recorder_complete and worker_stopped and sink is not None and sink.packets
                and not sink.release_failed and not cleanup_errors and final_frame is not None
                and len(policy_retries) == len(completed_retries)
                and sum(event['kind'] == 'expired' for event in events) == len(policy_retries)
                and not any((event['kind'] == 'rejected' and not
                             (event['reason'] == 'policy_abstained' and terminal is not None
                              and phase_frame[0] is not None and event['sequence'] == phase_frame[0].sequence)) or
                            (event['kind'] == 'dispatch' and event['reason'] != 'transmitted') for event in events))
    if eligible and reason == 'time_limit':
        releases = [event['send_started_at_ns'] for event in events if event['kind'] == 'release'
                    and event['send_started_at_ns'] >= sink.packets[-1].sent_at_ns]
        if not releases or final_frame.metadata['capture_started_at_ns'] > min(releases):
            eligible = False
            caught = 'no_post_action_final_observation_before_time_limit_release'
    if online_session is not None:
        online_session.end_combat(frame=final_frame, phase=final_phase, terminal=terminal, eligible=bool(eligible))
    if eligible and online_session is None:
        try:
            _json(session / 'final-observation.json', {**final_frame.metadata, 'available_at_ns': final_frame.available_at_ns})
            with (session / 'final-observation.bgra').open('xb') as output:
                output.write(final_frame.pixels)
            batch, flat_batch, states = _batch(behavior, sink.packets, final_frame, terminal=terminal,
                                        final_phase=final_phase, rollout_id=session.name, split=split,
                                        chunk_steps=chunk_steps, burn_in=burn_in, build_state=build_state)
            rollout_path = session / 'rollout.pt'
            with rollout_path.open('xb') as output:
                torch.save({'schema': SCHEMA, 'batch': batch.__dict__}, output)
            flat_rollout_path = session / 'flat-rollout.pt'
            with flat_rollout_path.open('xb') as output:
                torch.save({'schema': SCHEMA, 'batch': flat_batch.__dict__}, output)
            initial_states_path = session / 'initial-states.pt'
            with initial_states_path.open('xb') as output:
                torch.save({'schema': SCHEMA, 'initial_states': states}, output)
        except Exception as error:
            caught = f'rollout_rejected:{type(error).__name__}: {error}'
            rollout_path = flat_rollout_path = initial_states_path = None
    report = {'session_directory': str(session.resolve()), 'reason': reason, 'error': caught,
              'capture_error': stream.error if stream is not None else None,
              'safety_reason': safety.reason if safety is not None else None,
              'target_identity': {key: first.metadata.get(key) for key in ('hwnd', 'pid', 'executable')}
                                 if first is not None else None,
              'controller_guard_reason': controller_guard_reason,
              'policy_timings_path': str((session / 'policy-timings.json').resolve()),
              'guard_frame_path': str(guard_frame_path.resolve()) if guard_frame_path else None,
              'policy_deadline_retries': len(policy_retries),
              'policy_deadline_retries_completed': len(completed_retries),
              'deadline_retry_contract': 'one_fresh_frame_within_unchanged_previous_input_deadlines',
              'steps': sink.sent_count if sink else 0, 'rollout_path': str(rollout_path.resolve()) if rollout_path else None,
              'flat_rollout_path': str(flat_rollout_path.resolve()) if flat_rollout_path else None,
              'initial_states_path': str(initial_states_path.resolve()) if initial_states_path else None,
              'behavior_version': behavior.policy_version(), 'training_performed': False,
              'runtime_contract': RUNTIME_CONTRACT, 'phase_schema': PHASE_SCHEMA,
              'scope': 'online_versioned_combat' if online_session is not None else 'bounded_movement_trial',
              'full_run_complete': False, 'menu_heads_used': False,
              'acknowledgement': 'unknown', 'game_application_verified': False,
              'reward_schema': 'independent_death_minus1_wave_clear_plus1_other_zero',
              'terminal_kind': terminal.kind if terminal else None, 'final_phase': final_phase,
              'chunk_steps': chunk_steps, 'burn_in': burn_in,
              'gae_scope': 'chunk_traces_with_true_next_value_bootstrap', 'split': split,
              'last_hidden': last_hidden.tolist(), 'final_hidden': last_hidden.tolist(),
              'hidden_scope': 'after_last_transmitted_action_observation_before_bootstrap',
              'recorder_complete': recorder_complete,
              'rollout_eligible': bool(eligible) and online_session is None,
              'worker_stopped': worker_stopped, 'cleanup_errors': cleanup_errors}
    if online_session is not None:
        report.update(online_collection_eligible=bool(eligible), online_session=online_session.snapshot(),
                      behavior_versions=list(dict.fromkeys(packet.behavior_version for packet in sink.packets))
                        if sink else [])
    _json(session / 'report.json', report)
    # Disk logging stays outside the timed policy/dispatch loop. The durable
    # controller ledger includes the failed job even when it was still running.
    from playmodel.execution_log import event as execution_event
    if policy_retries or controller_guard_reason:
        execution_event('neural_controller_diagnostics', session_directory=str(session.resolve()),
                        reason=reason, error=caught, controller_guard_reason=controller_guard_reason,
                        control_events_path=str((session / 'control-events.json').resolve()),
                        policy_timings_path=report['policy_timings_path'],
                        guard_frame_path=report['guard_frame_path'],
                        latest_policy_timing=timing_snapshot[-1] if timing_snapshot else None,
                        retries=policy_retries, completed_retries=len(completed_retries),
                        failures=[event for event in events if event['kind'] in ('expired', 'discarded')],
                        actual_transmissions=report['steps'], rollout_eligible=bool(eligible))
    # A slow OCR call may finish after its stop request. Only this collector's
    # frozen files enter the manifest; its verified source image is copied above.
    files = [{'path': str(path.relative_to(session)).replace('\\', '/'), 'sha256': _sha(path)}
             for path in sorted(session.rglob('*')) if path.is_file() and path.relative_to(session).parts[0] != 'ocr']
    _json(session / 'manifest.json', {'schema': SCHEMA, 'behavior_version': behavior.policy_version(),
                                     'runtime_contract': RUNTIME_CONTRACT, 'phase_schema': PHASE_SCHEMA,
                                     'recorder_complete': recorder_complete and worker_stopped, 'files': files,
                                     'sources': build_sources})
    return report
