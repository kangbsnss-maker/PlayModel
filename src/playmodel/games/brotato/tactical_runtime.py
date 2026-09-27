"""CPU tactical execution with asynchronous local choices and actual-input receipts.

The neural collector supplies capture, authority, STOP and terminal verification.
These actions contain no recurrent state, value estimate, or PPO probability.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import time
import uuid

from .neural_runtime import CLOCK, MOVEMENT, NeuralMovementSink


def movement_keys(action):
    dx, dy = MOVEMENT[action]
    return sorted(([0x44 if dx > 0 else 0x41] if dx else []) +
                  ([0x53 if dy > 0 else 0x57] if dy else []))


@dataclass
class TacticalMovementAction:
    frame: object
    action: int
    decided_at_ns: int
    decision: dict | None
    situation_signature: str
    fallback_reason: str | None = None
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    legal_mask: tuple = (True,) * 9
    sent_at_ns: int | None = None
    generation: int | None = None
    transport_started_at_ns: int | None = None
    receipt: dict | None = None
    held_before: tuple = ()


class TacticalMovementSink(NeuralMovementSink):
    """Same bounded native transport as CNN; a separate non-tensor commit path."""
    def __init__(self, background, writer, hwnd):
        super().__init__(background, writer, hwnd, None)
        self.previous_execution_id = None

    def _valid_packet(self, action):
        if not isinstance(action, TacticalMovementAction):
            return False
        if action.decision is not None:
            expires = action.decision.get('expires_at_ns')
            if type(expires) is not int or time.perf_counter_ns() >= expires:
                return False
            if action.decision.get('signature') != action.situation_signature:
                return False
        return True

    def send(self, action, **kwargs):
        if self._valid_packet(action):
            action.held_before = tuple(sorted(self.background.held))
        return super().send(action, **kwargs)

    def _commit_packet(self, action, attempt):
        stages = dict(attempt.get('background_stages') or {})
        if stages.get('call_id') != attempt['expected_background_call_id']:
            raise OSError('tactical movement lacks current native-call receipt')
        after = tuple(sorted(self.background.held))
        if after != tuple(movement_keys(action.action)):
            raise OSError('tactical movement held keys differ from requested action')
        posts = stages.get('posted_keys')
        if type(posts) is not int or posts < 0 or stages.get('identity_check_passed') is not True:
            raise OSError('tactical movement lacks native identity/post proof')
        kind = 'native_transition' if posts else 'existing_owned_hold' if after else 'neutral_hold'
        if kind == 'existing_owned_hold' and self.previous_execution_id is None:
            raise OSError('held movement has no earlier owned execution')
        action.receipt = {
            'execution_id': action.execution_id, 'previous_execution_id': self.previous_execution_id,
            'actual_movement': action.action, 'actual_keys': list(after),
            'held_before': list(action.held_before), 'held_after': list(after),
            'native_post_count': posts, 'execution_kind': kind,
            'deadline_ns': attempt['deadline_ns'], 'background_stages': stages,
            'target_identity': {key: action.frame.metadata.get(key) for key in ('hwnd', 'pid', 'executable')},
        }
        self.previous_execution_id = action.execution_id

    def release(self, **kwargs):
        result = super().release(**kwargs)
        self.previous_execution_id = None
        return result


def action_record(packet):
    decision = packet.decision or {}
    return {
        **packet.receipt, 'schema': 'playmodel.tactical-execution.v1', 'domain': 'combat_tactic',
        'sequence': packet.frame.sequence, 'generation': packet.generation,
        'frame_ref': f'frames/{packet.frame.sequence:08d}.bgra',
        'frame_sha256': hashlib.sha256(packet.frame.pixels).hexdigest(),
        'observed_at_ns': packet.frame.metadata['capture_started_at_ns'],
        'available_at_ns': packet.frame.available_at_ns, 'decided_at_ns': packet.decided_at_ns,
        'sent_at_ns': packet.sent_at_ns, 'verified_at_ns': packet.sent_at_ns,
        'transport_started_at_ns': packet.transport_started_at_ns,
        'transport_finished_at_ns': packet.sent_at_ns,
        'decision_id': decision.get('decision_id'), 'action_id': decision.get('action_id'),
        'behavior_version': decision.get('behavior_version'), 'epoch': decision.get('epoch'),
        'signature': packet.situation_signature, 'decision_source': decision,
        'action_origin': 'local_laya' if decision else 'explicit_rule_fallback',
        'fallback_reason': packet.fallback_reason, 'transmitted': True, 'acknowledged': None,
        'accepted': True, 'successful_transport_reported': True,
        'game_application_verified': False, 'clock_domain': CLOCK,
        'cnn_training_eligible': False, 'legal_mask': packet.legal_mask,
    }


class TacticalCombatActor:
    """Fast geometry calls only nonblocking broker operations; writer persists proof."""
    action_record = staticmethod(action_record)

    def __init__(self, session, *, planner=None):
        if planner is None:
            from .tactical_state import TacticalPlanner
            planner = TacticalPlanner()
        self.session, self.planner = session, planner
        self.receipt_refs = {}
        self.ownership_receipt = None
        self.active = False

    def begin_combat(self, session_id):
        self.receipt_refs = {}
        self.ownership_receipt = None
        self.planner.reset()
        self.session.begin_combat()
        self.active = True

    def propose(self, frame, vision_state, *, build_state=None):
        from .tactical_state import execute_tactic, fallback_movement
        if self.session.error:
            raise OSError('tactical broker failed: ' + str(self.session.error))
        situation = self.planner.observe(vision_state,
            observed_at_ns=frame.metadata['capture_started_at_ns'], available_at_ns=frame.available_at_ns,
            build_state=build_state,
            aspect_ratio=frame.metadata['sample_width'] / frame.metadata['sample_height'])
        self.session.offer(frame, situation)
        decision = self.session.resolve(situation, frame)
        now = time.perf_counter_ns()
        world = situation.get('world', {})
        if (world.get('valid') is not True or not world.get('available_at_ns', now+1) <= now
                <= world.get('fresh_until_ns', -1)
                or (decision is not None and decision.get('action_id') not in situation['options'])):
            decision = None
        movement = (execute_tactic(decision['action_id'], situation, now_ns=now) if decision is not None
                    else fallback_movement(situation, now_ns=now))
        if type(movement) is not int or not 0 <= movement < 9:
            raise ValueError('tactical planner returned invalid movement')
        return TacticalMovementAction(frame, movement, time.perf_counter_ns(),
            deepcopy(decision), situation['signature'], None if decision else 'no_current_valid_local_choice')

    def record_action(self, packet, record, directory):
        # Recorder thread only, after source pixels and JSONL are durable.
        receipt = dict(record)
        receipt['frame_ref'] = str((directory / record['frame_ref']).resolve())
        receipt['currentframe_ref'] = receipt['frame_ref']
        receipt['currentframe_sha256'] = receipt['frame_sha256']
        previous = self.receipt_refs.get(record['previous_execution_id'])
        receipt['previous_receipt_path'] = previous[0] if previous else None
        receipt['previous_receipt_sha256'] = previous[1] if previous else None
        anchor = self.ownership_receipt if record['execution_kind'] == 'existing_owned_hold' else None
        receipt['ownership_receipt_path'] = anchor[0] if anchor else None
        receipt['ownership_receipt_sha256'] = anchor[1] if anchor else None
        receipt['run_id'] = self.session.run_id
        receipt['epoch'] = receipt.get('epoch') or self.session.epoch
        receipt['writer_authority'] = 'single_ai_writer'
        path = directory / ('receipt-' + packet.execution_id + '.json')
        raw = json.dumps(receipt, allow_nan=False, sort_keys=True).encode('utf-8')
        with path.open('xb') as stream:
            stream.write(raw)
        self.receipt_refs[packet.execution_id] = (str(path.resolve()), hashlib.sha256(raw).hexdigest())
        if record['execution_kind'] == 'native_transition':
            self.ownership_receipt = self.receipt_refs[packet.execution_id]
        if packet.decision is not None:
            self.session.record_execution({**receipt, 'receipt_path': str(path.resolve()),
                'receipt_sha256': hashlib.sha256(raw).hexdigest()}, packet.frame)

    def end_combat(self, *, eligible):
        if self.active:
            self.session.end_combat(valid=eligible, reason='verified_boundary' if eligible else 'unsafe_or_gapped_combat')
            self.active = False
