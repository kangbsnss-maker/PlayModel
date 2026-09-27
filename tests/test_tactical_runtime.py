"""Tactical integration with real realtime authority and fake native transport."""
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
import importlib.util
from unittest.mock import Mock, patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')

from playmodel.games.brotato.capture import _png
from playmodel.games.brotato.neural_runtime import run_neural_trial, _definitely_unsent_deadline, _Writer
from playmodel.games.brotato.pilot import CLOCK, PilotConfig, SafetyMonitor
from playmodel.games.brotato.stream import Frame
from playmodel.games.brotato.tactical_runtime import (
    TacticalMovementAction, TacticalMovementSink, TacticalCombatActor, action_record,
)
from playmodel.learning import StateEvidence


def frame(sequence=1):
    now = time.perf_counter_ns()
    return Frame(sequence, {'hwnd': 1, 'pid': 2, 'executable': 'fake.exe',
        'capture_started_at_ns': now, 'capture_finished_at_ns': now,
        'sample_width': 16, 'sample_height': 16,
        'backend': 'win32_printwindow_client', 'clock': CLOCK}, bytes([1, 2, 3, 255]) * 256, now)


class Background:
    def __init__(self, *args):
        self.held = set()
        self._movement_call_id = 0
        self.calls = 0

    def set_movement_before(self, keys, *, deadline_ns):
        started = time.perf_counter_ns()
        self.calls += 1
        self._movement_call_id += 1
        posts = len(keys ^ self.held)
        self.last_movement_timing = {'call_id': self._movement_call_id,
            'identity_check_passed': True, 'posted_keys': posts,
            'native_post_attempted': bool(posts), 'attempted_posts': posts,
            'started_at_ns': started, 'check_finished_at_ns': started,
            'posts_started_at_ns': started if posts else None,
            'posts_finished_at_ns': time.perf_counter_ns(), 'deadline_ns': deadline_ns}
        self.held = set(keys)

    def check(self):
        pass

    def release(self):
        self.held.clear()


class TacticalRuntimeTests(unittest.TestCase):
    def sink(self):
        writer = SimpleNamespace(error=None, pending=queue.Queue(4))
        return TacticalMovementSink(Background(), writer, 1)

    def send(self, sink, movement=3, sequence=1):
        packet = TacticalMovementAction(frame(sequence), movement, time.perf_counter_ns(), None, 'sig')
        sink.send(packet, generation=1, observation_sequence=sequence,
                  deadline_ns=time.perf_counter_ns() + 25_000_000)
        return packet

    def test_actual_native_and_existing_hold_without_repeat_key_posts(self):
        sink = self.sink()
        first = self.send(sink)
        second = self.send(sink, sequence=2)
        self.assertEqual(first.receipt['execution_kind'], 'native_transition')
        self.assertEqual(second.receipt['execution_kind'], 'existing_owned_hold')
        self.assertEqual(second.receipt['native_post_count'], 0)
        self.assertEqual(second.receipt['previous_execution_id'], first.execution_id)
        record = action_record(second)
        self.assertFalse(record['game_application_verified'])
        for key in ('hidden_before', 'old_value', 'old_log_probability', 'probabilities'):
            self.assertNotIn(key, record)

    def test_initial_neutral_hold_is_honest_no_post(self):
        packet = self.send(self.sink(), movement=0)
        self.assertEqual(packet.receipt['execution_kind'], 'neutral_hold')
        self.assertEqual(packet.receipt['native_post_count'], 0)

    def test_full_recorder_queue_fails_closed_before_native_call(self):
        sink = self.sink()
        for _ in range(4):
            sink.writer.pending.put_nowait(object())
        with self.assertRaises(OSError):
            self.send(sink)
        self.assertEqual(sink.background.calls, 0)

    def test_wrong_frame_and_illegal_action_never_send(self):
        sink = self.sink()
        packet = TacticalMovementAction(frame(), 3, time.perf_counter_ns(), None, 'sig')
        with self.assertRaises(OSError):
            sink.send(packet, generation=1, observation_sequence=2, deadline_ns=time.perf_counter_ns()+25_000_000)
        packet.legal_mask = (False,) * 9
        with self.assertRaises(OSError):
            sink.send(packet, generation=1, observation_sequence=1, deadline_ns=time.perf_counter_ns()+25_000_000)
        self.assertEqual(sink.background.calls, 0)

    def test_choice_expiring_between_proposal_and_send_is_rejected(self):
        sink = self.sink()
        packet = TacticalMovementAction(frame(), 3, time.perf_counter_ns(),
            {'expires_at_ns': time.perf_counter_ns()-1, 'signature': 'sig'}, 'sig')
        with self.assertRaises(OSError):
            sink.send(packet, generation=1, observation_sequence=1,
                      deadline_ns=time.perf_counter_ns()+25_000_000)
        self.assertEqual(sink.background.calls, 0)

    def test_stop_file_and_f8_release_held_tactical_input(self):
        from playmodel.control import RealtimeController, ControlLimits
        for reason in ('stop_file', 'F8'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp:
                sink = self.sink()
                controller = RealtimeController(lambda *_: None, sink,
                    ControlLimits(250_000_000,100_000_000,25_000_000,250_000_000))
                controller.arm()
                self.send(sink)
                stop = Path(temp)/'STOP'
                if reason == 'stop_file':
                    stop.touch()
                else:
                    sink.background.check = Mock(side_effect=OSError('F8 requested'))
                monitor = SafetyMonitor(controller, sink,
                    SimpleNamespace(error=None, latest=lambda **_: frame()), stop,
                    PilotConfig(), time.perf_counter_ns())
                until = time.perf_counter()+.5
                while monitor.reason is None and time.perf_counter()<until:
                    time.sleep(.002)
                monitor.close()
                controller.close()
                self.assertEqual(monitor.reason, reason)
                self.assertEqual(sink.background.held, set())

    def test_small_native_reserve_recovery_proof_requires_fixed_bound(self):
        proof = {'started_at_ns': 11, 'check_finished_at_ns': 20,
            'deadline_checked_at_ns': 21, 'deadline_ns': 1000, 'call_id': 1,
            'identity_check_passed': True, 'native_post_attempted': False,
            'attempted_posts': 0, 'posted_keys': 0, 'posts_started_at_ns': None,
            'posts_finished_at_ns': None, 'cancellation_reason': 'post_budget_insufficient',
            'minimum_post_budget_ns': 2_000_000}
        attempt = {'error': 'PrePostMovementDeadline', 'transmitted': False,
            'delivery_certainty': 'definitely_not_sent', 'cancellation_proof': proof,
            'background_stages': proof, 'expected_background_call_id': 1,
            'transport_started_at_ns': 10, 'transport_finished_at_ns': 30,
            'deadline_ns': 1000, 'sequence': 1, 'generation': 1}
        events = [{'kind': 'dispatch', 'reason': 'input_error:PrePostMovementDeadline',
            'sequence': 1, 'generation': 1, 'sink_deadline_ns': 1000,
            'send_started_at_ns': 9, 'send_finished_at_ns': 31}]
        self.assertTrue(_definitely_unsent_deadline(attempt, events))
        proof['minimum_post_budget_ns'] = 3_000_000
        self.assertFalse(_definitely_unsent_deadline(attempt, events))
        proof['minimum_post_budget_ns'] = 2_000_000
        proof['deadline_ns'] = attempt['deadline_ns'] = events[0]['sink_deadline_ns'] = 2_000_021
        self.assertFalse(_definitely_unsent_deadline(attempt, events))

    def actor(self, decision=None):
        session = SimpleNamespace(error=None, run_id='run', epoch='epoch', offer=Mock(), resolve=Mock(return_value=decision),
            record_execution=Mock(), begin_combat=Mock(), end_combat=Mock())
        now = time.perf_counter_ns()
        planner = SimpleNamespace(reset=Mock(), observe=Mock(return_value={
            'signature': 'sig', 'options': {'retreat': 'retreat'},
            'world': {'valid': True, 'available_at_ns': now, 'fresh_until_ns': now+5_000_000_000}}))
        return TacticalCombatActor(session, planner=planner)

    def test_missing_or_stale_whole_request_uses_explicit_fallback(self):
        actor = self.actor()
        with patch('playmodel.games.brotato.tactical_state.fallback_movement', return_value=7), \
             patch('playmodel.games.brotato.tactical_state.execute_tactic') as execute:
            packet = actor.propose(frame(), object())
        self.assertEqual(packet.action, 7)
        self.assertIsNone(packet.decision)
        execute.assert_not_called()
        actor.session.offer.assert_called_once()

    def test_new_geometry_recomputes_direction_for_same_semantic_choice(self):
        actor = self.actor({'action_id': 'retreat', 'decision_id': 'd'})
        with patch('playmodel.games.brotato.tactical_state.execute_tactic', side_effect=[3, 7]):
            first = actor.propose(frame(), object())
            second = actor.propose(frame(2), object())
        self.assertEqual((first.action, second.action), (3, 7))

    def test_invalid_world_neutral_is_not_labeled_as_laya_execution(self):
        actor = self.actor({'action_id': 'retreat', 'decision_id': 'd'})
        actor.planner.observe.return_value['world']['valid'] = False
        with patch('playmodel.games.brotato.tactical_state.execute_tactic') as execute, \
             patch('playmodel.games.brotato.tactical_state.fallback_movement', return_value=0):
            packet = actor.propose(frame(), object())
        execute.assert_not_called()
        self.assertIsNone(packet.decision)
        self.assertEqual(packet.action, 0)

    def test_writer_persists_native_ownership_anchor_before_accepting_hold(self):
        actor, sink = self.actor(), self.sink()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = self.send(sink)
            actor.record_action(first, action_record(first), directory)
            second = self.send(sink, sequence=2)
            second.decision = {'decision_id': 'd', 'action_id': 'retreat'}
            actor.record_action(second, action_record(second), directory)
            receipt = actor.session.record_execution.call_args.args[0]
            anchor = Path(receipt['ownership_receipt_path'])
            self.assertEqual(hashlib.sha256(anchor.read_bytes()).hexdigest(), receipt['ownership_receipt_sha256'])
            self.assertEqual(json.loads(anchor.read_text())['execution_kind'], 'native_transition')

    def test_broker_failure_fails_closed_and_gap_abandons_epoch(self):
        actor = self.actor()
        actor.begin_combat('trial')
        actor.planner.reset.assert_called_once()
        actor.session.error = 'receipt queue full'
        with self.assertRaises(OSError):
            actor.propose(frame(), object())
        actor.session.offer.assert_not_called()
        actor.end_combat(eligible=False)
        actor.session.end_combat.assert_called_once_with(valid=False, reason='unsafe_or_gapped_combat')

    def test_actual_writer_receipts_validate_fallback_native_then_laya_hold_chain(self):
        from playmodel.laya.records import digest, validate_tactical_application
        actor = self.actor()
        actor.begin_combat('trial')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source_frame = frame()
            source = directory/'source.bgra'
            source.write_bytes(source_frame.pixels)
            evidence = {'decision_domain':'combat_tactic','run_id':'run','epoch':'epoch','signature':'sig',
                'target_identity': {key:source_frame.metadata[key] for key in ('hwnd','pid','executable')},
                'frame_ref':str(source),'frame_sha256':digest(source),
                'observed_at_ns':source_frame.metadata['capture_started_at_ns'],
                'available_at_ns':source_frame.available_at_ns,
                'expires_at_ns':source_frame.available_at_ns+1_500_000_000}
            world = {'state':{},'options':{'retreat':'retreat'},'signature':'sig',
                     'world':{'valid':True,'observed_at_ns':evidence['observed_at_ns'],
                              'available_at_ns':evidence['available_at_ns']}}
            observation_path = directory/'observation.json'
            observation_path.write_text(json.dumps({'evidence':evidence,'metadata':source_frame.metadata,
                                                    'situation':world}),encoding='utf8')
            evidence = {**evidence,'observation_path':str(observation_path),'observation_sha256':digest(observation_path)}
            choice_time = time.perf_counter_ns()
            decision = {'decision_id':'choice','action_id':'retreat','behavior_version':'v1',
                        'epoch':'epoch','signature':'sig','expires_at_ns':evidence['expires_at_ns']}
            choice = {**decision,'decided_at_ns':choice_time,'decision_domain':'combat_tactic',
                      'state':world['state'],'options':world['options'],'evidence':evidence}
            writer = _Writer(directory,action_callback=actor.record_action,record_factory=action_record)
            sink = TacticalMovementSink(Background(),writer,1)
            packets = []
            for sequence in (1,2,3):
                packet = TacticalMovementAction(frame(sequence),3,time.perf_counter_ns(),
                    None if sequence==1 else dict(decision),'sig')
                sink.send(packet,generation=1,observation_sequence=sequence,
                          deadline_ns=time.perf_counter_ns()+25_000_000)
                packets.append(packet)
            self.assertTrue(writer.close(),writer.error)
            self.assertEqual(actor.session.record_execution.call_count,2)
            for invocation in actor.session.record_execution.call_args_list:
                receipt = invocation.args[0]
                validated = validate_tactical_application(choice,receipt,time.perf_counter_ns())
                self.assertEqual(validated['execution_kind'],'existing_owned_hold')
                self.assertEqual(validated['native_post_count'],0)
                self.assertFalse(validated['game_application_verified'])
            anchor = json.loads((directory/f'receipt-{packets[0].execution_id}.json').read_text())
            self.assertEqual(anchor['action_origin'],'explicit_rule_fallback')
            self.assertEqual(anchor['epoch'],'epoch')

    def test_collector_uses_real_lifecycle_but_never_constructs_ppo(self):
        actor = self.actor()
        class Stream:
            error = None
            def __init__(self, *args, **kwargs):
                self.sequence = 0
            def latest(self, **kwargs):
                self.sequence += 1
                return frame(self.sequence)
            def close(self):
                pass
        class Vision:
            def observe(self, *args, **kwargs):
                return SimpleNamespace(combat_likely=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'combat.png'
            source.write_bytes(_png(16, 16, frame().pixels))
            now = time.perf_counter_ns()
            entry = StateEvidence('combat', str(source), hashlib.sha256(source.read_bytes()).hexdigest(),
                now, now, now, 'developer_verified', 'fixture', True, True)
            with patch('playmodel.games.brotato.tactical_state.fallback_movement', return_value=3), \
                 patch('playmodel.games.brotato.neural_runtime._batch', side_effect=AssertionError('PPO forbidden')):
                report = run_neural_trial(root/'fake.exe', root/'trials', model=None,
                    combat_entry=entry, combat_actor=actor, mixed_control=True,
                    config=PilotConfig(max_steps=3,max_seconds=2,startup_seconds=1,tick_ms=2),
                    terminal_observer=lambda: None,stream_factory=Stream,
                    background_factory=Background,vision_factory=Vision)
            self.assertEqual(report['reason'], 'step_limit', report)
            self.assertEqual(report['steps'], 3)
            self.assertFalse(report['rollout_eligible'])
            self.assertIsNone(report['final_hidden'])
            self.assertEqual(report['run_id'], 'run')
            self.assertEqual(report['tactical_epoch'], 'epoch')
            self.assertEqual(len(report['tactical_receipts']), 3)
            self.assertEqual(hashlib.sha256(Path(report['actions_path']).read_bytes()).hexdigest(), report['actions_sha256'])
            self.assertFalse(list(Path(report['session_directory']).glob('*.pt')))
            records = [json.loads(row) for row in (Path(report['session_directory'])/'actions.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['domain']=='combat_tactic' for row in records))


if __name__ == '__main__':
    unittest.main()
