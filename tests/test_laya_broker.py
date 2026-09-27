"""Broker timing and whole-request provenance; no model or game required."""
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

from playmodel.laya.broker import LayaBroker


def observation(sequence=1):
    now = time.perf_counter_ns()
    return SimpleNamespace(sequence=sequence, pixels=b'pixels', available_at_ns=now,
        metadata={'capture_started_at_ns': now, 'hwnd': 1, 'pid': 2})


def situation(signature='same', other='other'):
    return {'state': 'observed world', 'signature': signature,
            'options': {'retreat': 'retreat safely', 'other': other}, 'world': {'valid': True}}


class Client:
    def __init__(self, root):
        self.output = Path(root)
        self.ready = {'behavior_version': 'v1'}
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.count = 0

    def choose(self, state, options, evidence):
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('test release missing')
        self.count += 1
        self.calls.append(('choose', evidence))
        return {'decision_id': f'd{self.count}', 'action_id': 'retreat',
                'behavior_version': 'v1', 'decided_at_ns': time.perf_counter_ns()}

    def discard(self, decision_id, reason):
        self.calls.append(('discard', decision_id, reason))
        return {}

    def accept_tactic(self, decision_id, application):
        self.calls.append(('accept', decision_id, application))
        return {}

    def abandon(self, reason):
        self.calls.append(('abandon', reason))
        return {}


class LayaBrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = Client(self.temp.name)
        self.broker = LayaBroker(self.client, queue_capacity=4)
        self.session = self.broker.new_tactical_session('run')
        self.session.offer_period_ns = 0
        self.session.begin_combat()

    def tearDown(self):
        self.client.release.set()
        self.broker.close()
        self.temp.cleanup()

    def ready(self):
        frame, world = observation(), situation()
        self.session.offer(frame, world)
        until = time.perf_counter()+1
        while self.session._result is None and time.perf_counter()<until:
            time.sleep(.002)
        self.assertIsNotNone(self.session._result, self.broker.error)
        return frame, world

    def test_delayed_model_never_blocks_offer_or_resolve(self):
        self.client.release.clear()
        self.session.offer(observation(), situation())
        self.assertTrue(self.client.started.wait(1))
        start = time.perf_counter()
        for n in range(50):
            frame = observation(n+2)
            self.session.offer(frame, situation())
            self.assertIsNone(self.session.resolve(situation(), frame))
        self.assertLess(time.perf_counter()-start, .1)
        self.assertEqual(self.session._offer[1].sequence, 51)

    def test_any_candidate_change_invalidates_entire_request(self):
        frame, world = self.ready()
        self.assertIsNotNone(self.session.resolve(world, frame))
        # Selected retreat remains legal: changing an unselected option must
        # still reject the whole distribution, preventing selective censoring.
        self.assertIsNone(self.session.resolve(situation(other='changed'), frame))
        self.assertIsNone(self.session.resolve(situation(signature='different'), frame))

    def test_epoch_version_ttl_and_future_frame_are_all_rejected(self):
        frame, world = self.ready()
        original = dict(self.session._result)
        changes = {'epoch': 'stale', 'behavior_version': 'old',
            'expires_at_ns': time.perf_counter_ns()-1,
            'source_observed_at_ns': frame.metadata['capture_started_at_ns']+1,
            'source_available_at_ns': frame.available_at_ns+1,
            'decided_at_ns': time.perf_counter_ns()+10_000_000_000}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.session._result = {**original, key: value}
                self.assertIsNone(self.session.resolve(world, frame))
        self.session._result = original

    def test_terminal_barrier_drains_receipt_before_pending_cleanup(self):
        frame, _ = self.ready()
        self.session.record_execution({'decision_id': 'd1', 'receipt_path': 'receipt.json',
                                       'receipt_sha256': 'hash'}, frame)
        self.session.record_execution({'decision_id': 'd1', 'receipt_path': 'receipt2.json',
                                       'receipt_sha256': 'hash'}, frame)
        self.session.end_combat(valid=True, reason='terminal')
        self.assertEqual(sum(call[0]=='accept' for call in self.client.calls), 1)
        self.assertFalse(any(call[0] in ('discard','abandon') for call in self.client.calls))

    def test_interrupted_epoch_cancels_inflight_result_and_abandons_reward(self):
        self.client.release.clear()
        self.session.offer(observation(), situation())
        self.assertTrue(self.client.started.wait(1))
        self.session.active = False
        self.client.release.set()
        self.session.end_combat(valid=False, reason='gap')
        self.assertIsNone(self.session._result)
        self.assertTrue(any(call[0]=='discard' for call in self.client.calls))
        self.assertEqual(self.client.calls[-1], ('abandon','gap'))

    def test_superseded_choice_remains_acceptable_until_writer_drain(self):
        first, _ = self.ready()
        self.session.offer(observation(2), situation())
        until = time.perf_counter()+1
        while self.session._result['decision_id']=='d1' and time.perf_counter()<until:
            time.sleep(.002)
        self.assertEqual(self.session._result['decision_id'], 'd2')
        self.session.record_execution({'decision_id':'d1','receipt_path':'old.json',
                                      'receipt_sha256':'hash'}, first)
        self.session.end_combat(valid=True, reason='terminal')
        self.assertTrue(any(call[0]=='accept' and call[1]=='d1' for call in self.client.calls))
        self.assertFalse(any(call[0]=='discard' and call[1]=='d1' for call in self.client.calls))
        self.assertTrue(any(call[0]=='discard' and call[1]=='d2' for call in self.client.calls))

    def test_receipt_queue_saturation_is_error_not_silent_drop(self):
        self.client.release.clear()
        self.session.offer(observation(), situation())
        self.assertTrue(self.client.started.wait(1))
        for n in range(4):
            self.broker._enqueue(lambda: {})
        with self.assertRaisesRegex(RuntimeError, 'saturated'):
            self.broker._enqueue(lambda: {})
        self.assertIsNotNone(self.session.error)

    def test_empty_candidate_set_does_not_invoke_model(self):
        self.session.offer(observation(), {'state':'unknown','signature':'invalid','options':{},'world':{'valid':False}})
        time.sleep(.025)
        self.assertFalse(self.client.started.is_set())


if __name__ == '__main__':
    unittest.main()
