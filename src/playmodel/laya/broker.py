"""Single IPC owner; gameplay offers observations without waiting for inference."""
from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import uuid
from .preferences import load_preferences


class LayaBroker:
    def __init__(self, client, *, queue_capacity=256):
        self.client = client
        self.output = Path(client.output)
        self.ready = client.ready
        self.version = self.ready['behavior_version']
        self.error = None
        self._commands = queue.Queue(maxsize=queue_capacity)
        self._sessions = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='laya-broker', daemon=True)
        self._thread.start()

    def _enqueue(self, callback):
        if self._closed or self.error:
            raise RuntimeError('Laya broker unavailable' + (': ' + str(self.error) if self.error else '')) from self.error
        future = Future()
        try:
            self._commands.put_nowait((callback, future))
        except queue.Full as error:
            self.error = RuntimeError('Laya evidence queue saturated; stopping without dropping evidence')
            raise self.error from error
        return future

    def _call(self, method, *args, **kwargs):
        def invoke():
            result = getattr(self.client, method)(*args, **kwargs)
            self.version = result.get('behavior_version', self.version)
            return result
        return self._enqueue(invoke).result(timeout=240)

    def choose(self, *args, **kwargs):
        return self._call('choose', *args, **kwargs)

    def accept(self, *args, **kwargs):
        return self._call('accept', *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._call('finish', *args, **kwargs)

    def abandon(self, *args, **kwargs):
        return self._call('abandon', *args, **kwargs)

    def apply_preferences(self, path):
        return self._call('configure_preferences', preferences=load_preferences(path))

    def new_tactical_session(self, run_id):
        session = TacticalSession(self, run_id)
        self._sessions.append(session)
        return session

    def _run(self):
        while not self._closed:
            try:
                callback, future = self._commands.get(timeout=0.005)
            except queue.Empty:
                if self.error:
                    continue
                for session in tuple(self._sessions):
                    offer = session._offer
                    if session.active and offer is not None and offer is not session._consumed:
                        session._consumed = offer
                        try:
                            session._choose(offer)
                        except BaseException as error:
                            self.error = error
                        break
                continue
            try:
                if self.error:
                    raise RuntimeError('Laya broker failed') from self.error
                future.set_result(callback())
            except BaseException as error:
                self.error = error
                future.set_exception(error)

    def close(self):
        if self._closed:
            return
        for session in self._sessions:
            session.active = False
        try:
            if not self.error:
                self._enqueue(lambda: None).result(timeout=240)
        finally:
            self._closed = True
            self._thread.join(timeout=35)
        if self._thread.is_alive():
            raise RuntimeError('Laya broker did not terminate')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TacticalSession:
    """Latest-only choices; control receipts are lossless FIFO writer work."""
    def __init__(self, broker, run_id, *, ttl_ns=1_500_000_000, offer_period_ns=400_000_000):
        self.broker, self.run_id = broker, str(run_id)
        self.ttl_ns, self.offer_period_ns = ttl_ns, offer_period_ns
        self.active = False
        self.epoch = None
        self._offer = self._consumed = self._result = None
        self._last_offer_ns = 0
        self._recorded = set()
        self._pending = set()

    @property
    def error(self):
        return self.broker.error

    def begin_combat(self):
        if self.active:
            raise RuntimeError('Tactical combat already active')
        self.broker._enqueue(lambda: None).result(timeout=240)
        self.epoch = uuid.uuid4().hex
        self._offer = self._consumed = self._result = None
        self._last_offer_ns = 0
        self._recorded = set()
        self.active = True

    def offer(self, frame, situation):
        # Only immutable references and scalar assignment on the control thread.
        now = time.perf_counter_ns()
        if self.active and situation['options'] and now - self._last_offer_ns >= self.offer_period_ns:
            self._offer = (self.epoch, frame, situation, self.broker.version)
            self._last_offer_ns = now

    def resolve(self, situation, frame):
        result = self._result
        now = time.perf_counter_ns()
        if (not self.active or self.error or result is None
                or result['epoch'] != self.epoch or result['behavior_version'] != self.broker.version
                or result['signature'] != situation['signature']
                or result['options'] != situation['options']
                or result['action_id'] not in situation['options']
                or now >= result['expires_at_ns']
                or result['source_observed_at_ns'] > frame.metadata['capture_started_at_ns']
                or result['source_available_at_ns'] > frame.available_at_ns
                or result['decided_at_ns'] > now):
            return None
        return result

    def _choose(self, offer):
        epoch, frame, situation, version = offer
        expires = frame.available_at_ns + self.ttl_ns
        if not self.active or epoch != self.epoch or time.perf_counter_ns() >= expires or len(self._pending) >= 96:
            return
        directory = self.broker.output / 'tactic-observations' / uuid.uuid4().hex
        directory.mkdir(parents=True)
        source = directory / 'frame.bgra'
        with source.open('xb') as stream:
            stream.write(frame.pixels)
        evidence = {'decision_domain': 'combat_tactic', 'run_id': self.run_id,
            'epoch': epoch, 'signature': situation['signature'], 'expires_at_ns': expires,
            'target_identity': {key: frame.metadata.get(key) for key in ('hwnd', 'pid', 'executable')},
            'frame_ref': str(source.resolve()), 'frame_sha256': hashlib.sha256(frame.pixels).hexdigest(),
            'observed_at_ns': frame.metadata['capture_started_at_ns'],
            'available_at_ns': frame.available_at_ns}
        observation_path = directory / 'observation.json'
        with observation_path.open('x', encoding='utf8') as stream:
            json.dump({'evidence': evidence, 'metadata': frame.metadata, 'situation': situation},
                      stream, ensure_ascii=False, allow_nan=False)
        evidence['observation_path'] = str(observation_path.resolve())
        evidence['observation_sha256'] = hashlib.sha256(observation_path.read_bytes()).hexdigest()
        response = self.broker.client.choose(situation['state'], situation['options'], evidence)
        self._pending.add(response['decision_id'])
        if (not self.active or epoch != self.epoch or version != self.broker.version
                or response['behavior_version'] != version or time.perf_counter_ns() >= expires):
            self._discard(response['decision_id'], 'expired_or_cancelled_whole_request')
            return
        self._result = {**response, 'epoch': epoch, 'signature': situation['signature'],
            'options': dict(situation['options']), 'source_observed_at_ns': evidence['observed_at_ns'],
            'source_available_at_ns': frame.available_at_ns, 'expires_at_ns': expires}
        # Retain superseded pending choices until writer join: an earlier packet
        # can still be in transport or awaiting its durable writer receipt.

    def _discard(self, decision_id, reason):
        if decision_id in self._pending:
            self.broker.client.discard(decision_id, reason)
            self._pending.remove(decision_id)

    def record_execution(self, receipt, frame):
        decision_id = receipt['decision_id']
        if decision_id in self._recorded:
            return
        application = {'receipt_path': receipt['receipt_path'], 'receipt_sha256': receipt['receipt_sha256']}
        def accept():
            self.broker.client.accept_tactic(decision_id, application)
            self._pending.discard(decision_id)
        self.broker._enqueue(accept)
        self._recorded.add(decision_id)

    def end_combat(self, *, valid, reason):
        # Called after input release and writer join. Epoch cancellation makes
        # any in-flight choose unusable before this FIFO drain barrier runs.
        self.active = False
        self._offer = None
        self._result = None
        def finish_boundary():
            for decision_id in tuple(self._pending):
                self._discard(decision_id, reason)
            if not valid:
                self.broker.client.abandon(reason)
        self.broker._enqueue(finish_boundary).result(timeout=240)
