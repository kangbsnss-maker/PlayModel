"""Control scheduling tests use only fake observations and recording sinks."""

from dataclasses import replace
from threading import Event, Lock, Thread
import time
import unittest

from playmodel.control import (
    ControlLimits, LatestObservationSlot, Observation, RealtimeController, SendReceipt,
)


class Clock:
    def __init__(self, now=1_000):
        self.now = now
        self.lock = Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, delta):
        with self.lock:
            self.now += delta


class RecordingSink:
    def __init__(self):
        self.calls = []
        self.held = None
        self.human_held = {"human-left"}

    def send(self, action, **metadata):
        self.calls.append(("send", action, metadata))
        self.held = action
        return SendReceipt(True, None)

    def release(self, **metadata):
        self.calls.append(("release", None, metadata))
        self.held = None
        return SendReceipt(True, True)

    @property
    def sends(self):
        return [call for call in self.calls if call[0] == "send"]


class LatestObservationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.slot = LatestObservationSlot(100, clock=self.clock)

    def observation(self, sequence=1, **changes):
        return replace(Observation(sequence, 0, 950, 990, b"immutable-frame"), **changes)

    def test_replaces_without_backlog_and_readers_do_not_consume(self):
        for sequence in range(100):
            self.assertTrue(self.slot.publish(self.observation(sequence)).accepted)
        self.assertEqual(self.slot.latest().sequence, 99)
        self.assertIs(self.slot.latest(), self.slot.latest())
        self.assertEqual(self.slot.publish(self.observation(99)).reason, "duplicate_or_out_of_order")
        self.assertEqual(self.slot.publish(self.observation(98)).reason, "duplicate_or_out_of_order")

    def test_ocr_arrival_does_not_refresh_stale_capture(self):
        self.assertEqual(self.slot.publish(self.observation(observed_at_ns=900)).reason, "stale_observation")
        self.assertIsNone(self.slot.latest())
        self.assertEqual(self.slot.publish(self.observation(available_at_ns=1001)).reason, "future_observation")
        with self.assertRaises(ValueError):
            self.observation(observed_at_ns=1000, available_at_ns=990)

    def test_old_producer_cannot_restore_previous_generation(self):
        self.assertTrue(self.slot.publish(self.observation()).accepted)
        self.slot.reset(1)
        self.assertIsNone(self.slot.latest())
        self.assertEqual(self.slot.publish(self.observation(100)).reason, "generation_changed")
        self.assertTrue(self.slot.publish(self.observation(0, generation=1)).accepted)
        with self.assertRaises(ValueError):
            self.slot.reset(1)

    def test_diagnostic_monotonic_clock_cannot_mix_with_runtime_qpc(self):
        self.assertEqual(self.slot.publish(self.observation(clock_domain="monotonic_ns_same_host")).reason,
                         "clock_domain_mismatch")
        self.assertIsNone(self.slot.latest())

    def test_out_of_order_capture_rejected_even_with_higher_sequence(self):
        self.slot.publish(self.observation(1))
        self.assertEqual(self.slot.publish(self.observation(2, observed_at_ns=949)).reason,
                         "capture_time_regressed")
        self.assertEqual(self.slot.latest().sequence, 1)

    def test_parallel_producers_leave_largest_sequence(self):
        threads = [Thread(target=lambda start=start: [self.slot.publish(self.observation(i))
                                                     for i in range(start, 100, 4)])
                   for start in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(1)
            self.assertFalse(thread.is_alive())
        self.assertEqual(self.slot.latest().sequence, 99)

    def test_invalid_contract_values_fail_early(self):
        for field in ("sequence", "generation", "observed_at_ns", "available_at_ns"):
            for value in (True, -1, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.observation(**{field: value})
        with self.assertRaises(ValueError):
            ControlLimits(100, 0, 5, 50)
        with self.assertRaises(ValueError):
            SendReceipt(False, True)


class RealtimeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sink = RecordingSink()
        self.limits = ControlLimits(100, 40, 10, 50, event_capacity=12)
        self.controllers = []
        self.unblock = Event()

    def tearDown(self):
        self.unblock.set()
        for controller in self.controllers:
            controller.close(timeout=0.2)

    def controller(self, policy=lambda observation, deadline: observation.payload, sink=None, limits=None):
        controller = RealtimeController(policy, sink or self.sink, limits or self.limits, clock=self.clock)
        self.controllers.append(controller)
        return controller

    def publish(self, controller, sequence=1, **changes):
        observation = replace(Observation(sequence, controller.generation, self.clock(), self.clock(),
                                          (1.0, 0.0)), **changes)
        self.assertTrue(controller.publish(observation).accepted)
        return observation

    def drive_until(self, controller, condition):
        end = time.monotonic() + 1
        while time.monotonic() < end:
            controller.tick()
            if condition():
                return
            time.sleep(0.001)
        self.fail("controller did not reach expected state")

    def test_disarmed_default_and_repeated_ticks_never_repeat_same_action(self):
        controller = self.controller()
        self.publish(controller)
        controller.tick()
        self.assertEqual(self.sink.sends, [])
        controller.arm()
        self.assertIsNone(controller.latest())
        self.publish(controller)
        self.drive_until(controller, lambda: bool(self.sink.sends))
        for _ in range(20):
            controller.tick()
        self.assertEqual(len(self.sink.sends), 1)
        event = next(event for event in controller.events() if event.kind == "dispatch")
        self.assertEqual(event.observed_at_ns, 1000)
        self.assertEqual(event.available_at_ns, 1000)
        self.assertEqual(event.policy_started_at_ns, 1000)
        self.assertEqual(event.send_finished_at_ns, 1000)
        self.assertIsNone(event.receipt.acknowledged)
        self.assertIsNone(event.game_applied)
        self.assertLessEqual(event.sink_deadline_ns, event.deadline_ns)

    def test_slow_ocr_and_training_do_not_block_movement_dispatch(self):
        controller = self.controller()
        controller.arm()
        self.publish(controller)
        started = [Event(), Event()]
        finished = [Event(), Event()]

        def auxiliary(index):
            if index == 0:
                self.assertIsNotNone(controller.latest())
            started[index].set()
            self.unblock.wait(2)
            finished[index].set()

        workers = [Thread(target=auxiliary, args=(index,), daemon=True) for index in range(2)]
        for worker in workers:
            worker.start()
        for signal in started:
            self.assertTrue(signal.wait(1))
        self.drive_until(controller, lambda: bool(self.sink.sends))
        self.assertFalse(any(signal.is_set() for signal in finished))
        self.assertEqual(self.sink.held, (1.0, 0.0))
        self.unblock.set()
        for worker in workers:
            worker.join(1)

    def test_slow_policy_has_one_inflight_job_and_only_latest_next_frame(self):
        entered = Event()
        processed = []

        def policy(observation, deadline):
            processed.append(observation.sequence)
            if observation.sequence == 1:
                entered.set()
                self.unblock.wait(2)
            return observation.payload

        controller = self.controller(policy)
        controller.arm()
        self.publish(controller, 1)
        controller.tick()
        self.assertTrue(entered.wait(1))
        for sequence in range(2, 101):
            self.publish(controller, sequence)
            controller.tick()
        self.assertEqual(processed, [1])
        self.assertEqual(self.sink.sends, [])
        self.unblock.set()
        self.drive_until(controller, lambda: len(self.sink.sends) == 2)
        self.assertEqual(processed, [1, 100])
        self.assertEqual([call[2]["observation_sequence"] for call in self.sink.sends], [1, 100])

    def test_takeover_cancels_inflight_result_and_releases_ai_only(self):
        entered = Event()

        def policy(observation, deadline):
            if observation.sequence == 2:
                entered.set()
                self.unblock.wait(2)
            return observation.payload

        controller = self.controller(policy)
        generation = controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: len(self.sink.sends) == 1)
        self.publish(controller, 2)
        controller.tick()
        self.assertTrue(entered.wait(1))
        controller.disarm()
        self.assertGreater(controller.generation, generation)
        self.assertFalse(controller.armed)
        self.assertIsNone(self.sink.held)
        self.assertEqual(self.sink.human_held, {"human-left"})
        self.assertFalse(self.unblock.is_set())
        self.unblock.set()
        self.drive_until(controller, lambda: any(event.reason == "authority_changed"
                                                 for event in controller.events()))
        self.assertEqual(len(self.sink.sends), 1)

    def test_scene_generation_changed_during_inference_never_sends_old_result(self):
        entered = Event()

        def policy(observation, deadline):
            entered.set()
            self.unblock.wait(2)
            return observation.payload

        controller = self.controller(policy)
        controller.arm()
        previous = self.publish(controller)
        controller.tick()
        self.assertTrue(entered.wait(1))
        controller.invalidate()
        self.assertTrue(controller.armed)
        self.assertEqual(controller.publish(replace(previous, sequence=2)).reason, "generation_changed")
        self.unblock.set()
        self.drive_until(controller, lambda: any(event.reason == "authority_changed"
                                                 for event in controller.events()))
        self.assertEqual(self.sink.sends, [])

    def test_policy_deadline_disarms_without_waiting_for_policy(self):
        entered = Event()

        def policy(observation, deadline):
            entered.set()
            self.unblock.wait(2)
            return observation.payload

        controller = self.controller(policy)
        controller.arm()
        self.publish(controller)
        controller.tick()
        self.assertTrue(entered.wait(1))
        self.clock.advance(40)
        controller.tick()
        self.assertFalse(controller.armed)
        self.assertEqual(self.sink.calls[-1][0], "release")
        self.assertFalse(controller.close(timeout=0))
        self.assertEqual(self.sink.sends, [])
        self.unblock.set()
        self.assertTrue(controller.close(timeout=0.2))
        controller.tick()
        self.assertEqual(self.sink.sends, [])

    def test_observation_age_checked_again_after_policy(self):
        def policy(observation, deadline):
            self.clock.advance(100)
            return observation.payload

        controller = self.controller(policy)
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: not controller.armed)
        self.assertEqual(self.sink.sends, [])
        self.assertTrue(any(event.reason in ("stale_observation", "policy_deadline")
                            for event in controller.events()))

    def retry_controller(self):
        entered = Event()

        def policy(observation, deadline):
            if observation.sequence == 2:
                entered.set()
                self.unblock.wait(2)
            return observation.payload

        controller = self.controller(policy, limits=replace(self.limits, event_capacity=64,
                                                            policy_deadline_retries=1))
        controller.arm()
        self.publish(controller, 1)
        self.drive_until(controller, lambda: len(self.sink.sends) == 1)
        self.publish(controller, 2)
        controller.tick()
        self.assertTrue(entered.wait(1))
        self.clock.advance(40)
        controller.tick()
        return controller

    def test_deadline_retry_discards_late_result_without_input_or_memory_commit(self):
        controller = self.retry_controller()
        self.assertTrue(controller.armed)
        self.assertEqual([call[0] for call in self.sink.calls], ["send"])
        self.assertEqual(controller.publish(Observation(2, controller.generation, 1040, 1040, 2)).reason,
                         "pre_retry_observation")
        self.publish(controller, 3, payload="fresh")
        self.unblock.set()
        self.drive_until(controller, lambda: len(self.sink.sends) == 2)
        self.assertEqual([call[2]['observation_sequence'] for call in self.sink.sends], [1, 3])
        events = controller.events()
        retry = next(event for event in events if event.kind == 'retry')
        self.assertEqual(retry.deadline_ns, 1050)  # Original watchdog, never extended.
        expired = next(event for event in events if event.kind == 'expired')
        self.assertEqual((expired.sequence, expired.observed_at_ns, expired.deadline_ns), (2, 1000, 1040))
        self.assertIsNone(expired.policy_finished_at_ns)  # Was still running, not fabricated.
        self.assertTrue(any(event.kind == 'discarded' and event.sequence == 2 for event in events))
        self.assertEqual(sum(event.kind == 'retry_completed' for event in events), 1)
        self.assertEqual(self.sink.sends[-1][2]['deadline_ns'], 1050)

    def test_retry_does_not_extend_held_watchdog_while_worker_hangs(self):
        controller = self.retry_controller()
        self.clock.advance(10)
        controller.tick()
        self.assertFalse(controller.armed)
        self.assertEqual(controller.disarm_reason, 'input_watchdog')
        self.assertIsNone(self.sink.held)
        self.assertEqual(len(self.sink.sends), 1)

    def test_retry_dispatch_rechecks_original_lease_after_proposal(self):
        controller = self.retry_controller()
        original = controller._event

        def delayed(kind, reason, result, **kwargs):
            original(kind, reason, result, **kwargs)
            if kind == 'proposed':
                self.clock.advance(10)

        controller._event = delayed
        self.publish(controller, 3)
        self.unblock.set()
        self.drive_until(controller, lambda: not controller.armed)
        self.assertEqual(len(self.sink.sends), 1)
        self.assertEqual(controller.disarm_reason, 'policy_deadline_retry_exhausted')

    def test_human_takeover_during_retry_cannot_rearm_or_send(self):
        controller = self.retry_controller()
        controller.disarm('human_takeover')
        self.unblock.set()
        self.drive_until(controller, lambda: any(event.kind == 'discarded' for event in controller.events()))
        self.assertFalse(controller.armed)
        self.assertEqual(controller.disarm_reason, 'human_takeover')
        self.assertEqual(len(self.sink.sends), 1)

    def test_first_policy_deadline_has_no_held_action_to_retry(self):
        controller = self.controller(lambda obs, deadline: self.clock.advance(40) or obs.payload,
                                     limits=replace(self.limits, policy_deadline_retries=1))
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: not controller.armed)
        self.assertEqual(controller.disarm_reason, 'policy_deadline')
        self.assertEqual(self.sink.sends, [])
        self.assertFalse(any(event.kind == 'retry' for event in controller.events()))

    def test_late_policy_exception_is_never_hidden_by_deadline_retry(self):
        def policy(observation, deadline):
            if observation.sequence == 2:
                self.clock.advance(40)
                raise ValueError('fixture')
            return observation.payload

        controller = self.controller(policy, limits=replace(self.limits, policy_deadline_retries=1))
        controller.arm()
        self.publish(controller, 1)
        self.drive_until(controller, lambda: bool(self.sink.sends))
        self.publish(controller, 2)
        controller.tick()
        # Poll the already completed error, independently of the inflight tick path.
        end = time.monotonic() + 1
        while controller._worker._result is None and time.monotonic() < end:
            time.sleep(.001)
        controller.tick()
        self.assertFalse(controller.armed)
        self.assertEqual(controller.disarm_reason, 'policy_error:ValueError')
        self.assertFalse(any(event.kind == 'retry' for event in controller.events()))

    def test_final_dispatch_check_rejects_expiry_after_proposal(self):
        controller = self.controller()
        controller.arm()
        self.publish(controller)
        original = controller._event

        def proposal_takes_time(kind, reason, result, **kwargs):
            original(kind, reason, result, **kwargs)
            if kind == "proposed":
                self.clock.advance(40)

        controller._event = proposal_takes_time
        self.drive_until(controller, lambda: not controller.armed)
        self.assertEqual(self.sink.sends, [])
        self.assertTrue(any(event.kind == "rejected" and event.reason == "policy_deadline"
                            for event in controller.events()))

    def test_watchdog_releases_held_action_even_if_no_new_frame_arrives(self):
        controller = self.controller()
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: bool(self.sink.sends))
        self.clock.advance(50)
        controller.tick()
        self.assertIsNone(self.sink.held)
        self.assertFalse(controller.armed)
        self.assertEqual(self.sink.calls[-1][2]["reason"], "input_watchdog")

    def test_held_input_released_when_source_expires_before_watchdog(self):
        controller = self.controller(limits=replace(self.limits, input_watchdog_ns=200))
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: bool(self.sink.sends))
        self.clock.advance(100)
        controller.tick()
        self.assertIsNone(self.sink.held)
        self.assertFalse(controller.armed)
        self.assertEqual(self.sink.calls[-1][2]["reason"], "held_observation_expired")
        guard = next(event for event in controller.events()
                     if event.kind == "authority" and event.reason == "held_observation_expired")
        self.assertIsNotNone(guard.guard_snapshot["latest_sequence"])
        self.assertEqual(guard.guard_snapshot["held_observation_deadline_ns"],
                         guard.guard_snapshot["latest_observed_at_ns"] + self.limits.observation_age_ns)
        self.assertIsNotNone(guard.guard_snapshot["last_sent_at_ns"])
        self.assertIsNone(controller.latest())  # Snapshot precedes generation reset.

    def test_takeover_is_serialized_after_already_started_sink_send(self):
        entered, takeover_started, takeover_finished = Event(), Event(), Event()

        class PausedSink(RecordingSink):
            def send(sink, action, **metadata):
                entered.set()
                self.unblock.wait(2)
                return super(PausedSink, sink).send(action, **metadata)

        sink = PausedSink()
        controller = self.controller(sink=sink)
        controller.arm()
        self.publish(controller)
        stop = Event()
        actor = Thread(target=controller.run, args=(stop,), kwargs={"interval_ns": 1_000_000}, daemon=True)

        def takeover():
            takeover_started.set()
            controller.disarm()
            takeover_finished.set()

        taker = Thread(target=takeover, daemon=True)
        actor.start()
        try:
            self.assertTrue(entered.wait(1))
            taker.start()
            self.assertTrue(takeover_started.wait(1))
            self.assertFalse(takeover_finished.is_set())
            self.assertEqual(sink.calls, [])
            self.unblock.set()
            self.assertTrue(takeover_finished.wait(1))
            self.assertEqual([call[0] for call in sink.calls], ["send", "release"])
            self.assertIsNone(sink.held)
        finally:
            self.unblock.set()
            stop.set()
            actor.join(1)
            if taker.ident is not None:
                taker.join(1)

    def test_explicit_negative_acknowledgement_faults_and_releases(self):
        class RejectedSink(RecordingSink):
            def send(sink, action, **metadata):
                super(RejectedSink, sink).send(action, **metadata)
                return SendReceipt(True, False)

        sink = RejectedSink()
        controller = self.controller(sink=sink)
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: not controller.armed)
        self.assertTrue(any(event.reason == "input_not_acknowledged" for event in controller.events()))
        self.assertIsNone(sink.held)

    def test_sink_overrun_is_recorded_and_release_requested(self):
        class SlowSink(RecordingSink):
            def send(sink, action, **metadata):
                receipt = super(SlowSink, sink).send(action, **metadata)
                self.clock.advance(10)
                return receipt

        sink = SlowSink()
        controller = self.controller(sink=sink)
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: not controller.armed)
        dispatch = next(event for event in controller.events() if event.kind == "dispatch")
        self.assertEqual(dispatch.reason, "sink_deadline")
        self.assertTrue(dispatch.receipt.transmitted)
        self.assertIsNone(sink.held)
        self.assertEqual(sink.calls[-1][0], "release")

    def test_policy_exception_and_release_failure_are_visible_not_success(self):
        class FailingSink(RecordingSink):
            def release(sink, **metadata):
                raise OSError("fixture")

        def failed_policy(observation, deadline):
            raise ValueError("private payload is not logged")

        controller = self.controller(failed_policy, FailingSink())
        controller.arm()
        self.publish(controller)
        self.drive_until(controller, lambda: not controller.armed)
        reasons = [event.reason for event in controller.events()]
        self.assertIn("policy_error:ValueError", reasons)
        self.assertIn("release_error:OSError", reasons)
        self.assertNotIn("private payload", repr(controller.events()))

    def test_event_storage_bounded_and_actor_stop_requests_release(self):
        controller = self.controller()
        for _ in range(30):
            controller.arm()
            controller.disarm()
        self.assertLessEqual(len(controller.events()), self.limits.event_capacity)
        stop = Event()
        actor = Thread(target=controller.run, args=(stop,), kwargs={"interval_ns": 1_000_000})
        actor.start()
        stop.set()
        actor.join(1)
        self.assertFalse(actor.is_alive())
        self.assertEqual(self.sink.calls[-1][2]["reason"], "actor_stopped")


if __name__ == "__main__":
    unittest.main()
