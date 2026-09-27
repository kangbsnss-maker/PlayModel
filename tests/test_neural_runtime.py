"""Real neural inference against fake capture/input; never opens or drives a game."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from playmodel.games.brotato.capture import _png
    from playmodel.games.brotato.neural_runtime import (
        NeuralAction, NeuralMovementSink, _action_record, chunk_rollout, image_tensor, load_rollout, run_neural_trial,
    )
    from playmodel.games.brotato.pilot import CLOCK, PilotConfig
    from playmodel.games.brotato.stream import Frame
    from playmodel.learning import StateEvidence
    from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic, PPOConfig, _validate_rollout


@unittest.skipIf(torch is None, 'optional PyTorch dependency not installed')
class NeuralRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.manual_seed(13)
        self.model = RecurrentActorCritic(ModelConfig(hidden_size=16, visual_size=16, candidate_hidden_size=8))
        self.state = SimpleNamespace(count=0, stop=False, menu=False, releases=0, mutate=None, requests=[])

    def tearDown(self):
        self.temp.cleanup()

    def evidence(self, kind):
        observed = time.perf_counter_ns()
        width, height = (1920, 1080) if kind == 'wave_clear' else (16, 16)
        pixels = bytes([4, 5, 6, 255]) * width * height
        path = self.root / (kind + '.png')
        path.write_bytes(_png(width, height, pixels))
        if kind == 'wave_clear':
            raw = {'lines': [{'words': [{'text': 'Shop', 'x': 30, 'y': 30, 'width': 100, 'height': 50}]},
                             {'words': [{'text': 'Go', 'x': 1500, 'y': 820, 'width': 100, 'height': 50}]}]}
            (self.root / 'ocr.json').write_text(json.dumps({'raw': raw}), encoding='utf-8')
        available = time.perf_counter_ns()
        return StateEvidence(kind, str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest(),
                             observed, available, available, 'developer_verified', 'unit-fixture-review', True, True)

    def factories(self):
        state = self.state

        class Stream:
            def __init__(self, *args, **kwargs):
                self.error, self.sequence, self.latest_frame = None, 0, None

            def latest(self, **kwargs):
                now = time.perf_counter_ns()
                if self.latest_frame is None or now - self.latest_frame.available_at_ns > 8_000_000:
                    self.sequence += 1
                    marker = 200 if state.menu and state.count >= 2 else 20
                    self.latest_frame = Frame(self.sequence, {
                        'hwnd': 123, 'backend': 'win32_printwindow_client', 'clock': CLOCK,
                        'capture_started_at_ns': now, 'capture_finished_at_ns': now,
                        'sample_width': 16, 'sample_height': 16,
                    }, bytes([marker, 10, 30, 255]) * 16 * 16, now)
                return self.latest_frame

            def close(self):
                pass

            def request_fresh(self, after_ns):
                assert state.count > 0, 'capture requests follow only actual transmissions'
                state.requests.append(after_ns)

        class Background:
            def __init__(self, *args):
                pass

            def set_movement(self, keys):
                state.count += 1
                if state.mutate:
                    state.mutate()
                    state.mutate = None

            def release(self):
                state.releases += 1

            def check(self):
                if state.stop and state.count:
                    raise OSError('F8 requested')

        class Vision:
            def observe(self, pixels, *args, **kwargs):
                return SimpleNamespace(combat_likely=pixels[0] != 200)

        return dict(stream_factory=Stream, background_factory=Background, vision_factory=Vision)

    def run_trial(self, *, steps=4, terminal=None, split='train', policy_budget_ms=500,
                  first_action_callback=None, online_session=None, initial_hidden=None, reset_first=None):
        entry = self.evidence('combat')
        config = PilotConfig(max_seconds=4, max_steps=steps, policy_budget_ms=policy_budget_ms,
                             max_frame_age_ms=1000, input_watchdog_ms=1000, terminal_wait_seconds=1,
                             startup_seconds=1, tick_ms=2)
        callback = terminal or (lambda: None)
        return run_neural_trial(self.root / 'fake.exe', self.root / 'trials', model=self.model,
                                combat_entry=entry, config=config, terminal_observer=callback,
                                chunk_steps=2, burn_in=2, split=split,
                                online_session=online_session, initial_hidden=initial_hidden, reset_first=reset_first,
                                first_action_callback=first_action_callback, **self.factories())

    def test_training_gate_receives_first_real_action_once_from_recorder_thread(self):
        received = []

        def callback(record):
            self.assertTrue(record['transmitted'])
            self.assertGreater(self.state.count, 0)
            self.assertGreater(record['sent_at_ns'], record['transport_started_at_ns'])
            path = self.root / 'trials' / record['session_id'] / 'actions.jsonl'
            saved = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual(saved['sent_at_ns'], record['sent_at_ns'])
            received.append((record, threading.current_thread().name))

        report = self.run_trial(steps=4, first_action_callback=callback)
        self.assertEqual(report['reason'], 'step_limit', report)
        self.assertTrue(report['rollout_eligible'], report)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][1], 'neural-rollout-writer')
        self.assertEqual(received[0][0]['session_id'], Path(report['session_directory']).name)

    def test_training_gate_failure_is_recording_failure_and_rejects_rollout(self):
        def callback(record):
            raise OSError('fixture training gate write failure')
        report = self.run_trial(steps=4, first_action_callback=callback)
        self.assertFalse(report['rollout_eligible'], report)
        self.assertIsNone(report['rollout_path'])

    def test_empty_recorder_does_not_release_training_gate(self):
        from playmodel.games.brotato.neural_runtime import _Writer
        callback = Mock()
        writer = _Writer(self.root, first_action_callback=callback)
        self.assertTrue(writer.close())
        callback.assert_not_called()

    def test_raw_bgra_is_rgb_and_chronological_step_limit_bootstraps(self):
        report = self.run_trial(steps=5)
        self.assertEqual(report['reason'], 'step_limit', report)
        self.assertIsNotNone(report['rollout_path'], report)
        batch = load_rollout(report['rollout_path'])
        self.assertEqual(tuple(batch.images.shape), (4, 3, 3, 96, 96))
        self.assertEqual(batch.images[2, 0, :, 0, 0].tolist(), [30, 10, 20])
        self.assertEqual(int(batch.valid[2:].sum()), 5)
        self.assertFalse(batch.valid[:2, 0].any())
        self.assertTrue(batch.reset[2, 0])
        self.assertFalse(batch.reset[0, 2])
        self.assertFalse(batch.terminated.any())
        self.assertTrue(batch.truncated[-2, -1])
        self.assertEqual(float(batch.rewards.sum()), 0.)
        self.assertGreater(float(batch.elapsed_seconds[batch.valid].min()), 0.)
        learning = _validate_rollout(self.model, batch, PPOConfig(burn_in=2))
        self.assertEqual(int(learning.sum()), 5)
        self.assertFalse(report['game_application_verified'])
        self.assertFalse(report['training_performed'])
        self.assertGreater(self.state.releases, 0)
        actual_rows = [json.loads(line) for line in (Path(report['session_directory']) / 'actions.jsonl').read_text().splitlines()]
        self.assertEqual(self.state.requests, [row['sent_at_ns'] for row in actual_rows])
        flat = load_rollout(report['flat_rollout_path'])
        states = torch.load(report['initial_states_path'], weights_only=True)['initial_states']
        self.assertEqual(tuple(flat.images.shape[:2]), (5, 1))
        self.assertEqual(tuple(states.shape), (5, 1, 16))
        with torch.no_grad():
            final = self.model.step(flat.images[-1], flat.context[-1], flat.phase[-1], flat.candidates[-1],
                                    flat.legal_mask[-1], states[-1], flat.reset[-1])
        self.assertTrue(torch.allclose(final.next_hidden, torch.tensor(report['final_hidden']), atol=1e-6))

    def test_wave_clear_truncates_and_keeps_verified_reward_separate_from_ack(self):
        self.state.menu = True
        cache = []

        def terminal():
            if self.state.count < 2:
                return None
            if not cache:
                cache.append(self.evidence('wave_clear'))
            return cache[0]

        report = self.run_trial(steps=100, terminal=terminal)
        self.assertEqual(report['reason'], 'terminal_wave_clear', report)
        self.assertIsNotNone(report['rollout_path'], report)
        batch = load_rollout(report['rollout_path'])
        self.assertFalse(batch.terminated.any())
        self.assertTrue(batch.truncated.any())
        self.assertEqual(float(batch.rewards[2:].sum()), 1.)
        self.assertFalse(report['full_run_complete'])
        self.assertEqual(report['final_phase'], 3)
        rows = [json.loads(line) for line in (Path(report['session_directory']) / 'actions.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['acknowledged'] is None and not row['game_application_verified'] for row in rows))
        self.assertTrue(all(row['proposed_action'] == row['actual_action'] for row in rows))

    def test_death_is_terminal_and_zero_bootstrap(self):
        self.state.menu = True
        cache = []

        def terminal():
            if self.state.count < 2:
                return None
            if not cache:
                cache.append(self.evidence('death'))
            return cache[0]

        report = self.run_trial(steps=100, terminal=terminal)
        self.assertEqual(report['reason'], 'terminal_death', report)
        self.assertIsNotNone(report['rollout_path'], report)
        batch = load_rollout(report['rollout_path'])
        self.assertEqual(float(batch.rewards[2:].sum()), -1.)
        self.assertTrue(batch.terminated.any())
        self.assertFalse(batch.truncated.any())
        self.assertEqual(float(batch.next_values[batch.terminated].abs().sum()), 0.)

    def test_f8_guard_releases_and_excludes_entire_trial(self):
        self.state.stop = True
        report = self.run_trial(steps=100)
        self.assertEqual(report['reason'], 'F8', report)
        self.assertIsNone(report['rollout_path'])
        self.assertGreater(self.state.releases, 0)

    def test_unconfirmed_screen_change_preserves_exact_rejected_frame(self):
        self.state.menu = True
        report = self.run_trial(steps=100)
        self.assertEqual(report['reason'], 'screen_changed', report)
        self.assertIsNone(report['rollout_path'])
        directory = Path(report['session_directory'])
        rejected = json.loads((directory / 'phase-rejection.json').read_text())
        pixels = (directory / rejected['frame_ref']).read_bytes()
        self.assertEqual(pixels, bytes([200, 10, 30, 255]) * 16 * 16)
        self.assertEqual(hashlib.sha256(pixels).hexdigest(), rejected['frame_sha256'])
        self.assertFalse(rejected['vision']['combat_likely'])
        self.assertEqual(rejected['decision'], 'abstain')
        self.assertTrue((directory / 'phase-rejection.png').is_file())
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertTrue({'phase-rejection.json', 'phase-rejection.bgra', 'phase-rejection.png'}
                        <= {row['path'] for row in manifest['files']})

    def action_and_sink(self):
        now = time.perf_counter_ns()
        frame = Frame(1, {'hwnd': 123, 'capture_started_at_ns': now}, b'pixels', now)
        inputs = (None, None, None, None, torch.ones(1, 9, dtype=torch.bool))
        action = NeuralAction(frame, inputs, torch.zeros(1, 16), torch.ones(1, 16),
                              True, 3, -2., 0., [1 / 9] * 9, now)
        writer = SimpleNamespace(error=None, pending=queue.Queue(2))
        background = SimpleNamespace(set_movement=Mock(), release=Mock())
        return action, NeuralMovementSink(background, writer, 123, torch.zeros(1, 16))

    def test_transport_and_recorder_use_native_snapshots_without_tensor_work(self):
        action, sink = self.action_and_sink()
        with (patch.object(torch.Tensor, '__getitem__', side_effect=AssertionError('tensor in input path')),
              patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('tensor in recorder'))):
            receipt = sink.send(action, generation=2, observation_sequence=1,
                                deadline_ns=time.perf_counter_ns() + 1_000_000_000)
            record = _action_record(sink.writer.pending.get_nowait())
        self.assertTrue(receipt.transmitted)
        sink.background.set_movement.assert_called_once_with({0x44})
        self.assertEqual(record['legal_mask'], (True,) * 9)
        self.assertEqual(record['hidden_before'], (0.,) * 16)
        attempt = sink.transport_attempts[0]
        self.assertEqual(attempt['transport_started_at_ns'], record['transport_started_at_ns'])
        self.assertEqual(attempt['transport_finished_at_ns'], action.sent_at_ns)
        self.assertLessEqual(attempt['transport_started_at_ns'], action.sent_at_ns)
        self.assertIs(sink.hidden, action.hidden_after)

    def test_expired_or_unrecordable_action_never_enters_transport(self):
        for rejected in ('expired', 'full', 'illegal'):
            with self.subTest(rejected=rejected):
                action, sink = self.action_and_sink()
                deadline = time.perf_counter_ns() + 1_000_000_000
                if rejected == 'expired':
                    deadline = 0
                elif rejected == 'full':
                    sink.writer.pending.put_nowait(object())
                    sink.writer.pending.put_nowait(object())
                else:
                    action.legal_mask = (False,) * 9
                with self.assertRaises(OSError):
                    sink.send(action, generation=2, observation_sequence=1, deadline_ns=deadline)
                sink.background.set_movement.assert_not_called()
                self.assertEqual(sink.sent_count, 0)
                self.assertEqual(sink.transport_attempts, [])

    def test_native_background_receives_original_deadline_and_preserves_stage_evidence(self):
        action, sink = self.action_and_sink()
        deadline = time.perf_counter_ns() + 1_000_000_000
        sink.background.set_movement_before = Mock()
        sink.background.last_movement_timing = {'check_finished_at_ns': 123, 'posted_keys': 1}
        sink.send(action, generation=2, observation_sequence=1, deadline_ns=deadline)
        sink.background.set_movement.assert_not_called()
        sink.background.set_movement_before.assert_called_once_with({0x44}, deadline_ns=deadline)
        self.assertEqual(sink.transport_attempts[0]['background_stages'],
                         sink.background.last_movement_timing)

    def test_typed_no_post_deadline_preserves_hidden_action_and_pending_records(self):
        from playmodel.games.brotato.background import BackgroundController, PrePostMovementDeadline
        action, sink = self.action_and_sink()
        background = object.__new__(BackgroundController)
        background.held, background._key, background.check = {0x57}, Mock(), Mock()
        sink.background = background
        sink.previous_action = 7
        old_hidden = sink.hidden.clone()
        # transport starts10, identity starts11 and completes30 after deadline25.
        with patch('playmodel.games.brotato.neural_runtime.time.perf_counter_ns', side_effect=[10, 11, 30, 31]):
            with self.assertRaises(PrePostMovementDeadline):
                sink.send(action, generation=2, observation_sequence=1, deadline_ns=25)
        attempt = sink.transport_attempts[-1]
        self.assertIs(attempt['transmitted'], False)
        self.assertEqual(attempt['delivery_certainty'], 'definitely_not_sent')
        self.assertEqual(attempt['cancellation_proof'], attempt['background_stages'])
        self.assertEqual(sink.sent_count, 0)
        self.assertEqual(sink.previous_action, 7)
        self.assertTrue(torch.equal(sink.hidden, old_hidden))
        self.assertEqual(sink.packets, [])
        self.assertTrue(sink.writer.pending.empty())
        self.assertEqual(background.held, {0x57})
        background.release()
        self.assertEqual(background.held, set())

    def test_transport_error_preserves_unknown_attempt_and_excludes_rollout(self):
        def fail_after_possible_partial_input():
            raise OSError('partially posted input cannot be confirmed')

        self.state.mutate = fail_after_possible_partial_input
        report = self.run_trial(steps=2)
        self.assertEqual(report['reason'], 'controller_guard', report)
        self.assertIsNone(report['rollout_path'])
        self.assertFalse(report['rollout_eligible'])
        self.assertEqual(report['steps'], 0)
        directory = Path(report['session_directory'])
        attempts = json.loads((directory / 'input-attempts.json').read_text())
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(attempts[0]['transmitted'])
        self.assertEqual(attempts[0]['error'], 'OSError')
        self.assertGreaterEqual(attempts[0]['transport_finished_at_ns'], attempts[0]['transport_started_at_ns'])
        self.assertEqual((directory / 'actions.jsonl').read_text(), '')
        self.assertGreater(self.state.releases, 0)

    def test_deadline_retry_records_only_actual_actions_and_preserves_hidden_seam(self):
        original = RecurrentActorCritic.step
        calls = [0]

        def delayed(model, *args, **kwargs):
            calls[0] += 1
            # Warmup, actual action 1, then an intentionally late unsent action.
            if calls[0] == 3:
                time.sleep(.12)
            return original(model, *args, **kwargs)

        with patch.object(RecurrentActorCritic, 'step', delayed):
            report = self.run_trial(steps=3, policy_budget_ms=80)
        self.assertEqual(report['reason'], 'step_limit', report)
        self.assertIsNone(report['controller_guard_reason'])
        self.assertEqual(report['policy_deadline_retries'], 1, report)
        self.assertEqual(report['policy_deadline_retries_completed'], 1)
        self.assertTrue(report['rollout_eligible'], report)
        self.assertEqual(report['steps'], 3)
        self.assertIsNotNone(report['flat_rollout_path'], report)
        directory = Path(report['session_directory'])
        events = json.loads((directory / 'control-events.json').read_text())
        expired, = [event for event in events if event['kind'] == 'expired']
        discarded, = [event for event in events if event['kind'] == 'discarded']
        rows = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
        self.assertNotIn(expired['sequence'], [row['sequence'] for row in rows])
        self.assertEqual(expired['sequence'], discarded['sequence'])
        self.assertEqual(len(rows), 3)
        self.assertFalse(any(event['kind'] == 'release' and event['at_ns'] <= rows[-1]['sent_at_ns']
                             for event in events))
        batch = load_rollout(report['flat_rollout_path'])
        states = torch.load(report['initial_states_path'], weights_only=True)['initial_states']
        with torch.no_grad():
            first = self.model.step(batch.images[0], batch.context[0], batch.phase[0], batch.candidates[0],
                                    batch.legal_mask[0], states[0], batch.reset[0])
        self.assertTrue(torch.allclose(states[1], first.next_hidden, atol=1e-6))
        self.assertEqual(int(batch.context[1, 0, :9].argmax()), rows[0]['actual_action'])
        self.assertAlmostEqual(float(batch.elapsed_seconds[0, 0]),
                               (rows[1]['observed_at_ns'] - rows[0]['decided_at_ns']) / 1e9, places=6)
        self.assertEqual(float(batch.rewards.sum()), 0.)

    def test_first_inference_deadline_reports_specific_guard_without_retry(self):
        original = RecurrentActorCritic.step
        calls = [0]

        def delayed(model, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                time.sleep(.10)
            return original(model, *args, **kwargs)

        with (patch.object(RecurrentActorCritic, 'step', delayed),
              patch('playmodel.execution_log.event') as log):
            report = self.run_trial(steps=3, policy_budget_ms=40)
        self.assertEqual(report['reason'], 'controller_guard', report)
        self.assertEqual(report['controller_guard_reason'], 'policy_deadline')
        self.assertEqual(report['error'], 'controller_guard:policy_deadline')
        self.assertFalse(report['rollout_eligible'])
        self.assertEqual(report['steps'], 0)
        self.assertEqual(report['policy_deadline_retries'], 0)
        self.assertTrue(any(call.args[0] == 'neural_controller_diagnostics' for call in log.call_args_list))
        timing, = json.loads(Path(report['policy_timings_path']).read_text())['records']
        self.assertEqual(timing['outcome'], 'proposal_ready')  # Late result is still never sent.
        self.assertGreaterEqual(timing['model_finished_at_ns'] - timing['inputs_finished_at_ns'], 100_000_000)
        for before, after in zip(('started_at_ns', 'vision_finished_at_ns', 'callback_finished_at_ns',
                                  'inputs_finished_at_ns', 'model_finished_at_ns'),
                                 ('vision_finished_at_ns', 'callback_finished_at_ns', 'inputs_finished_at_ns',
                                  'model_finished_at_ns', 'finished_at_ns')):
            self.assertLessEqual(timing[before], timing[after])
        self.assertEqual((Path(report['session_directory']) / 'actions.jsonl').read_text(), '')
        guard = Path(report['guard_frame_path'])
        proof = json.loads(guard.with_suffix('.json').read_text())
        self.assertEqual(proof['sequence'], timing['sequence'])
        self.assertEqual(proof['frame_sha256'], hashlib.sha256(guard.read_bytes()).hexdigest())

    def test_real_transport_overrun_still_releases_and_excludes_rollout(self):
        self.state.mutate = lambda: time.sleep(.06)
        report = self.run_trial(steps=2)
        self.assertEqual(report['reason'], 'controller_guard', report)
        self.assertIsNone(report['rollout_path'])
        self.assertFalse(report['rollout_eligible'])
        directory = Path(report['session_directory'])
        events = json.loads((directory / 'control-events.json').read_text())
        event = next(row for row in events if row['kind'] == 'dispatch')
        self.assertEqual(event['reason'], 'sink_deadline')
        self.assertEqual(event['sink_deadline_ns'] - event['send_started_at_ns'], 25_000_000)
        attempt, = json.loads((directory / 'input-attempts.json').read_text())
        self.assertTrue(attempt['transmitted'])
        self.assertGreater(attempt['transport_finished_at_ns'] - attempt['transport_started_at_ns'], 25_000_000)
        self.assertGreater(self.state.releases, 0)

    def test_parent_model_mutation_cannot_change_behavior(self):
        original = self.model.policy_version()

        def mutate():
            with torch.no_grad():
                next(self.model.parameters()).add_(.25)

        self.state.mutate = mutate
        report = self.run_trial(steps=2)
        self.assertIsNotNone(report['rollout_path'], report)
        self.assertNotEqual(self.model.policy_version(), original)
        self.assertEqual(report['behavior_version'], original)
        self.assertEqual(load_rollout(report['rollout_path']).behavior_version, original)

    def test_tampered_raw_evidence_rejected_before_rollout_load(self):
        report = self.run_trial(steps=2)
        directory = Path(report['session_directory'])
        source = next((directory / 'frames').glob('*.bgra'))
        source.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            load_rollout(report['rollout_path'])

    def test_unverified_entry_rejected_without_capture_or_input(self):
        evidence = replace(self.evidence('combat'), verified=False)
        with self.assertRaisesRegex(ValueError, 'independent state evidence'):
            run_neural_trial(self.root / 'fake.exe', self.root / 'trials', model=self.model,
                             combat_entry=evidence, terminal_observer=lambda: None, **self.factories())
        self.assertEqual(self.state.count, 0)

    def test_wave_clear_with_unknown_menu_phase_is_not_training_data(self):
        self.state.menu = True
        cache = []

        def terminal():
            if self.state.count < 2:
                return None
            if not cache:
                cache.append(self.evidence('wave_clear'))
                (self.root / 'ocr.json').write_text(json.dumps({'raw': {'lines': []}}), encoding='utf-8')
            return cache[0]

        report = self.run_trial(steps=100, terminal=terminal)
        self.assertEqual(report['reason'], 'runtime_error', report)
        self.assertIn('menu phase is unknown', report['error'])
        self.assertIsNone(report['rollout_path'])

    def test_loot_boundary_uses_v2_choice_phase_without_borrowing_weapon_phase(self):
        self.state.menu = True
        cache = []

        def terminal():
            if self.state.count < 2:
                return None
            if not cache:
                cache.append(self.evidence('wave_clear'))
                raw = {'lines': [
                    {'words': [{'text': 'ItemFound', 'x': 500, 'y': 200, 'width': 160, 'height': 40}]},
                    {'words': [{'text': 'Recycle', 'x': 500, 'y': 750, 'width': 120, 'height': 40}]},
                ]}
                (self.root / 'ocr.json').write_text(json.dumps({'raw': raw}), encoding='utf-8')
            return cache[0]

        report = self.run_trial(steps=100, terminal=terminal)
        from playmodel.learning.runtime_contract import RUNTIME_CONTRACT, PHASE_SCHEMA
        self.assertEqual(report['reason'], 'terminal_wave_clear', report)
        self.assertEqual(report['final_phase'], 1)
        self.assertEqual(report['runtime_contract'], RUNTIME_CONTRACT)
        self.assertEqual(report['phase_schema'], PHASE_SCHEMA)
        self.assertTrue(report['rollout_path'])
        self.assertTrue(report['flat_rollout_path'])
        batch = load_rollout(report['flat_rollout_path'])
        self.assertEqual(batch.runtime_contract, RUNTIME_CONTRACT)
        self.assertEqual(batch.rewards[-1, 0].item(), 1)
        self.assertTrue((Path(report['session_directory']) / 'terminal-source.png').exists())

    def test_evaluation_split_preserved_and_cannot_enter_ppo(self):
        report = self.run_trial(steps=2, split='evaluation')
        self.assertEqual(report['split'], 'evaluation')
        self.assertIsNotNone(report['rollout_path'], report)
        batch = load_rollout(report['rollout_path'])
        self.assertEqual(batch.split, 'evaluation')
        with self.assertRaisesRegex(ValueError, 'evaluation/replay'):
            _validate_rollout(self.model, batch, PPOConfig(burn_in=2))


if __name__ == '__main__':
    unittest.main()
