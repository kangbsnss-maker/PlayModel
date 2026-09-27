"""Released screen changes allow observation only; no terminal reward is invented."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional PyTorch unavailable')

from playmodel.games.brotato.neural_runtime import safe_observation_transition
from playmodel.games.brotato.pilot import CLOCK


class ObservationTransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = {'hwnd': 1, 'pid': 2, 'executable': 'game.exe'}
        self.report = dict(reason='screen_changed', error=None, capture_error=None, safety_reason=None,
            controller_guard_reason=None, cleanup_errors=[], recorder_complete=True, worker_stopped=True,
            terminal_kind=None, verified_terminal_boundary=False, session_directory=str(self.root),
            target_identity=self.identity, steps=0, policy_deadline_retries=0, policy_deadline_retries_completed=0)
        pixels = b'1234'
        (self.root / 'phase-rejection.bgra').write_bytes(pixels)
        self.phase = dict(sequence=3, available_at_ns=120, decision='abstain', clock_domain=CLOCK,
            metadata={**self.identity, 'clock': CLOCK, 'capture_started_at_ns': 100,
                'capture_finished_at_ns': 110, 'sample_width': 1, 'sample_height': 1},
            vision={'combat_likely': False, 'rejected_at_ns': 130}, frame_ref='phase-rejection.bgra',
            frame_sha256=hashlib.sha256(pixels).hexdigest())
        self.events = [dict(kind='authority', reason='screen_changed', clock_domain=CLOCK, at_ns=140,
            guard_snapshot={'latest_sequence': 3, 'latest_observed_at_ns': 100, 'latest_available_at_ns': 120}),
            dict(kind='release', reason='screen_changed', clock_domain=CLOCK,
                send_started_at_ns=141, send_finished_at_ns=145, deadline_ns=150,
                receipt={'transmitted': True, 'acknowledged': None})]
        self.attempts = []

    def write(self):
        for name, value in [('phase-rejection.json', self.phase), ('control-events.json', self.events),
                            ('input-attempts.json', self.attempts)]:
            (self.root / name).write_text(json.dumps(value), encoding='utf8')

    def test_released_abstention_is_observation_only(self):
        self.write()
        proof = safe_observation_transition(self.report)
        self.assertIsNotNone(proof)
        self.assertFalse(proof['training_eligible'])
        self.assertFalse(proof['terminal_reward_verified'])

    def test_errors_unknown_delivery_and_late_release_rejected(self):
        self.write()
        for key, value in [('error', 'fault'), ('capture_error', 'ended'), ('safety_reason', 'F8'),
                           ('controller_guard_reason', 'sink_deadline'), ('cleanup_errors', ['fault']),
                           ('recorder_complete', False), ('worker_stopped', False), ('terminal_kind', 'death')]:
            with self.subTest(key=key):
                self.assertIsNone(safe_observation_transition({**self.report, key: value}))
        self.events[-1]['send_finished_at_ns'] = 150
        self.write()
        self.assertIsNone(safe_observation_transition(self.report))
        self.events[-1]['send_finished_at_ns'] = 145
        self.events[-1]['receipt']['transmitted'] = None
        self.write()
        self.assertIsNone(safe_observation_transition(self.report))

    def test_changed_frame_identity_and_incomplete_retry_rejected(self):
        self.write()
        (self.root / 'phase-rejection.bgra').write_bytes(b'0000')
        self.assertIsNone(safe_observation_transition(self.report))
        (self.root / 'phase-rejection.bgra').write_bytes(b'1234')
        self.phase['metadata']['pid'] = 99
        self.write()
        self.assertIsNone(safe_observation_transition(self.report))
        self.phase['metadata']['pid'] = 2
        self.write()
        self.assertIsNone(safe_observation_transition({**self.report, 'policy_deadline_retries': 1}))

    def test_malformed_missing_and_failed_dispatch_rejected(self):
        self.assertIsNone(safe_observation_transition({}))
        self.write()
        self.events.append({'kind': 'dispatch', 'reason': 'sink_deadline', 'clock_domain': CLOCK})
        self.write()
        self.assertIsNone(safe_observation_transition(self.report))

    def test_exact_phase_abstention_release_is_allowed_without_input(self):
        self.events.insert(0, dict(kind='rejected', reason='policy_abstained', generation=1,
                                  sequence=3, clock_domain=CLOCK))
        self.events.insert(1, dict(kind='release', reason='policy_abstained', generation=1,
            clock_domain=CLOCK, send_started_at_ns=132, send_finished_at_ns=133, deadline_ns=139,
            receipt={'transmitted': True, 'acknowledged': None}))
        self.write()
        self.assertIsNotNone(safe_observation_transition(self.report))
        self.events[0]['sequence'] = 2
        self.write()
        self.assertIsNone(safe_observation_transition(self.report))
        (self.root / 'control-events.json').write_text('{', encoding='utf8')
        self.assertIsNone(safe_observation_transition(self.report))

    def test_original_442_action_pause_artifact_is_read_only_admissible(self):
        base = Path(__file__).resolve().parents[1] / 'artifacts/laya-learning/laya-20260927T132403Z-2382ddaf'
        files = list(base.glob('**/neural-60b84d8082fe4944863b3a231fa19968/report.json'))
        if not files:
            self.skipTest('private original game evidence absent')
        path = files[0]
        original = path.read_bytes()
        report = json.loads(original)
        self.assertEqual(report['steps'], 442)
        self.assertIsNotNone(safe_observation_transition(report))
        self.assertEqual(path.read_bytes(), original)
        self.assertIsNone(safe_observation_transition({**report, 'safety_reason': 'human_activity'}))


if __name__ == '__main__':
    unittest.main()
