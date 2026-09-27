"""Tactical transport admission is separate from menu/game application proof."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from playmodel.laya.records import digest
from playmodel.laya.worker import Learner
from playmodel.laya.client import LayaClient
from playmodel.laya.preferences import default_preferences, preference_hash


class TacticalWorkerTests(unittest.TestCase):
    def test_client_uses_separate_tactical_and_discard_protocol_methods(self):
        client = LayaClient.__new__(LayaClient)
        with patch.object(client, '_request', return_value={'status': 'accepted'}) as request:
            client.accept_tactic('d', {'receipt_path': 'proof'})
            request.assert_called_once_with('accept_tactic', decision_id='d', application={'receipt_path': 'proof'})
        with patch.object(client, '_request', return_value={'status': 'discarded'}) as request:
            client.discard('d', 'expired')
            request.assert_called_once_with('discard', decision_id='d', reason='expired')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = self.root / 'frame.png'
        self.frame.write_bytes(b'synthetic fixture')
        self.learner = Learner.__new__(Learner)
        self.learner.output = self.root
        self.learner.version = 'v3'
        self.learner.accepted = []
        self.learner.events_used = set()
        self.learner.torch = None
        self.learner.preferences = default_preferences()
        self.learner.preference_hash = preference_hash(self.learner.preferences)
        self.record = {'decision_id': 'd', 'decision_domain': 'combat_tactic',
                       'action_id': 'retreat', 'behavior_version': 'v3', 'decided_at_ns': 110,
                       'state': {'scene': 'combat'}, 'options': {'retreat': 'Retreat'},
                       'evidence': {'run_id': 'run', 'epoch': 'e', 'signature': 's',
                                    'decision_domain': 'combat_tactic', 'available_at_ns': 105,
                                    'expires_at_ns': 180, 'target_identity': {'hwnd': 1, 'pid': 2, 'executable': 'game.exe'},
                                    'observed_at_ns': 100, 'frame_ref': str(self.frame),
                                    'frame_sha256': digest(self.frame)}}
        self.choice = self.root / 'choice-d.json'
        visual = self.root / 'visual.rgb'
        visual.write_bytes(bytes(3 * 96 * 96))
        self.record['tokens'] = {'visual_window': [{'path': str(visual), 'sha256': digest(visual),
            'source_path': str(self.frame), 'source_sha256': digest(self.frame),
            'observed_at_ns': 100, 'available_at_ns': 105,
            'transform': 'bgra_to_rgb_area96_round_uint8_v1'}]}
        self.record['preference_hash'] = self.learner.preference_hash
        observation = self.root / 'observation.json'
        observation.write_text(json.dumps({'evidence': self.record['evidence'],
            'metadata': {'hwnd': 1, 'pid': 2, 'executable': 'game.exe', 'capture_started_at_ns': 100},
            'situation': {'state': self.record['state'], 'options': self.record['options'],
                'signature': 's', 'world': {'valid': True, 'observed_at_ns': 100,
                    'available_at_ns': 105, 'fresh_until_ns': 180}}}), encoding='utf8')
        self.record['evidence'].update(observation_path=str(observation), observation_sha256=digest(observation))
        self.choice.write_text(json.dumps(self.record), encoding='utf8')
        self.record['_choice_sha256'] = digest(self.choice)
        self.learner.pending = {'d': self.record}
        self.receipt = {k: self.record[k] for k in ('decision_id', 'action_id', 'behavior_version')}
        self.receipt.update(run_id='run', epoch='e', signature='s', accepted=True,
                            execution_id='exec1', previous_execution_id=None,
                            schema='playmodel.tactical-execution.v1', domain='combat_tactic',
                            action_origin='local_laya', cnn_training_eligible=False,
                            transmitted=True, acknowledged=None, generation=1, sequence=1,
                            successful_transport_reported=True, game_application_verified=False,
                            writer_authority='single_ai_writer', observed_at_ns=120, available_at_ns=125,
                            decided_at_ns=126, transport_started_at_ns=130, transport_finished_at_ns=140,
                            sent_at_ns=140, deadline_ns=150,
                            frame_ref=str(self.frame), frame_sha256=digest(self.frame),
                            actual_movement=1, actual_keys=[0x57], held_before=[], held_after=[0x57], native_post_count=1,
                            execution_kind='native_transition', target_identity={'hwnd': 1, 'pid': 2, 'executable': 'game.exe'},
                            background_stages={'posted_keys': 1, 'attempted_posts': 1,
                                'identity_check_passed': True, 'native_post_attempted': True,
                                'deadline_ns': 150, 'started_at_ns': 130, 'posts_started_at_ns': 134, 'call_id': 1,
                                'check_finished_at_ns': 132, 'posts_finished_at_ns': 138})

    def application(self):
        path = self.root / 'receipt.json'
        path.write_text(json.dumps(self.receipt), encoding='utf8')
        return {'receipt_path': str(path), 'receipt_sha256': digest(path)}

    def outcome(self, observed=160, run='run'):
        terminal = self.root / 'terminal.json'
        terminal.write_text(json.dumps({'kind': 'wave_clear', 'verified': True,
            'independent_of_policy': True, 'observed_at_ns': observed,
            'frame_ref': str(self.frame), 'frame_sha256': digest(self.frame)}), encoding='utf8')
        report = self.root / 'report.json'
        actions = self.root / 'tactical-actions.jsonl'
        actions.write_text(json.dumps(self.receipt) + '\n', encoding='utf8')
        receipts = [{'path': str(self.root / 'receipt.json'), 'sha256': self.record['application']['receipt_sha256']}] if 'application' in self.record else []
        report.write_text(json.dumps({'terminal_kind': 'wave_clear', 'reason': 'terminal_wave_clear',
            'session_directory': str(self.root), 'action_domain': 'combat_tactic', 'run_id': 'run',
            'tactical_epoch': 'e', 'tactical_receipts': receipts, 'actions_path': str(actions),
            'actions_sha256': digest(actions), 'verified_terminal_boundary': True,
            'tactical_collection_eligible': True, 'recorder_complete': True, 'worker_stopped': True}), encoding='utf8')
        return {'path': str(terminal), 'sha256': digest(terminal), 'report_path': str(report),
                'report_sha256': digest(report), 'run_id': run}

    def test_accept_preserves_transport_only_semantics(self):
        self.learner.accept_tactic('d', self.application())
        self.assertFalse(self.record['application']['game_application_verified'])
        self.assertNotIn('d', self.learner.pending)
        self.assertEqual(len(self.learner.accepted), 1)

    def test_menu_api_cannot_accept_combat(self):
        with self.assertRaisesRegex(ValueError, 'tactical acceptance'):
            self.learner.accept('d', {})

    def test_tactical_api_cannot_accept_menu(self):
        self.record['decision_domain'] = 'menu'
        with self.assertRaisesRegex(ValueError, 'combat decision'):
            self.learner.accept_tactic('d', self.application())

    def test_discard_only_pending_preserves_accepted(self):
        self.learner.accepted.append({'decision_id': 'earlier'})
        self.learner.discard('d', 'expired request')
        self.assertEqual(self.learner.accepted, [{'decision_id': 'earlier'}])
        self.assertTrue((self.root / 'discarded-d.json').exists())
        with self.assertRaises(ValueError):
            self.learner.discard('earlier', 'cannot erase accepted')

    def test_outcome_after_send_before_validation_is_causal(self):
        with patch('playmodel.laya.worker.time.perf_counter_ns', return_value=200):
            self.learner.accept_tactic('d', self.application())
        result = self.learner.finish('wave_clear', self.outcome(160))
        self.assertEqual(result['reason'], 'forced_actions_have_no_policy_gradient')

    def test_outcome_before_send_excluded_without_stopping(self):
        self.learner.accept_tactic('d', self.application())
        result = self.learner.finish('wave_clear', self.outcome(139))
        self.assertEqual(result['reason'], 'no_choices_before_terminal_observation')
        self.assertEqual(self.learner.accepted, [])
        evidence = json.loads(next(self.root.glob('boundary-excluded-*.json')).read_text())
        self.assertEqual(evidence['decisions'], ['d'])
        self.assertFalse(evidence['reward_assigned'])

    def test_wrong_version_still_rejected_even_after_outcome(self):
        self.learner.accept_tactic('d', self.application())
        self.record['behavior_version'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'follow'):
            self.learner.finish('wave_clear', self.outcome(139))

    def test_late_receipt_tampering_still_rejected(self):
        application = self.application()
        self.learner.accept_tactic('d', application)
        Path(application['receipt_path']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            self.learner.finish('wave_clear', self.outcome(139))

    def test_cross_run_outcome_rejected(self):
        self.learner.accept_tactic('d', self.application())
        with self.assertRaisesRegex(ValueError, 'decision run'):
            self.learner.finish('wave_clear', self.outcome(run='other'))

    def test_receipt_changed_before_learning_rejected(self):
        application = self.application()
        self.learner.accept_tactic('d', application)
        Path(application['receipt_path']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            self.learner.finish('wave_clear', self.outcome())

    def test_partial_native_post_rejected(self):
        self.receipt['background_stages']['attempted_posts'] = 2
        with self.assertRaises(ValueError):
            self.learner.accept_tactic('d', self.application())
        self.assertIn('d', self.learner.pending)

    def test_same_run_different_wave_report_rejected(self):
        self.learner.accept_tactic('d', self.application())
        evidence = self.outcome()
        path = Path(evidence['report_path'])
        value = json.loads(path.read_text())
        value['tactical_epoch'] = 'other-wave'
        path.write_text(json.dumps(value), encoding='utf8')
        evidence['report_sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'combat session'):
            self.learner.finish('wave_clear', evidence)

    def test_ledger_disagrees_with_receipt_rejected(self):
        self.learner.accept_tactic('d', self.application())
        self.receipt['action_id'] = 'orbit_left'
        with self.assertRaisesRegex(ValueError, 'action ledger'):
            self.learner.finish('wave_clear', self.outcome())

    def test_original_observation_mutation_rejected(self):
        self.learner.accept_tactic('d', self.application())
        Path(self.record['evidence']['observation_path']).write_text('{"changed":true}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'observation artifact'):
            self.learner.finish('wave_clear', self.outcome())


if __name__ == '__main__':
    unittest.main()
