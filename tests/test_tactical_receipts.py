"""Hash-bound tactical execution evidence; no game or model inference required."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from playmodel.laya.records import digest, validate_tactical_application, validate_tactical_observation


class TacticalReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.now = 1_000_000_000, 2_000_000_000
        self.identity = {'hwnd': 11, 'pid': 22, 'executable': 'game.exe'}
        source = self.root / 'source.bgra'
        source.write_bytes(b'original source')
        self.evidence = {
            'decision_domain': 'combat_tactic', 'run_id': 'run', 'epoch': 'epoch', 'signature': 'signature',
            'target_identity': self.identity.copy(), 'frame_ref': str(source), 'frame_sha256': digest(source),
            'observed_at_ns': self.base, 'available_at_ns': self.base + 1_000_000,
            'expires_at_ns': self.base + 1_500_000_000,
        }
        self.record = {
            'decision_domain': 'combat_tactic', 'decision_id': 'choice', 'action_id': 'retreat',
            'behavior_version': 'version', 'decided_at_ns': self.base + 2_000_000,
            'state': {'risks': 'unverified'}, 'options': {'retreat': 'Increase gap'},
            'evidence': self.evidence,
        }
        self.bind_observation()

    def bind_observation(self):
        original = {key: value for key, value in self.evidence.items()
                    if key not in ('observation_path', 'observation_sha256')}
        document = {
            'evidence': original,
            'metadata': {**self.evidence['target_identity'],
                         'capture_started_at_ns': self.evidence['observed_at_ns']},
            'situation': {'state': self.record['state'], 'options': self.record['options'],
                'signature': self.evidence['signature'], 'world': {
                    'valid': True, 'observed_at_ns': self.evidence['observed_at_ns'],
                    'available_at_ns': self.evidence['available_at_ns']}},
        }
        path = self.root / 'observation.json'
        path.write_text(json.dumps(document), encoding='utf8')
        self.evidence.update(observation_path=str(path), observation_sha256=digest(path))

    def receipt(self, number=1, kind='native_transition'):
        at = self.base + number * 50_000_000
        keys = [] if kind == 'neutral_hold' else [0x44]
        native = int(kind == 'native_transition')
        frame = self.root / f'frame-{number}.bgra'
        frame.write_bytes(f'frame {number}'.encode())
        return {
            'schema': 'playmodel.tactical-execution.v1', 'domain': 'combat_tactic',
            'execution_id': f'execution-{number}', 'previous_execution_id': None,
            'previous_receipt_path': None, 'previous_receipt_sha256': None,
            'ownership_receipt_path': None, 'ownership_receipt_sha256': None,
            'decision_id': 'choice', 'action_id': 'retreat', 'behavior_version': 'version',
            'run_id': 'run', 'epoch': 'epoch', 'signature': 'signature',
            'target_identity': self.identity.copy(), 'generation': 7, 'sequence': number,
            'action_origin': 'local_laya', 'accepted': True, 'successful_transport_reported': True,
            'transmitted': True, 'acknowledged': None, 'game_application_verified': False,
            'cnn_training_eligible': False, 'writer_authority': 'single_ai_writer',
            'frame_ref': str(frame), 'frame_sha256': digest(frame),
            'actual_movement': 0 if not keys else 3, 'actual_keys': keys,
            'held_before': [] if native else keys.copy(), 'held_after': keys.copy(),
            'native_post_count': native, 'execution_kind': kind,
            'observed_at_ns': at, 'available_at_ns': at + 1_000_000,
            'decided_at_ns': at + 2_000_000, 'transport_started_at_ns': at + 3_000_000,
            'sent_at_ns': at + 4_000_000, 'transport_finished_at_ns': at + 4_000_000,
            'deadline_ns': at + 28_000_000,
            'background_stages': {
                'call_id': number, 'started_at_ns': at + 3_100_000,
                'check_finished_at_ns': at + 3_200_000,
                'posts_started_at_ns': at + 3_300_000 if native else None,
                'posts_finished_at_ns': at + 3_400_000, 'deadline_ns': at + 28_000_000,
                'identity_check_passed': True, 'native_post_attempted': bool(native),
                'attempted_posts': native, 'posted_keys': native,
            },
        }

    def save(self, receipt):
        path = self.root / ('receipt-' + receipt['execution_id'] + '.json')
        path.write_text(json.dumps(receipt), encoding='utf8')
        return {'receipt_path': str(path), 'receipt_sha256': digest(path)}

    def validate(self, receipt):
        return validate_tactical_application(self.record, self.save(receipt), self.now)

    def hold_chain(self):
        anchor = self.receipt()
        anchor['action_origin'] = 'explicit_rule_fallback'
        anchor_ref = self.save(anchor)
        previous, previous_ref = anchor, anchor_ref
        for number in (2, 3):
            current = self.receipt(number, 'existing_owned_hold')
            current.update(previous_execution_id=previous['execution_id'],
                previous_receipt_path=previous_ref['receipt_path'],
                previous_receipt_sha256=previous_ref['receipt_sha256'],
                ownership_receipt_path=anchor_ref['receipt_path'],
                ownership_receipt_sha256=anchor_ref['receipt_sha256'])
            previous, previous_ref = current, self.save(current)
        return anchor, current

    def test_native_and_neutral_executions_keep_game_application_unknown(self):
        for kind in ('native_transition', 'neutral_hold'):
            with self.subTest(kind=kind):
                result = self.validate(self.receipt(kind=kind))
                self.assertIs(result['game_application_verified'], False)

    def test_existing_owned_hold_requires_immediate_and_original_ownership(self):
        _, held = self.hold_chain()
        self.assertEqual(self.validate(held)['native_post_count'], 0)
        self.assertEqual(self.validate(held)['execution_kind'], 'existing_owned_hold')

    def test_source_observation_binds_exact_state_options_metadata_and_world(self):
        validate_tactical_observation(self.record['state'], self.record['options'], self.evidence, self.now)
        for field, value in (('state', {'invented': True}), ('options', {'invented': 'Move'})):
            changed = deepcopy(self.record)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_tactical_application(changed, self.save(self.receipt()), self.now)
        path = Path(self.evidence['observation_path'])
        document = json.loads(path.read_text())
        document['metadata']['pid'] = 999
        path.write_text(json.dumps(document), encoding='utf8')
        self.evidence['observation_sha256'] = digest(path)
        with self.assertRaises(ValueError):
            self.validate(self.receipt())

    def test_source_observation_and_source_pixels_tampering_rejected(self):
        for key in ('observation_path', 'frame_ref'):
            with self.subTest(key=key):
                path = Path(self.evidence[key])
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaises(ValueError):
                    self.validate(self.receipt())
                path.write_bytes(original)

    def test_choice_expiry_including_exact_boundary_rejects_execution(self):
        receipt = self.receipt()
        for expires in (receipt['transport_started_at_ns'], receipt['sent_at_ns']):
            with self.subTest(expires=expires):
                self.evidence['expires_at_ns'] = expires
                self.bind_observation()
                with self.assertRaises(ValueError):
                    self.validate(receipt)

    def test_future_choice_or_source_availability_rejected(self):
        receipt = self.receipt()
        self.record['decided_at_ns'] = receipt['transport_started_at_ns'] + 1
        with self.assertRaises(ValueError):
            self.validate(receipt)
        self.record['decided_at_ns'] = receipt['decided_at_ns']
        self.evidence['available_at_ns'] = receipt['available_at_ns'] + 1
        self.bind_observation()
        with self.assertRaises(ValueError):
            self.validate(receipt)

    def test_wrong_target_domain_or_fallback_never_becomes_laya_application(self):
        for field, value in (('target_identity', {**self.identity, 'pid': 999}),
                             ('domain', 'menu'), ('action_origin', 'explicit_rule_fallback'),
                             ('game_application_verified', True), ('cnn_training_eligible', True),
                             ('transmitted', False), ('acknowledged', 1), ('generation', True)):
            with self.subTest(field=field):
                receipt = self.receipt()
                receipt[field] = value
                with self.assertRaises(ValueError):
                    self.validate(receipt)

    def test_native_stage_numbers_and_partial_post_counts_rejected(self):
        for field, value in (('call_id', True), ('call_id', 0), ('started_at_ns', float('nan')),
                             ('check_finished_at_ns', None), ('posts_started_at_ns', None),
                             ('attempted_posts', 2), ('posted_keys', 0)):
            with self.subTest(field=field, value=value):
                receipt = self.receipt()
                receipt['background_stages'][field] = value
                with self.assertRaises(ValueError):
                    self.validate(receipt)

    def test_changed_current_frame_and_receipt_hash_rejected(self):
        receipt = self.receipt()
        application = self.save(receipt)
        Path(application['receipt_path']).write_text('{}', encoding='utf8')
        with self.assertRaises(ValueError):
            validate_tactical_application(self.record, application, self.now)
        Path(receipt['frame_ref']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.validate(receipt)

    def test_hold_cannot_skip_predecessor_or_reuse_wrong_epoch_or_call(self):
        for field, value in (('previous_execution_id', None), ('previous_execution_id', 'wrong'),
                             ('previous_receipt_sha256', 'wrong'), ('epoch', 'other'),
                             ('generation', 8), ('sequence', 2)):
            with self.subTest(field=field):
                _, held = self.hold_chain()
                held[field] = value
                with self.assertRaises(ValueError):
                    self.validate(held)
        _, held = self.hold_chain()
        held['background_stages']['call_id'] = 5
        with self.assertRaises(ValueError):
            self.validate(held)

    def test_hold_rejects_predecessor_after_release_or_different_native_transition(self):
        _, held = self.hold_chain()
        previous = json.loads(Path(held['previous_receipt_path']).read_text())
        previous.update(held_before=[], held_after=[], actual_keys=[], actual_movement=0,
                        execution_kind='neutral_hold', ownership_receipt_path=None, ownership_receipt_sha256=None)
        proof = self.save(previous)
        held['previous_receipt_sha256'] = proof['receipt_sha256']
        with self.assertRaises(ValueError):
            self.validate(held)

    def test_hold_rejects_predecessor_anchor_change_even_with_matching_keys(self):
        _, held = self.hold_chain()
        previous = json.loads(Path(held['previous_receipt_path']).read_text())
        previous['ownership_receipt_sha256'] = 'another anchor'
        proof = self.save(previous)
        held['previous_receipt_sha256'] = proof['receipt_sha256']
        with self.assertRaises(ValueError):
            self.validate(held)

    def test_hold_anchor_cannot_be_copied_from_another_combat_directory(self):
        _, held = self.hold_chain()
        other = self.root / 'other-combat'
        other.mkdir()
        path = other / 'anchor.json'
        path.write_bytes(Path(held['ownership_receipt_path']).read_bytes())
        held['ownership_receipt_path'] = str(path)
        with self.assertRaises(ValueError):
            self.validate(held)


if __name__ == '__main__':
    unittest.main()
