"""Outcome admission must reject unsupported rewards and modified evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from playmodel.laya.records import (digest, validate_distribution, validate_options,
                                    validate_resume_report, verified_outcome)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = self.root / 'frame.png'
        self.frame.write_bytes(b'unit-test evidence, not game observations')
        self.event = self.root / 'terminal.json'
        self.data = {'kind': 'death', 'independent_of_policy': True, 'verified': True,
                     'observed_at_ns': 100, 'frame_ref': str(self.frame), 'frame_sha256': digest(self.frame)}

    def write(self):
        self.event.write_text(json.dumps(self.data), encoding='utf8')
        return {'path': str(self.event), 'sha256': digest(self.event)}

    def test_independent_game_event_has_reward(self):
        self.assertEqual(verified_outcome('death', self.write())[:2], (-1.0, 100))

    def test_stop_and_unknown_are_not_negative_examples(self):
        for kind in ('aborted', 'stopped', 'unknown', 'timeout'):
            with self.assertRaises(ValueError):
                verified_outcome(kind, self.write())

    def test_unverified_or_self_judged_result_rejected(self):
        for key in ('verified', 'independent_of_policy'):
            self.data[key] = False
            with self.assertRaises(ValueError):
                verified_outcome('death', self.write())
            self.data[key] = True

    def test_event_and_frame_mutation_rejected(self):
        proof = self.write()
        self.event.write_text('{}', encoding='utf8')
        with self.assertRaises(ValueError):
            verified_outcome('death', proof)
        proof = self.write()
        self.frame.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verified_outcome('death', proof)

    def test_wrong_event_kind_rejected(self):
        with self.assertRaises(ValueError):
            verified_outcome('wave_clear', self.write())

    def test_choice_distribution_contract(self):
        validate_options({'one': 'Buy', 'two': 'Skip'})
        validate_distribution(['one', 'two'], [.3, .7])
        for p in ([.3, .8], [float('nan'), 0], [-.1, 1.1], [1]):
            with self.assertRaises(ValueError):
                validate_distribution(['one', 'two'], p)
        with self.assertRaises(ValueError):
            validate_options({'one': ''})

    def test_rejected_checkpoint_cannot_resume(self):
        bundle = {'encoder_hash': 'encoder', 'head_hash': 'new', 'report': {
            'accepted': True, 'optimizer_steps': 1, 'encoder_hash_before': 'encoder',
            'encoder_hash_after': 'encoder', 'head_hash_before': 'old', 'head_hash_after': 'new',
            'kl_per_choice': [0.002]}}
        validate_resume_report(bundle)
        bundle['report']['accepted'] = False
        with self.assertRaises(ValueError):
            validate_resume_report(bundle)
        bundle['report']['accepted'] = True
        bundle['report']['kl_per_choice'] = [0.1]
        with self.assertRaises(ValueError):
            validate_resume_report(bundle)


if __name__ == '__main__':
    unittest.main()
