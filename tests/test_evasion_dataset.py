import unittest
from copy import deepcopy
from types import SimpleNamespace

from playmodel.games.brotato.combat_experience import CombatExperience
from playmodel.games.brotato.evasion_dataset import examples, features
from playmodel.learning.outcome_values import OutcomeValues
from test_combat_experience import frame, situation, track


class EvasionDatasetTests(unittest.TestCase):
    def rows(self):
        memory = CombatExperience()
        vision = SimpleNamespace(camera_status='unknown', camera_shift=None, hp_fill_fraction=.8)
        rows = []
        for time in (1000000000, 1100000000):
            sample = memory.observe(frame(time), situation(time, [track()]), vision)
            rows.append({'combat_experience': sample, 'actual_movement': 3,
                         'execution_id': str(time), 'previous_execution_id': str(time-100000000),
                         'sent_at_ns': time+1000, 'frame_ref': str(time), 'frame_sha256': 'hash'})
        return rows

    def test_factorization_unknown_map_and_weapon_effects(self):
        rows = self.rows()
        source = examples(rows, 'run')[0]['input']
        self.assertEqual(len(source['candidates']), 9)
        self.assertIsNone(source['environment']['map_size'])
        self.assertFalse(source['environment']['weapon_effects_known'])
        self.assertTrue(all(c['safe'] is None for c in source['candidates']))

    def test_future_targets_never_enter_features(self):
        row = examples(self.rows(), 'run')[0]
        before = features(row)
        row['target']['clearance_proxy'] = -100
        row['target']['hp_fraction'] = 0
        self.assertEqual(features(row), before)
        self.assertFalse(row['bc_label'])
        self.assertEqual(row['split_group'], 'run')

    def test_gap_and_epoch_censor(self):
        rows = self.rows()
        rows[1]['combat_experience']['environment']['epoch'] = 'other'
        row = examples(rows, 'run')[0]
        self.assertTrue(row['target']['censored'])
        self.assertIsNone(row['target']['clearance_proxy'])
        rows = self.rows()
        rows[1]['combat_experience']['observed_at_ns'] += 1000000000
        self.assertTrue(examples(rows, 'run')[0]['target']['censored'])

    def test_untransmitted_or_future_action_cannot_label(self):
        rows = self.rows()
        rows[0]['sent_at_ns'] = rows[1]['combat_experience']['observed_at_ns']
        self.assertEqual(examples(rows, 'run'), [])

    def test_camera_shift_is_not_map_resize(self):
        rows = self.rows()
        rows[1]['combat_experience']['environment']['camera_shift'] = [.3, .2]
        self.assertIsNone(examples(rows, 'run')[0]['input']['environment']['map_size'])

    def test_predictor_learns_without_changing_input(self):
        row = examples(self.rows(), 'run')[0]
        model = OutcomeValues()
        before = deepcopy(model.state())
        model.fit([(features(row), .2)])
        self.assertNotEqual(before, model.state())
        restored = OutcomeValues(model.state())
        self.assertEqual(restored.predict(features(row)), model.predict(features(row)))
