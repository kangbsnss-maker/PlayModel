"""Semantic phase changes restart comparisons without rewriting frozen history."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from playmodel.learning.runtime_contract import (
    LEGACY_PHASE_SCHEMA, LEGACY_RUNTIME_CONTRACT, PHASE_SCHEMA, RUNTIME_CONTRACT,
    contract_fields, contract_identity, require_current_contract,
)


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        specification = importlib.util.spec_from_file_location('contract_cycle_test',
            Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py')
        self.cycle = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.cycle)

    def legacy_report(self):
        return {'source_checkpoint': 'source.pt', 'evaluation_runs': 1,
            'training': {'run_id': 'old-training', 'session_ids': ['old-training-session'],
                'training_eligible': True, 'full_run_complete': True,
                'phase_schema': LEGACY_PHASE_SCHEMA, 'steps': 746,
                'setup_conditions': {'character_slot': 1}},
            'candidate': {'checkpoint': 'candidate.pt', 'final_kl_within_target': True,
                          'optimizer_updates': 48},
            'evaluations': [], 'excluded_evaluations': [],
            'pending': {'operation_id': 'old-pending-operation', 'role': 'source',
                        'checkpoint': 'source.pt'}}

    def test_missing_tags_are_legacy_and_inconsistent_pairs_are_rejected(self):
        document = {'phase_schema': LEGACY_PHASE_SCHEMA}
        self.assertEqual(contract_identity(document), (LEGACY_RUNTIME_CONTRACT, LEGACY_PHASE_SCHEMA))
        self.assertNotIn('runtime_contract', document)
        with self.assertRaisesRegex(ValueError, 'read-only'):
            require_current_contract(document)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            contract_identity({'runtime_contract': RUNTIME_CONTRACT, 'phase_schema': LEGACY_PHASE_SCHEMA})
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            contract_identity({'phase_schema': PHASE_SCHEMA})

    def test_old_pending_candidate_is_preserved_and_both_roles_use_fresh_contract(self):
        legacy = self.legacy_report()
        original = deepcopy(legacy)
        source_hashes = {'neural_choices.py': 'new-source-hash'}
        calls = []
        def collect(checkpoint, *, split, tag):
            calls.append((checkpoint, split))
            return {**contract_fields(), 'runtime_source_hashes': source_hashes,
                'run_id': tag, 'session_ids': [tag], 'full_run_complete': True,
                'setup_conditions': {'character_slot': 1}, 'verified_wave_clears': 3}
        result = self.cycle.run_pipeline('source.pt', collect_run=collect,
            train_candidate=lambda *a: self.fail('saved candidate must not train twice'),
            resume_report=legacy, runtime_source_hashes=source_hashes)
        self.assertEqual(legacy, original)
        self.assertEqual(result['training'], original['training'])
        self.assertEqual(result['candidate'], original['candidate'])
        self.assertEqual(calls, [('source.pt', 'evaluation'), ('candidate.pt', 'evaluation')])
        self.assertEqual(result['excluded_evaluation_scopes'][0]['pending'], original['pending'])
        self.assertEqual(result['mean_verified_wave_clears'], {'source': 3., 'candidate': 3.})
        self.assertTrue(all(row['evaluation_scope_id'] == result['evaluation_scope_id']
                            for row in result['evaluations']))

    def test_legacy_training_without_saved_candidate_never_reenters_ppo(self):
        report = self.legacy_report()
        report.pop('candidate')
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.cycle.run_pipeline('source.pt', collect_run=lambda *a, **k: self.fail('must reject first'),
                train_candidate=lambda *a: self.fail('old rollout must not train'), resume_report=report)

    def test_source_change_excludes_complete_scores_and_preserves_pending_archive(self):
        old = self.cycle._bind_evaluation_scope(self.legacy_report(), {'menu.py': 'before'})
        old['evaluations'] = [{**contract_fields(), 'run_id': 'old-evaluation',
            'session_ids': ['old-evaluation-session'], 'full_run_complete': True,
            'setup_conditions': {'character_slot': 1}, 'role': 'source', 'evaluation_index': 0,
            'evaluation_scope_id': old['evaluation_scope_id'], 'runtime_source_hashes': {'menu.py': 'before'},
            'verified_wave_clears': 999}]
        old['pending'] = {'operation_id': 'old-candidate-eval', 'role': 'candidate'}
        bound = self.cycle._bind_evaluation_scope(old, {'menu.py': 'after'})
        self.assertEqual(bound['evaluations'], [])
        self.assertIsNone(bound['pending'])
        self.assertNotEqual(bound['evaluation_scope_id'], old['evaluation_scope_id'])
        self.assertEqual(bound['excluded_evaluations'][-1]['verified_wave_clears'], 999)
        self.assertEqual(bound['excluded_evaluation_scopes'][-1]['pending'], old['pending'])
        again = self.cycle._bind_evaluation_scope(bound, {'menu.py': 'after'})
        self.assertEqual(again, bound)

    def test_incompatible_fresh_evaluation_is_rejected_before_scoring(self):
        legacy = self.legacy_report()
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.cycle.run_pipeline('source.pt', collect_run=lambda *a, **k: {
                'run_id': 'wrong-runtime', 'full_run_complete': True, 'verified_wave_clears': 999},
                train_candidate=lambda *a: self.fail('no retraining'), resume_report=legacy)


if __name__ == '__main__':
    unittest.main()
