"""Concurrent orchestration contracts with fake jobs; no game or subprocess I/O."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from playmodel.learning.runtime_contract import contract_fields


class PipelineOverlapTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py'
        spec = importlib.util.spec_from_file_location('overlap_cycle_fixture', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls, self.journal = [], []
        self.training = self.run_record('training', 'train')
        self.candidate = {'checkpoint': 'candidate.pt', 'final_kl_within_target': True}
        self.job = Mock()
        self.job.describe.return_value = {'job_id': 'one-fixed-training', 'status_path': 'unused-status'}
        def wait():
            self.calls.append(('wait',))
            return deepcopy(self.candidate)
        self.job.wait.side_effect = wait
        def start(checkpoint, training):
            self.calls.append(('start', str(checkpoint), training['run_id']))
            return self.job
        self.start = Mock(side_effect=start)

    def run_record(self, name, split='evaluation', complete=True):
        return {**contract_fields(), 'runtime_source_hashes': {}, 'run_id': name,
                'session_ids': [name + '-session'], 'split': split,
                'full_run_complete': complete, 'training_eligible': split == 'train' and complete,
                'setup_conditions': {'character_slot': 1}, 'verified_wave_clears': 2}

    def collect(self, checkpoint, *, split, tag):
        self.calls.append(('collect', str(checkpoint), split, tag))
        if tag == 'evaluation-0-source':
            self.assertFalse(self.job.wait.called, 'source gameplay must overlap pending training')
        if tag == 'evaluation-0-candidate':
            self.assertTrue(self.job.wait.called, 'candidate may run only after job result verification')
        return self.training if split == 'train' else self.run_record(tag)

    def pipeline(self, **kwargs):
        return self.module.run_pipeline('source.pt', collect_run=kwargs.pop('collect_run', self.collect),
            train_candidate=lambda *args: self.fail('background path must not train synchronously'),
            start_training=self.start, on_progress=lambda row: self.journal.append(deepcopy(row)), **kwargs)

    def test_source_evaluation_runs_between_worker_start_and_result_wait(self):
        result = self.pipeline()
        self.assertEqual(result['status'], 'comparison_recorded')
        self.assertEqual([row[0] for row in self.calls], ['collect', 'start', 'collect', 'wait', 'collect'])
        self.assertEqual([row['role'] for row in result['evaluations']], ['source', 'candidate'])
        self.assertEqual(result['training_job']['job_id'], 'one-fixed-training')
        self.assertTrue(any(row.get('training_job') and row.get('pending', {}).get('role') == 'source'
                            for row in self.journal if row.get('pending')))
        self.job.close.assert_called_once()

    def test_saved_candidate_never_starts_or_waits_for_another_update(self):
        report = {'source_checkpoint': 'source.pt', 'training': self.training,
                  'candidate': self.candidate, 'evaluations': []}
        result = self.pipeline(resume_report=report,
            collect_run=lambda checkpoint, split, tag: self.run_record(tag))
        self.assertEqual(result['status'], 'comparison_recorded')
        self.start.assert_not_called()
        self.job.wait.assert_not_called()

    def test_failed_source_evaluation_preserves_job_and_does_not_run_candidate(self):
        def collect(checkpoint, *, split, tag):
            if split == 'train':
                return self.training
            self.assertEqual(tag, 'evaluation-0-source')
            return {**self.run_record(tag, complete=False), 'error': 'fixture perception failure'}
        result = self.pipeline(collect_run=collect)
        self.assertEqual(result['status'], 'evaluation_run_incomplete')
        self.assertEqual(result['error'], 'fixture perception failure')
        self.assertIn('training_job', result)
        self.assertNotIn('candidate', result)
        self.job.wait.assert_not_called()
        self.job.close.assert_called_once()

    def test_source_collection_exception_keeps_durable_job_identity(self):
        def collect(checkpoint, *, split, tag):
            if split == 'train':
                return self.training
            raise RuntimeError('fixture process interruption')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.pipeline(collect_run=collect)
        saved = self.journal[-1]
        self.assertEqual(saved['training_job']['job_id'], 'one-fixed-training')
        self.assertEqual(saved['pending']['role'], 'source')
        self.job.close.assert_called_once()

    def test_numerical_rejection_after_source_eval_never_runs_candidate(self):
        self.candidate['final_kl_within_target'] = False
        result = self.pipeline()
        self.assertEqual(result['status'], 'candidate_failed_numerical_gate')
        self.assertEqual([row[0] for row in self.calls], ['collect', 'start', 'collect', 'wait'])
        self.assertEqual([row['role'] for row in result['evaluations']], ['source'])

    def test_resume_preserves_pending_source_collection_operation(self):
        def interrupted(checkpoint, *, split, tag):
            if split == 'train':
                return self.training
            raise RuntimeError('fixture crash')
        with self.assertRaises(RuntimeError):
            self.pipeline(collect_run=interrupted)
        saved = self.journal[-1]
        operation = saved['pending']['operation_id']
        self.journal.clear()
        self.pipeline(resume_report=saved)
        source_intent = next(row for row in self.journal
            if row.get('pipeline_phase') == 'collect_evaluation'
            and (row.get('pending') or {}).get('role') == 'source')
        self.assertEqual(source_intent['pending']['operation_id'], operation)

    def test_reports_actual_optimizer_overlap_separately_from_collection(self):
        self.candidate.update(optimization_started_at_ns=300, optimization_finished_at_ns=700)
        def collect(checkpoint, *, split, tag):
            if split == 'train':
                return self.training
            return {**self.run_record(tag), 'combat_intervals_ns': [[200, 600]]}
        with patch.object(self.module.time, 'perf_counter_ns', side_effect=[100, 1000, 2000, 3000]):
            result = self.pipeline(collect_run=collect)
        self.assertAlmostEqual(result['training_overlap']['source_collection_seconds'], 400 / 1e9)
        self.assertAlmostEqual(result['training_overlap']['actual_movement_seconds'], 300 / 1e9)

    def test_only_first_source_combat_releases_job_across_waves(self):
        coordinator = object.__new__(self.module.LocalCycle)
        coordinator._training_job = self.job
        coordinator._training_gate_released = False
        for split, tag, partial in [('train', 'training', False),
                                    ('evaluation', 'partial-recovery', True),
                                    ('evaluation', 'evaluation-0-candidate', False)]:
            self.assertIsNone(coordinator._training_combat_callback(
                run_id='ignored', split=split, tag=tag, partial=partial))
        callback = coordinator._training_combat_callback(run_id='source-run', split='evaluation',
                                                        tag='evaluation-0-source', partial=False)
        self.job.release_for_combat.assert_not_called()
        callback({'sent_at_ns': 1234, 'session_id': 'first-wave'})
        self.job.release_for_combat.assert_called_once_with(sent_at_ns=1234, run_id='source-run',
                                                            session_id='first-wave')
        self.assertIsNone(coordinator._training_combat_callback(run_id='source-run', split='evaluation',
                                                                tag='evaluation-0-source', partial=False))


if __name__ == '__main__':
    unittest.main()
