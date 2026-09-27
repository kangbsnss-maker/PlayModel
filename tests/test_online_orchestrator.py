"""Online journal/version handoff checks without game input or model loading."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class OnlineJournalTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('online_orchestrator_test',
            Path(__file__).resolve().parents[1] / 'scripts/run_online_learning.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'original')
        self.candidate = self.root / 'candidate.pt'
        self.candidate.write_bytes(b'updated')
        self.initial = {'checkpoint': str(self.source), 'checkpoint_sha256': self.module.digest(self.source),
                        'active_run_id': 'current', 'active_adoption_count': 0}

    def test_journal_rejects_changed_checkpoint_on_resume(self):
        self.module.OnlineState(self.root, self.initial)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            self.module.OnlineState(self.root)

    def test_only_current_run_monotonic_adoptions_move_checkpoint(self):
        state = self.module.OnlineState(self.root, self.initial)
        snapshot = {'run_id': 'current', 'checkpoint': str(self.candidate), 'adoptions': [{}, {}]}
        state.applied({**snapshot, 'run_id': 'previous'})
        self.assertEqual(state.value['checkpoint'], str(self.source))
        state.applied(snapshot)
        self.assertEqual(Path(state.value['checkpoint']), self.candidate.resolve())
        state.applied({'run_id': 'current', 'checkpoint': str(self.source), 'adoptions': [{}]})
        self.assertEqual(Path(state.value['checkpoint']), self.candidate.resolve())
        restored = self.module.OnlineState(self.root)
        self.assertEqual(restored.value['active_adoption_count'], 2)

    def test_online_seed_prefers_verified_saved_candidate_without_altering_legacy(self):
        directory = self.root / 'previous'
        directory.mkdir()
        path = directory / 'session-state.json'
        original = json.dumps({'checkpoint': str(self.source), 'pipeline': {'candidate': {
            'checkpoint': str(self.candidate), 'final_kl_within_target': True,
            'checkpoint_reload_verified': True}}})
        path.write_text(original)
        args = SimpleNamespace(resume_summary=None, resume_latest=True, output=self.root, checkpoint=self.source)
        selected, origin = self.module.seed_checkpoint(args)
        self.assertEqual(selected, self.candidate.resolve())
        self.assertEqual(Path(origin).resolve(), path.resolve())
        self.assertEqual(path.read_text(), original)

    def test_rejected_candidate_does_not_replace_seed(self):
        directory = self.root / 'previous'
        directory.mkdir()
        (directory / 'session-state.json').write_text(json.dumps({'checkpoint': str(self.source),
            'pipeline': {'candidate': {'checkpoint': str(self.candidate), 'final_kl_within_target': False,
                                       'checkpoint_reload_verified': True}}}))
        args = SimpleNamespace(resume_summary=None, resume_latest=True, output=self.root, checkpoint=self.candidate)
        self.assertEqual(self.module.seed_checkpoint(args)[0], self.source.resolve())
