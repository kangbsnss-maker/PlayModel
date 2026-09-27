"""Observed build context crosses the real menu/combat adapter boundaries."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional learning dependency')
class BuildStateIntegrationTests(unittest.TestCase):
    def test_menu_and_movement_use_same_extra_context_and_freeze_provenance(self):
        from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic
        from playmodel.games.brotato.neural_choices import ChoiceObservation, MenuCandidate, sample_choice
        from playmodel.games.brotato.neural_runtime import _inputs
        model = RecurrentActorCritic(ModelConfig(context_dim=64))
        class Build:
            value = 0.25
            def features(self, observed_at_ns, *, available_at_ns=None):
                return (self.value,) + (0.0,) * 47
            def snapshot(self):
                return {'schema': 'brotato-observed-build-v2', 'value': self.value, 'sources': []}
        state = Build()
        candidate = MenuCandidate('upgrade:0', 'choose_0', 'upgrade', 'stable',
                                  ('+1 Armor',), (0.0,) * 16, True, 'visible')
        observation = ChoiceObservation('frame', 'pixels', 'ocr', 100, 200, 110,
            'level_up', 1, bytes(96 * 96 * 3), (candidate,), 'cards')
        decision = sample_choice(model, observation, build_state=state, now_ns=300)
        frame = SimpleNamespace(metadata={'sample_width': 96, 'sample_height': 96,
                                'capture_started_at_ns': 400}, pixels=bytes(96 * 96 * 4), available_at_ns=500)
        movement = _inputs(model, frame, 2, build_state=state)
        self.assertEqual(len(decision.context), 64)
        self.assertTrue(torch.equal(movement[1][0, 16:], torch.tensor(decision.context[16:])))
        self.assertEqual(movement[1][0, 2], 1)
        state.value = 0.75
        self.assertEqual(json.loads(decision.build_state_json)['value'], 0.25)
        self.assertEqual(decision.context[16], 0.25)
        with self.assertRaisesRegex(ValueError, 'observed build'):
            _inputs(model, frame, 2)

    def test_build_evidence_binds_original_and_derived_files(self):
        from playmodel.learning.full_run import build_source_proofs
        with tempfile.TemporaryDirectory() as directory:
            frame, ocr = Path(directory) / 'frame.png', Path(directory) / 'stats-ocr.json'
            frame.write_bytes(b'original')
            ocr.write_bytes(b'local OCR observation')
            snapshot = {'sources': [{'frame_ref': str(frame),
                'source_png_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
                'derived_sources': [{'path': str(ocr), 'sha256': hashlib.sha256(ocr.read_bytes()).hexdigest()}]}]}
            self.assertEqual(len(build_source_proofs(snapshot)), 2)
            ocr.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'derived evidence modified'):
                build_source_proofs(snapshot)


if __name__ == '__main__':
    unittest.main()
