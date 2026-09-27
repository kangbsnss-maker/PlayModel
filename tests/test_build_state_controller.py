"""Synthetic ROI observations traverse the real context64 menu controller."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('optional learning dependency')

import torch

from playmodel.games.brotato.capture import _png
from playmodel.games.brotato.neural_menu_controller import NeuralMenuController
from playmodel.games.brotato.state_features import make_stats_observation, STAT_LABELS, STATS
from playmodel.learning.full_run import FullRunRecorder
from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic
from test_shop_learning import BASE_TIME, frame, report


class BuildStateControllerTests(unittest.TestCase):
    def test_shop_roi_pair_precedes_less_detailed_inventory_observation(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        model = RecurrentActorCritic(ModelConfig(context_dim=64))
        recorder = FullRunRecorder(model, 'synthetic-build-controller')
        now = [BASE_TIME]
        controller = NeuralMenuController(recorder, clock=lambda: now[0])
        # Full-frame OCR has no stat rows. Its independent ROI has all 16.
        values = (25, 0, 0, -3, 0, 4, 0, 5, 6, 0, 25, 3, 0, 5, 0, 10)
        with tempfile.TemporaryDirectory() as directory:
            for sequence in (1, 2):
                root = Path(directory) / f'frame-{sequence}'
                root.mkdir()
                pixels = frame()
                png = _png(1920, 1080, pixels, compression_level=1)
                source = root / 'frame.png'
                source.write_bytes(png)
                observed = BASE_TIME + sequence * 100_000_000
                available = observed + 20_000_000
                ocr = report(sequence=sequence)
                ocr['available_at_ns'] = available
                stats = make_stats_observation(ocr, frame_ref=str(source),
                    pixels_sha256=hashlib.sha256(pixels).hexdigest(), observed_at_ns=observed,
                    available_at_ns=available, scene='shop')
                stats['raw_text'] = tuple(f'{label} {value}' for label, value in zip(STAT_LABELS, values))
                ocr['_local_stats'] = stats
                shot = {'session_directory': str(root), 'capture_started_at_ns': observed,
                        'available_at_ns': observed + 5_000_000,
                        'frame_sha256': hashlib.sha256(png).hexdigest()}
                now[0] = available + 1_000_000
                directive = controller.handle(shot, ocr, pixels)
                if sequence == 1:
                    self.assertEqual(directive.status, 'wait')
                    self.assertIsNone(controller.pending_decision)
            self.assertEqual(directive.status, 'navigate')
            decision = controller.pending_decision
            self.assertEqual(len(decision.context), 64)
            self.assertEqual(decision.context[32:48], (1.0,) * 16)
            for actual, expected in zip(decision.context[16:32], values):
                normalized = math.copysign(math.log1p(abs(expected)) / math.log1p(1000), expected)
                self.assertAlmostEqual(actual, normalized)
            snapshot = json.loads(decision.build_state_json)
            self.assertEqual(snapshot['stats'], dict(zip(STATS, values)))
            self.assertEqual(snapshot['weapon_count'], 2)
            self.assertEqual(snapshot['stats_sources'][-1]['raw_text'], list(stats['raw_text']))
            self.assertEqual(controller.samples, 1)
            self.assertFalse(controller.awaiting_application)
            self.assertEqual(len(recorder.records), 0)


if __name__ == '__main__':
    unittest.main()
