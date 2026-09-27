from contextlib import nullcontext
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from playmodel.games.brotato import setup_run


class SetupRecoveryTests(unittest.TestCase):
    def test_interrupted_difficulty_backtracking_sends_once_per_scene(self):
        phases = iter(['difficulty', 'difficulty', 'weapon', 'weapon', 'character', 'weapon', 'difficulty'])
        active = {'phase': None, 'index': 0}
        pixels = bytes([255]) * (1920 * 1080 * 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def capture(directory, **kwargs):
                active.update(phase=next(phases), index=active['index'] + 1)
                target = root / str(active['index'])
                target.mkdir()
                return {'session_directory': str(target), 'hwnd': 1,
                        'frame_sha256': f"{active['index']:064x}",
                        'capture_started_at_ns': time.perf_counter_ns()}, pixels, 1920, 1080
            def rows(ocr, rect):
                if rect == (500,60,1450,160):
                    return [{'difficulty':'Difficultyselection', 'character':'Characterselection'}.get(active['phase'], '')]
                if rect == (20,20,300,100):
                    return ['Back']
                if rect == (1290,180,1520,220):
                    return ['SMG']
                if rect in ((210,175,975,280), (415,170,1180,265)):
                    return ['Well-Rounded']
                return []
            cap, reader = MagicMock(), MagicMock()
            cap.__enter__.return_value.read.side_effect = capture
            reader.__enter__.return_value.read.return_value = {}
            with patch.object(setup_run, 'MenuCapture', return_value=cap), \
                 patch.object(setup_run, 'MenuOcr', return_value=reader), \
                 patch.object(setup_run, 'session_lock', return_value=nullcontext()), \
                 patch.object(setup_run, 'rows_in_region', side_effect=rows), \
                 patch.object(setup_run, 'classify_scene', return_value=SimpleNamespace(scene='unknown')), \
                 patch.object(setup_run, 'focused_tile', return_value=1), \
                 patch.object(setup_run, 'BackgroundController') as controller:
                result = setup_run.prepare_next(root/'game.exe', root=root, character_slot=1, weapon='SMG', record=False)
            self.assertIsNone(result['error'], result)
            self.assertTrue(result['context']['setup_complete'])
            self.assertEqual([call.args[0] for call in controller.return_value.tap_menu.call_args_list],
                             ['escape', 'escape', 'enter', 'enter'])
