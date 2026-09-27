import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from playmodel.games.brotato.pilot import LocalTerminalObserver, SlowTerminalWorker


class TerminalWorkerReuseTests(unittest.TestCase):
    def test_two_independent_observations_reuse_capture_and_uncached_ocr(self):
        rules = Mock(version='fixture')
        rules.classify.return_value = 'wave_clear'
        with patch('playmodel.games.brotato.menu_capture.MenuCapture') as capture, \
             patch('playmodel.games.brotato.ocr.MenuOcr') as ocr, \
             patch('playmodel.games.brotato.pilot._json'):
            observer = LocalTerminalObserver(Path('Brotato.exe'), Path('temp'), Path('ocr.ps1'), rules)
            capture.return_value.read.side_effect = [
                ({'session_directory': 'frame1', 'width': 1920, 'height': 1080,
                  'capture_started_at_ns': 100_000_000, 'frame_sha256': 'a'*64}, None, 1920, 1080),
                ({'session_directory': 'frame2', 'width': 1920, 'height': 1080,
                  'capture_started_at_ns': 400_000_000, 'frame_sha256': 'b'*64}, None, 1920, 1080)]
            ocr.return_value.read.return_value = {'text': 'Shop Next', 'lines': []}
            self.assertIsNone(observer())
            evidence = observer()
            self.assertEqual(evidence.kind, 'wave_clear')
            self.assertEqual(evidence.frame_sha256, 'b'*64)
            capture.assert_called_once_with(Path('Brotato.exe'), fps=4)
            ocr.assert_called_once_with(Path('ocr.ps1'), cache_seconds=0)
            self.assertEqual(ocr.return_value.read.call_count, 2)
            observer.close()
            capture.return_value.close.assert_called_once()
            ocr.return_value.close.assert_called_once()

    def test_worker_closes_resources_on_its_own_thread_after_inflight_call(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        class Observer:
            def __call__(self):
                entered.set()
                release.wait(1)
            def close(self):
                closed.set()
        worker = SlowTerminalWorker(Observer())
        self.assertTrue(entered.wait(1))
        self.assertFalse(worker.close())
        self.assertFalse(closed.is_set())
        release.set()
        self.assertTrue(closed.wait(1))
        worker.thread.join(1)
        self.assertFalse(worker.thread.is_alive())

    def test_borrowed_uncached_reader_is_reused_and_never_closed_by_observer(self):
        reader = Mock(cache_seconds=0)
        rules = Mock(version='fixture')
        rules.classify.return_value = None
        with patch('playmodel.games.brotato.menu_capture.MenuCapture') as capture, \
             patch('playmodel.games.brotato.ocr.MenuOcr') as create_reader, \
             patch('playmodel.games.brotato.pilot._json'):
            capture.return_value.read.return_value = ({'session_directory': 'frame1', 'width': 1920,
                'height': 1080, 'capture_started_at_ns': 100_000_000, 'frame_sha256': 'a'*64},
                None, 1920, 1080)
            reader.read.return_value = {'text': 'combat', 'lines': []}
            observer = LocalTerminalObserver(Path('game.exe'), Path('temp'), Path('ocr.ps1'), rules,
                                             reader=reader)
            self.assertIsNone(observer())
            reader.read.assert_called_once_with(Path('frame1/frame.png'))
            create_reader.assert_not_called()
            observer.close()
            reader.close.assert_not_called()
            capture.return_value.close.assert_called_once()

    def test_cached_borrowed_reader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'uncached'):
            LocalTerminalObserver(Path('game.exe'), Path('temp'), Path('ocr.ps1'), Mock(),
                                  reader=Mock(cache_seconds=2.0))


if __name__ == '__main__':
    unittest.main()
