import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from playmodel.games.brotato.menu_capture import MenuCapture
from playmodel.games.brotato.stream import Frame


class MenuCaptureTests(unittest.TestCase):
    def test_stale_frame_never_acknowledges_input_and_worker_reused(self):
        pixels = bytes(1920 * 1080 * 4)
        def frame(sequence, started):
            return Frame(sequence, {'sequence': sequence, 'width': 1920, 'height': 1080,
                                   'capture_started_at_ns': started}, pixels, started + 1)
        with tempfile.TemporaryDirectory() as folder, \
             patch('playmodel.games.brotato.menu_capture.CaptureStream') as factory, \
             patch('playmodel.games.brotato.menu_capture.time.perf_counter_ns', side_effect=[100, 200]), \
             patch('playmodel.games.brotato.menu_capture._png', return_value=b'png'), \
             patch('playmodel.games.brotato.menu_capture.time.sleep'):
            stream = factory.return_value
            stream.error = None
            stream.latest.side_effect = [frame(0, 99), frame(1, 101), frame(2, 110),
                                         frame(2, 110), frame(3, 201), frame(4, 210)]
            reader = MenuCapture(Path('Brotato.exe'))
            first, *_ = reader.read(Path(folder))
            second, *_ = reader.read(Path(folder))
            self.assertEqual((first['sequence'], second['sequence']), (2, 4))
            self.assertFalse(second['fresh_render_verified'])
            factory.assert_called_once()
            reader.close()
            stream.close.assert_called_once()

    def test_guard_runs_before_observation(self):
        with patch('playmodel.games.brotato.menu_capture.CaptureStream') as factory:
            reader = MenuCapture(Path('Brotato.exe'))
            def stop():
                raise OSError('Stopped')
            with self.assertRaisesRegex(OSError, 'Stopped'):
                reader.read(Path('unused'), check=stop)
            factory.return_value.latest.assert_not_called()
            reader.close()


if __name__ == '__main__':
    unittest.main()
