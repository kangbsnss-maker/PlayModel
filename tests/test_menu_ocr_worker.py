"""Worker protocol tests, independent of Windows OCR and game input."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from playmodel.games.brotato.ocr import MenuOcr


FAKE = '''import sys,json,time
print("worker stderr evidence",file=sys.stderr,flush=True)
for line in sys.stdin:
    req=json.loads(line)
    if "slow" in req["path"]: time.sleep(2)
    if "badid" in req["path"]: req["id"]+=1
    print(json.dumps({"id":req["id"],"report":{"lines":[],"text":req["path"]}}),flush=True)
'''


class MenuOcrWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame = self.root / 'frame.png'
        self.frame.write_bytes(b'frame-one')
        self.original_popen = subprocess.Popen
        self.starts = []
        def start(args, **kwargs):
            self.starts.append(args)
            return self.original_popen([sys.executable, '-u', '-c', FAKE], **kwargs)
        self.patch = patch('playmodel.games.brotato.ocr.subprocess.Popen', side_effect=start)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_multiple_images_share_process_and_close(self):
        with MenuOcr(self.root / 'script.ps1', cache_seconds=0) as worker:
            first = worker.read(self.frame)
            second = worker.read(self.frame)
            process = worker.process
            errors = worker.stderr_path
            self.assertEqual(len(self.starts), 1)
            self.assertEqual(first['recognition_path'], 'persistent_local_ocr')
            self.assertFalse(second['verified'])
        self.assertIsNotNone(process.poll())
        self.assertIn('worker stderr evidence', errors.read_text(encoding='utf-8'))

    def test_cache_requires_exact_bytes_and_expires(self):
        with MenuOcr(self.root / 'script.ps1') as worker:
            original = worker.read(self.frame)
            hit = worker.read(self.frame)
            self.assertEqual(hit['recognition_path'], 'exact_image_cache')
            self.assertEqual(hit['cache_source_available_at_ns'], original['available_at_ns'])
            self.frame.write_bytes(b'changed-price-or-dialog')
            self.assertEqual(worker.read(self.frame)['recognition_path'], 'persistent_local_ocr')
            worker.cache_seconds = 0
            self.assertEqual(worker.read(self.frame)['recognition_path'], 'persistent_local_ocr')

    def test_evidence_uses_capture_clock_even_when_monotonic_epoch_differs(self):
        with MenuOcr(self.root / 'script.ps1') as worker:
            with patch('playmodel.games.brotato.ocr.time.perf_counter_ns', side_effect=[1000, 2000, 3000, 4000]), \
                 patch('playmodel.games.brotato.ocr.time.monotonic_ns', return_value=1):
                result = worker.read(self.frame)
                cached = worker.read(self.frame)
            self.assertEqual((result['processing_started_at_ns'], result['available_at_ns']), (1000, 2000))
            self.assertEqual((cached['processing_started_at_ns'], cached['available_at_ns']), (3000, 4000))
            self.assertEqual(result['clock_domain'], 'perf_counter_ns_same_host')

    def test_timed_out_reply_cannot_reach_next_request(self):
        slow = self.root / 'slow.png'
        slow.write_bytes(b'slow')
        with MenuOcr(self.root / 'script.ps1', timeout=.5) as worker:
            with self.assertRaisesRegex(OSError, 'timed out'):
                worker.read(slow)
            self.assertIsNone(worker.process)
            result = worker.read(self.frame)
            self.assertEqual(result['text'], str(self.frame.resolve()))
            self.assertEqual(len(self.starts), 2)

    def test_wrong_request_id_closes_worker(self):
        frame = self.root / 'badid.png'
        frame.write_bytes(b'bad')
        with MenuOcr(self.root / 'script.ps1') as worker:
            with self.assertRaisesRegex(OSError, 'identity mismatch'):
                worker.read(frame)
            self.assertIsNone(worker.process)

    def test_inflight_terminal_and_next_menu_share_serialized_reader_without_reply_mixup(self):
        slow = self.root / 'slow-terminal.png'
        slow.write_bytes(b'terminal')
        with MenuOcr(self.root / 'script.ps1', cache_seconds=0) as worker:
            worker.read(self.frame)  # Session menu warms the one child.
            completed, replies = threading.Event(), {}
            terminal = threading.Thread(target=lambda: replies.update(terminal=worker.read(slow)))
            terminal.start()
            until = time.monotonic() + 1
            while worker._sequence < 2 and time.monotonic() < until:
                time.sleep(.005)
            self.assertEqual(worker._sequence, 2)

            def next_menu():
                replies['menu'] = worker.read(self.frame)
                completed.set()
            menu = threading.Thread(target=next_menu)
            menu.start()
            self.assertFalse(completed.wait(.05), 'menu must not consume the in-flight terminal reply')
            terminal.join(4)
            menu.join(4)
            self.assertFalse(terminal.is_alive())
            self.assertFalse(menu.is_alive())
            self.assertEqual(replies['terminal']['text'], str(slow.resolve()))
            self.assertEqual(replies['menu']['text'], str(self.frame.resolve()))
            self.assertEqual(len(self.starts), 1)


if __name__ == '__main__':
    unittest.main()
