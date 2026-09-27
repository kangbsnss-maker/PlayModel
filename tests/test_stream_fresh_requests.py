"""Capture cadence/transport tests with a fake hidden child; no game capture/input."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from playmodel.games.brotato.stream import CaptureStream, _FreshCaptureRequests


class FreshCaptureTests(unittest.TestCase):
    def test_requests_coalesce_and_never_regress(self):
        requests = _FreshCaptureRequests()
        self.assertTrue(requests.publish(10))
        self.assertTrue(requests.publish(30))
        self.assertFalse(requests.publish(20))
        self.assertFalse(requests.publish(30))
        self.assertEqual(requests.next(-1), 30)
        for invalid in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                requests.publish(invalid)

    def test_actual_send_after_capture_wakes_remaining_cadence(self):
        requests = _FreshCaptureRequests()
        entered, finished = threading.Event(), threading.Event()
        def wait():
            entered.set()
            requests.wait(100, 1)
            finished.set()
        thread = threading.Thread(target=wait)
        thread.start()
        self.assertTrue(entered.wait(1))
        requests.publish(99)
        self.assertFalse(finished.wait(.03), 'a newer existing capture already satisfies this request')
        requests.publish(100)
        self.assertTrue(finished.wait(.5))
        thread.join(1)

    def test_shutdown_wakes_empty_request_writer(self):
        requests, values = _FreshCaptureRequests(), []
        thread = threading.Thread(target=lambda: values.append(requests.next(-1)))
        thread.start()
        requests.close()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(values, [None])
        self.assertFalse(requests.publish(10))

    def test_hidden_child_receives_request_and_returns_only_new_frame(self):
        fake = '''import sys,json,struct,time
sequence=0
while True:
    after=-1
    if sequence:
        data=sys.stdin.buffer.read(8)
        if not data: break
        after=struct.unpack('<Q',data)[0]
    at=time.perf_counter_ns()
    meta=json.dumps({'sequence':sequence,'capture_started_at_ns':at,'capture_finished_at_ns':at,
                     'sample_width':1,'sample_height':1,'request_after_ns':after}).encode()
    pixels=bytes([1,2,3,0])
    sys.stdout.buffer.write(struct.pack('<II',len(meta),len(pixels))+meta+pixels)
    sys.stdout.buffer.flush()
    sequence+=1
'''
        original = subprocess.Popen
        def spawn(args, **kwargs):
            self.assertEqual(kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            return original([sys.executable, '-u', '-c', fake], **kwargs)
        with patch('playmodel.games.brotato.stream.subprocess.Popen', side_effect=spawn):
            stream = CaptureStream(Path('unused.exe'))
            try:
                until = time.monotonic() + 2
                while stream.latest(max_age_ms=2000) is None and time.monotonic() < until:
                    time.sleep(.005)
                self.assertEqual(stream.latest(max_age_ms=2000).sequence, 0)
                sent = time.perf_counter_ns()
                self.assertTrue(stream.request_fresh(sent))
                until = time.monotonic() + 2
                while stream.frames_received < 2 and time.monotonic() < until:
                    time.sleep(.005)
                frame = stream.latest(max_age_ms=2000)
                self.assertEqual(frame.sequence, 1)
                self.assertEqual(frame.metadata['request_after_ns'], sent)
                self.assertGreater(frame.metadata['capture_started_at_ns'], sent)
                self.assertEqual(stream.fresh_requests_sent, 1)
            finally:
                stream.close()
            self.assertFalse(stream._request_writer.is_alive())


if __name__ == '__main__':
    unittest.main()
