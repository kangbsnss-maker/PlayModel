"""Persistent hidden capture worker; latest frame only, no OCR in the fast path."""
from __future__ import annotations

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module('playmodel.games.brotato.stream', capture_stdout=False)


import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

from .capture import _capture_worker


@dataclass(frozen=True)
class Frame:
    sequence: int
    metadata: dict
    pixels: bytes
    available_at_ns: int


class _FreshCaptureRequests:
    """One monotonic request, never a backlog or a second capture worker."""
    def __init__(self):
        self.condition = threading.Condition()
        self.latest_after_ns = -1
        self.closed = False

    def publish(self, after_ns):
        if type(after_ns) is not int or after_ns < 0:
            raise ValueError('fresh capture request requires a nonnegative timestamp')
        with self.condition:
            if self.closed or after_ns <= self.latest_after_ns:
                return False
            self.latest_after_ns = after_ns
            self.condition.notify_all()
            return True

    def next(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.latest_after_ns > previous)
            return None if self.closed else self.latest_after_ns

    def wait(self, captured_at_ns, seconds):
        # If capture already started after the actual send, let its ordinary
        # cadence stand. Otherwise wake only its remaining inter-frame wait.
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.latest_after_ns >= captured_at_ns,
                                    timeout=max(0, seconds))

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class CaptureStream:
    def __init__(self, executable: Path, *, fps: float = 20, stride: int = 6, backend: str = "printwindow"):
        self._lock = threading.Lock()
        self._latest = None
        self.error = None
        self.frames_received = 0
        self._requests = _FreshCaptureRequests()
        self.fresh_requests_sent = 0
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "playmodel.games.brotato.stream", "--exe", str(executable),
             "--fps", str(fps), "--stride", str(stride), "--backend", backend],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._reader = threading.Thread(target=self._read, daemon=True, name="brotato-latest-frame")
        self._reader.start()
        self._request_writer = threading.Thread(target=self._write_requests, daemon=True,
                                                name='brotato-capture-requests')
        self._request_writer.start()
        from playmodel.execution_log import event
        event('child_started', program='playmodel.games.brotato.stream', child_pid=self.process.pid)

    def request_fresh(self, after_ns):
        """Nonblocking in-memory request after actual input; no pipe I/O here."""
        return self._requests.publish(after_ns)

    def _write_requests(self):
        previous = -1
        try:
            while (after_ns := self._requests.next(previous)) is not None:
                self.process.stdin.write(struct.pack('<Q', after_ns))
                self.process.stdin.flush()
                self.fresh_requests_sent += 1
                previous = after_ns
        except (OSError, ValueError):
            if not self._requests.closed:
                self.error = 'Fresh capture request transport failed'

    def _read_exact(self, length):
        data = bytearray()
        while len(data) < length:
            part = self.process.stdout.read(length - len(data))
            if not part:
                raise EOFError("Capture worker ended")
            data.extend(part)
        return bytes(data)

    def _read(self):
        try:
            while True:
                meta_size, data_size = struct.unpack("<II", self._read_exact(8))
                if not 1 <= meta_size <= 32768 or not 1 <= data_size <= 7680 * 4320 * 4:
                    raise ValueError("Invalid frame message length")
                metadata = json.loads(self._read_exact(meta_size))
                pixels = self._read_exact(data_size)
                frame = Frame(metadata["sequence"], metadata, pixels, time.perf_counter_ns())
                with self._lock:
                    self._latest = frame
                    self.frames_received += 1
        except (EOFError, OSError, ValueError) as error:
            detail = self.process.stderr.read(16384).decode("utf-8", errors="replace") if isinstance(error, EOFError) else ""
            self.error = str(error) + (": " + detail.strip() if detail else "")

    def latest(self, *, max_age_ms: float = 150) -> Frame | None:
        with self._lock:
            frame = self._latest
        if frame is None or time.perf_counter_ns() - frame.metadata["capture_started_at_ns"] > max_age_ms * 1_000_000:
            return None
        return frame

    def close(self):
        self._requests.close()
        requested_termination = self.process.poll() is None
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self._reader.join(timeout=2)
        self._request_writer.join(timeout=2)
        from playmodel.execution_log import event
        event('child_exited', program='playmodel.games.brotato.stream', child_pid=self.process.pid,
              exit_code=self.process.returncode, requested_termination=requested_termination,
              frames_received=self.frames_received, fresh_requests_sent=self.fresh_requests_sent,
              error=self.error if not requested_termination else None)
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.stdin.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--backend", choices=("printwindow", "visible_client"), default="printwindow")
    args = parser.parse_args()
    if not 1 <= args.fps <= 60 or not 1 <= args.stride <= 32:
        raise ValueError("Unsupported capture rate or sample stride")
    sequence = 0
    requests = _FreshCaptureRequests()

    def receive_requests():
        while True:
            data = sys.stdin.buffer.read(8)
            if not data:
                return
            if len(data) != 8:
                return
            after_ns, = struct.unpack('<Q', data)
            if after_ns > time.perf_counter_ns():
                return  # No future timestamp may accelerate this local clock.
            requests.publish(after_ns)

    threading.Thread(target=receive_requests, daemon=True, name='capture-cadence-requests').start()
    preceding_wait_ns = 0
    burst_period = 1 / min(60, args.fps * 2)
    while True:
        tick = time.perf_counter()
        frame = _capture_worker(args.exe, pixels_only=True, sample_stride=args.stride, backend=args.backend)
        frame['worker_capture_returned_at_ns'] = time.perf_counter_ns()
        frame['preceding_cadence_wait_ns'] = preceding_wait_ns
        frame['latest_fresh_request_after_ns'] = requests.latest_after_ns
        frame['capture_nominal_fps'] = args.fps
        frame['capture_burst_fps_limit'] = min(60, args.fps * 2)
        pixels = frame.pop("pixels")
        frame["sequence"] = sequence
        metadata = json.dumps(frame, ensure_ascii=True).encode("utf-8")
        sys.stdout.buffer.write(struct.pack("<II", len(metadata), len(pixels)) + metadata + pixels)
        sys.stdout.buffer.flush()
        sequence += 1
        wait_started = time.perf_counter_ns()
        # A request cannot turn the worker into a free-running capture loop.
        time.sleep(max(0, burst_period - (time.perf_counter() - tick)))
        requests.wait(frame['capture_started_at_ns'], 1 / args.fps - (time.perf_counter() - tick))
        preceding_wait_ns = time.perf_counter_ns() - wait_started


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
