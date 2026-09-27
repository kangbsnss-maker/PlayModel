"""Local Windows Chinese OCR for menu observations, outside the movement loop."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import queue
import subprocess
import threading
import time
import unicodedata


class MenuOcr:
    """One hidden WinRT worker; serialized requests never consume a late reply.

    Exact-image reuse is a cache, not policy learning. It is scoped to this
    reader, carries the original observation provenance, and expires quickly.
    """
    def __init__(self, script_path: Path, *, language='zh-Hans-CN', timeout=15.0,
                 cache_seconds=2.0):
        self.script_path = script_path
        self.language = language
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self.process = None
        self._lock = threading.Lock()
        self._replies = queue.Queue()
        self._reader = None
        self._sequence = 0
        self._cached = None

    def _start(self):
        self._replies = queue.Queue()
        from playmodel.execution_log import child_stderr_path, event
        self.stderr_path = child_stderr_path('windows_ocr.ps1')
        with self.stderr_path.open('ab') as errors:
            self.process = subprocess.Popen(
                ['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                 str(self.script_path.resolve()), '-Language', self.language, '-Server'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                text=True, encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        event('child_started', program='windows_ocr.ps1', child_pid=self.process.pid,
              stderr_path=str(self.stderr_path), language=self.language)
        process, replies = self.process, self._replies
        def receive():
            try:
                for line in process.stdout:
                    replies.put(line)
            finally:
                replies.put(None)
        self._reader = threading.Thread(target=receive, daemon=True, name='menu-ocr-replies')
        self._reader.start()

    def _terminate(self):
        process, self.process = self.process, None
        if process is not None:
            requested_termination = process.poll() is None
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            from playmodel.execution_log import event
            event('child_exited', program='windows_ocr.ps1', child_pid=process.pid,
                  exit_code=process.returncode, requested_termination=requested_termination,
                  stderr_path=str(self.stderr_path))
            if self._reader is not None:
                self._reader.join(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        self._cached = None

    def close(self):
        with self._lock:
            self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self, image_path: Path) -> dict:
        with self._lock:
            # Windows/Python versions can use different monotonic clock sources.
            # Capture and input provenance use QPC via perf_counter_ns throughout.
            started = time.perf_counter_ns()
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if self._cached is not None:
                old_digest, old_time, old_path, old_report = self._cached
                if digest == old_digest and time.monotonic() - old_time < self.cache_seconds:
                    return {**old_report, 'processing_started_at_ns': started,
                            'available_at_ns': time.perf_counter_ns(), 'recognition_path': 'exact_image_cache',
                            'cache_source': old_path, 'cache_source_sha256': old_digest,
                            'cache_source_available_at_ns': old_report['available_at_ns'],
                            'verified': False}
            if self.process is None or self.process.poll() is not None:
                self._terminate()
                self._start()
            self._sequence += 1
            request_id = self._sequence
            try:
                self.process.stdin.write(json.dumps({'id': request_id, 'path': str(image_path.resolve())}) + '\n')
                self.process.stdin.flush()
                line = self._replies.get(timeout=self.timeout)
                if line is None:
                    raise OSError('Local OCR worker exited')
                response = json.loads(line.lstrip('\ufeff'))
                if response.get('id') != request_id:
                    raise OSError('Local OCR response identity mismatch')
                if 'error' in response:
                    raise OSError('Local OCR failed: ' + response['error'])
                report = response['report']
                if not isinstance(report.get('lines'), list):
                    raise OSError('Malformed local OCR report')
            except queue.Empty as error:
                from playmodel.execution_log import event
                event('ocr_request_failed', error='Local OCR timed out', request_id=request_id,
                      frame_path=str(image_path), stderr_path=str(self.stderr_path))
                self._terminate()
                raise OSError('Local OCR timed out; worker terminated') from error
            except (OSError, ValueError, KeyError, TypeError) as error:
                from playmodel.execution_log import exception
                exception('ocr_request_failed', error)
                self._terminate()
                raise
            report.update(processing_started_at_ns=started, available_at_ns=time.perf_counter_ns(),
                          clock_domain='perf_counter_ns_same_host',
                          verified=False, recognition_path='persistent_local_ocr')
            self._cached = (digest, time.monotonic(), str(image_path.resolve()), dict(report))
            return report


def read_menu(image_path: Path, script_path: Path, *, language: str = 'zh-Hans-CN') -> dict:
    started = time.perf_counter_ns()
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script_path.resolve()),
             "-ImagePath", str(image_path.resolve()), "-Language", language],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise OSError("Local OCR timed out") from error
    if result.returncode:
        raise OSError(result.stderr.strip() or "Local OCR failed")
    report = json.loads(result.stdout.lstrip("\ufeff"))
    report.update(processing_started_at_ns=started, available_at_ns=time.perf_counter_ns(),
                  clock_domain='perf_counter_ns_same_host', verified=False)
    return report


def rows_in_region(report: dict, region: tuple[int, int, int, int]) -> list[str]:
    """Rejoin OCR words geometrically; keep raw OCR separately for review."""
    left, top, right, bottom = region
    words = [word for line in report["lines"] for word in line["words"]
             if left <= word["x"] + word["width"] / 2 <= right
             and top <= word["y"] + word["height"] / 2 <= bottom]
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda item: item["y"] + item["height"] / 2):
        center = word["y"] + word["height"] / 2
        if rows:
            previous = rows[-1]
            baseline = sum(w["y"] + w["height"] / 2 for w in previous) / len(previous)
            tolerance = max(8, min(word["height"], max(w["height"] for w in previous)) * 0.65)
            if abs(center - baseline) <= tolerance:
                previous.append(word)
                continue
        rows.append([word])
    return [unicodedata.normalize("NFKC", "".join(w["text"] for w in sorted(row, key=lambda w: w["x"]))) for row in rows]
