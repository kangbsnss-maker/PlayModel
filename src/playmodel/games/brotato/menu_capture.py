"""Fresh menu observations from one hidden worker, without per-key process startup."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from .capture import _png
from .stream import CaptureStream


class MenuCapture:
    def __init__(self, executable: Path, *, fps: float = 20):
        self.executable = executable
        if not 1 <= fps <= 60:
            raise ValueError('menu capture fps must be 1..60')
        self.fps = fps
        self.stream = None

    def read(self, output: Path, *, check=lambda: None, timeout=4):
        requested = time.perf_counter_ns()
        if self.stream is None:
            self.stream = CaptureStream(self.executable, fps=self.fps, stride=1)
        deadline = time.perf_counter() + timeout
        first_sequence = None
        while time.perf_counter() < deadline:
            check()
            if self.stream.error:
                raise OSError(self.stream.error)
            frame = self.stream.latest(max_age_ms=300)
            # A cached frame predating this observation request cannot acknowledge a key.
            if frame is not None and frame.metadata['capture_started_at_ns'] >= requested:
                # PrintWindow can return a rendering from the preceding request.
                # Keep a second post-request capture without spawning two workers
                # or imposing a fixed per-button sleep.
                if first_sequence is None:
                    first_sequence = frame.sequence
                elif frame.sequence > first_sequence:
                    break
            time.sleep(.005)
        else:
            self.close()
            raise OSError('Fresh menu capture timed out')
        metadata = dict(frame.metadata)
        width, height = metadata['width'], metadata['height']
        if (width, height) != (1920, 1080) or len(frame.pixels) != width * height * 4:
            raise OSError('Menu capture dimensions changed')
        directory = output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True)
        png = _png(width, height, frame.pixels, compression_level=1)
        (directory / 'frame.png').write_bytes(png)
        metadata.update(session_directory=str(directory.resolve()), session_id=directory.name,
                        frame='frame.png', frame_sha256=hashlib.sha256(png).hexdigest(),
                        available_at_ns=frame.available_at_ns, observation_requested_at_ns=requested,
                        first_post_request_sequence=first_sequence,
                        capture_path='persistent_menu_worker', usable_for_training=False,
                        fresh_render_verified=False)
        (directory / 'observation.json').write_text(json.dumps(metadata), encoding='utf-8')
        return metadata, frame.pixels, width, height

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
