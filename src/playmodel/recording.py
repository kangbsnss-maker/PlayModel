"""Optional OBS scene signal. File I/O runs outside the movement loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time


class RecordingSignal:
    def __init__(self, path: Path, session_id: str, *, enabled: bool = False):
        self.path, self.session_id, self.enabled = path, session_id, enabled
        self.scene = "unknown"
        self.session_open = True
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._publish()
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True, name="obs-scene-signal")
            self._thread.start()

    def set_scene(self, scene: str):
        with self._lock:
            self.scene = scene if scene in ("combat", "level_up") else "excluded"

    def _publish(self):
        with self._lock:
            state = {"schema": "playmodel.obs-scene.v1", "enabled": self.enabled,
                     "session_id": self.session_id, "session_open": self.session_open,
                     "scene": self.scene, "updated_at": time.time(),
                     "allowed_scenes": ["combat", "level_up"]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, self.path)

    def _run(self):
        while not self._stop.wait(.1):
            try:
                self._publish()
            except OSError as error:
                self.error = str(error)  # OBS's freshness check pauses stale signals.

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1)
        with self._lock:
            self.session_open, self.scene = False, "excluded"
        self._publish()
