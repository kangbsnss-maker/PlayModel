"""One persistent hidden local worker. No HTTP, cloud API or input injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class LayaClient:
    def __init__(self, root: Path, output: Path, model_dir: Path, device='cuda', seed=0, checkpoint=None):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self._log = (self.output / 'worker.log').open('ab')
        python = self.root / '.venv-laya/Scripts/python.exe'
        command = [str(python), '-X', 'utf8', '-m', 'playmodel.laya.worker',
                   '--model', str(Path(model_dir).resolve()), '--output', str(self.output),
                   '--device', device, '--seed', str(seed)]
        if checkpoint:
            command.extend(['--checkpoint', str(Path(checkpoint).resolve())])
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0',
                   PYTHONPATH=str(self.root / 'src'))
        self.process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, encoding='utf8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self._responses = queue.Queue()
        self._sequence = 0
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            self.ready = self._receive(180)
            if self.ready.get('status') != 'ready':
                raise RuntimeError('Laya worker failed to initialize: ' + str(self.ready))
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self._responses.put(json.loads(line))
                except ValueError:
                    self._responses.put({'error': 'Invalid worker JSON'})
        finally:
            self._responses.put({'error': 'Laya worker exited; inspect worker.log'})

    def _receive(self, timeout):
        try:
            value = self._responses.get(timeout=timeout)
        except queue.Empty as error:
            self.close()
            raise TimeoutError('Local Laya worker deadline exceeded') from error
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def _request(self, method, **kwargs):
        with self._lock:
            self._sequence += 1
            request_id = self._sequence
            self.process.stdin.write(json.dumps({'id': request_id, 'method': method, **kwargs}, allow_nan=False) + '\n')
            self.process.stdin.flush()
            result = self._receive(180 if method == 'finish' else 30)
            if result.pop('request_id', None) != request_id:
                raise RuntimeError('Laya worker response identity mismatch')
            return result

    def choose(self, state: dict, options: dict[str, str], evidence: dict):
        return self._request('choose', state=state, options=options, evidence=evidence)

    def accept(self, decision_id: str, application: dict):
        return self._request('accept', decision_id=decision_id, application=application)

    def accept_tactic(self, decision_id: str, application: dict):
        return self._request('accept_tactic', decision_id=decision_id, application=application)

    def discard(self, decision_id: str, reason: str):
        return self._request('discard', decision_id=decision_id, reason=reason)

    def configure_preferences(self, preferences: dict):
        return self._request('configure_preferences', preferences=preferences)

    def finish(self, kind: str, evidence: dict):
        return self._request('finish', kind=kind, evidence=evidence)

    def abandon(self, reason: str):
        return self._request('abandon', reason=reason)

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=5)
        if getattr(self, '_log', None):
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
