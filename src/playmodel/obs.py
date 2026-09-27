"""Local OBS WebSocket v5 control; credentials never leave loopback or enter logs."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def ensure_obs_running(*, timeout=20):
    """Start installed OBS only when absent; verify its local control endpoint."""
    import csv
    import io
    import shutil
    import subprocess
    from .execution_log import event

    try:
        with ObsClient() as client:
            client.call('GetVersion')
        return
    except ConnectionError:
        pass
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    probe = subprocess.run(['tasklist.exe', '/FI', 'IMAGENAME eq obs64.exe', '/FO', 'CSV', '/NH'],
                           capture_output=True, text=True, creationflags=flags, timeout=5, check=True)
    running = any(row and row[0].lower() == 'obs64.exe' for row in csv.reader(io.StringIO(probe.stdout)))
    if not running:
        candidates = [Path(shutil.which('obs64.exe') or '__missing_obs__')]
        candidates += [Path(f'{drive}:/Program Files/obs-studio/bin/64bit/obs64.exe')
                       for drive in 'CDEFGHIJKLMNOPQRSTUVWXYZ']
        executable = next((path.resolve() for path in candidates if path.is_file()), None)
        if executable is None:
            raise OSError('Installed OBS executable not found')
        child = subprocess.Popen([str(executable), '--disable-shutdown-check'], cwd=executable.parent,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, creationflags=flags)
        event('obs_started', child_pid=child.pid, executable=str(executable))
    deadline = time.monotonic() + timeout
    while True:
        try:
            with ObsClient() as client:
                client.call('GetVersion')
            event('obs_ready', started=not running)
            return
        except ConnectionError:
            if time.monotonic() >= deadline:
                raise OSError('OBS local control endpoint did not become ready')
            time.sleep(.25)


class ObsClient:
    def __init__(self, config_path: Path | None = None):
        import websocket
        path = config_path or Path(os.environ['APPDATA']) / 'obs-studio/plugin_config/obs-websocket/config.json'
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        if not config.get('server_enabled'):
            raise OSError('OBS WebSocket server disabled')
        port = int(config.get('server_port', 4455))
        if not 1 <= port <= 65535:
            raise ValueError('Invalid local OBS port')
        self.socket = websocket.create_connection(f'ws://127.0.0.1:{port}', timeout=4,
                                                  http_no_proxy=['127.0.0.1', 'localhost'])
        try:
            hello = json.loads(self.socket.recv())
            if hello.get('op') != 0:
                raise OSError('OBS Hello missing')
            identify = {'rpcVersion': 1, 'eventSubscriptions': 0}
            auth = hello['d'].get('authentication')
            if auth:
                secret = base64.b64encode(hashlib.sha256((config['server_password'] + auth['salt']).encode()).digest()).decode()
                identify['authentication'] = base64.b64encode(hashlib.sha256((secret + auth['challenge']).encode()).digest()).decode()
            self.socket.send(json.dumps({'op': 1, 'd': identify}))
            if json.loads(self.socket.recv()).get('op') != 2:
                raise OSError('OBS identification failed')
        except Exception:
            self.socket.close()
            raise

    def call(self, request: str, **data) -> dict:
        request_id = uuid.uuid4().hex
        self.socket.send(json.dumps({'op': 6, 'd': {'requestType': request, 'requestId': request_id,
                                                   'requestData': data}}))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = json.loads(self.socket.recv())
            if message.get('op') != 7 or message['d'].get('requestId') != request_id:
                continue
            payload = message['d']
            if not payload['requestStatus']['result']:
                raise OSError(f"OBS {request} failed (code {payload['requestStatus']['code']})")
            return payload.get('responseData', {})
        raise OSError(f'OBS {request} timed out')

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
