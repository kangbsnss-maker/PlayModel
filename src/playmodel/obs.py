"""Local OBS WebSocket v5 control; credentials never leave loopback or enter logs."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


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
