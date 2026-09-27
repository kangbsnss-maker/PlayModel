"""Loopback-only Laya evidence dashboard and audited human tactical preferences."""
from __future__ import annotations

if __name__ == '__main__':
    from _execution_bootstrap import launch
    launch(__file__)

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit
import uuid

from playmodel.atomic_io import atomic_json
from playmodel.instance import session_lock
from playmodel.laya.preferences import (SCHEMA, TACTICAL_ACTIONS, load_preferences,
                                       preference_hash, validate_preferences)

SERVICE = 'playmodel-laya-dashboard'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Dashboard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.artifacts = self.root / 'artifacts/laya-learning'
        self.preference_path = self.root / 'configs/local/laya-preferences.json'
        self.journal = self.root / 'artifacts/laya-human-feedback'
        self.csrf = secrets.token_urlsafe(32)
        self.nonce = secrets.token_urlsafe(24)
        self.cache = {}
        self.hash_cache = {}
        self.snapshot = None
        self.snapshot_time = 0

    def _read(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Evidence path outside workspace')
        stat = path.stat()
        if stat.st_size > 8_000_000:
            raise ValueError('Evidence JSON exceeds dashboard read bound')
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(str(path))
        if cached and cached[0] == key:
            return cached[1], cached[2]
        raw = path.read_bytes()
        value, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
        self.cache[str(path)] = (key, value, digest)
        return value, digest

    def _proof_digest(self,path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.artifacts) or path.suffix.lower() not in ('.json','.jsonl','.bgra','.png'):
            raise ValueError('Unsupported training evidence file')
        if path.suffix.lower()=='.json':
            return self._read(path)[1]
        stat = path.stat()
        if stat.st_size>32_000_000:
            raise ValueError('Evidence frame exceeds dashboard read bound')
        key = (str(path),stat.st_mtime_ns,stat.st_size)
        cached = self.hash_cache.get(str(path))
        if cached and cached[0]==key:
            return cached[1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.hash_cache[str(path)] = (key,digest)
        return digest

    def _relative(self, path):
        return str(Path(path).resolve().relative_to(self.root)).replace('\\', '/')

    def status(self, *, refresh=False):
        if not refresh and self.snapshot is not None and time.monotonic()-self.snapshot_time < 2:
            return self.snapshot
        errors, sessions, decisions, reports, updates, abandoned, applied = [], [], {}, {}, [], set(), []

        def read(path):
            try:
                return self._read(path)[0]
            except (OSError, ValueError, TypeError) as error:
                if len(errors) < 20:
                    errors.append({'source': self._relative(path), 'error': str(error)})
                return None

        for directory in sorted(self.artifacts.glob('laya-*')):
            if not directory.is_dir():
                continue
            status_path = directory / 'status.json'
            if status_path.exists():
                current = read(status_path)
                if current:
                    sessions.append({**current, 'session': directory.name,
                                     'source': self._relative(status_path), '_mtime': status_path.stat().st_mtime_ns})
            for path in directory.glob('*/cycle-run.json'):
                value = read(path)
                if value and value.get('run_id'):
                    reports[value['run_id']] = value
            laya = directory / 'laya'
            for path in laya.glob('abandoned-*.json'):
                value = read(path)
                if value:
                    abandoned.update(value.get('accepted', []))
                    abandoned.update(value.get('pending', []))
            for path in laya.glob('choice-*.json'):
                value = read(path)
                if value and value.get('decision_id'):
                    decisions[value['decision_id']] = {**value, '_path': path,
                        'source': self._relative(path), '_mtime': path.stat().st_mtime_ns}
            for path in laya.glob('preferences-applied-*.json'):
                value = read(path)
                if value:
                    applied.append({**value, '_mtime': path.stat().st_mtime_ns})
            for path in laya.glob('update-*/report.json'):
                value = read(path)
                if value:
                    updates.append({**value, '_path': path, 'source': self._relative(path)})

        trained, verified_updates, seen_versions = set(), [], set()
        for update in sorted(updates, key=lambda row: row.get('created_utc', '')):
            eligible = update.get('accepted') is True and update.get('status') == 'updated'
            ids = update.get('decisions', [])
            manifest = update['_path'].parent / 'dataset-manifest.json'
            try:
                if not eligible:
                    continue
                version = update.get('behavior_version')
                if not version or version in seen_versions:
                    continue
                if (type(update.get('optimizer_steps')) is not int or update['optimizer_steps']<=0
                        or not update.get('head_hash_before') or not update.get('head_hash_after')
                        or update['head_hash_before']==update['head_hash_after']
                        or not update.get('encoder_hash_before')
                        or update['encoder_hash_before']!=update.get('encoder_hash_after')
                        or not isinstance(update.get('kl_per_choice'),list)
                        or len(update['kl_per_choice'])!=len(ids)
                        or any(type(k) not in (float,int) or not math.isfinite(k) or not 0<=k<=.03
                               for k in update['kl_per_choice'])):
                    raise ValueError('Optimizer/head/encoder/KL evidence is incomplete')
                outcome = update.get('verified_outcome') or {}
                if (outcome.get('origin')!='local_detector' or outcome.get('verified') is not True
                        or outcome.get('independent_of_policy') is not True
                        or outcome.get('kind') not in ('death','wave_clear')):
                    raise ValueError('No independent local game outcome')
                outcome_binding = update.get('outcome') or {}
                terminal_path = Path(outcome_binding.get('terminal_path','')).resolve()
                if not terminal_path.is_relative_to(self.artifacts):
                    raise ValueError('Terminal proof outside Laya artifacts')
                terminal, terminal_sha = self._read(terminal_path)
                if (terminal_sha!=outcome_binding.get('terminal_sha256') or terminal!=outcome
                        or outcome_binding.get('kind')!=outcome['kind']):
                    raise ValueError('Verified outcome does not match immutable terminal evidence')
                data, manifest_sha = self._read(manifest)
                if (manifest_sha != update.get('dataset_manifest_sha256') or data.get('split') != 'train'
                        or data.get('source_version') != update.get('source_version')
                        or not isinstance(ids, list) or not ids):
                    raise ValueError('Update dataset manifest binding mismatch')
                proof_map = {}
                for proof in data.get('files', []):
                    proof_path = Path(proof['path']).resolve()
                    if not proof_path.is_relative_to(self.artifacts):
                        raise ValueError('Training proof outside Laya artifacts')
                    # Immutable frame proofs are hashed once per file change.
                    # Model tensors/checkpoints are never opened by this service.
                    actual = self._proof_digest(proof_path)
                    if actual != proof['sha256']:
                        raise ValueError('Training proof digest mismatch')
                    proof_map[proof_path] = actual
                valid_ids = []
                for decision_id in ids:
                    decision = decisions.get(decision_id)
                    if decision is None or decision_id in abandoned:
                        raise ValueError('Training decision missing or abandoned')
                    choice_path = decision['_path'].resolve()
                    acceptance_path = choice_path.with_name(f'accepted-{decision_id}.json')
                    if choice_path not in proof_map or acceptance_path not in proof_map:
                        raise ValueError('Update lacks decision/acceptance proof pair')
                    if decision.get('behavior_version') != update.get('source_version'):
                        raise ValueError('Update source version differs from decision')
                    valid_ids.append(decision_id)
                trained.update(valid_ids)
                seen_versions.add(version)
                verified_updates.append({
                    key: update.get(key) for key in ('source_version','behavior_version','created_utc','method',
                        'head_hash_before','head_hash_after','loss','gradient_norm','kl_per_choice','reward','source',
                        'parameter_count','trainable_parameter_count','head_l2_norm','head_delta_l2_norm')
                } | {'decisions':len(valid_ids),'optimizer_steps':update.get('optimizer_steps',0)})
            except (OSError, ValueError, KeyError, TypeError) as error:
                if len(errors)<20:
                    errors.append({'source': self._relative(manifest), 'error': str(error)})

        ordered = sorted(decisions.values(), key=lambda row:row['_mtime'])
        recent = []
        for decision in ordered[-100:]:
            decision_id = decision['decision_id']
            acceptance = decision['_path'].with_name(f'accepted-{decision_id}.json')
            discarded = decision['_path'].with_name(f'discarded-{decision_id}.json')
            state = ('trained' if decision_id in trained else 'abandoned' if decision_id in abandoned else
                     'accepted' if acceptance.exists() else 'discarded' if discarded.exists() else 'pending')
            recent.append({key:decision.get(key) for key in ('decision_id','decision_domain','action_id','state',
                'options','distribution','raw_head_distribution','applied_bias','behavior_version',
                'preferences','preference_hash','inference_ms','source')} | {'evidence_status':state})
        current = max(sessions, key=lambda row: row['_mtime']) if sessions else None
        if current:
            current = {key:value for key,value in current.items() if key != '_mtime'}
            heartbeat = current.get('heartbeat_utc')
            try:
                age = (datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat)).total_seconds()
                current['heartbeat_fresh'] = 0 <= age < 30
            except (TypeError, ValueError):
                current['heartbeat_fresh'] = False
            current['wave'] = None  # No verified wave number in this status contract.
        try:
            preferences = load_preferences(self.preference_path)
            preferences_error = None
        except (OSError, ValueError) as error:
            preferences, preferences_error = None, str(error)
        latest_applied = max(applied,key=lambda row:row['_mtime']) if applied else None
        if latest_applied:
            latest_applied = {key:value for key,value in latest_applied.items() if key != '_mtime'}
        learning_runs = [row for row in reports.values() if row.get('schema')=='playmodel.laya-run.v1'
                         and row.get('split')=='train' and row.get('menu_choice_backend')=='local_laya'
                         and not row.get('recovery_only')]
        counts = {'runs_completed':sum(row.get('full_run_complete') is True for row in learning_runs),
            'runs_recorded':len(learning_runs),
            'menu_decisions':sum(row.get('decision_domain','menu')!='combat_tactic' for row in decisions.values()),
            'tactic_decisions':sum(row.get('decision_domain')=='combat_tactic' for row in decisions.values()),
            'trained_decisions':len(trained), 'accepted_updates':len(verified_updates),
            'optimizer_steps':sum(row['optimizer_steps'] for row in verified_updates)}
        self.snapshot = {'schema':'playmodel.laya-dashboard.v1','generated_utc':utc_now(),
            'csrf_token':self.csrf, 'counts':counts,'current':current,'updates':verified_updates[-80:],
            'decisions':recent,'preferences':preferences,'preferences_error':preferences_error,
            'applied_preferences':latest_applied,'errors':errors,
            'limitations':['현재 웨이브 번호는 확인된 자료가 없습니다.',
                '선택 확률은 서로 다른 관측에서 나온 기록이며 실력 향상 비교가 아닙니다.',
                '신경망 개별 가중치값은 표시하지 않습니다. 검증된 업데이트 해시·손실·KL을 표시합니다.']}
        self.snapshot_time = time.monotonic()
        return self.snapshot

    def save_preferences(self, request):
        if (not isinstance(request,dict) or not {'revision','values','reason'} <= set(request)
                or set(request)-{'revision','values','reason','related_decision_id'}):
            raise ValueError('revision, values, reason만 필요합니다.')
        current = load_preferences(self.preference_path)
        if type(request['revision']) is not int or request['revision'] != current['revision']:
            raise Conflict('다른 변경이 먼저 저장되었습니다. 최신 값을 새로 불러오세요.')
        if not isinstance(request['reason'],str) or not request['reason'].strip():
            raise ValueError('변경 사유를 입력하세요.')
        related = request.get('related_decision_id')
        if related is not None:
            if not isinstance(related,str) or len(related)>128 or not related.isalnum():
                raise ValueError('판단 식별자가 올바르지 않습니다.')
            matches = list(self.artifacts.glob(f'laya-*/laya/choice-{related}.json'))
            if not matches or not any(self._read(path)[0].get('decision_id')==related for path in matches):
                raise ValueError('관련 판단 기록을 찾을 수 없습니다.')
        next_value = validate_preferences({'schema':SCHEMA,'revision':current['revision']+1,
            'values':request['values'],'reason':request['reason'].strip(),'updated_utc':utc_now()})
        self.journal.mkdir(parents=True,exist_ok=True)
        identifier = uuid.uuid4().hex
        event = {'schema':'playmodel.laya-human-feedback.v1','id':identifier,'created_utc':utc_now(),
            'status':'requested','before':current,'after':next_value,'after_sha256':preference_hash(next_value),
            'meaning':'explicit_human_tactical_logit_bias','reward_source':False,'bc_label':False,
            'related_decision_id':related}
        with (self.journal/f'request-{identifier}.json').open('x',encoding='utf8') as stream:
            json.dump(event,stream,ensure_ascii=False,allow_nan=False,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.preference_path,next_value,durable=True)
        with (self.journal/f'committed-{identifier}.json').open('x',encoding='utf8') as stream:
            json.dump({'id':identifier,'revision':next_value['revision'],'committed_utc':utc_now(),
                       'preferences_sha256':preference_hash(next_value)},stream,ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.snapshot = None
        return {'status':'saved','preferences':next_value,'application':'next_safe_boundary'}


class Handler(BaseHTTPRequestHandler):
    server_version = SERVICE
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self,*args):
        pass

    @property
    def app(self):
        return self.server.app

    def _trusted_host(self):
        return self.headers.get_all('Host') == [f'127.0.0.1:{self.server.server_port}']

    def _respond(self,status,value,*,html=False):
        body = value.encode('utf8') if html else json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',f"default-src 'none'; script-src 'nonce-{self.app.nonce}'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._trusted_host():
            return self._respond(403,{'error':'허용되지 않은 Host'})
        path = urlsplit(self.path).path
        if path == '/api/health':
            return self._respond(200,{'service':SERVICE,'pid':os.getpid()})
        if path == '/':
            page = (self.app.root/'docs/guides/laya-dashboard.html').read_text(encoding='utf8')
            return self._respond(200,page.replace('__CSP_NONCE__',self.app.nonce),html=True)
        if path == '/api/status':
            try:
                return self._respond(200,self.app.status())
            except (OSError,ValueError,TypeError,KeyError) as error:
                return self._respond(500,{'error':str(error)})
        self._respond(404,{'error':'경로가 없습니다.'})

    def do_POST(self):
        origin = f'http://127.0.0.1:{self.server.server_port}'
        if (not self._trusted_host() or self.headers.get_all('Origin') != [origin]
                or not secrets.compare_digest(self.headers.get('X-CSRF-Token',''),self.app.csrf)
                or self.headers.get('Sec-Fetch-Site','same-origin') not in ('same-origin','none')):
            return self._respond(403,{'error':'동일한 로컬 대시보드에서만 변경할 수 있습니다.'})
        if urlsplit(self.path).path != '/api/preferences':
            return self._respond(404,{'error':'경로가 없습니다.'})
        if (self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding')
                or len(self.headers.get_all('Content-Length') or []) != 1):
            return self._respond(400,{'error':'길이가 명시된 JSON 요청이 필요합니다.'})
        try:
            length = int(self.headers['Content-Length'])
            if not 0 < length <= 16384:
                raise ValueError('요청 크기를 초과했습니다.')
            data = json.loads(self.rfile.read(length))
            result = self.app.save_preferences(data)
        except Conflict as error:
            return self._respond(409,{'error':str(error)})
        except (ValueError,TypeError) as error:
            return self._respond(400,{'error':str(error)})
        except OSError as error:
            return self._respond(500,{'error':str(error)})
        self._respond(200,result)


def create_server(root,port=0):
    server = HTTPServer(('127.0.0.1',port),Handler)
    server.app = Dashboard(root)
    server.timeout = 2
    return server


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root/'artifacts/local-learning'
    with session_lock(directory/'dashboard.lock'):
        with create_server(root) as server:
            atomic_json(directory/'dashboard.json',{'url':f'http://127.0.0.1:{server.server_port}/',
                                                   'pid':os.getpid(),'service':SERVICE},durable=True)
            server.serve_forever(poll_interval=.5)


if __name__ == '__main__':
    main()
