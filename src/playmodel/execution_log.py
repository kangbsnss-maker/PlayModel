"""Local, per-process execution evidence. No network or game input."""
from __future__ import annotations

from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid
from playmodel.atomic_io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
_current = None


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _arguments(args):
    result, hide = [], False
    for arg in args:
        arg = str(arg)
        sensitive = bool(re.search(r'(password|token|secret|api[-_]?key)', arg.split('=', 1)[0], re.I))
        result.append('[REDACTED]' if hide else arg.split('=', 1)[0] + '=[REDACTED]'
                      if sensitive and '=' in arg else arg)
        hide = sensitive and '=' not in arg
    return result


class _Tee:
    def __init__(self, original, output, lock):
        self.original, self.output, self.lock = original, output, lock

    def write(self, value):
        with self.lock:
            self.output.write(value)
            self.output.flush()
            if self.original is not None:
                self.original.write(value)
        return len(value)

    def flush(self):
        with self.lock:
            self.output.flush()
            if self.original is not None:
                self.original.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return 'utf-8'

    def __getattr__(self, name):
        return getattr(self.original, name)


def event(kind, **details):
    if _current is not None:
        try:
            _current.event(kind, **details)
        except (OSError, ValueError) as error:
            # A diagnostic write must not mask the original game failure or
            # turn a healthy input loop into a logging-induced crash.
            _current.had_error = True
            _current.state['logging_error'] = f'{type(error).__name__}: {error}'
            try:
                sys.stderr.write('EXECUTION_LOG_WRITE_FAILED: ' + str(error) + '\n')
            except (OSError, AttributeError):
                pass


def exception(kind, error):
    event(kind, error_type=type(error).__name__, error=str(error),
          traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))


def child_stderr_path(program):
    base = _current.directory / 'children' if _current else ROOT / 'artifacts/helper-logs'
    base.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(program).name)
    return base / f'{name}-{uuid.uuid4().hex}.stderr.log'


class Execution:
    def __init__(self, program, directory=None):
        base = Path(directory) if directory else ROOT / 'artifacts/program-logs'
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(program).name)
        self.directory = base / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                                 + f'-{os.getpid()}-{uuid.uuid4().hex[:8]}-{name}')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.events = (self.directory / 'events.jsonl').open('a', encoding='utf-8', buffering=1)
        self.started = time.perf_counter()
        self.state = dict(schema='playmodel.program-execution.v1', program=str(program),
                          pid=os.getpid(), parent_pid=os.getppid(), cwd=str(Path.cwd()),
                          interpreter=sys.executable, python=sys.version, arguments=_arguments(sys.argv),
                          parent_execution=os.environ.get('PLAYMODEL_PARENT_EXECUTION'),
                          started_at=_utc(), status='running', exit_code=None,
                          automatic_agent_call=False)
        self.had_error = False
        self.save()

    def save(self):
        target = self.directory / 'execution.json'
        atomic_json(target, self.state)

    def event(self, kind, **details):
        with self.lock:
            item = dict(at=_utc(), elapsed_seconds=time.perf_counter()-self.started,
                        event=kind, **details)
            self.events.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')
            if details.get('error') or details.get('status') == 'fault_paused':
                self.had_error = True
            self.state['last_event'] = item
            if details.get('phase'):
                self.state['last_phase_event'] = item
            if details.get('status_path'):
                self.state['status_path'] = str(details['status_path'])
            self.save()


def diagnose(directory):
    """Evidence summary, not an assertion of an inferred root cause."""
    directory = Path(directory)
    state = json.loads((directory / 'execution.json').read_text(encoding='utf-8'))
    errors = []
    with (directory / 'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except ValueError:
                continue  # A concurrently written final line may be incomplete.
            if item.get('error') or item.get('status') == 'fault_paused':
                errors.append(item)
                errors = errors[-20:]
    result = dict(program=state['program'], pid=state['pid'], status=state['status'],
                  parent_execution=state.get('parent_execution'),
                  exit_code=state['exit_code'], last_event=state.get('last_event'),
                  last_phase_event=state.get('last_phase_event'),
                  errors=errors, evidence_directory=str(directory.resolve()),
                  note='Observed errors only; abrupt termination may leave status running. '
                       'A heartbeat alone does not prove game progress.')
    status_path = state.get('status_path')
    if status_path:
        try:
            live_status = json.loads(Path(status_path).read_text(encoding='utf-8'))
            # The same status.json is reused after resume. Never attribute a
            # later process's state to an older execution's diagnosis.
            if (live_status.get('process_id') == state['pid']
                    and (not state.get('ended_at') or live_status.get('heartbeat_utc', '') <= state['ended_at'])):
                result['runtime_status'] = live_status
            else:
                result['runtime_status'] = state.get('last_phase_event') or {}
                result['runtime_status_source'] = 'recorded_event; status file now belongs to a different execution'
            result['runtime_status_path'] = status_path
            run_directory = result['runtime_status'].get('run_directory')
            if run_directory:
                reports = sorted(Path(run_directory).glob('segments/*/report.json'),
                                 key=lambda path: path.name + str(path.parent), reverse=True)[:10]
                result['session_reports'] = []
                for path in reports:
                    try:
                        report = json.loads(path.read_text(encoding='utf-8'))
                        result['session_reports'].append(dict(path=str(path),
                            **{key: report.get(key) for key in ('reason', 'error', 'session_directory')}))
                    except (OSError, ValueError) as error:
                        result['session_reports'].append(dict(path=str(path), read_error=str(error)))
        except (OSError, ValueError) as error:
            result['runtime_status_read_error'] = str(error)
    for name in ('stdout.log', 'stderr.log', 'fatal.log'):
        path = directory / name
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                result[name] = stream.read().decode('utf-8', errors='replace')
    return result


def run_logged(callback, *, program=None, directory=None, capture_stdout=True):
    global _current
    if _current is not None:
        return callback()
    run = Execution(program or sys.argv[0], directory)
    _current = run
    old_out, old_err, old_thread = sys.stdout, sys.stderr, threading.excepthook
    old_parent = os.environ.get('PLAYMODEL_PARENT_EXECUTION')
    os.environ['PLAYMODEL_PARENT_EXECUTION'] = str(run.directory.resolve())
    output = (run.directory / 'stdout.log').open('a', encoding='utf-8', buffering=1)
    errors = (run.directory / 'stderr.log').open('a', encoding='utf-8', buffering=1)
    fatal = (run.directory / 'fatal.log').open('a', encoding='utf-8')
    old_fault = faulthandler.is_enabled()
    sys.stdout = _Tee(old_out, output, run.lock) if capture_stdout else old_out
    sys.stderr = _Tee(old_err, errors, run.lock)
    def thread_error(args):
        exception('thread_exception', args.exc_value)
        old_thread(args)
    threading.excepthook = thread_error
    if not old_fault:
        faulthandler.enable(file=fatal, all_threads=True)
    code = 0
    try:
        event('started')
        result = callback()
        code = result if isinstance(result, int) else 0
        return result
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        if code:
            exception('system_exit', error)
        raise
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        exception('uncaught_exception', error)
        traceback.print_exc(file=errors)
        raise
    finally:
        run.state.update(status='failed' if code else 'completed_with_errors' if run.had_error else 'completed',
                         exit_code=code, ended_at=_utc(), duration_seconds=time.perf_counter()-run.started)
        try:
            event('finished', exit_code=code)
            try:
                (run.directory / 'diagnosis.json').write_text(
                    json.dumps(diagnose(run.directory), ensure_ascii=False, indent=2), encoding='utf-8')
            except (OSError, ValueError) as error:
                event('diagnosis_write_failed', error=str(error))
        finally:
            sys.stdout, sys.stderr, threading.excepthook = old_out, old_err, old_thread
            if not old_fault:
                faulthandler.disable()
            if old_parent is None:
                os.environ.pop('PLAYMODEL_PARENT_EXECUTION', None)
            else:
                os.environ['PLAYMODEL_PARENT_EXECUTION'] = old_parent
            _current = None
            for stream in (output, errors, fatal, run.events):
                try:
                    stream.close()
                except OSError:
                    pass  # Preserve the original process exit/exception.


def cli_main():
    # Import failures are inside the logged boundary as well.
    def invoke():
        from playmodel.cli import main
        return main()
    return run_logged(invoke, program='playmodel')


def launch_module(name, *, capture_stdout=True):
    """Start the log before a module's application imports are evaluated."""
    if _current is not None:
        return
    import runpy
    run_logged(lambda: runpy.run_module(name, run_name='__main__', alter_sys=True),
               program=name, capture_stdout=capture_stdout)
    raise SystemExit(0)
