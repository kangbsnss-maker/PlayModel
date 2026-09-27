import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from playmodel.execution_log import run_logged, event, diagnose


class ExecutionLogTests(unittest.TestCase):
    def setUp(self):
        # verify_preparation itself is logged; isolate the logger under test
        # while retaining and restoring that outer execution afterwards.
        active = patch('playmodel.execution_log._current', None)
        active.start()
        self.addCleanup(active.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def record(self):
        directory = next(self.base.iterdir())
        return directory, json.loads((directory/'execution.json').read_text(encoding='utf-8'))

    def test_output_keeps_original_contract_and_stage_source(self):
        status = self.base/'status.json'
        # Runtime status sits outside the per-execution directory.
        segment = self.base/'run/segments/one'
        segment.mkdir(parents=True)
        (segment/'report.json').write_text('{"reason":"Unrecognized screen"}', encoding='utf-8')
        status.write_text(json.dumps({'status': 'running', 'process_id': os.getpid(),
                                     'run_directory': str(self.base/'run')}), encoding='utf-8')
        def callback():
            print('한글 output')
            print('diagnostic', file=sys.stderr)
            event('runtime_status', phase='training', status_path=str(status))
            return 0
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(run_logged(callback, program='test', directory=self.base/'logs'), 0)
        self.assertEqual(out.getvalue(), '한글 output\n')
        self.assertEqual(err.getvalue(), 'diagnostic\n')
        directory = next((self.base/'logs').iterdir())
        report = diagnose(directory)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['runtime_status']['status'], 'running')
        self.assertEqual(report['session_reports'][0]['reason'], 'Unrecognized screen')
        self.assertIn('한글', report['stdout.log'])
        self.assertTrue((directory/'diagnosis.json').is_file())
        status.write_text(json.dumps({'status': 'unrelated-new-run', 'process_id': -1}), encoding='utf-8')
        old_report = diagnose(directory)
        self.assertEqual(old_report['runtime_status']['phase'], 'training')
        self.assertNotIn('unrelated-new-run', str(old_report))

    def test_exception_trace_and_streams_restored(self):
        original = sys.stdout
        def fail():
            raise ValueError('source evidence missing')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'source evidence'):
                run_logged(fail, program='test', directory=self.base)
        directory, state = self.record()
        self.assertIs(sys.stdout, original)
        self.assertEqual(state['exit_code'], 1)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('ValueError: source evidence missing', diagnose(directory)['stderr.log'])

    def test_system_exit_and_none_streams(self):
        def fail():
            print('pythonw output')
            raise SystemExit(7)
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            with self.assertRaises(SystemExit) as caught:
                run_logged(fail, program='pythonw', directory=self.base)
        self.assertEqual(caught.exception.code, 7)
        directory, state = self.record()
        self.assertEqual(state['exit_code'], 7)
        self.assertIn('pythonw output', diagnose(directory)['stdout.log'])

    def test_nested_run_does_not_duplicate_and_thread_failure_recorded(self):
        def fail():
            raise RuntimeError('worker died')
        def callback():
            run_logged(lambda: print('nested'), directory=self.base)
            thread = threading.Thread(target=fail)
            thread.start()
            thread.join()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_logged(callback, program='thread', directory=self.base)
        self.assertEqual(len(list(self.base.iterdir())), 1)
        directory, state = self.record()
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(diagnose(directory)['errors'][0]['error'], 'worker died')

    def test_bootstrap_catches_import_failure_without_game_input(self):
        # A real child verifies the pre-import boundary and original exit code.
        root = Path(__file__).resolve().parents[1]
        script = self.base/'broken.py'
        script.write_text("import sys\nsys.path.insert(0," + repr(str(root/'scripts')) + ")\n"
                          "from _execution_bootstrap import launch\nlaunch(__file__)\n"
                          "raise ImportError('intentional-bootstrap-test')\n", encoding='utf-8')
        result = subprocess.run([sys.executable, '-X', 'utf8', str(script)], capture_output=True,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'intentional-bootstrap-test', result.stderr)
        self.assertEqual(result.stderr.count(b'Traceback (most recent call last):'), 1)
        matches = [p for p in (root/'artifacts/program-logs').glob('*-broken.py')
                   if json.loads((p/'execution.json').read_text(encoding='utf-8'))['program'] == str(script)]
        self.assertEqual(len(matches), 1)
        self.assertEqual(diagnose(matches[0])['status'], 'failed')

    def test_metadata_write_failure_does_not_mask_original_exception(self):
        def fail():
            with patch('playmodel.execution_log.atomic_json', side_effect=PermissionError('locked')):
                event('runtime_status', phase='combat')
            raise RuntimeError('original failure')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'original failure'):
                run_logged(fail, program='write-failure', directory=self.base)
        directory, state = self.record()
        self.assertEqual(state['exit_code'], 1)
        self.assertIn('locked', state['logging_error'])

    def test_binary_stdout_stays_untouched(self):
        binary = io.BytesIO()
        original = io.TextIOWrapper(binary, encoding='utf-8')
        def callback():
            self.assertIs(sys.stdout, original)
            sys.stdout.buffer.write(b'\x00\xff\x01')
        with contextlib.redirect_stdout(original):
            run_logged(callback, program='capture', directory=self.base, capture_stdout=False)
        self.assertEqual(binary.getvalue(), b'\x00\xff\x01')
        directory, _ = self.record()
        self.assertEqual((directory/'stdout.log').read_bytes(), b'')

    def test_sensitive_arguments_not_stored(self):
        with patch.object(sys, 'argv', ['app', '--token', 'secret-value', '--password=hidden', '--device', 'cpu']):
            run_logged(lambda: None, program='args', directory=self.base)
        _, state = self.record()
        self.assertNotIn('secret-value', str(state['arguments']))
        self.assertNotIn('hidden', str(state['arguments']))
        self.assertIn('cpu', state['arguments'])


if __name__ == '__main__':
    unittest.main()
