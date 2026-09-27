"""Windows reader-sharing and status publication contracts; no game I/O."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from playmodel.atomic_io import atomic_json


def windows_error(code):
    error = PermissionError(errno.EACCES, 'synthetic Windows replace failure')
    error.winerror = code
    return error


class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / 'status.json'
        self.target.write_text('{"old": true}', encoding='utf-8')

    def test_bounded_transient_retries_preserve_atomic_reader_view(self):
        original = os.replace
        attempts = []
        def blocked(source, target):
            attempts.append(Path(source))
            if len(attempts) < 3:
                self.assertEqual(json.loads(self.target.read_text()), {'old': True})
                raise windows_error(5 if len(attempts) == 1 else 32)
            original(source, target)
        with patch('playmodel.atomic_io.os.replace', side_effect=blocked), \
                patch('playmodel.atomic_io.time.sleep') as sleep:
            atomic_json(self.target, {'new': True})
        self.assertEqual(json.loads(self.target.read_text()), {'new': True})
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(set(attempts)), 1)
        self.assertNotEqual(attempts[0].name, 'status.json.tmp')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_persistent_windows_permission_error_propagates_and_cleans_up(self):
        error = windows_error(5)
        with patch('playmodel.atomic_io.os.replace', side_effect=error) as replace, \
                patch('playmodel.atomic_io.time.sleep') as sleep:
            with self.assertRaises(PermissionError) as caught:
                atomic_json(self.target, {'new': True})
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 6)
        self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), .155)
        self.assertEqual(json.loads(self.target.read_text()), {'old': True})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_other_errors_are_not_retried_and_invalid_json_never_replaces(self):
        for error in (OSError(errno.ENOSPC, 'disk full'), windows_error(19)):
            with self.subTest(error=error), patch('playmodel.atomic_io.os.replace', side_effect=error), \
                    patch('playmodel.atomic_io.time.sleep') as sleep:
                with self.assertRaises(OSError):
                    atomic_json(self.target, {'new': True})
                sleep.assert_not_called()
                self.assertEqual(list(self.root.glob('*.tmp')), [])
        with self.assertRaises(ValueError):
            atomic_json(self.target, {'bad': float('nan')})
        self.assertEqual(json.loads(self.target.read_text()), {'old': True})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'actual Windows sharing semantics')
    def test_actual_windows_reader_blocks_replace_then_release_allows_retry(self):
        reader = self.target.open('rb')  # Python reader lacks FILE_SHARE_DELETE.
        self.addCleanup(reader.close)
        blocked = threading.Event()
        finished = threading.Event()
        errors = []
        original = os.replace
        def observe_replace(source, target):
            try:
                original(source, target)
            except OSError as error:
                errors.append(error.winerror)
                blocked.set()
                raise
        def close_reader():
            if blocked.wait(2):
                reader.close()
            finished.set()
        thread = threading.Thread(target=close_reader)
        thread.start()
        try:
            with patch('playmodel.atomic_io.os.replace', side_effect=observe_replace):
                atomic_json(self.target, {'after_reader_release': True})
        finally:
            blocked.set()
            thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertTrue(errors)
        self.assertTrue(all(code in (5, 32, 33) for code in errors))
        self.assertEqual(json.loads(self.target.read_text()), {'after_reader_release': True})
        self.assertEqual(list(self.root.glob('*.tmp')), [])


class LocalStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_recurrent_cycle.py'
        spec = importlib.util.spec_from_file_location('cycle_atomic_test', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_only_changed_runtime_fields_emit_events_and_failed_write_keeps_previous_state(self):
        with tempfile.TemporaryDirectory() as directory, patch('playmodel.execution_log.event') as event:
            status = self.module.LocalStatus(directory, interval_seconds=60)
            try:
                status.update(progress=False)
                event.assert_not_called()
                status.update(phase='combat', run_id='one')
                self.assertEqual(event.call_count, 1)
                self.assertEqual(event.call_args.args[0], 'runtime_status')
                status.update(progress=False)
                status.update(recorded_transitions=2)
                self.assertEqual(event.call_count, 1)
                with patch('playmodel.atomic_io.atomic_json', side_effect=PermissionError('persistent')):
                    with self.assertRaises(PermissionError):
                        status.update(phase='training')
                self.assertEqual(status.state['phase'], 'combat')
                self.assertEqual(event.call_count, 1)
                status.fault('persistent failure')
                self.assertEqual(event.call_args.kwargs['status'], 'fault_paused')
                self.assertEqual(event.call_args.kwargs['error'], 'persistent failure')
            finally:
                status.close()

    def test_unrecoverable_heartbeat_error_is_logged_and_surfaced_on_next_progress(self):
        with tempfile.TemporaryDirectory() as directory, patch('playmodel.execution_log.event') as event:
            status = self.module.LocalStatus(directory, interval_seconds=60)
            status.close()
            error = PermissionError('persistent heartbeat write failure')
            with patch.object(status.stop, 'wait', return_value=False), \
                    patch('playmodel.atomic_io.atomic_json', side_effect=error):
                status._heartbeat()
            self.assertEqual(event.call_args.args[0], 'runtime_status_write_failed')
            with self.assertRaises(PermissionError) as caught:
                status.update(phase='training')
            self.assertIs(caught.exception, error)
            status.fault(str(error))
            self.assertEqual(json.loads((Path(directory) / 'status.json').read_text())['status'], 'fault_paused')

    def test_repeated_fault_updates_latest_request_and_preserves_prior_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            status = self.module.LocalStatus(directory, interval_seconds=60)
            try:
                status.fault('first failure')
                status.fault('second failure')
                current = json.loads((Path(directory) / 'help-request.json').read_text())
                self.assertEqual(current['reason'], 'second failure')
                history = [json.loads(path.read_text())['reason']
                           for path in (Path(directory) / 'help-requests').glob('*.json')]
                self.assertIn('first failure', history)
                self.assertIn('second failure', history)
            finally:
                status.close()


if __name__ == '__main__':
    unittest.main()
