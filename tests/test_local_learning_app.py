"""Launcher boundary tests. Never launch a learner or send game input."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(sys.platform == 'win32', 'Windows desktop launcher')
class LocalLearningAppTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('local_learning_app_test',
            Path(__file__).resolve().parents[1] / 'scripts/local_learning_app.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module.ROOT = self.root
        self.module.CONFIG = self.root / 'configs/local/learner-launch.json'
        self.module.RUNTIME = self.root / 'artifacts/local-learning'
        self.module.RUNTIME.mkdir(parents=True)
        self.module.LOCK = self.module.RUNTIME / 'worker.lock'
        self.module.STOP = self.root / 'artifacts/BROTATO_STOP'
        self.model = self.root / 'candidate.pt'
        self.model.write_bytes(b'not loaded by launcher')
        python = self.root / '.venv/Scripts/python.exe'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'never executed')

    def test_desktop_counters_distinguish_decisions_updates_and_data(self):
        counts = {'menu_decisions': 12, 'tactic_decisions': 1000, 'accepted_updates': 7,
                  'trained_decisions': 800, 'runs_completed': 3}
        text = self.module.format_learning_counts(counts)
        self.assertIn('누적 판단 1,012회', text)
        self.assertIn('학습(가중치 갱신) 7회', text)
        self.assertIn('유효 학습 자료 800건', text)
        self.assertIn('완료 학습 3판', text)
        with self.assertRaises(ValueError):
            self.module.format_learning_counts({**counts, 'accepted_updates': True})

    def test_counter_error_preserves_previous_verified_values(self):
        from unittest.mock import Mock
        fetch = Mock(return_value={'counts': {'menu_decisions': 1, 'tactic_decisions': 4,
            'accepted_updates': 2, 'trained_decisions': 3, 'runs_completed': 1}, 'errors': []})
        reader = self.module.LearningCounterReader(self.root, fetch=fetch, start=False)
        reader.refresh_once()
        before = reader.snapshot
        fetch.side_effect = OSError('temporary stats read error')
        reader.refresh_once()
        self.assertIs(reader.snapshot, before)
        self.assertIn('temporary', reader.error)
        self.assertFalse(self.module.STOP.exists())

    def test_slow_counter_read_runs_outside_ui_thread(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        def fetch():
            entered.set()
            release.wait(2)
            return {'counts': {'menu_decisions': 0, 'tactic_decisions': 0, 'accepted_updates': 0,
                               'trained_decisions': 0, 'runs_completed': 0}}
        reader = self.module.LearningCounterReader(self.root, fetch=fetch)
        try:
            self.assertTrue(entered.wait(1))
            self.assertIsNone(reader.snapshot)
        finally:
            reader.stop.set()
            release.set()

    def test_dashboard_reuses_local_service_without_starting_learning(self):
        from unittest.mock import Mock
        endpoint = {'url': 'http://127.0.0.1:45678/', 'pid': 42}
        (self.module.RUNTIME / 'dashboard.json').write_text(json.dumps(endpoint))
        response = Mock()
        response.read.return_value = b'{"service":"playmodel-laya-dashboard","pid":42}'
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        self.module.STOP.write_text('keep stopped')
        with patch.object(self.module, 'build_opener', return_value=opener), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            self.assertEqual(self.module.open_dashboard(show_browser=False), endpoint['url'])
        spawn.assert_not_called()
        self.assertEqual(self.module.STOP.read_text(), 'keep stopped')
        self.assertEqual(opener.open.call_args.args[0], endpoint['url']+'api/health')

    def test_explicit_start_resumes_hidden_worker_and_saves_config(self):
        self.module.STOP.write_text('previous user stop')
        with patch.object(self.module, 'worker_running', return_value=False), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 123
            pid, log = self.module.launch_worker(self.model, None)
        self.assertEqual(pid, 123)
        self.assertFalse(self.module.STOP.exists())
        args, kwargs = spawn.call_args
        self.assertIn('--continuous', args[0])
        self.assertIn('--resume-latest', args[0])
        self.assertIn('--recover-active-run', args[0])
        self.assertEqual(kwargs['creationflags'], self.module.subprocess.CREATE_NO_WINDOW)
        self.assertEqual(kwargs['stdin'], self.module.subprocess.DEVNULL)
        self.assertNotIn('shell', kwargs)
        self.assertTrue(log.exists())
        self.assertEqual(json.loads(self.module.CONFIG.read_text())['checkpoint'], str(self.model.resolve()))

    def test_duplicate_or_invalid_launch_preserves_stop_file(self):
        self.module.STOP.write_text('user stop')
        with patch.object(self.module, 'worker_running', return_value=True), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            with self.assertRaises(RuntimeError):
                self.module.launch_worker(self.model, None)
            spawn.assert_not_called()
        self.assertTrue(self.module.STOP.exists())
        with patch.object(self.module, 'worker_running', return_value=False), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                self.module.launch_worker(self.root / 'missing.pt', None)
            spawn.assert_not_called()
        self.assertTrue(self.module.STOP.exists())

    def test_optional_summary_is_structured_argument_not_shell_text(self):
        summary = self.root / 'comparison with spaces.json'
        summary.write_text('{}')
        with patch.object(self.module, 'worker_running', return_value=False), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 123
            self.module.launch_worker(self.model, summary)
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index('--resume-summary') + 1], str(summary.resolve()))

    def test_live_os_lock_detected_and_release_not_confused_with_stale_file(self):
        import msvcrt
        self.assertFalse(self.module.worker_running())
        with self.module.LOCK.open('r+b') as stream:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                self.assertTrue(self.module.worker_running())
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.assertFalse(self.module.worker_running())

    def test_explicit_new_experiment_does_not_resume_old_model_state(self):
        with patch.object(self.module, 'worker_running', return_value=False), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 123
            self.module.launch_worker(self.model, None, resume_latest=False)
        self.assertNotIn('--resume-latest', spawn.call_args.args[0])
        self.assertNotIn('--resume-summary', spawn.call_args.args[0])
        self.assertTrue(json.loads(self.module.CONFIG.read_text())['resume_latest'])

    def test_laya_start_uses_local_worker_and_does_not_resume_cnn_ppo(self):
        model_dir = self.root / 'models/laya/base'
        model_dir.mkdir(parents=True)
        (model_dir / 'source-manifest.json').write_text('{}')
        python = self.root / '.venv-laya/Scripts/python.exe'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'not executed')
        with patch.object(self.module, 'worker_running', return_value=False), \
                patch.object(self.module.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 123
            self.module.launch_worker(self.model, None, backend='laya')
        command = spawn.call_args.args[0]
        self.assertIn(str(self.root / 'scripts/run_laya_learning.py'), command)
        self.assertNotIn('--online-updates', command)
        self.assertNotIn('--resume-summary', command)
        self.assertEqual(json.loads(self.module.CONFIG.read_text())['backend'], 'laya')


if __name__ == '__main__':
    unittest.main()
