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


if __name__ == '__main__':
    unittest.main()
