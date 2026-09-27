import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from playmodel.obs import ensure_obs_running


class ObsStartupTests(unittest.TestCase):
    def test_existing_connection_does_not_launch(self):
        with patch('playmodel.obs.ObsClient') as client, patch('subprocess.Popen') as launch:
            ensure_obs_running()
        client.return_value.__enter__.return_value.call.assert_called_once_with('GetVersion')
        launch.assert_not_called()

    def test_running_process_waits_without_duplicate(self):
        with patch('playmodel.obs.ObsClient', side_effect=[ConnectionRefusedError(), MagicMock()]), \
             patch('subprocess.run', return_value=SimpleNamespace(stdout='"obs64.exe","123"')), \
             patch('subprocess.Popen') as launch:
            ensure_obs_running()
        launch.assert_not_called()

    def test_absent_process_launches_and_verifies(self):
        with patch('playmodel.obs.ObsClient', side_effect=[ConnectionRefusedError(), MagicMock()]), \
             patch('subprocess.run', return_value=SimpleNamespace(stdout='')), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('shutil.which', return_value='C:/obs64.exe'), \
             patch('subprocess.Popen', return_value=SimpleNamespace(pid=123)) as launch:
            ensure_obs_running()
        self.assertIn('--disable-shutdown-check', launch.call_args.args[0])
        self.assertIn('creationflags', launch.call_args.kwargs)

    def test_configuration_failure_is_not_silently_retried(self):
        with patch('playmodel.obs.ObsClient', side_effect=OSError('server disabled')), \
             patch('subprocess.Popen') as launch:
            with self.assertRaisesRegex(OSError, 'disabled'):
                ensure_obs_running()
        launch.assert_not_called()
