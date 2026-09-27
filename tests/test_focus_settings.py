import json
from pathlib import Path
import tempfile
import unittest

from playmodel.games.brotato.focus_settings import focus_pause_status


class FocusSettingsTests(unittest.TestCase):
    def test_known_modes_and_legacy_precedence_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Brotato/profile/settings.json'
            path.parent.mkdir(parents=True)
            for settings, expected in [({'on_lost_focus': 0, 'pause_on_focus_lost': True}, '꺼짐'),
                    ({'on_lost_focus': 1}, '켜짐'), ({'on_lost_focus': 2}, '켜짐'),
                    ({'pause_on_focus_lost': False}, '꺼짐'),
                    ({'on_lost_focus': True}, '확인 불가'), ({'on_lost_focus': 9}, '확인 불가')]:
                with self.subTest(settings=settings):
                    original = json.dumps({'settings': settings}).encode()
                    path.write_bytes(original)
                    self.assertIn(expected, focus_pause_status(directory))
                    self.assertEqual(path.read_bytes(), original)

    def test_missing_corrupt_or_ambiguous_profile_never_claims_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn('확인 불가', focus_pause_status(directory))
            path = Path(directory) / 'Brotato/profile/settings.json'
            path.parent.mkdir(parents=True)
            path.write_text('{')
            self.assertIn('확인 불가', focus_pause_status(directory))
            path.write_text('{"settings":{"on_lost_focus":0}}')
            second = path.parents[1] / 'another/settings.json'
            second.parent.mkdir()
            second.write_text(path.read_text())
            self.assertIn('프로필 선택', focus_pause_status(directory))
