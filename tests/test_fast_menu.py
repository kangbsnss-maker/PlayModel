"""A learned perception cannot bypass current focus or confirm menu choices."""
from types import SimpleNamespace
import unittest
from unittest.mock import create_autospec, patch

from playmodel.games.brotato.fast_menu import fast_navigation
from playmodel.games.brotato.menu_model import MenuClassifier


class FastMenuTests(unittest.TestCase):
    def model(self, scene='difficulty', focus='danger_3', abstain=None):
        model = create_autospec(MenuClassifier, instance=True)
        model.predict.return_value = SimpleNamespace(scene=scene, focus=focus,
            label=f'{scene}:{focus}', abstain_reason=abstain)
        return model

    def test_navigation_towards_six_requires_matching_pixel_focus(self):
        with patch('playmodel.games.brotato.setup_run.focused_tile', return_value=3):
            result = fast_navigation(self.model(), b'', 1920, 1080, game_build_id='test')
            self.assertEqual((result.key, result.target), ('right', 'danger_6'))
        with patch('playmodel.games.brotato.setup_run.focused_tile', return_value=2):
            self.assertIsNone(fast_navigation(self.model(), b'', 1920, 1080, game_build_id='test'))

    def test_confirmation_and_dynamic_screens_always_fall_back(self):
        for scene, focus in [('difficulty', 'danger_6'), ('pause', 'continue'),
                             ('shop', 'buy_0'), ('level_up', 'choose_1'), ('result', 'restart')]:
            self.assertIsNone(fast_navigation(self.model(scene, focus), b'', 1920, 1080, game_build_id='test'))

    def test_abstention_and_resolution_do_not_navigate(self):
        self.assertIsNone(fast_navigation(self.model(abstain='unapproved_checkpoint'), b'',
                                         1920, 1080, game_build_id='test'))
        self.assertIsNone(fast_navigation(self.model(), b'', 960, 540, game_build_id='test'))


if __name__ == '__main__':
    unittest.main()
