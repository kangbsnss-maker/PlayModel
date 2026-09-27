from types import SimpleNamespace
import unittest
from unittest.mock import patch

from playmodel.games.brotato.neural_navigation import FrozenNavigation
from playmodel.games.brotato.menu import BUTTONS, ButtonSelection


class FrozenNavigationTests(unittest.TestCase):
    def setUp(self):
        self.pixels = bytes(1920 * 1080 * 4)
        self.guard = FrozenNavigation()
        self.decision = SimpleNamespace(decision_id='frozen', target='choose_3',
                                        observation=SimpleNamespace(scene='level_up'))
        self.shot = {'hwnd': 7, 'capture_started_at_ns': 1_000_000_000}

    def arm(self):
        self.guard.arm(self.decision, self.shot, self.pixels,
                       BUTTONS[self.decision.observation.scene], 1_100_000_000)
        self.guard.sent(1_110_000_000)

    def propose(self, pixels=None, **shot):
        return self.guard.propose(self.decision,
            {'hwnd': 7, 'capture_started_at_ns': 1_200_000_000, **shot},
            self.pixels if pixels is None else pixels, 1_300_000_000)

    def test_fresh_arrow_never_becomes_enter_at_target(self):
        self.arm()
        with patch('playmodel.games.brotato.neural_navigation.selected_button',
                   return_value=ButtonSelection('choose_1', BUTTONS['level_up']['choose_1'], 'fixture')) as focus:
            self.assertEqual(self.propose().key, 'right')
            focus.return_value = ButtonSelection('choose_3', BUTTONS['level_up']['choose_3'], 'fixture')
            self.assertIsNone(self.propose())

    def test_content_change_stale_frame_other_window_and_new_decision_require_ocr(self):
        self.arm()
        changed = bytearray(self.pixels)
        changed[(450*1920+300)*4] = 1
        self.assertIsNone(self.propose(changed))
        self.assertIsNone(self.propose(capture_started_at_ns=1_050_000_000))
        self.assertIsNone(self.propose(capture_started_at_ns=1_400_000_000))
        self.assertIsNone(self.propose(hwnd=8))
        self.decision.decision_id = 'other'
        self.assertIsNone(self.propose())

    def test_only_departure_shop_arrows_skip_offer_checks_and_still_never_enter(self):
        self.decision.observation.scene = 'shop'
        self.decision.target = 'buy_1'
        self.arm()
        self.assertIsNone(self.guard.anchor)
        self.decision.target = 'depart'
        self.arm()
        changed_offer = bytearray(self.pixels)
        changed_offer[(350*1920+300)*4] = 1
        with patch('playmodel.games.brotato.neural_navigation.selected_button',
                   return_value=ButtonSelection('lock_3', BUTTONS['shop']['lock_3'], 'fixture')) as focus:
            self.assertIn(self.propose(changed_offer).key, ('left', 'right', 'up', 'down'))
            focus.return_value = ButtonSelection('depart', BUTTONS['shop']['depart'], 'fixture')
            self.assertIsNone(self.propose(changed_offer))
        changed_offer[(50*1920+300)*4] = 1
        self.assertIsNone(self.propose(changed_offer))

    def test_navigation_budget_requires_full_ocr_again(self):
        self.arm()
        self.guard.steps = 8
        self.assertIsNone(self.propose())


if __name__ == '__main__':
    unittest.main()
