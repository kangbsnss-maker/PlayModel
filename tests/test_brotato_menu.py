"""Pure menu calibration regression tests; no Win32 calls or game input."""
from pathlib import Path
from copy import deepcopy
import json
import unittest

from playmodel.games.brotato.menu import ANCHORS, BUTTONS, classify_scene, navigation_key, selected_button, selected_stat_row


def report_for(scene):
    return {"lines": [{"words": [{"text": token, "x": region[0] + 5, "y": region[1] + 5,
                                  "width": 120, "height": 30}]} for token, region in ANCHORS[scene]]}


def pixels(selected=()):
    image = bytearray(bytes((20, 20, 20, 0)) * 1920 * 1080)
    for left, top, right, bottom in selected:
        row = bytes((200, 200, 200, 0)) * (right - left)
        for y in range(top, bottom):
            start = (y * 1920 + left) * 4
            image[start:start + len(row)] = row
    return bytes(image)


class MenuTests(unittest.TestCase):
    def test_loot_literal_ocr_alias_requires_both_positioned_anchors(self):
        report = {'lines': [
            {'words': [{'text': 'ltem found!', 'x': 570, 'y': 200, 'width': 300, 'height': 45}]},
            {'words': [{'text': 'RecycIe', 'x': 570, 'y': 765, 'width': 160, 'height': 30}]},
        ]}
        original = deepcopy(report)
        self.assertEqual(classify_scene(report).scene, 'loot')
        self.assertEqual(report, original)
        for index in range(2):
            self.assertEqual(classify_scene({'lines': [report['lines'][index]]}).scene, 'unknown')
        for text in ('1tem found!', 'ltem f0und!'):
            bad = deepcopy(report)
            bad['lines'][0]['words'][0]['text'] = text
            self.assertEqual(classify_scene(bad).scene, 'unknown')
        report['lines'][1]['words'][0]['x'] = 1500
        self.assertEqual(classify_scene(report).scene, 'unknown')

    def test_loot_stat_focus_requires_a_unique_calibrated_border_pair(self):
        borders = ((1115, 633, 1470, 635), (1115, 666, 1470, 668))
        focused = pixels(borders)
        selection = selected_stat_row(focused, scene='loot')
        self.assertEqual(selection.selected_id, 'stat_633')
        self.assertEqual(navigation_key(selection.rect, BUTTONS['loot']['recycle'], scene='loot'), 'left')
        self.assertIsNone(selected_stat_row(focused).selected_id)  # Shop geometry is separate.
        self.assertIsNone(selected_stat_row(focused, scene='level_up').selected_id)
        self.assertIsNone(selected_stat_row(pixels(), scene='loot').selected_id)
        self.assertIsNone(selected_stat_row(pixels(borders[:1]), scene='loot').selected_id)
        extra = ((1115, 733, 1470, 735), (1115, 766, 1470, 768))
        self.assertIsNone(selected_stat_row(pixels(borders + extra), scene='loot').selected_id)

    def test_actual_failed_startup_is_loot_with_stat_focus(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        root = (Path(__file__).resolve().parents[1] /
                'artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/'
                'evaluation-0-source-ccd15f4dc99d4941908066f214205a3d/initial-state')
        captures = sorted(root.glob('*/startup-ocr.json'))
        if not captures:
            self.skipTest('private failed startup evidence absent')
        self.assertEqual(len(captures), 3)
        for path in captures:
            raw = json.loads(path.read_text(encoding='utf-8'))
            original = deepcopy(raw)
            self.assertEqual(classify_scene(raw).scene, 'loot', str(path))
            self.assertEqual(raw, original)
            frame, width, height = read_diagnostic_png(path.with_name('frame.png'))
            self.assertEqual(selected_button(frame, width, height, scene='loot').reason, 'no_selected_candidate')
            focus = selected_stat_row(frame, width, height, scene='loot')
            self.assertEqual(focus.selected_id, 'stat_633', str(path))
            self.assertEqual(navigation_key(focus.rect, BUTTONS['loot']['recycle'], scene='loot'), 'left')

    def test_level_up_ocr_alias_keeps_raw_text_and_requires_positioned_choose(self):
        from playmodel.games.brotato.session import calibrated_rules

        rules = calibrated_rules()
        for title in ('Level up!', 'LeveI up!'):
            with self.subTest(title=title):
                report = {'text': title + ' Choose', 'lines': [
                    {'words': [{'text': title, 'x': 650, 'y': 280, 'width': 230, 'height': 40}]},
                    {'words': [{'text': 'Choose', 'x': 200, 'y': 640, 'width': 110, 'height': 30}]},
                ]}
                original = deepcopy(report)
                self.assertEqual(classify_scene(report).scene, 'level_up')
                self.assertEqual(rules.classify(report['text'], lines=report['lines'],
                                                image_size=(1920,1080)), 'wave_clear')
                self.assertEqual(report, original)
                for missing in ('title', 'choose', 'position'):
                    bad = deepcopy(report)
                    if missing == 'position':
                        bad['lines'][1]['words'][0]['y'] = 800
                    else:
                        bad['lines'].pop(0 if missing == 'title' else 1)
                    self.assertEqual(classify_scene(bad).scene, 'unknown')
                    self.assertIsNone(rules.classify(bad['text'], lines=bad['lines'], image_size=(1920,1080)))

    def test_level_up_alias_does_not_generalize_confusables_or_override_pause(self):
        from playmodel.games.brotato.session import calibrated_rules

        rules = calibrated_rules()
        for title in ('Leve1 up!', 'Leve! up!', 'IeveI up!'):
            report = {'text': title + ' Choose', 'lines': [
                {'words': [{'text': title, 'x': 650, 'y': 280, 'width': 230, 'height': 40}]},
                {'words': [{'text': 'Choose', 'x': 200, 'y': 640, 'width': 110, 'height': 30}]},
            ]}
            self.assertEqual(classify_scene(report).scene, 'unknown')
            self.assertIsNone(rules.classify(report['text'], lines=report['lines'], image_size=(1920,1080)))
        report['lines'][0]['words'][0]['text'] = 'LeveI up!'
        self.assertIsNone(rules.classify('LeveI up! Choose Resume', lines=report['lines'],
                                        image_size=(1920,1080)))

    def test_actual_level_up_ocr_alias_is_recognized_by_scene_and_terminal(self):
        from playmodel.games.brotato.session import calibrated_rules

        root = (Path(__file__).resolve().parents[1] /
                'artifacts/recurrent-recovery/20260927T150510/'
                'evaluation-0-source-a8f97c3dadf04230a348a528145cd911/segments/'
                '20260927T060803Z-26643c41/pilots/'
                'neural-5fbeb6b2b34f4b55acfd7b143bffb032/ocr')
        paths = sorted(root.glob('*/ocr.json'))
        if not paths:
            self.skipTest('private OCR calibration evidence absent')
        rules, matches = calibrated_rules(), 0
        for path in paths:
            raw = json.loads(path.read_text(encoding='utf-8'))['raw']
            if 'LeveI up!' not in raw.get('text', ''):
                continue
            original = deepcopy(raw)
            self.assertEqual(classify_scene(raw).scene, 'level_up', str(path))
            self.assertEqual(rules.classify(raw['text'], lines=raw['lines'], image_size=(1920,1080)),
                             'wave_clear', str(path))
            self.assertEqual(raw, original)
            matches += 1
        self.assertGreaterEqual(matches, 2)

    def test_shop_pause_without_enabled_restart_requires_other_anchors(self):
        report = {'lines': [
            {'words': [{'text': 'Resume', 'x': 272, 'y': 262, 'width': 155, 'height': 27}]},
            {'words': [{'text': 'End the run', 'x': 235, 'y': 438, 'width': 230, 'height': 31}]},
            {'words': [{'text': 'Options', 'x': 273, 'y': 618, 'width': 153, 'height': 40}]},
        ]}
        self.assertEqual(classify_scene(report).scene, 'pause')
        self.assertEqual(classify_scene({'lines': report['lines'][:1]}).scene, 'unknown')

    def test_difficulty_ocr_variant_requires_both_positioned_labels(self):
        report = {'lines': [
            {'words': [{'text': 'DifficuIty selection', 'x': 668, 'y': 88, 'width': 560, 'height': 60}]},
            {'words': [{'text': 'DifficuIty', 'x': 1485, 'y': 238, 'width': 120, 'height': 25}]},
        ]}
        self.assertEqual(classify_scene(report).scene, 'difficulty')
        self.assertEqual(classify_scene({'lines': report['lines'][:1]}).scene, 'unknown')
        report['lines'][1]['words'][0]['x'] = 200
        self.assertEqual(classify_scene(report).scene, 'unknown')

    def test_each_scene_requires_positioned_anchors(self):
        for scene in ANCHORS:
            with self.subTest(scene=scene):
                self.assertEqual(classify_scene(report_for(scene)).scene, scene)
                wrong = report_for(scene)
                for line in wrong["lines"]:
                    for word in line["words"]:
                        word["x"], word["y"] = 1700, 980
                self.assertEqual(classify_scene(wrong).scene, "unknown")

    def test_missing_conflicting_and_malformed_anchors_abstain(self):
        shop = report_for("shop")
        self.assertEqual(classify_scene({"lines": shop["lines"][:1]}).scene, "unknown")
        self.assertEqual(classify_scene({"lines": shop["lines"] + report_for("level_up")["lines"]}).reason,
                         "ambiguous_anchors")
        for bad in ({}, {"lines": None}, {"lines": [{"words": [{"text": "商店", "x": float("nan")}]}]}):
            self.assertEqual(classify_scene(bad).reason, "invalid_ocr")
        self.assertEqual(classify_scene(shop, 1280, 720).scene, "unknown")

    def test_confirmation_overlay_does_not_return_underlying_pause(self):
        report = {"lines": report_for("pause")["lines"] + report_for("restart_confirm")["lines"]}
        self.assertEqual(classify_scene(report).scene, "restart_confirm")

    def test_selected_button_unknown_and_ambiguity(self):
        candidates = BUTTONS["level_up"]
        result = selected_button(pixels((candidates["choose_1"],)), 1920, 1080, scene="level_up")
        self.assertEqual(result.selected_id, "choose_1")
        self.assertEqual(result.rect, candidates["choose_1"])
        result = selected_button(pixels((candidates["choose_1"], candidates["choose_2"])), 1920, 1080,
                                 scene="level_up")
        self.assertEqual(result.reason, "ambiguous_selection")
        self.assertIsNone(selected_button(pixels(), 1920, 1080, scene="level_up").selected_id)

    def test_unverified_size_scene_or_confirm_geometry_never_guessed(self):
        frame = pixels()
        for width, height, scene in ((960, 540, "shop"), (1920, 1080, "combat")):
            self.assertIsNone(selected_button(frame, width, height, scene=scene).selected_id)
        self.assertEqual(selected_button(frame, 1920, 1080, scene="restart_confirm").reason,
                         "explicit_candidates_required")
        self.assertIsNone(selected_button(bytearray(frame), 1920, 1080, scene="shop").selected_id)
        with self.assertRaises(ValueError):
            selected_button(frame, 1920, 1080, scene="shop", candidates={"bad": (-1, 0, 20, 20)})

    def test_navigation_is_one_arrow_only_and_overlapping_abstains(self):
        center = (400, 400, 600, 500)
        for target, key in (((700, 400, 900, 500), "right"), ((100, 400, 300, 500), "left"),
                            ((400, 550, 600, 650), "down"), ((400, 250, 600, 350), "up")):
            self.assertEqual(navigation_key(center, target), key)
        self.assertIsNone(navigation_key(center, center))
        self.assertIsNone(navigation_key(center, (450, 450, 650, 550)))

    def test_level_up_reroll_focus_routes_to_card_row_without_guessing_card(self):
        reroll = BUTTONS["level_up"]["reroll_focus"]
        result = selected_button(pixels((reroll,)), 1920, 1080, scene="level_up")
        self.assertEqual(result.selected_id, "reroll_focus")
        for index in range(4):
            target = BUTTONS["level_up"][f"choose_{index}"]
            self.assertEqual(navigation_key(result.rect, target, scene="level_up"), "up")
        # The next key depends on a newly observed card, not an assumed landing.
        first = BUTTONS["level_up"]["choose_0"]
        last = BUTTONS["level_up"]["choose_3"]
        self.assertEqual(navigation_key(first, last, scene="level_up"), "right")
        self.assertEqual(navigation_key(last, first, scene="level_up"), "left")
        ambiguous = selected_button(pixels((reroll, first)), 1920, 1080, scene="level_up")
        self.assertEqual(ambiguous.reason, "ambiguous_selection")
        self.assertIsNone(ambiguous.selected_id)

    def test_actual_failed_level_up_focus_is_reroll_not_a_card(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        root = (Path(__file__).resolve().parents[1] /
                "artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/"
                "training-d313da35b97c43c393eec8be1615ffc3/segments/"
                "20260927T070106Z-7a97133d/menus")
        names = ("20260927T070143Z-69a1c07e", "20260927T070143Z-713eba62")
        if not all((root / name / "frame.png").is_file() for name in names):
            self.skipTest("private failed-run calibration evidence absent")
        for name in names:
            directory = root / name
            raw = json.loads((directory / "ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(classify_scene(raw).scene, "level_up")
            selection = selected_button(*read_diagnostic_png(directory / "frame.png"), scene="level_up")
            self.assertEqual(selection.selected_id, "reroll_focus", selection)
            self.assertTrue(all(ratio < .5 for key, ratio in selection.ratios if key.startswith("choose_")))
            self.assertEqual(navigation_key(selection.rect, BUTTONS["level_up"]["choose_2"], scene="level_up"), "up")

    def test_private_real_frames_detect_expected_highlights(self):
        # Read only this project's diagnostic PNG; no private game images ship.
        from playmodel.games.brotato.capture import read_diagnostic_png
        root = Path(__file__).resolve().parents[1] / "data/raw"
        cases = (("brotato-observations/20260926T165020Z-6aea8781", "level_up", "choose_1"),
                 ("brotato-observations/20260926T165042Z-8e5c9b35", "shop", "buy_1"),
                 ("brotato-observations/20260926T165526Z-74075559", "shop", "buy_1"),
                 ("brotato-pilot-calibration/20260926T164826Z-d99c753a", "loot", "take"),
                 ("brotato-observations/20260926T165232Z-86ce921c", "pause", "continue"))
        if not all((root / name / "frame.png").exists() for name, _, _ in cases):
            self.skipTest("private game calibration frames are excluded from Git")
        for name, scene, expected in cases:
            with self.subTest(scene=scene):
                image, width, height = read_diagnostic_png(root / name / "frame.png")
                result = selected_button(image, width, height, scene=scene)
                self.assertEqual(result.selected_id, expected, result)


if __name__ == "__main__":
    unittest.main()
