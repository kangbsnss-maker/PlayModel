"""English/Chinese menu calibration. No input, purchases, OCR calls or training.

Callers bind OCR and pixels to the same fresh frame and reobserve after EVERY
key. Selection color is a focus hypothesis, not permission or applied action.
"""
from dataclasses import dataclass
import math
from typing import Mapping
import unicodedata

from .ocr import rows_in_region


Rect = tuple[int, int, int, int]
SCENES = ("shop", "pause", "loot", "level_up", "restart_confirm", "death", "result", "difficulty")
BUTTONS: dict[str, dict[str, Rect]] = {
    'death': {'ok':(660,654,1260,720)},
    'result': {'restart':(335,983,635,1049), 'new_run':(660,983,960,1049)},
    'difficulty': {f'danger_{i}':(599+i*106,723,695+i*106,820) for i in range(7)},
    "level_up": {**{f"choose_{i}": (left, 635, left + 321, 687)
                    for i, left in enumerate((52, 421, 790, 1159))},
                 # Observed keyboard focus can enter the reroll button after
                 # combat. This is a focus landmark, not a learned action.
                 "reroll_focus": (522, 756, 1009, 822)},
    "loot": {"take": (435, 676, 1035, 741), "recycle": (435, 751, 1035, 816),
             "ban": (435, 825, 1035, 890)},
    "pause": {name: (50, top, 650, top + 65) for name, top in
              zip(("continue", "restart", "end_run", "catalog", "settings", "main_menu"),
                  (244, 334, 424, 514, 604, 694))},
    "shop": {**{f'owned_{i}':(1131+106*i,845,1227+106*i,941) for i in range(3)},
             "buy_0": (122, 541, 278, 609), "buy_1": (486, 541, 638, 609),
             "buy_2": (845, 541, 1000, 609), "buy_3": (1208, 541, 1360, 609),
             "lock_0": (49, 634, 352, 686), "lock_1": (410, 634, 713, 686),
             "lock_2": (771, 634, 1074, 686), "lock_3": (1132, 634, 1435, 686),
             "ban_2": (771, 694, 1074, 741), "ban_3": (1132, 694, 1435, 741),
             "refresh": (1162, 25, 1459, 107), "depart": (1484, 823, 1884, 889)},
}
# Classification is based on separately placed anchors, not whole-screen text.
ANCHORS = {
    "shop": (("商店", (0, 0, 400, 125)), ("出发", (1450, 780, 1910, 940))),
    "pause": (("继续", (40, 220, 670, 325)), ("重新开始", (40, 325, 670, 410))),
    "loot": (("发现道具", (400, 150, 1050, 300)), ("回收", (400, 735, 1070, 825))),
    "level_up": (("升级", (450, 200, 1050, 350)), ("选择", (20, 610, 1510, 710))),
    "restart_confirm": (("是否重新开始", (400, 200, 1520, 850)),),
}
ENGLISH_ANCHORS = {
    'death': (('RunLost',(600,130,1300,270)),('Killedby',(760,330,1150,460))),
    'result': (('Stats',(80,100,480,190)),('NewRun',(650,970,980,1060)),('Restart',(330,970,650,1060))),
    'difficulty': (('DifficuItyselection',(600,60,1400,170)),('Difficulty',(1350,180,1770,320))),
    'shop': (('Shop', (0, 0, 400, 125)), ('Go', (1450, 780, 1910, 1070))),
    'pause': (('Resume', (40, 220, 670, 325)), ('Restart', (40, 325, 670, 410))),
    'loot': (('ItemFound', (400, 150, 1050, 300)), ('Recycle', (400, 735, 1070, 825))),
    'level_up': (('LevelUp', (450, 200, 1050, 350)), ('Choose', (20, 610, 1510, 710))),
    'restart_confirm': (('Restarttherun', (400, 200, 1520, 850)),),
}
# WinRT zh-Hans-CN sometimes reads this English font's lowercase l as I.
# Measured variants stay local to each scene's independently positioned anchors.
ENGLISH_OCR_VARIANTS = {
    # Shop pause disables Restart; require two other positioned labels.
    'pause': (('Resume', (40,220,670,325)), ('Endtherun', (40,410,670,500)),
              ('Options', (40,590,670,680))),
    'death': (('RunLost', (600,130,1300,270)), ('KiIIedby', (760,330,1150,460))),
    # Three startup captures 20260927T081913Z..081914Z: literal OCR spellings.
    'loot': (('ltemFound', (400,150,1050,300)), ('RecycIe', (400,735,1070,825))),
    # Captured 20260927T060828Z-9e81769c: LeveI up! with four Choose cards.
    'level_up': (('LeveIUp', (450,200,1050,350)), ('Choose', (20,610,1510,710))),
    # Captured 20260927T012159Z: both difficulty labels use uppercase I.
    'difficulty': (('DifficuItyselection', (600,60,1400,170)), ('DifficuIty', (1350,180,1770,320))),
}


@dataclass(frozen=True)
class MenuScene:
    scene: str
    reason: str
    matched_anchors: tuple[str, ...] = ()


def classify_scene(report: dict, width: int = 1920, height: int = 1080) -> MenuScene:
    if type(width) is not int or type(height) is not int or (width, height) != (1920, 1080):
        return MenuScene("unknown", "uncalibrated_dimensions")
    try:
        # Reject malformed/out-of-frame OCR geometry rather than guessing a ROI.
        if not isinstance(report, dict) or not isinstance(report.get("lines"), list):
            raise ValueError("missing OCR lines")
        for line in report["lines"]:
            for word in line["words"]:
                if type(word["text"]) is not str:
                    raise ValueError("invalid word")
                x, y, w, h = (word[key] for key in ("x", "y", "width", "height"))
                if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y, w, h))
                        or not (0 <= x < x + w <= width and 0 <= y < y + h <= height)):
                    raise ValueError("invalid OCR geometry")
        matches = {}
        for scene, anchors in list(ANCHORS.items()) + list(ENGLISH_ANCHORS.items()) + list(ENGLISH_OCR_VARIANTS.items()):
            found = tuple(token for token, region in anchors
                          if any(token.casefold() in "".join(unicodedata.normalize("NFKC", row).split()).casefold()
                                 for row in rows_in_region(report, region)))
            if len(found) == len(anchors):
                matches[scene] = found
        # A visible confirmation question supersedes the underlying pause page.
        if "restart_confirm" in matches:
            return MenuScene("restart_confirm", "question_overlay", matches["restart_confirm"])
        if len(matches) == 1:
            scene = next(iter(matches))
            return MenuScene(scene, "positioned_anchors", matches[scene])
        return MenuScene("unknown", "ambiguous_anchors" if matches else "anchors_missing")
    except (KeyError, TypeError, ValueError, IndexError):
        return MenuScene("unknown", "invalid_ocr")


def _rect(rect: Rect, width: int = 1920, height: int = 1080) -> None:
    if (type(rect) is not tuple or len(rect) != 4 or any(type(v) is not int for v in rect)
            or not (0 <= rect[0] < rect[2] <= width and 0 <= rect[1] < rect[3] <= height)):
        raise ValueError("invalid calibrated button rectangle")


@dataclass(frozen=True)
class ButtonSelection:
    selected_id: str | None
    rect: Rect | None
    reason: str
    ratios: tuple[tuple[str, float], ...] = ()


def selected_stat_row(bgra: bytes, width: int = 1920, height: int = 1080, *,
                      scene: str = 'shop') -> ButtonSelection:
    """Calibrated thin keyboard-focus border on a positively classified menu."""
    if (width,height)!=(1920,1080) or len(bgra)!=width*height*4:
        return ButtonSelection(None,None,'invalid_frame')
    regions = {'shop': (1498, 165, 1870, 780, 1515, 1850),
               # Actual loot Stats panel, independently positioned from shop.
               'loot': (1115, 380, 1470, 975, 1128, 1455)}
    if scene not in regions:
        return ButtonSelection(None,None,'uncalibrated_stats_scene')
    left, top_limit, right, bottom_limit, sample_left, sample_right = regions[scene]
    lines=[]
    for y in range(top_limit,bottom_limit):
        samples=[bgra[(y*width+x)*4:(y*width+x)*4+3] for x in range(sample_left,sample_right,12)]
        if sum(max(c)>155 for c in samples)>=len(samples)*.9:
            if not lines or y-lines[-1]>2: lines.append(y)
    pairs=[(a,b) for a,b in zip(lines,lines[1:]) if 25<=b-a<=40]
    if len(pairs)!=1: return ButtonSelection(None,None,'stat_focus_unknown')
    top,bottom=pairs[0]
    return ButtonSelection(f'stat_{top}',(left,top,right,bottom),'calibrated_stat_border')


def selected_button(bgra: bytes, width: int, height: int, *, scene: str,
                    candidates: Mapping[str, Rect] | None = None) -> ButtonSelection:
    if (type(width) is not int or type(height) is not int or (width, height) != (1920, 1080)
            or not isinstance(bgra, bytes) or len(bgra) != width * height * 4):
        return ButtonSelection(None, None, "invalid_or_uncalibrated_frame")
    if scene not in SCENES:
        return ButtonSelection(None, None, "unknown_scene")
    candidates = dict(BUTTONS.get(scene, {})) if candidates is None else dict(candidates)
    if not candidates:
        return ButtonSelection(None, None, "explicit_candidates_required")
    ratios = []
    for name, rect in candidates.items():
        if type(name) is not str or not name.strip():
            raise ValueError("candidate ID required")
        _rect(rect, width, height)
        left, top, right, bottom = rect
        # Include the button's flat gray background around price digits/icons.
        # A center-only crop can be mostly black text or a green currency icon.
        inset_x, inset_y = (right - left) // 10, (bottom - top) // 10
        bright = total = 0
        for y in range(top + inset_y, bottom - inset_y, 2):
            for x in range(left + inset_x, right - inset_x, 2):
                channels = bgra[(y * width + x) * 4:(y * width + x) * 4 + 3]
                bright += 145 <= min(channels) <= max(channels) <= 245 and max(channels) - min(channels) <= 18
                total += 1
        ratios.append((name, bright / total))
    ranked = sorted(ratios, key=lambda item: item[1], reverse=True)
    if ranked[0][1] < 0.50:
        return ButtonSelection(None, None, "no_selected_candidate", tuple(ratios))
    if len(ranked) > 1 and (ranked[1][1] >= 0.50 or ranked[0][1] - ranked[1][1] < 0.15):
        return ButtonSelection(None, None, "ambiguous_selection", tuple(ratios))
    name = ranked[0][0]
    return ButtonSelection(name, candidates[name], "calibrated_highlight", tuple(ratios))


def navigation_key(current_rect: Rect, target_rect: Rect, *, scene: str | None = None) -> str | None:
    """One geometric direction hint, never Enter or a promised UI focus path.

    Same/overlapping rectangles abstain. Reclassify and redetect selection after
    one key; Godot focus neighbors may differ from geometric nearest neighbors.
    """
    _rect(current_rect)
    _rect(target_rect)
    if (scene=='shop' and current_rect in (BUTTONS['shop'][f'owned_{i}'] for i in range(3))
            and target_rect[0]>=1480 and target_rect[1]>=820):
        # Human-reported route. Observe every transition; no blind multi-key send.
        return 'right'
    if (scene == "level_up" and current_rect == BUTTONS["level_up"]["reroll_focus"]
            and target_rect in (BUTTONS["level_up"][f"choose_{index}"] for index in range(4))):
        # First move geometrically toward the card ROW above the observed
        # reroll focus. Center-distance directly to an outer card would choose
        # a horizontal key while still below the row. The caller must observe
        # which card actually receives focus before any further arrow/Enter.
        cards = [BUTTONS["level_up"][f"choose_{index}"] for index in range(4)]
        target_rect = (min(rect[0] for rect in cards), min(rect[1] for rect in cards),
                       max(rect[2] for rect in cards), max(rect[3] for rect in cards))
    if (max(current_rect[0], target_rect[0]) < min(current_rect[2], target_rect[2])
            and max(current_rect[1], target_rect[1]) < min(current_rect[3], target_rect[3])):
        return None
    dx = (target_rect[0] + target_rect[2] - current_rect[0] - current_rect[2]) / 2
    dy = (target_rect[1] + target_rect[3] - current_rect[1] - current_rect[3]) / 2
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left" if dx < 0 else None
    return "down" if dy > 0 else "up"
