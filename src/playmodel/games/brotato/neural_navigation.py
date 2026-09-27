"""Fresh pixel arrows for a frozen macro. Never authorizes Enter or spending."""
from dataclasses import dataclass
import hashlib

from .menu import BUTTONS, navigation_key, selected_button, selected_stat_row


@dataclass(frozen=True)
class NeuralArrow:
    scene: str
    selected: str
    target: str
    key: str
    decision_id: str
    model_label: str = 'fresh_pixels_frozen_macro_navigation'


class FrozenNavigation:
    def __init__(self):
        self.anchor = None
        self.last_sent = 0
        self.steps = 0

    @staticmethod
    def digest(pixels, scene):
        if len(pixels) != 1920 * 1080 * 4:
            return None
        # Level-up content stays exact; only Choose focus and idle character
        # rendering are excluded. Shop departure arrows do NOT verify offers
        # or prices: full fresh OCR and macro verification remain mandatory.
        rectangles = ((0, 0, 400, 260), (400, 200, 1500, 375),
                      (30, 380, 1500, 632), (1515, 200, 1900, 980),
                      (522, 756, 1009, 822)) if scene == 'level_up' else (
                      (0, 0, 1100, 125), (0, 790, 1445, 1065))
        digest = hashlib.sha256()
        view = memoryview(pixels)
        for left, top, right, bottom in rectangles:
            for y in range(top, bottom):
                digest.update(view[(y*1920+left)*4:(y*1920+right)*4])
        return digest.hexdigest()

    def arm(self, decision, shot, pixels, candidates, now_ns):
        scene, target = decision.observation.scene, decision.target
        self.anchor = None
        if scene != 'level_up' and (scene != 'shop' or target != 'depart'):
            return
        observed = shot['capture_started_at_ns']
        if not 0 <= now_ns-observed <= 750_000_000:
            return
        self.anchor = (decision.decision_id, scene, target, shot['hwnd'], observed,
                       self.digest(pixels, scene), dict(candidates))
        self.last_sent, self.steps = observed, 0

    def propose(self, decision, shot, pixels, now_ns):
        if self.anchor is None or decision is None:
            return None
        identity, scene, target, hwnd, anchored, digest, candidates = self.anchor
        observed = shot['capture_started_at_ns']
        if (decision.decision_id != identity or decision.target != target
                or decision.observation.scene != scene or shot['hwnd'] != hwnd
                or not self.last_sent < observed <= now_ns
                or now_ns-observed > 500_000_000 or now_ns-anchored > 3_000_000_000
                or self.steps >= 8 or self.digest(pixels, scene) != digest):
            return None
        focus = selected_button(pixels, 1920, 1080, scene=scene, candidates=candidates)
        if scene == 'shop' and focus.reason == 'no_selected_candidate':
            focus = selected_stat_row(pixels, 1920, 1080, scene=scene)
        if focus.selected_id is None or focus.selected_id == target:
            return None  # Reaching target only requests full OCR, never Enter.
        key = navigation_key(focus.rect, candidates[target], scene=scene)
        if key not in ('left', 'right', 'up', 'down'):
            return None
        return NeuralArrow(scene, focus.selected_id, target, key, identity)

    def sent(self, at_ns):
        self.last_sent = at_ns
        self.steps += 1
