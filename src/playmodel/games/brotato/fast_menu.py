"""Learned menu perception may accelerate navigation, never confirm a choice."""
from dataclasses import dataclass

from .menu import BUTTONS, navigation_key, selected_button


@dataclass(frozen=True)
class FastNavigation:
    scene: str
    selected: str
    target: str
    key: str
    model_label: str


def fast_navigation(model, pixels: bytes, width: int, height: int, *, game_build_id: str):
    """Only static navigation, cross-checked against current-frame geometry.

    Enter, purchases, rewards, changing card text and result actions require
    fresh OCR. Learned scene/focus does not make the fixed target learned.
    """
    if (width, height) != (1920, 1080):
        return None
    prediction = model.predict(pixels, width, height, game_build_id=game_build_id, language='en')
    if prediction.abstain_reason:
        return None
    targets = {'pause': 'continue', 'difficulty': 'danger_6'}
    scene, focus = prediction.scene, prediction.focus
    target = targets.get(scene)
    if target is None or focus == target or focus not in BUTTONS[scene]:
        return None
    if scene == 'difficulty':
        from .setup_run import focused_tile
        boxes = {i: BUTTONS['difficulty'][f'danger_{i}'] for i in range(7)}
        index = focused_tile(pixels, width, boxes)
        actual = f'danger_{index}' if index is not None else None
    else:
        actual = selected_button(pixels, width, height, scene=scene).selected_id
    if actual != focus:
        return None
    key = navigation_key(BUTTONS[scene][focus], BUTTONS[scene][target])
    if key not in ('left', 'right', 'up', 'down'):
        return None
    return FastNavigation(scene, focus, target, key, prediction.label)
