"""Calibrated menu-scroll geometry; no OCR, game input, or learned weights.

This observes a known scrollbar, not arbitrary scrolling or game-camera motion.
The caller must independently identify the menu/viewport. Unknown geometry and
unchanged pixels cannot establish that navigation reached the end of a list.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ScrollbarCalibration:
    scene_id: str
    width: int
    height: int
    track: tuple[int, int, int, int]  # half-open left, top, right, bottom
    calibration_id: str
    edge_tolerance_px: int = 1
    near_edge_px: int = 6
    minimum_thumb_px: int = 12

    def __post_init__(self):
        if not all(type(v) is str and v.strip() for v in (self.scene_id, self.calibration_id)):
            raise ValueError("scene and calibration IDs are required")
        if any(type(v) is not int or v < 1 for v in (self.width, self.height, self.minimum_thumb_px)):
            raise ValueError("positive integer dimensions are required")
        if (type(self.track) is not tuple or len(self.track) != 4
                or any(type(v) is not int for v in self.track)):
            raise ValueError("track must be an immutable integer rectangle")
        left, top, right, bottom = self.track
        if not 0 <= left < right <= self.width or not 0 <= top < bottom <= self.height:
            raise ValueError("track is outside frame")
        if (type(self.edge_tolerance_px) is not int or type(self.near_edge_px) is not int
                or not 0 <= self.edge_tolerance_px < self.near_edge_px
                or bottom - top < self.minimum_thumb_px + 2 * self.near_edge_px):
            raise ValueError("invalid scrollbar tolerances")


# Two local 1920x1080 captures, Brotato 1.1.15.4, Chinese accessibility tab.
# Different resolutions, tabs, themes and layouts need their own calibration.
ACCESSIBILITY_1080 = ScrollbarCalibration(
    "settings.accessibility", 1920, 1080, (1450, 223, 1460, 980),
    "brotato-1.1.15.4-accessibility-1920x1080-v1",
)


@dataclass(frozen=True)
class ScrollObservation:
    status: str
    reason: str
    scene_id: str
    calibration_id: str
    observed_at_ns: int | None
    frame_sha256: str | None = None
    thumb: tuple[int, int] | None = None  # half-open top, bottom
    position: float | None = None  # thumb travel fraction, not selected item
    can_scroll_up: bool | None = None
    can_scroll_down: bool | None = None
    edge: str = "unknown"


def detect_scrollbar(bgra: bytes, width: int, height: int, *, scene_id: str,
                     observed_at_ns: int,
                     calibration: ScrollbarCalibration = ACCESSIBILITY_1080) -> ScrollObservation:
    """Read a light neutral thumb on a dark neutral track, failing closed.

    Alpha is ignored because PrintWindow BGRA need not carry meaningful alpha.
    A bright full-height strip is ambiguous (no overflow, occlusion or wrong
    region), so it is NOT evidence that the whole menu has been visited.
    """
    if not isinstance(calibration, ScrollbarCalibration):
        raise ValueError("explicit scrollbar calibration required")

    def unknown(reason):
        return ScrollObservation("unknown", reason, scene_id if type(scene_id) is str else "",
                                 calibration.calibration_id,
                                 observed_at_ns if type(observed_at_ns) is int and observed_at_ns >= 0 else None)

    if (type(width) is not int or type(height) is not int
            or type(observed_at_ns) is not int or observed_at_ns < 0
            or not isinstance(bgra, bytes) or len(bgra) != width * height * 4):
        return unknown("invalid_frame")
    if scene_id != calibration.scene_id:
        return unknown("unverified_or_changed_menu")
    if (width, height) != (calibration.width, calibration.height):
        return unknown("uncalibrated_dimensions")
    left, top, right, bottom = calibration.track
    row_states = []
    for y in range(top, bottom):
        bright = dark = 0
        for x in range(left, right):
            offset = (y * width + x) * 4
            channels = bgra[offset:offset + 3]
            neutral = max(channels) - min(channels) <= 16
            bright += neutral and 145 <= min(channels) and max(channels) <= 245
            dark += neutral and max(channels) <= 65
        required = (4 * (right - left) + 4) // 5  # >=80% across each row
        row_states.append(1 if bright >= required else 0 if dark >= required else -1)
    if -1 in row_states:
        return unknown("track_occluded_or_appearance_changed")
    runs = []
    start = None
    for index, bright in enumerate((*row_states, 0)):
        if bright and start is None:
            start = index
        elif not bright and start is not None:
            runs.append((top + start, top + index))
            start = None
    if len(runs) != 1:
        return unknown("missing_or_multiple_thumbs")
    thumb_top, thumb_bottom = runs[0]
    thumb_height = thumb_bottom - thumb_top
    travel = bottom - top - thumb_height
    if thumb_height < calibration.minimum_thumb_px or travel <= 2 * calibration.near_edge_px:
        return unknown("ambiguous_thumb_extent")
    above, below = thumb_top - top, bottom - thumb_bottom

    def more(gap):
        if gap <= calibration.edge_tolerance_px:
            return False
        return True if gap > calibration.near_edge_px else None

    can_up, can_down = more(above), more(below)
    edge = ("top" if can_up is False else "bottom" if can_down is False
            else "near_top" if can_up is None else "near_bottom" if can_down is None else "middle")
    return ScrollObservation("observed", "calibrated_geometry", scene_id, calibration.calibration_id,
                             observed_at_ns, hashlib.sha256(bgra).hexdigest(), runs[0], above / travel,
                             can_up, can_down, edge)


@dataclass(frozen=True)
class ScrollProgress:
    motion: str  # up, down, unchanged, unknown
    boundary_confirmed: str | None
    no_scroll_count: int
    stop_navigation: bool
    reason: str


class MenuScrollTracker:
    """Track one menu instance; the caller gives a new ID on each menu entry.

    A request is intent, not proof that input reached the game. Repeated no
    scroll may mean selection moved inside the viewport, a lost input or a
    stale render. Stop for inspection without relabelling it as a list end.
    Fresh capture timestamps also do not prove a newly rendered game frame.
    """

    def __init__(self, *, unchanged_limit: int = 3):
        if type(unchanged_limit) is not int or unchanged_limit < 2:
            raise ValueError("unchanged_limit must be an integer >=2")
        self.unchanged_limit = unchanged_limit
        self.reset()

    def reset(self):
        self._previous = None
        self._instance = None
        self._direction = None
        self._unchanged = 0
        self._boundary = None
        self._boundary_count = 0

    def observe(self, observation: ScrollObservation, *, menu_instance: str,
                requested_direction: str | None = None) -> ScrollProgress:
        if type(menu_instance) is not str or not menu_instance.strip():
            raise ValueError("a menu instance ID is required")
        if requested_direction not in (None, "up", "down"):
            raise ValueError("requested_direction must be up, down or None")
        if not isinstance(observation, ScrollObservation):
            raise ValueError("expected ScrollObservation")
        if observation.status != "observed" or observation.thumb is None or observation.observed_at_ns is None:
            self.reset()
            return ScrollProgress("unknown", None, 0, True, "unknown_geometry")
        old = self._previous
        if (menu_instance != self._instance or (old is not None and
                (old.calibration_id != observation.calibration_id or old.scene_id != observation.scene_id))):
            self.reset()
            old = None
        if old is not None and observation.observed_at_ns <= old.observed_at_ns:
            return ScrollProgress("unknown", None, self._unchanged, True, "duplicate_or_out_of_order")
        motion = "unknown"
        if old is not None:
            old_height = old.thumb[1] - old.thumb[0]
            new_height = observation.thumb[1] - observation.thumb[0]
            if abs(new_height - old_height) > 1:
                self.reset()
                return ScrollProgress("unknown", None, 0, True, "content_extent_changed")
            delta = observation.thumb[0] - old.thumb[0]
            motion = "down" if delta > 1 else "up" if delta < -1 else "unchanged"
        same_request = requested_direction is not None and requested_direction == self._direction
        self._unchanged = self._unchanged + 1 if same_request and motion == "unchanged" else 0
        boundary = observation.edge if observation.edge in ("top", "bottom") else None
        self._boundary_count = self._boundary_count + 1 if boundary and boundary == self._boundary else 1 if boundary else 0
        self._boundary = boundary
        self._previous, self._instance, self._direction = observation, menu_instance, requested_direction
        confirmed = boundary if self._boundary_count >= 2 else None
        remaining = (observation.can_scroll_up if requested_direction == "up"
                     else observation.can_scroll_down if requested_direction == "down" else True)
        if requested_direction is not None and remaining is False:
            return ScrollProgress(motion, confirmed, self._unchanged, True, "visual_boundary")
        if requested_direction is not None and remaining is None:
            return ScrollProgress(motion, confirmed, self._unchanged, True, "near_boundary_unconfirmed")
        if requested_direction and motion in ("up", "down") and motion != requested_direction:
            return ScrollProgress(motion, confirmed, self._unchanged, True, "opposite_motion")
        if self._unchanged >= self.unchanged_limit:
            return ScrollProgress(motion, confirmed, self._unchanged, True, "stalled_unknown")
        return ScrollProgress(motion, confirmed, self._unchanged, False, "observed")
