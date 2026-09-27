"""Small, local-only Brotato color/geometry baseline; no trained detector.

Coordinates and navigation are normalized. Enemy/projectile identities are
unknown: ``hazards`` are unverified high-contrast candidates, never labels for
training or independently measured damage/death. This module cannot send input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


EXTRACTOR_VERSION = "brotato-color-geometry-v2"


@dataclass(frozen=True)
class VisualCandidate:
    x: float
    y: float
    radius: float
    confidence: float
    label: str
    evidence: str


@dataclass(frozen=True)
class VisionObservation:
    navigation: tuple[float, float, bool] = (0.0, 0.0, False)
    player: tuple[float, float] | None = None
    player_confidence: float = 0.0
    pickups: tuple[VisualCandidate, ...] = ()
    hazards: tuple[VisualCandidate, ...] = ()
    hp_fill_fraction: float | None = None
    combat_likely: bool = False
    status: str = "unknown"
    reasons: tuple[str, ...] = ()
    observed_at_ns: int | None = None
    extractor_version: str = EXTRACTOR_VERSION
    camera_status: str = "unknown"
    camera_shift: tuple[float, float] | None = None
    camera_confidence: float = 0.0


@dataclass(frozen=True)
class _Component:
    x: float
    y: float
    count: int
    width: float
    height: float

    def radius(self, aspect_ratio):
        return math.hypot(self.width * aspect_ratio, self.height) / 2


def _components(points: set[int], width: int, height: int) -> list[_Component]:
    """Eight-connected components on the small sampled grid."""
    found = []
    while points:
        seed = points.pop()
        stack = [seed]
        sx = sy = count = 0
        min_x = max_x = seed % width
        min_y = max_y = seed // width
        while stack:
            index = stack.pop()
            x, y = index % width, index // width
            sx += x
            sy += y
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = ny * width + nx
                    if neighbor in points:
                        points.remove(neighbor)
                        stack.append(neighbor)
        found.append(_Component((sx / count + 0.5) / width,
                                (sy / count + 0.5) / height, count,
                                (max_x - min_x + 1) / width,
                                (max_y - min_y + 1) / height))
    return found


def _hud_mask(x: float, y: float) -> bool:
    # Health, level, materials and the center wave/timer, not the whole top edge.
    # The thin top band also excludes clipped parachute highlights at spawn.
    return y < 0.03 or (x < 0.195 and y < 0.20) or (0.47 < x < 0.63 and y < 0.155)


def _unit(x: float, y: float) -> tuple[float, float]:
    magnitude = math.hypot(x, y)
    return (x / magnitude, y / magnitude) if magnitude > 1e-9 else (0.0, 0.0)


def _merge_white_fragments(components: list[_Component]) -> list[_Component]:
    # Black eyes/mouth can split one white face at stride 12. Join vertically
    # adjacent aligned pieces, but not distant damage numerals or other sprites.
    merged = []
    for item in sorted(components, key=lambda part: part.y):
        for index, previous in enumerate(merged):
            combined_width = max(previous.x + previous.width / 2, item.x + item.width / 2) - min(
                previous.x - previous.width / 2, item.x - item.width / 2)
            combined_height = max(previous.y + previous.height / 2, item.y + item.height / 2) - min(
                previous.y - previous.height / 2, item.y - item.height / 2)
            vertical_gap = abs(previous.y - item.y) - (previous.height + item.height) / 2
            if (abs(previous.x - item.x) <= 0.01 and vertical_gap <= 0.015
                    and combined_width <= 0.04 and combined_height <= 0.075):
                count = previous.count + item.count
                merged[index] = _Component(
                    (previous.x * previous.count + item.x * item.count) / count,
                    (previous.y * previous.count + item.y * item.count) / count,
                    count, combined_width, combined_height)
                break
        else:
            merged.append(item)
    return merged


def plan_navigation(player: tuple[float, float] | None,
                    pickups: tuple[VisualCandidate, ...] = (),
                    hazards: tuple[VisualCandidate, ...] = (),
                    *, aspect_ratio: float = 16 / 9, avoidance_radius: float = .17) -> tuple[float, float, bool]:
    """Experimental steering, with close hazards ahead of resource attraction.

    Radii/distances use screen-height units after correcting horizontal aspect.
    The arena edge is a screen-edge proxy, not an inferred world/map boundary.
    """
    if (not isinstance(player, tuple) or len(player) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not 0 <= v <= 1 for v in player)
            or type(aspect_ratio) not in (int, float)
            or not math.isfinite(aspect_ratio) or aspect_ratio <= 0):
        return (0.0, 0.0, False)
    for candidate in (*pickups, *hazards):
        if (not isinstance(candidate, VisualCandidate)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in (candidate.x, candidate.y, candidate.radius,
                                 candidate.confidence))
                or not (0 <= candidate.x <= 1 and 0 <= candidate.y <= 1
                        and 0 <= candidate.radius <= 1
                        and 0 <= candidate.confidence <= 1)):
            return (0.0, 0.0, False)
    px, py = player
    threats = []
    for candidate in hazards:
        if candidate.confidence <= 0:
            continue
        dx, dy = (px - candidate.x) * aspect_ratio, py - candidate.y
        distance = math.hypot(dx, dy)
        if distance < avoidance_radius + candidate.radius:
            threats.append((distance - candidate.radius, dx, dy, candidate))
    edge_x = max(0.0, (0.065 - px) / 0.065) - max(0.0, (px - 0.935) / 0.065)
    edge_y = max(0.0, (0.08 - py) / 0.08) - max(0.0, (py - 0.92) / 0.08)
    if threats:
        threats.sort(key=lambda item: item[0])
        steer_x = steer_y = 0.0
        for gap, dx, dy, candidate in threats[:12]:
            if math.hypot(dx, dy) < 1e-9:
                # An overlapping candidate has no trustworthy escape bearing.
                continue
            ux, uy = _unit(dx, dy)
            weight = candidate.confidence / max(0.015, gap + 0.03) ** 2
            steer_x += ux * weight
            steer_y += uy * weight
        steer_x, steer_y = _unit(steer_x, steer_y)
        if steer_x == steer_y == 0:
            _, dx, dy, _ = threats[0]
            steer_x, steer_y = _unit(dx, dy)
            if steer_x == steer_y == 0:
                steer_x, steer_y = _unit((0.5 - px) * aspect_ratio, 0.5 - py)
            if steer_x == steer_y == 0:
                return (0.0, 0.0, False)
        # Do not let a pickup across the threat reverse the escape direction.
        steer_x += edge_x * 0.8
        steer_y += edge_y * 0.8
    elif edge_x or edge_y:
        steer_x, steer_y = edge_x, edge_y
    else:
        eligible = [candidate for candidate in pickups if candidate.confidence > 0]
        if eligible:
            target = min(eligible, key=lambda item:
                         math.hypot((item.x - px) * aspect_ratio, item.y - py)
                         / max(0.1, item.confidence))
            steer_x, steer_y = (target.x - px) * aspect_ratio, target.y - py
        else:
            # Screen-centered exploration is a baseline, not learned XP seeking.
            dx, dy = (px - 0.5) * aspect_ratio, py - 0.5
            steer_x, steer_y = -dy - dx * 0.7, dx - dy * 0.7
            if abs(steer_x) + abs(steer_y) < 0.01:
                steer_x = 1.0
    dx, dy = _unit(steer_x, steer_y)
    return (dx, dy, True)


class BrotatoVision:
    """Stateful color/component tracker for one episode's captured client.

    ``alpha_mode='ignore'`` matches GDI's unused alpha byte. Use ``'straight'``
    for actual BGRA with alpha; transparent pixels then provide no evidence.
    Timestamps, when supplied, must increase and use one monotonic clock. The
    caller still owns current-frame age checks and externally known UI state.
    """
    def __init__(self, *, avoidance_radius: float = .17):
        if type(avoidance_radius) not in (int, float) or not .10 <= avoidance_radius <= .35:
            raise ValueError('Invalid avoidance radius')
        self.avoidance_radius = avoidance_radius
        self.reset()

    def reset(self):
        self._previous_player = None
        self._previous_time = None
        self._previous_grid = None
        self._misses = 0

    def _invalid(self, status: str, observed_at_ns: int | None = None,
                 *, hp_fill_fraction: float | None = None,
                 combat_likely: bool = False) -> VisionObservation:
        self._misses += 1
        self._previous_grid = None
        if self._misses >= 3:
            self._previous_player = None
        return VisionObservation(status=status, reasons=(status,),
                                 observed_at_ns=observed_at_ns,
                                 hp_fill_fraction=hp_fill_fraction,
                                 combat_likely=combat_likely)

    def observe(self, bgra: bytes, width: int, height: int, *,
                observed_at_ns: int | None = None, paused: bool = False,
                alpha_mode: str = "ignore") -> VisionObservation:
        if (type(width) is not int or type(height) is not int
                or not 80 <= width <= 7680 or not 45 <= height <= 4320
                or not 1.2 <= width / height <= 2.5
                or not isinstance(bgra, (bytes, bytearray))
                or len(bgra) != width * height * 4
                or alpha_mode not in ("ignore", "straight")
                or type(paused) is not bool
                or (observed_at_ns is not None
                    and (type(observed_at_ns) is not int or observed_at_ns < 0))):
            return self._invalid("invalid_frame")
        if (observed_at_ns is not None and self._previous_time is not None
                and observed_at_ns <= self._previous_time):
            return self._invalid("non_increasing_frame_time", observed_at_ns)
        elapsed = ((observed_at_ns - self._previous_time) / 1e9
                   if observed_at_ns is not None and self._previous_time is not None else None)
        if elapsed is not None and elapsed > 0.5:
            self._previous_player = None
        self._previous_time = observed_at_ns
        if paused:
            self._previous_player = None
            return self._invalid("paused", observed_at_ns)

        def rgb(x, y):
            ix = max(0, min(width - 1, round(x * width)))
            iy = max(0, min(height - 1, round(y * height)))
            offset = (iy * width + ix) * 4
            if alpha_mode == "straight" and bgra[offset + 3] < 128:
                return None
            return bgra[offset + 2], bgra[offset + 1], bgra[offset]

        # Repeated dark card panels + right stats panel are a menu layout even
        # while the combat HUD remains visible behind a level-up overlay.
        panel_columns=0
        for x in (.025,.22,.41,.60):
            colors=[rgb(x,y) for y in (.42,.55)]
            panel_columns += all(c is not None and max(c)<42 for c in colors)
        stats=[rgb(.80,y) for y in (.23,.29,.34)]
        if panel_columns>=3 and all(c is not None and max(c)<42 for c in stats):
            return self._invalid('menu_panel_layout',observed_at_ns)
        hp, hud_ok = self._health_bar(rgb, width)
        if not hud_ok:
            return self._invalid("combat_hud_not_found", observed_at_ns)
        # A large dark central panel commonly means a menu/pause overlay. This
        # is intentionally a rejection heuristic, not a complete UI classifier.
        center = [rgb(x / 20, y / 20) for x in range(6, 15) for y in range(5, 16)]
        dark = sum(color is not None and max(color) < 42 for color in center)
        if dark > len(center) * 0.65:
            return self._invalid("possible_menu_overlay", observed_at_ns,
                                 hp_fill_fraction=hp)

        # Bound Python work to <=240x135 points (320x180 input becomes 160x90).
        step = max(1, math.ceil(width / 240), math.ceil(height / 135))
        gw, gh = math.ceil(width / step), math.ceil(height / step)
        whites, greens, contrasts = set(), set(), set()
        gray_grid = bytearray(gw * gh)
        opaque = brown = samples = 0
        x_samples = [(gx, x * 4) for gx, x in enumerate(range(0, width, step))]
        hud_x_samples = [(gx, offset) for gx, offset in x_samples
                         if (gx + 0.5) / gw >= 0.195]
        top_x_samples = [(gx, offset) for gx, offset in hud_x_samples
                         if not 0.47 < (gx + 0.5) / gw < 0.63]
        use_alpha = alpha_mode == "straight"
        for gy, y in enumerate(range(0, height, step)):
            yn = (gy + 0.5) / gh
            if yn < 0.03:
                continue
            row = y * width * 4
            row_x_samples = (top_x_samples if yn < 0.155 else
                             hud_x_samples if yn < 0.20 else x_samples)
            grid_row = gy * gw
            for gx, x_offset in row_x_samples:
                offset = row + x_offset
                samples += 1
                if use_alpha and bgra[offset + 3] < 128:
                    continue
                opaque += 1
                b, g, r = bgra[offset], bgra[offset + 1], bgra[offset + 2]
                index = grid_row + gx
                gray_grid[index] = (r + g + b) // 3
                # Both brown and desaturated blue/gray ground were observed in
                # local captures. Colored sprites are handled below separately.
                if (42 <= r <= 145 and
                        ((0 <= r - g <= 40 and 0 <= g - b <= 35)
                         or (abs(r - g) <= 24 and abs(g - b) <= 20))):
                    brown += 1
                    if r < 110:
                        continue
                if min(r, g, b) >= 202 and max(r, g, b) - min(r, g, b) <= 42:
                    whites.add(index)
                elif g >= 110 and g - r >= 28 and g - b >= 22:
                    greens.add(index)
                elif ((r >= 130 and r - g > 45 and r - b > 35)
                      or (r >= 140 and b >= 120 and g < min(r, b) * 0.72)
                      or (65 <= r <= 165 and 95 <= b <= 210 and b - g >= 30 and r - g >= 5)):
                    contrasts.add(index)
        if opaque < samples * 0.8 or brown < opaque * 0.25:
            return self._invalid("arena_background_unconfirmed", observed_at_ns,
                                 hp_fill_fraction=hp)
        # Reject overwhelmed color masks before connected-component traversal.
        if len(whites) > gw * gh * 0.06 or len(contrasts) > gw * gh * 0.16:
            return self._invalid("color_candidates_overwhelmed", observed_at_ns,
                                 hp_fill_fraction=hp)
        camera_status, camera_shift, camera_confidence = self._camera_motion(gray_grid, gw, gh)
        if camera_status == "shift_candidate":
            # Common translation cancels in same-frame relative navigation.
            # Discard prior position association, then require an independently
            # identifiable current player. No world velocity/event is inferred.
            self._previous_player = None
        white_components = _merge_white_fragments(_components(whites, gw, gh))
        player_components = [item for item in white_components
                             if 3 <= item.count <= 0.004 * gw * gh
                             and 0.008 <= item.width <= 0.05
                             and 0.012 <= item.height <= 0.085
                             and 0.25 <= item.width / item.height <= 2.3]
        previous = self._previous_player
        if previous is not None:
            max_shift = min(0.16, 0.04 + (elapsed if elapsed is not None else 0.05) * 1.2)
            player_components = [item for item in player_components
                                 if math.hypot((item.x - previous[0]) * width / height,
                                               item.y - previous[1]) <= max_shift]
            player_components.sort(key=lambda item:
                                   math.hypot((item.x - previous[0]) * width / height,
                                              item.y - previous[1]))
        else:
            # The initial parachute body is cream; the visible white face is
            # distinct in the inspected capture. No character-general claim.
            player_components.sort(key=lambda item: (-item.count, item.x, item.y))
        if not player_components:
            return self._invalid("player_unknown", observed_at_ns,
                                 hp_fill_fraction=hp, combat_likely=True)
        selected = player_components[0]
        if previous is None and len(player_components) > 1:
            runner_up = player_components[1]
            # At this sampling stride a two-pixel advantage can be a transient
            # level-up glyph, not a larger face. Do not cold-start from it.
            if (runner_up.count >= selected.count * 0.75
                    or selected.count - runner_up.count <= 2):
                return self._invalid("player_ambiguous", observed_at_ns,
                                     hp_fill_fraction=hp, combat_likely=True)
        player = (selected.x, selected.y)
        pickups = tuple(VisualCandidate(item.x, item.y, item.radius(width / height), 0.4,
                                        "pickup_candidate", "green_component_unverified")
                        for item in _components(greens, gw, gh)
                        if 1 <= item.count <= 0.003 * gw * gh
                        and item.width <= 0.045 and item.height <= 0.075)
        risks = _components(contrasts, gw, gh)
        # White/gray components include observed damage numerals and floor
        # decorations, so they are not promoted to hazard evidence.
        hazards = tuple(VisualCandidate(item.x, item.y, item.radius(width / height), 0.2,
                                        "unknown", "contrast_component_unverified")
                        for item in risks
                        if 2 <= item.count <= 0.01 * gw * gh
                        and item.width <= 0.1 and item.height <= 0.15
                        and math.hypot((item.x - player[0]) * width / height,
                                       item.y - player[1]) > 0.035 + selected.radius(width / height))
        if len(pickups) > 128 or len(hazards) > 96:
            return self._invalid("too_many_candidates", observed_at_ns,
                                 hp_fill_fraction=hp, combat_likely=True)
        self._previous_player = player
        self._misses = 0
        return VisionObservation(
            navigation=plan_navigation(player, pickups, hazards, aspect_ratio=width / height,
                                       avoidance_radius=self.avoidance_radius),
            player=player, player_confidence=0.65 if previous is not None else 0.45,
            pickups=pickups, hazards=hazards, hp_fill_fraction=hp,
            combat_likely=True, status="heuristic_observation",
            reasons=("player_color_unverified", "pickup_identity_unverified",
                     "enemy_and_projectile_identity_unknown",
                     "hp_flash_fill_unknown" if hp is None else "hp_color_fraction_not_ocr",
                     "screen_coordinates_not_world_motion"),
            observed_at_ns=observed_at_ns, camera_status=camera_status,
            camera_shift=camera_shift, camera_confidence=camera_confidence)

    def _camera_motion(self, current: bytearray, width: int, height: int):
        """Sparse common texture translation, never player/world velocity.

        A +/-2 sampled-pixel patch search cannot rule out slow/subpixel motion,
        zoom, large jumps or repeating texture. Its confidence is not calibrated.
        """
        previous = self._previous_grid
        self._previous_grid = (current, width, height)
        if previous is None or previous[1:] != (width, height):
            return "unknown", None, 0.0
        old = previous[0]
        patch = (-width, -1, 0, 1, width)
        anchors = []
        for y in range(max(4, height // 5), height - 4, 5):
            for x in range(4, width - 4, 5):
                index = y * width + x
                if not 25 <= old[index] <= 115:
                    continue
                contrast = max(abs(old[index] - old[index + delta]) for delta in patch)
                if contrast >= 12:
                    anchors.append((contrast, index, int(x >= width / 2) + 2 * int(y >= height / 2)))
        anchors.sort(reverse=True)
        anchors = anchors[:64]
        if len(anchors) < 12 or len({item[2] for item in anchors}) < 3:
            return "unknown", None, 0.0
        shifts = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]
        votes = {}
        for _, index, quadrant in anchors:
            scores = []
            for dx, dy in shifts:
                offset = dy * width + dx
                error = sum(abs(old[index + delta] - current[index + offset + delta])
                            for delta in patch)
                scores.append((error, dx, dy))
            scores.sort()
            best, runner_up = scores[:2]
            if best[0] <= 30 and runner_up[0] - best[0] >= 8:
                displacement = (best[1], best[2])
                votes.setdefault(displacement, []).append(quadrant)
        if not votes:
            return "unknown", None, 0.0
        shift, supporters = max(votes.items(), key=lambda item: len(item[1]))
        support = len(supporters) / len(anchors)
        if len(supporters) < 10 or support < 0.65 or len(set(supporters)) < 3:
            return "unknown", None, 0.0
        if shift == (0, 0):
            return "no_shift_detected", (0.0, 0.0), min(0.5, support * 0.5)
        return "shift_candidate", (shift[0] / width, shift[1] / height), min(0.5, support * 0.5)

    @staticmethod
    def _health_bar(rgb, width):
        # ROI from the inspected 1920x1080 client capture. Display scaling and
        # layouts need separate calibration; no numeric HP is read here.
        columns = max(12, min(120, round(width * 0.155)))
        xs = [(35 + i * 295 / (columns - 1)) / 1920 for i in range(columns)]
        red_columns = []
        white_columns = []
        interior_dark = 0
        for x in xs:
            colors = [rgb(x, y / 1080) for y in (36, 42, 49, 58, 63)]
            reds = sum(c is not None and c[0] >= 130 and c[0] > c[1] * 1.6 + 20
                       and c[0] > c[2] * 1.6 + 20 for c in colors)
            red_columns.append(reds >= 2)
            # The inspected damage-flash frame replaces the HP fill with white.
            # Four vertical samples reject ordinary HP digits, which occupy
            # only the middle of the bar. This supplies layout evidence only.
            whites = sum(c is not None and min(c) >= 202 and max(c) - min(c) <= 35
                         for c in colors)
            white_columns.append(whites >= 4)
            interior_dark += sum(c is not None and max(c) <= 105 for c in colors)
        borders = [rgb(x, y / 1080) for x in xs[::2] for y in (28, 70, 80, 120)]
        black_border = sum(c is not None and max(c) <= 35 for c in borders)
        xp = [rgb(x, 98 / 1080) for x in xs]
        xp_dark_or_green = sum(c is not None and
                              (max(c) <= 105 or (c[1] > c[0] + 20 and c[1] > c[2] + 20))
                              for c in xp)
        red_count = sum(red_columns)
        structure = (black_border >= len(borders) * 0.65
                     and xp_dark_or_green >= len(xp) * 0.60)
        if (red_count == 0 and structure
                and all(white_columns[:max(2, columns // 25)])
                and (sum(white_columns) * 5 + interior_dark) >= columns * 2.5):
            # Do not turn an animation color into a health estimate or damage
            # event. Arena/menu checks must still pass before combat is likely.
            return None, True
        plausible = (red_count >= 1 and structure
                     and (red_count * 5 + interior_dark) >= columns * 2.5)
        if not plausible or not any(red_columns[:max(2, columns // 10)]):
            return None, False
        # Holes from white HP digits are tolerated; a red object far to the
        # right is not accepted as the end of a contiguous fill.
        last_red, gap = -1, 0
        for index, red in enumerate(red_columns):
            gap = 0 if red else gap + 1
            if red:
                last_red = index
            if gap > max(2, columns // 12):
                break
        return min(1.0, (last_red + 1) / columns), True
