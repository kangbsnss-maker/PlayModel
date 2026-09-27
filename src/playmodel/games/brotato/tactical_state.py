"""Unverified screen-geometry tactics; no detector training or game-mechanics oracle.

All distances, association gates and tactic thresholds below are initial
engineering proposals, not measured optimal values. Enemy/projectile identity,
ownership, world speed, current numeric HP and true weapon range stay unknown.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict

from playmodel.learning import MOVEMENTS
from .state_features import STATS
from .vision import VisionObservation

MAX_AGE_NS = 250_000_000
TRACK_GAP_NS = 300_000_000
STAT_NAMES = ('melee_damage', 'ranged_damage', 'speed', 'armor', 'dodge',
              'hp_regeneration', 'attack_speed')
SCHEMA = 'brotato-visual-target-goals-v2'


def _norm(vector):
    return math.hypot(*vector)


def _unit(vector):
    length = _norm(vector)
    return (vector[0] / length, vector[1] / length) if length > 1e-9 else (0., 0.)


def _valid_pair(pair):
    return (isinstance(pair, (tuple, list)) and len(pair) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in pair))


def _build_observation(build_state, observed, available):
    if build_state is None:
        return {}, {'status': 'unknown', 'age_seconds': None, 'confidence': None}
    # Reuse the runtime's causal known masks; snapshot alone does not apply them.
    features = build_state.features(observed, available_at_ns=available)
    snapshot = build_state.snapshot()
    if len(features) != 48:
        return {}, {'status': 'unknown_feature_contract', 'age_seconds': None, 'confidence': None}
    values = {name: snapshot.get('stats', {}).get(name) for index, name in enumerate(STATS)
              if features[16 + index] == 1 and type(snapshot.get('stats', {}).get(name)) in (int, float)
              and math.isfinite(snapshot['stats'][name])}
    sources = snapshot.get('stats_sources', [])
    age = ((observed - max(source['observed_at_ns'] for source in sources)) / 1e9
           if values and sources else None)
    return values, {'status': snapshot.get('stats_status', 'unknown'), 'age_seconds': age,
                    'confidence': snapshot.get('confidence'), 'snapshot': snapshot,
                    'semantics': 'last_observed_display_not_current_combat_effects'}


def _profile(stats):
    melee, ranged = stats.get('melee_damage'), stats.get('ranged_damage')
    profile = ('melee_bias' if melee is not None and ranged is not None and melee > ranged
               else 'ranged_bias' if melee is not None and ranged is not None and ranged > melee
               else 'unknown_or_balanced')
    ring = {'melee_bias': .12, 'ranged_bias': .20, 'unknown_or_balanced': .16}[profile]
    # Displayed values alter a proposed spacing preference, never assert safety.
    if stats.get('speed', 0) < 0:
        ring += .02
    if stats.get('armor', 0) < 0 or stats.get('dodge', 0) < 0:
        ring += .02
    if stats.get('hp_regeneration', 0) > 0 and stats.get('attack_speed', 0) < 0:
        ring += .02
    return profile, round(ring, 2)


class TacticalPlanner:
    def __init__(self, *, clock=time.perf_counter_ns, training_intent='survive and progress'):
        self.clock = clock
        self.training_intent = training_intent
        self._tracks = {}
        self._serial = 0
        self._previous_time = None
        self._previous_player = None
        self._aspect = None

    def reset(self):
        self._tracks.clear()
        self._previous_time = self._previous_player = self._aspect = None

    def observe(self, vision: VisionObservation, *, observed_at_ns: int,
                available_at_ns: int, build_state=None, aspect_ratio: float = 16 / 9) -> dict:
        now = self.clock()
        valid_time = (type(observed_at_ns) is int and type(available_at_ns) is int
                      and 0 < observed_at_ns <= available_at_ns <= now
                      and now - observed_at_ns <= MAX_AGE_NS
                      and (self._previous_time is None or observed_at_ns > self._previous_time))
        valid_geometry = (isinstance(vision, VisionObservation) and _valid_pair(vision.player)
                          and all(0 <= x <= 1 for x in vision.player)
                          and vision.combat_likely and vision.status == 'heuristic_observation'
                          and type(vision.player_confidence) in (int, float)
                          and math.isfinite(vision.player_confidence) and 0 < vision.player_confidence <= 1
                          and vision.observed_at_ns == observed_at_ns
                          and type(aspect_ratio) in (int, float) and math.isfinite(aspect_ratio)
                          and 1.2 <= aspect_ratio <= 2.5)
        if not valid_time or not valid_geometry:
            self._tracks.clear()
            self._previous_player = None
            if valid_time:
                self._previous_time = observed_at_ns
            return self._result({}, {}, {'valid': False, 'reason': 'stale_or_noncausal_observation'
                if not valid_time else 'player_or_combat_unknown', 'observed_at_ns': observed_at_ns,
                'available_at_ns': available_at_ns, 'tracks': []}, {})
        player = tuple(vision.player)
        reset_reason = None
        if self._previous_time is not None:
            if observed_at_ns - self._previous_time > TRACK_GAP_NS or self._aspect != aspect_ratio:
                reset_reason = 'gap_or_resolution_ratio_change'
            elif self._previous_player is not None:
                displacement = (player[0] - self._previous_player[0], player[1] - self._previous_player[1])
                if vision.camera_status == 'shift_candidate' and _valid_pair(vision.camera_shift):
                    displacement = tuple(a - b for a, b in zip(displacement, vision.camera_shift))
                if _norm((displacement[0] * aspect_ratio, displacement[1])) > .18:
                    reset_reason = 'player_teleport_or_bad_association'
        if reset_reason:
            self._tracks.clear()
        previous = [track for track in self._tracks.values()
                    if observed_at_ns - track['observed_at_ns'] <= TRACK_GAP_NS]
        used, tracks = set(), []
        for candidate in vision.hazards:
            if (not all(type(v) in (int, float) and math.isfinite(v) for v in
                        (candidate.x, candidate.y, candidate.radius, candidate.confidence))
                    or not 0 <= candidate.x <= 1 or not 0 <= candidate.y <= 1
                    or not 0 < candidate.confidence <= 1 or not 0 <= candidate.radius <= 1):
                continue
            relative = ((candidate.x - player[0]) * aspect_ratio, candidate.y - player[1])
            matches = sorted((_norm((relative[0] - old['relative'][0], relative[1] - old['relative'][1])),
                              old['track_id'], old) for old in previous if old['track_id'] not in used)
            old = None
            if matches:
                distance, _, nearest = matches[0]
                dt = (observed_at_ns - nearest['observed_at_ns']) / 1e9
                ambiguous = len(matches) > 1 and matches[1][0] - distance < .012
                if not ambiguous and .005 <= dt <= .3 and distance <= min(.25, .035 + dt * 1.5):
                    old = nearest
            velocity, tca, dca, collision_time = None, None, None, None
            patterns = []
            if old is not None:
                used.add(old['track_id'])
                dt = (observed_at_ns - old['observed_at_ns']) / 1e9
                velocity = tuple((a - b) / dt for a, b in zip(relative, old['relative']))
                speed2 = sum(v * v for v in velocity)
                dot = sum(a * b for a, b in zip(relative, velocity))
                if dot < 0:
                    patterns.append('approaching')
                if speed2 > 1e-8:
                    tca = max(0., -dot / speed2)
                    dca = _norm(tuple(a + b * tca for a, b in zip(relative, velocity)))
                    if dot < 0 and tca <= 1.2 and dca < candidate.radius + .07:
                        patterns.append('crossing')
                    # Circle interception under constant relative velocity;
                    # .03 is an uncalibrated player-radius proxy, not a hitbox.
                    clearance2 = sum(v * v for v in relative) - (candidate.radius + .03) ** 2
                    discriminant = dot * dot - speed2 * clearance2
                    if dot < 0 and discriminant >= 0:
                        collision_time = max(0., (-dot - math.sqrt(discriminant)) / speed2)
                    if math.sqrt(speed2) > .45 and candidate.radius < .03:
                        patterns.append('fast_small_candidate')
                    prior_velocity = old.get('velocity')
                    if prior_velocity is not None and _norm(prior_velocity) > .05 and math.sqrt(speed2) > .05:
                        if sum(a * b for a, b in zip(_unit(prior_velocity), _unit(velocity))) < .5:
                            patterns.append('turning')
                identity = old['track_id']
            else:
                self._serial += 1
                identity = self._serial
            tracks.append({'track_id': identity, 'observed_at_ns': observed_at_ns,
                'position': [candidate.x, candidate.y], 'relative': list(relative),
                'radius': candidate.radius, 'confidence': candidate.confidence,
                'identity': 'unknown', 'owner': 'unknown', 'evidence': candidate.evidence,
                'velocity': list(velocity) if velocity is not None else None,
                'velocity_space': 'player_relative_screen_height_per_second',
                'association_verified': False, 'pattern_hypotheses': patterns,
                'closest_approach_seconds': tca, 'closest_approach_distance': dca,
                'collision_time_hypothesis_seconds': collision_time,
                'collision_radius_proxy': candidate.radius + .03,
                'ttc': None, 'ttc_reason': 'collision_geometry_and_identity_unverified'})
        self._tracks = {track['track_id']: track for track in tracks}
        self._previous_time, self._previous_player, self._aspect = observed_at_ns, player, aspect_ratio
        stats, build = _build_observation(build_state, observed_at_ns, available_at_ns)
        profile, ring = _profile(stats)
        tracks.sort(key=lambda row: _norm(row['relative']) - row['radius'])
        crossing = any('crossing' in row['pattern_hypotheses'] for row in tracks)
        options = {}
        if tracks:
            gap = _norm(tracks[0]['relative']) - tracks[0]['radius']
            options.update(retreat='Increase gap from nearby risk candidates.',
                           hold_distance=f'Keep heuristic {ring:.2f} screen-height gap; not weapon range.',
                           orbit_left='Circle candidate left with local avoidance.',
                           orbit_right='Circle candidate right with local avoidance.',
                           kite='Retreat diagonally and preserve spacing.')
            if not crossing and gap > ring * (1.0 if profile == 'melee_bias' else 1.4):
                options['approach'] = 'Close candidate gap toward heuristic ring; identity unknown.'
            if crossing:
                options['avoid_crossing'] = 'Move sideways from predicted relative crossing.'
        else:
            options['move_open'] = 'Move toward open screen area.'
        # Candidate identity stays stable as coordinates change. These are
        # positional targets, not asserted enemy/tree identities or kill labels.
        targets = sorted(tracks[:2], key=lambda row: row['track_id']) if not crossing else []
        for target in targets:
            options[f'hold_distance@{target["track_id"]}'] = (
                f'Prioritize target {target["track_id"]}; maintain spacing.')
        pickups = [item for item in vision.pickups
                   if all(type(v) in (int, float) and math.isfinite(v)
                          for v in (item.x, item.y, item.radius, item.confidence))
                   and 0 <= item.x <= 1 and 0 <= item.y <= 1
                   and 0 <= item.radius <= 1 and 0 < item.confidence <= 1]
        if pickups and not crossing:
            options['collect'] = 'Approach green pickup candidate with local avoidance.'
        parameters = {'spacing': ring, 'profile': profile, 'crossing': crossing,
                      'executor': 'fresh_geometry_v1'}
        world = {'valid': True, 'observed_at_ns': observed_at_ns, 'available_at_ns': available_at_ns,
                 'fresh_until_ns': observed_at_ns + MAX_AGE_NS, 'player': list(player),
                 'player_confidence': vision.player_confidence, 'aspect_ratio': aspect_ratio,
                 'tracks': tracks, 'pickups': [asdict(item) for item in pickups],
                 'source_vision': asdict(vision),
                 'stats': stats, 'build': build, 'camera_status': vision.camera_status,
                 'camera_shift': vision.camera_shift, 'camera_confidence': vision.camera_confidence,
                 'reset_reason': reset_reason, 'numeric_hp': None, 'weapon_range_screen_units': None,
                 'parameters': parameters, 'spacing_semantics': 'stats_based_heuristic_not_verified_weapon_type_or_range',
                 'threshold_status': 'initial_unvalidated_engineering_proposal'}
        state = {'risks': [[round(v, 2) for v in row['relative']] +
                          [round(v, 2) for v in (row['velocity'] or [0., 0.])] +
                          [','.join(row['pattern_hypotheses']) or
                           ('untracked' if row['velocity'] is None else 'stable_relative')] for row in tracks[:2]],
                 'risk_fields': 'relative x,y; relative vx,vy; pattern hypotheses',
                 'player': [round(value, 2) for value in player],
                 'risk_count': len(tracks), 'profile': profile,
                 'training_intent': self.training_intent,
                 'stats': {key: stats[key] for key in STAT_NAMES if key in stats},
                 'stats_age_s': round(build['age_seconds'], 1) if build['age_seconds'] is not None else None,
                 'targets_id_xy': [[row['track_id'], *[round(v, 2) for v in row['position']]] for row in targets],
                 'unknown': 'risk identity/owner; current HP; true weapon range; velocities are unverified relative estimates'}
        return self._result(state, options, world, parameters)

    @staticmethod
    def _result(state, options, world, parameters):
        semantic = {'schema': SCHEMA, 'options': options, 'parameters': parameters}
        signature = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return {'state': state, 'options': options, 'signature': signature, 'world': world}


def execute_tactic(action_id: str, situation: dict, *, now_ns: int | None = None) -> int:
    """Map a selected tactic to fresh geometry; this executor is hand-written.

    Invalid/stale/unknown-player/outside-option requests return neutral (0),
    never a substituted tactic. Caller still owns the final input deadline.
    """
    now = time.perf_counter_ns() if now_ns is None else now_ns
    world = situation.get('world', {})
    if (action_id not in situation.get('options', {}) or world.get('valid') is not True
            or not _valid_pair(world.get('player'))
            or not world.get('available_at_ns', now + 1) <= now <= world.get('fresh_until_ns', -1)):
        return 0
    px, py = world['player']
    aspect = world['aspect_ratio']
    tracks = world['tracks']
    nearest = min(tracks, key=lambda row: _norm(row['relative']) - row['radius']) if tracks else None
    if '@' in action_id:
        family, identity = action_id.split('@', 1)
        selected = next((row for row in tracks if str(row['track_id']) == identity), None)
        if selected is None or family != 'hold_distance':
            return 0
        nearest, action_id = selected, family
    ring = world['parameters']['spacing']
    toward = _unit(nearest['relative']) if nearest else (0., 0.)
    away = (-toward[0], -toward[1])
    tangent = (toward[1], -toward[0])
    if action_id == 'retreat': vector = away
    elif action_id == 'approach': vector = toward
    elif action_id == 'hold_distance':
        gap = _norm(nearest['relative']) - nearest['radius']
        vector = toward if gap > ring + .02 else away if gap < ring - .02 else tangent
    elif action_id in ('orbit_left', 'orbit_right'):
        sign = 1 if action_id == 'orbit_left' else -1
        radial = max(-1., min(1., (_norm(nearest['relative']) - nearest['radius'] - ring) * 5))
        vector = tuple(sign * t + radial * r for t, r in zip(tangent, toward))
    elif action_id == 'kite': vector = tuple(a + .6 * t for a, t in zip(away, tangent))
    elif action_id == 'avoid_crossing':
        threat = next((row for row in tracks if 'crossing' in row['pattern_hypotheses']), nearest)
        velocity = _unit(threat['velocity'] or threat['relative'])
        lateral = (velocity[1], -velocity[0])
        vector = lateral if sum(a * b for a, b in zip(lateral, threat['relative'])) < 0 else tuple(-v for v in lateral)
    elif action_id == 'collect' and world['pickups']:
        item = min(world['pickups'], key=lambda row: math.hypot((row['x'] - px) * aspect, row['y'] - py))
        vector = ((item['x'] - px) * aspect, item['y'] - py)
    else:
        vector = ((.5 - px) * aspect, .5 - py)
        if _norm(vector) < .02: vector = (1., 0.)
    # Current nearby geometry can alter a tactic's direction without changing
    # its identity. No claim of verified collision safety or optimal steering.
    for row in tracks:
        gap = _norm(row['relative']) - row['radius']
        if gap < .10:
            repel = _unit(row['relative'])
            weight = min(3., (.10 - gap) * 25)
            vector = tuple(v - r * weight for v, r in zip(vector, repel))
    edge = (max(0., .06 - px) * 20 - max(0., px - .94) * 20,
            max(0., .07 - py) * 20 - max(0., py - .93) * 20)
    vector = _unit(tuple(v + e for v, e in zip(vector, edge)))
    if _norm(vector) < 1e-9:
        return 0
    return max(range(1, 9), key=lambda i: sum(a * b for a, b in zip(_unit(MOVEMENTS[i]), vector)))


def fallback_movement(situation: dict, *, now_ns: int | None = None) -> int:
    """Explicit geometric fallback, never relabeled as a Laya decision."""
    options = situation.get('options', {})
    action = next((name for name in ('avoid_crossing', 'retreat', 'move_open', 'collect') if name in options), '')
    return execute_tactic(action, situation, now_ns=now_ns)
