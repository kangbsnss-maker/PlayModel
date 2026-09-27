"""Bounded per-frame object evidence, temporal hypotheses and fast arbitration.

No disappearing object is labelled killed/collected. HP crops are uncalibrated
bar hypotheses. Only independent terminal verification supplies outcome labels.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math


def bar_crop(frame, position, radius):
    """Sample source-resolution pixels above a track, without resizing the arena."""
    width, height = frame.metadata['sample_width'], frame.metadata['sample_height']
    if len(frame.pixels) != width * height * 4:
        raise ValueError('Object crop source size mismatch')
    cx, cy = position[0] * width, position[1] * height
    half = max(8, min(70, int(radius * height * 1.8)))
    x0, x1 = max(0, int(cx)-half), min(width, int(cx)+half)
    y0, y1 = max(0, int(cy-radius*height)-12), max(0, int(cy-radius*height)-2)
    y1 = min(height, y1)
    if x1-x0 < 8 or y1-y0 < 2:
        return {'ratio': None, 'status': 'not_observable'}
    raw = b''.join(frame.pixels[(y*width+x0)*4:(y*width+x1)*4] for y in range(y0, y1))
    rows = []
    for y in range(y0, y1):
        colors = [frame.pixels[(y*width+x)*4:(y*width+x)*4+3] for x in range(x0, x1)]
        red = [i for i, (b,g,r) in enumerate(colors) if r > 130 and r > g*1.6 and r > b*1.6]
        if len(red) >= 5 and max(red)-min(red)+1 == len(red):
            rows.append(len(red)/(x1-x0))
    ratio = sum(rows)/len(rows) if len(rows) >= 2 and max(rows)-min(rows) < .08 else None
    return {'ratio': ratio, 'status': 'unverified_bar_candidate' if ratio is not None else 'not_observable',
            'bbox': [x0,y0,x1,y1], 'crop_sha256': hashlib.sha256(raw).hexdigest(),
            'numeric_hp': None, 'reward_eligible': False}


class CombatExperience:
    def __init__(self, motion_state=None):
        self.previous = None
        self.tracks = {}
        self.blocked = [0] * 9
        self.motion_scale = float((motion_state or {}).get('velocity_scale', 1.))
        self.motion_samples = int((motion_state or {}).get('samples', 0))
        self.bar_decay = (motion_state or {}).get('bar_decay_per_second')
        self.last_execution = None

    def record_execution(self, receipt):
        # Atomic reference replacement; recorder thread never mutates observations.
        self.last_execution = {'movement': receipt['actual_movement'],
                               'sent_at_ns': receipt['sent_at_ns'],
                               'successful_transport_reported': receipt['successful_transport_reported']}

    def observe(self, frame, situation, vision):
        world = situation['world']
        if not world.get('valid'):
            self.previous, self.tracks = None, {}
            return {'valid': False}
        now = world['observed_at_ns']
        old = self.previous
        dt = (now-old['observed_at_ns'])/1e9 if old else None
        if dt is None or not .005 <= dt <= .3:
            self.tracks, old = {}, None
        objects, motion, bar_pairs = [], [], []
        target_hp_drop = 0.
        for track in world['tracks'][:8]:
            prior = self.tracks.get(track['track_id'])
            hp = bar_crop(frame, track['position'], track['radius'])
            age = (now-prior['first_seen_ns'])/1e9 if prior else 0.
            row = {'track_id': track['track_id'], 'first_seen_ns': prior['first_seen_ns'] if prior else now,
                   'position': track['position'], 'relative': track['relative'], 'velocity': track['velocity'],
                   'radius': track['radius'], 'hp_observation': hp, 'visible_seconds': age,
                   'class': 'fast_small_hypothesis' if 'fast_small_candidate' in track['pattern_hypotheses'] else 'unknown',
                   'identity_verified': False}
            if (prior and prior['hp_observation'].get('ratio') is not None and hp.get('ratio') is not None):
                target_hp_drop += max(0., prior['hp_observation']['ratio'] - hp['ratio'])
                bar_pairs.append({'input': prior['hp_observation']['ratio'], 'target': hp['ratio'],
                                  'dt_seconds': dt, 'input_observed_at_ns': old['observed_at_ns'],
                                  'target_observed_at_ns': now})
            row['bar_empty_seconds_hypothesis'] = (round(hp['ratio']/self.bar_decay,2)
                if hp.get('ratio') is not None and self.bar_decay and self.bar_decay > .001 else None)
            if prior and old and prior.get('velocity') is not None:
                actual = [(a-b)/dt for a,b in zip(row['relative'], prior['relative'])]
                if all(math.isfinite(x) and abs(x)<5 for x in actual):
                    motion.append({'input_velocity': prior['velocity'], 'target_velocity': actual,
                                   'input_observed_at_ns': old['observed_at_ns'], 'target_observed_at_ns': now,
                                   'track_id': row['track_id'], 'association_verified': False})
            objects.append(row)
        missing = sorted(set(self.tracks)-{r['track_id'] for r in objects})
        execution = self.last_execution
        displacement = None
        if old and execution and execution['successful_transport_reported'] and (
                old['observed_at_ns'] < execution['sent_at_ns'] < now):
            shift = vision.camera_shift if vision.camera_status == 'shift_candidate' else (0.,0.)
            if shift is not None:
                displacement = math.hypot(world['player'][0]-old['player'][0]-shift[0],
                                          world['player'][1]-old['player'][1]-shift[1])
                action = execution['movement']
                # This is evidence of no observed motion, not proof of a wall.
                self.blocked[action] = min(12, self.blocked[action]+1) if displacement < .0015 else 0
        hp_fraction = getattr(vision, 'hp_fill_fraction', None)
        hp_drop = max(0., old['player_hp_fraction']-hp_fraction) if (old and
                    old.get('player_hp_fraction') is not None and hp_fraction is not None) else None
        previous_pickups = old.get('pickups', []) if old else []
        pickup_missing = sum(not any(math.hypot(p['x']-q['x'], p['y']-q['y']) < .03
                                    for q in world['pickups']) for p in previous_pickups)
        result = {'schema': 'playmodel.combat-experience.v1', 'valid': True,
                  'observed_at_ns': now, 'available_at_ns': world['available_at_ns'],
                  'objects': objects, 'motion_pairs': motion,
                  'bar_pairs': bar_pairs,
                  'missing_tracks': missing, 'missing_track_semantics': 'censored_not_kill',
                  'pickups': world['pickups'], 'pickup_disappearances': pickup_missing,
                  'pickup_semantics': 'unverified_not_collection_reward',
                  'player': world['player'], 'player_hp_fraction': hp_fraction,
                  'hp_drop_hypothesis': hp_drop, 'numeric_enemy_hp': None,
                  'target_hp_drop_proxy': target_hp_drop,
                  'movement_not_observed': list(self.blocked), 'map_boundary_verified': False,
                  'motion_model_samples': self.motion_samples, 'motion_scale': self.motion_scale,
                  'stats': world.get('stats', {}), 'reward_eligible': False}
        self.tracks = {r['track_id']: r for r in objects}
        self.previous = deepcopy(result)
        return result

    def emergency(self, situation, evidence, now):
        world = situation['world']
        if not world.get('valid') or now > world['fresh_until_ns']:
            return None
        threat = any(t.get('collision_time_hypothesis_seconds') is not None
                     and t['collision_time_hypothesis_seconds'] / max(.5, self.motion_scale) < .25
                     for t in world['tracks'])
        if threat:
            from .tactical_state import execute_tactic
            if 'avoid_crossing' in situation['options']:
                return execute_tactic('avoid_crossing', situation, now_ns=now), 'fast_collision_hypothesis'
        blocked = evidence.get('movement_not_observed', [])
        if blocked and max(blocked[1:], default=0) >= 5:
            from playmodel.learning import MOVEMENTS
            worst = max(range(1,9), key=lambda i: blocked[i])
            self.blocked[worst] = 0
            opposite = tuple(-x for x in MOVEMENTS[worst])
            return next(i for i, v in enumerate(MOVEMENTS) if tuple(v) == opposite), 'repeated_motion_not_observed'
        return None
