"""Factorized, causal evasion observations. Screen geometry is not a map survey."""
from copy import deepcopy
import math

from playmodel.learning import MOVEMENTS


def context(world, frame, epoch):
    snapshot = world.get('build', {}).get('snapshot', {})
    # Keep raw provenance, but do not promote unmasked inventory to known effects.
    weapons = deepcopy(snapshot.get('weapons', []))
    return {'epoch': epoch, 'viewport': [frame.metadata['sample_width'], frame.metadata['sample_height']],
            'map_size': None, 'map_bounds': None, 'obstacles': None,
            'map_status': 'unobserved_world_geometry',
            'camera_status': world.get('camera_status', 'unknown'),
            'camera_shift': world.get('camera_shift'),
            'camera_confidence': world.get('camera_confidence'),
            'visible_risk_count': len(world['tracks']), 'visible_pickup_count': len(world['pickups']),
            'player_screen_position': world['player'],
            'stats': deepcopy(world.get('stats', {})),
            'weapon_observations': weapons, 'weapon_status': snapshot.get('weapon_status', 'unknown'),
            'weapon_count': snapshot.get('weapon_count'),
            'weapon_effects_known': False, 'weapon_range_screen_units': None,
            'game_build_id': snapshot.get('game_build_id'),
            'semantics': 'visible_context_not_verified_map_or_projectile_identity'}


def candidates(world):
    """Fixed test distance, not learned speed or a guaranteed reachable destination."""
    aspect = world.get('aspect_ratio', 1.)
    result = []
    for action, (x, y) in enumerate(MOVEMENTS):
        length = math.hypot(x, y) or 1.
        dx, dy = .08*x/length, .08*y/length
        px, py = world['player'][0]+dx/aspect, world['player'][1]+dy
        gaps = []
        for track in world['tracks'][:8]:
            if track.get('velocity') is None:
                continue
            rx, ry = track['relative']
            vx, vy = track['velocity']
            gaps.append(math.hypot(rx+vx*.2-dx, ry+vy*.2-dy)-track['radius'])
        result.append({'movement': action, 'destination_screen': [px, py],
                       'offset_screen_height': [dx, dy], 'horizon_seconds': .2,
                       'predicted_clearance_proxy': min(gaps) if gaps else None,
                       'outside_viewport': not (0 <= px <= 1 and 0 <= py <= 1),
                       'reachable': None, 'safe': None, 'speed_calibrated': False})
    return result


def examples(rows, run_id):
    """Adjacent preserved receipts only; future observations live exclusively in target."""
    result = []
    for before, after in zip(rows, rows[1:]):
        a, b = before['combat_experience'], after['combat_experience']
        if 'environment' not in a or 'environment' not in b:
            continue
        start, end = a['observed_at_ns'], b['observed_at_ns']
        if not start <= a['available_at_ns'] <= before['sent_at_ns'] < end:
            continue
        action = before.get('actual_movement')
        if type(action) is not int or not 0 <= action < 9:
            continue
        gap = (end-start)/1e9
        same = a['environment']['epoch'] == b['environment']['epoch']
        chain = (before.get('execution_id') is not None and
                 after.get('previous_execution_id') == before['execution_id'])
        admissible = .005 <= gap <= .3 and same and chain
        # Never equate no HP loss or missing tracks with successful evasion.
        distances = [math.hypot(*o['relative'])-o['radius'] for o in b['objects']]
        clearance = min(distances) if distances and admissible else None
        result.append({'schema': 'playmodel.factorized-evasion.v1', 'run_id': run_id,
            'split_group': run_id, 'input': {'observed_at_ns': start,
                'environment': a['environment'], 'threats': a['objects'],
                'candidates': a['evasion_candidates'], 'actual_movement': action},
            'execution': {k: before.get(k) for k in ('execution_id', 'sent_at_ns', 'actual_keys', 'execution_kind')},
            'target': {'observed_at_ns': end, 'elapsed_seconds': gap,
                'player_screen_position': b['player'], 'clearance_proxy': clearance,
                'hp_fraction': b.get('player_hp_fraction'), 'evasion_success': None,
                'censored': not admissible, 'reason': None if admissible else 'gap_context_or_execution_chain'},
            'sources': [{k: r[k] for k in ('frame_ref', 'frame_sha256')} for r in (before, after)],
            'bc_label': False, 'reward_eligible': False})
    return result


def features(row):
    source = row['input']
    env = source['environment']
    names = sorted(str(w.get('name', 'unknown')) for w in env['weapon_observations'])
    nearest = sorted(source['threats'], key=lambda t: math.hypot(*t['relative']))[:3]
    candidate = source['candidates'][source['actual_movement']]
    return {'choice': str(source['actual_movement']), 'weapons_observed': '|'.join(names),
            'weapon_status': env['weapon_status'], 'camera': env['camera_status'],
            'viewport': str(env['viewport']), 'density': env['visible_risk_count'],
            'pickup_density': env['visible_pickup_count'],
            **{f'stat:{k}': v for k,v in env['stats'].items()},
            'nearest_risk': min((math.hypot(*t['relative']) for t in source['threats']), default=1.),
            'predicted_clearance': candidate['predicted_clearance_proxy'],
            'outside_viewport': candidate['outside_viewport'],
            **{f'threat:{i}:{axis}': value for i,t in enumerate(nearest)
               for axis,value in zip(('x','y','vx','vy','radius'),
                    [*t['relative'], *(t.get('velocity') or [None,None]), t['radius']])}}
