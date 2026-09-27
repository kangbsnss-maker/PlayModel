"""Human tactical preferences, separate from learned tensors and game rewards."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'playmodel.laya-preferences.v1'
TACTICAL_ACTIONS = ('retreat', 'hold_distance', 'approach', 'orbit_left', 'orbit_right',
                    'kite', 'avoid_crossing', 'collect', 'move_open', 'neutral')


def default_preferences():
    return {'schema': SCHEMA, 'revision': 0, 'values': {key: 0.0 for key in TACTICAL_ACTIONS},
            'reason': '', 'updated_utc': '1970-01-01T00:00:00+00:00'}


def validate_preferences(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'revision', 'values', 'reason', 'updated_utc'}:
        raise ValueError('Preference document fields mismatch')
    if value['schema'] != SCHEMA or type(value['revision']) is not int or value['revision'] < 0:
        raise ValueError('Preference schema/revision invalid')
    if not isinstance(value['reason'], str) or len(value['reason']) > 10000:
        raise ValueError('Human preference reason must be text of at most 10000 characters')
    if value['revision'] > 0 and not value['reason'].strip():
        raise ValueError('A saved human preference revision requires a reason')
    if not isinstance(value['updated_utc'], str):
        raise ValueError('Preference UTC timestamp required')
    try:
        stamp = datetime.fromisoformat(value['updated_utc'].replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Preference UTC timestamp required') from error
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('Preference timestamp must use UTC')
    values = value['values']
    if not isinstance(values, dict) or set(values) - set(TACTICAL_ACTIONS):
        raise ValueError('Unknown tactical preference action')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not -2 <= v <= 2 for v in values.values()):
        raise ValueError('Tactical logit bias must be finite within [-2, 2]')
    return {**value, 'values': {key: float(values.get(key, 0)) for key in TACTICAL_ACTIONS}}


def preference_hash(value):
    canonical = json.dumps(validate_preferences(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf8')).hexdigest()


def load_preferences(path):
    path = Path(path)
    return validate_preferences(json.loads(path.read_text(encoding='utf8'))) if path.exists() else default_preferences()


def checkpoint_preferences(bundle):
    """Legacy means both report and bundle predate preferences, never half missing."""
    report = bundle.get('report', {})
    fields = ('preferences', 'preference_hash')
    present = [key in value for value in (bundle, report) for key in fields]
    if not any(present):
        return default_preferences()
    if not all(present):
        raise ValueError('Checkpoint/report preference evidence incomplete')
    value = validate_preferences(bundle['preferences'])
    reported = validate_preferences(report['preferences'])
    fingerprint = preference_hash(value)
    if (value != reported or bundle['preference_hash'] != fingerprint
            or report['preference_hash'] != fingerprint):
        raise ValueError('Checkpoint/report preference evidence mismatch')
    return value
