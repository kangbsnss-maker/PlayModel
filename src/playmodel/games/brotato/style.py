"""User-editable bootstrap preferences, distinct from learned model weights."""
import hashlib
import json
import math
from pathlib import Path

DEFAULT = {
    'name': 'balanced',
    'stat_weights': {'ranged_damage': 10, 'attack_speed': 9, 'armor': 8, 'max_hp': 7,
                     'hp_regeneration': 6, 'speed': 5, 'life_steal': 5, 'dodge': 5,
                     'harvesting': 4, 'damage': 3, 'crit_chance': 2, 'luck': 1,
                     'melee_damage': 0, 'elemental_damage': 0, 'engineering': 0},
    'behavior': {'avoidance_radius': .17},
    'shop': {'max_rerolls_per_wave': 1, 'reserve': 50, 'max_budget_fraction': .10},
}
ALIASES = {
    'ranged_damage': ('rangeddamage', '远程伤害'), 'attack_speed': ('attackspeed', '攻击速度'),
    'armor': ('armor', '护甲'), 'max_hp': ('maxhp', '最大生命'),
    'hp_regeneration': ('hpregeneration', '生命再生'), 'speed': ('speed', '速度'),
    'life_steal': ('lifesteal', '生命窃取'), 'dodge': ('dodge', '闪避'),
    'harvesting': ('harvesting', '收获'), 'damage': ('damage', '伤害'),
    'crit_chance': ('critchance', '暴击率'), 'luck': ('luck', '幸运'),
    'melee_damage': ('meleedamage', '近战伤害'), 'elemental_damage': ('elementaldamage', '元素伤害'),
    'engineering': ('engineering', '工程学'),
}


def load_style(path: Path | None = None) -> dict:
    data = json.loads(path.read_text(encoding='utf-8')) if path else json.loads(json.dumps(DEFAULT))
    if set(data) != set(DEFAULT) or set(data['stat_weights']) != set(DEFAULT['stat_weights']):
        raise ValueError('Style keys must match the balanced template')
    if not isinstance(data['name'], str) or not data['name'].strip():
        raise ValueError('Style name required')
    for value in data['stat_weights'].values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Stat weights must be finite, 0..100')
    radius = data['behavior']['avoidance_radius']
    shop = data['shop']
    if type(radius) not in (int, float) or not .10 <= radius <= .35:
        raise ValueError('Avoidance radius must be 0.10..0.35 screen heights')
    if type(shop['max_rerolls_per_wave']) is not int or not 0 <= shop['max_rerolls_per_wave'] <= 3:
        raise ValueError('Reroll count must be 0..3')
    if type(shop['reserve']) is not int or not 0 <= shop['reserve'] <= 10000:
        raise ValueError('Invalid material reserve')
    if type(shop['max_budget_fraction']) not in (int, float) or not 0 <= shop['max_budget_fraction'] <= .25:
        raise ValueError('Reroll fraction must be 0..0.25')
    return data


def style_digest(style):
    return hashlib.sha256(json.dumps(style, sort_keys=True, allow_nan=False).encode()).hexdigest()
