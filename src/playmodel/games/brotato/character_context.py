"""Source-backed character conditioning, not a complete mechanics interpreter."""
import hashlib
import re
from pathlib import Path

TERMS = ('melee damage', 'ranged damage', 'elemental damage', 'engineering',
         'attack speed', 'max hp', 'hp regeneration', 'life steal', 'armor',
         'dodge', 'speed', 'range', 'luck', 'harvesting', 'damage')


def effect_text(text, term):
    compact=term.replace(' ','')
    for other in TERMS:
        longer=other.replace(' ','')
        if longer!=compact and compact in longer:
            text=text.replace(longer,'#')
    return text


def modifiers(lines):
    result = {}
    for line in lines:
        text = re.sub(r'\s+', '', line.casefold()).replace('−','-')
        for term in TERMS:
            match = re.search(r'([+-]\d+(?:\.\d+)?)(%?)'+term.replace(' ',''), effect_text(text,term))
            if match:
                # Keep displayed units; no conversion into physical effect/range.
                result[term] = match[0]
    return result


def affinity(traits, option):
    """Small compatibility hint from explicit signed display modifiers only."""
    text = re.sub(r'\s+', '', option.casefold()).replace('−','-')
    score = 0
    for term, raw in modifiers(traits).items():
        compact = term.replace(' ','')
        scoped=effect_text(text,term)
        if compact not in scoped:
            continue
        # Longest-term matching avoids counting ranged damage as generic damage.
        amount = re.search(r'([+-]\d+(?:\.\d+)?)(?:%)?'+compact, scoped)
        option_sign = -1 if amount and float(amount[1]) < 0 else 1
        score += (-1 if raw.startswith('-') else 1)*option_sign
    return max(-5,min(5,score))


def validated_context(context):
    source = context.get('character_source')
    expected = context.get('character_source_sha256')
    if not source or not expected:
        return {}
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
        raise ValueError('Character observation source changed')
    return {'name':context.get('character','Unknown'), 'traits':context.get('traits',[]),
            'frame_ref':source,'frame_sha256':expected,
            'observed_at_ns':context.get('character_observed_at_ns'),
            'unlock_goals':context.get('unlock_goals',[]),
            'semantics':'observed_display_traits_partial_rule_interpretation'}


def validate_choice_character(character, observed_at_ns):
    when=character.get('observed_at_ns')
    if (type(when) is not int or not 0 < when <= observed_at_ns
            or hashlib.sha256(Path(character['frame_ref']).read_bytes()).hexdigest()!=character['frame_sha256']):
        raise ValueError('Character source changed or is noncausal')
    return {'path':character['frame_ref'],'sha256':character['frame_sha256']}
