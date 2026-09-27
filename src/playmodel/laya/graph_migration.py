"""Explicit weight-preserving migration for the terminal admission bug fix."""
import hashlib
from .records import canonical


def validate_character_provenance_migration(bundle, current_sources):
    old = bundle.get('report', {}).get('graph_sources', {})
    fingerprint = hashlib.sha256(canonical(old).encode()).hexdigest()
    if (fingerprint != 'ce5add78e12e558691a18b2ca7d4700dd46cdf87125612201ab526cafe908e80'
            or bundle.get('graph_hash') != fingerprint
            or bundle.get('report', {}).get('graph_hash') != fingerprint
            or set(old) != set(current_sources)
            or any(old[k] != v for k,v in current_sources.items() if k != 'laya/worker.py')):
        raise ValueError('Unsupported character provenance checkpoint migration')
    return {'migration':'character_source_provenance_v1','old_graph_hash':fingerprint,
            'weights_preserved':True,'behavior_version_changes':True}


def validate_object_branch_migration(bundle):
    sources = bundle.get('report', {}).get('graph_sources', {})
    fingerprint = hashlib.sha256(canonical(sources).encode()).hexdigest()
    if (fingerprint not in (
            'baaee8ea1b837445bcdf86cbe2cf78aec3cbfb7adff0fa254d563162b50e4f41',
            '863543b2e688ce573fcbae04380378c537d6a1361acbbe88fe3611efc47303f6')
            or bundle.get('graph_hash') != fingerprint
            or bundle.get('report', {}).get('graph_hash') != fingerprint):
        raise ValueError('Unsupported object branch checkpoint migration')
    return {'migration': 'zero_residual_object_crops_v1', 'old_graph_hash': fingerprint,
            'weights_preserved': True, 'new_branch_initial_effect': 0,
            'behavior_version_changes': True}


def validate_boundary_fix_migration(bundle, current_sources):
    old = bundle.get('report', {}).get('graph_sources', {})
    fingerprint = hashlib.sha256(canonical(old).encode()).hexdigest()
    if (bundle.get('graph_hash') != fingerprint
            or bundle.get('report', {}).get('graph_hash') != fingerprint
            or set(old) != set(current_sources)
            or old.get('laya/worker.py') !=
            '8bf18a20ea5965c6dfada5f4054568876a578e63426d71a80b2911b00fe980e4'
            or any(old[key] != value for key, value in current_sources.items()
                   if key != 'laya/worker.py')):
        raise ValueError('Visual decision checkpoint graph/preprocessing mismatch')
    return {'migration': 'terminal_boundary_admission_fix_v1',
            'old_graph_hash': fingerprint, 'weights_preserved': True,
            'behavior_version_changes': True}
