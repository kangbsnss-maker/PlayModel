"""Evidence validation shared by the local decision worker and tests."""
from pathlib import Path
import hashlib
import json
import math

SCHEMA = 'playmodel.laya-choice.v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_options(options):
    if not isinstance(options, dict) or not 1 <= len(options) <= 16:
        raise ValueError('Laya needs 1..16 legal options')
    for key, description in options.items():
        if not isinstance(key, str) or not key or not isinstance(description, str) or not description.strip():
            raise ValueError('Nonempty option identifiers and descriptions required')


def verified_outcome(kind, evidence):
    """Only independently observed events may supply a return; no stop penalty."""
    if kind not in ('death', 'wave_clear'):
        raise ValueError('Unknown/aborted outcomes cannot provide a learning reward')
    if not isinstance(evidence, dict):
        raise ValueError('Independent event evidence required')
    path = evidence.get('path') or evidence.get('evidence_path')
    if not path or evidence.get('sha256') != digest(path):
        raise ValueError('Outcome evidence file hash mismatch')
    value = json.loads(Path(path).read_text(encoding='utf8'))
    # Caller supplies the original terminal.json written by the combat verifier.
    event_kind = value.get('kind') or value.get('terminal_kind')
    independent = value.get('independent_of_policy')
    verified = value.get('verified')
    if event_kind != kind or independent is not True or verified is not True:
        raise ValueError('Outcome must be independently verified death/wave_clear')
    observed = value.get('observed_at_ns')
    if type(observed) is not int or observed <= 0:
        raise ValueError('Outcome observation time required')
    frame = value.get('frame_ref')
    frame_hash = value.get('frame_sha256')
    if not frame or not frame_hash or digest(frame) != frame_hash:
        raise ValueError('Outcome source frame hash mismatch')
    return (1.0 if kind == 'wave_clear' else -1.0), observed, value


def validate_distribution(options, probabilities):
    if len(options) != len(probabilities) or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in probabilities
    ) or abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError('Invalid decision probability distribution')


def validate_resume_report(bundle):
    report = bundle.get('report', {})
    visual_keys = any(key.startswith('visual_context.') for key in bundle.get('head', {}))
    if visual_keys and (bundle.get('model_schema') != 'playmodel.visual-goal.v1'
                       or report.get('model_schema') != 'playmodel.visual-goal.v1'
                       or report.get('encoder_frozen_scope') != 'text_encoder_only'
                       or report.get('visual_encoder_trainable') is not True):
        raise ValueError('Visual decision checkpoint lacks its learning contract')
    kls = report.get('kl_per_choice')
    if (report.get('accepted') is not True or report.get('optimizer_steps') != 1
            or report.get('encoder_hash_before') != bundle.get('encoder_hash')
            or report.get('encoder_hash_after') != bundle.get('encoder_hash')
            or report.get('head_hash_after') != bundle.get('head_hash')
            or report.get('head_hash_before') == report.get('head_hash_after')
            or not isinstance(kls, list) or not kls
            or any(not isinstance(k, (int, float)) or not math.isfinite(k) or k > 0.03 for k in kls)):
        raise ValueError('Rejected or unverified Laya candidate cannot resume')


def _bound_json(path, sha256, label):
    try:
        if not path or digest(path) != sha256:
            raise ValueError(label + ' hash mismatch')
        value = json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(label + ' is missing or invalid') from error
    if not isinstance(value, dict):
        raise ValueError(label + ' must be an object')
    return value


def _tactical_identity(identity):
    if (not isinstance(identity, dict)
            or any(type(identity.get(key)) is not int or identity[key] <= 0 for key in ('hwnd', 'pid'))
            or not isinstance(identity.get('executable'), str) or not identity['executable'].strip()):
        raise ValueError('Tactical target identity required')
    return identity


def validate_tactical_observation(state, options, evidence, now_ns):
    """Bind the exact model input and unverified world observations to their source."""
    validate_options(options)
    names = ('observed_at_ns', 'available_at_ns', 'expires_at_ns')
    if (not isinstance(evidence, dict) or type(now_ns) is not int
            or any(type(evidence.get(key)) is not int for key in names)):
        raise ValueError('Tactical observation timestamps required')
    observed, available, expires = (evidence[key] for key in names)
    if not 0 < observed <= available <= now_ns or expires <= available:
        raise ValueError('Noncausal tactical observation')
    if evidence.get('decision_domain') != 'combat_tactic' or any(
            not isinstance(evidence.get(key), str) or not evidence[key] for key in ('run_id', 'epoch', 'signature')):
        raise ValueError('Tactical observation identity required')
    identity = _tactical_identity(evidence.get('target_identity'))
    document = _bound_json(evidence.get('observation_path'), evidence.get('observation_sha256'),
                           'Tactical observation')
    original = {key: value for key, value in evidence.items()
                if key not in ('observation_path', 'observation_sha256')}
    metadata, situation = document.get('metadata'), document.get('situation')
    if (document.get('evidence') != original or not isinstance(metadata, dict)
            or any(metadata.get(key) != value for key, value in identity.items())
            or metadata.get('capture_started_at_ns') != observed
            or not isinstance(situation, dict) or situation.get('state') != state
            or situation.get('options') != options or situation.get('signature') != evidence['signature']):
        raise ValueError('Tactical observation differs from model input or source identity')
    world = situation.get('world')
    if (not isinstance(world, dict) or world.get('valid') is not True
            or world.get('observed_at_ns') != observed or world.get('available_at_ns') != available):
        raise ValueError('Tactical world observation is invalid or noncausal')
    frame = evidence.get('frame_ref')
    if not frame or digest(frame) != evidence.get('frame_sha256'):
        raise ValueError('Tactical observation frame hash mismatch')
    return document


def _validate_tactical_execution(receipt, now_ns):
    """Validate one controller execution, without claiming visible game effect."""
    if (receipt.get('schema') != 'playmodel.tactical-execution.v1' or receipt.get('domain') != 'combat_tactic'
            or receipt.get('accepted') is not True or receipt.get('successful_transport_reported') is not True
            or receipt.get('transmitted') is not True
            or (receipt.get('acknowledged') is not None and receipt.get('acknowledged') is not True)
            or receipt.get('game_application_verified') is not False
            or receipt.get('cnn_training_eligible') is not False
            or receipt.get('writer_authority') != 'single_ai_writer'
            or receipt.get('action_origin') not in ('local_laya', 'explicit_rule_fallback')
            or any(not isinstance(receipt.get(key), str) or not receipt[key]
                   for key in ('execution_id', 'run_id', 'epoch'))):
        raise ValueError('Tactical control provenance is incomplete')
    _tactical_identity(receipt.get('target_identity'))
    if (type(receipt.get('generation')) is not int or receipt['generation'] <= 0
            or type(receipt.get('sequence')) is not int or receipt['sequence'] < 0):
        raise ValueError('Tactical controller generation and sequence required')
    names = ('observed_at_ns', 'available_at_ns', 'decided_at_ns', 'transport_started_at_ns',
             'sent_at_ns', 'transport_finished_at_ns', 'deadline_ns')
    if type(now_ns) is not int or any(type(receipt.get(key)) is not int for key in names):
        raise ValueError('Tactical control timestamps required')
    observed, available, decided, started, sent, finished, deadline = (receipt[key] for key in names)
    if not (0 < observed <= available <= decided <= started <= sent == finished < deadline
            and sent <= now_ns and sent - observed < 250_000_000):
        raise ValueError('Stale/noncausal tactical execution')
    frame = receipt.get('frame_ref')
    if not frame or digest(frame) != receipt.get('frame_sha256'):
        raise ValueError('Tactical execution frame hash mismatch')
    movement = receipt.get('actual_movement')
    movements = ((), (0x57,), (0x57, 0x44), (0x44,), (0x53, 0x44),
                 (0x53,), (0x53, 0x41), (0x41,), (0x57, 0x41))
    before, after, actual = (receipt.get(key) for key in ('held_before', 'held_after', 'actual_keys'))
    if (type(movement) is not int or not 0 <= movement <= 8
            or not isinstance(before, list) or not isinstance(after, list) or actual != after
            or any(type(key) is not int or key not in (0x57, 0x41, 0x53, 0x44) for key in before + after)
            or len(set(before)) != len(before) or len(set(after)) != len(after)
            or set(after) != set(movements[movement])):
        raise ValueError('Tactical movement/owned key mismatch')
    stages = receipt.get('background_stages')
    native = receipt.get('native_post_count')
    stage_names = ('call_id', 'started_at_ns', 'check_finished_at_ns', 'posts_finished_at_ns',
                   'posted_keys', 'attempted_posts', 'deadline_ns')
    if (not isinstance(stages, dict) or any(type(stages.get(key)) is not int for key in stage_names)
            or stages['call_id'] <= 0 or type(native) is not int or native < 0
            or stages['posted_keys'] != native or stages['attempted_posts'] != native
            or stages.get('identity_check_passed') is not True or stages['deadline_ns'] != deadline
            or not started <= stages['started_at_ns'] <= stages['check_finished_at_ns']
            <= stages['posts_finished_at_ns'] <= sent):
        raise ValueError('Tactical native execution stages mismatch')
    kind = receipt.get('execution_kind')
    if kind == 'native_transition':
        post_start = stages.get('posts_started_at_ns')
        if (native != len(set(before) ^ set(after)) or native < 1
                or stages.get('native_post_attempted') is not True or type(post_start) is not int
                or not stages['check_finished_at_ns'] <= post_start <= stages['posts_finished_at_ns']):
            raise ValueError('Native transition has no matching successful posts')
    elif kind in ('existing_owned_hold', 'neutral_hold'):
        if (native != 0 or before != after or stages.get('native_post_attempted') is not False
                or stages.get('posts_started_at_ns') is not None
                or bool(after) != (kind == 'existing_owned_hold')):
            raise ValueError('Hold is not a no-post owned control interval')
    else:
        raise ValueError('Unknown tactical execution kind')


def _same_control_interval(earlier, later):
    if (any(earlier.get(key) != later.get(key) for key in
            ('run_id', 'epoch', 'generation', 'target_identity'))
            or earlier['execution_id'] == later['execution_id']
            or earlier['sent_at_ns'] > later['transport_started_at_ns']
            or earlier['sequence'] >= later['sequence']):
        raise ValueError('Tactical ownership link is from a different control interval')


def validate_tactical_application(record, application, now_ns):
    """Validate exact choice, fresh control execution and immutable ownership links."""
    if record.get('decision_domain') != 'combat_tactic':
        raise ValueError('Tactical acceptance requires a combat decision')
    receipt_path = application.get('receipt_path')
    receipt = _bound_json(receipt_path, application.get('receipt_sha256'), 'Tactical receipt')
    _validate_tactical_execution(receipt, now_ns)
    if receipt['action_origin'] != 'local_laya':
        raise ValueError('Fallback is not an executed Laya choice')
    for key in ('decision_id', 'action_id', 'behavior_version'):
        if not record.get(key) or receipt.get(key) != record[key]:
            raise ValueError('Tactical receipt decision/version mismatch')
    evidence = record['evidence']
    validate_tactical_observation(record.get('state'), record.get('options'), evidence, now_ns)
    if record['action_id'] not in record['options']:
        raise ValueError('Tactical action was not among the sampled options')
    for key in ('epoch', 'signature', 'run_id', 'target_identity'):
        if receipt.get(key) != evidence[key]:
            raise ValueError('Tactical receipt request identity mismatch')
    decided = record.get('decided_at_ns')
    if (type(decided) is not int
            or not evidence['available_at_ns'] <= decided <= receipt['decided_at_ns']
            <= receipt['transport_started_at_ns'] <= receipt['sent_at_ns'] < evidence['expires_at_ns']
            or evidence['observed_at_ns'] > receipt['observed_at_ns']
            or evidence['available_at_ns'] > receipt['available_at_ns']):
        raise ValueError('Expired or noncausal tactical choice execution')
    previous_id = receipt.get('previous_execution_id')
    previous = None
    if previous_id is not None:
        previous_path = receipt.get('previous_receipt_path')
        previous = _bound_json(previous_path, receipt.get('previous_receipt_sha256'), 'Previous tactical receipt')
        _validate_tactical_execution(previous, now_ns)
        _same_control_interval(previous, receipt)
        if (previous['execution_id'] != previous_id or previous['held_after'] != receipt['held_before']
                or Path(previous_path).resolve().parent != Path(receipt_path).resolve().parent
                or previous['background_stages']['call_id'] + 1 != receipt['background_stages']['call_id']):
            raise ValueError('Tactical execution does not continue its immediate predecessor')
    elif receipt.get('previous_receipt_path') is not None or receipt.get('previous_receipt_sha256') is not None:
        raise ValueError('Unexpected tactical predecessor evidence')
    if receipt['execution_kind'] == 'existing_owned_hold':
        anchor_path = receipt.get('ownership_receipt_path')
        anchor = _bound_json(anchor_path, receipt.get('ownership_receipt_sha256'), 'Tactical ownership anchor')
        _validate_tactical_execution(anchor, now_ns)
        _same_control_interval(anchor, receipt)
        if (previous is None or anchor['execution_kind'] != 'native_transition'
                or anchor['held_after'] != receipt['held_after']
                or Path(anchor_path).resolve().parent != Path(receipt_path).resolve().parent):
            raise ValueError('Held keys have no matching ownership chain')
        if previous['execution_kind'] == 'native_transition':
            if (previous['execution_id'] != anchor['execution_id']
                    or receipt['previous_receipt_sha256'] != receipt['ownership_receipt_sha256']):
                raise ValueError('Held keys skip a native ownership transition')
        elif (previous['execution_kind'] != 'existing_owned_hold'
                or previous.get('ownership_receipt_sha256') != receipt['ownership_receipt_sha256']
                or not previous.get('ownership_receipt_path')
                or Path(previous['ownership_receipt_path']).resolve() != Path(anchor_path).resolve()):
            raise ValueError('Held keys changed ownership anchor')
    elif receipt.get('ownership_receipt_path') is not None or receipt.get('ownership_receipt_sha256') is not None:
        raise ValueError('Unexpected tactical ownership anchor')
    return receipt
