"""Freeze causal, actually executed movement transitions for offline RL.

Recorded AI actions are environment actions, never correct-action BC labels.
Only completed, independently verified training combat segments are admitted.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from playmodel.laya.records import digest, canonical, verified_outcome, _validate_tactical_execution, _same_control_interval

SCHEMA = 'playmodel.offline-screen-replay.v1'
REWARD = {'horizon_seconds': 60., 'death': -5., 'wave_clear': 1.,
          'movement_bonus': 0., 'purchase_bonus': 0., 'max_interval_seconds': .5,
          'scope': 'wave_local_observed_combat_survival_not_full_run_growth'}


def group_split(run_id):
    bucket=int(hashlib.sha256(('screen-replay-v1:'+run_id).encode()).hexdigest()[:8],16)%10
    return 'validation' if bucket==8 else 'test' if bucket==9 else 'train'


def interrupted_interval(events,start,end):
    for event in events:
        if event['kind'] in ('release','released'):
            began=event.get('send_started_at_ns') or event['at_ns']
            if began<end and event['at_ns']>start:return True
        elif event['kind']=='authority' and event.get('reason')!='ai_granted':
            if start<event['at_ns']<end:return True
    return False


def terminal_bridge(first,terminal_at,phase,events):
    """Normal executor release followed by observation-only terminal verification.

    This is not permission to bridge arbitrary authority changes or pauses.
    The learned executor must retain the same screen-change release behavior.
    """
    metadata=phase.get('metadata',{})
    observed=metadata.get('capture_started_at_ns',0)
    available=phase.get('available_at_ns',0)
    rejected=phase.get('vision',{}).get('rejected_at_ns',0)
    if (phase.get('decision')!='abstain' or phase.get('vision',{}).get('combat_likely') is not False
            or any(metadata.get(k)!=v for k,v in first['target_identity'].items())
            or not first['sent_at_ns']<observed<=available<=rejected<terminal_at
            or observed-first['sent_at_ns']>500_000_000
            or terminal_at-observed>5_000_000_000):return None
    tail=[e for e in events if first['sent_at_ns']<e['at_ns']<=terminal_at or
          (e['kind']=='release' and (e.get('send_started_at_ns') or e['at_ns'])<terminal_at
           and e['at_ns']>first['sent_at_ns'])]
    releases=[];authorities=[]
    for e in tail:
        if e['kind']=='dispatch' and e.get('reason')=='transmitted' and e.get('sequence')==first['sequence']:
            if (e.get('generation')!=first['generation'] or e.get('receipt',{}).get('transmitted') is not True
                    or not e['send_started_at_ns']<=first['sent_at_ns']<=e['send_finished_at_ns']<=observed):
                return None
        elif e['kind']=='release' and e.get('reason') in ('screen_changed','policy_abstained'):
            start=e.get('send_started_at_ns');end=e.get('send_finished_at_ns')
            if (not isinstance(start,int) or not isinstance(end,int)
                    or not rejected<=start<=end==e['at_ns']<e['deadline_ns']
                    or e.get('receipt',{}).get('transmitted') is not True
                    or e.get('receipt',{}).get('acknowledged') is False):return None
            releases.append(e)
        elif e['kind']=='authority' and e.get('reason') in ('screen_changed','policy_abstained') and e['at_ns']>=rejected:
            authorities.append(e)
        elif (e['kind']=='rejected' and e.get('reason')=='policy_abstained'
              and e.get('sequence')==phase.get('sequence') and e['at_ns']>=rejected):
            pass
        else:return None
    if not releases or not authorities or authorities[0]['at_ns']>releases[0]['send_started_at_ns']:
        return None
    return {'schema':'playmodel.screen-change-terminal-bridge.v1',
            'phase_observed_at_ns':observed,'release_events':releases,'authority_events':authorities,
            'terminal_delay_seconds':(terminal_at-observed)/1e9,
            'controlled_seconds':(releases[0]['send_started_at_ns']-first['sent_at_ns'])/1e9,
            'executor':'movement_until_screen_change_then_release_and_observe',
            'observed_alive_seconds':0.}


def transition_values(first, second, *, terminal=None, terminal_at=None):
    """Return None for a censored control interval, not a zero-reward sample."""
    if terminal is not None:
        if terminal not in ('death','wave_clear') or type(terminal_at) is not int:
            raise ValueError('Verified terminal kind/time required')
        dt=(terminal_at-first['sent_at_ns'])/1e9
        if not 0<dt<=1.:return None
        # The terminal detector can be delayed. No invented alive reward in
        # the unobserved final interval.
        return {'reward':REWARD[terminal], 'discount':0., 'done':True,
                'elapsed_seconds':dt, 'observed_alive_seconds':0.}
    if (any(first[k]!=second[k] for k in ('run_id','epoch','generation','target_identity'))
            or second['previous_execution_id']!=first['execution_id']
            or second['sequence']<=first['sequence']
            or second['held_before']!=first['held_after']
            or not first['sent_at_ns']<second['observed_at_ns']<=second['transport_started_at_ns']):
        return None
    dt=(second['observed_at_ns']-first['sent_at_ns'])/1e9
    if not 0<dt<=REWARD['max_interval_seconds']:return None
    discount=math.exp(-dt/REWARD['horizon_seconds'])
    # Identical capture is not proof of elapsed active gameplay.
    alive=dt if first['frame_sha256']!=second['frame_sha256'] else 0.
    return {'reward':REWARD['horizon_seconds']*(1-math.exp(-alive/REWARD['horizon_seconds'])),
            'discount':discount,'done':False,'elapsed_seconds':dt,'observed_alive_seconds':alive}


def freeze_replay(root, session_directories, output, *, per_segment=64, max_segments=None):
    import torch
    from torch.nn import functional as F
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Replay output must be new; preserve prior manifests')
    if not 2<=per_segment<=1024:raise ValueError('per_segment must be 2..1024')
    if max_segments is not None and max_segments<1:raise ValueError('Positive segment budget required')
    output.mkdir(parents=True)
    proofs={}; images={}; tensors=[]; rows=[]; excluded=Counter(); origins=Counter()

    def proof(path, expected=None):
        path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('Replay source outside workspace')
        sha=digest(path)
        if expected is not None and sha!=expected:raise ValueError('Replay source hash mismatch')
        proofs[str(path)]=sha
        return path

    def read(path, expected=None):
        return json.loads(proof(path,expected).read_text(encoding='utf8'))

    def frame(receipt):
        path=proof(receipt['frame_ref'],receipt['frame_sha256'])
        if str(path) in images:return images[str(path)]
        metadata=read(path.with_suffix('.json'))
        if (metadata['capture_started_at_ns']!=receipt['observed_at_ns']
                or any(metadata[k]!=receipt['target_identity'][k] for k in ('hwnd','pid','executable'))):
            raise ValueError('Replay frame metadata/identity mismatch')
        w,h=metadata['sample_width'],metadata['sample_height']
        raw=path.read_bytes()
        if len(raw)!=w*h*4:raise ValueError('Replay frame geometry mismatch')
        rgb=torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(h,w,4)[:,:,[2,1,0]].permute(2,0,1)
        rgb=F.interpolate(rgb[None].float(),size=(96,96),mode='area').round().to(torch.uint8)[0]
        images[str(path)]=len(tensors);tensors.append(rgb)
        return images[str(path)]

    reports=[]
    for directory in session_directories:
        directory=Path(directory).resolve()
        if not directory.is_relative_to(root):raise ValueError('Session outside workspace')
        reports.extend(directory.glob('*/segments/*/pilots/*/report.json'))
    selected_reports=sorted(set(reports))
    if max_segments is not None:selected_reports=selected_reports[:max_segments]
    for report_path in selected_reports:
        report=json.loads(report_path.read_text(encoding='utf8'))
        required=('verified_terminal_boundary','recorder_complete','worker_stopped','tactical_collection_eligible')
        if (report.get('split')!='train' or any(report.get(k) is not True for k in required)
                or report.get('error') or report.get('safety_reason') or report.get('capture_error')
                or report.get('terminal_kind') not in ('death','wave_clear')):
            excluded['ineligible_segment']+=1;continue
        proof(report_path)
        folder=report_path.parent
        terminal_path=proof(folder/'terminal.json')
        _,terminal_at,terminal=verified_outcome(report['terminal_kind'],
            {'path':str(terminal_path),'sha256':digest(terminal_path)})
        proof(terminal['frame_ref'],terminal['frame_sha256'])
        proof(folder/'terminal-source.png',terminal['frame_sha256'])
        ledger_path=proof(report['actions_path'],report['actions_sha256'])
        ledger=[json.loads(line) for line in ledger_path.read_text(encoding='utf8').splitlines() if line.strip()]
        events=read(folder/'control-events.json')
        attempts=read(folder/'input-attempts.json')
        if any(a.get('error') or a.get('transmitted') is not True for a in attempts):
            excluded['incomplete_input_attempt']+=1;continue
        receipts=[read(p['path'],p['sha256']) for p in report['tactical_receipts']]
        receipts.sort(key=lambda r:r['sent_at_ns'])
        if not receipts:
            excluded['no_executed_actions']+=1;continue
        terminal_metadata=read(Path(terminal['frame_ref']).parent/'observation.json')
        if (terminal_metadata.get('frame_sha256')!=terminal['frame_sha256']
                or terminal_metadata.get('capture_started_at_ns')!=terminal_at
                or not receipts or any(terminal_metadata.get(k)!=v for k,v in receipts[0]['target_identity'].items())):
            raise ValueError('Terminal capture identity or time mismatch')
        if (not any(e['kind']=='authority' and e.get('reason')=='ai_granted'
                    and e['at_ns']<=receipts[0]['sent_at_ns'] for e in events)
                or not any(e['kind']=='authority' and e.get('reason')=='closed'
                           and e['at_ns']>=terminal_at for e in events)):
            excluded['incomplete_control_lifecycle']+=1;continue
        if len({r['generation'] for r in receipts})!=1:
            excluded['multiple_control_generations']+=1;continue
        ledger_by_id={r['execution_id']:r for r in ledger}
        if len(ledger_by_id)!=len(ledger) or set(ledger_by_id)!={r['execution_id'] for r in receipts}:
            raise ValueError('Full action ledger/receipt set mismatch')
        by_attempt={(a['generation'],a['sequence']):a for a in attempts}
        if (len(by_attempt)!=len(attempts) or set(by_attempt)!=
                {(r['generation'],r['sequence']) for r in receipts}):
            raise ValueError('Native attempts contain missing, duplicate or unrecorded control')
        for index,r in enumerate(receipts):
            # Validate archived clocks against their own recorded boundary;
            # perf_counter values cannot be compared across Windows boots.
            _validate_tactical_execution(r,r['verified_at_ns'])
            if r['run_id']!=report['run_id'] or r['epoch']!=report['tactical_epoch']:
                raise ValueError('Mixed replay run/epoch')
            original=ledger_by_id[r['execution_id']]
            keys=('actual_movement','held_before','held_after','sent_at_ns','background_stages','sequence','generation')
            if any(original[k]!=r[k] for k in keys):raise ValueError('Action receipt differs from control ledger')
            attempt=by_attempt.get((r['generation'],r['sequence']))
            if (not attempt or attempt['transport_finished_at_ns']!=r['sent_at_ns']
                    or attempt['background_stages']!=r['background_stages']):
                raise ValueError('Native attempt differs from receipt')
            if index:
                previous=read(r['previous_receipt_path'],r['previous_receipt_sha256'])
                _same_control_interval(previous,r)
                if (previous!=receipts[index-1] or previous['held_after']!=r['held_before']
                        or previous['execution_id']!=r['previous_execution_id']
                        or previous['background_stages']['call_id']+1!=r['background_stages']['call_id']):
                    raise ValueError('Broken immediate control chain')
            elif r['previous_execution_id'] is not None or r['held_before']:
                raise ValueError('Missing initial control ownership')
            if r['execution_kind']=='existing_owned_hold':
                anchor=read(r['ownership_receipt_path'],r['ownership_receipt_sha256'])
                _same_control_interval(anchor,r)
                if (not index or anchor['execution_kind']!='native_transition'
                        or anchor['held_after']!=r['held_after']
                        or (previous['execution_kind']=='native_transition' and previous!=anchor)
                        or (previous['execution_kind']!='native_transition' and
                            previous.get('ownership_receipt_sha256')!=r['ownership_receipt_sha256'])):
                    raise ValueError('Invalid held-key ownership')
        setup=read(report_path.parents[4]/'setup-evidence.json')['setup']['context']
        context={k:setup.get(k) for k in ('character','weapons','concept','difficulty','traits')}
        context['semantics']='starting_build_observation_not_current_inventory'
        character_at=setup.get('character_observed_at_ns',terminal_at)
        if setup.get('character_source'):
            character_meta=read(Path(setup['character_source']).parent/'observation.json')
            character_at=character_meta['capture_started_at_ns']
            if (any(character_meta.get(k)!=v for k,v in receipts[0]['target_identity'].items())
                    or (setup.get('character_observed_at_ns') is not None
                        and setup['character_observed_at_ns']!=character_at)):
                raise ValueError('Character context source identity or time mismatch')
            proof(setup['character_source'],character_meta['frame_sha256'])
            if setup.get('character_source_sha256') not in (None,character_meta['frame_sha256']):
                raise ValueError('Character context hash mismatch')
        if character_at>=receipts[0]['observed_at_ns']:
            raise ValueError('Setup context is not causal')
        candidates=[]
        for i,r in enumerate(receipts):
            if r['sent_at_ns']>=terminal_at:
                excluded['post_terminal_input']+=1;continue
            next_r=receipts[i+1] if i+1<len(receipts) else None
            is_terminal=next_r is None or next_r['sent_at_ns']>=terminal_at
            values=transition_values(r,next_r,terminal=report['terminal_kind'] if is_terminal else None,
                                     terminal_at=terminal_at if is_terminal else None)
            end=terminal_at if is_terminal else next_r['observed_at_ns']
            interrupted=interrupted_interval(events,r['observed_at_ns'],end)
            bridge=None
            if is_terminal and i==len(receipts)-1 and (values is None or interrupted) and (folder/'phase-rejection.json').exists():
                phase=read(folder/'phase-rejection.json')
                proof(folder/phase['frame_ref'],phase['frame_sha256'])
                bridge=terminal_bridge(r,terminal_at,phase,events)
                if bridge:
                    values={'reward':REWARD[report['terminal_kind']],'discount':0.,'done':True,
                            'elapsed_seconds':(terminal_at-r['sent_at_ns'])/1e9,'observed_alive_seconds':0.,
                            'terminal_bridge':bridge}
                    interrupted=False
            if values is None or interrupted:
                excluded['censored_interval']+=1;continue
            if not is_terminal:
                if (next_r.get('previous_receipt_path') is None
                        or digest(next_r['previous_receipt_path'])!=next_r.get('previous_receipt_sha256')):
                    raise ValueError('Next receipt has no intact previous link')
                previous=read(next_r['previous_receipt_path'],next_r['previous_receipt_sha256'])
                if previous['execution_id']!=r['execution_id']:raise ValueError('Skipped control execution')
            candidates.append((i,r,next_r,is_terminal,values))
        if not any(item[3] for item in candidates):
            excluded['no_valid_terminal_transition']+=1
            continue
        # Deterministic spread, always retaining the terminal if one was valid.
        admitted_indices={item[0] for item in candidates}
        if len(candidates)>per_segment:
            indices={round(i*(len(candidates)-1)/(per_segment-1)) for i in range(per_segment)}
            candidates=[candidates[i] for i in sorted(indices)]
        for i,r,next_r,done,values in candidates:
            current=frame(r)
            previous=current
            if i and i-1 in admitted_indices and transition_values(receipts[i-1],r) is not None:
                previous=frame(receipts[i-1])
            # A terminal state's value is never bootstrapped. Keep the actual
            # terminal source in proofs, use zero tensor sentinel in the batch.
            next_image=current if done else frame(next_r)
            rows.append({'run_id':r['run_id'],'segment':str(folder),'split':group_split(r['run_id']),
                         'state':[previous,current],'next_state':[current,next_image],
                         'action':r['actual_movement'],'action_origin':r['action_origin'],
                         'build_context':context,
                         'execution_id':r['execution_id'],'observed_at_ns':r['observed_at_ns'],
                         'sent_at_ns':r['sent_at_ns'],'next_observed_at_ns':terminal_at if done else next_r['observed_at_ns'],
                         'terminal_kind':report['terminal_kind'] if done else None,**values})
            origins[r['action_origin']]+=1
    if not rows:raise ValueError('No verified offline transitions; exclusions='+str(dict(excluded)))
    data_path=output/'frames.pt'
    torch.save(torch.stack(tensors),data_path)
    manifest={'schema':SCHEMA,'purpose':'offline_conservative_movement_Q_not_BC',
              'reward':REWARD,'rows':rows,'unique_transitions':len(rows),
              'required_executor_schema':'movement_until_screen_change_then_release_and_observe_v1',
              'terminal_bridge_count':sum('terminal_bridge' in r for r in rows),
              'completed_games_added':0,'original_action_origins':dict(origins),
              'segment_budget':max_segments,'reports_considered':len(selected_reports),
              'split_counts':dict(Counter(r['split'] for r in rows)),
              'excluded':dict(excluded),'frames':{'path':str(data_path),'sha256':digest(data_path)},
              'source_files':[{'path':p,'sha256':s} for p,s in sorted(proofs.items())]}
    path=output/'manifest.json'
    path.write_text(canonical(manifest),encoding='utf8')
    return path


def load_replay(path):
    import torch
    manifest=json.loads(Path(path).read_text(encoding='utf8'))
    if manifest.get('schema')!=SCHEMA or manifest.get('reward')!=REWARD:raise ValueError('Unsupported replay contract')
    for proof in [manifest['frames'],*manifest['source_files']]:
        if digest(proof['path'])!=proof['sha256']:raise ValueError('Frozen offline source changed')
    rows=manifest['rows'];groups={}
    if not rows or len({r['execution_id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate replay transitions')
    for r in rows:
        if r['split']!=group_split(r['run_id']):raise ValueError('Offline group split changed')
        groups.setdefault(r['run_id'],set()).add(r['split'])
        if (r['action'] not in range(9) or not r['observed_at_ns']<r['sent_at_ns']<r['next_observed_at_ns']
                or not math.isfinite(r['reward']) or not 0<=r['discount']<=1
                or (r['done'] and r['discount']!=0)):
            raise ValueError('Invalid offline transition')
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Offline group leakage')
    frames=torch.load(manifest['frames']['path'],map_location='cpu',weights_only=True)
    if frames.dtype!=torch.uint8 or frames.ndim!=4 or frames.shape[1:]!=(3,96,96):
        raise ValueError('Invalid offline frame tensor')
    for r in rows:
        if any(len(r[k])!=2 or any(type(i) is not int or not 0<=i<len(frames) for i in r[k])
               for k in ('state','next_state')):
            raise ValueError('Invalid frame history indices')
    return manifest,frames
