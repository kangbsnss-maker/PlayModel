"""Bounded, screen-verified next-run selection. No global cursor/focus calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import hashlib
import re

from .background import BackgroundController
from .capture import capture_session,read_diagnostic_png,_png
from .ocr import MenuOcr,rows_in_region
from .menu import classify_scene,selected_button,navigation_key,BUTTONS
from playmodel.video import SessionRecording
from playmodel.media_titles import label_recording
from playmodel.instance import session_lock
from .menu_capture import MenuCapture
from .menu_transition import MenuTransitionGate


CHARACTER_GRID = {i:(69+(i%17)*106,723+(i//17)*106,165+(i%17)*106,819+(i//17)*106) for i in range(51)}


def _main_menu_labels(text):
    # Measured on 1.1.15.4: WinRT reads the l in Profile as uppercase I.
    return (all(word in text for word in ('start', 'options', 'quit'))
            and any(word in text for word in ('profile', 'profiie')))


def is_main_menu(ocr, pixels, width, height):
    if (width, height) != (1920, 1080):
        return False
    text=''.join(rows_in_region(ocr,(25,600,310,1030))).casefold()
    selected=pixels[(640*width+48)*4:(640*width+48)*4+3]
    return _main_menu_labels(text) and len(selected)==3 and min(selected)>180


def recognize_main_menu(reader, source, ocr, pixels, width, height):
    """Bounded menu ROI avoids title artwork skewing full-frame OCR coordinates."""
    if is_main_menu(ocr, pixels, width, height):
        return True
    if (width, height) != (1920, 1080) or len(pixels) != width * height * 4:
        return False
    selected = pixels[(640 * width + 48) * 4:(640 * width + 48) * 4 + 3]
    if min(selected) <= 180:
        return False
    left, top, right, bottom = 25, 600, 310, 1030
    crop = b''.join(pixels[(y * width + left) * 4:(y * width + right) * 4]
                    for y in range(top, bottom))
    path = Path(source).with_name('main-menu-roi.png')
    path.write_bytes(_png(right-left, bottom-top, crop))
    report = reader.read(path)
    path.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    text = ''.join(rows_in_region(report, (0, 0, right-left, bottom-top))).casefold()
    return _main_menu_labels(text)


def focused_tile(pixels: bytes, width: int, boxes: dict, *, low=145, high=245) -> int | None:
    scores=[]
    for index,(left,top,right,bottom) in boxes.items():
        samples=[(x,y) for x in (left+7,right-8) for y in range(top+22,bottom-22,3)]
        samples += [(x,y) for y in (top+7,bottom-8) for x in range(left+22,right-22,3)]
        colors=[pixels[(y*width+x)*4:(y*width+x)*4+3] for x,y in samples]
        score=sum(low<=min(c)<=max(c)<=high and max(c)-min(c)<=18 for c in colors)/len(colors)
        scores.append((score,index))
    scores.sort(reverse=True)
    return scores[0][1] if scores[0][0]>.65 and scores[1][0]<.35 else None


def locked_character(pixels, width, ocr):
    """Calibrated dark focused tile plus locked-card condition, never dimness alone."""
    header=''.join(rows_in_region(ocr,(500,60,1450,160))).casefold()
    name=''.join(rows_in_region(ocr,(210,175,975,280))).strip()
    condition=' '.join(rows_in_region(ocr,(440,445,750,550))).strip()
    records=''.join(rows_in_region(ocr,(760,380,1120,530))).casefold()
    slot=focused_tile(pixels,width,CHARACTER_GRID,low=105,high=160)
    if ('characterselection' not in header or name or not condition
            or 'records' not in records or slot is None or slot==0):
        return None
    match=re.fullmatch(r'Recycle(\d+)weaponsduringarun',re.sub(r'\s+','',condition),re.I)
    return {'slot':slot,'condition_text':condition,'status':'locked_observed',
            'objective': {'kind':'recycle_weapons_in_one_run','count':int(match[1])}
                         if match else None,
            'objective_semantics':'screen_condition_not_verified_completion'}


def prepare_next(executable: Path, *, root: Path, character_slot: int, weapon: str, record: bool,
                 concept: str | None = None) -> dict:
    if not 1<=character_slot<=50: raise ValueError('Visible calibrated character slot must be 1..50')
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-setup-'+uuid.uuid4().hex[:6]
    directory=root/'artifacts/run-setup'/token
    media=root/'media/captures'/token
    directory.mkdir(parents=True)
    controller=None
    recording=None
    context={'character_slot':character_slot,'concept':concept or weapon+'-build','endless_requested':True,
             'character':'Unknown','weapons':[],'difficulty':'Nightmare'}
    visited=[]
    attempted_slot=None
    weapon_names=[]
    weapon_profiles={}
    preferred_weapon=None
    weapon_move=None
    character_mismatch_observations=0
    error=None
    recovery_wait=None
    transition_gate=MenuTransitionGate()
    locked_pending=None
    unlock_goals=[]
    with session_lock(root/'artifacts/brotato-input.lock'), MenuOcr(root/'scripts/windows_ocr.ps1') as menu_reader, MenuCapture(executable) as menu_capture:
        try:
            if record: recording=SessionRecording(media,max_seconds=240)
            for _ in range(100):
                if (root/'artifacts/BROTATO_STOP').exists(): raise OSError('User stop')
                def check_stop():
                    if (root/'artifacts/BROTATO_STOP').exists(): raise OSError('User stop')
                shot,pixels,w,h=menu_capture.read(directory,check=check_stop)
                if not transition_gate.allow(shot['frame_sha256'], captured_at_ns=shot['capture_started_at_ns']):
                    continue
                source=Path(shot['session_directory'])/'frame.png'
                ocr=menu_reader.read(source)
                (source.parent/'setup-ocr.json').write_text(json.dumps(ocr,ensure_ascii=False),encoding='utf-8')
                if (w,h)!=(1920,1080): raise OSError('Setup resolution changed')
                if controller is None: controller=BackgroundController(shot['hwnd'],executable)
                header=''.join(rows_in_region(ocr,(500,60,1450,160))).casefold()
                scene=classify_scene(ocr).scene
                if recognize_main_menu(menu_reader,source,ocr,pixels,w,h):
                    recovery_wait=None
                    key='enter'
                elif scene=='result':
                    focus=selected_button(pixels,w,h,scene='result')
                    if focus.selected_id is None: raise OSError('Result focus unknown')
                    key='enter' if focus.selected_id=='new_run' else navigation_key(focus.rect,BUTTONS['result']['new_run'])
                elif 'characterselection' in header:
                    recovery_wait=None
                    locked=locked_character(pixels,w,ocr)
                    if locked:
                        proof={**locked,'frame_ref':str(source),'frame_sha256':shot['frame_sha256'],
                               'observed_at_ns':shot['capture_started_at_ns'],
                               'available_at_ns':ocr['available_at_ns'],
                               'target_identity':{k:shot.get(k) for k in ('hwnd','pid','executable')}}
                        if (locked_pending is None or locked_pending['slot']!=locked['slot']
                                or locked_pending['condition_text']!=locked['condition_text']
                                or proof['target_identity']!=locked_pending['target_identity']
                                or proof['observed_at_ns']<=locked_pending['available_at_ns']):
                            locked_pending=proof
                            continue
                        goal={**locked,'observations':[locked_pending,proof],
                              'completion_verified':False}
                        unlock_goals.append(goal)
                        visited.append(goal)
                        ledger=root/'artifacts/local-learning/unlock-goals.jsonl'
                        ledger.parent.mkdir(parents=True,exist_ok=True)
                        with ledger.open('a',encoding='utf8') as stream:
                            stream.write(json.dumps(goal,ensure_ascii=False)+'\n')
                        context['unlock_goals']=unlock_goals
                        # Leave the locked tile using arrows; never confirm a lock.
                        character_slot=1 if locked['slot']!=1 else 2
                        attempted_slot=None
                        slot=locked['slot']
                        locked_pending=None
                    else:
                        slot=focused_tile(pixels,w,CHARACTER_GRID)
                        locked_pending=None
                    if slot is None: raise OSError('Character selection focus unknown')
                    if attempted_slot==slot:
                        visited.append({'slot':slot,'selection_not_confirmed':True,'source':str(source)})
                        character_slot=1+(slot%50)
                        attempted_slot=None
                    if slot != character_slot:
                        row,col=divmod(slot,17)
                        target_row,target_col=divmod(character_slot,17)
                        key=('down' if target_row>row else 'up') if row!=target_row else ('right' if target_col>col else 'left')
                    else:
                        name=' '.join(rows_in_region(ocr,(210,175,975,280))).strip()
                        if not name or 'random' in name.casefold(): raise OSError('Character name unreadable')
                        context.update(character=name,character_slot=slot,character_source=str(source),
                                       traits=rows_in_region(ocr,(520,285,975,690)),
                                       character_source_sha256=shot['frame_sha256'],
                                       character_observed_at_ns=shot['capture_started_at_ns'])
                        # The observed ON switch is white on the right. Do not silently run normal mode.
                        switch=pixels[(320*w+1640)*4:(320*w+1640)*4+3]
                        if min(switch)<180: raise OSError('Endless option not confirmed enabled')
                        context['endless_verified']=True
                        key='enter'
                        attempted_slot=slot
                elif 'selection' in header and ('diffic' in header):
                    # Leave on this screen: the recorded gameplay session selects highest difficulty.
                    if not context.get('character_source') or not context.get('weapon_source'):
                        # Interrupted setup: return through the observed pre-run
                        # menus and collect fresh character/weapon evidence.
                        if recovery_wait is not None:
                            continue
                        key='escape'
                        recovery_wait='weapon'
                    else:
                        context['difficulty_menu_source']=str(source)
                        context['setup_complete']=True
                        break
                elif ('back' in ''.join(rows_in_region(ocr,(20,20,300,100))).casefold()
                      and context['character']=='Unknown'):
                    if recovery_wait=='character':
                        continue
                    key='escape'
                    recovery_wait='character'
                elif 'back' in ''.join(rows_in_region(ocr,(20,20,300,100))).casefold():
                    name=' '.join(rows_in_region(ocr,(1290,180,1520,220))).strip()
                    # This is the calibrated weapon-card layout, with character name in the left card.
                    char=''.join(rows_in_region(ocr,(415,170,1180,265)))
                    if context['character'].replace(' ','').casefold() not in char.casefold():
                        character_mismatch_observations+=1
                        if character_mismatch_observations<=3:
                            controller.release()
                            continue
                        raise OSError('Weapon page character mismatch')
                    character_mismatch_observations=0
                    if weapon_move is not None:
                        previous_name, posted_at = weapon_move
                        if name == previous_name:
                            # Arrows are asynchronous too. Never call a stale card a cycle.
                            if time.perf_counter_ns()-posted_at < 1_000_000_000:
                                continue
                            # Some characters have one available starting weapon.
                            # Select the observed card only, with incomplete coverage explicit.
                            if len(weapon_names)==1 and preferred_weapon is None:
                                preferred_weapon=name
                                context['weapon_scan_complete']=False
                                context['weapon_scan_reason']='only_one_observed_after_navigation_wait'
                            else:
                                raise OSError('Weapon navigation did not change observed selection')
                        weapon_move=None
                    repeated = bool(name and name in weapon_names)
                    if name and name not in weapon_names: weapon_names.append(name)
                    from .character_context import affinity
                    if name:
                        weapon_profiles[name]={'text':rows_in_region(ocr,(1200,180,1880,690)),
                                               'frame_ref':str(source),'frame_sha256':shot['frame_sha256']}
                    wanted=weapon.casefold().replace(' ','')
                    rotating = weapon.startswith('@rotate:')
                    rotation_match = rotating and len(weapon_names) > int(weapon.split(':', 1)[1])
                    if rotating and context.get('traits'):
                        cycle_complete = repeated and len(weapon_names)>1 and name==weapon_names[0]
                        if preferred_weapon is None and (cycle_complete or len(weapon_names)>=12):
                            offset=int(weapon.split(':',1)[1]) % len(weapon_names)
                            order=weapon_names[offset:]+weapon_names[:offset]
                            preferred_weapon=max(order,key=lambda n:affinity(context['traits'],' '.join(weapon_profiles[n]['text'])))
                            context['weapon_character_affinity']={n:affinity(context['traits'],' '.join(p['text'])) for n,p in weapon_profiles.items()}
                            context['weapon_choice_semantics']='display_trait_affinity_hint_not_optimal_build'
                            context['weapon_scan_complete']=cycle_complete
                        choose_weapon = name and preferred_weapon == name
                    else:
                        choose_weapon = name and (rotation_match or (not rotating and wanted in name.casefold().replace(' ',''))
                                 or repeated or len(weapon_names)>=12)
                    if choose_weapon:
                        context.update(weapons=[name],weapon_source=str(source),concept=concept or name+'-build',
                                       requested_weapon=weapon, observed_weapon_names=list(weapon_names),
                                       weapon_rotation_match=rotation_match if rotating else None)
                        key='enter'
                    else:
                        key=('left' if preferred_weapon in weapon_names and name in weapon_names
                             and weapon_names.index(preferred_weapon)<weapon_names.index(name) else 'right')
                        weapon_move=(name,time.perf_counter_ns())
                else:
                    raise OSError('Unrecognized setup screen')
                if recording and key=='enter': recording.event('menu_choice','다음 판의 캐릭터·무기 조건을 선택합니다.',context=dict(context))
                if time.perf_counter_ns()-shot['capture_started_at_ns']>2_500_000_000: raise OSError('Setup observation expired')
                controller.tap_menu(key)
                transition_gate.record(shot['frame_sha256'],'enter' if key=='escape' else key,posted_at_ns=time.perf_counter_ns())
            else:
                raise OSError('Setup observation budget exhausted')
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            if controller: controller.release()
            if recording:
                recording.event('stop','다음 판 조건 선택 기록을 저장합니다.')
                report=recording.close()
                if report.get('output_path'):
                    label_recording(media,context)
                    (media/'edit-pending.json').write_text(json.dumps({'recording_directory':str(media)}),encoding='utf-8')
    result={'context':context,'error':error,'visited':visited,'directory':str(directory)}
    (directory/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
