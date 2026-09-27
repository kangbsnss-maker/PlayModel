"""Bounded, screen-verified next-run selection. No global cursor/focus calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from .background import BackgroundController
from .capture import capture_session,read_diagnostic_png
from .ocr import MenuOcr,rows_in_region
from .menu import classify_scene,selected_button,navigation_key,BUTTONS
from playmodel.video import SessionRecording
from playmodel.media_titles import label_recording
from playmodel.instance import session_lock
from .menu_capture import MenuCapture
from .menu_transition import MenuTransitionGate


CHARACTER_GRID = {i:(69+(i%17)*106,723+(i//17)*106,165+(i%17)*106,819+(i//17)*106) for i in range(51)}


def focused_tile(pixels: bytes, width: int, boxes: dict) -> int | None:
    scores=[]
    for index,(left,top,right,bottom) in boxes.items():
        samples=[(x,y) for x in (left+7,right-8) for y in range(top+22,bottom-22,3)]
        samples += [(x,y) for y in (top+7,bottom-8) for x in range(left+22,right-22,3)]
        colors=[pixels[(y*width+x)*4:(y*width+x)*4+3] for x,y in samples]
        score=sum(145<=min(c)<=max(c)<=245 and max(c)-min(c)<=18 for c in colors)/len(colors)
        scores.append((score,index))
    scores.sort(reverse=True)
    return scores[0][1] if scores[0][0]>.65 and scores[1][0]<.35 else None


def prepare_next(executable: Path, *, root: Path, character_slot: int, weapon: str, record: bool) -> dict:
    if not 1<=character_slot<=50: raise ValueError('Visible calibrated character slot must be 1..50')
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-setup-'+uuid.uuid4().hex[:6]
    directory=root/'artifacts/run-setup'/token
    media=root/'media/captures'/token
    directory.mkdir(parents=True)
    controller=None
    recording=None
    context={'character_slot':character_slot,'concept':weapon+'-build','endless_requested':True,
             'character':'Unknown','weapons':[],'difficulty':'Nightmare'}
    visited=[]
    attempted_slot=None
    weapon_names=[]
    character_mismatch_observations=0
    error=None
    transition_gate=MenuTransitionGate()
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
                if (w,h)!=(1920,1080): raise OSError('Setup resolution changed')
                if controller is None: controller=BackgroundController(shot['hwnd'],executable)
                header=''.join(rows_in_region(ocr,(500,60,1450,160))).casefold()
                scene=classify_scene(ocr).scene
                if scene=='result':
                    focus=selected_button(pixels,w,h,scene='result')
                    if focus.selected_id is None: raise OSError('Result focus unknown')
                    key='enter' if focus.selected_id=='new_run' else navigation_key(focus.rect,BUTTONS['result']['new_run'])
                elif 'characterselection' in header:
                    slot=focused_tile(pixels,w,CHARACTER_GRID)
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
                                       traits=rows_in_region(ocr,(520,285,975,690)))
                        # The observed ON switch is white on the right. Do not silently run normal mode.
                        switch=pixels[(320*w+1640)*4:(320*w+1640)*4+3]
                        if min(switch)<180: raise OSError('Endless option not confirmed enabled')
                        context['endless_verified']=True
                        key='enter'
                        attempted_slot=slot
                elif 'selection' in header and ('diffic' in header):
                    # Leave on this screen: the recorded gameplay session selects highest difficulty.
                    context['difficulty_menu_source']=str(source)
                    context['setup_complete']=True
                    break
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
                    repeated = bool(name and name in weapon_names)
                    if name and name not in weapon_names: weapon_names.append(name)
                    wanted=weapon.casefold().replace(' ','')
                    if name and (wanted in name.casefold().replace(' ','') or repeated or len(weapon_names)>=12):
                        context.update(weapons=[name],weapon_source=str(source),concept=name+'-build')
                        key='enter'
                    else:
                        key='right'
                else:
                    raise OSError('Unrecognized setup screen')
                if recording and key=='enter': recording.event('menu_choice','다음 판의 캐릭터·무기 조건을 선택합니다.',context=dict(context))
                if time.perf_counter_ns()-shot['capture_started_at_ns']>2_500_000_000: raise OSError('Setup observation expired')
                controller.tap_menu(key)
                transition_gate.record(shot['frame_sha256'],key,posted_at_ns=time.perf_counter_ns())
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
