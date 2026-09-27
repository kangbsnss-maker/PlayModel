"""Bounded, screen-verified next-run selection. No global cursor/focus calls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

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
    character_mismatch_observations=0
    error=None
    recovery_wait=None
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
                    repeated = bool(name and name in weapon_names)
                    if name and name not in weapon_names: weapon_names.append(name)
                    wanted=weapon.casefold().replace(' ','')
                    rotating = weapon.startswith('@rotate:')
                    rotation_match = rotating and len(weapon_names) > int(weapon.split(':', 1)[1])
                    if name and (rotation_match or (not rotating and wanted in name.casefold().replace(' ',''))
                                 or repeated or len(weapon_names)>=12):
                        context.update(weapons=[name],weapon_source=str(source),concept=concept or name+'-build',
                                       requested_weapon=weapon, observed_weapon_names=list(weapon_names),
                                       weapon_rotation_match=rotation_match if rotating else None)
                        key='enter'
                    else:
                        key='right'
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
