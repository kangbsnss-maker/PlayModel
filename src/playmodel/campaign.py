"""Local continuing training coordinator. Segments are checkpoints, not task completion."""
from __future__ import annotations

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module('playmodel.campaign')


import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .games.brotato.installation import inspect_installation
from .games.brotato.session import run_session
from .games.brotato.setup_run import prepare_next
from .instance import session_lock


def launch_worker(root: Path, *, media: bool = False):
    status=root/('artifacts/media-worker-status.json' if media else 'artifacts/learning/worker-status.json')
    if status.exists():
        try:
            if time.time()-json.loads(status.read_text(encoding='utf-8'))['updated_at']<10:
                return None
        except (ValueError,KeyError,OSError):
            pass
    log = (root/('artifacts/media-worker.log' if media else 'artifacts/learning-worker.log')).open('ab')
    try:
        return subprocess.Popen([sys.executable,'-X','utf8','-m','playmodel.media_worker' if media else 'playmodel.learning.worker'],cwd=root,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)
            |getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    finally:
        log.close()


def write_state(path: Path, state: dict):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--style',type=Path,default=Path('configs/styles/balanced.json'))
    parser.add_argument('--context',type=Path,default=Path('configs/local/current-run.json'))
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    os.chdir(root)
    path=root/'artifacts/campaign/status.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    stop=root/'artifacts/BROTATO_STOP'
    queue=root/'artifacts/learning/queue.sqlite3'
    context=json.loads(args.context.read_text(encoding='utf-8')) if args.context.exists() else {}
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    installation=next(i for i in inspect_installation()['installations'] if i['status']=='files_present')
    state={'schema':'playmodel.campaign.v1','pid':os.getpid(),'state':'starting','segments':previous.get('segments',[]),
           'objective':'all_characters_highest_difficulty_endless_personal_best',
           'continue_after_stage_clear':True,'automatic_agent_calls':False,'completed':False,
           'roster_coverage_verified':False,'context':context}
    with session_lock(root/'artifacts/campaign/owner.lock'):
        launch_worker(root)
        launch_worker(root,media=True)
        if previous.get('setup_directory') and previous.get('state')=='needs_review':
            pending=json.loads((Path(previous['setup_directory'])/'report.json').read_text(encoding='utf-8'))
            setup=prepare_next(Path(installation['path'])/'Brotato.exe',root=root,
                               character_slot=pending['context']['character_slot'],weapon='SMG',record=args.record)
            if setup['error']:
                state.update(state='needs_review',reason=setup['error'],setup_directory=setup['directory'],updated_at=time.time())
                write_state(path,state)
                return
            context=setup['context']
            args.context.write_text(json.dumps(context,ensure_ascii=False,indent=2),encoding='utf-8')
            state['context']=context
        recoveries=0
        while not stop.exists():
            if shutil.disk_usage(root).free < 10*1024**3:
                state.update(state='waiting_for_disk_space',reason='less_than_10_GiB_free')
                break
            state.update(state='playing',updated_at=time.time())
            write_state(path,state)
            report=run_session(Path(installation['path'])/'Brotato.exe',root/'artifacts/brotato-sessions',
                waves=10,seconds=600,stop_file=stop,ocr_script=root/'scripts/windows_ocr.ps1',
                style_path=args.style,record=args.record,edit=False,learning_queue=queue,run_context=context)
            state['segments'].append({'directory':report['session_directory'],'reason':report['reason'],
                                      'jobs':report.get('background_learning_jobs',[])})
            if (report.get('recording') or {}).get('output_path'):
                # Editing is queued separately from movement. Media worker integration is explicit.
                media_path=Path(report['recording']['output_path']).parent.parent
                (media_path/'edit-pending.json').write_text(json.dumps({'recording_directory':str(media_path)}),encoding='utf-8')
            if report['reason'] in ('wave_limit','segment_limit'):
                context=report['run_context']
                args.context.write_text(json.dumps(context,ensure_ascii=False,indent=2),encoding='utf-8')
                state['context']=context
                recoveries=0
                continue
            if report['reason']=='run_finished':
                result=report['run_context']
                ledger=root/'artifacts/campaign/run-results.jsonl'
                with ledger.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'context':result,'session':report['session_directory'],
                                             'completed_at':time.time()},ensure_ascii=False)+'\n')
                state.update(state='selecting_next_run',reason='result_recorded')
                write_state(path,state)
                next_slot=1+(int(context.get('character_slot',1))%50)
                concepts=['SMG','Stick','Fist','Laser Gun','Wrench','Wand','Shotgun','Knife']
                completed_runs=sum(1 for line in ledger.read_text(encoding='utf-8').splitlines() if line.strip())
                planned_weapon=concepts[(completed_runs//50)%len(concepts)]
                setup=prepare_next(Path(installation['path'])/'Brotato.exe',root=root,
                                   character_slot=next_slot,weapon=planned_weapon,record=args.record)
                if setup['error']:
                    state.update(state='needs_review',reason=setup['error'],setup_directory=setup['directory'])
                    break
                context=setup['context']
                args.context.write_text(json.dumps(context,ensure_ascii=False,indent=2),encoding='utf-8')
                state['context']=context
                recoveries=0
                continue
            if report['reason'] in ('screen_changed','perception_timeout','controller_guard') and recoveries<2:
                recoveries+=1
                time.sleep(1)
                continue
            state.update(state='needs_review',reason=report['reason'])
            break
        if stop.exists(): state.update(state='stopped_by_user')
        state['updated_at']=time.time()
        write_state(path,state)


if __name__=='__main__':
    from playmodel.execution_log import run_logged
    run_logged(main, program='playmodel.campaign')
