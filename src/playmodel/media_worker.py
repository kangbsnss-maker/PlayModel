"""Low-priority local editing service. Separate from game control and model training."""

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module('playmodel.media_worker')

import json
import os
from pathlib import Path
import time
import uuid
from .instance import session_lock
from .video import make_highlights


def main():
    root=Path(__file__).resolve().parents[2]
    with session_lock(root/'artifacts/media-worker.lock'):
        while not (root/'artifacts/MEDIA_STOP').exists():
            for pending in sorted((root/'media/captures').glob('*/edit-pending.json')):
                directory=pending.parent
                if (directory/'edit-done.json').exists() or (directory/'edit-failed.json').exists(): continue
                target=directory/('auto-edit-'+uuid.uuid4().hex[:6])
                try:
                    result=make_highlights(directory,output_directory=target)
                    (directory/'edit-done.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                except Exception as error:
                    (directory/'edit-failed.json').write_text(json.dumps({'error':str(error),'output_directory':str(target)},ensure_ascii=False),encoding='utf-8')
            (root/'artifacts/media-worker-status.json').write_text(json.dumps({'pid':os.getpid(),'updated_at':time.time(),'state':'idle'}),encoding='utf-8')
            time.sleep(5)


if __name__=='__main__': main()
