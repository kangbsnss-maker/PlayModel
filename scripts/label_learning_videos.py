"""Add curriculum titles to finalized recordings; no game or API access."""
if __name__ == '__main__':
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import json
from pathlib import Path
import time

from playmodel.execution_log import event, exception
from playmodel.instance import session_lock
from playmodel.learning_video_titles import label_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with session_lock(root / 'artifacts/video-titles.lock'):
        while True:
            try:
                changed = label_completed(root)
                if changed:
                    event('learning_video_titles_updated', count=len(changed),
                          catalog=str(root / 'media/learning-videos.html'))
                    print(json.dumps({'labelled': len(changed)}, ensure_ascii=False), flush=True)
            except (OSError, ValueError) as error:
                exception('learning_video_titles_failed', error)
                if not args.watch:
                    raise
            if not args.watch or (root / 'artifacts/VIDEO_TITLES_STOP').exists():
                break
            time.sleep(15)


if __name__ == '__main__':
    main()
