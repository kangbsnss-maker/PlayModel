"""Create a reviewable highlight cut and external Korean subtitles from a local recording."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import json
from pathlib import Path
from playmodel.video import make_highlights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path, help='Session folder containing recording.json')
    args = parser.parse_args()
    result = make_highlights(args.directory)
    print(json.dumps({key: result[key] for key in ('highlights', 'original_seconds', 'captions')}, ensure_ascii=True))
