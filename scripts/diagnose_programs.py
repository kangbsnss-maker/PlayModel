"""Summarize latest local program failures without LLM or game input."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import json
from pathlib import Path
from playmodel.execution_log import ROOT, diagnose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    paths = [args.directory] if args.directory else sorted(
        (ROOT / 'artifacts/program-logs').glob('*'), key=lambda p: p.name, reverse=True)
    reports = []
    for path in paths:
        if len(reports) >= max(1, args.limit):
            break
        try:
            report = diagnose(path)
            if Path(report['program']).name != 'diagnose_programs.py':
                reports.append(report)
        except (OSError, ValueError, KeyError):
            continue
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
