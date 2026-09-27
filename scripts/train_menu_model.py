"""Train a local menu perception candidate from an explicitly reviewed manifest."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import json
from pathlib import Path

from playmodel.games.brotato.menu_model import train_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path, help='New checkpoint path; never overwrites an existing candidate')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--learning-rate', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    report = train_manifest(args.manifest, args.output, epochs=args.epochs,
                            learning_rate=args.learning_rate, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
