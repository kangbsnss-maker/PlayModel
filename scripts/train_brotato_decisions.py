"""Make one unapproved local high-level RL candidate from a frozen full run."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import json
from pathlib import Path

from playmodel.games.brotato.decision_learning import train_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--value-learning-rate", type=float, default=0.1)
    parser.add_argument("--max-kl", type=float, default=0.02)
    args = parser.parse_args(argv)
    report = train_run(args.manifest, args.source_checkpoint, args.output,
                       learning_rate=args.learning_rate,
                       value_learning_rate=args.value_learning_rate, max_kl=args.max_kl)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
