"""Create a context64 warm start from an immutable legacy context16 checkpoint.

No game access, optimizer, rollout relabeling, or automatic model deployment.
"""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    import torch
    from playmodel.learning.recurrent_ppo import migrate_build_state_checkpoint
    torch.set_num_threads(2)
    try:
        report = migrate_build_state_checkpoint(args.source, args.target)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "migration_failed", "error": str(error),
                          "training_performed": False, "deployment_approved": False}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "warm_start_created", **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
