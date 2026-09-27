"""Freeze reusable training pixels for encoder-only self-supervised learning."""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
from pathlib import Path
import json

from playmodel.learning.replay_data import build_visual_manifest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--max-source-frames", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protected-manifest", type=Path, action="append", default=[])
    parser.add_argument("--group-splits", type=Path,
                        help="Frozen group split configuration established before training")
    args = parser.parse_args()
    document = build_visual_manifest(args.root, args.output, max_frames=args.max_frames,
                                     max_episodes=args.max_episodes, max_source_frames=args.max_source_frames,
                                     seed=args.seed, protected_manifests=tuple(args.protected_manifest),
                                     group_splits=args.group_splits)
    print(json.dumps({"manifest": str(args.output.resolve()), "sha256": sha256_file(args.output),
                      "samples": len(document["samples"]), "split_counts": document["split_counts"],
                      "statistics": document["statistics"], "purpose": document["purpose"]}, indent=2))


if __name__ == "__main__":
    main()
