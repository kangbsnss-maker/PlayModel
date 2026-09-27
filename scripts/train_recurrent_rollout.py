"""Train an experimental PPO candidate from a verified local neural rollout."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--burn-in', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 8 or not 0 <= args.burn_in <= 32:
        parser.error('epochs must be 1..8, burn-in 0..32')
    import torch
    from playmodel.learning.recurrent_ppo import load_checkpoint, save_checkpoint, ppo_update, PPOConfig
    from playmodel.games.brotato.neural_runtime import load_rollout
    from playmodel.learning.full_run import chunk_full_trajectory
    torch.set_num_threads(2)
    model, metadata = load_checkpoint(args.checkpoint, device=args.device)
    # Validate the collector's complete frozen manifest before reading any tensor.
    load_rollout(args.rollout, device='cpu')
    flat_path = args.rollout.parent/'flat-rollout.pt'
    flat = load_rollout(flat_path, device='cpu')
    states = torch.load(args.rollout.parent/'initial-states.pt', map_location='cpu', weights_only=True)
    rollout, indices = chunk_full_trajectory(flat, states['initial_states'],
                                            chunk_steps=32, burn_in=args.burn_in)
    rollout = rollout.to(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = ppo_update(model, rollout, PPOConfig(epochs=args.epochs, minibatch_sequences=2,
                                                burn_in=args.burn_in, gamma=.997,
                                                discount_time_unit_seconds=1.0),
                        trajectory=flat, transition_indices=indices)
    result.update(elapsed_seconds=time.perf_counter()-started,
                  source_checkpoint=str(args.checkpoint.resolve()),
                  source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  rollout_path=str(args.rollout.resolve()),
                  rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
                  gae_scope='complete_chronological_trajectory_before_chunking',
                  gpu_peak_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    checkpoint = args.output/'ppo-candidate.pt'
    save_checkpoint(model, checkpoint, metadata={**metadata, 'purpose':'experimental_ppo_candidate',
                    'source_checkpoint_sha256':result['source_sha256'],
                    'rollout_sha256':result['rollout_sha256'], 'ppo':result,
                    'deployment_approved':False, 'gameplay_improvement_verified':False})
    restored, _ = load_checkpoint(checkpoint)
    if restored.policy_version() != model.policy_version():
        raise ValueError('Candidate reload mismatch')
    result['checkpoint_reload_verified'] = True
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
