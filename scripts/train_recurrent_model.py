"""Pretrain the visual encoder on preserved frames and measure local resources.

Old actions are never used as PPO or behavior-cloning targets. This produces an
experimental checkpoint, not a gameplay performance certificate or deployment.
"""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--resume', type=Path, help='Continue encoder learning while retaining learned policy/GRU weights')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch_size <= 64:
        parser.error('steps must be 1..10000 and batch-size 1..64')
    # Optional heavyweight imports never affect the preparation-only CLI.
    import torch
    from torch.utils.data import DataLoader
    from playmodel.learning.recurrent_ppo import (
        ModelConfig, RecurrentActorCritic, pretrain_encoder, save_checkpoint, load_checkpoint, reconstruction_loss,
    )
    from playmodel.learning.replay_data import VisualReplayDataset, register_training_use

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose CPU explicitly')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    dataset = VisualReplayDataset(args.manifest, split='train', size=96, expected_manifest_sha256=source_hash)
    if not len(dataset):
        raise ValueError('No eligible train frames; no checkpoint written')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    def batches():
        while True:
            yield from loader
    model = load_checkpoint(args.resume, device=args.device)[0] if args.resume else RecurrentActorCritic(ModelConfig()).to(args.device)
    initial_version = model.policy_version()
    heldout = {split: VisualReplayDataset(args.manifest, split=split, size=96,
                                         expected_manifest_sha256=source_hash)
               for split in ('validation', 'test')}
    def evaluate():
        results = {}
        model.eval()
        with torch.inference_mode():
            for split, frames in heldout.items():
                total, count = 0.0, 0
                for batch in DataLoader(frames, batch_size=args.batch_size, num_workers=0):
                    total += reconstruction_loss(model, batch.to(args.device)).item() * len(batch)
                    count += len(batch)
                results[split] = {'frames': count, 'mse': total/count if count else None}
        return results
    heldout_before = evaluate()
    register_training_use(args.manifest, run_id=str(args.output.resolve()))
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    training = pretrain_encoder(model, batches(), steps=args.steps, device=args.device)
    elapsed = time.perf_counter() - started
    memory = ({'allocated_peak_bytes': torch.cuda.max_memory_allocated(),
               'reserved_peak_bytes': torch.cuda.max_memory_reserved()}
              if args.device == 'cuda' else {})
    heldout_after = evaluate()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != source_hash:
        raise ValueError('Training manifest changed during execution')
    checkpoint = args.output / 'encoder-pretrained.pt'
    metadata = {'purpose': 'visual_encoder_pretraining', 'algorithm': 'self_supervised_reconstruction',
                'manifest': str(args.manifest.resolve()), 'manifest_sha256': source_hash,
                'seed': args.seed, 'old_actions_used_as_targets': False,
                'resume_checkpoint_sha256': hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                'deployment_approved': False, 'gameplay_improvement_verified': False}
    save_checkpoint(model, checkpoint, metadata=metadata)
    restored, restored_metadata = load_checkpoint(checkpoint, device='cpu')
    if restored.policy_version() != model.policy_version():
        raise ValueError('Checkpoint reload changed model parameters')
    # Inference measurements include model inference only, not capture/OCR/input.
    images = torch.zeros(1, 3, 96, 96, dtype=torch.uint8)
    context = torch.zeros(1, restored.config.context_dim)
    phase = torch.zeros(1, dtype=torch.long)
    candidates = torch.zeros(1, 9, restored.config.candidate_dim)
    legal = torch.ones(1, 9, dtype=torch.bool)
    reset = torch.ones(1, dtype=torch.bool)
    durations = []
    restored.eval()
    with torch.inference_mode():
        for index in range(35):
            start = time.perf_counter()
            output = restored.step(images, context, phase, candidates, legal, reset=reset)
            if not torch.isfinite(output.value).all():
                raise ValueError('Nonfinite inference value')
            if index >= 5:
                durations.append((time.perf_counter()-start)*1000)
    report = {**metadata, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'device': args.device, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'train_frames': len(dataset), 'batch_size': args.batch_size, 'steps_requested': args.steps,
              'training': training, 'elapsed_seconds': elapsed, 'gpu_memory': memory,
              'heldout_reconstruction_before': heldout_before, 'heldout_reconstruction_after': heldout_after,
              'parameters': sum(p.numel() for p in model.parameters()),
              'initial_version': initial_version, 'final_version': model.policy_version(),
              'weights_changed': initial_version != model.policy_version(),
              'checkpoint_reload_verified': restored_metadata == metadata,
              'cpu_inference_only_ms': {'median': statistics.median(durations),
                                        'p95': sorted(durations)[int(len(durations)*.95)-1],
                                        'samples': len(durations), 'threads': 2},
              'evaluation_status': 'resource_and_reconstruction_checks_not_gameplay_evaluation'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
