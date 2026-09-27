"""Local Laya menu/tactical learning with asynchronous choices and control evidence."""
from __future__ import annotations

if __name__ == '__main__':
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def run_laya(args, root):
    import torch
    from playmodel.atomic_io import atomic_json
    from playmodel.games.brotato.laya_menu import LayaMenuController
    from playmodel.laya.client import LayaClient
    from playmodel.laya.broker import LayaBroker
    from playmodel.games.brotato.tactical_runtime import TacticalCombatActor
    from playmodel.obs import ensure_obs_running
    from run_recurrent_cycle import LocalCycle, LocalStatus

    torch.set_num_threads(2)
    source = args.checkpoint.resolve()
    source_sha = digest(source)
    directory = args.output.resolve() / ('laya-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                         + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    status = LocalStatus(directory)
    state = {'schema': 'playmodel.laya-session.v1', 'combat_checkpoint': str(source),
             'combat_checkpoint_sha256': source_sha, 'cnn_online_ppo': False,
             'laya_model_directory': str(args.model_dir.resolve()), 'reports': [], 'updates': [],
             'learning_target': 'Laya menu and combat tactic decision head from independent game outcomes',
             'combat_controller': 'asynchronous_laya_tactics_with_fresh_geometry',
             'fallback_controller': 'explicit_nonlearned_geometry',
             'evaluation_score_eligible': False, 'deployment_approved': False}
    def save():
        atomic_json(directory / 'laya-state.json', state, durable=True)
    try:
        save()
        status.update(mode='laya_choice_learning', phase='laya_model_preload',
                      checkpoint=str(source), cnn_online_ppo=False)
        with LayaClient(root=root, output=directory / 'laya', model_dir=args.model_dir.resolve(),
                        device=args.device, seed=args.seed, checkpoint=args.laya_checkpoint) as raw_client, LayaBroker(raw_client) as client:
            status.update(phase='obs_start')
            ensure_obs_running()
            def apply_preferences():
                applied = client.apply_preferences(root / 'configs/local/laya-preferences.json')
                state['preference_application'] = applied
                save()
                status.update(laya_preferences=applied)
            def menu_factory(recorder, **kwargs):
                client.abandon('new_verified_run_boundary')
                apply_preferences()
                return LayaMenuController(recorder, client=client, **kwargs)
            def terminal(kind, evidence):
                status.update(phase='laya_training')
                update = client.finish(kind, evidence)
                state['updates'].append(update)
                save()
                apply_preferences()
                status.update(phase='menu', laya_last_update=update)
                return update
            def tactical_factory(recorder, output_directory, seed):
                return TacticalCombatActor(client.new_tactical_session(recorder.run_id))
            coordinator = LocalCycle(root=root, output=directory,
                character_slot=args.character_slot, weapon=args.weapon,
                max_run_seconds=args.max_run_seconds, device=args.device, seed=args.seed,
                status=status, recover_active_run=args.recover_active_run,
                menu_factory=menu_factory, terminal_callback=terminal,
                tactical_factory=tactical_factory)
            count = 0
            while args.continuous or count < args.cycles:
                if coordinator.stop_file.exists():
                    status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                    break
                if digest(source) != source_sha:
                    raise ValueError('Frozen CNN source checkpoint changed')
                coordinator.operation_id = uuid.uuid4().hex
                coordinator.evaluation_scope_id = 'laya-adaptive-' + coordinator.operation_id
                # One physical learning run keeps one train split. Mixed-control
                # provenance explicitly blocks both standalone and full-run CNN PPO.
                report = coordinator.collect_run(source, split='train', tag='laya-learning')
                report['evaluation_score_eligible'] = False
                state['reports'].append(report)
                client.abandon('run_boundary_or_interruption')
                save()
                if not report.get('full_run_complete'):
                    if coordinator.stop_file.exists():
                        status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                        return 0
                    status.fault(report.get('error') or 'Laya run ended without verified death')
                    return 1
                count += 1
                status.update(status='running', phase='run_complete', completed_runs=count)
            else:
                status.update(status='completed', phase='stopped', completed_runs=count)
        return 0
    except Exception as error:
        state['error'] = f'{type(error).__name__}: {error}'
        save()
        status.fault(state['error'])
        return 1
    finally:
        status.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='fixed CNN combat checkpoint')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--laya-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/laya-learning'))
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--character-slot', type=int, default=1)
    parser.add_argument('--weapon', default='SMG')
    parser.add_argument('--max-run-seconds', type=int, default=1800)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--recover-active-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= 5 or not 1 <= args.character_slot <= 50 or not 30 <= args.max_run_seconds <= 7200:
        parser.error('cycles 1..5, character-slot 1..50, max-run-seconds 30..7200 required')
    root = Path(__file__).resolve().parents[1]
    from playmodel.instance import session_lock
    with session_lock(root / 'artifacts/local-learning/worker.lock'):
        return run_laya(args, root)


if __name__ == '__main__':
    raise SystemExit(main())
