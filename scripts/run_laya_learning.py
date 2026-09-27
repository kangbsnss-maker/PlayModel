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
    from playmodel.laya.campaign import LearningCampaign
    from playmodel.games.brotato.tactical_runtime import TacticalCombatActor
    from playmodel.obs import ensure_obs_running
    from run_recurrent_cycle import LocalCycle, LocalStatus

    torch.set_num_threads(2)
    source = args.checkpoint.resolve()
    source_sha = digest(source)
    directory = args.output.resolve() / ('laya-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                         + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    args._last_session_directory = directory
    status = LocalStatus(directory)
    state = {'schema': 'playmodel.laya-session.v1', 'combat_checkpoint': str(source),
             'combat_checkpoint_sha256': source_sha, 'cnn_online_ppo': False,
             'laya_model_directory': str(args.model_dir.resolve()), 'reports': [], 'updates': [],
             'learning_target': 'Causal visual context and Laya decision weights from independent game outcomes',
             'combat_controller': 'asynchronous_laya_tactics_with_fresh_geometry',
             'fallback_controller': 'explicit_nonlearned_geometry',
             'evaluation_score_eligible': False, 'deployment_approved': False}
    def save():
        atomic_json(directory / 'laya-state.json', state, durable=True)
    try:
        evaluation = getattr(args, 'evaluation', False)
        campaign = (LearningCampaign(root / 'artifacts/local-learning/laya-campaign.json')
                    if getattr(args, 'campaign', False) and not evaluation else None)
        from playmodel.laya.combat_economy import CombatEconomy, fixed_evaluation_eligible
        economy = CombatEconomy(root / 'artifacts/local-learning/combat-economy')
        save()
        status.update(mode='laya_choice_learning', phase='laya_model_preload',
                      checkpoint=str(source), cnn_online_ppo=False)
        with LayaClient(root=root, output=directory / 'laya', model_dir=args.model_dir.resolve(),
                        device=args.device, seed=args.seed, checkpoint=args.laya_checkpoint) as raw_client, LayaBroker(raw_client) as client:
            status.update(phase='obs_start')
            ensure_obs_running()
            def fixed_configuration():
                navigation = root / 'artifacts/local-learning/ui-navigation.jsonl'
                return {'laya_version': client.version, 'economic_model': economy.model_evidence(),
                        'ui_graph_sha256': digest(navigation) if navigation.exists() else None,
                        'preference_hash': state.get('preference_application', {}).get('preference_hash')}
            def apply_preferences():
                if evaluation and state.get('preference_application') is not None:
                    return
                applied = client.apply_preferences(root / 'configs/local/laya-preferences.json')
                state['preference_application'] = applied
                save()
                status.update(laya_preferences=applied)
            def menu_factory(recorder, **kwargs):
                client.abandon('new_verified_run_boundary')
                apply_preferences()
                menu = LayaMenuController(recorder, client=client, **kwargs)
                economy.begin_run(recorder.run_id)
                menu.economy = economy
                if evaluation:
                    state.setdefault('fixed_configuration', fixed_configuration())
                menu.training_intent = campaign.current()['concept'] if campaign else 'survive and progress'
                return menu
            def terminal(kind, evidence):
                status.update(phase='laya_training')
                update = (client.abandon('fixed_evaluation_terminal') if evaluation
                          else client.finish(kind, evidence))
                economic_update = economy.finish(kind, evidence, learn=not evaluation)
                state['combat_economy'] = economic_update
                status.update(combat_economy=economic_update)
                state['updates'].append(update)
                save()
                apply_preferences()
                status.update(phase='menu', laya_last_update=update)
                return update
            def tactical_factory(recorder, output_directory, seed):
                from playmodel.games.brotato.tactical_state import TacticalPlanner
                return TacticalCombatActor(client.new_tactical_session(recorder.run_id),
                    planner=TacticalPlanner(training_intent=campaign.current()['concept'] if campaign else 'survive and progress'),
                    economy=economy)
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
                if campaign:
                    selection = campaign.current()
                    coordinator.character_slot, coordinator.weapon = selection['character_slot'], selection['weapon']
                    coordinator.campaign_concept = selection['concept']
                    state['campaign'] = selection
                    status.update(campaign=selection)
                coordinator.evaluation_scope_id = 'laya-adaptive-' + coordinator.operation_id
                # One physical learning run keeps one train split. Mixed-control
                # provenance explicitly blocks both standalone and full-run CNN PPO.
                report = coordinator.collect_run(source, split='evaluation' if evaluation else 'train',
                                                 tag='laya-evaluation' if evaluation else 'laya-learning')
                report['evaluation_score_eligible'] = bool(evaluation and fixed_evaluation_eligible(
                    report, state.get('fixed_configuration'), fixed_configuration()))
                report['fixed_configuration'] = state.get('fixed_configuration') if evaluation else None
                report['fixed_laya_behavior_version'] = client.version if evaluation else None
                state['reports'].append(report)
                client.abandon('run_boundary_or_interruption')
                save()
                if not report.get('full_run_complete'):
                    if coordinator.stop_file.exists():
                        status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
                        return 0
                    status.fault(report.get('error') or 'Laya run ended without verified death')
                    args._last_failure = report.get('error') or 'Laya run ended without verified death'
                    return 1
                count += 1
                if campaign:
                    campaign.record(report)
                status.update(status='running', phase='run_complete', completed_runs=count)
            else:
                status.update(status='completed', phase='stopped', completed_runs=count)
        return 0
    except Exception as error:
        state['error'] = f'{type(error).__name__}: {error}'
        args._last_failure = state['error']
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
    parser.add_argument('--campaign', action='store_true', help='persist character/weapon/concept rotation')
    parser.add_argument('--evaluation', action='store_true', help='freeze Laya, preferences and auxiliary models; no learning')
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
        result = run_laya(args, root)
        from playmodel.laya.safety_watch import watch_ui, user_stop
        stop = root / 'artifacts/BROTATO_STOP'
        failure = getattr(args, '_last_failure', '')
        if result and args.continuous and not stop.exists() and not user_stop(failure):
            from playmodel.games.brotato.installation import inspect_installation
            from run_recurrent_cycle import LocalStatus
            installation = next(item for item in inspect_installation()['installations']
                                if item['status'] == 'files_present')
            directory = args._last_session_directory
            status = LocalStatus(directory)
            try:
                return watch_ui(Path(installation['path']) / 'Brotato.exe', directory,
                    stop_file=stop, status=status, incident={
                        'category': 'unclassified_runtime_fault', 'original_error': failure,
                        'authority_revoked': None, 'writer_quiescent': None, 'release_state': 'unconfirmed',
                        'target_identity': None, 'evidence_refs': [str(directory / 'laya-state.json')],
                        'learning_quarantined': True, 'automatic_rearm_allowed': False})
            finally:
                status.close()
        return result


if __name__ == '__main__':
    raise SystemExit(main())
