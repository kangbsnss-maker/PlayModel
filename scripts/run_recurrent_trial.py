"""Run a bounded recorded neural combat experiment with existing menu controls.

Menu/build rules are still the explicit bootstrap baseline in this command.
Neural action samples are saved for separate PPO training; no model promotion.
"""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--waves', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output', type=Path, default=Path('artifacts/neural-sessions'))
    args = parser.parse_args()
    import torch
    from playmodel.learning.recurrent_ppo import load_checkpoint
    from playmodel.games.brotato.neural_runtime import run_neural_trial
    from playmodel.games.brotato.session import run_session
    from playmodel.games.brotato.installation import inspect_installation
    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.checkpoint, device='cpu')
    hidden = None
    trials = []
    def neural_runner(executable, output_root, *, policy=None, config, **kwargs):
        nonlocal hidden
        kwargs.pop('vision_factory', None)
        result = run_neural_trial(executable, output_root, model=model,
                    config=replace(config, train=False, defer_training=False),
                    initial_hidden=hidden, chunk_steps=32, burn_in=8, **kwargs)
        if result.get('last_hidden') is not None:
            hidden = torch.tensor(result['last_hidden'], dtype=torch.float32)
        trials.append(result)
        result['training_performed'] = False
        result['status'] = 'neural_rollout_ready' if result.get('rollout_path') else 'aborted'
        return result
    install = next(i for i in inspect_installation()['installations'] if i['status']=='files_present')
    context_path = Path('configs/local/current-run.json')
    context = json.loads(context_path.read_text(encoding='utf-8')) if context_path.exists() else {}
    context['experimental_partial_character_run'] = True
    report = run_session(Path(install['path'])/'Brotato.exe', args.output,
                         waves=args.waves, seconds=args.seconds, stop_file=Path('artifacts/BROTATO_STOP'),
                         ocr_script=Path('scripts/windows_ocr.ps1'), record=True, edit=False,
                         run_context=context, combat_runner=neural_runner)
    summary = {'session_directory':report['session_directory'], 'reason':report['reason'],
               'neural_checkpoint':str(args.checkpoint.resolve()), 'neural_policy_version':model.policy_version(),
               'menu_choices':'bootstrap_rules_not_learned', 'full_character_cycle_verified':False,
               'trials':[{'directory':trial.get('session_directory'), 'rollout_path':trial.get('rollout_path'),
                          'reason':trial.get('reason'), 'status':trial.get('status')} for trial in trials],
               'deployment_approved':False, 'performance_improvement_verified':False}
    Path(report['session_directory'],'neural-experiment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 0 if any(trial.get('rollout_path') for trial in trials) else 1


if __name__ == '__main__':
    raise SystemExit(main())
