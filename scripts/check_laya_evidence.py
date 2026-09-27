"""Verify real-game Laya choice -> outcome -> update -> applied new choice."""
from pathlib import Path
import argparse
import json
from playmodel.laya.records import digest, verified_outcome, validate_tactical_application
import time


def verify(root):
    runs = []
    applied_by_version = {}
    for worker in (root / 'artifacts/laya-learning').glob('laya-*/laya'):
        for path in worker.glob('choice-*.json'):
            choice = json.loads(path.read_text(encoding='utf8'))
            accepted_path = worker / ('accepted-' + choice['decision_id'] + '.json')
            if not accepted_path.is_file():
                continue
            application = json.loads(accepted_path.read_text(encoding='utf8'))
            if choice.get('decision_domain') == 'combat_tactic':
                validate_tactical_application(choice, application, time.perf_counter_ns())
            elif application.get('game_application_verified') is not True:
                continue
            applied_by_version.setdefault(choice['behavior_version'], set()).add(choice['decision_id'])
    for directory in sorted((root / 'artifacts/laya-learning').glob('laya-*')):
        worker = directory / 'laya'
        choices = {p.stem[7:]: json.loads(p.read_text(encoding='utf8')) for p in worker.glob('choice-*.json')}
        accepted = {p.stem[9:]: json.loads(p.read_text(encoding='utf8')) for p in worker.glob('accepted-*.json')}
        for path in worker.glob('update-*/report.json'):
            report = json.loads(path.read_text(encoding='utf8'))
            if report.get('status') != 'updated' or report.get('accepted') is not True:
                continue
            assert report['encoder_hash_before'] == report['encoder_hash_after']
            assert report['head_hash_before'] != report['head_hash_after']
            assert report['optimizer_steps'] > 0
            assert digest(report['checkpoint']) == report['checkpoint_sha256']
            dataset = path.parent / 'dataset-manifest.json'
            assert digest(dataset) == report['dataset_manifest_sha256']
            manifest = json.loads(dataset.read_text(encoding='utf8'))
            assert manifest['split'] == 'train'
            for proof in manifest['files']:
                assert digest(proof['path']) == proof['sha256']
            _, observed, outcome = verified_outcome(report['verified_outcome']['kind'], report['outcome'])
            assert outcome['origin'] == 'local_detector'
            assert report['outcome'].get('run_id') in manifest['run_ids']
            for identifier in report['decisions']:
                choice, application = choices[identifier], accepted[identifier]
                assert choice['behavior_version'] == report['source_version']
                assert application['successful_transport_reported'] is True
                if choice.get('decision_domain') == 'combat_tactic':
                    validate_tactical_application(choice, application, time.perf_counter_ns())
                    assert application['game_application_verified'] is False
                    assert choice['decided_at_ns'] <= application['sent_at_ns'] < observed
                else:
                    assert application['game_application_verified'] is True
                    assert choice['decided_at_ns'] <= application['sent_at_ns'] < application['verified_at_ns'] < observed
            assert any(len(choices[i]['options']) > 1 for i in report['decisions'])
            applied = applied_by_version.get(report['behavior_version'], set())
            runs.append({'report': str(path), 'choices_trained': len(report['decisions']),
                         'combat_tactics_trained': sum(choices[i].get('decision_domain') == 'combat_tactic' for i in report['decisions']),
                         'new_version_applied_choices': len(applied), 'updated_version': report['behavior_version']})
    if not runs:
        raise ValueError('No verified real-game Laya weight update yet')
    if not any(r['new_version_applied_choices'] for r in runs):
        raise ValueError('Updated Laya version has no verified game application yet')
    print(json.dumps({'updates': runs, 'game_improvement_proven': False}, indent=2))
    print('LAYA_REAL_GAME_LEARNING_VERIFIED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().root)
