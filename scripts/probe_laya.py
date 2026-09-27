"""Explicit synthetic integration test; never game training or a resumable run."""
from pathlib import Path
import json
import time
import uuid
from playmodel.laya.client import LayaClient
from playmodel.laya.records import digest


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'artifacts/laya-probe' / uuid.uuid4().hex
    output.mkdir(parents=True)
    frame = output / 'SYNTHETIC-FRAME.fixture'
    frame.write_bytes(b'Synthetic integration fixture. Not a game observation or training dataset.')
    with LayaClient(root, output / 'worker', root / 'models/laya/base') as client:
        now = time.perf_counter_ns()
        result = client.choose({'scene': 'shop', 'health': 'unknown', 'money': 10},
            {'a': 'Save money', 'b': 'Buy armor. This synthetic option deliberately exceeds the upstream option limit. '
             'Keep every described effect in the model input: armor increases by one; damage stays unchanged; '
             'health regeneration stays unchanged; harvesting stays unchanged; attack speed stays unchanged; '
             'ranged damage stays unchanged; maximum health decreases by five.'},
            {'frame_ref': str(frame), 'frame_sha256': digest(frame),
             'observed_at_ns': now, 'available_at_ns': now, 'run_id': 'synthetic_test_only'})
        sent = time.perf_counter_ns()
        verified = time.perf_counter_ns()
        client.accept(result['decision_id'], {'accepted': True, 'sent_at_ns': sent,
            'successful_transport_reported': True, 'game_application_verified': True,
            'verified_at_ns': verified, 'actual_target': result['action_id'],
            'after_frames': [{'frame_ref': str(frame), 'frame_sha256': digest(frame)}]})
        event = output / 'SYNTHETIC-TERMINAL.json'
        event.write_text(json.dumps({'kind': 'wave_clear', 'verified': True, 'independent_of_policy': True,
            'origin': 'synthetic_integration_fixture', 'observed_at_ns': time.perf_counter_ns(),
            'frame_ref': str(frame), 'frame_sha256': digest(frame)}), encoding='utf8')
        update = client.finish('wave_clear', {'path': str(event), 'sha256': digest(event)})
        assert update['status'] == 'updated', update
        assert update['head_hash_before'] != update['head_hash_after']
        assert update['encoder_hash_before'] == update['encoder_hash_after']
        checkpoint = update['checkpoint']
    with LayaClient(root, output / 'reload', root / 'models/laya/base', checkpoint=checkpoint) as client:
        assert client.ready['behavior_version'] == update['behavior_version']
    report = {'synthetic_test_only': True, 'game_training_performed': False,
              'initial_choice': result, 'update': update, 'reload_version_verified': True}
    (output / 'probe.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('LAYA_SYNTHETIC_INTEGRATION_PASS', output, flush=True)


if __name__ == '__main__':
    main()
