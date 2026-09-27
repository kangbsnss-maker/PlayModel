"""Local combat summaries, frozen observational value regression and curriculum.

Terminal calls occur after actor/writer join and verified Laya outcome admission.
Original action ledgers remain immutable; derived model revisions are append-only.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

from playmodel.learning.outcome_values import OutcomeValues
from playmodel.games.brotato.evasion_dataset import examples as evasion_examples, features as evasion_features
from .records import canonical, digest, verified_outcome


def fixed_evaluation_eligible(report, expected, current):
    return bool(expected and expected == current and report.get('full_run_complete') is True
                and not report.get('error') and not report.get('observation_gaps')
                and report.get('scheduling_recovery_count') == 0
                and not any(row.get('status') == 'excluded' for row in report.get('choice_updates', [])))


class CombatEconomy:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        states = sorted(self.directory.glob('revision-*.json'))
        state = json.loads(states[-1].read_text(encoding='utf8')) if states else {}
        self.values = OutcomeValues(state.get('values'))
        self.evasion = OutcomeValues(state.get('evasion'))
        self.motion = state.get('motion', {'velocity_scale': 1., 'samples': 0})
        self.revision = int(state.get('revision', 0))
        self.used = set(state.get('used_outcomes', []))
        self.history, self.run_id = [], None
        self.summary = {}
        self.snapshot_path = self.directory / ('snapshot-' + uuid.uuid4().hex + '.json')
        self._snapshot()

    def _snapshot(self):
        self.snapshot_path = self.directory / ('snapshot-' + uuid.uuid4().hex + '.json')
        with self.snapshot_path.open('x', encoding='utf8') as stream:
            json.dump({'values': self.values.state(), 'evasion': self.evasion.state(), 'motion': self.motion, 'model_hash': self.model_hash},
                      stream, allow_nan=False)

    def model_evidence(self):
        return {'path': str(self.snapshot_path.resolve()), 'sha256': digest(self.snapshot_path),
                'model_hash': self.model_hash}

    @property
    def model_hash(self):
        return hashlib.sha256(canonical({'values': self.values.state(), 'evasion': self.evasion.state(), 'motion': self.motion}).encode()).hexdigest()

    def begin_run(self, run_id):
        if self.run_id != run_id:
            self.run_id, self.history, self.summary = run_id, [], {}

    def abandon(self):
        self.history, self.summary = [], {}

    def menu_context(self):
        return {'wave_memory': self.summary, 'value_revision': self.revision,
                'value_model_hash': self.model_hash,
                'value_semantics': 'observational_outcome_not_causal_item_effect'}

    def features(self, state, option):
        return {'choice': option, 'wave': state.get('wave'), 'currency': state.get('currency'),
                'intent': state.get('training_intent'), 'weapons': ','.join(state.get('weapon_names', [])),
                **{f'stat:{key}': val for key,val in state.get('last_observed_stats', {}).items()},
                **{f'combat:{key}': val for key,val in self.summary.items() if isinstance(val,(int,float))}}

    def estimates(self, state, options):
        return {key: round(self.values.predict(self.features(state, text)), 3)
                for key,text in options.items()}

    def record_purchase(self, record, application):
        # Called only for a verified menu application, never hover or failed buy.
        if application.get('game_application_verified') is not True:
            raise ValueError('Economic choice requires verified application')
        if record['evidence']['run_id'] != self.run_id:
            raise ValueError('Economic choice run mismatch')
        row = {'decision_id': record['decision_id'], 'run_id': self.run_id,
               'features': self.features(record['state'], record['options'][record['action_id']]),
               'verified_at_ns': application['verified_at_ns'], 'age_waves': 0,
               'choice_sha256': digest(record['_source_path']),
               'choice_path': record['_source_path']}
        if not any(r['decision_id'] == row['decision_id'] for r in self.history):
            self.history.append(row)

    def finish(self, kind, evidence, *, learn=True):
        reward, observed, _ = verified_outcome(kind, evidence)
        if evidence['sha256'] in self.used:
            raise ValueError('Economic outcome already consumed')
        report_path = Path(evidence['report_path'])
        if digest(report_path) != evidence['report_sha256']:
            raise ValueError('Economic combat report changed')
        report = json.loads(report_path.read_text(encoding='utf8'))
        if (report.get('run_id') != self.run_id or evidence.get('run_id') != self.run_id
                or report.get('verified_terminal_boundary') is not True
                or report.get('recorder_complete') is not True or report.get('worker_stopped') is not True):
            raise ValueError('Economic outcome lacks verified run boundary')
        actions = Path(report['actions_path'])
        if digest(actions) != report['actions_sha256']:
            raise ValueError('Economic actions changed')
        rows = [json.loads(line) for line in actions.read_text(encoding='utf8').splitlines() if line.strip()]
        valid = [r for r in rows if r.get('sent_at_ns', observed) < observed and
                 r.get('successful_transport_reported') is True and r.get('combat_experience', {}).get('valid')]
        samples = [r['combat_experience'] for r in valid]
        dropped, pickup_missing, tracks, hp = 0., 0, set(), []
        dot = norm = 0.
        pairs = 0
        source_times = {r['combat_experience']['observed_at_ns'] for r in valid}
        checked_models = set()
        for row in valid:
            frame = actions.parent / row['frame_ref']
            if digest(frame) != row['frame_sha256']:
                raise ValueError('Economic source frame changed')
            sample = row['combat_experience']
            model = sample.get('auxiliary_model')
            if model and model['path'] not in checked_models:
                if digest(model['path']) != model['sha256']:
                    raise ValueError('Motion model snapshot changed')
                checked_models.add(model['path'])
            if not sample['observed_at_ns'] <= sample['available_at_ns'] <= row['sent_at_ns']:
                raise ValueError('Noncausal combat experience')
            dropped += sample.get('hp_drop_hypothesis') or 0.
            pickup_missing += sample['pickup_disappearances']
            for obj in sample['objects']:
                tracks.add(obj['track_id'])
                if obj['hp_observation'].get('ratio') is not None:
                    hp.append(obj['hp_observation']['ratio'])
            for pair in sample['motion_pairs']:
                if not pair['input_observed_at_ns'] < pair['target_observed_at_ns'] == sample['observed_at_ns']:
                    raise ValueError('Noncausal motion target')
                if pair['input_observed_at_ns'] not in source_times:
                    continue  # No preserved source frame for this training pair.
                v, target = pair['input_velocity'], pair['target_velocity']
                dot += sum(a*b for a,b in zip(v,target))
                norm += sum(a*a for a in v)
                pairs += 1
        summary = {'tracked': len(tracks), 'hp_drop_proxy': round(dropped,3),
                   'pickup_missing_proxy': pickup_missing, 'hp_bar_candidates': len(hp),
                   'clear': kind == 'wave_clear', 'samples': len(samples),
                   'kills': None, 'collected': None, 'enemy_hp': None}
        summary['target_hp_drop_proxy'] = round(sum(s.get('target_hp_drop_proxy',0) for s in samples),3)
        lifetimes = [o['visible_seconds'] for s in samples for o in s['objects']]
        summary['visible_duration_proxy'] = round(sum(lifetimes)/len(lifetimes),2) if lifetimes else None
        if not learn:
            self.summary = summary
            return {'status': 'fixed_evaluation', 'summary': summary, 'model_hash': self.model_hash,
                    'evaluation_score_eligible': True, 'model_updates': 0}
        examples, credited = [], []
        for row in self.history:
            if row['verified_at_ns'] >= observed:
                continue
            if digest(row['choice_path']) != row['choice_sha256']:
                raise ValueError('Economic choice changed')
            # Older versions train a separate value estimator, not policy-gradient replay.
            examples.append((row['features'], reward * (.9 ** row['age_waves'])))
            credited.append(deepcopy(row))
            row['age_waves'] += 1
        before = self.model_hash
        evasion_rows = evasion_examples(valid, self.run_id)
        evasion_training = [(evasion_features(r), math.tanh(r['target']['clearance_proxy']))
                            for r in evasion_rows if r['target']['clearance_proxy'] is not None]
        # Auxiliary prediction only. This is not a safe-action label or policy reward.
        self.evasion.fit(evasion_training)
        dataset_path = self.directory / ('evasion-' + uuid.uuid4().hex + '.jsonl')
        with dataset_path.open('x', encoding='utf8') as stream:
            for row in evasion_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        self.values.fit(examples)
        if pairs >= 8 and norm > 1e-6:
            self.motion = {**self.motion, 'velocity_scale': max(.5, min(1.5, dot/norm)),
                           'samples': self.motion['samples'] + pairs,
                           'semantics': 'self_supervised_track_motion_not_verified_projectile_identity'}
        decay = [max(0.,p['input']-p['target'])/p['dt_seconds'] for sample in samples
                 for p in sample.get('bar_pairs', []) if .005 <= p['dt_seconds'] <= .3
                 and p['input_observed_at_ns'] in source_times
                 and p['target_observed_at_ns'] == sample['observed_at_ns']]
        if len(decay) >= 8:
            self.motion['bar_decay_per_second'] = sum(decay)/len(decay)
            self.motion['bar_samples'] = self.motion.get('bar_samples',0) + len(decay)
        self.summary = summary
        self.used.add(evidence['sha256'])
        self.revision += 1
        state = {'schema': 'playmodel.combat-economy.v1', 'revision': self.revision,
                 'run_id': self.run_id, 'values': self.values.state(), 'motion': self.motion,
                 'evasion': self.evasion.state(),
                 'used_outcomes': sorted(self.used), 'summary': summary,
                 'model_hash_before': before, 'model_hash_after': self.model_hash,
                 'dataset': {'outcome': evidence, 'actions_path': str(actions),
                             'actions_sha256': digest(actions), 'choices': credited},
                 'evasion_dataset': {'path': str(dataset_path), 'sha256': digest(dataset_path),
                     'rows': len(evasion_rows), 'split_group': self.run_id,
                     'semantics': 'observational_next_frame_clearance_not_safe_action_policy'},
                 'curriculum': {'motion_pairs': pairs, 'economic_examples': len(examples),
                     'factorized_evasion_rows': len(evasion_rows), 'evasion_prediction_examples': len(evasion_training),
                     'bar_prediction_examples': len(decay),
                     'hp_labeled_examples': 0, 'confirmed_kills': 0, 'confirmed_pickups': 0,
                     'unsupported_semantics_remain_unknown': True},
                 'game_improvement_proven': False}
        target = self.directory / f'revision-{self.revision:09d}-{uuid.uuid4().hex}.json'
        with target.open('x', encoding='utf8') as stream:
            json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
        self._snapshot()
        return {'revision': self.revision, 'model_hash': self.model_hash, 'summary': summary,
                'curriculum': state['curriculum'], 'record': str(target)}
