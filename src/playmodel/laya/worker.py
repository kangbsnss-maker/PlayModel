"""Local Laya decisions with trainable causal visual context; serialized IPC.

This is REINFORCE from verified game events, not RLCD teacher distillation.
The text encoder is frozen. Actor and learner use eval mode so dropout cannot change
the behavior probabilities. No output is a BC label or a CNN PPO transition.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from .records import (SCHEMA, canonical, digest, validate_distribution, validate_options,
                      validate_resume_report, validate_tactical_application, validate_tactical_observation, verified_outcome)
from .encoding import encode_complete
from .preferences import checkpoint_preferences, default_preferences, preference_hash, validate_preferences


def _save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _tensor_hash(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Learner:
    def __init__(self, model_dir, output, device='cuda', seed=0, checkpoint=None):
        import torch
        from laya.agent import Agent
        self.torch = torch
        torch.set_num_threads(1)
        # Identical head math for no-grad actor and grad-enabled learner.
        torch.backends.mha.set_fastpath_enabled(False)
        self.model_dir, self.output = Path(model_dir).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.source = json.loads((self.model_dir / 'source-manifest.json').read_text(encoding='utf8'))
        for relative, expected in self.source['files'].items():
            if digest(self.model_dir / relative) != expected:
                raise ValueError('Laya source hash mismatch: ' + relative)
        self.base_hash = digest(self.model_dir / 'source-manifest.json')
        package = Path(__file__).resolve().parents[1]
        self.graph_sources = {name: digest(package / name) for name in (
            'learning/visual_decision.py', 'laya/forward.py', 'laya/visual.py',
            'laya/worker.py', 'laya/encoding.py', 'games/brotato/capture.py')}
        self.graph_hash = hashlib.sha256(canonical(self.graph_sources).encode()).hexdigest()
        self.agent = Agent(str(self.model_dir), device=device, fast=False, compile=False)
        self.model, self.device = self.agent.model, self.agent.device
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not name.startswith(('encoder.', 'act_head.')))
        self.encoder_hash = _tensor_hash(self.model.encoder.state_dict().items())
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.pending, self.accepted, self.events_used = {}, [], set()
        self.preferences = default_preferences()
        self.preference_hash = preference_hash(self.preferences)
        self.checkpoint = None
        self.visual_history, self.visual_context_key = [], None
        self.preference_resume = 'new_session_default'
        object_migration = False
        if checkpoint:
            bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
            validate_resume_report(bundle)
            if any(key.startswith('visual_context.') for key in bundle['head']):
                if bundle.get('graph_hash') != self.graph_hash:
                    from .graph_migration import validate_object_branch_migration, validate_character_provenance_migration
                    if bundle.get('graph_hash') == 'ce5add78e12e558691a18b2ca7d4700dd46cdf87125612201ab526cafe908e80':
                        migration = validate_character_provenance_migration(bundle, self.graph_sources)
                    else:
                        migration = validate_object_branch_migration(bundle)
                        object_migration = True
                    _save(self.output / 'graph-migration.json', {
                        **migration, 'checkpoint': str(Path(checkpoint).resolve()),
                        'checkpoint_sha256': digest(checkpoint), 'new_graph_hash': self.graph_hash})
                self._attach_visual_context(seed)
            restored_preferences = checkpoint_preferences(bundle)
            if bundle['base_hash'] != self.base_hash or bundle['encoder_hash'] != self.encoder_hash:
                raise ValueError('Laya checkpoint base/encoder mismatch')
            heads = bundle['head']
            if object_migration:
                additions = {k:v for k,v in self._heads().items() if k not in heads}
                if not additions or any(not k.startswith(('visual_context.object_encoder.',
                            'visual_context.object_project.')) for k in additions):
                    raise ValueError('Unexpected object migration tensor additions')
                self._restore({**heads, **additions})
            else:
                self._restore(heads)
            if _tensor_hash((k,v) for k,v in self._heads().items() if k in heads) != bundle['head_hash']:
                raise ValueError('Laya checkpoint head hash mismatch')
            self.checkpoint = str(Path(checkpoint).resolve())
            self.preference_resume = ('checkpoint_preferences_restored' if 'preferences' in bundle
                                      else 'legacy_checkpoint_default_preferences_new_behavior_version')
            self.preferences = restored_preferences
            self.preference_hash = preference_hash(self.preferences)
            if bundle.get('preference_hash', self.preference_hash) != self.preference_hash:
                raise ValueError('Checkpoint preference hash mismatch')
        if not hasattr(self.model, 'visual_context'):
            self._attach_visual_context(seed)
        from .forward import ChoiceForward
        self.choice_forward = ChoiceForward(self.model)
        self.version = self._version()
        self.preference_application = self._record_preferences()
        # Warm the actual forward path before fresh gameplay observations arrive.
        item, _ = self.encode({'scene': 'shop', 'health': 'unknown'}, {'a': 'Buy an item', 'b': 'Save money'})
        with torch.no_grad():
            self.probs(item)

    def _attach_visual_context(self, seed):
        from playmodel.learning.visual_decision import VisualDecisionContext
        # Deterministic migration without changing the actor sampling generator.
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(seed)
            self.model.visual_context = VisualDecisionContext(self.model.encoder.config.hidden_size)
        self.model.visual_context.to(self.device).eval()

    def _heads(self):
        return {name: tensor for name, tensor in self.model.state_dict().items() if not name.startswith('encoder.')}

    def head_hash(self):
        return _tensor_hash(self._heads().items())

    def head_summary(self, before=None):
        """Tensor magnitudes describe the head, never gameplay competence."""
        parameters = {name: value for name, value in self.model.named_parameters()
                      if not name.startswith('encoder.')}
        squared = sum(float(value.detach().double().square().sum()) for value in parameters.values())
        delta = (math.sqrt(sum(float((value.detach().double() - before[name].to(device=value.device,
                                    dtype=self.torch.float64)).square().sum())
                               for name, value in parameters.items())) if before is not None else None)
        return {'parameter_count': sum(value.numel() for value in parameters.values()),
                'trainable_parameter_count': sum(value.numel() for value in parameters.values() if value.requires_grad),
                'head_l2_norm': math.sqrt(squared), 'head_delta_l2_norm': delta,
                'head_tensor_summary_semantics': 'tensor_magnitude_not_gameplay_improvement'}

    def _version(self):
        return hashlib.sha256((self.base_hash + self.head_hash() + ':visual-goal-v1:causal-window4:preferences:'
            + self.preference_hash + self.graph_hash + ':lossless-options:temperature=1').encode()).hexdigest()

    def configure_preferences(self, preferences):
        if self.pending or self.accepted:
            raise ValueError('Preferences may change only at an empty decision boundary')
        value = validate_preferences(preferences)
        fingerprint = preference_hash(value)
        self.preferences, self.preference_hash = value, fingerprint
        self.version = self._version()
        self.preference_application = self._record_preferences()
        return {'status': 'configured', **self.preference_application}

    def _record_preferences(self):
        value = {'revision': self.preferences['revision'], 'preferences_hash': self.preference_hash,
                 'preference_hash': self.preference_hash, 'behavior_version': self.version,
                 'applied_at_ns': time.perf_counter_ns(), 'preferences': self.preferences,
                 'checkpoint_preference_provenance': self.preference_resume,
                 'reward_source': False, 'bc_label': False}
        _save(self.output / f'preferences-applied-{uuid.uuid4().hex}.json', value)
        return value

    def _restore(self, heads):
        current = self._heads()
        if set(heads) != set(current):
            raise ValueError('Laya head keys mismatch')
        with self.torch.no_grad():
            for key, value in heads.items():
                if value.shape != current[key].shape or not self.torch.isfinite(value).all():
                    raise ValueError('Invalid Laya head tensor')
                current[key].copy_(value.to(current[key].device))

    def encode(self, state, options):
        from laya.common import serialize_state, render_options
        validate_options(options)
        # Use short stable option aliases; IDs remain application metadata.
        q = {'t': 'choice', 'ins': 'Choose the next action to survive and progress in Brotato.',
             'crit': {str(i): text for i, text in enumerate(options.values())}}
        tok = self.agent.tok
        state_ids = tok(serialize_state(state).replace(tok.mask_token, ' '), add_special_tokens=False)['input_ids']
        max_len = int(self.agent.cfg.get('max_len', 512))
        ids, markers = encode_complete(tok, q['ins'], render_options(q), state_ids, max_len=max_len)
        return {'ids': ids, 'markers': markers, 'qtype': 0}, q

    def logits(self, item):
        from laya.common import collate_items
        torch = self.torch
        batch = collate_items([[item]], self.agent.tok.pad_token_id)
        tensors = {key: batch[key].to(self.device) for key in
                   ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            from .visual import tensor_window, object_tensor
            visual = tensor_window(item['visual_window'], self.device) if item.get('visual_window') else None
            objects = object_tensor(item['visual_window'], self.device) if item.get('visual_window') else None
            logits = self.choice_forward(tensors,
                key=(tuple(item['ids']), tuple(item['markers']), item['qtype']), visual_frames=visual,
                object_frames=objects)
        # Neutral temperature defines the explicit training policy. Do not claim
        # the source model's task-specific confidence calibration transfers.
        return logits[0].float()

    def probs(self, item, bias=None, *, return_raw=False):
        logits = self.logits(item)
        raw = self.torch.softmax(logits, dim=-1)
        if bias is not None:
            logits = logits + self.torch.tensor(bias, dtype=logits.dtype, device=logits.device)
        actual = self.torch.softmax(logits, dim=-1)
        return (actual, raw) if return_raw else actual

    def choose(self, state, options, evidence):
        if len(self.pending) > 128:
            raise ValueError('Too many unaccepted Laya decisions')
        validate_options(options)
        domain = evidence.get('decision_domain', 'menu')
        if domain not in ('menu', 'combat_tactic'):
            raise ValueError('Unknown Laya decision domain')
        if domain == 'combat_tactic' and any(not evidence.get(k) for k in ('epoch', 'signature', 'run_id')):
            raise ValueError('Tactical decision identity required')
        if domain == 'combat_tactic':
            if not evidence.get('observation_path') or digest(evidence['observation_path']) != evidence.get('observation_sha256'):
                raise ValueError('Tactical observation artifact hash mismatch')
        frame = evidence.get('frame_ref')
        if not frame or digest(frame) != evidence.get('frame_sha256'):
            raise ValueError('Laya observation frame hash mismatch')
        observed, available = evidence.get('observed_at_ns'), evidence.get('available_at_ns')
        now = time.perf_counter_ns()
        if type(observed) is not int or type(available) is not int or not 0 < observed <= available <= now:
            raise ValueError('Causal Laya observation times required')
        if domain == 'combat_tactic':
            validate_tactical_observation(state, options, evidence, now)
            if now >= evidence['expires_at_ns']:
                raise ValueError('Tactical request expired before inference')
        item, question = self.encode(state, options)
        from .visual import snapshot, validate_window
        context_key = (evidence.get('run_id'), evidence.get('epoch'), domain)
        if context_key != self.visual_context_key or domain == 'menu':
            self.visual_history = []
        self.visual_context_key = context_key
        self.visual_history = [row for row in self.visual_history
            if 0 < observed - row['observed_at_ns'] <= 2_500_000_000
            and row['available_at_ns'] <= available][-3:]
        current_visual = snapshot(evidence, self.output)
        item['visual_window'] = [*self.visual_history, current_visual]
        validate_window(item['visual_window'], evidence)
        self.visual_history = item['visual_window']
        bias = [self.preferences['values'].get(key.split('@', 1)[0], 0.0)
                if domain == 'combat_tactic' else 0.0 for key in options]
        with self.torch.no_grad():
            probabilities, raw_probabilities = self.probs(item, bias, return_raw=True)
            probabilities, raw_probabilities = probabilities.cpu(), raw_probabilities.cpu()
            index = int(self.torch.multinomial(probabilities, 1, generator=self.generator).item())
        values = probabilities.tolist()
        validate_distribution(options, values)
        decided = time.perf_counter_ns()
        decision_id = uuid.uuid4().hex
        record = {'schema': SCHEMA, 'decision_id': decision_id, 'decision_domain': domain, 'state': state, 'options': options,
                  'question': question, 'tokens': item, 'evidence': evidence,
                  'model_schema': 'playmodel.visual-goal.v1',
                  'graph_hash': self.graph_hash,
                  'action_id': list(options)[index], 'action_index': index,
                  'distribution': dict(zip(options, values)), 'log_probability': math.log(values[index]),
                  'raw_head_distribution': dict(zip(options, raw_probabilities.tolist())),
                  'applied_bias': dict(zip(options, bias)), 'preferences': self.preferences,
                  'options_order': list(options), 'applied_bias_vector': bias,
                  'preference_revision': self.preferences['revision'],
                  'preference_hash': self.preference_hash,
                  'choice_basis': 'causal_pixels_and_observed_state_with_learned_visual_and_decision_weights_plus_human_bias',
                  'behavior_version': self.version, 'temperature': 1.0,
                  'sampling': 'categorical', 'decided_at_ns': decided,
                  'inference_ms': (decided - now) / 1e6, 'source': self.source,
                  'probability_semantics': 'uncalibrated action preference, not success probability',
                  'training_target_source': 'future_verified_game_outcome', 'bc_label': False}
        _save(self.output / f'choice-{decision_id}.json', record)
        record['_choice_sha256'] = digest(self.output / f'choice-{decision_id}.json')
        self.pending[decision_id] = record
        return {key: record[key] for key in ('decision_id', 'action_id', 'distribution', 'log_probability',
                                            'behavior_version', 'decided_at_ns', 'inference_ms')}

    def accept(self, decision_id, application):
        record = self.pending.get(decision_id)
        if record is None or record['behavior_version'] != self.version:
            raise ValueError('Unknown or stale Laya decision')
        if record.get('decision_domain', 'menu') != 'menu':
            raise ValueError('Combat decisions require tactical acceptance')
        sent, verified = application.get('sent_at_ns'), application.get('verified_at_ns')
        if (application.get('accepted') is not True or application.get('successful_transport_reported') is not True
                or application.get('game_application_verified') is not True
                or type(sent) is not int or type(verified) is not int
                or not record['decided_at_ns'] <= sent < verified <= time.perf_counter_ns()):
            raise ValueError('Actual send and verified application required')
        candidates = record['evidence'].get('candidates')
        if candidates:
            chosen = next((c for c in candidates if c['candidate_id'] == record['action_id']), None)
            if chosen is None or chosen.get('legal') is not True or chosen['target'] != application.get('actual_target'):
                raise ValueError('Actual target differs from chosen legal candidate')
            if (application.get('decision_id') != decision_id
                    or application.get('before_frame_ref') != record['evidence']['frame_ref']
                    or list(application.get('after_frame_ids', [])) !=
                       [p['frame_ref'] for p in application.get('after_frames', [])]):
                raise ValueError('Application refers to another Laya decision or frame')
            auth = application.get('authorization') or {}
            if auth.get('sent_at_ns') != sent or auth.get('target') != chosen['target']:
                raise ValueError('Laya transmission receipt mismatch')
        frames = application.get('after_frames', [])
        if len(frames) < 1:
            raise ValueError('Post-action source frame required')
        for frame in frames:
            if digest(frame['frame_ref']) != frame['frame_sha256']:
                raise ValueError('Application frame hash mismatch')
        _save(self.output / f'accepted-{decision_id}.json', application)
        record['_accepted_sha256'] = digest(self.output / f'accepted-{decision_id}.json')
        record['application'] = application
        self.accepted.append(record)
        del self.pending[decision_id]
        return {'status': 'accepted', 'decision_id': decision_id}

    def accept_tactic(self, decision_id, application):
        record = self.pending.get(decision_id)
        if record is None or record['behavior_version'] != self.version:
            raise ValueError('Unknown or stale Laya decision')
        now = time.perf_counter_ns()
        receipt = validate_tactical_application(record, application, now)
        stored = {**receipt, 'receipt_path': str(Path(application['receipt_path']).resolve()),
                  'receipt_sha256': application['receipt_sha256'], 'receipt_validated_at_ns': now}
        path = self.output / f'accepted-{decision_id}.json'
        _save(path, stored)
        record['_accepted_sha256'] = digest(path)
        record['application'] = stored
        self.accepted.append(record)
        del self.pending[decision_id]
        return {'status': 'accepted', 'decision_id': decision_id, 'decision_domain': 'combat_tactic'}

    def discard(self, decision_id, reason):
        if decision_id not in self.pending:
            raise ValueError('Only a pending decision can be discarded')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Discard reason required')
        _save(self.output / f'discarded-{decision_id}.json', {
            'decision_id': decision_id, 'reason': reason, 'behavior_version': self.version,
            'discarded_at_ns': time.perf_counter_ns()})
        del self.pending[decision_id]
        return {'status': 'discarded', 'decision_id': decision_id}

    def abandon(self, reason):
        self.visual_history, self.visual_context_key = [], None
        _save(self.output / f'abandoned-{uuid.uuid4().hex}.json',
              {'reason': reason, 'accepted': [r['decision_id'] for r in self.accepted],
               'pending': list(self.pending), 'behavior_version': self.version})
        self.pending.clear()
        self.accepted.clear()
        return {'status': 'abandoned', 'reason': reason}

    def finish(self, kind, evidence):
        reward, observed, outcome = verified_outcome(kind, evidence)
        if evidence['sha256'] in self.events_used:
            raise ValueError('Outcome already consumed')
        self.events_used.add(evidence['sha256'])
        if not self.accepted:
            return {'status': 'no_update', 'reason': 'no_accepted_laya_choices'}
        torch = self.torch
        records = list(self.accepted)
        proofs = []
        if any(r['evidence'].get('candidates') or r.get('decision_domain') == 'combat_tactic' for r in records):
            terminal_path = Path(evidence['path']).resolve()
            report_path = Path(evidence.get('report_path', '')).resolve()
            if report_path.parent != terminal_path.parent or evidence.get('report_sha256') != digest(report_path):
                raise ValueError('Outcome combat report hash/location mismatch')
            report = json.loads(report_path.read_text(encoding='utf8'))
            if (report.get('terminal_kind') != kind or report.get('reason') != 'terminal_' + kind
                    or report.get('error') or Path(report.get('session_directory', '')).resolve() != terminal_path.parent
                    or any(r['evidence']['run_id'] != evidence.get('run_id') for r in records)):
                raise ValueError('Outcome is not from this decision run and verified combat')
            proofs.append({'path': str(report_path), 'sha256': evidence['report_sha256']})
            tactics = [r for r in records if r.get('decision_domain') == 'combat_tactic']
            if tactics:
                if (report.get('action_domain') != 'combat_tactic'
                        or any(report.get(k) is not True for k in ('verified_terminal_boundary',
                            'tactical_collection_eligible', 'recorder_complete', 'worker_stopped'))
                        or report.get('run_id') != evidence.get('run_id')):
                    raise ValueError('Tactical combat report is not eligible')
                actions = Path(report.get('actions_path', '')).resolve()
                if actions.parent != terminal_path.parent or digest(actions) != report.get('actions_sha256'):
                    raise ValueError('Tactical action ledger hash/location mismatch')
                proofs.append({'path': str(actions), 'sha256': report['actions_sha256']})
                ledger = [json.loads(line) for line in actions.read_text(encoding='utf8').splitlines() if line.strip()]
                for record in tactics:
                    app = record['application']
                    if (Path(app['receipt_path']).resolve().parent != terminal_path.parent
                            or record['evidence']['epoch'] != report.get('tactical_epoch')
                            or not any(Path(p['path']).resolve() == Path(app['receipt_path']).resolve()
                                and p['sha256'] == app['receipt_sha256'] for p in report.get('tactical_receipts', []))):
                        raise ValueError('Tactical receipt is not from this combat session')
                    linked = [row for row in ledger if row.get('execution_id') == app.get('execution_id')]
                    keys = ('execution_id', 'decision_id', 'action_id', 'behavior_version', 'epoch',
                            'signature', 'sent_at_ns', 'transport_started_at_ns', 'frame_sha256')
                    if (not app.get('execution_id') or len(linked) != 1
                            or any(linked[0].get(k) != app.get(k) for k in keys)):
                        raise ValueError('Tactical receipt does not match actual action ledger')
        for record in records:
            tactical = record.get('decision_domain') == 'combat_tactic'
            application = record['application']
            causal_time = application['sent_at_ns'] if tactical else application['verified_at_ns']
            if record['behavior_version'] != self.version or (not tactical and observed <= causal_time):
                raise ValueError('Outcome must follow same-version accepted choices')
            if record.get('preference_hash') != self.preference_hash:
                raise ValueError('Learning preference version differs from behavior')
            if tactical:
                observation = record['evidence']
                if (not observation.get('observation_path') or
                        digest(observation['observation_path']) != observation.get('observation_sha256')):
                    raise ValueError('Tactical observation artifact changed before learning')
                proofs.append({'path': observation['observation_path'], 'sha256': observation['observation_sha256']})
                receipt = validate_tactical_application(record, application, time.perf_counter_ns())
                if any(application.get(k) != v for k, v in receipt.items()):
                    raise ValueError('Accepted tactical receipt contents changed')
                proofs.append({'path': application['receipt_path'], 'sha256': application['receipt_sha256']})
                if receipt.get('ownership_receipt_path'):
                    proofs.append({'path': receipt['ownership_receipt_path'],
                                   'sha256': receipt['ownership_receipt_sha256']})
            observation = record['evidence']
            character = observation.get('character_context')
            if character:
                from playmodel.games.brotato.character_context import validate_choice_character
                proofs.append(validate_choice_character(character, observation['observed_at_ns']))
            economic = observation.get('economic_model')
            if economic:
                if digest(economic['path']) != economic['sha256']:
                    raise ValueError('Economic model snapshot changed')
                proofs.append({'path': economic['path'], 'sha256': economic['sha256']})
            from .visual import validate_window
            validate_window(record['tokens']['visual_window'], observation)
            for visual in record['tokens']['visual_window']:
                proofs.extend({'path': crop['path'], 'sha256': crop['sha256']}
                              for crop in visual.get('object_crops', []))
                proofs.extend([{'path': visual['path'], 'sha256': visual['sha256']},
                               {'path': visual['source_path'], 'sha256': visual['source_sha256']}])
            frames = [application] if tactical else application['after_frames']
            for proof in [observation, *frames]:
                if digest(proof['frame_ref']) != proof['frame_sha256']:
                    raise ValueError('Laya source frame changed before learning')
            for prefix in ('choice', 'accepted'):
                path = self.output / f'{prefix}-{record["decision_id"]}.json'
                expected = record['_' + prefix + '_sha256']
                if digest(path) != expected:
                    raise ValueError('Laya decision evidence changed before learning')
                proofs.append({'path': str(path), 'sha256': expected})
        # An asynchronous terminal detector can finish after the writer sent
        # another action. Keep its transport evidence, but never reward that
        # post-outcome action or carry it into the next wave.
        excluded = [r for r in records if r.get('decision_domain') == 'combat_tactic'
                    and r['application']['sent_at_ns'] >= observed]
        if excluded:
            _save(self.output / f'boundary-excluded-{uuid.uuid4().hex}.json', {
                'reason': 'action_not_before_terminal_observation', 'outcome': evidence,
                'behavior_version': self.version, 'reward_assigned': False,
                'decisions': [r['decision_id'] for r in excluded], 'files': proofs})
            excluded_ids = {r['decision_id'] for r in excluded}
            records = [r for r in records if r['decision_id'] not in excluded_ids]
        if not records:
            self.accepted.clear()
            self.pending.clear()
            return {'status': 'no_update', 'reason': 'no_choices_before_terminal_observation'}
        if all(len(r['options']) == 1 for r in records):
            self.accepted.clear()
            return {'status': 'no_update', 'reason': 'forced_actions_have_no_policy_gradient'}
        update_id = uuid.uuid4().hex
        directory = self.output / ('update-' + update_id)
        directory.mkdir()
        _save(directory / 'dataset-manifest.json', {
            'schema': 'playmodel.visual-goal-dataset.v1', 'split': 'train', 'source_version': self.version,
            'files': proofs, 'outcome': evidence,
            'run_ids': sorted(set(r['evidence']['run_id'] for r in records)),
            'decision_domains': sorted(set(r.get('decision_domain', 'menu') for r in records))})
        before = {k: v.detach().cpu().clone() for k, v in self._heads().items()}
        old_version, old_head_hash = self.version, self.head_hash()
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        losses, logp_errors = [], []
        for record in records:
            p = self.probs(record['tokens'], list(record['applied_bias'].values()))
            logp = p[record['action_index']].clamp_min(1e-12).log()
            mismatch = abs(float(logp.detach()) - record['log_probability'])
            logp_errors.append(mismatch)
            if mismatch > 1e-4:
                raise ValueError('Laya behavior log probability does not reproduce')
            elapsed = (observed - record['application']['sent_at_ns']) / 1e9
            actual_return = reward * (0.997 ** elapsed)
            loss = -logp * actual_return / len(records)
            losses.append(float(loss.detach()))
            loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 0.5))
        if not math.isfinite(gradient_norm) or any(p.grad is not None for p in self.model.encoder.parameters()):
            raise ValueError('Invalid/frozen-encoder gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        kls = []
        with torch.no_grad():
            for record in records:
                p = self.probs(record['tokens'], list(record['applied_bias'].values())).cpu()
                old = torch.tensor(list(record['distribution'].values()))
                kls.append(float((old * (old.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum()))
        changed_hash = self.head_hash()
        encoder_after = _tensor_hash(self.model.encoder.state_dict().items())
        accepted = (changed_hash != old_head_hash and encoder_after == self.encoder_hash
                    and all(math.isfinite(k) and k <= 0.03 for k in kls)
                    and all(torch.isfinite(v).all() for v in self._heads().values()))
        visual_deltas = {}
        for component in ('encoder', 'temporal', 'project', 'object_encoder', 'object_project'):
            prefix = 'visual_context.' + component + '.'
            visual_deltas[component] = math.sqrt(sum(float(
                (value.detach().double() - before[name].to(value.device).double()).square().sum())
                for name, value in self.model.named_parameters() if name.startswith(prefix)))
        report = {'schema': 'playmodel.laya-outcome-update.v1', 'source_version': old_version,
                  **self.head_summary(before),
                  'model_schema': 'playmodel.visual-goal.v1',
                  'graph_hash': self.graph_hash, 'graph_sources': self.graph_sources,
                  'method': 'visual_context_and_decision_REINFORCE', 'encoder_frozen': True,
                  'encoder_frozen_scope': 'text_encoder_only', 'visual_encoder_trainable': True,
                  'visual_component_delta_l2': visual_deltas,
                  'text_feature_cache': {'hits': self.choice_forward.hits,
                      'misses': self.choice_forward.misses, 'bytes': self.choice_forward.bytes},
                  'preference_hash': self.preference_hash, 'preferences': self.preferences,
                  'encoder_hash_before': self.encoder_hash, 'encoder_hash_after': encoder_after,
                  'head_hash_before': old_head_hash, 'head_hash_after': changed_hash,
                  'accepted': accepted, 'optimizer_steps': 1, 'loss': sum(losses),
                  'gradient_norm': gradient_norm, 'kl_per_choice': kls,
                  'max_log_probability_error': max(logp_errors), 'reward': reward,
                  'outcome': evidence, 'verified_outcome': outcome,
                  'decisions': [r['decision_id'] for r in records],
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'game_improvement_proven': False, 'calibrated': False}
        report['dataset_manifest_sha256'] = digest(directory / 'dataset-manifest.json')
        bundle = {'base_hash': self.base_hash, 'encoder_hash': self.encoder_hash,
                  'model_schema': 'playmodel.visual-goal.v1',
                  'graph_hash': self.graph_hash,
                  'preferences': self.preferences, 'preference_hash': self.preference_hash,
                  'head_hash': changed_hash, 'head': {k: v.detach().cpu().clone() for k, v in self._heads().items()},
                  'report': report}
        checkpoint = directory / 'head.pt'
        torch.save(bundle, checkpoint)
        reloaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if _tensor_hash(reloaded['head'].items()) != changed_hash:
            accepted = report['accepted'] = False
        if accepted:
            self._restore(reloaded['head'])
            self.version = self._version()
            self.checkpoint = str(checkpoint)
        else:
            self._restore(before)
        report.update(status='updated' if accepted else 'rejected',
                      behavior_version=self.version, checkpoint=self.checkpoint,
                      candidate_checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        _save(directory / 'report.json', report)
        self.accepted.clear()
        self.pending.clear()
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    protocol = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            learner = Learner(args.model, args.output, args.device, args.seed, args.checkpoint)
        protocol.write(canonical({'status': 'ready', 'behavior_version': learner.version,
                                  'model_schema': 'playmodel.visual-goal.v1',
                                  'checkpoint': learner.checkpoint, 'encoder_hash': learner.encoder_hash,
                                  **learner.head_summary(),
                                  **learner.preference_application}) + '\n')
        protocol.flush()
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request.pop('id')
            method = request.pop('method')
            try:
                if method not in ('choose', 'accept', 'accept_tactic', 'discard', 'configure_preferences', 'finish', 'abandon'):
                    raise ValueError('Unknown worker method')
                with redirect_stdout(sys.stderr):
                    result = getattr(learner, method)(**request)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                result = {'error': type(error).__name__ + ': ' + str(error)}
            protocol.write(canonical({'request_id': request_id, **result}) + '\n')
            protocol.flush()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        protocol.write(canonical({'error': type(error).__name__ + ': ' + str(error)}) + '\n')
        protocol.flush()
        raise


if __name__ == '__main__':
    main()
