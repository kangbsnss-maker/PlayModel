"""Laya decisions with the existing fresh-input and game-application verifier.

No CNN action tensors, values or PPO labels are fabricated for these choices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .neural_choices import ChoiceObservation
from .neural_menu_controller import NeuralMenuController

CNN_EXCLUSION = 'mixed_control_excluded_from_cnn_ppo'


@dataclass(frozen=True)
class LayaMenuDecision:
    decision_id: str
    observation: ChoiceObservation
    action_index: int
    target: str
    behavior_version: str
    decided_at_ns: int
    backend_record: dict


class LayaMenuController(NeuralMenuController):
    def __init__(self, recorder, *, client, **kwargs):
        super().__init__(recorder, **kwargs)
        self.client = client
        self.economy = None

    def _sample(self, current, previous_action, prepared_at):
        options = {}
        for candidate in current.candidates:
            if not candidate.legal:
                continue
            text = candidate.kind + ': ' + '; '.join(candidate.raw_text)
            if current.scene == 'shop' and self._ready_shop is not None:
                shop = self._ready_shop.current
                if candidate.candidate_id.startswith('buy:'):
                    slot = int(candidate.candidate_id.split(':')[1])
                    text += '; cost=' + str(next(offer.cost for offer in shop.offers if offer.slot == slot))
                elif candidate.candidate_id == 'reroll':
                    text += '; cost=' + str(shop.reroll_cost)
            options[candidate.candidate_id] = text
        frame = Path(current.frame_id)
        build = self.recorder.build_state.snapshot() if self.recorder.build_state else {}
        character = getattr(self.recorder, 'character_context', {})
        from .character_context import affinity
        state = {'game': 'Brotato', 'scene': current.scene, 'wave': current.wave,
                 'training_intent': getattr(self, 'training_intent', 'survive and progress'),
                 'currency': current.currency, 'weapon_fill': current.weapon_fill,
                 'previous_actual_movement': previous_action,
                 'last_observed_stats': {key: value for key, value in build.get('stats', {}).items()
                                         if value is not None},
                 'weapon_names': [weapon['name'] for weapon in build.get('weapons', [])],
                 'weapon_count': build.get('weapon_count'),
                 'unknown': 'missing stats; current HP; item effects; unlisted inventory'}
        if character:
            state['character'] = character['name']
            state['trait_fit'] = {key:affinity(character['traits'],text) for key,text in options.items()}
            # Full raw traits and source remain in evidence, beyond token budget.
            state.pop('unknown')
            state.pop('previous_actual_movement')
        if self.economy is not None:
            state['combat_memory'] = {key: self.economy.summary[key] for key in (
                'tracked', 'hp_drop_proxy', 'target_hp_drop_proxy', 'visible_duration_proxy',
                'pickup_missing_proxy', 'clear') if key in self.economy.summary}
            state['value_predictions'] = self.economy.estimates(state, options)
            state['value_revision'] = self.economy.revision
        evidence = {'frame_ref': str(frame), 'frame_sha256': hashlib.sha256(frame.read_bytes()).hexdigest(),
                    'observed_at_ns': current.observed_at_ns, 'available_at_ns': current.available_at_ns,
                    'game_build_id': current.game_build_id, 'ocr_sha256': current.ocr_sha256,
                    'run_id': self.recorder.run_id, 'build_snapshot': build,
                    'candidates': [asdict(candidate) for candidate in current.candidates]}
        if character:
            evidence['character_context'] = character
        if self.economy is not None:
            evidence['economic_model'] = self.economy.model_evidence()
        result = self.client.choose(state, options, evidence)
        action_id = result.get('action_id')
        if action_id not in options:
            raise ValueError('Laya returned an unavailable action')
        decided_at = result.get('decided_at_ns')
        if type(decided_at) is not int or not prepared_at <= decided_at <= self.clock():
            raise ValueError('Laya decision clock invalid')
        index = next(i for i, c in enumerate(current.candidates) if c.candidate_id == action_id)
        # Invalidate once a foreign policy intervenes; invalidate does not close.
        if CNN_EXCLUSION not in self.recorder.rejection_reasons:
            self.recorder.invalidate(CNN_EXCLUSION)
        return LayaMenuDecision(result['decision_id'], current, index,
                                current.candidates[index].target, result['behavior_version'],
                                decided_at, result)

    def _save_decision(self, stage, decision, application=None):
        if self.output_directory is None:
            return
        directory = self.output_directory / decision.decision_id / stage
        directory.mkdir(parents=True, exist_ok=False)
        value = {'schema': 'playmodel.laya-menu.v1', 'backend': decision.backend_record,
                 'frame_ref': decision.observation.frame_id, 'target': decision.target,
                 'action_index': decision.action_index, 'cnn_ppo_eligible': False,
                 'application': asdict(application) if application is not None else None}
        (directory / 'decision.json').write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def _record_application(self, decision, application):
        pending = self._pending
        proof = {**asdict(application), 'game_application_verified': True,
                 'successful_transport_reported': True, 'authorization': self._pending_details(),
                 'before_frame_ref': decision.observation.frame_id,
                 'before_frame_sha256': hashlib.sha256(Path(decision.observation.frame_id).read_bytes()).hexdigest(),
                 'after_frames': [{'frame_ref': frame,
                                  'frame_sha256': hashlib.sha256(Path(frame).read_bytes()).hexdigest()}
                                 for frame in application.after_frame_ids]}
        if pending is None or pending.sent_at_ns != application.sent_at_ns or not application.accepted:
            raise ValueError('Laya application lacks actual transport')
        self.client.accept(decision.decision_id, proof)
        if self.economy is not None and decision.observation.scene == 'shop':
            path = self.client.output / f'choice-{decision.decision_id}.json'
            record = json.loads(path.read_text(encoding='utf8'))
            record['_source_path'] = str(path.resolve())
            self.economy.record_purchase(record, proof)

    def discard_interrupted_outcome(self):
        """A released combat gap cannot reward earlier menu choices."""
        self.client.abandon('released_combat_gap_outcome_excluded')
        if self.economy is not None:
            self.economy.abandon()
        self._pending = None
        self._previous_shop = self._ready_shop = None
        self._previous_loot = self._previous_stats = None
        self._prepared_observation = self._prepared_fingerprint = self._prepared_shop = None
        self._preparation_observations = 0

    def _fail(self, reason, *, details=None):
        if self.economy is not None:
            self.economy.abandon()
        self.client.abandon('menu_verification_failed:' + str(reason))
        return super()._fail(reason, details=details)
