"""Pure shared-CNN/GRU menu decisions and application evidence.

No capture, key input, optimizer, reward assignment or model promotion occurs
here. Candidate text is a lossy lexical observation, not parsed game-effect
truth. Unknown effects remain explicit. The actor samples legal candidates;
there is no hand-written preference ranking or action copied from another AI.

The caller owns one unchanged behavior model, exclusive input, original-frame
preservation, focus verification before Enter, and run-level outcome rewards.
Navigation arrows are execution details of the sampled macro action, not extra
unrecorded policy choices. Failed/unknown application must not enter PPO.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import torch
from torch.nn import functional as F

from playmodel.learning import StateEvidence
from playmodel.learning.recurrent_ppo import RecurrentActorCritic
from playmodel.learning.runtime_contract import PHASE_SCHEMA, RUNTIME_CONTRACT
from .menu import BUTTONS, classify_scene
from .ocr import rows_in_region
from .shop_learning import VerifiedShop, purchase_candidates, verify_purchase, verify_reroll


SCHEMA = "playmodel.neural-menu-macro.v2"
FEATURE_SCHEMA = "lexical8_kind4_cost2_text_present_effect_unknown-v1"
CONTEXT_SCHEMA = "previous_actual_movement9_wave2_currency2_weaponfill2_unknown-v1"
MAX_AGE_NS = 750_000_000
# V2 shared contract: movement0, upgrade/loot1, weapon2, shop3, character4.
PHASE_UPGRADE, PHASE_SHOP = 1, 3


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _lexical(text: tuple[str, ...]) -> tuple[float, ...]:
    """Fixed signed character-trigram sketch. Collisions/unknown semantics remain."""
    normalized = _compact("|".join(text))[:4096]
    result = [0.0] * 8
    if not normalized:
        return tuple(result)
    grams = [normalized[index:index + 3] for index in range(max(1, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        result[digest[0] % 8] += 1.0 if digest[1] & 1 else -1.0
    scale = max(1.0, math.sqrt(len(grams)))
    return tuple(max(-1.0, min(1.0, value / scale)) for value in result)


@dataclass(frozen=True)
class MenuCandidate:
    candidate_id: str
    target: str
    kind: str
    semantic_id: str
    raw_text: tuple[str, ...]
    features: tuple[float, ...]
    legal: bool
    mask_reason: str


@dataclass(frozen=True)
class ChoiceObservation:
    frame_id: str
    source_pixels_sha256: str
    ocr_sha256: str
    observed_at_ns: int
    available_at_ns: int
    ocr_started_at_ns: int
    scene: str
    phase: int | None
    rgb96: bytes
    candidates: tuple[MenuCandidate, ...]
    card_set_sha256: str
    wave: int | None = None
    currency: int | None = None
    weapon_fill: float | None = None
    game_build_id: str = "23429717"
    # Literal title from the calibrated loot card. Empty is unknown. Tooltip
    # prose and action highlighting never establish or change card identity.
    loot_card_title: tuple[str, ...] = ()

    @property
    def legal_mask(self) -> tuple[bool, ...]:
        return tuple(candidate.legal for candidate in self.candidates) + (False,) * (9 - len(self.candidates))


@dataclass(frozen=True)
class FrozenMacroDecision:
    decision_id: str
    observation: ChoiceObservation
    context: tuple[float, ...]
    action_index: int
    target: str
    old_log_probability: float
    old_value: float
    probabilities: tuple[float, ...]
    hidden_before: tuple[float, ...]
    next_hidden: tuple[float, ...]
    reset: bool
    behavior_version: str
    decided_at_ns: int
    build_state_json: str | None = None

    def tensors(self, *, device="cpu") -> dict[str, torch.Tensor]:
        """Detached fresh tensors; caller mutation cannot alter the frozen record."""
        image = torch.frombuffer(bytearray(self.observation.rgb96), dtype=torch.uint8).reshape(
            96, 96, 3).permute(2, 0, 1).unsqueeze(0).contiguous()
        return {
            "images": image.to(device),
            "context": torch.tensor([self.context], dtype=torch.float32, device=device),
            "phase": torch.tensor([self.observation.phase], dtype=torch.long, device=device),
            "candidates": torch.tensor([[candidate.features for candidate in self.observation.candidates]],
                                       dtype=torch.float32, device=device),
            "legal_mask": torch.tensor([self.observation.legal_mask], dtype=torch.bool, device=device),
            "hidden_before": torch.tensor([self.hidden_before], dtype=torch.float32, device=device),
            "next_hidden": torch.tensor([self.next_hidden], dtype=torch.float32, device=device),
            "actions": torch.tensor([self.action_index], dtype=torch.long, device=device),
            "old_log_probs": torch.tensor([self.old_log_probability], dtype=torch.float32, device=device),
            "old_values": torch.tensor([self.old_value], dtype=torch.float32, device=device),
            "reset": torch.tensor([self.reset], dtype=torch.bool, device=device),
        }


@dataclass(frozen=True)
class MacroApplication:
    decision_id: str
    accepted: bool
    reason: str
    actual_target: str
    sent_at_ns: int
    verified_at_ns: int
    after_frame_ids: tuple[str, ...]
    after_pixel_sha256: tuple[str, ...]
    next_observed_at_ns: int | None
    feature_effects_verified: bool = False
    # No reward: successful application says nothing about run quality.


def _features(raw_text: tuple[str, ...], kind: str, *, cost: int | None = None,
              currency: int | None = None) -> tuple[float, ...]:
    # Reroll keeps the existing 16-feature contract: its lexical observation and
    # cost carry the action evidence, all four existing kind bits remain zero,
    # and its game effects remain explicitly unknown.
    cost_known = cost is not None and currency is not None
    return (*_lexical(raw_text), float(kind == "upgrade"), float(kind == "weapon"),
            float(kind == "item"), float(kind == "skip"),
            min(cost / max(currency, 1), 1.0) if cost_known else 0.0,
            float(cost_known), float(any(raw_text)), float(kind != "skip"))


def _rgb96(pixels: bytes, width: int, height: int) -> bytes:
    raw = torch.frombuffer(bytearray(pixels), dtype=torch.uint8).reshape(height, width, 4)
    rgb = raw[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    small = F.interpolate(rgb.float(), size=(96, 96), mode="area").round().to(torch.uint8)
    # Storage iteration crosses Python once per byte. NumPy copies the exact
    # contiguous uint8 HWC buffer without changing the calibrated resize.
    return small[0].permute(1, 2, 0).contiguous().numpy().tobytes()


def observe_menu(ocr: dict, *, pixels: bytes, frame_id: str, observed_at_ns: int,
                 available_at_ns: int, game_build_id: str = "23429717", width: int = 1920,
                 height: int = 1080, shop: VerifiedShop | None = None) -> ChoiceObservation:
    """Read current menu evidence. A shop without verified prices is post-state only."""
    if (game_build_id != "23429717" or (width, height) != (1920, 1080)
            or not isinstance(pixels, bytes) or len(pixels) != width * height * 4
            or not isinstance(frame_id, str) or not frame_id.strip()
            or type(observed_at_ns) is not int or type(available_at_ns) is not int
            or observed_at_ns <= 0 or available_at_ns < observed_at_ns):
        raise ValueError("Uncalibrated or malformed menu frame")
    started = ocr.get("processing_started_at_ns")
    if (type(started) is not int or not observed_at_ns <= started <= available_at_ns
            or ocr.get("recognition_path") == "exact_image_cache" or "cache_source" in ocr):
        raise ValueError("Fresh, independently executed OCR required")
    scene = classify_scene(ocr, width, height).scene
    if scene == "unknown":
        raise ValueError("Unknown menu scene")
    candidates, phase, wave, currency, weapon_fill = [], None, None, None, None
    loot_card_title = ()
    pixels_sha = _digest(pixels)
    if scene == "level_up":
        phase = PHASE_UPGRADE
        for index, left in enumerate((30, 400, 770, 1140)):
            target = f"choose_{index}"
            raw = tuple(rows_in_region(ocr, (left, 450, left + 360, 610)))
            labels = [_compact(row) for row in rows_in_region(ocr, BUTTONS["level_up"][target])]
            legal = any("choose" in label for label in labels)
            candidates.append(MenuCandidate(f"upgrade:{index}", target, "upgrade", _digest(_canonical(raw)),
                                            raw, _features(raw, "upgrade"), legal,
                                            "visible_choose_button" if legal else "choose_button_unreadable"))
    elif scene == "loot":
        phase = PHASE_UPGRADE
        loot_card_title = tuple(rows_in_region(ocr, (550, 305, 935, 360)))
        title_known = bool(loot_card_title) and any(character.isalpha()
                                                     for row in loot_card_title for character in row)
        body = tuple(rows_in_region(ocr, (550, 400, 930, 670)))
        for target in ('take', 'recycle'):
            # The selected button has a controller glyph at x448..493 (OCR
            # reads it as 囗). Read the positioned central label, retaining the
            # untouched full OCR as source, rather than repairing its text.
            button = BUTTONS['loot'][target]
            labels = tuple(rows_in_region(ocr, (550, button[1], 960, button[3])))
            visible = any((_compact(row) == 'take' if target == 'take' else
                           re.fullmatch(r'(?:recycle|recycie)(?:\(\+[0-9]{1,5}\))?', _compact(row)) is not None)
                          for row in labels)
            # Action text and item prose are observations, not parsed effects or
            # a guessed salvage payout. Ban is deliberately outside this adapter.
            raw = (*loot_card_title, *body, *labels)
            kind = 'item' if target == 'take' else 'recycle'
            identity = _digest(_canonical((tuple(_compact(row) for row in loot_card_title), target)))
            legal = title_known and visible
            candidates.append(MenuCandidate('loot:' + target, target, kind, identity, raw,
                                            _features(raw, kind), legal,
                                            'visible_loot_choice' if legal else 'loot_identity_or_button_unknown'))
    elif scene == "shop":
        phase = PHASE_SHOP
        if shop is not None:
            current = shop.current
            if (current.frame_id != frame_id or current.frame_sha256 != pixels_sha
                    or current.observed_at_ns != observed_at_ns or current.available_at_ns != available_at_ns
                    or current.game_build_id != game_build_id):
                raise ValueError("Verified shop does not belong to this frame")
            wave, currency = current.wave, current.currency
            if current.weapon_count is not None and current.weapon_capacity:
                weapon_fill = current.weapon_count / current.weapon_capacity
            for candidate in purchase_candidates(shop, now_ns=available_at_ns):
                candidates.append(MenuCandidate(candidate.action_id, candidate.target, candidate.kind,
                                                candidate.semantic_id, candidate.text,
                                                _features(candidate.text, candidate.kind, cost=candidate.cost,
                                                          currency=currency), candidate.legal, candidate.reason))
    return ChoiceObservation(frame_id, pixels_sha, _digest(_canonical(ocr["lines"])), observed_at_ns,
                              available_at_ns, started, scene, phase, _rgb96(pixels, width, height),
                              tuple(candidates), _digest(_canonical([(c.semantic_id, c.legal) for c in candidates])),
                              wave, currency, weapon_fill, game_build_id, loot_card_title)


def stable_loot(first: ChoiceObservation | None, second: ChoiceObservation) -> bool:
    """Independent agreeing title/action observations before sampling loot."""
    return bool(first is not None and first.scene == second.scene == 'loot'
                and first.loot_card_title and second.loot_card_title
                and first.frame_id != second.frame_id
                and first.observed_at_ns < first.available_at_ns < second.observed_at_ns
                and first.ocr_started_at_ns != second.ocr_started_at_ns
                and first.game_build_id == second.game_build_id
                and first.card_set_sha256 == second.card_set_sha256
                and any(second.legal_mask))


def observe_upgrade(ocr: dict, **kwargs) -> ChoiceObservation:
    observation = observe_menu(ocr, **kwargs)
    if observation.scene != "level_up":
        raise ValueError("Upgrade menu required")
    return observation


def observe_shop_choice(shop: VerifiedShop, ocr: dict, **kwargs) -> ChoiceObservation:
    observation = observe_menu(ocr, shop=shop, **kwargs)
    if observation.scene != "shop" or not observation.candidates:
        raise ValueError("Verified shop required")
    return observation


def sample_choice(model: RecurrentActorCritic, observation: ChoiceObservation, *, hidden=None,
                   previous_action: int = 0, reset: bool = False, generator=None,
                   now_ns: int | None = None, build_state=None) -> FrozenMacroDecision:
    """Sample the same recurrent actor used for movement; never rank by fixed stats."""
    now = time.perf_counter_ns() if now_ns is None else now_ns
    if (model.config.context_dim not in (16, 64) or model.config.candidate_dim != 16
            or model.config.phase_count <= PHASE_SHOP):
        raise ValueError("Menu adapter requires context16/64, candidate16 and phases0..3")
    if (observation.phase not in (PHASE_UPGRADE, PHASE_SHOP) or not observation.candidates
            or not any(observation.legal_mask) or type(previous_action) is not int or not 0 <= previous_action < 9
            or type(reset) is not bool or not observation.available_at_ns <= now
            or now - observation.observed_at_ns > MAX_AGE_NS):
        raise ValueError("Stale, unsupported or entirely illegal menu choice")
    context = [float(index == previous_action) for index in range(9)]
    context += [min((observation.wave or 0) / 100, 1.0), float(observation.wave is not None),
                min((observation.currency or 0) / 5000, 1.0), float(observation.currency is not None),
                observation.weapon_fill or 0.0, float(observation.weapon_fill is not None), 1.0]
    build_state_json = None
    if model.config.context_dim == 64:
        if build_state is None:
            raise ValueError("context64 requires source-backed observed build state")
        context.extend(build_state.features(observation.observed_at_ns,
                                            available_at_ns=observation.available_at_ns))
        if len(context) != 64:
            raise ValueError("Invalid observed build feature width")
        build_state_json = _canonical(build_state.snapshot()).decode('utf-8')
    device = next(model.parameters()).device
    initial = model.initial_hidden(1) if hidden is None else hidden.detach().clone().to(device)
    if initial.shape != (1, model.config.hidden_size) or not torch.isfinite(initial).all():
        raise ValueError("Invalid recurrent choice memory")
    initial = torch.zeros_like(initial) if reset else initial
    image = torch.frombuffer(bytearray(observation.rgb96), dtype=torch.uint8).reshape(96, 96, 3).permute(
        2, 0, 1).unsqueeze(0).contiguous().to(device)
    behavior_version = model.policy_version()
    with torch.no_grad():
        output = model.step(image, torch.tensor([context], dtype=torch.float32, device=device),
                            torch.tensor([observation.phase], dtype=torch.long, device=device),
                            torch.tensor([[candidate.features for candidate in observation.candidates]],
                                         dtype=torch.float32, device=device),
                            torch.tensor([observation.legal_mask], dtype=torch.bool, device=device),
                            hidden=initial, reset=torch.tensor([reset], dtype=torch.bool, device=device))
        action, log_probability = output.sample(generator)
    if model.policy_version() != behavior_version:
        raise ValueError("Behavior weights changed during menu sampling")
    index = int(action.item())
    # Store CPU tuples/bytes; even tensors returned later cannot alter behavior evidence.
    decided = time.perf_counter_ns() if now_ns is None else now_ns
    if decided - observation.observed_at_ns > MAX_AGE_NS:
        raise ValueError("Menu observation expired during inference")
    return FrozenMacroDecision(_digest(_canonical([behavior_version, observation.frame_id, index, decided])),
                               observation, tuple(context), index, observation.candidates[index].target,
                               float(log_probability.item()), float(output.value.item()),
                               tuple(float(p) for p in output.probabilities[0].cpu().tolist()),
                               tuple(float(v) for v in initial[0].cpu().tolist()),
                               tuple(float(v) for v in output.next_hidden[0].cpu().tolist()),
                               reset, behavior_version, decided, build_state_json)


def _post_pair(decision: FrozenMacroDecision, first: ChoiceObservation, second: ChoiceObservation,
                sent_at_ns: int, actual_target: str, now_ns: int) -> str | None:
    if actual_target != decision.target:
        return "actual_target_differs_from_policy"
    if (not decision.decided_at_ns <= sent_at_ns < first.observed_at_ns
            or not first.available_at_ns < second.observed_at_ns <= second.available_at_ns <= now_ns
            or now_ns - second.observed_at_ns > MAX_AGE_NS
            or len({decision.observation.frame_id, first.frame_id, second.frame_id}) != 3
            or first.ocr_started_at_ns == second.ocr_started_at_ns
            or first.game_build_id != decision.observation.game_build_id
            or second.game_build_id != decision.observation.game_build_id):
        return "post_action_frame_sequence_unverified"
    if first.scene != second.scene or first.card_set_sha256 != second.card_set_sha256:
        return "post_action_menu_not_stable"
    return None


def verify_upgrade_choice(decision: FrozenMacroDecision, first: ChoiceObservation, second: ChoiceObservation, *,
                           sent_at_ns: int, actual_target: str, now_ns: int | None = None) -> MacroApplication:
    now = time.perf_counter_ns() if now_ns is None else now_ns
    reason = _post_pair(decision, first, second, sent_at_ns, actual_target, now)
    if decision.observation.phase != PHASE_UPGRADE or decision.observation.scene != 'level_up':
        reason = "not_upgrade_decision"
    if reason is None:
        changed = (first.scene == "level_up" and all(c.raw_text for c in decision.observation.candidates)
                   and all(c.raw_text for c in first.candidates)
                   and first.card_set_sha256 != decision.observation.card_set_sha256)
        changed_pixels = decision.observation.source_pixels_sha256 != first.source_pixels_sha256
        reason = ("upgrade_selection_accepted" if changed_pixels and
                  (first.scene in ("shop", "loot") or changed) else "upgrade_application_unconfirmed")
    return MacroApplication(decision.decision_id, reason == "upgrade_selection_accepted", reason,
                             actual_target, sent_at_ns, now, (first.frame_id, second.frame_id),
                             (first.source_pixels_sha256, second.source_pixels_sha256),
                             first.observed_at_ns if reason == "upgrade_selection_accepted" else None)


def verify_loot_choice(decision: FrozenMacroDecision, first: ChoiceObservation, second: ChoiceObservation, *,
                       sent_at_ns: int, actual_target: str, now_ns: int | None = None) -> MacroApplication:
    """Verify consumption of the selected card, never its stat/currency effects.

    The caller also verifies source-bound focus before Enter. Two fresh stable
    post-send observations must show the next menu or a different literal loot
    title. A hover/focus/pixel change on the same item cannot confirm acceptance.
    Consecutive identical items stay ambiguous rather than inventing an effect.
    """
    now = time.perf_counter_ns() if now_ns is None else now_ns
    before = decision.observation
    reason = _post_pair(decision, first, second, sent_at_ns, actual_target, now)
    if (before.phase != PHASE_UPGRADE or before.scene != 'loot' or not before.loot_card_title
            or decision.target not in ('take', 'recycle')):
        reason = 'not_loot_decision'
    if reason is None:
        next_card = (first.scene == 'loot' and first.loot_card_title and second.loot_card_title
                     and tuple(_compact(row) for row in first.loot_card_title)
                         == tuple(_compact(row) for row in second.loot_card_title)
                     and tuple(_compact(row) for row in first.loot_card_title)
                         != tuple(_compact(row) for row in before.loot_card_title))
        changed_pixels = (before.source_pixels_sha256 != first.source_pixels_sha256
                          and before.source_pixels_sha256 != second.source_pixels_sha256)
        reason = ('loot_selection_accepted' if changed_pixels and
                  (first.scene in ('shop', 'level_up') or next_card) else 'loot_application_unconfirmed')
    accepted = reason == 'loot_selection_accepted'
    return MacroApplication(decision.decision_id, accepted, reason, actual_target, sent_at_ns, now,
                            (first.frame_id, second.frame_id),
                            (first.source_pixels_sha256, second.source_pixels_sha256),
                            first.observed_at_ns if accepted else None)


def verify_shop_choice(decision: FrozenMacroDecision, before: VerifiedShop, after: VerifiedShop | None, *,
                        sent_at_ns: int, actual_target: str, now_ns: int | None = None) -> MacroApplication:
    """Verify sampled buys/rerolls; departure needs verified combat entry."""
    now = time.perf_counter_ns() if now_ns is None else now_ns
    reason = "shop_application_unconfirmed"
    accepted, ids, hashes, next_time = False, (), (), None
    candidate = decision.observation.candidates[decision.action_index]
    if (decision.observation.phase != PHASE_SHOP or actual_target != decision.target
            or before.current.frame_id != decision.observation.frame_id
            or before.current.frame_sha256 != decision.observation.source_pixels_sha256):
        reason = "purchase_decision_evidence_mismatch"
    elif candidate.candidate_id == "skip":
        reason = "skip_requires_verified_combat_entry"
    elif (after is None or not decision.decided_at_ns <= sent_at_ns < after.first.observed_at_ns
          or not after.current.available_at_ns <= now or now - after.current.observed_at_ns > MAX_AGE_NS):
        reason = "post_purchase_observation_unverified"
    else:
        result = (verify_reroll(before, after) if candidate.candidate_id == "reroll"
                  else verify_purchase(before, after, int(candidate.candidate_id.split(":")[1])))
        accepted, reason = result.verified, result.reason
        ids = after.evidence_ids
        hashes = (after.first.frame_sha256, after.current.frame_sha256)
        next_time = after.first.observed_at_ns if accepted else None
    return MacroApplication(decision.decision_id, accepted, reason, actual_target, sent_at_ns,
                             now, ids, hashes, next_time)


def verify_shop_skip(decision: FrozenMacroDecision, combat_entry: StateEvidence | None, *,
                     sent_at_ns: int, actual_target: str, now_ns: int | None = None) -> MacroApplication:
    """Leaving a shop is accepted only after independently verified combat entry."""
    now = time.perf_counter_ns() if now_ns is None else now_ns
    candidate = decision.observation.candidates[decision.action_index]
    valid = (decision.observation.phase == PHASE_SHOP and candidate.candidate_id == "skip"
             and actual_target == decision.target and isinstance(combat_entry, StateEvidence)
             and combat_entry.kind == "combat" and combat_entry.verified is True
             and combat_entry.independent_of_policy is True and bool(combat_entry.verifier_id)
             and combat_entry.verifier_id != decision.behavior_version
             and combat_entry.origin in ("local_detector", "developer_verified")
             and combat_entry.clock_domain == "perf_counter_ns_same_host")
    if valid:
        valid = (decision.decided_at_ns <= sent_at_ns < combat_entry.observed_at_ns
                 <= combat_entry.available_at_ns <= combat_entry.verified_at_ns <= now
                 and now - combat_entry.observed_at_ns <= MAX_AGE_NS)
    if valid:
        try:
            valid = _digest(Path(combat_entry.frame_ref).read_bytes()) == combat_entry.frame_sha256
        except OSError:
            valid = False
    return MacroApplication(decision.decision_id, valid,
                             "shop_departure_observed" if valid else "verified_combat_entry_required",
                             actual_target, sent_at_ns, now,
                             (combat_entry.frame_ref,) if valid else (),
                             (combat_entry.frame_sha256,) if valid else (),
                             combat_entry.observed_at_ns if valid else None)


def save_macro_record(path: Path, decision: FrozenMacroDecision, application: MacroApplication | None = None) -> dict:
    """Exclusive immutable record, with exact RGB96 observation as a separate file.

    A pending record has application=None and is not eligible for PPO. Store a
    later acceptance in a new directory; never rewrite the original decision.
    """
    if application is not None and application.decision_id != decision.decision_id:
        raise ValueError("Application belongs to another decision")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    image_path = path / "observation.rgb96"
    with image_path.open("xb") as stream:
        stream.write(decision.observation.rgb96)
    record = asdict(decision)
    del record["observation"]["rgb96"]
    document = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
                "runtime_contract": RUNTIME_CONTRACT, "phase_schema": PHASE_SCHEMA,
                "context_schema": 'brotato-observed-build-v2' if len(decision.context) == 64 else CONTEXT_SCHEMA,
                "decision": record, "application": asdict(application) if application else None,
                "observation_file": image_path.name, "observation_sha256": _digest(decision.observation.rgb96),
                "reward_assigned": False, "requires_run_outcome_credit": True,
                "ppo_application_eligible": application is not None and application.accepted}
    with (path / "record.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return document
