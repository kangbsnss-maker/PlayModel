"""Pure neural menu decisions; synthetic fixtures are not gameplay evidence."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch is not installed")

import torch

from playmodel.learning import StateEvidence
from playmodel.learning.recurrent_ppo import RecurrentActorCritic
from playmodel.games.brotato.neural_choices import (
    _rgb96,
    observe_menu, observe_shop_choice, observe_upgrade, sample_choice, save_macro_record,
    verify_shop_choice, verify_shop_skip, verify_upgrade_choice, verify_loot_choice, stable_loot,
)
from test_shop_learning import frame as shop_pixels, report as shop_report, verified as verified_shop


BASE = 1_000_000_000


def word(text, x, y, width=150, height=25):
    return {"words": [{"text": text, "x": x, "y": y, "width": width, "height": height}]}


def upgrade_report(sequence=1, *, values=None, buttons=(0, 1, 2, 3)):
    values = values or ("+4 Max HP", "+5% Attack Speed", "+2 Armor", "+3 Harvesting")
    rows = [word("Level Up", 650, 240, 200, 40)]
    for index, left in enumerate((30, 400, 770, 1140)):
        if values[index] is not None:
            rows.append(word(values[index], left + 50, 520, 250, 35))
        if index in buttons:
            rows.append(word("Choose", left + 90, 648))
    return {"lines": rows, "processing_started_at_ns": BASE + sequence * 100_000_000 + 10_000_000,
            "recognition_path": "persistent_local_ocr"}


def arguments(sequence, *, color=95, pixels=None):
    return {"pixels": pixels or bytes((color, color, color, 255)) * 1920 * 1080,
            "frame_id": f"frame-{sequence}", "observed_at_ns": BASE + sequence * 100_000_000,
            "available_at_ns": BASE + sequence * 100_000_000 + 20_000_000}


def upgrade(sequence=1, *, color=95, **kwargs):
    return observe_upgrade(upgrade_report(sequence, **kwargs), **arguments(sequence, color=color))


def loot_report(sequence=1, *, title='Baby Gecko', body='+10 Range', take='Take', recycle='RecycIe (+5)'):
    rows = [word('ltem found!', 580, 210, 300, 40), word(recycle, 570, 775, 280, 30)]
    if title:
        rows.append(word(title, 640, 330, 240, 25))
    if body:
        rows.append(word(body, 590, 450, 260, 30))
    if take:
        rows.append(word(take, 660, 702, 130, 30))
    return {'lines': rows, 'processing_started_at_ns': BASE + sequence * 100_000_000 + 10_000_000,
            'recognition_path': 'persistent_local_ocr'}


def loot(sequence=1, *, color=95, **kwargs):
    return observe_menu(loot_report(sequence, **kwargs), **arguments(sequence, color=color))


def with_reroll(shop, *, cost=2, refreshed=False):
    """Synthetic observed cost/change; never derive a label from policy output."""
    def update(observation):
        offers = observation.offers
        if refreshed:
            # One changed unlocked slot is sufficient; other slots can be locked.
            first = offers[0]
            changed = replace(first, name="Refreshed weapon", semantic_id="weapon:refreshed",
                              pixels_sha256=hashlib.sha256(b"refreshed-card-pixels").hexdigest())
            offers = (changed, *offers[1:])
        return replace(observation, reroll_cost=cost,
                       reroll_text=("Reroll", str(cost)) if cost is not None else ("Reroll",),
                       offers=offers)
    return replace(shop, first=update(shop.first), current=update(shop.current))


class NeuralChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = RecurrentActorCritic()

    def choose(self, observation=None, **kwargs):
        observation = observation or upgrade()
        return sample_choice(self.model, observation, now_ns=observation.available_at_ns,
                             generator=torch.Generator().manual_seed(10), **kwargs)

    def test_loot_two_candidates_use_the_same_actor_with_exact_logp_and_unknown_effects(self):
        observation = loot()
        self.assertEqual(observation.phase, 1)
        self.assertEqual(observation.loot_card_title, ('Baby Gecko',))
        self.assertEqual(tuple(c.target for c in observation.candidates), ('take', 'recycle'))
        self.assertEqual(observation.legal_mask, (True, True) + (False,) * 7)
        targets = set()
        for seed in range(12):
            decision = sample_choice(self.model, observation, now_ns=observation.available_at_ns,
                                     generator=torch.Generator().manual_seed(seed))
            data = decision.tensors()
            with torch.no_grad():
                output = self.model.step(data['images'], data['context'], data['phase'], data['candidates'],
                                         data['legal_mask'], hidden=data['hidden_before'], reset=data['reset'])
            self.assertAlmostEqual(decision.old_log_probability,
                                   output.logits.log_softmax(-1)[0, decision.action_index].item(), places=6)
            self.assertEqual(tuple(data['candidates'].shape), (1, 2, 16))
            self.assertTrue(all(c.features[13] == 0 and c.features[15] == 1 for c in observation.candidates))
            targets.add(decision.target)
        self.assertEqual(targets, {'take', 'recycle'})

    def test_loot_unreadable_identity_or_action_is_not_guessed(self):
        unknown = loot(title=None)
        self.assertFalse(any(unknown.legal_mask))
        with self.assertRaisesRegex(ValueError, 'entirely illegal'):
            self.choose(unknown)
        self.assertEqual(loot(take='T4ke').legal_mask[:2], (False, True))
        self.assertEqual(loot(recycle='RecycIe unreadable').legal_mask[:2], (True, False))

    def test_loot_pre_pair_requires_independent_literal_identity_not_tooltip_body(self):
        first, second = loot(1), loot(2, body='tooltip occlusion is unknown')
        self.assertTrue(stable_loot(first, second))
        self.assertFalse(stable_loot(first, first))
        self.assertFalse(stable_loot(first, loot(2, title='Other item')))
        self.assertFalse(stable_loot(first, replace(second, frame_id=first.frame_id)))
        self.assertFalse(stable_loot(first, replace(second, observed_at_ns=first.available_at_ns)))
        self.assertFalse(stable_loot(first, replace(second, ocr_started_at_ns=first.ocr_started_at_ns)))

    def test_loot_application_needs_two_post_send_frames_and_a_consumed_card(self):
        decision = self.choose(loot())
        sent = decision.decided_at_ns + 1_000_000
        def verify(first, second, **kwargs):
            return verify_loot_choice(decision, first, second, sent_at_ns=sent,
                                      actual_target=kwargs.get('target', decision.target),
                                      now_ns=kwargs.get('now', second.available_at_ns))
        # Mere focus/tooltip changes leave the same card and are not acceptance.
        self.assertFalse(verify(loot(2, color=110, body='changed tooltip'), loot(3, color=111)).accepted)
        next_first, next_second = loot(2, color=110, title='New item'), loot(3, color=110, title='New item')
        self.assertTrue(verify(next_first, next_second).accepted)
        self.assertFalse(verify(next_first, replace(next_second, frame_id=next_first.frame_id)).accepted)
        self.assertFalse(verify(next_first, loot(3, color=110, title='Another item')).accepted)
        self.assertFalse(verify(next_first, next_second, target='ban').accepted)
        self.assertFalse(verify(next_first, next_second, now=next_second.observed_at_ns + 800_000_000).accepted)
        accepted = verify(upgrade(2, color=110), upgrade(3, color=110))
        self.assertTrue(accepted.accepted)
        self.assertFalse(accepted.feature_effects_verified)
        self.assertEqual(accepted.reason, 'loot_selection_accepted')
        self.assertFalse(verify_upgrade_choice(decision, next_first, next_second, sent_at_ns=sent,
                                              actual_target=decision.target, now_ns=next_second.available_at_ns).accepted)

    def test_actual_loot_evidence_keeps_literal_title_and_two_supported_actions(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        from playmodel.games.brotato.session import calibrated_rules
        root = (Path(__file__).resolve().parents[1] /
                'artifacts/recurrent-cycles/20260927T065557Z-81c8e1db/'
                'partial-recovery-e21c0140420a4c51a64d0e020ce957a2/segments/'
                '20260927T082700Z-a80eae59/menus')
        names = ('20260927T082702Z-5a12accb', '20260927T082702Z-a4665a90', '20260927T082703Z-89c07778')
        if not all((root / name / 'ocr.json').is_file() for name in names):
            self.skipTest('private loot calibration evidence absent')
        identities = []
        for name in names:
            directory = root / name
            raw = json.loads((directory / 'ocr.json').read_text(encoding='utf-8'))
            metadata = json.loads((directory / 'observation.json').read_text(encoding='utf-8'))
            original = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            pixels, width, height = read_diagnostic_png(directory / 'frame.png')
            observation = observe_menu(raw, pixels=pixels, width=width, height=height,
                frame_id=str(directory / 'frame.png'), observed_at_ns=metadata['capture_started_at_ns'],
                available_at_ns=raw['available_at_ns'])
            self.assertEqual(observation.loot_card_title, ('BabyGecko',))
            self.assertEqual(observation.legal_mask[:2], (True, True))
            self.assertEqual(calibrated_rules().classify(raw['text'], lines=raw['lines'], image_size=(width, height)), 'wave_clear')
            self.assertEqual(json.dumps(raw, ensure_ascii=False, sort_keys=True), original)
            identities.append(observation.card_set_sha256)
        self.assertEqual(len(set(identities)), 1)

    def test_fast_rgb_serialization_preserves_the_exact_calibrated_pixels(self):
        width, height = 192, 108
        raw = torch.randint(0, 256, (height, width, 4), dtype=torch.uint8)
        pixels = raw.numpy().tobytes()
        rgb = raw[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        small = torch.nn.functional.interpolate(rgb.float(), size=(96, 96), mode="area").round().to(torch.uint8)
        legacy = bytes(small[0].permute(1, 2, 0).contiguous().untyped_storage())
        self.assertEqual(_rgb96(pixels, width, height), legacy)
        self.assertEqual(len(legacy), 96 * 96 * 3)

    def test_shared_actor_output_and_exact_old_log_probability_are_preserved(self):
        decision = self.choose()
        data = decision.tensors()
        self.assertEqual(data["phase"].item(), 1)
        self.assertEqual(tuple(data["images"].shape), (1, 3, 96, 96))
        self.assertEqual(tuple(data["candidates"].shape), (1, 4, 16))
        with torch.no_grad():
            output = self.model.step(data["images"], data["context"], data["phase"], data["candidates"],
                                     data["legal_mask"], hidden=data["hidden_before"], reset=data["reset"])
        self.assertAlmostEqual(decision.old_log_probability,
                               float(output.logits.log_softmax(-1)[0, decision.action_index]), places=6)
        self.assertEqual(decision.behavior_version, self.model.policy_version())
        self.assertTrue(torch.equal(output.next_hidden, data["next_hidden"]))

    def test_sampled_choice_uses_mask_and_does_not_rank_by_handwritten_stat_weights(self):
        observation = upgrade(buttons=(1,))
        decision = self.choose(observation)
        self.assertEqual(decision.target, "choose_1")
        self.assertEqual(decision.probabilities[1], 1.0)
        all_cards = upgrade()
        choices = {sample_choice(self.model, all_cards, now_ns=all_cards.available_at_ns,
                                  generator=torch.Generator().manual_seed(seed)).action_index for seed in range(12)}
        self.assertGreater(len(choices), 1)

    def test_unknown_text_effects_remain_explicit_and_no_slot_in_features(self):
        first = upgrade(values=("+2 Armor", None, "+2 Armor", "+2 Armor"))
        self.assertEqual(first.candidates[0].features, first.candidates[2].features)
        self.assertEqual(first.candidates[1].features[14:], (0.0, 1.0))
        self.assertTrue(first.candidates[1].legal)  # Visible free choice remains an exploratory option.
        self.assertTrue(all(-1 <= value <= 1 for c in first.candidates for value in c.features))

    def test_immutable_decision_survives_mutated_input_and_returned_tensors(self):
        hidden = torch.ones(1, 128)
        decision = self.choose(hidden=hidden)
        hidden.zero_()
        self.assertEqual(decision.hidden_before, (1.0,) * 128)
        first = decision.tensors()
        first["images"].zero_()
        first["context"].zero_()
        self.assertGreater(float(decision.tensors()["images"].sum()), 0)
        self.assertEqual(decision.context[0], 1.0)
        with self.assertRaises(FrozenInstanceError):
            decision.target = "choose_0"

    def test_cached_stale_wrong_build_and_no_legal_choice_abstain(self):
        report = upgrade_report()
        with self.assertRaisesRegex(ValueError, "Fresh"):
            observe_upgrade({**report, "recognition_path": "exact_image_cache"}, **arguments(1))
        with self.assertRaisesRegex(ValueError, "Uncalibrated"):
            observe_upgrade(report, **arguments(1), game_build_id="changed")
        observation = upgrade()
        with self.assertRaisesRegex(ValueError, "Stale"):
            sample_choice(self.model, observation, now_ns=observation.available_at_ns + 800_000_000)
        no_legal = replace(observation, candidates=tuple(replace(c, legal=False) for c in observation.candidates))
        with self.assertRaisesRegex(ValueError, "illegal"):
            self.choose(no_legal)

    def test_upgrade_acceptance_requires_post_send_stable_change(self):
        decision = self.choose()
        new_values = ("+1 Speed", "+1 Luck", "+2 Damage", "+1 Harvesting")
        first, second = upgrade(3, color=100, values=new_values), upgrade(4, color=100, values=new_values)
        sent = BASE + 250_000_000
        accepted = verify_upgrade_choice(decision, first, second, sent_at_ns=sent,
                                          actual_target=decision.target, now_ns=second.available_at_ns)
        self.assertTrue(accepted.accepted, accepted)
        self.assertFalse(accepted.feature_effects_verified)
        self.assertFalse(hasattr(accepted, "reward"))
        unchanged = verify_upgrade_choice(decision, upgrade(3), upgrade(4), sent_at_ns=sent,
                                           actual_target=decision.target, now_ns=second.available_at_ns)
        self.assertFalse(unchanged.accepted)
        wrong_target = verify_upgrade_choice(decision, first, second, sent_at_ns=sent,
                                              actual_target="choose_99", now_ns=second.available_at_ns)
        self.assertFalse(wrong_target.accepted)
        too_early = verify_upgrade_choice(decision, first, second, sent_at_ns=first.observed_at_ns,
                                           actual_target=decision.target, now_ns=second.available_at_ns)
        self.assertFalse(too_early.accepted)

    def test_upgrade_accepts_independently_observed_shop_transition(self):
        decision = self.choose()
        first = observe_menu(shop_report(sequence=3), **arguments(3, pixels=shop_pixels()))
        second = observe_menu(shop_report(sequence=4), **arguments(4, pixels=shop_pixels()))
        result = verify_upgrade_choice(decision, first, second, sent_at_ns=BASE + 250_000_000,
                                         actual_target=decision.target, now_ns=second.available_at_ns)
        self.assertTrue(result.accepted)

    def test_shop_candidates_and_currency_bound_to_exact_verified_frame(self):
        shop = verified_shop(currency=20)
        observation = observe_shop_choice(shop, shop_report(sequence=2, currency=20),
                                           **arguments(2, pixels=shop_pixels()))
        decision = self.choose(observation)
        self.assertEqual(decision.observation.phase, 3)
        self.assertEqual(observation.legal_mask, (True, True, False, False, True, False, False, False, False))
        self.assertEqual(decision.context[12], 1.0)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            observe_shop_choice(shop, shop_report(sequence=2, currency=20),
                                 **arguments(2, color=10))

    def test_verified_purchase_required_for_sampled_shop_action(self):
        shop = verified_shop()
        observation = observe_shop_choice(shop, shop_report(sequence=2), **arguments(2, pixels=shop_pixels()))
        # Legal mask comes from collector facts. Restrict to one legal action to
        # exercise its exact transaction without test dependence on RNG output.
        observation = replace(observation, candidates=tuple(replace(c, legal=index == 0)
                                                            for index, c in enumerate(observation.candidates)))
        decision = self.choose(observation)
        after = verified_shop(3, currency=66, absent=(0,), count=3)
        result = verify_shop_choice(decision, shop, after, sent_at_ns=BASE + 250_000_000,
                                      actual_target=decision.target, now_ns=after.current.available_at_ns)
        self.assertTrue(result.accepted, result)
        wrong_spend = verified_shop(3, currency=65, absent=(0,), count=3)
        self.assertFalse(verify_shop_choice(decision, shop, wrong_spend, sent_at_ns=BASE + 250_000_000,
                                              actual_target=decision.target, now_ns=after.current.available_at_ns).accepted)

    def test_reroll_uses_existing_features_and_requires_observed_spend_and_changed_offer(self):
        shop = with_reroll(verified_shop(currency=20))
        observation = observe_shop_choice(shop, shop_report(sequence=2, currency=20),
                                           **arguments(2, pixels=shop_pixels()))
        self.assertEqual([candidate.candidate_id for candidate in observation.candidates],
                         ["buy:0", "buy:1", "buy:2", "buy:3", "skip", "reroll"])
        reroll = observation.candidates[5]
        self.assertTrue(reroll.legal)
        self.assertEqual((reroll.target, reroll.kind, reroll.semantic_id),
                         ("refresh", "reroll", "shop:reroll"))
        self.assertEqual(len(reroll.features), 16)
        self.assertEqual(reroll.features[8:12], (0.0,) * 4)
        self.assertEqual(reroll.features[12:], (.1, 1.0, 1.0, 1.0))
        self.assertTrue(any(reroll.features[:8]))
        observation = replace(observation, candidates=tuple(replace(candidate, legal=index == 5)
                                                            for index, candidate in enumerate(observation.candidates)))
        decision = self.choose(observation)
        self.assertEqual((decision.action_index, decision.target, decision.probabilities[5]),
                         (5, "refresh", 1.0))
        self.assertEqual(tuple(decision.tensors()["candidates"].shape), (1, 6, 16))
        after = with_reroll(verified_shop(3, currency=18), cost=3, refreshed=True)
        result = verify_shop_choice(decision, shop, after, sent_at_ns=BASE + 250_000_000,
                                    actual_target="refresh", now_ns=after.current.available_at_ns)
        self.assertTrue(result.accepted, result)
        self.assertEqual(result.reason, "reroll_observed")
        self.assertEqual(result.after_frame_ids, after.evidence_ids)
        self.assertEqual(result.next_observed_at_ns, after.first.observed_at_ns)
        self.assertFalse(result.feature_effects_verified)
        self.assertFalse(hasattr(result, "reward"))
        for invalid in (with_reroll(verified_shop(3, currency=19), refreshed=True),
                        with_reroll(verified_shop(3, currency=18)),
                        with_reroll(verified_shop(3, currency=18, image=shop_pixels(inventory_change=True)),
                                    refreshed=True)):
            result = verify_shop_choice(decision, shop, invalid, sent_at_ns=BASE + 250_000_000,
                                        actual_target="refresh", now_ns=invalid.current.available_at_ns)
            self.assertFalse(result.accepted, result)
            self.assertIsNone(result.next_observed_at_ns)
        self.assertFalse(verify_shop_choice(decision, shop, after, sent_at_ns=after.first.observed_at_ns,
                                            actual_target="refresh", now_ns=after.current.available_at_ns).accepted)
        self.assertFalse(verify_shop_choice(decision, shop, after, sent_at_ns=BASE + 250_000_000,
                                            actual_target="buy_0", now_ns=after.current.available_at_ns).accepted)

    def test_unknown_currency_masks_every_spending_action_and_preserves_known_zero_distinction(self):
        shop = with_reroll(verified_shop(currency="?"))
        observation = observe_shop_choice(shop, shop_report(sequence=2, currency="?"),
                                           **arguments(2, pixels=shop_pixels()))
        self.assertIsNone(observation.currency)
        self.assertEqual(observation.legal_mask, (False, False, False, False, True, False, False, False, False))
        self.assertEqual(observation.candidates[5].features[12:14], (0.0, 0.0))
        decision = self.choose(observation)
        self.assertEqual((decision.target, decision.probabilities[4]), ("depart", 1.0))
        self.assertEqual(decision.context[11:13], (0.0, 0.0))
        zero = with_reroll(verified_shop(currency=0))
        known_zero = observe_shop_choice(zero, shop_report(sequence=2, currency=0),
                                          **arguments(2, pixels=shop_pixels()))
        self.assertEqual(self.choose(known_zero).context[11:13], (0.0, 1.0))

    def test_unknown_or_unaffordable_reroll_cost_cannot_be_sampled(self):
        for cost in (None, 21):
            with self.subTest(cost=cost):
                shop = with_reroll(verified_shop(currency=20), cost=cost)
                observation = observe_shop_choice(shop, shop_report(sequence=2, currency=20),
                                                   **arguments(2, pixels=shop_pixels()))
                self.assertFalse(observation.candidates[5].legal)
                decision = self.choose(observation)
                self.assertEqual(decision.probabilities[5], 0.0)
                self.assertNotEqual(decision.action_index, 5)

    def test_pending_and_accepted_evidence_saved_separately_without_reward(self):
        decision = self.choose()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pending"
            document = save_macro_record(path, decision)
            self.assertFalse(document["ppo_application_eligible"])
            self.assertFalse(document["reward_assigned"])
            self.assertEqual((path / "observation.rgb96").read_bytes(), decision.observation.rgb96)
            self.assertEqual(json.loads((path / "record.json").read_text())["schema"], "playmodel.neural-menu-macro.v2")
            with self.assertRaises(FileExistsError):
                save_macro_record(path, decision)

    def test_skip_needs_backed_independent_combat_entry(self):
        shop = verified_shop()
        observation = observe_shop_choice(shop, shop_report(sequence=2), **arguments(2, pixels=shop_pixels()))
        observation = replace(observation, candidates=tuple(replace(c, legal=index == 4)
                                                            for index, c in enumerate(observation.candidates)))
        decision = self.choose(observation)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "combat.png"
            path.write_bytes(b"independent synthetic evidence")
            evidence = StateEvidence("combat", str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                                     BASE + 300_000_000, BASE + 310_000_000, BASE + 320_000_000,
                                     "local_detector", "calibrated-local-combat", True, True)
            result = verify_shop_skip(decision, evidence, sent_at_ns=BASE + 250_000_000,
                                        actual_target="depart", now_ns=BASE + 330_000_000)
            self.assertTrue(result.accepted, result)
            invalid = replace(evidence, independent_of_policy=False)
            self.assertFalse(verify_shop_skip(decision, invalid, sent_at_ns=BASE + 250_000_000,
                                               actual_target="depart", now_ns=BASE + 330_000_000).accepted)


if __name__ == "__main__":
    unittest.main()
