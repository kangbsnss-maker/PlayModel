"""Shop action contracts: synthetic geometry plus optional private real capture."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from playmodel.games.brotato.shop_learning import (
    CARD_LEFTS, GAME_BUILD, observe_shop, purchase_candidates, stable_shop, verify_purchase, verify_reroll,
)


BASE_TIME = 1_000_000_000


def word(text, x, y, width=70, height=22):
    return {"words": [{"text": text, "x": x, "y": y, "width": width, "height": height}]}


def report(currency=80, count=2, capacity=6, *, absent=(), prices=(14, 18, 23, 28), sequence=1,
           reroll_cost=None):
    rows = [word("Shop (Wave 1)", 25, 45, 285, 40), word("GO", 1550, 995),
            word(str(currency), 835, 45), word(f"Weapons ({count}/{capacity})", 1130, 785, 280)]
    if reroll_cost is not None:
        rows += [word('REROLL -', 1170, 45, 175, 40), word(str(reroll_cost), 1350, 45, 40, 40)]
    for slot, left in enumerate(CARD_LEFTS):
        if slot in absent:
            continue
        rows += [word(("Ghost Flint", "Crossbow", "Insanity", "Cute Monkey")[slot], left + 125, 160, 180),
                 word("Precise" if slot < 2 else "Item", left + 125, 195, 100)]
        if slot < 2:
            rows += [word("Damage: 6", left + 18, 270, 200),
                     word("Cooldown: 0.98s", left + 18, 320, 200),
                     word("Range: 170", left + 18, 370, 200)]
        else:
            rows += [word("+6% Critical Chance", left + 18, 270, 280)]
        if prices[slot] is not None:
            rows += [word(str(prices[slot]), (142, 508, 867, 1230)[slot], 560, 50, 34)]
        # Currency icon commonly OCRs as zero; pixel color must exclude it.
        rows += [word("0", (212, 575, 940, 1300)[slot], 553, 40, 42)]
    return {"lines": rows, "processing_started_at_ns": BASE_TIME + sequence * 100_000_000 + 10_000_000,
            "recognition_path": "persistent_local_ocr", "verified": False}


def frame(*, absent=(), inventory_change=False):
    image = bytearray(bytes((95, 95, 95, 255)) * 1920 * 1080)
    def fill(rect, color):
        left, top, right, bottom = rect
        row = bytes(color) * (right - left)
        for y in range(top, bottom):
            start = (y * 1920 + left) * 4
            image[start:start + len(row)] = row
    for slot, left in enumerate(CARD_LEFTS):
        if slot not in absent:
            fill((left, 142, left + 350, 625), (12, 12, 12, 255))
            icon = (212, 575, 940, 1300)[slot]
            fill((icon, 553, icon + 40, 595), (40, 170, 50, 255))
    if inventory_change:
        fill((1138, 852, 1200, 925), (40, 110, 180, 255))
        fill((35, 855, 85, 920), (110, 40, 180, 255))
    return bytes(image)


def observation(sequence, *, image=None, **kwargs):
    return observe_shop(report(sequence=sequence, **kwargs), pixels=image or frame(absent=kwargs.get("absent", ())),
                        observed_at_ns=BASE_TIME + sequence * 100_000_000,
                        available_at_ns=BASE_TIME + sequence * 100_000_000 + 20_000_000,
                        frame_id=f"frame-{sequence}", game_build_id=GAME_BUILD)


def verified(start=1, **kwargs):
    first, second = observation(start, **kwargs), observation(start + 1, **kwargs)
    return stable_shop(first, second, now_ns=BASE_TIME + (start + 1) * 100_000_000 + 20_000_000)


class ShopLearningTests(unittest.TestCase):
    def test_affordable_observed_choices_and_expensive_mask(self):
        shop = verified(currency=20)
        candidates = purchase_candidates(shop, now_ns=shop.current.available_at_ns)
        self.assertEqual([c.action_id for c in candidates], ["buy:0", "buy:1", "buy:2", "buy:3", "skip", "reroll"])
        self.assertEqual([c.cost for c in candidates], [14, 18, 23, 28, 0, None])
        self.assertEqual([c.legal for c in candidates], [True, True, False, False, True, False])
        self.assertEqual([c.kind for c in candidates], ["weapon", "weapon", "item", "item", "skip", "reroll"])
        self.assertTrue(all(0 <= f <= 1 for c in candidates for f in c.features))

    def test_missing_price_is_unknown_never_free(self):
        shop = verified(prices=(None, 18, None, None))
        candidates = purchase_candidates(shop, now_ns=shop.current.available_at_ns)
        self.assertEqual([c.legal for c in candidates], [False, True, False, False, True, False])
        self.assertEqual(candidates[0].reason, "price_unreadable")
        self.assertIsNone(candidates[0].cost)

    def test_currency_with_leading_zeros_is_unknown_not_zero(self):
        for text in ("00", "064", "00001"):
            self.assertIsNone(observation(1, currency=text).currency)
        self.assertEqual(observation(1, currency="0").currency, 0)
        self.assertEqual(observation(1, currency="64").currency, 64)

    def test_unknown_currency_keeps_only_independently_verified_departure(self):
        shop = verified(currency=None, reroll_cost=2)
        self.assertIsNone(shop.current.currency)
        self.assertEqual([c.legal for c in purchase_candidates(shop, now_ns=shop.current.available_at_ns)],
                         [False, False, False, False, True, False])
        first = observation(1, currency=None)
        self.assertIsNone(stable_shop(first, first, now_ns=first.available_at_ns))
        self.assertIsNone(stable_shop(first, observation(2, currency=62),
                                     now_ns=BASE_TIME + 220_000_000))

    def test_reroll_cost_is_literal_affordable_and_independently_agrees(self):
        for cost, legal in ((0, True), (2, True), (90, False), ('00', False), ('O', False)):
            shop = verified(reroll_cost=cost)
            candidate = purchase_candidates(shop, now_ns=shop.current.available_at_ns)[5]
            self.assertEqual(candidate.legal, legal)
            self.assertEqual(candidate.target, 'refresh')
        self.assertIsNone(stable_shop(observation(1, reroll_cost=2), observation(2, reroll_cost=3),
                                     now_ns=BASE_TIME + 220_000_000))

    def test_reroll_requires_exact_spend_changed_offer_and_unchanged_inventory(self):
        before = verified(reroll_cost=2)
        after = verified(3, currency=78, reroll_cost=3)
        def changed(shop):
            def frame_change(row):
                offers = list(row.offers)
                offers[0] = replace(offers[0], name='New weapon', semantic_id='weapon:new', pixels_sha256='new-pixels')
                return replace(row, offers=tuple(offers))
            return replace(shop, first=frame_change(shop.first), current=frame_change(shop.current))
        after = changed(after)
        self.assertTrue(verify_reroll(before, after).verified)
        self.assertEqual(verify_reroll(before, after).spent, 2)
        # Three unchanged offers may be locked; they do not invalidate a reroll.
        self.assertEqual(verify_reroll(before, verified(3, currency=78)).reason,
                         'reroll_offers_unchanged_or_unreadable')
        for invalid in (changed(verified(3, currency=77)), changed(verified(3, currency=None)),
                        changed(verified(3, currency=78, image=frame(inventory_change=True))),
                        replace(after, item_inventory_stable=False),
                        replace(after, current=replace(after.current, wave=2))):
            self.assertFalse(verify_reroll(before, invalid).verified)
        free = verified(reroll_cost=0)
        self.assertTrue(verify_reroll(free, changed(verified(3, currency=80))).verified)
        self.assertFalse(verify_purchase(before, changed(verified(3, currency=66, absent=(0,))), 0).verified)

    def test_full_weapon_slots_abstain_but_item_purchase_stays_legal(self):
        shop = verified(count=6)
        candidates = purchase_candidates(shop, now_ns=shop.current.available_at_ns)
        self.assertFalse(candidates[0].legal)
        self.assertEqual(candidates[0].reason, "weapon_combine_or_replace_unsupported")
        self.assertTrue(candidates[2].legal)
        unknown = replace(shop, current=replace(shop.current, weapon_count=None, weapon_capacity=None))
        self.assertEqual(purchase_candidates(unknown, now_ns=shop.current.available_at_ns)[0].reason,
                         "weapon_capacity_unreadable")

    def test_two_frames_must_be_independent_ordered_fresh_and_agree(self):
        first, second = observation(1), observation(2)
        now = second.available_at_ns
        self.assertIsNone(stable_shop(first, first, now_ns=now))
        self.assertIsNone(stable_shop(second, first, now_ns=now))
        self.assertIsNone(stable_shop(first, replace(second, ocr_started_at_ns=first.ocr_started_at_ns), now_ns=now))
        self.assertIsNone(stable_shop(first, replace(second, currency=79), now_ns=now))
        self.assertIsNone(stable_shop(first, second, now_ns=now + 800_000_000))
        self.assertIsNotNone(stable_shop(first, second, now_ns=now))
        self.assertFalse(any(c.legal for c in purchase_candidates(verified(), now_ns=now + 800_000_000)))

    def test_cached_ocr_and_bad_calibration_never_observe(self):
        ocr = report(sequence=1)
        args = dict(pixels=frame(), observed_at_ns=BASE_TIME + 100_000_000,
                    available_at_ns=BASE_TIME + 120_000_000, frame_id="one", game_build_id=GAME_BUILD)
        self.assertIsNone(observe_shop({**ocr, "recognition_path": "exact_image_cache"}, **args))
        self.assertIsNone(observe_shop({**ocr, "cache_source": "older.png"}, **args))
        self.assertIsNone(observe_shop(ocr, **{**args, "game_build_id": "new-build"}))
        self.assertIsNone(observe_shop(ocr, **{**args, "width": 1280}))
        self.assertIsNone(observe_shop(ocr, **{**args, "pixels": b""}))
        modal = deepcopy(ocr)
        modal["lines"].append(word("Replace weapon", 800, 400, 250))
        self.assertIsNone(observe_shop(modal, **args))

    def test_exact_spend_and_disappeared_card_confirm_purchase_without_reward(self):
        before = verified()
        after = verified(3, currency=66, absent=(0,), count=3)
        result = verify_purchase(before, after, 0)
        self.assertTrue(result.verified, result)
        self.assertEqual(result.spent, 14)
        self.assertTrue(result.offer_removed)
        self.assertFalse(hasattr(result, "reward"))

    def test_currency_change_alone_never_confirms_application(self):
        before = verified()
        same_offer = verified(3, currency=66)
        result = verify_purchase(before, same_offer, 0)
        self.assertFalse(result.verified)
        self.assertEqual(result.reason, "purchase_application_unconfirmed")
        wrong_delta = verified(3, currency=65, absent=(0,))
        self.assertEqual(verify_purchase(before, wrong_delta, 0).reason, "currency_delta_mismatch")
        self.assertEqual(verify_purchase(before, None, 0).reason, "post_purchase_unverified")

    def test_missing_ocr_title_on_visible_card_is_not_purchase(self):
        before = verified()
        # OCR vanished but panel pixels still show the offered card.
        after = verified(3, currency=66, absent=(0,), image=frame())
        self.assertFalse(verify_purchase(before, after, 0).verified)

    def test_inventory_change_and_slot_increment_confirm_weapon_application(self):
        before = verified()
        after = verified(3, currency=66, count=3, image=frame(inventory_change=True))
        result = verify_purchase(before, after, 0)
        self.assertTrue(result.verified)
        self.assertTrue(result.inventory_changed)
        self.assertFalse(result.offer_removed)
        unchanged_count = verified(3, currency=66, image=frame(inventory_change=True))
        self.assertFalse(verify_purchase(before, unchanged_count, 0).verified)

    def test_item_inventory_evidence_and_unstable_inventory_rejection(self):
        before = verified()
        after = verified(3, currency=57, image=frame(inventory_change=True))
        self.assertTrue(verify_purchase(before, after, 2).verified)
        self.assertFalse(verify_purchase(before, replace(after, item_inventory_stable=False), 2).verified)

    def test_reroll_and_old_run_cannot_be_misattributed_as_purchase(self):
        before = verified()
        after = verified(3, currency=66, absent=(0,))
        changed_offers = list(after.current.offers)
        changed_offers[1] = replace(changed_offers[1], semantic_id="another-offer")
        rerolled = replace(after, current=replace(after.current, offers=tuple(changed_offers)))
        self.assertEqual(verify_purchase(before, rerolled, 0).reason, "unrelated_offers_changed")
        other_wave = replace(after, current=replace(after.current, wave=2))
        self.assertEqual(verify_purchase(before, other_wave, 0).reason, "mismatched_purchase_sequence")
        self.assertFalse(verify_purchase(before, before, 0).verified)

    def test_private_real_english_shop_parses_available_price_without_guessing_others(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        folder = (Path(__file__).resolve().parents[1] / "artifacts/brotato-sessions"
                  / "20260927T012307Z-fd93badc/menus/20260927T012335Z-21a89140")
        if not (folder / "frame.png").exists():
            self.skipTest("private game images are excluded from Git")
        pixels, width, height = read_diagnostic_png(folder / "frame.png")
        ocr = json.loads((folder / "ocr.json").read_text(encoding="utf-8"))
        observed = observe_shop(ocr, pixels=pixels, width=width, height=height,
                                observed_at_ns=ocr["processing_started_at_ns"] - 1,
                                available_at_ns=ocr["available_at_ns"], frame_id=str(folder),
                                game_build_id=GAME_BUILD)
        self.assertIsNotNone(observed)
        self.assertEqual((observed.currency, observed.wave), (51, 1))
        self.assertEqual((observed.weapon_count, observed.weapon_capacity), (2, 6))
        self.assertEqual([o.cost for o in observed.offers], [None, 18, None, None])
        self.assertEqual([o.kind for o in observed.offers], ["weapon", "weapon", "item", "item"])

    def test_private_actual_62_ocr_failure_remains_unknown_but_allows_skip(self):
        from playmodel.games.brotato.capture import read_diagnostic_png
        directory = (Path(__file__).resolve().parents[1] / 'artifacts/recurrent-cycles'
            / '20260927T062951Z-23f23cf4/partial-recovery-345998b813ae40eba187334caa317ea9'
            / 'segments/20260927T062955Z-0f0e4724/menus')
        folders = [directory / name for name in ('20260927T063128Z-b199b837', '20260927T063128Z-dab1786a')]
        if not all((folder / 'frame.png').exists() for folder in folders):
            self.skipTest('private game images are excluded from Git')
        rows = []
        for folder in folders:
            shot = json.loads((folder / 'observation.json').read_text(encoding='utf-8'))
            ocr = json.loads((folder / 'ocr.json').read_text(encoding='utf-8'))
            fallback = json.loads((folder / 'currency-roi.json').read_text(encoding='utf-8'))
            self.assertIsNone(fallback['currency'])
            self.assertEqual(fallback['raw_text'], '℃ 2 62 62')
            pixels, width, height = read_diagnostic_png(folder / 'frame.png')
            rows.append(observe_shop(ocr, pixels=pixels, width=width, height=height,
                observed_at_ns=shot['capture_started_at_ns'],
                available_at_ns=max(ocr['available_at_ns'], fallback['available_at_ns']),
                frame_id=str((folder / 'frame.png').resolve()), game_build_id=GAME_BUILD,
                currency_observation=fallback))
        rows.sort(key=lambda row: row.observed_at_ns)
        shop = stable_shop(*rows, now_ns=rows[1].available_at_ns)
        self.assertIsNotNone(shop)
        self.assertIsNone(shop.current.currency)
        self.assertEqual([c.target for c in purchase_candidates(shop, now_ns=shop.current.available_at_ns) if c.legal],
                         ['depart'])


if __name__ == "__main__":
    unittest.main()
