"""Observed shop actions for local reinforcement learning, without game input.

The calibrated adapter exposes affordable purchases, rerolls and leaving. Two
fresh captures with separate OCR executions must agree before spending; cached
OCR never supplies a second verification sample.  Agreement is an operational
check, not a claim that OCR is ground truth.  Purchase verification is separate
from policy reward: buying something is not evidence that it improves a run.

Only the normal English 1920x1080 shop in build 23429717 is supported.  Weapon
replacement/combining and unreadable prices abstain.  Callers keep exclusive
input ownership, bind OCR to the supplied frame, and revalidate before Enter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
import time
import unicodedata

from .menu import BUTTONS, classify_scene
from .ocr import rows_in_region


GAME_BUILD = "23429717"
WIDTH, HEIGHT = 1920, 1080
CARD_LEFTS = (25, 386, 747, 1108)
MAX_OBSERVATION_AGE_NS = 750_000_000


def _compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _region_digest(pixels: bytes, rect: tuple[int, int, int, int]) -> str:
    left, top, right, bottom = rect
    digest = hashlib.sha256()
    for y in range(top, bottom):
        digest.update(pixels[(y * WIDTH + left) * 4:(y * WIDTH + right) * 4])
    return digest.hexdigest()


@dataclass(frozen=True)
class ShopOffer:
    slot: int
    name: str
    category: str
    kind: str
    text: tuple[str, ...]
    cost: int | None
    semantic_id: str | None
    present: bool | None
    pixels_sha256: str


@dataclass(frozen=True)
class ShopObservation:
    frame_id: str
    frame_sha256: str
    game_build_id: str
    observed_at_ns: int
    available_at_ns: int
    ocr_started_at_ns: int
    currency: int | None
    wave: int
    weapon_count: int | None
    weapon_capacity: int | None
    offers: tuple[ShopOffer, ...]
    build_text: tuple[str, ...]
    item_inventory_sha256: str
    weapon_inventory_sha256: str
    currency_evidence: dict | None = None
    reroll_cost: int | None = None
    reroll_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedShop:
    """Two agreeing observations; not a human label or semantic game oracle."""
    first: ShopObservation
    current: ShopObservation
    item_inventory_stable: bool
    weapon_inventory_stable: bool

    @property
    def evidence_ids(self) -> tuple[str, str]:
        return self.first.frame_id, self.current.frame_id


@dataclass(frozen=True)
class ShopCandidate:
    action_id: str
    target: str
    semantic_id: str
    kind: str
    slot: int | None
    cost: int | None
    text: tuple[str, ...]
    legal: bool
    reason: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class PurchaseVerification:
    verified: bool
    reason: str
    action_id: str
    evidence_ids: tuple[str, ...]
    spent: int | None = None
    offer_removed: bool = False
    inventory_changed: bool = False
    # There is intentionally no reward field. Run outcome supplies RL credit.


@dataclass(frozen=True)
class RerollVerification:
    verified: bool
    reason: str
    action_id: str
    evidence_ids: tuple[str, ...]
    spent: int | None = None
    offers_changed: bool = False


def _number(ocr: dict, region: tuple[int, int, int, int]) -> int | None:
    text = "".join(rows_in_region(ocr, region))
    # Currency uses canonical integer spelling. A leading-zero OCR result such
    # as "00" is ambiguous, not zero, and needs an independent ROI reading.
    return int(text) if re.fullmatch(r"(?:0|[1-9][0-9]{0,4})", text) else None


def _price(ocr: dict, pixels: bytes, slot: int) -> int | None:
    """Exclude the green currency icon, including icons OCR misreads as zero."""
    left, top, right, bottom = BUTTONS["shop"][f"buy_{slot}"]
    words = []
    for line in ocr["lines"]:
        for word in line["words"]:
            x, y, width, height = (word[key] for key in ("x", "y", "width", "height"))
            if not (left - 35 <= x + width / 2 <= right + 35
                    and top <= y + height / 2 <= bottom):
                continue
            colored = total = 0
            for py in range(int(y), int(y + height), 2):
                for px in range(int(x), int(x + width), 2):
                    offset = (py * WIDTH + px) * 4
                    blue, green, red = pixels[offset:offset + 3]
                    colored += green > 55 and green > red * 1.35 and green > blue * 1.35
                    total += 1
            if total and colored / total > .12:
                continue
            words.append(word)
    if not words:
        return None
    # One horizontal numeric group only. No O->0, l->1, punctuation removal,
    # text-level icon stripping, or best-guess price recovery.
    if max(w["y"] for w in words) - min(w["y"] for w in words) > 15:
        return None
    text = "".join(w["text"] for w in sorted(words, key=lambda w: w["x"]))
    return int(text) if re.fullmatch(r"[0-9]{1,5}", text) else None


def _presence(pixels: bytes, left: int) -> bool | None:
    # Calibrated dark card panel, excluding most art, buttons and scrollbars.
    dark = total = 0
    for y in range(255, 535, 12):
        for x in range(left + 8, left + 342, 12):
            offset = (y * WIDTH + x) * 4
            dark += max(pixels[offset:offset + 3]) < 40
            total += 1
    ratio = dark / total
    return True if ratio >= .60 else False if ratio <= .15 else None


def observe_shop(ocr: dict, *, pixels: bytes, observed_at_ns: int,
                 available_at_ns: int, frame_id: str, game_build_id: str,
                 width: int = WIDTH, height: int = HEIGHT,
                 currency_observation=None) -> ShopObservation | None:
    """Parse one bound frame. Invalid, cached, modal or uncalibrated data abstain."""
    if (game_build_id != GAME_BUILD or (width, height) != (WIDTH, HEIGHT)
            or not isinstance(pixels, bytes) or len(pixels) != WIDTH * HEIGHT * 4
            or not isinstance(frame_id, str) or not frame_id.strip()
            or type(observed_at_ns) is not int or type(available_at_ns) is not int
            or observed_at_ns <= 0 or available_at_ns < observed_at_ns
            or classify_scene(ocr, width, height).scene != "shop"):
        return None
    started = ocr.get("processing_started_at_ns")
    if (type(started) is not int or not observed_at_ns <= started <= available_at_ns
            or ocr.get("recognition_path") == "exact_image_cache"
            or "cache_source" in ocr):
        return None
    # A normal shop page may remain readable beneath a replacement overlay.
    lines = [_compact(row) for row in rows_in_region(ocr, (300, 220, 1600, 950))]
    if any(row in {"combine", "replace", "replaceweapon", "chooseaweapontoreplace",
                   "weaponlimitreached", "recycleweapon"} for row in lines):
        return None
    currency = _number(ocr, (805, 25, 1020, 110))
    currency_evidence = None
    if currency_observation is not None:
        from .shop_currency_ocr import bound_currency
        fallback = bound_currency(currency_observation, pixels=pixels, frame_id=frame_id,
                                  game_build_id=game_build_id, observed_at_ns=observed_at_ns,
                                  available_at_ns=available_at_ns)
        if fallback is not None and currency is not None and currency != fallback:
            return None
        # Unreadable fallback remains unknown. It cannot authorize spending,
        # but it need not hide a separately recognized shop's departure action.
        currency = fallback
        currency_evidence = (dict(currency_observation) if isinstance(currency_observation, dict)
                             else asdict(currency_observation))
    header = _compact("".join(rows_in_region(ocr, (0, 0, 400, 120))))
    wave = re.search(r"wave([0-9]{1,2})", header)
    if wave is None or not 1 <= int(wave[1]) <= 99:
        return None
    reroll_text = tuple(rows_in_region(ocr, (1162, 25, 1398, 107)))
    reroll_cost = (_number(ocr, (1345, 35, 1398, 95))
                   if any("reroll" in _compact(row) for row in reroll_text) else None)
    weapon_header = _compact("".join(rows_in_region(ocr, (1100, 765, 1460, 830))))
    inventory = re.fullmatch(r"weapons\(([0-9]{1,2})/([0-9]{1,2})\)", weapon_header)
    count = capacity = None
    if inventory and 0 <= int(inventory[1]) <= int(inventory[2]) <= 99:
        count, capacity = int(inventory[1]), int(inventory[2])
    offers = []
    for slot, left in enumerate(CARD_LEFTS):
        name = " ".join(rows_in_region(ocr, (left + 118, 145, left + 345, 191)))
        category = " ".join(rows_in_region(ocr, (left + 118, 191, left + 345, 230)))
        body = tuple(rows_in_region(ocr, (left + 10, 255, left + 339, 535)))
        compact_body, compact_category = _compact(" ".join(body)), _compact(category)
        if (compact_category in {"item", "ltem"} or compact_category.startswith("limited(")):
            kind = "item"
        elif (any(token in compact_body for token in ("cooldown", "c001down", "coold0wn"))
              and "damage" in compact_body and "range" in compact_body):
            kind = "weapon"
        else:
            kind = "unknown"
        semantic_id = (f"{kind}:" + _digest(_compact(name))) if re.search(r"[A-Za-z]", name) else None
        offers.append(ShopOffer(slot, name, category, kind, body, _price(ocr, pixels, slot),
                                semantic_id, _presence(pixels, left),
                                _region_digest(pixels, (left + 4, 145, left + 345, 535))))
    return ShopObservation(frame_id, hashlib.sha256(pixels).hexdigest(), game_build_id,
                           observed_at_ns, available_at_ns, started, currency, int(wave[1]),
                           count, capacity, tuple(offers),
                           tuple(rows_in_region(ocr, (1540, 150, 1880, 780))),
                           _region_digest(pixels, (28, 850, 1090, 975)),
                           _region_digest(pixels, (1132, 848, 1445, 975)), currency_evidence,
                           reroll_cost, reroll_text)


def stable_shop(first: ShopObservation | None, second: ShopObservation | None, *,
                now_ns: int | None = None,
                max_age_ns: int = MAX_OBSERVATION_AGE_NS) -> VerifiedShop | None:
    """No wait or sleep. Caller supplies two independent, ordered observations."""
    now_ns = time.perf_counter_ns() if now_ns is None else now_ns
    if first is None or second is None:
        return None
    if (first.frame_id == second.frame_id or first.ocr_started_at_ns == second.ocr_started_at_ns
            or not first.observed_at_ns < second.observed_at_ns <= second.available_at_ns <= now_ns
            or first.available_at_ns > second.observed_at_ns
            or now_ns - second.observed_at_ns > max_age_ns
            or second.observed_at_ns - first.observed_at_ns > 2_000_000_000):
        return None
    def key(observation):
        return (observation.game_build_id, observation.currency, observation.wave,
                observation.weapon_count, observation.weapon_capacity,
                observation.reroll_cost,
                tuple((o.slot, _compact(o.name), _compact(o.category), o.kind,
                       tuple(map(_compact, o.text)), o.cost, o.present) for o in observation.offers))
    if key(first) != key(second):
        return None
    return VerifiedShop(first, second,
                        first.item_inventory_sha256 == second.item_inventory_sha256,
                        first.weapon_inventory_sha256 == second.weapon_inventory_sha256)


def purchase_candidates(shop: VerifiedShop, *, now_ns: int | None = None) -> tuple[ShopCandidate, ...]:
    """Expose masks, never silently reinterpret unreadable as free or legal."""
    now_ns = time.perf_counter_ns() if now_ns is None else now_ns
    current = shop.current
    stale = not 0 <= now_ns - current.observed_at_ns <= MAX_OBSERVATION_AGE_NS
    candidates = []
    for offer in current.offers:
        reason = "affordable_verified_offer"
        if stale:
            reason = "stale_observation"
        elif current.currency is None:
            reason = "currency_unreadable"
        elif offer.present is not True or offer.semantic_id is None:
            reason = "offer_not_readable"
        elif offer.kind == "unknown":
            reason = "unsupported_offer_kind"
        elif offer.cost is None:
            reason = "price_unreadable"
        elif offer.cost > current.currency:
            reason = "insufficient_currency"
        elif offer.kind == "weapon" and (current.weapon_count is None or current.weapon_capacity is None):
            reason = "weapon_capacity_unreadable"
        elif offer.kind == "weapon" and current.weapon_count >= current.weapon_capacity:
            reason = "weapon_combine_or_replace_unsupported"
        features = (1.0, current.wave / 100, min(current.currency or 0, 5000) / 5000,
                    float(offer.kind == "weapon"), float(offer.kind == "item"),
                    min((offer.cost or 0) / max(current.currency or 0, 1), 1.0),
                    (current.weapon_count or 0) / max(current.weapon_capacity or 1, 1), 0.0)
        candidates.append(ShopCandidate(f"buy:{offer.slot}", f"buy_{offer.slot}",
                                        offer.semantic_id or f"unreadable:{offer.slot}", offer.kind,
                                        offer.slot, offer.cost, (offer.name, offer.category, *offer.text),
                                        reason == "affordable_verified_offer", reason, features))
    candidates.append(ShopCandidate("skip", "depart", "shop:skip", "skip", None, 0,
                                    ("Continue to next wave",), not stale,
                                    "stale_observation" if stale else "leave_shop",
                                    (1.0, current.wave / 100, min(current.currency or 0, 5000) / 5000,
                                     0.0, 0.0, 0.0, 0.0, 1.0)))
    reroll_reason = ("stale_observation" if stale else
                     "currency_unreadable" if current.currency is None else
                     "reroll_price_unreadable" if current.reroll_cost is None else
                     "insufficient_currency" if current.reroll_cost > current.currency else
                     "affordable_verified_reroll")
    # Append so the existing four purchases and departure keep their indices.
    candidates.append(ShopCandidate("reroll", "refresh", "shop:reroll", "reroll", None,
                                    current.reroll_cost, current.reroll_text,
                                    reroll_reason == "affordable_verified_reroll", reroll_reason,
                                    (1.0, current.wave / 100, min(current.currency or 0, 5000) / 5000,
                                     0.0, 0.0, 0.0, 0.0, 0.0)))
    return tuple(candidates)


def verify_purchase(before: VerifiedShop, after: VerifiedShop | None, slot: int) -> PurchaseVerification:
    """Verify application only. Unknown state is never a negative RL reward."""
    action_id = f"buy:{slot}"
    ids = before.evidence_ids + (() if after is None else after.evidence_ids)
    def result(reason, **kwargs):
        return PurchaseVerification(reason == "purchase_observed", reason, action_id, ids, **kwargs)
    if type(slot) is not int or slot not in range(4):
        return result("invalid_slot")
    if after is None:
        return result("post_purchase_unverified")
    if (after.first.observed_at_ns <= before.current.available_at_ns
            or len(set(ids)) != 4 or before.current.wave != after.current.wave
            or before.current.game_build_id != after.current.game_build_id):
        return result("mismatched_purchase_sequence")
    candidate = purchase_candidates(before, now_ns=before.current.available_at_ns)[slot]
    if not candidate.legal:
        return result("purchase_was_not_legal:" + candidate.reason)
    if after.current.currency is None:
        return result("post_purchase_currency_unreadable")
    old, new = before.current.offers[slot], after.current.offers[slot]
    spent = before.current.currency - after.current.currency
    if spent != old.cost:
        return result("currency_delta_mismatch", spent=spent)
    removed = (new.present is False and not new.name and new.cost is None
               and old.pixels_sha256 != new.pixels_sha256)
    if old.kind == "weapon":
        changed = (before.weapon_inventory_stable and after.weapon_inventory_stable
                   and before.current.weapon_capacity == after.current.weapon_capacity
                   and after.current.weapon_count == before.current.weapon_count + 1
                   and before.current.weapon_inventory_sha256 != after.current.weapon_inventory_sha256)
    else:
        changed = (before.item_inventory_stable and after.item_inventory_stable
                   and before.current.item_inventory_sha256 != after.current.item_inventory_sha256)
    if not (removed or changed):
        return result("purchase_application_unconfirmed", spent=spent)
    # Other offers must stay put. A reroll cannot be credited as a purchase.
    for index in range(4):
        if index != slot and before.current.offers[index].semantic_id != after.current.offers[index].semantic_id:
            return result("unrelated_offers_changed", spent=spent)
    return result("purchase_observed", spent=spent, offer_removed=removed, inventory_changed=changed)


def verify_reroll(before: VerifiedShop, after: VerifiedShop | None) -> RerollVerification:
    """Check this sampled reroll only; new waves are unrelated offer sets."""
    ids = before.evidence_ids + (() if after is None else after.evidence_ids)
    def result(reason, **kwargs):
        return RerollVerification(reason == "reroll_observed", reason, "reroll", ids, **kwargs)
    if after is None:
        return result("post_reroll_unverified")
    if (after.first.observed_at_ns <= before.current.available_at_ns
            or len(set(ids)) != 4 or before.current.wave != after.current.wave
            or before.current.game_build_id != after.current.game_build_id):
        return result("mismatched_reroll_sequence")
    candidate = next(c for c in purchase_candidates(before, now_ns=before.current.available_at_ns)
                     if c.action_id == "reroll")
    if not candidate.legal:
        return result("reroll_was_not_legal:" + candidate.reason)
    if after.current.currency is None:
        return result("post_reroll_currency_unreadable")
    spent = before.current.currency - after.current.currency
    if spent != candidate.cost:
        return result("reroll_currency_delta_mismatch", spent=spent)
    unchanged_inventory = (before.item_inventory_stable and before.weapon_inventory_stable
        and after.item_inventory_stable and after.weapon_inventory_stable
        and before.current.item_inventory_sha256 == after.current.item_inventory_sha256
        and before.current.weapon_inventory_sha256 == after.current.weapon_inventory_sha256
        and (before.current.weapon_count, before.current.weapon_capacity)
            == (after.current.weapon_count, after.current.weapon_capacity))
    if not unchanged_inventory:
        return result("reroll_inventory_changed_or_unverified", spent=spent)
    # Locked offers may stay; identical RNG outcomes remain unverified.
    changed = any(old.semantic_id is not None and new.semantic_id is not None
                  and old.semantic_id != new.semantic_id and new.present is True
                  and old.pixels_sha256 != new.pixels_sha256
                  for old, new in zip(before.current.offers, after.current.offers))
    if not changed:
        return result("reroll_offers_unchanged_or_unreadable", spent=spent)
    return result("reroll_observed", spent=spent, offers_changed=True)
