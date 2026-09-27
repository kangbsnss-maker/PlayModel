"""Evidence-bound, last-observed build features. No input, torch or rewards.

The 48 columns extend the old context16 without changing its meaning. Numeric
values are displayed primary stats, not inferred item effects or current HP.
Two OCR readings agreeing is provenance, not a claim of perfect recognition.
Weapon names come only from verified selection/application records; incomplete
inventories remain explicitly incomplete. No game-mechanics lookup is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import unicodedata

from .ocr import rows_in_region
from .shop_learning import VerifiedShop, verify_purchase

SCHEMA = "brotato-observed-build-v2"
EXTRA_DIM, CONTEXT_DIM = 48, 64
GAME_BUILD = "23429717"
PARSER_VERSION = "literal-primary-stats-en-v1"
CLOCK = "perf_counter_ns_same_host"
# Fixed feature normalization horizons, NOT game limits or freshness claims.
VALUE_SCALE, AGE_SCALE_NS = 1000.0, 120_000_000_000
STATS = ("max_hp", "hp_regeneration", "life_steal", "damage", "melee_damage",
         "ranged_damage", "elemental_damage", "attack_speed", "crit_chance",
         "engineering", "range", "armor", "dodge", "speed", "luck", "harvesting")
STAT_LABELS = ("Max HP", "HP Regeneration", "% Life Steal", "% Damage", "Melee Damage",
               "Ranged Damage", "Elemental Damage", "% Attack Speed", "% Crit Chance",
               "Engineering", "Range", "Armor", "% Dodge", "% Speed", "Luck", "Harvesting")
# Separately inspected 1920x1080 English shop/level-up captures, build 23429717.
STATS_ROI = {"shop": (1540, 212, 1880, 767), "level_up": (1570, 387, 1900, 950)}
_ALIASES = {"maxhp": "max_hp", "hpregeneration": "hp_regeneration", "%lifesteal": "life_steal",
            "%damage": "damage", "meleedamage": "melee_damage", "rangeddamage": "ranged_damage",
            "elementaldamage": "elemental_damage", "eiementaidamage": "elemental_damage",
            "elementaidamage": "elemental_damage", "%attackspeed": "attack_speed",
            "%critchance": "crit_chance", "engineering": "engineering", "range": "range",
            "armor": "armor", "%dodge": "dodge", "%d0dge": "dodge", "%speed": "speed",
            "luck": "luck", "harvesting": "harvesting"}


def _compact(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def parse_stats(raw_text) -> dict[str, float | None]:
    """Only whole labelled rows with literal signed decimal numbers survive.

    '%' belongs to the displayed label. Missing numbers, duplicate labels and
    ambiguous punctuation are unknown; numeric O/I substitutions are forbidden.
    Label spelling aliases above are explicit observed OCR errors only.
    """
    values = {key: None for key in STATS}
    seen = set()
    for row in raw_text:
        if not isinstance(row, str):
            continue
        text = _compact(row)
        key = next((key for label, key in sorted(_ALIASES.items(), key=lambda p: -len(p[0]))
                    if text.startswith(label)), None)
        if key is None:
            continue
        label = next(label for label, candidate in sorted(_ALIASES.items(), key=lambda p: -len(p[0]))
                     if candidate == key and text.startswith(label))
        literal = text[len(label):]
        if key in seen:
            values[key] = None
            continue
        seen.add(key)
        # Bound numeric length; the bound rejects corrupted OCR, not game values.
        if not re.fullmatch(r"[+-]?(?:0|[1-9][0-9]{0,6})(?:\.[0-9]{1,3})?", literal):
            continue
        value = float(literal)
        if key == "max_hp" and value <= 0:
            continue
        values[key] = value
    return values


def _file_hash(path):
    try:
        return _sha(Path(path).read_bytes())
    except (OSError, TypeError):
        return None


def make_stats_observation(ocr: dict, *, frame_ref: str, pixels_sha256: str,
                           observed_at_ns: int, available_at_ns: int, scene: str,
                           game_build_id: str = GAME_BUILD, language: str = "en") -> dict:
    """Bind current full-frame OCR to the calibrated stats panel, no re-OCR."""
    roi = STATS_ROI.get(scene)
    return {"frame_ref": str(frame_ref), "source_pixels_sha256": pixels_sha256,
            "source_png_sha256": _file_hash(frame_ref), "observed_at_ns": observed_at_ns,
            "available_at_ns": available_at_ns, "ocr_started_at_ns": ocr.get("processing_started_at_ns"),
            "recognition_path": ocr.get("recognition_path"), "cache_source": ocr.get("cache_source"),
            "raw_text": tuple(rows_in_region(ocr, roi)) if roi else (), "scene": scene,
            "game_build_id": game_build_id, "language": language, "ocr_language": ocr.get("language"),
            "confidence": None, "clock_domain": CLOCK, "roi": roi, "evidence_files": (),
            "extractor_version": PARSER_VERSION, "verified": False}


def _valid_observation(record, build, language):
    try:
        return (record["scene"] in STATS_ROI and record["game_build_id"] == build == GAME_BUILD
                and record["language"] == language == "en" and record.get("clock_domain", CLOCK) == CLOCK
                and isinstance(record["frame_ref"], str) and bool(record["frame_ref"].strip())
                and _digest(record["source_pixels_sha256"]) and _digest(record["source_png_sha256"])
                and all(type(record[k]) is int for k in ("observed_at_ns", "available_at_ns", "ocr_started_at_ns"))
                and 0 < record["observed_at_ns"] <= record["ocr_started_at_ns"] <= record["available_at_ns"]
                and record.get("recognition_path") == "persistent_local_ocr" and not record.get("cache_source")
                and isinstance(record["raw_text"], (tuple, list))
                and all(isinstance(row, str) for row in record["raw_text"]))
    except (TypeError, KeyError):
        return False


def _lexical(names):
    values = [0.0] * 8
    # Sort for inventory permutation invariance; duplicates preserve quantity.
    text = "|".join(sorted(_compact(name) for name in names))[:4096]
    if not text:
        return values
    grams = [text[i:i + 3] for i in range(max(1, len(text) - 2))]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        values[digest[0] % 8] += 1.0 if digest[1] & 1 else -1.0
    return [max(-1.0, min(1.0, value / math.sqrt(len(grams)))) for value in values]


class BoundedBuildState:
    """One run's last-observed state. Discard on episode change or takeover.

    `features(frame_time, available_at_ns=decision_available_time)` enforces both
    clocks. Age keeps increasing during combat; it never makes a menu snapshot
    into current HP or an observation of temporary effects.
    """
    def __init__(self, episode_id: str, game_build_id: str = GAME_BUILD, language: str = "en"):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode identity required")
        self.episode_id, self.game_build_id, self.language = episode_id, game_build_id, language
        self._stats = {key: None for key in STATS}
        self._stats_sources = []
        self._stats_reason = "unobserved"
        self._weapons = []
        self._weapon_count = None
        self._weapon_sources = []
        self._inventory_hash = None
        self._weapon_reason = "unobserved"
        self._history = []

    def invalidate_stats(self, reason: str):
        self._stats = {key: None for key in STATS}
        self._stats_reason = str(reason)

    def observe_stats_pair(self, first: dict, second: dict) -> bool:
        # A less detailed adapter may revisit the already-bound menu pair after
        # ROI OCR. Re-observation of the same frame is neither a new vote nor a
        # reason to erase its independently obtained numeric reading.
        if (self._stats_sources and second.get("frame_ref") == self._stats_sources[-1]["frame_ref"]
                and type(second.get("available_at_ns")) is int
                and second["available_at_ns"] <= self._stats_sources[-1]["available_at_ns"]):
            return any(value is not None for value in self._stats.values())
        self.invalidate_stats("unverified_stats_pair")
        if not all(_valid_observation(row, self.game_build_id, self.language) for row in (first, second)):
            return False
        if (first["frame_ref"] == second["frame_ref"] or first["scene"] != second["scene"]
                or first["ocr_started_at_ns"] == second["ocr_started_at_ns"]
                or not first["observed_at_ns"] < first["available_at_ns"] < second["observed_at_ns"]
                or second["observed_at_ns"] - first["observed_at_ns"] > 2_000_000_000
                or (self._stats_sources and second["observed_at_ns"] < self._stats_sources[-1]["observed_at_ns"])):
            return False
        left, right = parse_stats(first["raw_text"]), parse_stats(second["raw_text"])
        self._stats = {key: right[key] if left[key] is not None and left[key] == right[key] else None
                       for key in STATS}
        self._stats_sources = deepcopy([first, second])
        self._stats_reason = "two_independent_ocr_agreement"
        self._history.append({"kind": "stats_pair", "sources": deepcopy(self._stats_sources),
                              "values": dict(self._stats), "confidence": None})
        return any(value is not None for value in self._stats.values())

    def _shop_record(self, observation):
        return {"frame_ref": observation.frame_id, "source_pixels_sha256": observation.frame_sha256,
                "source_png_sha256": _file_hash(observation.frame_id), "observed_at_ns": observation.observed_at_ns,
                "available_at_ns": observation.available_at_ns, "ocr_started_at_ns": observation.ocr_started_at_ns,
                "raw_text": tuple(observation.build_text), "scene": "shop", "game_build_id": observation.game_build_id,
                "language": self.language, "recognition_path": "persistent_local_ocr", "clock_domain": CLOCK,
                "roi": STATS_ROI["shop"], "confidence": None, "extractor_version": PARSER_VERSION,
                "evidence_files": (), "verified": False}

    def observe_shop_pair(self, shop: VerifiedShop) -> bool:
        if not isinstance(shop, VerifiedShop):
            self.invalidate_stats("invalid_shop")
            return False
        first, current = shop.first, shop.current
        records = [self._shop_record(first), self._shop_record(current)]
        accepted = self.observe_stats_pair(*records)
        if (not all(_valid_observation(r, self.game_build_id, self.language) for r in records)
                or first.frame_id == current.frame_id or first.available_at_ns >= current.observed_at_ns
                or not shop.weapon_inventory_stable or current.weapon_count is None
                or current.weapon_count != first.weapon_count
                or type(current.weapon_count) is not int or not 0 <= current.weapon_count <= 99):
            self._weapons, self._weapon_count, self._inventory_hash = [], None, None
            self._weapon_sources = []
            self._weapon_reason = "inventory_unverified"
            return accepted
        if self._inventory_hash is not None and self._inventory_hash != current.weapon_inventory_sha256:
            self._weapons = []
            self._weapon_reason = "unexplained_inventory_change"
        if len(self._weapons) > current.weapon_count:
            self._weapons = []
            self._weapon_reason = "inventory_count_conflict"
        self._weapon_count = current.weapon_count
        self._inventory_hash = current.weapon_inventory_sha256
        self._weapon_sources = records
        return accepted

    def seed_weapon(self, name: str, evidence: dict) -> bool:
        """Verified setup selection only. Inventory total stays unknown until seen."""
        if self._weapons or self._weapon_count is not None:
            return False
        try:
            available = max(evidence["available_at_ns"], evidence.get("verified_at_ns", 0))
            valid = (isinstance(name, str) and bool(name.strip()) and evidence["verified"] is True
                     and evidence["independent_of_policy"] is True and _digest(evidence["frame_sha256"])
                     and isinstance(evidence["frame_ref"], str) and bool(evidence["frame_ref"].strip())
                     and type(evidence["observed_at_ns"]) is int and type(available) is int
                     and 0 < evidence["observed_at_ns"] <= available
                     and evidence.get("episode_id", self.episode_id) == self.episode_id)
        except (TypeError, KeyError):
            valid = False
        if not valid:
            return False
        source = {**deepcopy(evidence), "source_png_sha256": evidence["frame_sha256"],
                  "available_at_ns": available}
        self._weapons = [{"name": name, "raw_text": (name,), "source": source,
                          "origin": "verified_starting_selection", "tier": None, "effects_known": False}]
        self._weapon_sources = [source]
        self._weapon_reason = "verified_name_inventory_total_unknown"
        return True

    def apply_verified_purchase(self, before: VerifiedShop, after: VerifiedShop, slot: int,
                                application=None) -> bool:
        result = verify_purchase(before, after, slot)
        if not result.verified or (application is not None and not application.accepted):
            return False
        offer = before.current.offers[slot]
        if offer.kind == "weapon":
            if (self._inventory_hash is not None and self._inventory_hash != before.current.weapon_inventory_sha256
                    or self._weapon_count is not None and self._weapon_count != before.current.weapon_count
                    or len(self._weapons) > (before.current.weapon_count or 0)):
                self._weapons = []
            self._weapons.append({"name": offer.name, "raw_text": (offer.name, offer.category, *offer.text),
                                  "origin": "verified_purchase", "tier": None, "effects_known": False,
                                  "source": self._shop_record(before.current),
                                  "application_sources": [self._shop_record(after.first), self._shop_record(after.current)]})
            self._inventory_hash = after.current.weapon_inventory_sha256
            self._weapon_reason = "verified_purchase_names_partial_if_count_exceeds_names"
        self.invalidate_stats("purchase_requires_post_state")
        self.observe_shop_pair(after)
        return True

    @staticmethod
    def _usable(sources, observed, available):
        return bool(sources) and all(source["observed_at_ns"] <= observed and source["available_at_ns"] <= available
                                     for source in sources)

    def features(self, observed_at_ns: int, *, available_at_ns: int | None = None) -> tuple[float, ...]:
        if type(observed_at_ns) is not int or observed_at_ns < 0:
            raise ValueError("nonnegative observation time required")
        available = observed_at_ns if available_at_ns is None else available_at_ns
        if type(available) is not int or available < observed_at_ns:
            raise ValueError("availability must follow observation")
        stats_ok = self._usable(self._stats_sources, observed_at_ns, available)
        known = [float(stats_ok and self._stats[key] is not None) for key in STATS]
        values = [math.copysign(min(math.log1p(abs(self._stats[key])) / math.log1p(VALUE_SCALE), 1.0), self._stats[key])
                  if mask else 0.0 for key, mask in zip(STATS, known)]
        weapons_ok = self._usable(self._weapon_sources, observed_at_ns, available)
        names = [row["name"] for row in self._weapons if self._usable([row["source"], *row.get("application_sources", [])],
                                                                    observed_at_ns, available)] if weapons_ok else []
        count = self._weapon_count if weapons_ok else None
        stats_present, weapon_present = float(any(known)), float(bool(names))
        stats_age = min((observed_at_ns - self._stats_sources[-1]["observed_at_ns"]) / AGE_SCALE_NS, 1.0) if stats_present else 0.0
        weapon_age = min((observed_at_ns - self._weapon_sources[-1]["observed_at_ns"]) / AGE_SCALE_NS, 1.0) if weapons_ok else 0.0
        coverage = len(names) / count if count else (1.0 if count == 0 else 0.0)
        status = [stats_present, stats_age, weapon_present, weapon_age, coverage,
                  min(count / 99, 1.0) if count is not None else 0.0,
                  min(max(count - len(names), 0) / 99, 1.0) if count is not None else 0.0, 1.0]
        return tuple(values + known + _lexical(names) + status)

    def snapshot(self) -> dict:
        sources = [*self._stats_sources, *self._weapon_sources]
        for weapon in self._weapons:
            sources += [weapon["source"], *weapon.get("application_sources", [])]
        return deepcopy({"schema": SCHEMA, "episode_id": self.episode_id, "game_build_id": self.game_build_id,
                         "language": self.language, "parser_version": PARSER_VERSION, "stats": self._stats,
                         "stats_status": self._stats_reason, "weapons": self._weapons,
                         "weapon_count": self._weapon_count, "weapon_status": self._weapon_reason,
                         "sources": sources, "stats_sources": self._stats_sources,
                         "weapon_sources": self._weapon_sources, "confidence": None,
                         "effect_semantics_known": False, "current_hp_known": False,
                         "observation_semantics": "last_observed_display_not_current_combat_effects",
                         "clock_domain": CLOCK, "history": self._history})
