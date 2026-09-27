"""Brotato preparation checks: discovery is not gameplay readiness."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from playmodel.cli import main
from playmodel.games.brotato.contracts import Proposal, Screen, ScreenSnapshot, validate_proposal
from playmodel.games.brotato.installation import inspect_installation, parse_vdf


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def install(self, library, folder="Brotato", appid="1942280", pack=True):
        steamapps = library / "steamapps"
        steamapps.mkdir(parents=True, exist_ok=True)
        (steamapps / "appmanifest_1942280.acf").write_text(
            f'"AppState" {{ "appid" "{appid}" "installdir" "{folder}" '
            '"buildid" "123" "StateFlags" "4" "LastOwner" "PRIVATE" }', encoding="utf-8")
        if folder == "Brotato":
            game = steamapps / "common/Brotato"
            game.mkdir(parents=True, exist_ok=True)
            (game / "Brotato.exe").write_bytes(b"fixture")
            if pack:
                (game / "Brotato.pck").write_bytes(b"fixture")

    def test_quoted_vdf_with_comments_and_escaped_paths(self):
        parsed = parse_vdf('\ufeff// header\n"root" { "path" "D:\\\\steam" "title" "a \\"quote\\"" }')
        self.assertEqual(parsed["root"], {"path": "D:\\steam", "title": 'a "quote"'})

    def test_ambiguous_or_broken_vdf_rejected(self):
        for text in ('"a" "1" "a" "2"', '"a" {', '"a" }', '}', '"a" "b" junk', '"a"'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_vdf(text)

    def test_files_found_does_not_mean_autoplay_ready_or_expose_owner(self):
        self.install(self.root)
        report = inspect_installation(self.root)
        self.assertEqual(report["status"], "files_present")
        self.assertEqual(report["installations"][0]["steam_build_id"], "123")
        self.assertFalse(report["runtime_ready"])
        self.assertFalse(report["capabilities"]["game_input"])
        self.assertNotIn("PRIVATE", json.dumps(report))
        self.assertEqual((self.root / "steamapps/common/Brotato/Brotato.exe").read_bytes(), b"fixture")

    def test_secondary_library_found_with_malformed_primary_manifest(self):
        primary, secondary = self.root / "primary", self.root / "secondary"
        self.install(primary, appid="wrong")
        self.install(secondary)
        value = str(secondary).replace("\\", "\\\\")
        (primary / "steamapps/libraryfolders.vdf").write_text(
            f'"libraryfolders" {{ "1" {{ "path" "{value}" }} }}', encoding="utf-8")
        report = inspect_installation(primary)
        self.assertEqual(report["status"], "files_present")
        self.assertEqual(len(report["installations"]), 1)
        self.assertTrue(report["warnings"])

    def test_missing_pack_is_incomplete(self):
        self.install(self.root, pack=False)
        report = inspect_installation(self.root)
        self.assertEqual(report["status"], "incomplete_files")
        self.assertFalse(report["installations"][0]["resource_pack_found"])

    def test_manifest_cannot_point_outside_common(self):
        for folder in ("..", "../elsewhere", "C:/elsewhere"):
            with self.subTest(folder=folder):
                self.install(self.root, folder=folder)
                report = inspect_installation(self.root)
                self.assertEqual(report["installations"], [])
                self.assertTrue(report["warnings"])

    def test_cli_not_found_and_files_present_exit_codes(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["brotato-inspect", "--steam-root", str(self.root)]), 2)
        self.assertFalse(json.loads(stream.getvalue())["runtime_ready"])
        self.install(self.root)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["brotato-inspect", "--steam-root", str(self.root)]), 0)
        self.assertFalse(json.loads(stream.getvalue())["runtime_ready"])


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = ScreenSnapshot(
            Screen.SHOP, 7, 2, 100, 100, True, True, frozenset({"buy", "reroll_shop", "next_wave"}),
            {"buy": frozenset({"weapon-1"}), "reroll_shop": frozenset({"reroll-button"})},
            20, {"weapon-1": 20, "reroll-button": 0})
        self.proposal = Proposal("buy", 7, 2, 201, "weapon-1")

    def check(self, snapshot=None, proposal=None, now=150):
        return validate_proposal(snapshot or self.snapshot, proposal or self.proposal, now, 100)

    def test_affordable_observed_purchase_only_validates_not_sends(self):
        self.assertEqual(self.check().reason, "contract_valid_not_sent")
        self.assertTrue(self.check().allowed)

    def test_unknown_and_insufficient_payment_rejected(self):
        for snapshot in (replace(self.snapshot, materials=None), replace(self.snapshot, materials=19),
                         replace(self.snapshot, costs={}), replace(self.snapshot, costs={"weapon-1": True})):
            with self.subTest(snapshot=snapshot):
                self.assertEqual(self.check(snapshot).reason, "unknown_or_unaffordable_cost")
        self.assertTrue(self.check(proposal=replace(self.proposal, action="reroll_shop", target="reroll-button")).allowed)

    def test_stale_observation_and_authority_never_reused(self):
        cases = [(replace(self.proposal, control_epoch=1), "authority_changed"),
                 (replace(self.proposal, observation_sequence=6), "observation_changed"),
                 (replace(self.proposal, expires_at_ns=150), "proposal_expired")]
        for proposal, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(self.check(proposal=proposal).reason, reason)
        self.assertEqual(self.check(now=99).reason, "stale_or_future_observation")
        self.assertEqual(self.check(now=201).reason, "stale_or_future_observation")

    def test_no_input_on_uncertain_or_unfocused_screen(self):
        for changes in ({"foreground": False}, {"verified": False}, {"screen": Screen.UNKNOWN}):
            self.assertFalse(self.check(replace(self.snapshot, **changes)).allowed)

    def test_late_ocr_cannot_make_an_old_frame_fresh(self):
        self.assertEqual(self.check(replace(self.snapshot, observed_at_ns=1, available_at_ns=149)).reason,
                         "stale_or_future_observation")
        self.assertFalse(self.check(replace(self.snapshot, observed_at_ns=101, available_at_ns=100)).allowed)

    def test_malformed_masks_cannot_match_substrings(self):
        malformed = replace(self.snapshot, targets={"buy": "weapon-12"})
        self.assertEqual(self.check(malformed).reason, "invalid_state_contract")
        self.assertFalse(self.check(replace(self.snapshot, legal_actions="buy")).allowed)
        self.assertFalse(self.check(replace(self.snapshot, targets={"buy": frozenset({1})})).allowed)

    def test_observed_action_and_target_masks_both_required(self):
        self.assertEqual(self.check(replace(self.snapshot, legal_actions=frozenset())).reason, "action_not_available")
        self.assertEqual(self.check(proposal=replace(self.proposal, target="hidden-weapon")).reason, "target_not_available")
        self.assertEqual(self.check(proposal=replace(self.proposal, action="select_character")).reason, "action_not_available")

    def test_character_selection_and_result_retry_supported(self):
        character = replace(self.snapshot, screen=Screen.CHARACTER,
                            legal_actions=frozenset({"select_character"}),
                            targets={"select_character": frozenset({"unlocked-character"})})
        select = replace(self.proposal, action="select_character", target="unlocked-character")
        self.assertTrue(self.check(character, select).allowed)
        self.assertFalse(self.check(character, replace(select, target="locked-character")).allowed)
        result = replace(self.snapshot, screen=Screen.RESULT, legal_actions=frozenset({"retry_run"}))
        retry = replace(self.proposal, action="retry_run", target=None)
        self.assertTrue(self.check(result, retry).allowed)
        self.assertFalse(self.check(self.snapshot, retry).allowed)

    def test_combat_vector_must_be_finite_and_bounded(self):
        combat = replace(self.snapshot, screen=Screen.COMBAT, legal_actions=frozenset({"move"}))
        move = replace(self.proposal, action="move", target=None, movement=(1, -1))
        self.assertTrue(self.check(combat, move).allowed)
        for vector in ((float("nan"), 0), (float("inf"), 0), (2, 0), (True, 0), None, (0,)):
            self.assertEqual(self.check(combat, replace(move, movement=vector)).reason, "invalid_vector")

    def test_stop_is_internal_only_and_does_not_require_game_focus(self):
        stop = replace(self.proposal, action="stop", target=None)
        result = self.check(replace(self.snapshot, foreground=False), stop)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "internal_stop_only")


if __name__ == "__main__":
    unittest.main()
