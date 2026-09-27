"""State-machine tests with synthetic preserved frames; no actual game input."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch is not installed")

import torch

from playmodel.games.brotato.capture import _png
from playmodel.games.brotato.menu import BUTTONS
from playmodel.games.brotato.neural_menu_controller import NeuralMenuController, NeuralMenuError
from playmodel.games.brotato.neural_choices import MAX_AGE_NS
from playmodel.learning import StateEvidence
from playmodel.learning.full_run import FullRunRecorder
from playmodel.learning.recurrent_ppo import RecurrentActorCritic
from test_neural_choices import BASE, upgrade_report, loot_report
from test_shop_learning import frame as shop_pixels, report as shop_report


def highlighted(pixels, rect):
    image = bytearray(pixels)
    left, top, right, bottom = rect
    for y in range(top, bottom):
        for x in range(left, right):
            offset = (y * 1920 + x) * 4
            blue, green, red = pixels[offset:offset + 3]
            # Real focus changes the button background, not the currency icon.
            if not (green > 55 and green > red * 1.35 and green > blue * 1.35):
                image[offset:offset + 4] = bytes((200, 200, 200, 255))
    return bytes(image)


class NeuralMenuControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(0)
        self.model = RecurrentActorCritic()
        self.recorder = FullRunRecorder(self.model, "test-run")
        self.now = BASE + 100_000_000
        self.controller = NeuralMenuController(self.recorder, output_directory=self.root / "macros",
                                                clock=lambda: self.now, seed=1)
        self.serial = 0

    def observation(self, sequence, *, scene="level_up", target=None, color=95, values=None,
                    currency=80, prices=(14, 18, 23, 28), absent=(), count=2,
                    reroll_cost=None, changed_offer=False, loot_title='Baby Gecko'):
        if scene == "level_up":
            ocr = upgrade_report(sequence, values=values)
            pixels = bytes((color, color, color, 255)) * 1920 * 1080
        elif scene == 'loot':
            ocr = loot_report(sequence, title=loot_title)
            pixels = bytes((color, color, color, 255)) * 1920 * 1080
        else:
            ocr = shop_report(sequence=sequence, currency=currency, prices=prices, absent=absent,
                              count=count, reroll_cost=reroll_cost)
            pixels = shop_pixels(absent=absent)
            if changed_offer:
                next(line['words'][0] for line in ocr['lines']
                     if line['words'][0]['text'] == 'Ghost Flint')['text'] = 'New Weapon'
                changed = bytearray(pixels)
                changed[(300 * 1920 + 50) * 4] ^= 1
                pixels = bytes(changed)
        if target is not None:
            rect = (1484, 981, 1884, 1048) if target == "depart" else BUTTONS[scene][target]
            pixels = highlighted(pixels, rect)
        self.serial += 1
        directory = self.root / f"frame-{sequence}-{self.serial}"
        directory.mkdir()
        png = _png(1920, 1080, pixels, compression_level=1)
        (directory / "frame.png").write_bytes(png)
        ocr["available_at_ns"] = BASE + sequence * 100_000_000 + 20_000_000
        shot = {"session_directory": str(directory), "capture_started_at_ns": BASE + sequence * 100_000_000,
                "available_at_ns": BASE + sequence * 100_000_000 + 5_000_000,
                "frame_sha256": hashlib.sha256(png).hexdigest()}
        self.now = BASE + sequence * 100_000_000 + 21_000_000
        return shot, ocr, pixels

    def send(self, args):
        pending = self.controller.pending_decision
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        self.now += 1_000_000
        self.controller.mark_sent(sent_at_ns=self.now, actual_target=pending.target)

    def test_loot_is_sampled_once_then_committed_only_after_two_accepted_post_frames(self):
        self.assertEqual(self.controller.handle(*self.observation(1, scene='loot')).status, 'wait')
        self.assertEqual(self.controller.samples, 0)
        first = self.controller.handle(*self.observation(2, scene='loot'))
        self.assertEqual(first.status, 'navigate')
        decision = self.controller.pending_decision
        self.assertIn(decision.target, ('take', 'recycle'))
        args = self.observation(3, scene='loot', target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        self.assertEqual(self.controller.samples, 1)
        self.assertEqual(len(self.recorder.records), 0)
        self.assertEqual(self.controller.handle(*self.observation(4, scene='loot', target=decision.target)).status, 'wait')
        same = self.controller.handle(*self.observation(5, scene='loot', target=decision.target, color=110))
        self.assertEqual(same.status, 'wait')
        self.assertEqual(same.reason, 'loot_application_unconfirmed')
        self.assertEqual(len(self.recorder.records), 0)
        self.assertEqual(self.controller.handle(*self.observation(6, color=110)).status, 'wait')
        accepted = self.controller.handle(*self.observation(7, color=110))
        self.assertEqual(accepted.status, 'accepted')
        self.assertEqual(accepted.reason, 'loot_selection_accepted')
        record = self.recorder.records[0]
        self.assertEqual(record['evidence']['action_origin'], 'policy')
        self.assertEqual(record['evidence']['actual_action'], decision.action_index)
        self.assertEqual(record['reward'], 0.0)
        self.assertFalse(record['evidence']['feature_effects_verified'])
        self.assertTrue(torch.equal(self.recorder.hidden, decision.tensors()['next_hidden']))
        self.assertEqual(self.controller.samples, 1)

    def test_loot_same_named_next_card_stays_unknown_and_never_resends(self):
        self.controller.max_result_observations = 2
        self.controller.handle(*self.observation(1, scene='loot'))
        self.controller.handle(*self.observation(2, scene='loot'))
        decision = self.controller.pending_decision
        args = self.observation(3, scene='loot', target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        self.controller.handle(*self.observation(4, scene='loot', color=110))
        self.controller.handle(*self.observation(5, scene='loot', color=110))
        with self.assertRaisesRegex(NeuralMenuError, 'remained unverified'):
            self.controller.handle(*self.observation(6, scene='loot', color=110))
        self.assertEqual(self.controller.samples, 1)
        self.assertEqual(len(self.recorder.records), 0)

    def test_loot_acceptance_invalidates_stats_without_claiming_effects(self):
        self.controller.handle(*self.observation(1, scene='loot'))
        self.controller.handle(*self.observation(2, scene='loot'))
        decision = self.controller.pending_decision
        args = self.observation(3, scene='loot', target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        # The source decision remains the real frozen behavior. The state spy
        # only checks invalidation after accepted application, not any label.
        state = Mock()
        self.recorder.build_state = state
        self.controller.handle(*self.observation(4, scene='loot', color=110, loot_title='Other item'))
        accepted = self.controller.handle(*self.observation(5, scene='loot', color=110, loot_title='Other item'))
        self.assertEqual(accepted.status, 'accepted')
        state.invalidate_stats.assert_called_once_with('loot_selected_reobservation_required')
        state.apply_verified_purchase.assert_not_called()

    def test_loot_card_change_before_enter_and_wrong_focus_fail_closed(self):
        self.controller.handle(*self.observation(1, scene='loot'))
        self.controller.handle(*self.observation(2, scene='loot'))
        decision = self.controller.pending_decision
        changed = self.observation(3, scene='loot', target=decision.target, loot_title='Other item')
        self.assertEqual(self.controller.handle(*changed).status, 'wait')
        with self.assertRaisesRegex(NeuralMenuError, 'Candidate changed'):
            self.controller.authorize_enter(decision.decision_id, decision.target, *changed)
        self.assertEqual(len(self.recorder.records), 0)

    def test_navigation_keeps_one_sample_and_does_not_commit_hidden_early(self):
        before = self.recorder.hidden.clone()
        first = self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        second = self.controller.handle(*self.observation(2, target=pending.target))
        third = self.controller.handle(*self.observation(3, target=pending.target))
        self.assertEqual((first.target, second.target, third.target), (pending.target,) * 3)
        self.assertEqual(self.controller.samples, 1)
        self.assertEqual(len(self.recorder.records), 0)
        self.assertTrue(torch.equal(before, self.recorder.hidden))

    def test_enter_reuses_only_the_exact_fresh_prepared_observation(self):
        from playmodel.games.brotato import neural_menu_controller as module
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        with patch.object(module, 'observe_menu', wraps=module.observe_menu) as observer:
            self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
            observer.assert_not_called()  # No duplicate RGB96/candidate preparation.
        self.assertEqual(self.controller.samples, 1)

    def test_enter_cache_rejects_same_frame_ocr_pixels_or_availability_changes(self):
        from copy import deepcopy
        for change in ('ocr', 'pixels', 'available'):
            with self.subTest(change=change):
                recorder = FullRunRecorder(self.model, 'cache-' + change)
                controller = NeuralMenuController(recorder, clock=lambda: self.now)
                controller.handle(*self.observation(1))
                pending = controller.pending_decision
                args = self.observation(2, target=pending.target)
                controller.handle(*args)
                shot, ocr, pixels = deepcopy(args)
                if change == 'ocr':
                    ocr['lines'][1]['words'][0]['text'] = '+999 Changed'
                elif change == 'pixels':
                    pixels = bytes((pixels[0] ^ 1,)) + pixels[1:]
                else:
                    ocr['available_at_ns'] += 1
                with self.assertRaisesRegex(NeuralMenuError, 'Prepared menu evidence changed'):
                    controller.authorize_enter(pending.decision_id, pending.target, shot, ocr, pixels)
                self.assertIsNone(controller._pending.authorized_frame)
                self.assertEqual(len(recorder.records), 0)

    def test_repeated_enter_expiry_is_bounded_even_when_each_handle_succeeds(self):
        from playmodel.games.brotato import neural_menu_controller as module
        self.controller.max_result_observations = 2
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        real_selection = module.selected_button
        def slow_selection(*args, **kwargs):
            selected = real_selection(*args, **kwargs)
            self.now += 800_000_000
            return selected
        with patch.object(module, 'selected_button', side_effect=slow_selection), \
             patch('playmodel.execution_log.event') as log:
            for sequence in (10, 20):
                args = self.observation(sequence, target=pending.target)
                self.assertEqual(self.controller.handle(*args).status, 'navigate')
                self.assertFalse(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
            args = self.observation(30, target=pending.target)
            self.assertEqual(self.controller.handle(*args).status, 'navigate')
            with self.assertRaisesRegex(NeuralMenuError, 'Fresh Enter authorization remained unavailable'):
                self.controller.authorize_enter(pending.decision_id, pending.target, *args)
        self.assertEqual(self.controller._authorization_expirations, 3)
        self.assertEqual(self.controller.samples, 1)
        self.assertFalse(any(p.name.startswith('sent') for p in (self.root / 'macros').rglob('*')))
        deferred = [call.kwargs for call in log.call_args_list if call.args[0] == 'neural_menu_deferred']
        self.assertEqual([row['authorization_expirations'] for row in deferred], [1, 2, 3])
        self.assertTrue(all(row['stage'] == 'authorize_verification' and row['observation_age_ns'] > MAX_AGE_NS
                            and row['transmitted'] is False and row['actual_target'] is None for row in deferred))

    def test_pending_decision_keeps_frozen_build_without_reprocessing_stats(self):
        self.assertTrue(self.controller.needs_build_observation)
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        self.assertFalse(self.controller.needs_build_observation)
        state = Mock()
        self.recorder.build_state = state
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        state.observe_stats_pair.assert_not_called()
        state.observe_shop_pair.assert_not_called()
        self.send(args)
        self.assertFalse(self.controller.needs_build_observation)
        self.controller.handle(*self.observation(3, color=105, values=('A', 'B', 'C', 'D')))
        self.controller.handle(*self.observation(4, color=105, values=('A', 'B', 'C', 'D')))
        self.assertTrue(self.controller.needs_build_observation)
        state.invalidate_stats.assert_called_once()

    def test_upgrade_appends_only_after_two_independent_post_send_observations(self):
        self.controller.handle(*self.observation(1))
        decision = self.controller.pending_decision
        args = self.observation(2, target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        self.assertEqual(len(self.recorder.records), 0)
        next_cards = ("+2 Speed", "+2 Luck", "+3 Damage", "+4 Harvesting")
        first = self.controller.handle(*self.observation(3, color=105, values=next_cards))
        self.assertEqual(first.status, "wait")
        self.assertEqual(len(self.recorder.records), 0)
        second = self.controller.handle(*self.observation(4, color=105, values=next_cards))
        self.assertEqual(second.status, "accepted")
        self.assertEqual(len(self.recorder.records), 1)
        self.assertTrue(torch.equal(self.recorder.hidden, decision.tensors()["next_hidden"]))
        self.assertIsNone(self.controller.pending_decision)
        self.assertEqual(self.recorder.records[0]["reward"], 0.0)
        folder = self.root / "macros" / decision.decision_id
        self.assertTrue((folder / "proposed/record.json").exists())
        self.assertTrue((folder / "sent.json").exists())
        self.assertTrue((folder / "accepted/record.json").exists())

    def test_changed_card_or_wrong_pixel_focus_never_authorizes_enter(self):
        self.controller.handle(*self.observation(1))
        decision = self.controller.pending_decision
        changed = self.observation(2, target=decision.target, values=("changed", "other", "third", "fourth"))
        self.assertEqual(self.controller.handle(*changed).status, "wait")
        self.assertEqual(self.controller.samples, 1)
        with self.assertRaisesRegex(NeuralMenuError, "Candidate changed"):
            self.controller.authorize_enter(decision.decision_id, decision.target, *changed)
        self.assertTrue(self.recorder.rejection_reasons)

    def test_caller_target_string_does_not_replace_actual_focus_check(self):
        self.controller.handle(*self.observation(1))
        decision = self.controller.pending_decision
        other = "choose_" + str((decision.action_index + 1) % 4)
        wrong_focus = self.observation(2, target=other)
        self.controller.handle(*wrong_focus)
        with self.assertRaisesRegex(NeuralMenuError, "pixel focus"):
            self.controller.authorize_enter(decision.decision_id, decision.target, *wrong_focus)

    def test_no_sample_after_enter_and_unknown_application_is_bounded(self):
        self.controller.max_result_observations = 2
        self.controller.handle(*self.observation(1))
        decision = self.controller.pending_decision
        args = self.observation(2, target=decision.target)
        self.controller.handle(*args)
        self.send(args)
        self.assertEqual(self.controller.handle(*self.observation(3)).status, "wait")
        last_wait = self.controller.handle(*self.observation(4))
        self.assertEqual(last_wait.status, "wait")
        with self.assertRaisesRegex(NeuralMenuError, "remained unverified"):
            self.controller.handle(*self.observation(5))
        self.assertEqual(self.controller.samples, 1)
        self.assertEqual(len(self.recorder.records), 0)
        failure = json.loads(next((self.root / "macros").glob("failure-*.json")).read_text())
        self.assertEqual(failure["pending_state"]["last_result_reason"], last_wait.reason)
        self.assertEqual(failure["pending_state"]["last_application_check"]["reason"], last_wait.reason)

    def test_shop_requires_stability_and_price_change_blocks_old_macro(self):
        self.assertEqual(self.controller.handle(*self.observation(1, scene="shop", currency=14)).status, "wait")
        self.assertEqual(self.controller.samples, 0)
        second = self.controller.handle(*self.observation(2, scene="shop", currency=14))
        self.assertEqual(second.status, "navigate")
        decision = self.controller.pending_decision
        # Currency changing during navigation invalidates buy and skip alike.
        args = self.observation(3, scene="shop", currency=13, target=decision.target)
        self.assertEqual(self.controller.handle(*args).status, "wait")
        args = self.observation(4, scene="shop", currency=13, target=decision.target)
        self.assertEqual(self.controller.handle(*args).status, "wait")
        with self.assertRaisesRegex(NeuralMenuError, "Candidate changed|independently revalidated shop legality"):
            self.controller.authorize_enter(decision.decision_id, decision.target, *args)
        self.assertEqual(self.controller.samples, 1)

    def test_skip_commits_only_when_independent_combat_entry_arrives(self):
        self.controller.handle(*self.observation(1, scene="shop", currency=0))
        self.controller.handle(*self.observation(2, scene="shop", currency=0))
        decision = self.controller.pending_decision
        self.assertEqual(decision.target, "depart")
        args = self.observation(3, scene="shop", currency=0, target="depart")
        self.controller.handle(*args)
        self.send(args)
        before = self.recorder.hidden.clone()
        self.assertEqual(self.controller.handle(*self.observation(4, scene="shop", currency=0)).status, "wait")
        self.assertTrue(torch.equal(self.recorder.hidden, before))
        shot, _, _ = self.observation(5, color=30)
        source = str(Path(shot["session_directory"]) / "frame.png")
        evidence = StateEvidence("combat", source, shot["frame_sha256"], shot["capture_started_at_ns"],
                                 shot["available_at_ns"], self.now, "local_detector", "combat-test-verifier", True, True)
        self.assertEqual(self.controller.combat_entry(evidence).status, "accepted")
        self.assertEqual(len(self.recorder.records), 1)
        self.assertTrue(torch.equal(self.recorder.hidden, decision.tensors()["next_hidden"]))

    def test_sampled_purchase_is_confirmed_before_append_and_memory_commit(self):
        first = self.observation(1, scene="shop", currency=14)
        second = self.observation(2, scene="shop", currency=14)
        for seed in range(10):
            self.controller = NeuralMenuController(self.recorder, clock=lambda: self.now, seed=seed)
            self.controller.handle(*first)
            self.controller.handle(*second)
            if self.controller.pending_decision.target == "buy_0":
                break
        decision = self.controller.pending_decision
        self.assertEqual(decision.target, "buy_0")
        args = self.observation(3, scene="shop", currency=14, target="buy_0")
        self.controller.handle(*args)
        self.send(args)
        first_after = self.controller.handle(*self.observation(4, scene="shop", currency=0, absent=(0,), count=3))
        self.assertEqual(first_after.status, "wait")
        self.assertEqual(len(self.recorder.records), 0)
        second_after = self.controller.handle(*self.observation(5, scene="shop", currency=0, absent=(0,), count=3))
        self.assertEqual(second_after.status, "accepted")
        self.assertEqual(len(self.recorder.records), 1)
        self.assertEqual(self.recorder.records[0]["reward"], 0.0)
        self.assertTrue(torch.equal(self.recorder.hidden, decision.tensors()["next_hidden"]))

    def test_unknown_currency_reaches_neural_departure_without_spending(self):
        self.assertEqual(self.controller.handle(*self.observation(1, scene='shop', currency=None)).status,
                         'wait')
        self.assertIsNone(self.controller.pending_decision)
        self.assertEqual(self.controller.handle(*self.observation(2, scene='shop', currency=None)).target,
                         'depart')
        decision = self.controller.pending_decision
        self.assertIsNone(decision.observation.currency)
        self.assertEqual(decision.context[12], 0.0)
        self.assertEqual(decision.observation.legal_mask, (False, False, False, False, True, False, False, False, False))
        args = self.observation(3, scene='shop', currency=None, target='depart')
        self.controller.handle(*args)
        self.send(args)
        self.assertEqual(len(self.recorder.records), 0)
        shot, _, _ = self.observation(4, color=30)
        evidence = StateEvidence('combat', str(Path(shot['session_directory']) / 'frame.png'),
                                 shot['frame_sha256'], shot['capture_started_at_ns'],
                                 shot['available_at_ns'], self.now, 'local_detector', 'test-combat', True, True)
        self.assertEqual(self.controller.combat_entry(evidence).status, 'accepted')
        self.assertEqual(len(self.recorder.records), 1)

    def test_reroll_commits_only_after_two_new_agreeing_changed_offers(self):
        first = self.observation(1, scene='shop', currency=2, reroll_cost=2)
        second = self.observation(2, scene='shop', currency=2, reroll_cost=2)
        for seed in range(10):
            self.controller = NeuralMenuController(self.recorder, clock=lambda: self.now, seed=seed)
            self.controller.handle(*first)
            self.controller.handle(*second)
            if self.controller.pending_decision.target == 'refresh':
                break
        decision = self.controller.pending_decision
        self.assertEqual(decision.target, 'refresh')
        args = self.observation(3, scene='shop', currency=2, reroll_cost=2, target='refresh')
        self.controller.handle(*args)
        self.send(args)
        after = self.controller.handle(*self.observation(4, scene='shop', currency=0,
                                                         reroll_cost=3, changed_offer=True))
        self.assertEqual(after.status, 'wait')
        self.assertEqual(len(self.recorder.records), 0)
        after = self.controller.handle(*self.observation(5, scene='shop', currency=0,
                                                         reroll_cost=3, changed_offer=True))
        self.assertEqual(after.status, 'accepted')
        self.assertEqual(after.reason, 'reroll_observed')
        self.assertEqual(len(self.recorder.records), 1)
        self.assertEqual(self.recorder.records[0]['reward'], 0.0)
        self.assertIsNone(self.controller._previous_shop)
        self.assertIsNone(self.controller._ready_shop)
        self.assertEqual(self.controller.handle(*self.observation(6, scene='shop', currency=0,
                                                                 reroll_cost=3, changed_offer=True)).status, 'wait')

    def test_cold_ocr_expiry_reobserves_without_sampling_or_aborting(self):
        args = self.observation(1)
        self.now += 900_000_000
        self.assertEqual(self.controller.handle(*args).status, "wait")
        self.assertIsNone(self.controller.pending_decision)
        self.assertEqual(self.recorder.rejection_reasons, [])
        fresh = self.observation(12)
        self.assertEqual(self.controller.handle(*fresh).status, "navigate")
        self.assertEqual(self.controller.samples, 1)

    def test_expiry_during_preparation_defers_before_sampling_then_recovers(self):
        from playmodel.games.brotato.neural_menu_controller import observe_menu
        args = self.observation(1)
        def slow_preparation(*values, **kwargs):
            observation = observe_menu(*values, **kwargs)
            self.now = observation.observed_at_ns + MAX_AGE_NS + 1
            return observation
        with patch('playmodel.games.brotato.neural_menu_controller.observe_menu', side_effect=slow_preparation), \
                patch('playmodel.games.brotato.neural_menu_controller.sample_choice') as sample:
            directive = self.controller.handle(*args)
            self.assertEqual(directive.reason, 'stale_menu_observation_reobserve_without_input')
            sample.assert_not_called()
        self.assertIsNone(self.controller.pending_decision)
        self.assertEqual(self.controller._stale_observations, 1)
        self.assertEqual(self.recorder.rejection_reasons, [])
        self.assertEqual(self.controller.handle(*self.observation(12)).status, 'navigate')
        self.assertEqual(self.controller._stale_observations, 0)
        self.assertEqual(self.controller.samples, 1)

    def test_repeated_mid_preparation_expiry_keeps_bound_across_fresh_entries(self):
        from playmodel.games.brotato.neural_menu_controller import observe_menu
        self.controller.max_result_observations = 2
        def slow_preparation(*values, **kwargs):
            observation = observe_menu(*values, **kwargs)
            self.now = observation.observed_at_ns + MAX_AGE_NS + 1
            return observation
        with patch('playmodel.games.brotato.neural_menu_controller.observe_menu', side_effect=slow_preparation), \
                patch('playmodel.games.brotato.neural_menu_controller.sample_choice') as sample:
            for sequence, expected in ((1, 1), (12, 2)):
                self.assertEqual(self.controller.handle(*self.observation(sequence)).status, 'wait')
                self.assertEqual(self.controller._stale_observations, expected)
            with self.assertRaisesRegex(NeuralMenuError, 'Fresh menu observations remained unavailable'):
                self.controller.handle(*self.observation(23))
            sample.assert_not_called()
        self.assertEqual(self.controller.samples, 0)

    def test_preparation_defer_does_not_suppress_malformed_or_illegal_errors(self):
        with patch('playmodel.games.brotato.neural_menu_controller.observe_menu',
                   side_effect=ValueError('malformed observation')):
            with self.assertRaisesRegex(NeuralMenuError, 'malformed observation'):
                self.controller.handle(*self.observation(1))
        self.controller = NeuralMenuController(FullRunRecorder(self.model, 'illegal-test'), clock=lambda: self.now)
        with patch('playmodel.games.brotato.neural_menu_controller.sample_choice',
                   side_effect=ValueError('entirely illegal menu choice')):
            with self.assertRaisesRegex(NeuralMenuError, 'entirely illegal'):
                self.controller.handle(*self.observation(2))

    def test_cached_source_mutation_and_model_change_fail_closed(self):
        args = self.observation(1)
        args[1]["recognition_path"] = "exact_image_cache"
        with self.assertRaisesRegex(NeuralMenuError, "uncached"):
            self.controller.handle(*args)
        self.assertTrue(self.recorder.rejection_reasons)
        recorder = FullRunRecorder(self.model, "second-test")
        controller = NeuralMenuController(recorder, clock=lambda: self.now)
        with torch.no_grad():
            recorder.model.value_head.bias.add_(1)
        with self.assertRaisesRegex(NeuralMenuError, "changed mid-run"):
            controller.handle(*self.observation(2))

    def test_enter_without_authorization_and_duplicate_transmission_are_rejected(self):
        self.controller.handle(*self.observation(1))
        with self.assertRaisesRegex(NeuralMenuError, "Unapproved"):
            self.controller.mark_sent(sent_at_ns=self.now, actual_target=self.controller.pending_decision.target)

    def test_expiry_during_pre_send_verification_defers_without_input(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        from playmodel.games.brotato.menu import selected_button

        def slow_focus(*values, **kwargs):
            result = selected_button(*values, **kwargs)
            self.now = args[0]["capture_started_at_ns"] + MAX_AGE_NS + 1
            return result

        with patch("playmodel.games.brotato.neural_menu_controller.selected_button", side_effect=slow_focus):
            self.assertFalse(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        self.assertFalse(self.controller.awaiting_application)
        self.assertIsNone(self.controller._pending.authorized_at_ns)
        self.assertEqual(self.recorder.rejection_reasons, [])
        next_args = self.observation(12, target=pending.target)
        self.assertEqual(self.controller.handle(*next_args).target, pending.target)
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *next_args))
        self.assertEqual(self.controller.samples, 1)

    def test_transport_report_delay_does_not_expire_a_fresh_actual_send(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        actual_sent_at = self.now + 1_000_000
        self.now = args[0]["capture_started_at_ns"] + MAX_AGE_NS + 100_000_000
        self.controller.mark_sent(sent_at_ns=actual_sent_at, actual_target=pending.target)
        self.assertTrue(self.controller.awaiting_application)
        self.assertEqual(self.recorder.rejection_reasons, [])
        self.assertEqual(len(self.recorder.records), 0)
        receipt = json.loads(next((self.root / "macros" / pending.decision_id).glob("transmission-*.json")).read_text())
        self.assertGreater(receipt["observation_age_at_report_ns"], MAX_AGE_NS)
        self.assertLess(receipt["observation_age_at_send_start_ns"], MAX_AGE_NS)
        self.assertEqual(receipt["guard_failures"], [])

    def test_exact_start_before_expiry_and_late_completion_preserve_transport(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        start = args[0]["capture_started_at_ns"] + MAX_AGE_NS - 10_000_000
        finish = start + 20_000_000
        self.now = finish + 100_000_000
        self.controller.mark_sent(send_started_at_ns=start, sent_at_ns=finish, actual_target=pending.target)
        self.assertTrue(self.controller.awaiting_application)
        self.assertEqual(self.controller._pending.sent_at_ns, finish)
        self.assertEqual(self.recorder.rejection_reasons, [])

    def test_stale_actual_send_and_duplicate_report_are_saved_but_not_accepted(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        self.now = args[0]["capture_started_at_ns"] + MAX_AGE_NS + 1
        with self.assertRaisesRegex(NeuralMenuError, "observation_expired_before_transmission"):
            self.controller.mark_sent(sent_at_ns=self.now, actual_target=pending.target)
        directory = self.root / "macros" / pending.decision_id
        receipt = json.loads(next(directory.glob("transmission-*.json")).read_text())
        self.assertTrue(receipt["successful_transport_reported"])
        self.assertFalse(receipt["game_application_verified"])
        self.assertEqual(receipt["guard_failures"], ["observation_expired_before_transmission"])
        self.assertFalse((directory / "sent.json").exists())
        failure = json.loads(next((self.root / "macros").glob("failure-*.json")).read_text())
        self.assertEqual(failure["details"]["guard_failures"], receipt["guard_failures"])

    def test_model_check_after_actual_transport_keeps_receipt(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        self.assertTrue(self.controller.authorize_enter(pending.decision_id, pending.target, *args))
        self.now += 1_000_000
        with torch.no_grad():
            self.recorder.model.value_head.bias.add_(1)
        with self.assertRaisesRegex(NeuralMenuError, "changed mid-run"):
            self.controller.mark_sent(sent_at_ns=self.now, actual_target=pending.target)
        directory = self.root / "macros" / pending.decision_id
        first_receipt = (directory / "sent.json").read_bytes()
        self.assertTrue(self.controller.awaiting_application)
        self.now += 1_000_000
        with self.assertRaisesRegex(NeuralMenuError, "duplicate_transmission"):
            self.controller.mark_sent(sent_at_ns=self.now, actual_target=pending.target)
        self.assertEqual((directory / "sent.json").read_bytes(), first_receipt)
        self.assertEqual(len(list(directory.glob("transmission-*.json"))), 2)

    def test_new_observation_revokes_prior_enter_authorization(self):
        self.controller.handle(*self.observation(1))
        pending = self.controller.pending_decision
        args = self.observation(2, target=pending.target)
        self.controller.handle(*args)
        self.controller.authorize_enter(pending.decision_id, pending.target, *args)
        self.controller.handle(*self.observation(3, target=pending.target))
        with self.assertRaisesRegex(NeuralMenuError, "Unapproved"):
            self.controller.mark_sent(sent_at_ns=self.now, actual_target=pending.target)


if __name__ == "__main__":
    unittest.main()
