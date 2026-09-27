"""Local pilot tests: mathematical updates and strict rollout eligibility only."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from playmodel.learning import (
    ACTION_NAMES, ALL_ACTIONS_MASK, NEUTRAL_ONLY_MASK, FEATURE_COUNT, MOVEMENTS, NAVIGATION_OFFSET,
    EpisodeRecord, LinearMovementPolicy, MovementStep,
    StateEvidence, extract_features, load_checkpoint, reinforce_update, save_checkpoint, validate_episode,
)


class MovementLearningTests(unittest.TestCase):
    def setUp(self):
        self.frame = bytes((30, 60, 90, 255)) * (320 * 180)
        self.digest = hashlib.sha256(self.frame).hexdigest()
        self.policy = LinearMovementPolicy.random(seed=42)
        self.features = extract_features(self.frame, 320, 180, 0)
        self.decision = self.policy.sample(self.features, rng=random.Random(1))
        self.entry = StateEvidence("combat", "entry.bgra", self.digest, 100, 101, 200,
                                   "developer_verified", "developer-screen-review-v1", True, True)
        self.terminal = StateEvidence("wave_clear", "terminal.bgra", self.digest, 200, 201, 220,
                                      "developer_verified", "developer-screen-review-v1", True, True)
        self.step = MovementStep(1, 3, "step-1.bgra", self.digest, 110, 111, 112, 113,
                                 self.decision, self.decision.action_index, True, "input.jsonl#1")
        self.episode = EpisodeRecord("episode-1", self.policy.version, "manifest.json", "a" * 64,
                                     "capture-320x180-v1.json", 3, 0, self.entry, (self.step,), self.terminal)

    def test_feature_rgb_order_grid_and_actual_previous_action(self):
        self.assertEqual(len(self.features), 157)
        expected_rgb = (90 / 127.5 - 1, 60 / 127.5 - 1, 30 / 127.5 - 1)
        self.assertEqual(self.features[:3], expected_rgb)
        self.assertEqual(self.features[:144], expected_rgb * 48)
        self.assertEqual(self.features[144:153], (1.0,) + (0.0,) * 8)
        self.assertEqual(self.features[153:156], (0.0, 0.0, 0.0))
        self.assertEqual(self.features[-1], 1)
        other = extract_features(self.frame, 320, 180, 7)
        self.assertEqual(other[144:153], (0.0,) * 7 + (1.0, 0.0))
        small = bytes(channel for y in range(6) for x in range(8) for channel in (x, y, x + y, 255))
        sampled = extract_features(small, 8, 6, 0)
        self.assertEqual(sampled[141:144], ((7 + 5) / 127.5 - 1, 5 / 127.5 - 1, 7 / 127.5 - 1))

    def test_feature_input_must_be_immutable_complete_and_legal(self):
        for frame, width, height, previous in ((self.frame[:-1], 320, 180, 0),
                                               (bytearray(self.frame), 320, 180, 0),
                                               (self.frame, 320, 180, 9),
                                               (self.frame, 320, 180, True),
                                               (self.frame, 7, 6, 0)):
            with self.subTest(width=width, previous=previous), self.assertRaises(ValueError):
                extract_features(frame, width, height, previous)

    def test_seeded_policy_and_sampling_are_reproducible(self):
        self.assertEqual(self.policy, LinearMovementPolicy.random(seed=42))
        self.assertNotEqual(self.policy.version, LinearMovementPolicy.random(seed=43).version)
        self.assertEqual(self.decision, self.policy.sample(self.features, rng=random.Random(1)))
        self.assertEqual(self.decision.movement, MOVEMENTS[self.decision.action_index])
        self.assertAlmostEqual(sum(self.decision.probabilities), 1.0)
        self.assertAlmostEqual(self.decision.log_probability,
                               math.log(self.decision.probabilities[self.decision.action_index]))

    def test_geometry_prior_points_toward_all_eight_target_directions(self):
        policy = LinearMovementPolicy.initialized_for_collection(seed=42)
        self.assertEqual(policy.baseline_origin, "geometry_prior")
        self.assertEqual(policy.navigation_prior_strength, 6)
        for action, (dx, dy) in enumerate(MOVEMENTS[1:], 1):
            features = extract_features(self.frame, 320, 180, 0, navigation=(dx, dy, True))
            probabilities = policy.probabilities(features)
            with self.subTest(action=ACTION_NAMES[action]):
                self.assertEqual(max(range(9), key=probabilities.__getitem__), action)
                expected_x = sum(probability * movement[0]
                                 for probability, movement in zip(probabilities, MOVEMENTS))
                expected_y = sum(probability * movement[1]
                                 for probability, movement in zip(probabilities, MOVEMENTS))
                self.assertGreater((expected_x * dx + expected_y * dy) / math.hypot(dx, dy), 0.7)
                self.assertTrue(all(0 < probability < 1 for probability in probabilities))
        # Equal-length movement vectors: diagonal receives strength/sqrt(2) per axis.
        self.assertAlmostEqual(policy.weights[2][NAVIGATION_OFFSET], 6 / math.sqrt(2))
        self.assertAlmostEqual(policy.weights[2][NAVIGATION_OFFSET + 1], -6 / math.sqrt(2))

    def test_no_target_retains_seeded_exploration_and_masks_invalid_geometry(self):
        policy = LinearMovementPolicy.initialized_for_collection(seed=42)
        invalid = extract_features(self.frame, 320, 180, 0, navigation=(1, -1, False))
        self.assertEqual(invalid, self.features)
        self.assertEqual(policy.probabilities(invalid), self.policy.probabilities(self.features))
        self.assertTrue(all(0.05 < probability < 0.2 for probability in policy.probabilities(invalid)))
        selected = {policy.sample(invalid, rng=random.Random(seed)).action_index for seed in range(100)}
        self.assertEqual(selected, set(range(9)))

    def test_navigation_requires_finite_normalized_vector_and_binary_validity(self):
        for navigation in ((2, 0, True), (0, -2, True), (float("nan"), 0, True),
                           (0, 0, 2), (0, 0, 1.0), (True, 0, True), (0, 0), [0, 0, True]):
            with self.subTest(navigation=navigation), self.assertRaises(ValueError):
                extract_features(self.frame, 320, 180, 0, navigation=navigation)
        malformed = list(self.features)
        malformed[NAVIGATION_OFFSET] = 1
        with self.assertRaises(ValueError):
            self.policy.probabilities(tuple(malformed))

    def test_geometry_prior_is_trainable_and_origin_survives_checkpoint(self):
        policy = LinearMovementPolicy.initialized_for_collection(seed=42)
        features = extract_features(self.frame, 320, 180, 0, navigation=(1, 0, True))
        decision = policy.sample(features, rng=random.Random(1))
        step = replace(self.step, decision=decision, actual_action_index=decision.action_index)
        episode = replace(self.episode, policy_version=policy.version, steps=(step,))
        positive = reinforce_update(policy, episode)
        negative = reinforce_update(policy, replace(episode, terminal=replace(self.terminal, kind="death")))
        selected = decision.action_index
        self.assertGreater(positive.policy.probabilities(features)[selected], decision.probabilities[selected])
        self.assertLess(negative.policy.probabilities(features)[selected], decision.probabilities[selected])
        self.assertNotEqual(positive.policy.weights[selected][NAVIGATION_OFFSET],
                            policy.weights[selected][NAVIGATION_OFFSET])
        self.assertEqual(positive.report["baseline_origin"], "geometry_prior")
        self.assertEqual(positive.policy.baseline_origin, "geometry_prior")
        with tempfile.TemporaryDirectory() as directory:
            target = save_checkpoint(positive.policy, Path(directory) / "geometry-candidate.json")
            restored = load_checkpoint(target)
            self.assertEqual(restored, positive.policy)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "playmodel.linear-movement.v3")
            self.assertEqual(payload["navigation_features"], ["goal_dx", "goal_dy", "valid"])
            payload["schema"] = "playmodel.linear-movement.v1"
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint(target)

    def test_neutral_only_mask_has_unit_probability_and_exactly_zero_gradient(self):
        decision = self.policy.sample(self.features, rng=random.Random(1), mask=NEUTRAL_ONLY_MASK)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(decision.movement, (0, 0))
        self.assertEqual(decision.probabilities, (1.0,) + (0.0,) * 8)
        self.assertEqual(decision.log_probability, 0)
        self.assertEqual(decision.allowed_actions, NEUTRAL_ONLY_MASK)
        step = replace(self.step, decision=decision, actual_action_index=0)
        result = reinforce_update(self.policy, replace(self.episode, steps=(step,)))
        self.assertEqual(result.policy.weights, self.policy.weights)
        self.assertEqual(result.report["gradient_norm_before_clip"], 0)
        self.assertEqual(result.report["neutral_only_steps"], 1)
        self.assertFalse(result.report["weights_changed"])

    def test_invalid_masks_and_mask_distribution_mismatch_are_rejected(self):
        for mask in ((False,) * 9, (True,) * 8, (1,) * 9, [True] * 9):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.policy.sample(self.features, rng=random.Random(1), mask=mask)
        for allowed in (NEUTRAL_ONLY_MASK, None):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                wrong = replace(self.step, decision=replace(self.decision, allowed_actions=allowed))
                validate_episode(self.policy, replace(self.episode, steps=(wrong,)))

    def test_recorded_neutral_step_keeps_actual_previous_action_history(self):
        before = self.step.actual_action_index
        neutral_features = extract_features(self.frame, 320, 180, before)
        neutral = self.policy.sample(neutral_features, rng=random.Random(1), mask=NEUTRAL_ONLY_MASK)
        neutral_step = replace(self.step, sequence=2, observed_at_ns=120, available_at_ns=121,
                               decided_at_ns=122, sent_at_ns=123, decision=neutral, actual_action_index=0,
                               transmission_ref="input.jsonl#2")
        recovered = self.policy.sample(extract_features(self.frame, 320, 180, 0), rng=random.Random(1))
        recovered_step = replace(self.step, sequence=3, observed_at_ns=130, available_at_ns=131,
                                 decided_at_ns=132, sent_at_ns=133, decision=recovered,
                                 actual_action_index=recovered.action_index, transmission_ref="input.jsonl#3")
        episode = replace(self.episode, steps=(self.step, neutral_step, recovered_step))
        result = reinforce_update(self.policy, episode)
        self.assertEqual(result.report["neutral_only_steps"], 1)
        self.assertGreater(result.report["gradient_norm_before_clip"], 0)
        self.assertEqual(recovered.allowed_actions, ALL_ACTIONS_MASK)

    def test_positive_and_negative_terminal_return_change_selected_probability(self):
        selected = self.decision.action_index
        before = self.policy.probabilities(self.features)[selected]
        positive = reinforce_update(self.policy, self.episode)
        negative = reinforce_update(self.policy, replace(self.episode,
                                     terminal=replace(self.terminal, kind="death")))
        self.assertGreater(positive.policy.probabilities(self.features)[selected], before)
        self.assertLess(negative.policy.probabilities(self.features)[selected], before)
        self.assertEqual(self.policy.probabilities(self.features)[selected], before)
        self.assertEqual(positive.policy.parent_version, self.policy.version)
        self.assertEqual(positive.report["return"], 1)
        self.assertEqual(negative.report["return"], -1)
        self.assertFalse(positive.report["performance_improvement_verified"])
        self.assertFalse(positive.report["promotion_approved"])
        self.assertEqual(positive.report["terminal_origin"], "developer_verified")

    def test_analytic_policy_gradient_matches_finite_difference(self):
        learning_rate, epsilon = 0.0001, 0.000001
        update = reinforce_update(self.policy, self.episode, learning_rate=learning_rate, gradient_clip=1000)
        selected = self.decision.action_index
        for action, column in ((selected, 0), ((selected + 1) % 9, 0), (selected, 144),
                               ((selected + 1) % 9, FEATURE_COUNT - 1)):
            def perturbed(delta):
                rows = [list(row) for row in self.policy.weights]
                rows[action][column] += delta
                return LinearMovementPolicy(tuple(tuple(row) for row in rows))

            numerical = (math.log(perturbed(epsilon).probabilities(self.features)[selected])
                         - math.log(perturbed(-epsilon).probabilities(self.features)[selected])) / (2 * epsilon)
            analytic = (update.policy.weights[action][column] - self.policy.weights[action][column]) / learning_rate
            self.assertAlmostEqual(analytic, numerical, places=7)

    def test_episode_constant_return_is_not_centered_to_zero(self):
        second_features = extract_features(self.frame, 320, 180, self.step.actual_action_index)
        second_decision = self.policy.sample(second_features, rng=random.Random(1))
        second = replace(self.step, sequence=2, frame_ref="step-2.bgra", observed_at_ns=120,
                         available_at_ns=121, decided_at_ns=122, sent_at_ns=123, decision=second_decision,
                         actual_action_index=second_decision.action_index, transmission_ref="input.jsonl#2")
        update = reinforce_update(self.policy, replace(self.episode, steps=(self.step, second)))
        self.assertTrue(update.report["weights_changed"])
        self.assertGreater(update.report["gradient_norm_before_clip"], 0)
        self.assertEqual(update.report["baseline"], 0)
        self.assertEqual(update.report["gamma"], 1)

    def test_gradient_global_clip_and_all_candidate_weights_remain_finite(self):
        update = reinforce_update(self.policy, self.episode, gradient_clip=0.05)
        self.assertAlmostEqual(update.report["gradient_norm_after_clip"], 0.05)
        self.assertGreater(update.report["gradient_norm_before_clip"], 0.05)
        self.assertTrue(all(math.isfinite(value) for row in update.policy.weights for value in row))

    def test_large_finite_logits_have_stable_softmax(self):
        rows = tuple((0.0,) * (FEATURE_COUNT - 1) + (1000.0 if index == 3 else -1000.0,)
                     for index in range(len(ACTION_NAMES)))
        policy = LinearMovementPolicy(rows)
        probabilities = policy.probabilities(self.features)
        self.assertEqual(probabilities[3], 1)
        self.assertEqual(sum(probabilities), 1)
        self.assertEqual(policy.sample(self.features, rng=random.Random(3)).action_index, 3)

    def test_missing_or_unverified_combat_and_terminal_are_rejected(self):
        for changes in ({"combat_entry": None}, {"terminal": None},
                        {"terminal": replace(self.terminal, verified=False)},
                        {"terminal": replace(self.terminal, independent_of_policy=False)},
                        {"terminal": replace(self.terminal, kind="time_limit")},
                        {"terminal": replace(self.terminal, origin="policy_prediction")},
                        {"terminal": replace(self.terminal, verifier_id=self.policy.version)},
                        {"terminal": replace(self.terminal, observed_at_ns=112)},
                        {"terminal": replace(self.terminal, clock_domain="monotonic_ns_same_host")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                reinforce_update(self.policy, replace(self.episode, **changes))

    def test_off_policy_replay_and_mixed_step_policy_are_rejected(self):
        updated = reinforce_update(self.policy, self.episode).policy
        with self.assertRaisesRegex(ValueError, "policy version mismatch"):
            reinforce_update(updated, self.episode)
        decision = replace(self.decision, policy_version="b" * 64)
        with self.assertRaisesRegex(ValueError, "policy version mismatch"):
            validate_episode(self.policy, replace(self.episode, steps=(replace(self.step, decision=decision),)))

    def test_intervention_guards_truncation_and_empty_rollout_rejected(self):
        for changes in ({"truncated": True}, {"human_intervention": True}, {"guard_failure": True},
                        {"steps": ()}, {"truncated": 0}, {"manifest_sha256": "missing"},
                        {"capture_config_ref": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_episode(self.policy, replace(self.episode, **changes))

    def test_actual_action_time_source_and_probability_ledger_are_required(self):
        wrong_previous = self.policy.sample(extract_features(self.frame, 320, 180, 4), rng=random.Random(1))
        for changes in ({"transmitted": False}, {"acknowledged": False}, {"action_origin": "recovery"},
                        {"actual_action_index": (self.step.actual_action_index + 1) % 9},
                        {"generation": 4}, {"frame_sha256": "unknown"}, {"transmission_ref": ""},
                        {"available_at_ns": 115}, {"sent_at_ns": 201}, {"observed_at_ns": 99},
                        {"clock_domain": "monotonic_ns_same_host"},
                        {"decision": replace(self.decision, movement=list(self.decision.movement))},
                        {"decision": replace(self.decision, probabilities=(1 / 9,) * 9)},
                        {"decision": replace(self.decision, log_probability=float("nan"))},
                        {"decision": wrong_previous, "actual_action_index": wrong_previous.action_index}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_episode(self.policy, replace(self.episode, steps=(replace(self.step, **changes),)))

    def test_duplicate_transmission_and_sequence_cannot_double_weight_rollout(self):
        with self.assertRaises(ValueError):
            validate_episode(self.policy, replace(self.episode, steps=(self.step, self.step)))
        second_features = extract_features(self.frame, 320, 180, self.step.actual_action_index)
        decision = self.policy.sample(second_features, rng=random.Random(2))
        second = replace(self.step, sequence=2, observed_at_ns=120, available_at_ns=121,
                         decided_at_ns=122, sent_at_ns=123, decision=decision,
                         actual_action_index=decision.action_index)
        with self.assertRaisesRegex(ValueError, "duplicate transmission"):
            validate_episode(self.policy, replace(self.episode, steps=(self.step, second)))

    def test_hyperparameters_and_nonfinite_weights_rejected(self):
        for kwargs in ({"learning_rate": 0}, {"learning_rate": -1}, {"learning_rate": float("nan")},
                       {"learning_rate": True}, {"gradient_clip": 0}, {"gradient_clip": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reinforce_update(self.policy, self.episode, **kwargs)
        weights = list(self.policy.weights)
        weights[0] = (float("inf"),) + weights[0][1:]
        with self.assertRaises(ValueError):
            LinearMovementPolicy(tuple(weights))

    def test_checkpoint_roundtrip_preserves_exact_inference_and_protects_existing_file(self):
        candidate = reinforce_update(self.policy, self.episode).policy
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "candidate.json"
            save_checkpoint(candidate, target)
            original = target.read_bytes()
            restored = load_checkpoint(target)
            self.assertEqual(restored, candidate)
            self.assertEqual(restored.probabilities(self.features), candidate.probabilities(self.features))
            with self.assertRaises(FileExistsError):
                save_checkpoint(self.policy, target)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            payload = json.loads(original)
            payload["weights"][0][0] += 1
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_checkpoint(target)


if __name__ == "__main__":
    unittest.main()
