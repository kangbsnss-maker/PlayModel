"""Deterministic contract/gradient tests; these are not real gameplay evaluation."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from playmodel.games.brotato.decision_learning import (
    AppliedEvidence, ChoiceCandidate, ChoicePolicy, ChoiceStep, DecisionRun,
    Evidence, RunOutcome, load_run_manifest, save_run_manifest, train_run,
    validate_run, verify_split_separation,
)


class DecisionLearningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.policy = ChoicePolicy.initial(7)
        self.policy.save(self.root / "source.json")
        self.candidates = (
            ChoiceCandidate("slot0", "upgrade", {"damage": 1, "unknown_effect": 1}, semantic_id="raw-text:damage"),
            ChoiceCandidate("slot1", "upgrade", {"speed": 1, "unknown_effect": 1}, semantic_id="raw-text:speed"),
            ChoiceCandidate("slot2", "upgrade", {"speed": -1}, legal=False),
        )
        self.decision = self.policy.sample({"ranged_build": 1}, self.candidates, random.Random(9))

    def evidence(self, name, observed, *, verified=True):
        path = self.root / (name + ".frame")
        payload = name.encode()
        path.write_bytes(payload)
        return Evidence(str(path.resolve()), hashlib.sha256(payload).hexdigest(),
                        observed, observed + 1, observed + 2,
                        "synthetic-test-verifier", verified, True)

    def run_record(self, name="run1", split="train", terminal="death"):
        before = self.evidence(name + "before", 10)
        after = self.evidence(name + "after", 30)
        end = self.evidence(name + "end", 100)
        step = ChoiceStep(name + ":choice1", self.decision, before, 15, 20,
                          self.decision.selected_id, "synthetic-send:1",
                          AppliedEvidence("upgrade_selected", before, after, True),
                          acknowledged=True)
        return DecisionRun(name, split, "fixed-movement-test", self.policy.version,
                           (step,), RunOutcome(terminal, end, objective_id="synthetic-objective" if terminal == "objective_reached" else None),
                           (name + ":session",))

    def test_masks_permutation_and_ephemeral_ids(self):
        probabilities = self.policy.distribution({}, self.candidates)
        self.assertEqual(probabilities[2], 0)
        reversed_candidates = tuple(reversed(self.candidates))
        reverse = self.policy.distribution({}, reversed_candidates)
        self.assertEqual(probabilities, tuple(reversed(reverse)))
        renamed = tuple(replace(c, candidate_id="new-" + c.candidate_id) for c in self.candidates)
        self.assertEqual(probabilities, self.policy.distribution({}, renamed))
        with self.assertRaisesRegex(ValueError, "no legal"):
            self.policy.sample({}, [replace(c, legal=False) for c in self.candidates], random.Random(1))

    def test_candidate_state_interactions_change_relative_preferences(self):
        first = self.policy.distribution({"ranged_build": 1}, self.candidates)
        second = self.policy.distribution({"ranged_build": -1}, self.candidates)
        self.assertNotEqual(first, second)

    def test_source_snapshot_is_immutable(self):
        values = {"damage": 1}
        candidate = ChoiceCandidate("a", "upgrade", values)
        values["damage"] = -1
        self.assertEqual(candidate.features, (("damage", 1.0),))
        with self.assertRaises(ValueError):
            ChoiceCandidate("a", "upgrade", {"damage": float("nan")})

    def test_real_update_uses_return_and_value_not_self_labels(self):
        for terminal, direction in (("death", -1), ("objective_reached", 1)):
            with self.subTest(terminal=terminal):
                run = self.run_record(terminal, terminal=terminal)
                manifest = save_run_manifest(run, self.root / (terminal + "-manifest"))
                self.assertEqual(load_run_manifest(manifest), run)
                target = self.root / (terminal + "-candidate.json")
                report = train_run(manifest, self.root / "source.json", target)
                candidate = ChoicePolicy.load(target)
                index = [c.candidate_id for c in self.candidates].index(self.decision.selected_id)
                before = self.decision.probabilities[index]
                after = candidate.distribution(self.decision.state, self.candidates)[index]
                self.assertGreater(direction * (after - before), 0)
                self.assertNotEqual(candidate.actor_weights, self.policy.actor_weights)
                self.assertNotEqual(candidate.value_weights, self.policy.value_weights)
                self.assertEqual(report["deployment_status"], "unapproved_candidate")
                self.assertEqual(candidate.updates, 1)
                self.assertEqual(candidate.parent_version, self.policy.version)
                with self.assertRaises(FileExistsError):
                    train_run(manifest, self.root / "source.json", target)
                with self.assertRaises(ValueError):
                    train_run(manifest, target, self.root / (terminal + "-replay.json"))

    def test_pending_excluded_not_fabricated_reward(self):
        run = self.run_record()
        pending = replace(run.steps[0], applied=None)
        run = replace(run, steps=(pending,))
        report = validate_run(self.policy, run)
        self.assertEqual(report["eligible_indices"], [])
        self.assertEqual(report["excluded"], {pending.decision_id: "application_unknown"})
        manifest = save_run_manifest(run, self.root / "pending")
        with self.assertRaisesRegex(ValueError, "no verified-applied"):
            train_run(manifest, self.root / "source.json", self.root / "must-not-exist.json")
        self.assertFalse((self.root / "must-not-exist.json").exists())

    def test_purchase_requires_delta_and_ownership(self):
        run = self.run_record()
        candidates = [ChoiceCandidate("offer0", "buy", {"cost": .5}, semantic_id="item0")]
        decision = self.policy.sample({}, candidates, random.Random(1))
        before = run.steps[0].observation
        evidence = AppliedEvidence("purchase_applied", before, run.steps[0].applied.after,
                                   True, currency_before=100, currency_after=75, cost=25)
        step = replace(run.steps[0], decision=decision, actual_candidate_id="offer0", applied=evidence)
        self.assertFalse(validate_run(self.policy, replace(run, steps=(step,)))["eligible_indices"])
        evidence = replace(evidence, ownership_changed=True)
        step = replace(step, applied=evidence)
        self.assertEqual(validate_run(self.policy, replace(run, steps=(step,)))["eligible_indices"], [0])
        step = replace(step, applied=replace(evidence, currency_after=70))
        self.assertFalse(validate_run(self.policy, replace(run, steps=(step,)))["eligible_indices"])

    def test_forged_probabilities_future_stale_and_interventions_rejected(self):
        run = self.run_record()
        corruptions = [
            replace(run, steps=(replace(run.steps[0], decision=replace(self.decision, probabilities=(.5, .5, 0))),)),
            replace(run, steps=(replace(run.steps[0], decided_at_ns=9),)),
            replace(run, steps=(replace(run.steps[0], sent_at_ns=600_000_010),),
                    outcome=RunOutcome("death", self.evidence("later", 700_000_000))),
            replace(run, human_intervention=True),
            replace(run, steps=(replace(run.steps[0], actual_candidate_id="other"),)),
            replace(run, steps=(replace(run.steps[0], action_origin="recovery"),)),
            replace(run, outcome=RunOutcome("truncated", None)),
            replace(run, outcome=replace(run.outcome, evidence=replace(run.outcome.evidence, origin="ocr"))),
        ]
        for invalid in corruptions:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_run(self.policy, invalid)

    def test_evaluation_cannot_train_and_split_leakage_is_rejected(self):
        training = self.run_record("train")
        evaluation = self.run_record("eval", split="evaluation")
        with self.assertRaisesRegex(ValueError, "evaluation"):
            validate_run(self.policy, evaluation)
        self.assertEqual(validate_run(self.policy, evaluation, for_training=False)["split"], "evaluation")
        train_path = save_run_manifest(training, self.root / "training")
        eval_path = save_run_manifest(evaluation, self.root / "evaluation")
        self.assertTrue(verify_split_separation([train_path], [eval_path])["independent_by_manifest"])
        leaked = replace(evaluation, session_ids=training.session_ids)
        leaked_path = save_run_manifest(leaked, self.root / "leaked")
        with self.assertRaisesRegex(ValueError, "leakage"):
            verify_split_separation([train_path], [leaked_path])

    def test_manifest_checkpoint_and_evidence_tampering_fail(self):
        run = self.run_record()
        manifest = save_run_manifest(run, self.root / "frozen")
        Path(run.steps[0].observation.frame_ref).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "modified"):
            train_run(manifest, self.root / "source.json", self.root / "bad.json")
        with (manifest.parent / "run.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_run_manifest(manifest)
        payload = json.loads((self.root / "source.json").read_text())
        payload["actor_weights"][0] += .01
        (self.root / "forged.json").write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ChoicePolicy.load(self.root / "forged.json")

    def test_only_verified_distinct_wave_events_shape_return(self):
        run = self.run_record()
        wave = self.evidence("wave1", 50)
        outcome = replace(run.outcome, wave_clears=((1, wave),))
        self.assertEqual(validate_run(self.policy, replace(run, outcome=outcome))["return"], 1 / 21 - 1)
        with self.assertRaises(ValueError):
            validate_run(self.policy, replace(run, outcome=replace(outcome, wave_clears=((1, wave), (2, wave)))))
        with self.assertRaises(ValueError):
            validate_run(self.policy, replace(run, outcome=replace(outcome, wave_clears=((1, replace(wave, verified=False)),))))

    def test_endless_progress_remains_monotonic_after_wave_twenty(self):
        run = self.run_record()
        end = self.evidence("endless-terminal", 1000)
        waves = tuple((index, self.evidence(f"endless-wave-{index}", 100 + index))
                      for index in range(1, 31))
        first = replace(run, outcome=RunOutcome("death", end, waves[:20]))
        second = replace(run, outcome=RunOutcome("death", end, waves))
        self.assertGreater(validate_run(self.policy, second)["return"],
                           validate_run(self.policy, first)["return"])

    def test_later_action_cannot_supply_earlier_application_evidence(self):
        run = self.run_record()
        second_observation = self.evidence("second-before", 22)
        second = replace(run.steps[0], decision_id="second", observation=second_observation,
                         decided_at_ns=25, sent_at_ns=26, applied=None)
        # First application is observed at t=30, after this second send at 26.
        with self.assertRaisesRegex(ValueError, "before the next action"):
            validate_run(self.policy, replace(run, steps=(*run.steps, second)))


if __name__ == "__main__":
    unittest.main()
