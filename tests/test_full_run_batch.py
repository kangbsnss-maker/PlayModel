"""Synthetic artifact joins: bounded validation, identical results, atomic failure.

No game, capture worker, OCR, or operating-system input is used.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("optional PyTorch unavailable")
import torch

from playmodel.games.brotato.neural_runtime import SCHEMA as COMBAT_SCHEMA
from playmodel.learning.full_run import CLOCK, FullRunRecorder
from playmodel.learning.recurrent_ppo import ModelConfig, RecurrentActorCritic, RolloutBatch
from playmodel.learning.runtime_contract import contract_fields


class FullRunBatchTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.manual_seed(71)
        model = RecurrentActorCritic(ModelConfig(context_dim=4, candidate_dim=3,
            hidden_size=16, visual_size=16, candidate_hidden_size=8))
        self.recorder = FullRunRecorder(model, "synthetic-batched-run", session_ids=("prefix-session",))
        self.reference = FullRunRecorder(model, "synthetic-batched-run", session_ids=("prefix-session",))
        data, evidence = self.decision(self.recorder.model, self.recorder.hidden, 0, phase=1, reset=True)
        for recorder in (self.recorder, self.reference):
            recorder.append_decision(data, evidence=evidence)
            recorder.source_manifests.append({"path": evidence["frame_ref"], "sha256": evidence["frame_sha256"]})

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def decision(self, model, hidden, index, *, phase=0, reset=False):
        count = 3 if phase else 1
        legal = torch.zeros(1, 9, dtype=torch.bool)
        legal[:, :count if phase else 9] = True
        data = {"images": torch.randint(0, 256, (1, 3, 96, 96), dtype=torch.uint8),
                "context": torch.tensor([[index / 100, 0., 1., 0.]]),
                "phase": torch.tensor([phase]), "candidates": torch.rand(1, count, 3),
                "legal_mask": legal, "hidden_before": hidden.clone(), "reset": torch.tensor([reset])}
        with torch.no_grad():
            output = model.step(*(data[key] for key in
                ("images", "context", "phase", "candidates", "legal_mask")),
                hidden=data["hidden_before"], reset=data["reset"])
        action = output.logits.argmax(-1)
        data.update(actions=action, old_log_probs=output.logits.log_softmax(-1).gather(1, action[:, None]).squeeze(1),
                    old_values=output.value, next_hidden=output.next_hidden)
        source = self.root / f"frame-{index}.bgra"
        source.write_bytes(f"synthetic original frame {index}".encode())
        observed = (index + 1) * 1_000_000_000
        evidence = {"frame_ref": str(source.resolve()), "frame_sha256": self.sha(source),
                    "observed_at_ns": observed, "available_at_ns": observed + 1,
                    "decided_at_ns": observed + 2, "sent_at_ns": observed + 3,
                    "action_origin": "policy", "actual_action": int(action.item()),
                    "transmitted": True, "acknowledged": None,
                    "game_application_verified": phase != 0, "clock_domain": CLOCK}
        return data, evidence

    def artifact(self, count=65):
        """Create a real load_rollout-compatible, hash-bound collector artifact."""
        directory = self.root / "combat-segment"
        directory.mkdir()
        hidden = self.recorder.hidden.clone()
        rows = []
        for index in range(1, count + 1):
            data, evidence = self.decision(self.recorder.model, hidden, index)
            hidden = data["next_hidden"].clone()
            rows.append((data, evidence))
        fields = {name: torch.stack([data[name] for data, _ in rows]) for name in
                  ("images", "context", "phase", "candidates", "legal_mask", "actions",
                   "old_log_probs", "old_values", "reset")}
        valid = torch.ones(count, 1, dtype=torch.bool)
        truncated = torch.zeros_like(valid)
        truncated[-1] = True
        batch = RolloutBatch(**fields, rewards=torch.zeros(count, 1),
            next_values=torch.cat((fields["old_values"][1:], torch.zeros(1, 1))),
            terminated=torch.zeros_like(valid), truncated=truncated, valid=valid,
            elapsed_seconds=torch.ones(count, 1), initial_hidden=rows[0][0]["hidden_before"],
            behavior_version=self.recorder.behavior_version, rollout_id=directory.name, split="train")
        torch.save({"schema": COMBAT_SCHEMA, "batch": batch.__dict__}, directory / "flat-rollout.pt")
        torch.save({"initial_states": torch.stack([data["hidden_before"] for data, _ in rows])},
                   directory / "initial-states.pt")
        (directory / "actions.jsonl").write_text("".join(json.dumps(evidence) + "\n" for _, evidence in rows), encoding="utf-8")
        report = {"session_directory": str(directory), "rollout_eligible": True,
                  "flat_rollout_path": str(directory / "flat-rollout.pt"),
                  "initial_states_path": str(directory / "initial-states.pt"),
                  "final_hidden": hidden.tolist()}
        (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "combat-entry-source").write_bytes(b"synthetic independent entry")
        (directory / "combat-entry.json").write_text(json.dumps({
            "frame_sha256": self.sha(directory / "combat-entry-source")}), encoding="utf-8")
        self.freeze(directory)
        return report, rows

    def freeze(self, directory):
        # Mutated test fixtures are new synthetic evidence versions. Real originals are never rewritten.
        manifest = {"schema": COMBAT_SCHEMA, **contract_fields(), "recorder_complete": True,
                    "behavior_version": self.recorder.behavior_version, "sources": [],
                    "files": [{"path": path.name, "sha256": self.sha(path)}
                              for path in sorted(directory.iterdir()) if path.name != "manifest.json"]}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def snapshot(self):
        return deepcopy({"records": self.recorder.records, "hidden": self.recorder.hidden,
                         "session_ids": self.recorder.session_ids,
                         "source_manifests": self.recorder.source_manifests,
                         "rejection_reasons": self.recorder.rejection_reasons})

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected), "tensor changed")
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_batch_join_matches_sequential_records_hidden_and_provenance(self):
        report, rows = self.artifact()
        for data, evidence in rows:
            self.reference.append_decision(data, evidence=evidence)
        directory = Path(report["session_directory"])
        self.reference.session_ids.append(directory.name)
        self.reference.source_manifests.append({"path": str(directory / "manifest.json"),
                                                "sha256": self.sha(directory / "manifest.json")})
        self.recorder.append_combat_report(report)
        for name in ("records", "hidden", "session_ids", "source_manifests", "rejection_reasons"):
            self.assert_nested_equal(getattr(self.recorder, name), getattr(self.reference, name))

    def test_late_invalid_evidence_never_partially_commits(self):
        report, _ = self.artifact()
        directory = Path(report["session_directory"])
        originals = {path.name: path.read_bytes() for path in directory.iterdir()}
        before = self.snapshot()
        for fault in ("log_probability", "value", "hidden", "source", "transport"):
            with self.subTest(fault=fault):
                for name, data in originals.items():
                    (directory / name).write_bytes(data)
                if fault in ("log_probability", "value"):
                    path = directory / "flat-rollout.pt"
                    payload = torch.load(path, weights_only=True)
                    key = "old_log_probs" if fault == "log_probability" else "old_values"
                    payload["batch"][key][40, 0] += 0.25
                    torch.save(payload, path)
                elif fault == "hidden":
                    path = directory / "initial-states.pt"
                    payload = torch.load(path, weights_only=True)
                    payload["initial_states"][40, 0, 0] += 0.25
                    torch.save(payload, path)
                else:
                    path = directory / "actions.jsonl"
                    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                    if fault == "source":
                        rows[40]["frame_sha256"] = "0" * 64
                    else:
                        rows[40]["transmitted"] = False
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                self.freeze(directory)
                with self.assertRaises(ValueError):
                    self.recorder.append_combat_report(report)
                self.assert_nested_equal(self.snapshot(), before)

    def test_model_changed_during_validation_rejects_entire_join(self):
        report, _ = self.artifact(count=1)
        before = self.snapshot()
        original_step = self.recorder.model.step

        def mutate_after_forward(*args, **kwargs):
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                next(self.recorder.model.parameters()).add_(0.01)
            return result

        # Mutate after the only validation forward so no later forward comparison
        # can catch it: the final frozen-model check must prevent the commit.
        with patch.object(self.recorder.model, "step", side_effect=mutate_after_forward):
            with self.assertRaisesRegex(ValueError, "behavior model changed|policy.*changed|model.*changed"):
                self.recorder.append_combat_report(report)
        self.assert_nested_equal(self.snapshot(), before)

    def test_validation_hashes_twice_and_bounds_forward_batch_size(self):
        report, _ = self.artifact()
        model = self.recorder.model
        with (patch.object(model, "policy_version", wraps=model.policy_version) as version,
              patch.object(model, "step", wraps=model.step) as forward):
            self.recorder.append_combat_report(report)
        self.assertEqual(version.call_count, 2)
        self.assertEqual(forward.call_count, 3)
        self.assertEqual([call.args[0].shape[0] for call in forward.call_args_list], [32, 32, 1])
        self.assertEqual(len(self.recorder.records), 66)


if __name__ == "__main__":
    unittest.main()
