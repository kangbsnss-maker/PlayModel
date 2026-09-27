"""Immutable visual replay: source hashes, run splits and bounded lazy IO."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from playmodel.learning.replay_data import (
    VisualReplayDataset, build_visual_manifest, register_training_use, sha256_file, split_for_group,
)


def write(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def run_for(split: str, start: int = 0) -> str:
    for number in range(start, start + 1000):
        name = f"run-{number}"
        if split_for_group("run:" + name) == split:
            return name
    raise AssertionError("No deterministic split bucket")


def episode(root, session, run, *, count=3, train=True, color=20, collection="brotato-sessions"):
    directory = root / "artifacts" / collection / session / "pilots" / (session + "-pilot")
    directory.mkdir(parents=True)
    context = {"run_id": run} if run is not None else {}
    write(directory.parent.parent / "report.json", {"run_context": context})
    write(directory / "capture-config.json", {"train": train, "stride": 6})
    (directory / "frames").mkdir()
    steps, frames = [], []
    for index in range(count):
        path = directory / "frames" / f"{index:06d}.bgra"
        # Private frame data is never included in source control.
        raw = bytes((color + index, color + index + 1, color + index + 2, 255)) * 8 * 8
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        write(path.with_suffix(".json"), {
            "metadata": {"sample_width": 8, "sample_height": 8, "capture_started_at_ns": 100 + index * 20},
            "available_at_ns": 110 + index * 20,
        })
        relative = path.relative_to(directory).as_posix()
        frames.append({"path": relative, "sha256": sha})
        steps.append({"frame_ref": relative, "frame_sha256": sha, "observed_at_ns": 100 + index * 20,
                      "available_at_ns": 110 + index * 20,
                      "actual_action_index": 8, "decision": {"deliberately_not_a_label": True}})
    write(directory / "manifest.json", {
        "schema": "brotato-pilot-manifest-v1", "steps": count, "recorder_complete": True,
        "frames": frames, "files": [{"path": "capture-config.json",
                                       "sha256": sha256_file(directory / "capture-config.json")}],
    })
    write(directory / "episode.json", {
        "episode_id": directory.name, "policy_version": "old-policy", "manifest_ref": "manifest.json",
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "capture_config_ref": "capture-config.json", "steps": steps,
    })
    return directory


def neural_episode(root, session, run, *, split="train", color=20, collection="neural-sessions"):
    directory = episode(root, session, run, count=3, color=color, collection=collection)
    old = json.loads((directory / "episode.json").read_text())
    (directory / "episode.json").unlink()
    write(directory / "capture-config.json", {"train": False, "split": split, "stride": 6})
    write(directory.parent.parent / "report.json", {
        "run_context": {"run_id": run, "experimental_partial_character_run": True},
    })
    with (directory / "actions.jsonl").open("w", encoding="utf-8") as stream:
        for step in old["steps"]:
            stream.write(json.dumps({**step, "transmitted": True, "old_log_probability": -1.5,
                                     "reward": -1.0}) + "\n")
            metadata = directory / Path(step["frame_ref"]).with_suffix(".json")
            write(metadata, json.loads(metadata.read_text())["metadata"])
    files = [directory / "capture-config.json", directory / "actions.jsonl",
             *(directory / "frames").iterdir()]
    write(directory / "manifest.json", {"schema": "playmodel.neural-movement-rollout.v1",
          "behavior_version": "neural-policy", "recorder_complete": True,
          "files": [{"path": path.relative_to(directory).as_posix(), "sha256": sha256_file(path)} for path in files]})
    return directory


class VisualReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def build(self, name="replay.json", **kwargs):
        path = self.root / name
        document = build_visual_manifest(self.root, path, **kwargs)
        return path, document

    def test_actual_pixels_reused_without_actions_and_immutable_manifest(self):
        episode(self.root, "s1", run_for("train"))
        path, document = self.build(max_frames=2)
        self.assertEqual(len(document["samples"]), 2)
        self.assertNotIn("actual_action_index", document["samples"][0])
        self.assertNotIn("decision", document["samples"][0])
        dataset = VisualReplayDataset(path, size=8)
        rgb = dataset.read_rgb(0)
        self.assertEqual(len(rgb), 8 * 8 * 3)
        self.assertEqual(rgb[:3], bytes((22, 21, 20)))
        with self.assertRaises(FileExistsError):
            build_visual_manifest(self.root, path)

    def test_run_splits_are_shared_across_wave_and_session_boundaries(self):
        training_run, test_run = run_for("train"), run_for("test")
        episode(self.root, "s1", training_run, color=20)
        episode(self.root, "s2", training_run, color=40)
        episode(self.root, "s3", test_run, color=60)
        path, document = self.build(max_frames=20)
        roles = {}
        for sample in document["samples"]:
            roles.setdefault(sample["group_id"], set()).add(sample["split"])
        self.assertEqual(roles["run:" + training_run], {"train"})
        self.assertEqual(roles["run:" + test_run], {"test"})
        self.assertEqual(len(VisualReplayDataset(path, split="train")), 6)
        self.assertEqual(len(VisualReplayDataset(path, split="test")), 3)
        _, repeated = self.build("other.json", max_frames=20, seed=40)
        self.assertEqual({s["group_id"]: s["split"] for s in document["samples"]},
                         {s["group_id"]: s["split"] for s in repeated["samples"]})

    def test_reviewed_menu_holdout_excludes_whole_run_not_only_its_session(self):
        protected_run = run_for("train")
        episode(self.root, "menu-session", protected_run, color=20)
        episode(self.root, "later-same-run", protected_run, color=40)
        episode(self.root, "another-run", run_for("train", 100), color=60)
        write(self.root / "data/menu/review/reviewed-manifest.json", {
            "samples": [{"session_id": "menu-session", "split": "test"}],
        })
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"another-run"})
        self.assertIn("run:" + protected_run, document["protected_groups"])

    def test_evaluation_configuration_without_episode_excludes_related_training(self):
        run = run_for("train")
        evaluation = episode(self.root, "evaluation", run, train=False, color=20)
        (evaluation / "episode.json").unlink()
        episode(self.root, "related-train", run, color=40)
        episode(self.root, "safe-train", run_for("train", 100), color=60)
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"safe-train"})

    def test_neural_train_split_is_reusable_when_inline_training_flag_is_false(self):
        neural_episode(self.root, "neural", run_for("train"))
        path, document = self.build()
        self.assertEqual(len(document["samples"]), 3)
        source = document["sources"][0]
        self.assertEqual(source["source_kind"], "neural_movement")
        self.assertTrue(source["partial_character_run"])
        self.assertNotIn("old_log_probability", document["samples"][0])
        self.assertNotIn("reward", document["samples"][0])
        self.assertEqual(len(VisualReplayDataset(path).read_rgb(0)), 96 * 96 * 3)

    def test_recurrent_cycle_training_segments_are_discovered(self):
        collection = 'recurrent-cycles/cycle-1/training-run/segments'
        neural_episode(self.root, 'segment-1', run_for('train'), collection=collection)
        _, document = self.build()
        self.assertEqual(len(document['samples']), 3)
        self.assertEqual({row['session_id'] for row in document['samples']}, {'segment-1'})
        self.assertEqual(document['sources'][0]['source_kind'], 'neural_movement')

    def test_neural_evaluation_protects_its_entire_run_and_metadata_is_frozen(self):
        run = run_for("train")
        neural_episode(self.root, "evaluation", run, split="validation")
        neural_episode(self.root, "same-run-train", run, color=40)
        safe = neural_episode(self.root, "safe", run_for("train", 100), color=60)
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"safe"})
        write(safe / "frames/000000.json", {"sample_width": 8, "sample_height": 8, "capture_started_at_ns": 999})
        with self.assertRaisesRegex(ValueError, "neural frame metadata changed"):
            self.build("changed-neural.json")

    def test_full_run_evaluation_manifest_protects_child_sources_even_if_child_says_train(self):
        run = run_for("train")
        child = neural_episode(self.root, "evaluation-parent", run, split="train")
        neural_episode(self.root, "same-run-other-segment", run, color=40)
        neural_episode(self.root, "safe", run_for("train", 100), color=60)
        write(self.root / "artifacts/recurrent-cycles/cycle-1/evaluation-0-source-1/trajectory/manifest.json", {
            "schema": "playmodel.full-recurrent-run.v1", "split": "evaluation", "run_id": "full-run-identity",
            "session_ids": [child.name], "sources": [{"path": str(child / "frames/000000.bgra")}],
        })
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"safe"})
        self.assertIn("run:" + run, document["protected_groups"])

    def test_unknown_complete_run_excluded_without_deleting_original(self):
        original = episode(self.root, "unknown", None)
        episode(self.root, "known", run_for("train"), color=40)
        _, document = self.build()
        self.assertEqual(document["statistics"]["excluded_unknown_run_episodes"], 1)
        self.assertTrue((original / "frames/000000.bgra").exists())

    def test_source_mutation_detected_at_build_construction_and_frame_read(self):
        directory = episode(self.root, "s1", run_for("train"))
        path, document = self.build()
        dataset = VisualReplayDataset(path)
        frame_path = Path(document["samples"][0]["path"])
        raw = frame_path.read_bytes()
        frame_path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        with self.assertRaisesRegex(ValueError, "frame changed"):
            dataset.read_rgb(0)
        with self.assertRaisesRegex(ValueError, "frame changed"):
            self.build("changed.json")
        frame_path.write_bytes(raw)
        write(directory / "capture-config.json", {"train": False})
        with self.assertRaisesRegex(ValueError, "source metadata changed"):
            VisualReplayDataset(path)

    def test_frame_metadata_and_manifest_digest_are_checked(self):
        episode(self.root, "s1", run_for("train"))
        path, document = self.build()
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            VisualReplayDataset(path, expected_manifest_sha256="0" * 64)
        dataset = VisualReplayDataset(path)
        write(Path(document["samples"][0]["metadata_path"]), {"changed": True})
        with self.assertRaisesRegex(ValueError, "frame metadata changed"):
            dataset.read_rgb(0)

    def test_duplicate_pixels_cannot_cross_train_and_heldout(self):
        episode(self.root, "train", run_for("train"), count=1, color=20)
        episode(self.root, "test", run_for("test"), count=1, color=20)
        episode(self.root, "safe", run_for("train", 100), color=60)
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"safe"})

    def test_modified_split_and_group_contracts_are_rejected(self):
        episode(self.root, "s1", run_for("train"))
        path, document = self.build()
        document["samples"][0]["split"] = "test"
        write(path, document)
        with self.assertRaisesRegex(ValueError, "leakage"):
            VisualReplayDataset(path)

    def test_unused_snapshot_can_get_frozen_holdout_before_training(self):
        run = run_for("train")
        episode(self.root, "s1", run)
        path, _ = self.build("data/replay/unused.json")
        config = self.root / "configs/local/replay-group-splits.json"
        write(config, {"schema": "playmodel.visual-replay-group-splits.v1", "frozen_before_training": True,
                       "assignments": {"run:" + run: "validation"}})
        heldout, document = self.build("data/replay/frozen.json")
        self.assertEqual(document["split_counts"], {"validation": 3})
        self.assertEqual(len(VisualReplayDataset(heldout, split="validation")), 3)
        self.assertTrue(path.exists())

    def test_used_training_group_cannot_become_heldout_in_another_manifest(self):
        run = run_for("train")
        episode(self.root, "s1", run)
        path, _ = self.build("data/replay/used/manifest.json")
        usage = register_training_use(path, run_id="training-1")
        self.assertTrue(usage.exists())
        self.assertEqual(register_training_use(path, run_id="training-2"), usage)
        config = self.root / "configs/local/replay-group-splits.json"
        write(config, {"schema": "playmodel.visual-replay-group-splits.v1", "frozen_before_training": True,
                       "assignments": {"run:" + run: "test"}})
        with self.assertRaisesRegex(ValueError, "previously used replay group"):
            self.build("data/replay/reassigned.json")

    def test_sampling_and_loading_have_explicit_bounds(self):
        episode(self.root, "s1", run_for("train"), count=8)
        path, document = self.build(max_frames=3)
        self.assertEqual(len(document["samples"]), 3)
        with self.assertRaisesRegex(ValueError, "scan bound"):
            self.build("too-much.json", max_frames=2, max_source_frames=3)
        dataset = VisualReplayDataset(path, max_buffer_bytes=200)
        with self.assertRaisesRegex(ValueError, "memory bound"):
            dataset.read_rgb(0)
        dataset = VisualReplayDataset(path)
        with patch.object(dataset, "read_rgb", wraps=dataset.read_rgb) as read:
            iterator = dataset.iter_rgb(max_samples=1)
            self.assertEqual(read.call_count, 0)
            self.assertEqual(len(next(iterator)), 96 * 96 * 3)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(list(iterator), [])

    def test_optional_torch_output_is_cpu_uint8_chw(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional for replay preparation")
        episode(self.root, "s1", run_for("train"))
        path, _ = self.build()
        tensor = VisualReplayDataset(path)[0]
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(tuple(tensor.shape), (3, 96, 96))
        self.assertEqual(tensor.device.type, "cpu")

    def test_same_process_menu_collection_maps_to_previous_setup_for_exclusion(self):
        protected = episode(self.root, "live", run_for("train"), color=20)
        safe = episode(self.root, "safe", run_for("train", 100), color=60)
        setup = self.root / "artifacts/run-setup/20260927T010000Z-setup-a"
        setup_frame = setup / "20260927T010010Z-frame/frame.png"
        write(setup / "report.json", {"context": {"difficulty_menu_source": str(setup_frame)}})
        write(setup_frame.parent / "observation.json", {"pid": 123, "capture_started_at_ns": 1000})
        write(protected.parent.parent / "report.json", {
            "run_context": {"character_source": str(setup_frame)},
        })
        heldout_frame = self.root / "menu-collection/20260927T010011Z-frame/frame.png"
        write(heldout_frame.parent / "observation.json", {"pid": 123, "capture_started_at_ns": 1100})
        write(self.root / "data/menu/review/reviewed-manifest.json", {
            "samples": [{"session_id": "different-menu-session", "split": "validation",
                         "frame_path": str(heldout_frame)}],
        })
        _, document = self.build()
        self.assertEqual({s["session_id"] for s in document["samples"]}, {"safe"})
        self.assertIn("setup:20260927T010000Z-setup-a", document["protected_groups"])


if __name__ == "__main__":
    unittest.main()
