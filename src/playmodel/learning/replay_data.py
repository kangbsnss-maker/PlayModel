"""Immutable visual-only replay from preserved local training episodes.

Old actions, old log probabilities and old rewards are never dataset targets.
The output supports encoder reconstruction only; it is not a PPO rollout or a
behavior-cloning dataset.  Existing game evaluation runs and reviewed menu
validation/test runs are excluded, including matching complete-run groups.
New splits assess the encoder on prior training footage, not game performance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Iterator


SCHEMA = "playmodel.visual-replay.v1"
PURPOSE = "self_supervised_visual_reconstruction_only"
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
SPLIT_SALT = "playmodel-full-run-visual-split-v1"


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"Source exceeds byte bound: {path}")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024 if max_bytes is None else min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"Source exceeds byte bound: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def _document(path: Path) -> tuple[dict | list, str]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds byte bound: {path}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds byte bound: {path}")
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def _directories(root: Path, limit: int) -> list[Path]:
    if not root.exists():
        return []
    result = []
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= limit:
                raise ValueError(f"Discovery entry bound exceeded: {root}")
            if entry.is_dir(follow_symlinks=False):
                result.append(Path(entry.path))
    return sorted(result, key=lambda p: p.name)


def _child(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Source-relative path required")
    candidate = (directory / relative).resolve()
    if not candidate.is_relative_to(directory.resolve()):
        raise ValueError("Replay source escapes episode directory")
    return candidate


def _setup_group(path: str) -> str | None:
    parts = re.split(r"[\\/]", path)
    if "run-setup" in parts:
        position = parts.index("run-setup")
        if len(parts) > position + 1 and parts[position + 1]:
            return "setup:" + parts[position + 1]
    return None


def _context_group(context: dict) -> str | None:
    if not isinstance(context, dict):
        return None
    if isinstance(context.get("run_id"), str) and context["run_id"].strip():
        return "run:" + context["run_id"]
    return _setup_group(str(context.get("character_source", "")))


def split_for_group(group_id: str) -> str:
    """Fixed grouping independent of sampling seed and file enumeration order."""
    value = int(hashlib.sha256((SPLIT_SALT + "\0" + group_id).encode()).hexdigest()[:16], 16) % 10
    return "validation" if value == 8 else "test" if value == 9 else "train"


def register_training_use(manifest_path: Path, *, run_id: str) -> Path:
    """Freeze all group roles before the first optimizer step using this replay.

    A sidecar preserves usage without changing the immutable source manifest.
    Preparation/inspection alone does not mark unused footage as trained.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Training run ID required")
    manifest_path = manifest_path.resolve()
    document, digest = _document(manifest_path)
    if document.get("schema") != SCHEMA or document.get("purpose") != PURPOSE:
        raise ValueError("Unsupported visual replay contract")
    assignments = {}
    for sample in document["samples"]:
        group, split = sample["group_id"], sample["split"]
        if group in assignments and assignments[group] != split:
            raise ValueError("Replay group has multiple roles")
        assignments[group] = split
    usage_path = manifest_path.with_name(manifest_path.name + ".training-use.json")
    usage = {"schema": "playmodel.visual-replay-training-use.v1", "manifest_path": str(manifest_path),
             "manifest_sha256": digest, "first_training_run_id": run_id,
             "split_assignments": assignments}
    if usage_path.exists():
        old, _ = _document(usage_path)
        if old.get("manifest_sha256") != digest or old.get("split_assignments") != assignments:
            raise ValueError("Used replay identity or group roles changed")
        return usage_path
    with usage_path.open("x", encoding="utf-8") as stream:
        json.dump(usage, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return usage_path


def _training_locks(root: Path, limit: int):
    replay_root = root / "data/replay"
    locations = [replay_root, *_directories(replay_root, limit)]
    assignments, evidence = {}, []
    for directory in locations:
        if not directory.exists():
            continue
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    raise ValueError("Replay training-use discovery bound exceeded")
                if not entry.is_file() or not entry.name.endswith(".training-use.json"):
                    continue
                path = Path(entry.path)
                usage, digest = _document(path)
                if usage.get("schema") != "playmodel.visual-replay-training-use.v1":
                    raise ValueError("Invalid replay training-use record")
                if sha256_file(Path(usage["manifest_path"])) != usage["manifest_sha256"]:
                    raise ValueError("Previously used replay manifest changed")
                for group, split in usage["split_assignments"].items():
                    if group in assignments and assignments[group] != split:
                        raise ValueError("Previously used replay groups already conflict")
                    assignments[group] = split
                evidence.append({"path": str(path.resolve()), "sha256": digest, "role": "prior_training_split_lock"})
    return assignments, evidence


@dataclass(frozen=True)
class _Source:
    episode: Path
    session_id: str
    group_id: str | None
    group_report: Path | None
    config: Path
    train: bool
    kind: str = "legacy_movement"
    partial_run: bool = False


def _catalog(root: Path, max_episodes: int) -> tuple[list[_Source], dict[str, str | None]]:
    sources, session_groups = [], {}
    sessions = (_directories(root / "artifacts/brotato-sessions", max_episodes * 4)
                + _directories(root / "artifacts/neural-sessions", max_episodes * 4))
    for cycle in _directories(root / 'artifacts/recurrent-cycles', max_episodes * 4):
        for run in _directories(cycle, max_episodes):
            sessions.extend(_directories(run / 'segments', max_episodes * 4))
            if len(sessions) > max_episodes * 4:
                raise ValueError('Recurrent session discovery bound exceeded')
    for session in sessions:
        report_path = session / "report.json"
        report = _document(report_path)[0] if report_path.exists() else {}
        group = _context_group(report.get("run_context", {}))
        session_groups[session.name] = group
        for pilot in _directories(session / "pilots", max_episodes):
            session_groups[pilot.name] = group
            if (pilot / "capture-config.json").exists():
                config = _document(pilot / "capture-config.json")[0]
                if "split" in config and (pilot / "manifest.json").exists():
                    neural = _document(pilot / "manifest.json")[0]
                    if neural.get("schema") == "playmodel.neural-movement-rollout.v1":
                        # train=False means collection does not run an inline
                        # optimizer. The explicit data split decides eligibility.
                        sources.append(_Source(pilot / "actions.jsonl", session.name, group,
                                               report_path if report_path.exists() else None,
                                               pilot / "capture-config.json", config.get("split") == "train",
                                               "neural_movement",
                                               report.get("run_context", {}).get("experimental_partial_character_run") is True))
                        if len(sources) > max_episodes:
                            raise ValueError("Episode discovery bound exceeded")
                        continue
                if config.get("train") is True and not (pilot / "episode.json").exists():
                    continue
                sources.append(_Source(pilot / "episode.json", session.name, group,
                                       report_path if report_path.exists() else None,
                                       pilot / "capture-config.json", config.get("train") is True))
                if len(sources) > max_episodes:
                    raise ValueError("Episode discovery bound exceeded")
    for pilot in _directories(root / "artifacts/brotato-pilot", max_episodes):
        if (pilot / "capture-config.json").exists():
            report_path = pilot / "report.json"
            report = _document(report_path)[0] if report_path.exists() else {}
            config = _document(pilot / "capture-config.json")[0]
            if config.get("train") is True and not (pilot / "episode.json").exists():
                continue
            group = _context_group(report.get("run_context", {}))
            session_groups[pilot.name] = group
            sources.append(_Source(pilot / "episode.json", pilot.name, group,
                                   report_path if report_path.exists() else None,
                                   pilot / "capture-config.json", config.get("train") is True))
            if len(sources) > max_episodes:
                raise ValueError("Episode discovery bound exceeded")
    return sorted(sources, key=lambda s: str(s.episode)), session_groups


def _neural_records(path: Path):
    """Yield provenance only; never retain/relabel old actions or hidden states."""
    total = 0
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(2 * 1024 * 1024 + 1), b""):
            total += len(line)
            if total > MAX_JSON_BYTES or len(line) > 2 * 1024 * 1024:
                raise ValueError("Neural transmission ledger exceeds byte bound")
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("transmitted") is not True:
                raise ValueError("Neural replay frame lacks its recorded transmission provenance")
            yield {key: item[key] for key in ("frame_ref", "frame_sha256", "observed_at_ns", "available_at_ns")}


def _stamp(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = re.match(r"(\d{8}T\d{6}Z)", part)
        if match:
            return match[1]
    return None


def _protected(root: Path, session_groups: dict, extra: tuple[Path, ...], max_episodes: int):
    groups, sessions, hashes, evidence = set(), set(), set(), []
    paths = list(extra)
    for directory in _directories(root / "data/menu", 100):
        path = directory / "reviewed-manifest.json"
        if path.exists():
            paths.append(path)
    inspected_runs = 0
    for cycle in _directories(root / "artifacts/recurrent-cycles", max_episodes * 4):
        for run in _directories(cycle, max_episodes):
            inspected_runs += 1
            if inspected_runs > max_episodes * 4:
                raise ValueError("Full-run evaluation discovery bound exceeded")
            path = run / "trajectory/manifest.json"
            if path.exists():
                paths.append(path)
    # Setup timestamps are used only to exclude more footage conservatively.
    # This maps separately collected menu validation images back to their run.
    setups = []
    for directory in _directories(root / "artifacts/run-setup", max_episodes * 4):
        path = directory / "report.json"
        if not path.exists():
            continue
        report, report_sha = _document(path)
        context = report.get("context", {})
        source = context.get("difficulty_menu_source") or context.get("character_source")
        if not source or not _stamp(directory):
            continue
        observation = Path(source).parent / "observation.json"
        if observation.exists():
            meta, meta_sha = _document(observation)
            setups.append((_stamp(directory), meta.get("pid"), meta.get("capture_started_at_ns"),
                           "setup:" + directory.name, path, report_sha, observation, meta_sha))
    setups.sort(key=lambda item: item[0])
    for path in sorted(set(p.resolve() for p in paths)):
        document, digest = _document(path)
        evidence.append({"path": str(path), "sha256": digest, "role": "protected_manifest"})
        if not isinstance(document, dict):
            raise ValueError("Protected manifest must be an object")
        if (document.get("schema") == "playmodel.full-recurrent-run.v1"
                and document.get("split") in ("validation", "test", "evaluation", "heldout")):
            if document.get("run_id"):
                groups.add("run:" + document["run_id"])
            for session in document.get("session_ids", []):
                for identity in (session, *re.split(r"[\\/]", session)):
                    sessions.add(identity)
                    if session_groups.get(identity):
                        groups.add(session_groups[identity])
            start = document.get("start_evidence") or {}
            source_paths = [entry.get("path", "") for entry in document.get("sources", [])]
            source_paths += [start.get("setup_directory", ""), start.get("frame_ref", "")]
            for source in source_paths:
                direct_group = _setup_group(source)
                if direct_group:
                    groups.add(direct_group)
                for identity in re.split(r"[\\/]", source):
                    if session_groups.get(identity):
                        groups.add(session_groups[identity])
            continue
        for sample in document.get("samples", []):
            if sample.get("split") not in ("validation", "test", "evaluation", "heldout"):
                continue
            session = sample.get("session_id")
            if session:
                sessions.add(session)
                if session_groups.get(session):
                    groups.add(session_groups[session])
            if sample.get("group_id"):
                groups.add(sample["group_id"])
            hashes.update(sample[key] for key in ("frame_sha256", "decoded_bgra_sha256", "sha256") if sample.get(key))
            source = sample.get("frame_path") or sample.get("path")
            if not source:
                continue
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            direct_group = _setup_group(str(source_path))
            if direct_group:
                groups.add(direct_group)
            observation = source_path.parent / "observation.json"
            if observation.exists() and _stamp(source_path):
                meta, meta_sha = _document(observation)
                possible = [item for item in setups if item[0] <= _stamp(source_path)
                            and item[1] == meta.get("pid") and type(item[2]) is int
                            and item[2] <= meta.get("capture_started_at_ns", -1)]
                if possible:
                    selected = possible[-1]
                    groups.add(selected[3])
                    evidence.extend((
                        {"path": str(observation), "sha256": meta_sha, "role": "protected_menu_run_link"},
                        {"path": str(selected[4]), "sha256": selected[5], "role": "protected_setup_report"},
                        {"path": str(selected[6]), "sha256": selected[7], "role": "protected_setup_observation"},
                    ))
    return groups, sessions, hashes, list({entry["path"]: entry for entry in evidence}.values())


def build_visual_manifest(root: Path, output: Path, *, max_frames: int = 2000, seed: int = 0,
                          max_episodes: int = 1000, max_source_frames: int = 100_000,
                          protected_manifests: tuple[Path, ...] = (), group_splits: Path | None = None) -> dict:
    """Select old TRAIN pixels, check selected hashes, and exclusively create JSON.

    Directory discovery never descends into frame directories. Reservoir sampling
    retains at most max_frames lightweight references; only selected raw frames
    are read. Unknown full-run identity is excluded rather than split by wave.
    """
    if (type(max_frames) is not int or not 1 <= max_frames <= 20_000
            or type(max_episodes) is not int or not 1 <= max_episodes <= 10_000
            or type(max_source_frames) is not int or not max_frames <= max_source_frames <= 1_000_000
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid bounded replay preparation settings")
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Immutable replay manifest already exists: {output}")
    catalog, session_groups = _catalog(root, max_episodes)
    protected_groups, protected_sessions, protected_hashes, exclusions = _protected(
        root, session_groups, protected_manifests, max_episodes)
    assignments = {}
    group_splits = group_splits or root / "configs/local/replay-group-splits.json"
    if group_splits.exists():
        configuration, configuration_sha = _document(group_splits)
        if (configuration.get("schema") != "playmodel.visual-replay-group-splits.v1"
                or configuration.get("frozen_before_training") is not True
                or not isinstance(configuration.get("assignments"), dict)):
            raise ValueError("Invalid frozen replay group split configuration")
        assignments = configuration["assignments"]
        if any(not isinstance(group, str) or not group or role not in ("train", "validation", "test")
               for group, role in assignments.items()):
            raise ValueError("Invalid replay group role")
        exclusions.append({"path": str(group_splits.resolve()), "sha256": configuration_sha,
                           "role": "frozen_group_split_configuration"})
    locked_assignments, lock_evidence = _training_locks(root, max_episodes * 4)
    exclusions.extend(lock_evidence)
    for group, role in locked_assignments.items():
        proposed = assignments.get(group, split_for_group(group))
        if role != proposed:
            raise ValueError("Cannot reassign a previously used replay group")
    protected_groups.update(s.group_id for s in catalog if not s.train and s.group_id)
    rng, reservoir, seen, scan_count = random.Random(seed), [], 0, 0
    statistics = Counter()
    documents, hash_splits, cross_split_hashes = {}, {}, set()
    for source in catalog:
        if not source.train:
            statistics["excluded_not_training_episodes"] += 1
            continue
        if source.group_id is None:
            statistics["excluded_unknown_run_episodes"] += 1
            continue
        if source.group_id in protected_groups or source.session_id in protected_sessions:
            statistics["excluded_protected_run_episodes"] += 1
            continue
        directory = source.episode.parent
        config, config_sha = _document(source.config)
        if source.kind == "neural_movement":
            original_path = directory / "manifest.json"
            original, original_sha = _document(original_path)
            if original.get("recorder_complete") is not True:
                statistics["excluded_incomplete_episodes"] += 1
                continue
            originals = {entry["path"]: entry["sha256"] for entry in original["files"]}
            episode_sha = sha256_file(source.episode, max_bytes=MAX_JSON_BYTES)
            if episode_sha != originals.get("actions.jsonl"):
                raise ValueError("Original neural transmission ledger changed")
            if config_sha != originals.get("capture-config.json"):
                raise ValueError("Original neural capture configuration changed")
            episode = {"episode_id": directory.name, "policy_version": original["behavior_version"]}
            records = _neural_records(source.episode)
            frames = originals
            statistics["eligible_neural_training_episodes"] += 1
        else:
            episode, episode_sha = _document(source.episode)
            original_path = _child(directory, episode["manifest_ref"])
            original, original_sha = _document(original_path)
            if original_sha != episode["manifest_sha256"]:
                raise ValueError("Original episode manifest changed")
            if (original.get("recorder_complete") is not True or original.get("steps") != len(episode.get("steps", []))
                    or episode.get("truncated") is True or episode.get("guard_failure") is True):
                statistics["excluded_incomplete_episodes"] += 1
                continue
            originals = {entry["path"]: entry["sha256"] for entry in original["files"]}
            if config_sha != originals.get(episode["capture_config_ref"]):
                raise ValueError("Original capture configuration changed")
            frames = {entry["path"]: entry["sha256"] for entry in original["frames"]}
            records = episode["steps"]
        split = assignments.get(source.group_id, split_for_group(source.group_id))
        source_id = hashlib.sha256(str(source.episode).encode()).hexdigest()
        source_docs = [{"path": str(source.episode), "sha256": episode_sha,
                        "role": "transmission_ledger" if source.kind == "neural_movement" else "episode"},
                       {"path": str(original_path), "sha256": original_sha, "role": "original_manifest"},
                       {"path": str(source.config), "sha256": config_sha, "role": "capture_configuration"}]
        if source.group_report:
            source_docs.append({"path": str(source.group_report), "sha256": sha256_file(source.group_report),
                                "role": "full_run_group_report"})
        documents[source_id] = {"source_id": source_id, "episode_id": episode["episode_id"],
                                "session_id": source.session_id, "group_id": source.group_id,
                                "source_policy_version": episode.get("policy_version"), "train": True,
                                "source_kind": source.kind, "partial_character_run": source.partial_run,
                                "documents": source_docs}
        statistics["eligible_training_episodes"] += 1
        for index, step in enumerate(records):
            scan_count += 1
            if scan_count > max_source_frames:
                raise ValueError("Source frame-reference scan bound exceeded; raise explicit bound")
            frame_sha = step["frame_sha256"]
            if frames.get(step["frame_ref"]) != frame_sha:
                raise ValueError("Original step and frame manifest disagree")
            if frame_sha in protected_hashes:
                statistics["excluded_protected_frames"] += 1
                continue
            if frame_sha in hash_splits and hash_splits[frame_sha] != split:
                cross_split_hashes.add(frame_sha)
            hash_splits[frame_sha] = split
            record = {"source_id": source_id, "session_id": source.session_id,
                      "episode_id": episode["episode_id"], "group_id": source.group_id, "split": split,
                      "step_index": index, "path": str(_child(directory, step["frame_ref"])),
                      "sha256": frame_sha, "observed_at_ns": step["observed_at_ns"],
                      "available_at_ns": step["available_at_ns"], "source_kind": source.kind}
            if source.kind == "neural_movement":
                metadata_ref = str(Path(step["frame_ref"]).with_suffix(".json")).replace("\\", "/")
                if metadata_ref not in originals:
                    raise ValueError("Neural frame metadata absent from original manifest")
                record["original_metadata_sha256"] = originals[metadata_ref]
            seen += 1
            if len(reservoir) < max_frames:
                reservoir.append(record)
            else:
                position = rng.randrange(seen)
                if position < max_frames:
                    reservoir[position] = record
    selected, unique_hashes = [], set()
    for record in sorted(reservoir, key=lambda item: (item["source_id"], item["step_index"])):
        if record["sha256"] in cross_split_hashes or record["sha256"] in unique_hashes:
            statistics["excluded_duplicate_or_cross_split_frames"] += 1
            continue
        path = Path(record["path"])
        metadata_path = path.with_suffix(".json")
        metadata, metadata_sha = _document(metadata_path)
        neural = record["source_kind"] == "neural_movement"
        if neural and metadata_sha != record["original_metadata_sha256"]:
            raise ValueError("Original neural frame metadata changed")
        frame_metadata = metadata if neural else metadata["metadata"]
        width, height = (frame_metadata[key] for key in ("sample_width", "sample_height"))
        if (type(width) is not int or type(height) is not int or width < 1 or height < 1
                or width * height * 4 > MAX_FRAME_BYTES or path.stat().st_size != width * height * 4):
            raise ValueError("Invalid or oversized preserved BGRA frame")
        if (frame_metadata["capture_started_at_ns"] != record["observed_at_ns"]
                or (not neural and metadata["available_at_ns"] != record["available_at_ns"])
                or frame_metadata.get("capture_finished_at_ns", record["observed_at_ns"]) > record["available_at_ns"]
                or record["available_at_ns"] < record["observed_at_ns"]):
            raise ValueError("Frame provenance time mismatch")
        if sha256_file(path, max_bytes=MAX_FRAME_BYTES) != record["sha256"]:
            raise ValueError("Preserved frame changed")
        selected.append({**record, "width": width, "height": height, "format": "bgra_uint8",
                         "metadata_path": str(metadata_path), "metadata_sha256": metadata_sha})
        unique_hashes.add(record["sha256"])
    if not selected:
        raise ValueError("No eligible nonprotected training frames with known complete-run identity")
    selected_sources = {sample["source_id"] for sample in selected}
    document = {"schema": SCHEMA, "purpose": PURPOSE, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root), "split_contract": SPLIT_SALT,
                "group_split_overrides": assignments,
                "input_contract": "CPU uint8 RGB; old actions/rewards excluded; not PPO or BC",
                "evaluation_scope": "encoder reconstruction on prior training runs, not unseen gameplay evaluation",
                "unknown_run_policy": "exclude", "max_frames": max_frames, "sampling_seed": seed,
                "max_source_frames": max_source_frames, "source_frames_considered": scan_count,
                "protected_groups": sorted(protected_groups), "protected_sessions": sorted(protected_sessions),
                "protection_sources": exclusions, "statistics": dict(statistics),
                "split_counts": dict(Counter(sample["split"] for sample in selected)),
                "sources": [documents[key] for key in sorted(selected_sources)], "samples": selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return document


class VisualReplayDataset:
    """One-frame-at-a-time CPU dataset; torch import occurs only in __getitem__.

    No image cache, GPU allocation or prefetch thread. With a zero-worker
    DataLoader, peak owned image storage is one <=16 MiB BGRA frame plus RGB96.
    Source contracts are checked at construction; frame and sidecar are checked
    each read. Instantiate again for a new training execution.
    """
    def __init__(self, manifest_path: Path, *, split: str = "train", image_size: int = 96, size: int | None = None,
                 expected_manifest_sha256: str | None = None, max_buffer_bytes: int = MAX_FRAME_BYTES):
        if size is not None:
            if image_size != 96 and image_size != size:
                raise ValueError("Conflicting replay image sizes")
            image_size = size
        if split not in ("train", "validation", "test") or type(image_size) is not int or not 8 <= image_size <= 256:
            raise ValueError("Invalid replay split or image size")
        if type(max_buffer_bytes) is not int or not 4 <= max_buffer_bytes <= MAX_FRAME_BYTES:
            raise ValueError("Invalid frame memory bound")
        document, digest = _document(Path(manifest_path))
        if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
            raise ValueError("Replay manifest changed")
        if document.get("schema") != SCHEMA or document.get("purpose") != PURPOSE or document.get("split_contract") != SPLIT_SALT:
            raise ValueError("Unsupported visual replay contract")
        samples = document["samples"]
        groups, hashes = {}, {}
        overrides = document.get("group_split_overrides", {})
        protected = set(document.get("protected_groups", []))
        sources = {source["source_id"]: source for source in document["sources"]}
        for sample in samples:
            group, role, digest = sample["group_id"], sample["split"], sample["sha256"]
            if (sample.get("format") != "bgra_uint8" or type(sample.get("width")) is not int
                    or type(sample.get("height")) is not int or sample["width"] < 1 or sample["height"] < 1
                    or sample["width"] * sample["height"] * 4 > MAX_FRAME_BYTES):
                raise ValueError("Invalid replay frame dimensions")
            if (role != overrides.get(group, split_for_group(group)) or group in protected
                    or group in groups and groups[group] != role or digest in hashes and hashes[digest] != role):
                raise ValueError("Replay training/heldout leakage")
            if sample["source_id"] not in sources or sources[sample["source_id"]].get("train") is not True:
                raise ValueError("Replay source is not a training episode")
            if sources[sample["source_id"]]["group_id"] != group:
                raise ValueError("Replay group differs from source")
            groups[group], hashes[digest] = role, role
        self.samples = tuple(sample for sample in samples if sample["split"] == split)
        relevant = {sample["source_id"] for sample in self.samples}
        for source in document["sources"]:
            if source["source_id"] in relevant:
                for item in source["documents"]:
                    if sha256_file(Path(item["path"]), max_bytes=MAX_JSON_BYTES) != item["sha256"]:
                        raise ValueError("Replay source metadata changed")
        for item in document.get("protection_sources", []):
            if sha256_file(Path(item["path"]), max_bytes=MAX_JSON_BYTES) != item["sha256"]:
                raise ValueError("Replay holdout protection source changed")
        self.image_size, self.max_buffer_bytes = image_size, max_buffer_bytes
        self.manifest_sha256 = sha256_file(Path(manifest_path))
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def read_rgb(self, index: int) -> bytes:
        """Return HWC uint8 RGB bytes using deterministic nearest-neighbor resize."""
        sample = self.samples[index]
        width, height = sample["width"], sample["height"]
        expected = width * height * 4
        path = Path(sample["path"])
        if expected > self.max_buffer_bytes or path.stat().st_size != expected:
            raise ValueError("Replay frame exceeds memory bound or changed size")
        if sha256_file(Path(sample["metadata_path"]), max_bytes=MAX_JSON_BYTES) != sample["metadata_sha256"]:
            raise ValueError("Replay frame metadata changed")
        with path.open("rb") as stream:
            raw = stream.read(expected + 1)
        if len(raw) != expected:
            raise ValueError("Replay frame changed size while reading")
        if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
            raise ValueError("Replay frame changed")
        size = self.image_size
        rgb = bytearray(size * size * 3)
        for y in range(size):
            source_y = min(height - 1, ((2 * y + 1) * height) // (2 * size))
            for x in range(size):
                source_x = min(width - 1, ((2 * x + 1) * width) // (2 * size))
                source_offset, target = (source_y * width + source_x) * 4, (y * size + x) * 3
                rgb[target:target + 3] = raw[source_offset:source_offset + 3][::-1]
        return bytes(rgb)

    def __getitem__(self, index: int):
        import torch
        return torch.frombuffer(bytearray(self.read_rgb(index)), dtype=torch.uint8).reshape(
            self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()

    def iter_rgb(self, *, max_samples: int | None = None) -> Iterator[bytes]:
        count = len(self) if max_samples is None else min(len(self), max_samples)
        if count < 0:
            raise ValueError("Negative sample bound")
        for index in range(count):
            yield self.read_rgb(index)
