"""Offline learned menu perception; never an action policy or OCR cache.

Only explicitly reviewed labels may train this model. An approval sidecar bound
to the exact checkpoint is required for runtime predictions. These local review
records are provenance, not cryptographic attestations. Freshness, foreground
ownership, layout checks and the final action remain the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .capture import read_diagnostic_png
from .menu import BUTTONS

SCHEMA = 'brotato-menu-softmax-v1'
MANIFEST_SCHEMA = 'menu-model-reviewed-manifest-v1'
RESOLUTION = (1920, 1080)
# General scene samples plus button interiors and borders; no hand-coded labels.
_POINTS = [(int((x + .5) * 1920 / 24), int((y + .5) * 1080 / 14))
           for y in range(14) for x in range(24)]
for _scene in sorted(BUTTONS):
    for _name, (_l, _t, _r, _b) in sorted(BUTTONS[_scene].items()):
        _POINTS.extend([(int(_l + (_r - _l) * x), int(_t + (_b - _t) * y))
                        for x, y in ((.1, .1), (.5, .1), (.9, .1), (.1, .5),
                                     (.5, .5), (.9, .5), (.1, .9), (.5, .9), (.9, .9))])
        if _scene == 'difficulty':
            _POINTS.extend([(x, y) for x in (_l + 7, _r - 8)
                            for y in (_t + 24, (_t + _b) // 2, _b - 24)])
            _POINTS.extend([(x, y) for y in (_t + 7, _b - 8)
                            for x in (_l + 24, (_l + _r) // 2, _r - 24)])
FEATURE_POINTS = tuple(dict.fromkeys(_POINTS))
FEATURE_SCHEMA = hashlib.sha256(json.dumps(FEATURE_POINTS).encode()).hexdigest()
FEATURE_COUNT = len(FEATURE_POINTS) * 3


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _valid_label(label: str) -> bool:
    if not isinstance(label, str) or label.count(':') != 1:
        return False
    scene, focus = label.split(':')
    return focus in BUTTONS.get(scene, {})


def features(pixels: bytes, width: int, height: int) -> list[float]:
    if ((width, height) != RESOLUTION or type(width) is not int or type(height) is not int
            or not isinstance(pixels, bytes) or len(pixels) != width * height * 4):
        raise ValueError('Menu model requires a complete 1920x1080 BGRA frame')
    return [pixels[(y * width + x) * 4 + channel] / 127.5 - 1
            for x, y in FEATURE_POINTS for channel in range(3)]


def _probabilities(weights, biases, vector):
    scores = [sum(w * x for w, x in zip(row, vector)) + bias
              for row, bias in zip(weights, biases)]
    maximum = max(scores)
    values = [math.exp(score - maximum) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def _review_metrics_pass(evaluation, labels) -> bool:
    if not isinstance(evaluation, dict):
        return False
    for split in ('validation', 'test'):
        result = evaluation.get(split)
        if (not isinstance(result, dict) or type(result.get('samples')) is not int
                or result['samples'] < len(labels) or result.get('wrong_accepted') != 0):
            return False
        per_label = result.get('per_label')
        if not isinstance(per_label, dict):
            return False
        for label in labels:
            row = per_label.get(label)
            if (not isinstance(row, dict) or type(row.get('accepted')) is not int
                    or type(row.get('samples')) is not int or not 0 < row['accepted'] <= row['samples']
                    or row.get('wrong_accepted') != 0):
                return False
    return True


@dataclass(frozen=True)
class MenuPrediction:
    label: str | None
    scene: str | None
    focus: str | None
    confidence: float
    margin: float
    abstain_reason: str | None
    elapsed_ms: float


class MenuClassifier:
    def __init__(self, checkpoint: dict, *, approved: bool = False):
        self.checkpoint = checkpoint
        self.approved = approved
        if (not isinstance(checkpoint, dict) or checkpoint.get('schema') != SCHEMA
                or checkpoint.get('feature_schema') != FEATURE_SCHEMA
                or checkpoint.get('resolution') != list(RESOLUTION)):
            raise ValueError('Unsupported menu checkpoint schema or resolution')
        self.labels = checkpoint.get('labels', [])
        if (not isinstance(self.labels, list) or len(self.labels) < 2
                or not all(_valid_label(label) for label in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise ValueError('At least two valid scene:focus labels required')
        if not all(_nonempty(checkpoint.get(key)) for key in ('game_build_id', 'language', 'manifest_sha256')):
            raise ValueError('Checkpoint scope and manifest hash required')
        self.weights, self.biases = checkpoint.get('weights', []), checkpoint.get('biases', [])
        self.prototypes = checkpoint.get('prototypes', [])
        for matrix, limit in ((self.weights, 1e6), (self.prototypes, 1)):
            if (not isinstance(matrix, list) or len(matrix) != len(self.labels)
                    or any(not isinstance(row, list) or len(row) != FEATURE_COUNT for row in matrix)
                    or not all(_finite(v) and abs(v) <= limit for row in matrix for v in row)):
                raise ValueError('Invalid menu model matrix')
        if (not isinstance(self.biases, list) or len(self.biases) != len(self.labels)
                or not all(_finite(v) and abs(v) <= 1e6 for v in self.biases)):
            raise ValueError('Invalid menu model biases')
        self.thresholds = checkpoint.get('thresholds', {})
        if not isinstance(self.thresholds, dict):
            raise ValueError('Invalid abstention thresholds')
        for key in ('confidence', 'margin', 'max_distance'):
            value = self.thresholds.get(key)
            if not _finite(value) or not 0 < value <= 1:
                raise ValueError('Invalid abstention threshold')

    @classmethod
    def load(cls, path: Path, *, approval_path: Path | None = None) -> 'MenuClassifier':
        raw = path.read_bytes()
        checkpoint = json.loads(raw)
        model = cls(checkpoint)
        approved = False
        if approval_path is not None:
            approval = json.loads(approval_path.read_text(encoding='utf-8'))
            if not isinstance(approval, dict):
                raise ValueError('Invalid approval record')
            evaluation = checkpoint.get('evaluation', {})
            approved = (approval.get('schema') == 'menu-model-approval-v1'
                        and approval.get('checkpoint_sha256') == _sha(raw)
                        and approval.get('approved') is True
                        and all(_nonempty(approval.get(key)) for key in ('reviewer', 'reviewed_at', 'review_ref'))
                        and _review_metrics_pass(evaluation, checkpoint['labels']))
            if not approved:
                raise ValueError('Approval does not authorize this evaluated checkpoint')
        model.approved = approved
        return model

    def _predict_vector(self, vector, started) -> MenuPrediction:
        probabilities = _probabilities(self.weights, self.biases, vector)
        ranked = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
        winner = ranked[0]
        confidence = probabilities[winner]
        margin = confidence - probabilities[ranked[1]]
        # Softmax confidence alone cannot detect unknown images. This support
        # guard rejects large deviations, but does not guarantee OOD detection.
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, self.prototypes[winner])) / len(vector))
        reason = ('low_confidence' if confidence < self.thresholds['confidence'] else
                  'low_margin' if margin < self.thresholds['margin'] else
                  'outside_training_support' if distance > self.thresholds['max_distance'] else None)
        label = None if reason else self.labels[winner]
        scene, focus = label.split(':') if label else (None, None)
        return MenuPrediction(label, scene, focus, confidence, margin, reason,
                              (time.perf_counter() - started) * 1000)

    def predict(self, pixels: bytes, width: int, height: int, *, game_build_id: str,
                language: str, require_approved: bool = True) -> MenuPrediction:
        started = time.perf_counter()
        reason = ('unapproved_checkpoint' if require_approved and not self.approved else
                  'scope_mismatch' if (game_build_id != self.checkpoint['game_build_id']
                                      or language != self.checkpoint['language']) else None)
        if reason:
            return MenuPrediction(None, None, None, 0, 0, reason, (time.perf_counter() - started) * 1000)
        try:
            vector = features(pixels, width, height)
        except ValueError:
            return MenuPrediction(None, None, None, 0, 0, 'invalid_frame', (time.perf_counter() - started) * 1000)
        return self._predict_vector(vector, started)


def _load_manifest(path: Path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('resolution') != list(RESOLUTION)
            or not all(_nonempty(manifest.get(k)) for k in ('game_build_id', 'language'))):
        raise ValueError('Invalid reviewed menu manifest scope')
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Reviewed menu samples required; OCR is not a ground-truth label')
    splits, session_splits, frame_hashes = {k: [] for k in ('train', 'validation', 'test')}, {}, set()
    pixel_hashes = set()
    for sample in samples:
        provenance = sample.get('label_provenance', {})
        if (provenance.get('kind') not in ('human_review', 'developer_review')
                or provenance.get('reviewed') is not True
                or not all(_nonempty(provenance.get(k)) for k in ('reviewer', 'reviewed_at', 'review_ref'))):
            raise ValueError('Each label needs explicit review provenance; automatic OCR labels forbidden')
        session, split = sample.get('session_id'), sample.get('split')
        if not _nonempty(session) or split not in splits:
            raise ValueError('Every sample requires a session and fixed train/validation/test split')
        if session_splits.setdefault(session, split) != split:
            raise ValueError('Session leakage between dataset splits')
        label = f"{sample.get('scene')}:{sample.get('focus')}"
        if not _valid_label(label):
            raise ValueError('Unsupported scene:focus label')
        source = (path.parent / sample['frame_path']).resolve()
        frame_raw = source.read_bytes()
        digest = _sha(frame_raw)
        if digest != sample.get('frame_sha256'):
            raise ValueError('Frame SHA256 mismatch')
        if digest in frame_hashes:
            raise ValueError('Duplicate frame hash; duplicate source frames are not independent evidence')
        frame_hashes.add(digest)
        pixels, width, height = read_diagnostic_png(source)
        pixel_digest = _sha(pixels)
        if pixel_digest in pixel_hashes:
            raise ValueError('Duplicate decoded frame; container differences do not create independent evidence')
        pixel_hashes.add(pixel_digest)
        # Protect the separate decoder read from concurrent frame replacement.
        if _sha(source.read_bytes()) != digest:
            raise ValueError('Source frame changed during extraction')
        splits[split].append((features(pixels, width, height), label))
    labels = sorted({label for _, label in splits['train']})
    if len(labels) < 2:
        raise ValueError('Training requires at least two reviewed labels')
    for split, rows in splits.items():
        if {label for _, label in rows} != set(labels):
            raise ValueError(f'Every known label requires an independent session in {split}')
    return raw, manifest, splits, labels


def _evaluate(model: MenuClassifier, rows) -> dict:
    accepted = wrong = 0
    timings = []
    per_label = {label: {'samples': 0, 'accepted': 0, 'wrong_accepted': 0} for label in model.labels}
    for vector, label in rows:
        prediction = model._predict_vector(vector, time.perf_counter())
        timings.append(prediction.elapsed_ms)
        accepted += prediction.label is not None
        wrong += prediction.label is not None and prediction.label != label
        per_label[label]['samples'] += 1
        per_label[label]['accepted'] += prediction.label is not None
        per_label[label]['wrong_accepted'] += prediction.label is not None and prediction.label != label
    timings.sort()
    return {'samples': len(rows), 'accepted': accepted, 'wrong_accepted': wrong,
            'abstained': len(rows) - accepted, 'coverage': accepted / len(rows),
            'wrong_accept_rate': wrong / len(rows),
            'prediction_only_p50_ms': timings[(len(timings) - 1) // 2],
            'prediction_only_p95_ms': timings[math.ceil(len(timings) * .95) - 1],
            'per_label': per_label, 'unknown_screen_evaluation_performed': False,
            'timing_excludes_capture_decode_and_features': True}


def train_manifest(manifest_path: Path, output_path: Path, *, epochs: int = 80,
                   learning_rate: float = 1.0, seed: int = 0,
                   confidence: float = .90, margin: float = .20, max_distance: float = .12) -> dict:
    """Fit real softmax weights; write an unapproved, immutable candidate once.

    Fixed thresholds are declared before loading validation/test data. Evaluation
    does not tune them. Changing settings after inspecting test results requires
    a new evaluation set. Output files are exclusive-create, never overwritten.
    """
    if type(epochs) is not int or not 1 <= epochs <= 2000 or not _finite(learning_rate) or not 0 < learning_rate <= 10:
        raise ValueError('Invalid training settings')
    if output_path.exists():
        raise FileExistsError('Checkpoint already exists; choose a new candidate path')
    started = time.perf_counter()
    raw, manifest, splits, labels = _load_manifest(manifest_path)
    weights = [[0.0] * FEATURE_COUNT for _ in labels]
    biases = [0.0] * len(labels)
    class_counts = [sum(label == item for _, label in splits['train']) for item in labels]
    prototypes = [[sum(vector[i] for vector, label in splits['train'] if label == item) / class_counts[j]
                   for i in range(FEATURE_COUNT)] for j, item in enumerate(labels)]
    checkpoint = {'schema': SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'resolution': list(RESOLUTION),
                  'game_build_id': manifest['game_build_id'], 'language': manifest['language'],
                  'labels': labels, 'weights': weights, 'biases': biases, 'prototypes': prototypes,
                  'thresholds': {'confidence': confidence, 'margin': margin, 'max_distance': max_distance},
                  'manifest_sha256': _sha(raw), 'reviewed_manifest': manifest,
                  'training': {'epochs': epochs, 'learning_rate': learning_rate, 'seed': seed,
                               'algorithm': 'softmax_sgd_cross_entropy', 'policy_learning': False},
                  'created_at': datetime.now(timezone.utc).isoformat(), 'approved': False}
    model = MenuClassifier(checkpoint)
    training = [(vector, labels.index(label)) for vector, label in splits['train']]
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(training)
        for vector, target in training:
            probabilities = _probabilities(weights, biases, vector)
            scale = learning_rate / (1 + sum(value * value for value in vector))
            for j, probability in enumerate(probabilities):
                delta = scale * (probability - (j == target))
                weights[j] = [weight - delta * value for weight, value in zip(weights[j], vector)]
                biases[j] -= delta
    checkpoint['evaluation'] = {split: _evaluate(model, splits[split]) for split in ('validation', 'test')}
    checkpoint['training']['elapsed_seconds'] = time.perf_counter() - started
    if manifest_path.read_bytes() != raw:
        raise ValueError('Manifest changed during training; candidate not saved')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(checkpoint, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    with output_path.open('xb') as handle:
        handle.write(serialized)
    return {'checkpoint': str(output_path.resolve()), 'checkpoint_sha256': _sha(serialized),
            'manifest_sha256': _sha(raw), 'training_performed': True, 'approved': False,
            'runtime_deployed': False, 'evaluation': checkpoint['evaluation'],
            'training': checkpoint['training']}
