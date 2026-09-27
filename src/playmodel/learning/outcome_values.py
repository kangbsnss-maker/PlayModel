"""Small local outcome regressors; observational predictions, never causal labels."""
from __future__ import annotations
import hashlib
import math


class OutcomeValues:
    def __init__(self, state=None, *, dimensions=256):
        self.dimensions = dimensions
        self.weights = list((state or {}).get('weights', [0.] * dimensions))
        self.samples = int((state or {}).get('samples', 0))
        if len(self.weights) != dimensions or not all(math.isfinite(x) for x in self.weights):
            raise ValueError('Invalid local value checkpoint')

    def features(self, fields):
        features = [0.] * self.dimensions
        features[0] = 1.
        expanded = dict(fields)
        choice = fields.get('choice')
        if choice is not None:
            expanded.update({f'option:{choice}|{key}': value for key,value in fields.items()
                             if key != 'choice'})
        for key, value in sorted(expanded.items()):
            if value is None:
                continue
            token = f'{key}={value}' if not isinstance(value, (int, float)) else key
            fingerprint = hashlib.sha256(token.encode()).digest()
            index = 1 + int.from_bytes(fingerprint[:4], 'little') % (self.dimensions - 1)
            scale = math.tanh(float(value) / 20) if isinstance(value, (int, float)) else 1.
            features[index] += scale * (1 if fingerprint[4] & 1 else -1)
        norm = max(1., math.sqrt(sum(x*x for x in features)))
        return [x / norm for x in features]

    def predict(self, fields):
        features = self.features(fields)
        score = sum(x*w for x, w in zip(features, self.weights))
        return math.tanh(score)

    def fit(self, examples):
        for fields, target in examples:
            if not math.isfinite(target) or not -1 <= target <= 1:
                raise ValueError('Invalid verified outcome target')
            features = self.features(fields)
            prediction = self.predict(fields)
            gradient = (prediction - target) * (1 - prediction*prediction)
            self.weights = [w - .03 * (gradient*x + .0001*w)
                            for w, x in zip(self.weights, features)]
            self.samples += 1

    def state(self):
        return {'schema': 'playmodel.observational-values.v1', 'weights': self.weights,
                'samples': self.samples, 'semantics': 'observed_outcome_not_item_causal_effect'}
