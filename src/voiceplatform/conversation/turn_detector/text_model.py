"""A small, data-only Vietnamese turn completion model.

The training script exports TF-IDF character features and logistic weights as
JSON. Loading JSON rather than a pickle keeps a model file from executing code
inside the voice server. Inference has no scikit-learn dependency.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path


class TextTurnModel:
    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != 1 or payload.get("analyzer") != "char_tfidf_logistic":
            raise ValueError("unsupported turn model schema")
        ngrams = payload.get("ngram_range")
        if ngrams != [2, 4]:
            raise ValueError("unsupported turn model ngram range")
        features = payload.get("features")
        if not isinstance(features, dict) or not features or len(features) > 50000:
            raise ValueError("invalid turn model features")
        self.features: dict[str, tuple[float, float]] = {}
        for gram, values in features.items():
            if (not isinstance(gram, str) or not 2 <= len(gram) <= 4
                    or not isinstance(values, list) or len(values) != 2
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                    or values[0] <= 0):
                raise ValueError("invalid turn model feature")
            self.features[gram] = (float(values[0]), float(values[1]))
        self.intercept = float(payload.get("intercept", float("nan")))
        if not math.isfinite(self.intercept):
            raise ValueError("invalid turn model intercept")

    def score(self, text: str) -> float:
        source = text.lower()[:512]
        counts = Counter(source[i:i+n] for n in (2, 3, 4)
                         for i in range(max(0, len(source)-n+1)))
        weights = [(count * self.features[gram][0], self.features[gram][1])
                   for gram, count in counts.items() if gram in self.features]
        norm = math.sqrt(sum(value * value for value, _ in weights))
        raw = self.intercept + (sum(value * coef for value, coef in weights) / norm if norm else 0.0)
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, raw))))


@lru_cache(maxsize=4)
def load_text_turn_model(path: str) -> TextTurnModel:
    file = Path(path)
    if file.stat().st_size > 8_000_000:
        raise ValueError("turn model exceeds 8 MB")
    return TextTurnModel(json.loads(file.read_text(encoding="utf-8")))
