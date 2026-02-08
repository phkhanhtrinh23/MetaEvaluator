"""Accuracy label loading for MetaEvaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class AccuracyLookupError(KeyError):
    """Raised when a label is missing for a particular model/split."""


@dataclass
class ModelAccuracySource:
    """Helper that loads per-model accuracy labels from a JSON mapping."""

    accuracy_map: Mapping[str, Mapping[str, float]]

    def get(self, model_name: str, split_name: str) -> float:
        try:
            split_scores = self.accuracy_map[model_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AccuracyLookupError(f"Missing accuracy entry for model '{model_name}'.") from exc

        try:
            return float(split_scores[split_name])
        except KeyError as exc:  # pragma: no cover - defensive
            raise AccuracyLookupError(
                f"Missing accuracy entry for model '{model_name}' split '{split_name}'."
            ) from exc
