"""Accuracy label loading for FusionSQL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping


class AccuracyLookupError(KeyError):
    """Raised when a label is missing for a particular model/split."""


@dataclass
class ModelAccuracySource:
    """Helper that loads per-model accuracy labels from a JSON mapping."""

    accuracy_map: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        *,
        key_candidates: tuple[str, ...] = ("model_accuracy", "labels", "metrics", "models"),
    ) -> "ModelAccuracySource":
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Accuracy label file not found: {json_path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            # Either the mapping lives directly at the root or under a known key.
            if all(isinstance(v, Mapping) for v in payload.values()):
                return cls(accuracy_map=payload)  # type: ignore[arg-type]
            for key in key_candidates:
                maybe = payload.get(key)
                if isinstance(maybe, Mapping):
                    return cls(accuracy_map=maybe)  # type: ignore[arg-type]

        raise ValueError(
            "Unable to locate a per-model accuracy mapping in "
            f"{json_path}. Expected a dict like {'model': {'meta_val': 0.7, ...}}."
        )

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

    def to_serializable(self) -> Dict[str, Dict[str, float]]:
        return {model: {split: float(val) for split, val in splits.items()} for model, splits in self.accuracy_map.items()}
