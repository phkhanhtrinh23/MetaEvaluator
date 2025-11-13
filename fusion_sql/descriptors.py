"""Shift descriptor helpers tailored for FusionSQL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from shift_descriptor.metrics import (
    DistributionStats,
    compute_stats,
    frechet_distance,
    mahalanobis_distance,
    sliced_wasserstein_distance,
)


DEFAULT_FEATURE_ORDER: Sequence[str] = (
    "frechet_distance",
    "frechet_mean_shift",
    "mahalanobis_distance",
    "swd_mean",
    "swd_std",
    "swd_max",
)


@dataclass(frozen=True)
class ShiftDescriptor:
    """Stores shift descriptor metadata for a pair of splits."""

    model_name: str
    split_a: str
    split_b: str
    features: Dict[str, float]

    def as_tensor(self, feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER, device: torch.device | None = None) -> torch.Tensor:
        vec = [self.features[feature] for feature in feature_order]
        return torch.tensor(vec, dtype=torch.float32, device=device)


def _maybe_compute_stats(cache: Dict[int, DistributionStats], emb: np.ndarray) -> DistributionStats:
    cache_key = id(emb)
    if cache_key not in cache:
        cache[cache_key] = compute_stats(emb)
    return cache[cache_key]


def compute_shift_descriptor(
    model_name: str,
    split_a: str,
    emb_a: np.ndarray,
    split_b: str,
    emb_b: np.ndarray,
    *,
    num_projections: int = 128,
    seed: int = 13,
    feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
) -> ShiftDescriptor:
    """Compute the requested shift descriptor features between two embedding clouds."""

    stats_cache: Dict[int, DistributionStats] = {}
    stats_a = _maybe_compute_stats(stats_cache, emb_a)
    stats_b = _maybe_compute_stats(stats_cache, emb_b)

    frechet = frechet_distance(stats_a, stats_b)
    mahala = mahalanobis_distance(stats_a, stats_b)
    swd = sliced_wasserstein_distance(emb_a, emb_b, num_projections=num_projections, seed=seed)

    features = {
        "frechet_distance": frechet["frechet_distance"],
        "frechet_mean_shift": frechet["frechet_mean_shift"],
        "mahalanobis_distance": mahala,
        "swd_mean": swd["swd_mean"],
        "swd_std": swd["swd_std"],
        "swd_max": swd["swd_max"],
    }
    # Validate feature order early.
    for feature_name in feature_order:
        if feature_name not in features:
            raise KeyError(f"Feature '{feature_name}' missing from computed descriptor.")

    return ShiftDescriptor(model_name=model_name, split_a=split_a, split_b=split_b, features=features)


def stack_descriptors(
    descriptors: Iterable[ShiftDescriptor],
    feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
    device: torch.device | None = None,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for descriptor in descriptors:
        rows.append(descriptor.as_tensor(feature_order=feature_order, device=device))
    return torch.stack(rows, dim=0)
