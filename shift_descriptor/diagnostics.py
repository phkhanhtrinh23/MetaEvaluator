"""Diagnostic utilities for shift descriptor pipeline."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes


def match_dimensions(arr_a: np.ndarray, arr_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the smaller feature dimension so both arrays share the same width."""
    dim = max(arr_a.shape[1], arr_b.shape[1])
    if arr_a.shape[1] < dim:
        pad = np.zeros((arr_a.shape[0], dim - arr_a.shape[1]), dtype=arr_a.dtype)
        arr_a = np.hstack([arr_a, pad])
    if arr_b.shape[1] < dim:
        pad = np.zeros((arr_b.shape[0], dim - arr_b.shape[1]), dtype=arr_b.dtype)
        arr_b = np.hstack([arr_b, pad])
    return arr_a, arr_b


def match_pair_samples(
    arr_a: np.ndarray, arr_b: np.ndarray, rng: np.random.Generator, min_rows: int = 2
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Downsample two arrays so they share the same number of rows."""
    rows = min(arr_a.shape[0], arr_b.shape[0])
    if rows < min_rows:
        return None, None
    if arr_a.shape[0] > rows:
        idx = rng.choice(arr_a.shape[0], size=rows, replace=False)
        arr_a = arr_a[idx]
    if arr_b.shape[0] > rows:
        idx = rng.choice(arr_b.shape[0], size=rows, replace=False)
        arr_b = arr_b[idx]
    return arr_a, arr_b


def build_normalization_view(
    analysis_embeddings: Dict[str, Dict[str, Any]], mode: str
) -> Dict[str, np.ndarray]:
    """Construct a normalization view across models."""
    if not analysis_embeddings:
        return {}

    available: Dict[str, np.ndarray] = {}
    max_dim = 0
    for name, info in analysis_embeddings.items():
        arr = info["combined"]
        if arr.size == 0:
            continue
        available[name] = arr
        max_dim = max(max_dim, arr.shape[1])
    if not available:
        return {}

    if mode == "per_model":
        return dict(available)

    def pad(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] == max_dim:
            return arr
        pad_width = max_dim - arr.shape[1]
        return np.pad(arr, ((0, 0), (0, pad_width)))

    padded = {name: pad(arr) for name, arr in available.items()}
    combined = np.vstack(list(padded.values()))
    mean = combined.mean(axis=0, keepdims=True)

    if mode == "global_l2":
        views: Dict[str, np.ndarray] = {}
        for name, arr in padded.items():
            centered = arr - mean
            norm = np.linalg.norm(centered, axis=1, keepdims=True)
            norm = np.clip(norm, 1e-6, None)
            views[name] = centered / norm
        return views

    if mode == "global_zscore":
        std = combined.std(axis=0, keepdims=True)
        std = np.clip(std, 1e-6, None)
        views: Dict[str, np.ndarray] = {}
        for name, arr in padded.items():
            views[name] = (arr - mean) / std
        return views

    raise ValueError(f"Unknown normalization mode: {mode}")


def view_to_scatter_payload(view: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    payload: Dict[str, Dict[str, np.ndarray]] = {}
    for name, arr in view.items():
        payload[name] = {
            "train": arr,
            "test": np.empty((0, arr.shape[1]), dtype=arr.dtype),
        }
    return payload


def _sampling_limit(size: int, limit: int | None) -> int:
    if limit is None or limit <= 0:
        return size
    return min(size, limit)


def sample_cosine_hist(
    view: Dict[str, np.ndarray],
    rng: np.random.Generator,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    within = []
    between = []
    pair_means: Dict[str, float] = {}
    model_names = list(view.keys())

    for name, arr in view.items():
        rows = _sampling_limit(arr.shape[0], max_samples)
        if rows < 2:
            continue
        idx = rng.choice(arr.shape[0], size=rows, replace=False)
        sample = arr[idx]
        sample = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-6)
        dots = sample @ sample.T
        mask = np.triu(np.ones_like(dots, dtype=bool), k=1)
        within.extend(dots[mask].tolist())

    for i, j in combinations(range(len(model_names)), 2):
        arr_i = view[model_names[i]]
        arr_j = view[model_names[j]]
        rows = min(
            _sampling_limit(arr_i.shape[0], max_samples),
            _sampling_limit(arr_j.shape[0], max_samples),
        )
        if rows < 2:
            continue
        idx_i = rng.choice(arr_i.shape[0], size=rows, replace=False)
        idx_j = rng.choice(arr_j.shape[0], size=rows, replace=False)
        samp_i = arr_i[idx_i]
        samp_j = arr_j[idx_j]
        samp_i, samp_j = match_dimensions(samp_i, samp_j)
        samp_i = samp_i / (np.linalg.norm(samp_i, axis=1, keepdims=True) + 1e-6)
        samp_j = samp_j / (np.linalg.norm(samp_j, axis=1, keepdims=True) + 1e-6)
        pair_vals = (samp_i * samp_j).sum(axis=1)
        between.extend(pair_vals.tolist())
        pair_key = "__".join(sorted((model_names[i], model_names[j])))
        pair_means[pair_key] = float(np.mean(pair_vals))

    return np.array(within), np.array(between), pair_means


def plot_cosine_histogram(
    within: np.ndarray,
    between: np.ndarray,
    mode: str,
    output_path: Path,
    num_bins: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    plt.figure(figsize=(6, 4), dpi=120)
    if within.size:
        plt.hist(within, bins=bins, alpha=0.6, label="within-model", density=True)
    if between.size:
        plt.hist(between, bins=bins, alpha=0.6, label="between-model", density=True)
    plt.title(f"Cosine similarity histogram ({mode})")
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pairwise_cosines(pair_means: Dict[str, float], output_path: Path) -> bool:
    if not pair_means:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(pair_means.keys())
    values = [pair_means[k] for k in labels]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.4)), dpi=120)
    ax.barh(labels, values, color="tab:purple")
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val
    if span < 1e-9 and abs(max_val) < 1e-6:
        plt.close(fig)
        if output_path.exists():
            output_path.unlink()
        return False
    pad = max(1e-3, span * 0.15 if span > 0 else abs(max_val) * 0.15)
    left = min_val - pad
    right = max_val + pad
    if left == right:
        left -= 1e-3
        right += 1e-3
    ax.set_xlim(left, right)
    ax.set_xlabel("Mean cosine similarity")
    ax.set_title("Pairwise cosine similarity (per_model normalization)", pad=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)
    return True


def compute_procrustes(view: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    names = list(view.keys())
    for a, b in combinations(names, 2):
        arr_a, arr_b = match_pair_samples(view[a], view[b], rng)
        if arr_a is None:
            continue
        try:
            arr_a, arr_b = match_dimensions(arr_a, arr_b)
            r, scale = orthogonal_procrustes(arr_a, arr_b)
            recon = scale * arr_a @ r
            residual = np.linalg.norm(recon - arr_b, ord="fro") / max(1, arr_a.shape[0])
        except Exception:
            continue
        distances["__".join(sorted((a, b)))] = float(residual)
    return distances


def plot_mean_shift_anisotropy(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    if not metrics:
        return

    names = list(metrics.keys())
    mean_shifts = [metrics[n].get("mean_shift_norm", 0.0) for n in names]
    anisotropies = [metrics[n].get("train_anisotropy", 0.0) for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    axes[0].bar(x, mean_shifts, color="tab:green")
    axes[0].set_title("Mean shift norm")
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].set_ylabel("||μ_train - μ_test||")

    axes[1].bar(x, anisotropies, color="tab:red")
    axes[1].set_title("Train anisotropy (λ_max / λ_min)")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].set_ylabel("anisotropy")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
