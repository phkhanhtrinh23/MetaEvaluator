"""Visualization helpers for shift descriptor analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA


def plot_pca(points: np.ndarray, labels: Iterable[str], output_path: Path) -> None:
    if points.shape[1] != 2:
        raise ValueError("Expect 2D PCA points.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=120)
    xs, ys = points[:, 0], points[:, 1]
    for x, y, label in zip(xs, ys, labels):
        plt.scatter(x, y, label=label, s=80)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)
    plt.title("Shift Descriptor PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _ensure_2d(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("Expected 2D array.")
    n_samples, dim = features.shape
    if dim == 2:
        return features
    if dim == 1:
        return np.hstack([features, np.zeros((n_samples, 1), dtype=features.dtype)])
    reducer = PCA(n_components=2)
    return reducer.fit_transform(features)


def plot_multi_model_scatter(
    model_embeddings: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot each model's embedding space separately but on shared figure axes."""

    if not model_embeddings:
        return

    models = sorted(model_embeddings.keys())
    n_models = len(models)
    cols = min(3, n_models)
    rows = int(np.ceil(n_models / cols))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=120)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes[n_models:]:
        ax.axis("off")

    for idx, model in enumerate(models):
        ax = axes[idx]
        splits = model_embeddings[model]
        train_emb = splits.get("train")
        test_emb = splits.get("test")
        if test_emb is None:
            test_emb = splits.get("dev")
        if train_emb is None and test_emb is None:
            ax.axis("off")
            continue
        available = [arr for arr in [train_emb, test_emb] if arr is not None and arr.size > 0]
        if not available:
            ax.axis("off")
            continue
        combined = np.vstack(available)
        projected = _ensure_2d(combined)
        offset = 0
        legends: List[str] = []
        if train_emb is not None and train_emb.size > 0:
            count = train_emb.shape[0]
            train_points = projected[offset : offset + count]
            offset += count
            ax.scatter(train_points[:, 0], train_points[:, 1], s=8, alpha=0.5, label="train")
            legends.append("train")
        if test_emb is not None and test_emb.size > 0:
            count = test_emb.shape[0]
            test_points = projected[offset : offset + count]
            offset += count
            ax.scatter(test_points[:, 0], test_points[:, 1], s=8, alpha=0.5, label="test", marker="x")
            legends.append("test")
        ax.set_title(model)
        ax.set_xlabel("Dim 1 (PCA)" if combined.shape[1] > 2 else "Dim 1")
        ax.set_ylabel("Dim 2 (PCA)" if combined.shape[1] > 2 else "Dim 2")
        ax.grid(True, linestyle="--", alpha=0.3)
        if legends:
            ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _gather_embedding_entries(
    model_embeddings: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[List[Tuple[str, str, np.ndarray]], int]:
    entries: List[Tuple[str, str, np.ndarray]] = []
    max_dim = 0
    for model, splits in model_embeddings.items():
        for split, arr in splits.items():
            if arr is None or arr.size == 0:
                continue
            entries.append((model, split, arr))
            max_dim = max(max_dim, arr.shape[1])
    return entries, max_dim


def plot_overlay_scatter(
    model_embeddings: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Overlay all models in a single PCA projection."""

    entries, max_dim = _gather_embedding_entries(model_embeddings)
    if not entries:
        return

    padded_chunks = []
    lengths = []
    for _, _, arr in entries:
        if arr.shape[1] < max_dim:
            pad_width = max_dim - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_width)))
        padded_chunks.append(arr)
        lengths.append(arr.shape[0])

    combined = np.vstack(padded_chunks)
    projected = _ensure_2d(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=120)

    models = sorted({model for model, _, _ in entries})
    color_map = {model: plt.get_cmap("tab10")(idx % 10) for idx, model in enumerate(models)}
    marker_map = {"train": "o", "test": "x", "dev": "x"}

    start = 0
    for (model, split, _), length in zip(entries, lengths):
        points = projected[start : start + length]
        start += length
        color = color_map[model]
        marker = marker_map.get(split, "o")
        label = f"{model} ({split})"
        plt.scatter(points[:, 0], points[:, 1], s=25, alpha=0.6, color=color, marker=marker, label=label)

    plt.title("Embedding distributions per model")
    plt.xlabel("Dim 1 (PCA)" if max_dim > 2 else "Dim 1")
    plt.ylabel("Dim 2 (PCA)" if max_dim > 2 else "Dim 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_cka_heatmap(matrix: np.ndarray, labels: List[str], output_path: Path) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("CKA heatmap expects a square matrix.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=120)
    norm = Normalize(vmin=0.0, vmax=1.0)
    plt.imshow(matrix, cmap="viridis", norm=norm)
    plt.colorbar(label="CKA similarity")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("CKA similarity across models")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_architecture_alignment(summary: Dict[str, Dict], output_path: Path) -> None:
    within = summary.get("within_family", {})
    between = summary.get("between_family", {})
    if not within and not between:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 2 if between else 1
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), dpi=120)
    if cols == 1:
        axes = [axes]

    if within:
        families = list(within.keys())
        means = [within[f]["mean_cka"] for f in families]
        x = np.arange(len(families))
        axes[0].bar(x, means, color="tab:blue")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Within-family mean CKA")
        axes[0].set_ylabel("CKA")
        axes[0].set_xticks(x, families, rotation=30, ha="right")
    else:
        axes[0].axis("off")

    if between:
        keys = list(between.keys())
        means = [between[k]["mean_cka"] for k in keys]
        x = np.arange(len(keys))
        axes[-1].bar(x, means, color="tab:orange")
        axes[-1].set_ylim(0, 1)
        axes[-1].set_title("Between-family mean CKA")
        axes[-1].set_ylabel("CKA")
        axes[-1].set_xticks(x, keys, rotation=30, ha="right")
    elif cols == 2:
        axes[-1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
