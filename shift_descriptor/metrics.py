"""Shift descriptor metrics: Fréchet, Mahalanobis, and Sliced Wasserstein."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


EPS = 1e-6


@dataclass
class DistributionStats:
    mean: np.ndarray
    cov: np.ndarray


def sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be of shape (n, d).")
    if embeddings.size == 0:
        raise ValueError("Empty embedding array.")
    if np.isfinite(embeddings).all():
        return embeddings
    mask = np.all(np.isfinite(embeddings), axis=1)
    filtered = embeddings[mask]
    if filtered.size == 0:
        raise ValueError("All embeddings contain NaN or inf values.")
    return filtered


def compute_stats(embeddings: np.ndarray) -> DistributionStats:
    embeddings = sanitize_embeddings(embeddings)
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov += np.eye(cov.shape[0]) * EPS
    return DistributionStats(mean=mean, cov=cov)


def frechet_distance(stats_a: DistributionStats, stats_b: DistributionStats) -> float:
    diff = stats_a.mean - stats_b.mean
    cov_prod = stats_a.cov @ stats_b.cov
    sqrt_cov_prod = linalg.sqrtm(cov_prod)
    if np.iscomplexobj(sqrt_cov_prod):
        sqrt_cov_prod = sqrt_cov_prod.real
    trace_term = np.trace(stats_a.cov + stats_b.cov - 2 * sqrt_cov_prod)
    return float(diff @ diff + trace_term)


def mahalanobis_distance(stats_a: DistributionStats, stats_b: DistributionStats) -> float:
    diff = stats_a.mean - stats_b.mean
    pooled = 0.5 * (stats_a.cov + stats_b.cov)
    inv = np.linalg.pinv(pooled)
    dist = diff @ inv @ diff
    return float(np.sqrt(max(dist, 0.0)))


def sliced_wasserstein_distance(
    emb_a: np.ndarray, emb_b: np.ndarray, num_projections: int = 128, seed: int = 13
) -> float:
    if emb_a.shape[1] != emb_b.shape[1]:
        raise ValueError("Embedding dimensions must match.")
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(num_projections, emb_a.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    distances = []
    for direction in projections:
        proj_a = emb_a @ direction
        proj_b = emb_b @ direction
        distances.append(wasserstein_distance(proj_a, proj_b))
    return float(np.mean(distances))


def build_descriptor_matrix(metric_records: Dict[str, Dict[str, float]], metric_order: Iterable[str]) -> np.ndarray:
    rows = []
    for model_name in metric_records:
        rows.append([metric_records[model_name][metric] for metric in metric_order])
    return np.array(rows)


def compute_pairwise_similarities(matrix: np.ndarray, labels: Iterable[str]) -> Dict[Tuple[str, str], float]:
    label_list = list(labels)
    sims: Dict[Tuple[str, str], float] = {}
    for i, j in itertools.combinations(range(len(label_list)), 2):
        vec_i = matrix[i]
        vec_j = matrix[j]
        sims[(label_list[i], label_list[j])] = 1.0 - cosine(vec_i, vec_j)
    return sims


def pca_project(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    if matrix.shape[0] < n_components:
        raise ValueError("Need at least as many samples as PCA components.")
    return PCA(n_components=n_components).fit_transform(matrix)
