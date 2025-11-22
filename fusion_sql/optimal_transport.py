"""Lightweight optimal transport helpers for mapping shift descriptors."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def emd_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Approximate Earth Mover's Distance between two equal-length vectors.

    Treat each feature as a unit mass at its value; compute the minimal-cost
    assignment between the supports using L1 cost.
    """

    a = np.asarray(vec_a, dtype=float).flatten()
    b = np.asarray(vec_b, dtype=float).flatten()
    n = max(len(a), len(b))
    # Pad shorter vector with zeros to allow assignment
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    cost = np.abs(a[:, None] - b[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum() / n)


def sinkhorn_distance(vec_a: np.ndarray, vec_b: np.ndarray, epsilon: float = 0.1, n_iters: int = 100) -> float:
    """Entropy-regularized OT (Sinkhorn) between two histograms with uniform weights."""

    a = np.asarray(vec_a, dtype=float).flatten()
    b = np.asarray(vec_b, dtype=float).flatten()
    n = max(len(a), len(b))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    a_weights = np.full(n, 1.0 / n)
    b_weights = np.full(n, 1.0 / n)
    cost = np.abs(a[:, None] - b[None, :])
    K = np.exp(-cost / epsilon)
    u = np.ones(n) / n
    v = np.ones(n) / n
    for _ in range(n_iters):
        u = a_weights / (K @ v + 1e-9)
        v = b_weights / (K.T @ u + 1e-9)
    transport = np.outer(u, v) * K
    return float((transport * cost).sum())


def map_to_nearest_descriptor(
    target: np.ndarray,
    references: np.ndarray,
    ref_labels: np.ndarray,
    *,
    strategy: str = "emd",
    epsilon: float = 0.1,
) -> tuple[float, int, float]:
    """Map target descriptor to the closest reference via selected OT distance."""

    distances = []
    for ref in references:
        if strategy == "sinkhorn":
            distances.append(sinkhorn_distance(target, ref, epsilon=epsilon))
        else:
            distances.append(emd_distance(target, ref))
    distances = np.array(distances)
    idx = int(np.argmin(distances))
    return float(ref_labels[idx]), idx, float(distances[idx])
