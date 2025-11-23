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


def _knn_degrees(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Approximate graph smoothness via kNN degrees (no edges stored)."""

    from scipy.spatial.distance import cdist

    dists = cdist(X, X)
    # ignore self
    np.fill_diagonal(dists, np.inf)
    knn_dists = np.partition(dists, kth=k, axis=1)[:, :k]
    # degree proxy: sum of neighbor distances
    return knn_dists.sum(axis=1)


def laplacian_sinkhorn_distance(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    epsilon: float = 0.1,
    n_iters: int = 100,
    laplace_alpha: float = 0.1,
    knn: int = 5,
) -> float:
    """Sinkhorn distance with a Laplacian-inspired smoothness prior."""

    a = np.asarray(vec_a, dtype=float).flatten()
    b = np.asarray(vec_b, dtype=float).flatten()
    n = max(len(a), len(b))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    a_weights = np.full(n, 1.0 / n)
    b_weights = np.full(n, 1.0 / n)

    a_deg = _knn_degrees(a[:, None], k=knn)
    b_deg = _knn_degrees(b[:, None], k=knn)

    base_cost = np.abs(a[:, None] - b[None, :])
    cost = base_cost + laplace_alpha * (a_deg[:, None] + b_deg[None, :])

    K = np.exp(-cost / epsilon)
    u = np.ones(n) / n
    v = np.ones(n) / n
    for _ in range(n_iters):
        u = a_weights / (K @ v + 1e-9)
        v = b_weights / (K.T @ u + 1e-9)
    transport = np.outer(u, v) * K
    return float((transport * cost).sum())


def sinkhorn_plan(
    Xs: np.ndarray,
    Xt: np.ndarray,
    reg: float = 0.05,
    num_iter: int = 200,
) -> np.ndarray:
    """Full Sinkhorn OT plan between two point clouds with uniform weights.

    - Xs: (n_s, d) source points
    - Xt: (n_t, d) target points
    Returns: transport plan gamma (n_s x n_t)
    """

    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    n_s = Xs.shape[0]
    n_t = Xt.shape[0]
    a = np.ones(n_s) / n_s
    b = np.ones(n_t) / n_t
    # Squared Euclidean cost
    C = np.sum(Xs**2, axis=1)[:, None] + np.sum(Xt**2, axis=1)[None, :] - 2 * Xs @ Xt.T
    K = np.exp(-C / reg) + 1e-9
    u = np.ones(n_s) / n_s
    v = np.ones(n_t) / n_t
    for _ in range(num_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
    gamma = np.diag(u) @ K @ np.diag(v)
    return gamma


def barycentric_mapping(gamma: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Barycentric mapping of source points using OT plan gamma onto target support."""

    gamma = np.asarray(gamma, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    row_sums = gamma.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (gamma @ Xt) / row_sums


def map_to_nearest_descriptor(
    target: np.ndarray,
    references: np.ndarray,
    ref_labels: np.ndarray,
    *,
    strategy: str = "emd",
    epsilon: float = 0.1,
    laplace_alpha: float = 0.1,
    knn: int = 5,
) -> tuple[float, int, float]:
    """Map target descriptor to the closest reference via selected OT distance."""

    distances = []
    for ref in references:
        if strategy == "sinkhorn":
            distances.append(sinkhorn_distance(target, ref, epsilon=epsilon))
        elif strategy == "laplace":
            distances.append(laplacian_sinkhorn_distance(target, ref, epsilon=epsilon, laplace_alpha=laplace_alpha, knn=knn))
        else:
            distances.append(emd_distance(target, ref))
    distances = np.array(distances)
    idx = int(np.argmin(distances))
    return float(ref_labels[idx]), idx, float(distances[idx])
