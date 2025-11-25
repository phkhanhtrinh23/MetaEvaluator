"""Lightweight optimal transport helpers for mapping shift descriptors."""

from __future__ import annotations

import numpy as np


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
