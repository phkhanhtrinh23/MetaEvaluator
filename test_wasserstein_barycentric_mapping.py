import numpy as np
import ot  # pip install POT

def wasserstein_barycentric_map(X, a, Y, b, reg=1e-2):
    """
    X: (n,d) source descriptors
    a: (n,)  source masses, sum=1
    Y: (m,d) target descriptors
    b: (m,)  target masses, sum=1
    reg: entropic regularization (Sinkhorn). Smaller -> closer to exact OT.
    Returns:
      T: (n,d) mapped points T_WB(x_i)
      P: (n,m) transport plan pi
      W: (n,m) barycentric weights (row-normalized P)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    # 1) Cost matrix C_ij = ||x_i - y_j||^2
    C = ot.dist(X, Y, metric="sqeuclidean")

    # 2) Solve for transport plan pi (Sinkhorn is fast + stable)
    P = ot.sinkhorn(a, b, C, reg=reg)   # shape (n,m)

    # 3) WB weights = conditional distribution of target given source
    #    w_ij = pi_ij / a_i
    W = P / a[:, None]

    # 4) Barycentric projection: T_i = sum_j w_ij * y_j
    T = W @ Y
    return T, P, W

# ---- Example (same style as our worked example) ----
X = np.array([[0.10, 0.20],
              [0.90, 0.80]])
a = np.array([0.6, 0.4])

Y = np.array([[0.00, 0.00],
              [1.00, 1.00],
              [0.80, 0.20]])
b = np.array([0.2, 0.5, 0.3])

T, P, W = wasserstein_barycentric_map(X, a, Y, b, reg=5e-2)

print("Transport plan P (pi):\n", P)
print("Row-normalized weights W:\n", W)
print("WB-mapped points T:\n", T)


import numpy as np
import ot  # pip install POT

def wasserstein_barycentric_map_uniform(
    X,
    Y,
    reg=1e-2,
    metric="sqeuclidean",
):
    """
    Wasserstein barycentric mapping with UNIFORM mass.

    Parameters
    ----------
    X : (n, d) array
        Source shift descriptors (new / target workload)
    Y : (m, d) array
        Reference shift descriptors (meta-training pool)
    reg : float
        Entropic regularization for Sinkhorn.
        Smaller -> closer to exact OT.
    metric : str
        Distance metric passed to ot.dist.

    Returns
    -------
    T : (n, d) array
        WB-mapped descriptors for each x_i
    P : (n, m) array
        Optimal transport plan (pi)
    W : (n, m) array
        Barycentric weights (row-normalized pi)
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n = X.shape[0]
    m = Y.shape[0]

    # 1) Uniform masses (AUTOMATIC)
    a = np.ones(n) / n
    b = np.ones(m) / m

    # 2) Cost matrix
    C = ot.dist(X, Y, metric=metric)

    # 3) Sinkhorn OT (unique, stable solution)
    P = ot.sinkhorn(a, b, C, reg=reg)

    # 4) WB weights: conditional distribution P(y_j | x_i)
    W = P / a[:, None]

    # 5) Barycentric projection
    T = W @ Y

    return T, P, W


X = np.array([[0.10, 0.20],
                  [0.90, 0.80]])

Y = np.array([[0.00, 0.00],
                [1.00, 1.00],
                [0.80, 0.20]])

T, P, W = wasserstein_barycentric_map_uniform(
    X, Y, reg=5e-2
)

print("Transport plan P:\n", P)
print("WB weights W:\n", W)
print("Mapped descriptors T:\n", T)

# import numpy as np
# import ot  # pip install POT

# def _normalize_mass(w, eps=1e-12):
#     w = np.asarray(w, dtype=float)
#     w = np.clip(w, 0.0, None)
#     s = w.sum()
#     if s <= eps:
#         raise ValueError("Mass weights sum to 0. Check your inputs.")
#     return w / s

# def wasserstein_barycentric_map(
#     X,
#     Y,
#     a,
#     b,
#     reg=1e-2,
#     metric="sqeuclidean",
# ):
#     """
#     General WB mapping with user-provided masses a,b.
#     """
#     X = np.asarray(X, dtype=float)
#     Y = np.asarray(Y, dtype=float)
#     a = _normalize_mass(a)
#     b = _normalize_mass(b)

#     C = ot.dist(X, Y, metric=metric)
#     P = ot.sinkhorn(a, b, C, reg=reg)     # (n,m)
#     W = P / a[:, None]                   # row-normalized -> conditional weights
#     T = W @ Y                            # barycentric projection
#     return T, P, W

# # --------------------------
# # Case B: mass ∝ sample count
# # --------------------------
# def wb_mass_by_sample_count(
#     X,
#     Y,
#     N_source,
#     N_target,
#     reg=1e-2,
#     metric="sqeuclidean",
# ):
#     """
#     N_source: (n,) number of samples used to compute each source descriptor x_i
#     N_target: (m,) number of samples used to compute each target descriptor y_j
#     """
#     a = _normalize_mass(N_source)
#     b = _normalize_mass(N_target)
#     return wasserstein_barycentric_map(X, Y, a, b, reg=reg, metric=metric)

# # --------------------------
# # Case C: confidence-weighted mass (advanced)
# # --------------------------
# def wb_mass_by_confidence(
#     X,
#     Y,
#     conf_source,
#     conf_target,
#     reg=1e-2,
#     metric="sqeuclidean",
#     transform="identity",
#     temperature=1.0,
#     eps=1e-12,
# ):
#     """
#     conf_source: (n,) reliability/confidence score for each x_i (higher = more reliable)
#     conf_target: (m,) reliability/confidence score for each y_j

#     transform controls how confidence becomes mass:
#       - "identity": a_i ∝ conf_i
#       - "softmax":  a = softmax(conf / temperature)
#       - "inv_var":  if conf is variance, treat as a_i ∝ 1/(var+eps)
#     """
#     conf_source = np.asarray(conf_source, dtype=float)
#     conf_target = np.asarray(conf_target, dtype=float)

#     if transform == "identity":
#         a = _normalize_mass(conf_source)
#         b = _normalize_mass(conf_target)

#     elif transform == "softmax":
#         def softmax(z):
#             z = z / max(temperature, eps)
#             z = z - np.max(z)
#             ez = np.exp(z)
#             return ez / (ez.sum() + eps)
#         a = softmax(conf_source)
#         b = softmax(conf_target)

#     elif transform == "inv_var":
#         # Here "confidence" inputs are actually variances (lower variance => higher weight)
#         a = _normalize_mass(1.0 / (conf_source + eps))
#         b = _normalize_mass(1.0 / (conf_target + eps))

#     else:
#         raise ValueError(f"Unknown transform={transform}")

#     return wasserstein_barycentric_map(X, Y, a, b, reg=reg, metric=metric)

# X = np.array([[0.10, 0.20],
#                 [0.90, 0.80],
#                 [0.20, 0.85]])          # n=3
# Y = np.array([[0.00, 0.00],
#                 [1.00, 1.00],
#                 [0.80, 0.20],
#                 [0.10, 0.90]])          # m=4

# # Case B: sample-count masses
# N_source = np.array([200, 800, 100])  # each x_i computed from N_i samples
# N_target = np.array([500, 500, 200, 50])
# T_B, P_B, W_B = wb_mass_by_sample_count(X, Y, N_source, N_target, reg=5e-2)
# print("Case B mapped T:\n", T_B)

# # Case C (identity): confidence scores (higher = more reliable)
# conf_source = np.array([0.2, 0.9, 0.4])
# conf_target = np.array([0.6, 0.7, 0.2, 0.9])
# T_C, P_C, W_C = wb_mass_by_confidence(
#     X, Y, conf_source, conf_target, reg=5e-2, transform="identity"
# )
# print("Case C(identity) mapped T:\n", T_C)

# # Case C (inv_var): if you have per-descriptor variance estimates
# var_source = np.array([0.8, 0.1, 0.4])   # lower variance => higher weight
# var_target = np.array([0.2, 0.3, 1.0, 0.15])
# T_C2, P_C2, W_C2 = wb_mass_by_confidence(
#     X, Y, var_source, var_target, reg=5e-2, transform="inv_var"
# )
# print("Case C(inv_var) mapped T:\n", T_C2)

