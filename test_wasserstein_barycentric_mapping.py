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
