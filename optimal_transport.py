import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. Generate toy data
# ----------------------

def make_two_moons(n_samples=100, noise=0.05, angle=0.0, shift=(0.0, 0.0)):
    """
    Simple 'two moons' generator (similar to sklearn.datasets.make_moons),
    plus optional rotation and shift to create a target domain.
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer moon (class 0)
    theta_out = np.linspace(0, np.pi, n_samples_out)
    x_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])

    # Inner moon (class 1), shifted down
    theta_in = np.linspace(0, np.pi, n_samples_in)
    x_in = np.column_stack([1 - np.cos(theta_in), 1 - np.sin(theta_in) - 0.5])

    X = np.vstack([x_out, x_in])
    y = np.hstack([np.zeros(n_samples_out, dtype=int),
                   np.ones(n_samples_in, dtype=int)])

    # Add noise
    X += noise * np.random.randn(*X.shape)

    # Apply rotation
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s],
                  [s,  c]])
    X = X @ R.T

    # Apply shift
    X += np.array(shift)

    return X, y

# Source domain: standard two moons
Xs, ys = make_two_moons(n_samples=200, noise=0.08, angle=0.0, shift=(0.0, 0.0))

# Target domain: rotated + shifted two moons
Xt, yt = make_two_moons(n_samples=200, noise=0.08, angle=np.pi / 4, shift=(1.0, 0.3))

n_s, d = Xs.shape
n_t, _ = Xt.shape

# ----------------------
# 2. Entropic OT via Sinkhorn
# ----------------------

def sinkhorn_ot(Xs, Xt, reg=0.05, num_iter=200):
    """
    Compute entropic OT coupling between uniform measures on Xs and Xt.
    Returns gamma (n_s x n_t).
    """
    n_s = Xs.shape[0]
    n_t = Xt.shape[0]

    # Uniform marginals
    a = np.ones(n_s) / n_s
    b = np.ones(n_t) / n_t

    # Cost matrix (squared Euclidean)
    C = np.sum(Xs**2, axis=1)[:, None] + np.sum(Xt**2, axis=1)[None, :] - 2 * Xs @ Xt.T

    # Gibbs kernel
    K = np.exp(-C / reg)

    # Avoid numerical issues
    K += 1e-9

    u = np.ones(n_s) / n_s
    v = np.ones(n_t) / n_t

    for _ in range(num_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    # Transport plan
    gamma = np.diag(u) @ K @ np.diag(v)
    return gamma

gamma = sinkhorn_ot(Xs, Xt, reg=0.05, num_iter=300)

# ----------------------
# 3. Barycentric mapping
# ----------------------

# For uniform marginals, each row sums to 1/n_s,
# so the barycentric mapping is Xs_hat = n_s * gamma @ Xt
Xs_mapped = n_s * gamma @ Xt

# ----------------------
# 4. Visualization
# ----------------------

# Before adaptation
plt.figure(figsize=(6, 5))
plt.scatter(Xs[:, 0], Xs[:, 1], s=15, alpha=0.7, label="Source (original)")
plt.scatter(Xt[:, 0], Xt[:, 1], s=15, alpha=0.7, label="Target")
plt.title("Before OT adaptation")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
# plt.show()
plt.savefig("before_ot_adaptation.png", dpi=300, bbox_inches='tight')

# After adaptation: mapped source vs target
plt.figure(figsize=(6, 5))
plt.scatter(Xs_mapped[:, 0], Xs_mapped[:, 1], s=15, alpha=0.7, label="Source (mapped)")
plt.scatter(Xt[:, 0], Xt[:, 1], s=15, alpha=0.7, label="Target")
plt.title("After OT adaptation (barycentric mapping)")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
# plt.show()
plt.savefig("after_ot_adaptation.png", dpi=300, bbox_inches='tight')
