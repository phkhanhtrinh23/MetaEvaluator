# ====== Minimal First-Order MAML (Sinusoid Few-Shot Regression) ======
# Setup: Each task T is y = A * sin(x + phi). We learn an init θ so that
# ONE small gradient step on K support points fits the task, and we evaluate on query points.

import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Task distribution: y = A sin(x + phi) ----
def sample_task():
    A = random.uniform(0.1, 5.0)
    phi = random.uniform(0.0, math.pi)
    return A, phi

def sample_data(A, phi, n):
    x = torch.empty(n, 1).uniform_(-5.0, 5.0)
    y = A * torch.sin(x + phi)
    return x, y

# ---- Small MLP and a functional forward that takes explicit params ----
class MLP(nn.Module):
    def __init__(self, in_dim=1, hid=40, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, out_dim)

def forward_with(params, x):
    W1,b1,W2,b2,W3,b3 = params
    h1 = torch.relu(x @ W1.T + b1)
    h2 = torch.relu(h1 @ W2.T + b2)
    return h2 @ W3.T + b3

def get_param_list(model):
    return [model.fc1.weight, model.fc1.bias,
            model.fc2.weight, model.fc2.bias,
            model.fc3.weight, model.fc3.bias]

# ---- One inner (task) step: θ' = θ - α ∇θ L_sup(θ) (First-Order: no 2nd derivatives) ----
def inner_adapt(params, x_sup, y_sup, lr_inner=0.01):
    y_hat = forward_with(params, x_sup)
    loss  = F.mse_loss(y_hat, y_sup)
    grads = torch.autograd.grad(loss, params, create_graph=False)  # FOMAML
    new_params = [p - lr_inner*g for p,g in zip(params, grads)]
    # ensure they require grad for the meta-backward
    new_params = [p.clone().detach().requires_grad_(True) for p in new_params]
    return new_params, loss.detach()

# ---- Meta-training loop ----
def meta_train(steps=2000, tasks_per_batch=20, K=10, Q=10, lr_inner=0.01, lr_meta=1e-3, print_every=200):
    torch.manual_seed(0); random.seed(0)
    model = MLP()
    meta_optim = torch.optim.Adam(model.parameters(), lr=lr_meta)

    for step in range(1, steps+1):
        meta_optim.zero_grad()
        meta_loss = 0.0

        # Build one meta-batch of tasks
        for _ in range(tasks_per_batch):
            A, phi = sample_task()
            x_sup, y_sup = sample_data(A, phi, K)
            x_qry, y_qry = sample_data(A, phi, Q)

            # Start from current meta-params θ
            params = [p for p in get_param_list(model)]
            params = [p.clone().detach().requires_grad_(True) for p in params]

            # Inner step on support
            adapted_params, _ = inner_adapt(params, x_sup, y_sup, lr_inner)

            # Query loss at θ' (post-adaptation)
            y_hat_q = forward_with(adapted_params, x_qry)
            task_loss = F.mse_loss(y_hat_q, y_qry)
            meta_loss = meta_loss + task_loss

        # Average over tasks and update θ (first-order gradient through θ only)
        meta_loss = meta_loss / tasks_per_batch
        meta_loss.backward()
        meta_optim.step()

        if step % print_every == 0:
            print(f"[meta-step {step:4d}] meta-loss {meta_loss.item():.4f}")

    return model

# ---- Meta-test (inference): freeze θ*, do ONLY the inner step(s) on a NEW task ----
def meta_test(model, K=10, lr_inner=0.01, inner_steps=1):
    # fresh unseen task
    A, phi = sample_task()
    x_plot = torch.linspace(-5, 5, 200).unsqueeze(1)
    y_true = A * torch.sin(x_plot + phi)

    # BEFORE adaptation
    with torch.no_grad():
        # quick forward using model params directly
        W1, b1 = model.fc1.weight, model.fc1.bias
        W2, b2 = model.fc2.weight, model.fc2.bias
        W3, b3 = model.fc3.weight, model.fc3.bias
        y_pred_before = forward_with([W1,b1,W2,b2,W3,b3], x_plot)

    # Build tiny support set and adapt from θ*
    x_sup, y_sup = sample_data(A, phi, K)
    params = [p.clone().detach().requires_grad_(True) for p in get_param_list(model)]
    for _ in range(inner_steps):
        params, _ = inner_adapt(params, x_sup, y_sup, lr_inner)

    with torch.no_grad():
        y_pred_after = forward_with(params, x_plot)

    mse_before = F.mse_loss(y_pred_before, y_true).item()
    mse_after  = F.mse_loss(y_pred_after,  y_true).item()
    print(f"[meta-test] A={A:.3f}, phi={phi:.3f} | MSE before: {mse_before:.4f} -> after: {mse_after:.4f}")

# ===== Run it =====
if __name__ == "__main__":
    model = meta_train(steps=2000, tasks_per_batch=20, K=10, Q=10, lr_inner=0.01, lr_meta=1e-3, print_every=200)
    # Test on a brand-new task: do 1 inner step, then evaluate
    meta_test(model, K=10, lr_inner=0.01, inner_steps=1)
