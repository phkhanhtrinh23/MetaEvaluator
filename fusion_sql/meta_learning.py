"""Meta-learning loop for FusionSQL."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from .descriptors import DEFAULT_FEATURE_ORDER, ShiftDescriptor
from .model import FusionSQL


@dataclass
class ShiftDescriptorTask:
    """Encapsulates the support/query descriptors for a single model task."""

    model_name: str
    support_descriptor: torch.Tensor
    support_label: torch.Tensor
    query_descriptor: torch.Tensor
    query_label: torch.Tensor
    transfer_descriptor: torch.Tensor | None = None
    transfer_label: torch.Tensor | None = None

    @classmethod
    def from_descriptors(
        cls,
        *,
        model_name: str,
        support: ShiftDescriptor,
        support_label: float,
        query: ShiftDescriptor,
        query_label: float,
        transfer: ShiftDescriptor | None = None,
        transfer_label: float | None = None,
        device: torch.device | None = None,
        feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
    ) -> "ShiftDescriptorTask":
        support_tensor = support.as_tensor(feature_order, device=device).unsqueeze(0)
        query_tensor = query.as_tensor(feature_order, device=device).unsqueeze(0)
        transfer_tensor = None
        if transfer:
            transfer_tensor = transfer.as_tensor(feature_order, device=device).unsqueeze(0)
        return cls(
            model_name=model_name,
            support_descriptor=support_tensor,
            support_label=torch.tensor([support_label], dtype=torch.float32, device=device),
            query_descriptor=query_tensor,
            query_label=torch.tensor([query_label], dtype=torch.float32, device=device),
            transfer_descriptor=transfer_tensor,
            transfer_label=(
                torch.tensor([transfer_label], dtype=torch.float32, device=device) if transfer_label is not None else None
            ),
        )


@dataclass
class MetaLearningConfig:
    inner_lr: float = 0.01
    outer_lr: float = 1e-3
    inner_steps: int = 1
    tasks_per_batch: int = 4
    num_epochs: int = 100
    device: str | None = None
    first_order: bool = True
    eval_inner_steps: int | None = None
    val_interval: int = 10
    early_stopping_patience: int = 20
    meta_reg_lambda: float = 0.0  # L2 regularization on meta-parameters
    meta_reg_beta: float = 0.0  # KL-style regularizer on adapted activations


class FusionSQLMetaLearner:
    """First-order MAML-style trainer for FusionSQL."""

    def __init__(self, model: FusionSQL, config: MetaLearningConfig):
        self.cfg = config
        self.device = torch.device(
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.feature_order = DEFAULT_FEATURE_ORDER

    def _prepare_params(self, *, detach: bool = False) -> List[torch.Tensor]:
        params = [p for p in self.model.parameter_list()]
        if detach:
            return FusionSQL.clone_parameters(params)
        return list(params)

    def _kl_reg(self, tensor: torch.Tensor) -> torch.Tensor:
        # Approximate KL(q(z|x)||N(0,I)) assuming unit variance for q: 0.5 * ||mu||^2
        return 0.5 * torch.mean(tensor ** 2)

    def _inner_step(self, params: List[torch.Tensor], descriptor: torch.Tensor, label: torch.Tensor) -> List[torch.Tensor]:
        preds = self.model.functional_forward(descriptor, params)
        loss = F.mse_loss(preds, label)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=not self.cfg.first_order,
            retain_graph=not self.cfg.first_order,
        )
        updated: List[torch.Tensor] = []
        for param, grad in zip(params, grads):
            if grad is None:
                updated.append(param)
            else:
                updated.append(param - self.cfg.inner_lr * grad)
        return updated

    def meta_train(
        self,
        tasks: Sequence[ShiftDescriptorTask],
        *,
        val_tasks: Sequence[ShiftDescriptorTask] | None = None,
        checkpoint_path: Path | None = None,
    ) -> List[float]:
        if not tasks:
            raise ValueError("Meta-training requires at least one task.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.outer_lr)
        history: List[float] = []
        best_val_mae = float("inf")
        epochs_without_improve = 0

        for epoch in trange(1, self.cfg.num_epochs + 1, desc="Meta-train", leave=False):
            batch = random.sample(tasks, k=min(self.cfg.tasks_per_batch, len(tasks)))
            meta_loss = 0.0
            optimizer.zero_grad()

            for task in batch:
                params = self._prepare_params(detach=False)
                for _ in range(self.cfg.inner_steps):
                    params = self._inner_step(params, task.support_descriptor, task.support_label)
                preds = self.model.functional_forward(task.query_descriptor, params)
                loss = F.mse_loss(preds, task.query_label)
                if self.cfg.meta_reg_beta > 0.0:
                    loss = loss + self.cfg.meta_reg_beta * self._kl_reg(preds)
                meta_loss = meta_loss + loss

            meta_loss = meta_loss / len(batch)
            if self.cfg.meta_reg_lambda > 0.0:
                reg = 0.0
                for p in self.model.parameters():
                    reg = reg + torch.sum(p ** 2)
                meta_loss = meta_loss + self.cfg.meta_reg_lambda * reg
            meta_loss.backward()
            optimizer.step()
            history.append(meta_loss.item())

            if val_tasks and epoch % max(1, self.cfg.val_interval) == 0:
                val_results = self.evaluate(val_tasks)
                val_mae = float(np.mean([entry["mae"] for entry in val_results]))
                if val_mae + 1e-6 < best_val_mae:
                    best_val_mae = val_mae
                    epochs_without_improve = 0
                    if checkpoint_path:
                        torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    epochs_without_improve += 1
                    if self.cfg.early_stopping_patience > 0 and epochs_without_improve >= self.cfg.early_stopping_patience:
                        break

        if checkpoint_path and checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return history

    def _adapt_and_predict(
        self,
        support_descriptor: torch.Tensor,
        support_label: torch.Tensor,
        target_descriptor: torch.Tensor,
        *,
        inner_steps: int | None = None,
    ) -> torch.Tensor:
        params = self._prepare_params(detach=True)
        steps = inner_steps or self.cfg.eval_inner_steps or self.cfg.inner_steps
        for _ in range(steps):
            params = self._inner_step(params, support_descriptor, support_label)
        preds = self.model.functional_forward(target_descriptor, params)
        return preds.squeeze()

    def evaluate(self, tasks: Iterable[ShiftDescriptorTask], *, return_mae: bool = False) -> List[dict] | tuple[List[dict], float]:
        results: List[dict] = []
        for task in tqdm(list(tasks), desc="Meta-eval", leave=False):
            preds = self._adapt_and_predict(task.support_descriptor, task.support_label, task.query_descriptor)
            mae = torch.abs(preds - task.query_label.squeeze()).item()
            results.append(
                {
                    "model": task.model_name,
                    "predicted_accuracy": float(preds.item()),
                    "true_accuracy": float(task.query_label.squeeze().item()),
                    "mae": mae,
                }
            )
        if return_mae:
            mean_mae = float(np.mean([entry["mae"] for entry in results])) if results else 0.0
            return results, mean_mae
        return results

    def evaluate_transfer(self, tasks: Iterable[ShiftDescriptorTask]) -> List[dict]:
        """Evaluate on held-out descriptors (e.g., real dev set)."""

        results: List[dict] = []
        for task in tqdm(list(tasks), desc="Transfer-eval", leave=False):
            if task.transfer_descriptor is None or task.transfer_label is None:
                continue
            preds = self._adapt_and_predict(
                task.support_descriptor,
                task.support_label,
                task.transfer_descriptor,
            )
            mae = torch.abs(preds - task.transfer_label.squeeze()).item()
            results.append(
                {
                    "model": task.model_name,
                    "predicted_accuracy": float(preds.item()),
                    "true_accuracy": float(task.transfer_label.squeeze().item()),
                    "mae": mae,
                }
            )
        return results
