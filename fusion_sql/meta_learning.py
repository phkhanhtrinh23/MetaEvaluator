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
from .optimal_transport import barycentric_mapping, sinkhorn_plan


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
class TaskContextEmbedding:
    """Stores the adapted context for a training task together with its descriptor."""

    model_name: str
    descriptor: torch.Tensor
    context: torch.Tensor


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
    eval_context_steps: int | None = None
    val_interval: int = 10
    early_stopping_patience: int = 20
    meta_reg_lambda: float = 0.0  # L2 regularization on meta-parameters
    meta_reg_beta: float = 0.0  # KL-style regularizer on adapted activations
    context_dim: int = 32
    ot_num_iter: int = 200


class FusionSQLMetaLearner:
    """CAVIA-style meta-learner that adapts per-task context vectors."""

    def __init__(self, model: FusionSQL, config: MetaLearningConfig):
        self.cfg = config
        self.device = torch.device(
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.feature_order = DEFAULT_FEATURE_ORDER
        self.context_dim = max(0, int(config.context_dim))
        self.eval_context_steps = config.eval_context_steps
        self.ot_num_iter = max(1, int(config.ot_num_iter))

    def _prepare_params(self, *, detach: bool = False) -> List[torch.Tensor]:
        params = [p for p in self.model.parameter_list()]
        if detach:
            return FusionSQL.clone_parameters(params)
        return list(params)

    def _init_context(self) -> torch.Tensor:
        if self.context_dim == 0:
            return torch.zeros(0, device=self.device)
        return torch.zeros(self.context_dim, device=self.device, requires_grad=True)

    def _augment_with_context(self, descriptor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.context_dim == 0:
            return descriptor
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if descriptor.shape[0] != context.shape[0]:
            context = context.expand(descriptor.shape[0], -1)
        return torch.cat([descriptor, context], dim=-1)

    def _context_step(
        self,
        params: List[torch.Tensor],
        descriptor: torch.Tensor,
        label: torch.Tensor,
        context: torch.Tensor,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        if self.context_dim == 0:
            return context
        preds = self.model.functional_forward(self._augment_with_context(descriptor, context), params)
        loss = F.mse_loss(preds, label)
        (grad_context,) = torch.autograd.grad(
            loss,
            context,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        updated = context - self.cfg.inner_lr * grad_context
        if self.cfg.first_order and not create_graph:
            updated = updated.detach()
        return updated.requires_grad_(True)

    def _param_step(
        self,
        params: List[torch.Tensor],
        descriptor: torch.Tensor,
        label: torch.Tensor,
        context: torch.Tensor,
        *,
        create_graph: bool,
    ) -> List[torch.Tensor]:
        preds = self.model.functional_forward(self._augment_with_context(descriptor, context), params)
        loss = F.mse_loss(preds, label)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        updated: List[torch.Tensor] = []
        for param, grad in zip(params, grads):
            if grad is None:
                updated.append(param)
            else:
                updated.append(param - self.cfg.inner_lr * grad)
        return updated

    def _predict_with_context(
        self,
        params: List[torch.Tensor] | None,
        descriptor: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.functional_forward(self._augment_with_context(descriptor, context), params)

    def _kl_reg(self, tensor: torch.Tensor) -> torch.Tensor:
        # Approximate KL(q(z|x)||N(0,I)) assuming unit variance for q: 0.5 * ||mu||^2
        return 0.5 * torch.mean(tensor ** 2)

    def _adapt_context(
        self,
        params: List[torch.Tensor],
        descriptor: torch.Tensor,
        label: torch.Tensor,
        *,
        steps: int,
        create_graph: bool,
        context_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = context_init if context_init is not None else self._init_context()
        if self.context_dim == 0:
            return context
        for _ in range(steps):
            context = self._context_step(
                params,
                descriptor,
                label,
                context,
                create_graph=create_graph,
            )
        return context

    def _context_mapping(
        self,
        context_bank: Sequence[TaskContextEmbedding],
        tasks: Sequence[ShiftDescriptorTask],
        *,
        ot_reg: float,
        descriptor_kind: str,
    ) -> tuple[List[int], List[np.ndarray | None]]:
        source = np.stack([entry.descriptor.squeeze(0).cpu().numpy() for entry in context_bank])
        target_desc: List[np.ndarray] = []
        valid_indices: List[int] = []
        for idx, task in enumerate(tasks):
            desc = task.query_descriptor if descriptor_kind == "query" else task.transfer_descriptor
            if desc is None:
                continue
            target_desc.append(desc.squeeze(0).cpu().numpy())
            valid_indices.append(idx)
        if not target_desc:
            return [-1 for _ in tasks], [None for _ in tasks]
        target = np.stack(target_desc)
        gamma = sinkhorn_plan(source, target, reg=ot_reg, num_iter=self.ot_num_iter)
        top = np.argmax(gamma, axis=0)
        mapping: List[int] = [-1 for _ in tasks]
        mapped_list: List[np.ndarray | None] = [None for _ in tasks]
        mapped_targets = barycentric_mapping(gamma.T, source)
        for j, task_idx in enumerate(valid_indices):
            mapping[task_idx] = int(top[j])
            mapped_list[task_idx] = mapped_targets[j].astype(np.float32, copy=False)
        return mapping, mapped_list

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
                context = self._adapt_context(
                    params,
                    task.support_descriptor,
                    task.support_label,
                    steps=self.cfg.inner_steps,
                    create_graph=not self.cfg.first_order,
                )
                preds = self._predict_with_context(params, task.query_descriptor, context)
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
                    if (
                        self.cfg.early_stopping_patience > 0
                        and epochs_without_improve >= self.cfg.early_stopping_patience
                    ):
                        break

        if checkpoint_path and checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return history

    def _evaluate_task(
        self,
        task: ShiftDescriptorTask,
        *,
        context_init: torch.Tensor | None,
        adapt_context_steps: int,
        adapt_weight_steps: int,
        target_descriptor: torch.Tensor,
        target_label: torch.Tensor,
        adapt_descriptor: torch.Tensor | None = None,
        adapt_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_params = self._prepare_params(detach=True)
        context = context_init if context_init is not None else self._init_context()
        context = context.to(self.device)

        adapt_desc = adapt_descriptor if adapt_descriptor is not None else task.support_descriptor
        adapt_lab = adapt_label if adapt_label is not None else task.support_label

        if adapt_context_steps > 0 and self.context_dim > 0:
            context = context.detach().clone().requires_grad_(True)
            context = self._adapt_context(
                base_params,
                adapt_desc,
                adapt_lab,
                steps=adapt_context_steps,
                create_graph=False,
                context_init=context,
            ).detach()

        params = base_params
        if adapt_weight_steps > 0:
            for _ in range(adapt_weight_steps):
                params = self._param_step(
                    params,
                    adapt_desc,
                    adapt_lab,
                    context,
                    create_graph=False,
                )

        if target_descriptor is None:
            raise ValueError(f"Task {task.model_name} missing target descriptor.")
        preds = self._predict_with_context(params, target_descriptor, context)
        return preds

    def build_context_bank(
        self,
        tasks: Sequence[ShiftDescriptorTask],
        *,
        context_steps: int | None = None,
    ) -> List[TaskContextEmbedding]:
        steps = context_steps if context_steps is not None else self.cfg.inner_steps
        bank: List[TaskContextEmbedding] = []
        base_params = self._prepare_params(detach=True)
        for task in tqdm(list(tasks), desc="Context-adapt", leave=False):
            context = self._adapt_context(
                base_params,
                task.support_descriptor,
                task.support_label,
                steps=steps,
                create_graph=False,
            )
            bank.append(
                TaskContextEmbedding(
                    model_name=task.model_name,
                    descriptor=task.support_descriptor.detach().clone(),
                    context=context.detach(),
                )
            )
        return bank

    def evaluate(
        self,
        tasks: Iterable[ShiftDescriptorTask],
        *,
        context_bank: Sequence[TaskContextEmbedding] | None = None,
        ot_reg: float = 0.1,
        adapt_context_steps: int | None = None,
        adapt_weight_steps: int | None = None,
        return_mae: bool = False,
    ) -> List[dict] | tuple[List[dict], float]:
        results: List[dict] = []
        tasks_list = list(tasks)

        if adapt_context_steps is None:
            adapt_context_steps = self.eval_context_steps or self.cfg.inner_steps
        if adapt_weight_steps is None:
            adapt_weight_steps = self.cfg.eval_inner_steps or self.cfg.inner_steps

        mapping: List[int] = [-1 for _ in tasks_list]
        ref_names: List[str] | None = None
        mapped_descs: List[np.ndarray | None] = [None for _ in tasks_list]
        if context_bank:
            ref_names = [entry.model_name for entry in context_bank]
            mapping, mapped_descs = self._context_mapping(
                context_bank,
                tasks_list,
                ot_reg=ot_reg,
                descriptor_kind="query",
            )

        for idx, task in enumerate(tqdm(tasks_list, desc="Meta-eval", leave=False)):
            init_context = None
            ref_model = None
            mapped_idx = mapping[idx] if mapping else -1
            if mapped_idx != -1 and context_bank is not None:
                init_context = context_bank[mapped_idx].context
                ref_model = ref_names[mapped_idx] if ref_names else None
            target_descriptor = (
                torch.tensor(
                    mapped_descs[idx],
                    device=self.device,
                    dtype=task.query_descriptor.dtype,
                ).unsqueeze(0)
                if mapped_descs[idx] is not None
                else task.query_descriptor
            )
            preds = self._evaluate_task(
                task,
                context_init=init_context,
                adapt_context_steps=adapt_context_steps,
                adapt_weight_steps=adapt_weight_steps,
                target_descriptor=target_descriptor,
                target_label=task.query_label,
                adapt_descriptor=target_descriptor,
                adapt_label=task.query_label,
            )
            mae = torch.abs(preds - task.query_label.squeeze()).item()
            entry = {
                "model": task.model_name,
                "predicted_accuracy": float(preds.item()),
                "true_accuracy": float(task.query_label.squeeze().item()),
                "mae": mae,
            }
            if ref_model is not None:
                entry["ref_model"] = ref_model
                entry["ref_index"] = mapped_idx
            results.append(entry)

        if return_mae:
            mean_mae = float(np.mean([entry["mae"] for entry in results])) if results else 0.0
            return results, mean_mae
        return results

    def evaluate_transfer(
        self,
        tasks: Iterable[ShiftDescriptorTask],
        *,
        context_bank: Sequence[TaskContextEmbedding] | None = None,
        ot_reg: float = 0.1,
        adapt_context_steps: int | None = None,
        adapt_weight_steps: int | None = None,
    ) -> List[dict]:
        results: List[dict] = []
        tasks_list = list(tasks)

        if adapt_context_steps is None:
            adapt_context_steps = self.eval_context_steps or self.cfg.inner_steps
        if adapt_weight_steps is None:
            adapt_weight_steps = self.cfg.eval_inner_steps or self.cfg.inner_steps

        mapping: List[int] = [-1 for _ in tasks_list]
        ref_names: List[str] | None = None
        mapped_descs: List[np.ndarray | None] = [None for _ in tasks_list]
        if context_bank:
            ref_names = [entry.model_name for entry in context_bank]
            mapping, mapped_descs = self._context_mapping(
                context_bank,
                tasks_list,
                ot_reg=ot_reg,
                descriptor_kind="transfer",
            )

        for idx, task in enumerate(tqdm(tasks_list, desc="Transfer-eval", leave=False)):
            if task.transfer_descriptor is None or task.transfer_label is None:
                continue
            init_context = None
            ref_model = None
            mapped_idx = mapping[idx] if mapping else -1
            if mapped_idx != -1 and context_bank is not None:
                init_context = context_bank[mapped_idx].context
                ref_model = ref_names[mapped_idx] if ref_names else None
            target_descriptor = (
                torch.tensor(
                    mapped_descs[idx],
                    device=self.device,
                    dtype=task.transfer_descriptor.dtype,
                ).unsqueeze(0)
                if mapped_descs[idx] is not None
                else task.transfer_descriptor
            )
            preds = self._evaluate_task(
                task,
                context_init=init_context,
                adapt_context_steps=adapt_context_steps,
                adapt_weight_steps=adapt_weight_steps,
                target_descriptor=target_descriptor,
                target_label=task.transfer_label,
                adapt_descriptor=target_descriptor,
                adapt_label=task.transfer_label,
            )
            mae = torch.abs(preds - task.transfer_label.squeeze()).item()
            entry = {
                "model": task.model_name,
                "predicted_accuracy": float(preds.item()),
                "true_accuracy": float(task.transfer_label.squeeze().item()),
                "mae": mae,
            }
            if ref_model is not None:
                entry["ref_model"] = ref_model
                entry["ref_index"] = mapped_idx
            results.append(entry)
        return results
