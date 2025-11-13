"""FusionSQL evaluator model."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionSQL(nn.Module):
    """Three-layer MLP that maps shift descriptors to accuracy estimates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("FusionSQL expects exactly two hidden dimensions (three layers total).")
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h)
        return out.squeeze(-1)

    def functional_forward(self, x: torch.Tensor, params: Sequence[torch.Tensor] | None) -> torch.Tensor:
        if params is None:
            return self.forward(x)
        if len(params) != 6:
            raise ValueError("Expected 6 parameter tensors (weight/bias pairs for three layers).")
        h = F.linear(x, params[0], params[1])
        h = F.relu(h)
        h = F.linear(h, params[2], params[3])
        h = F.relu(h)
        out = F.linear(h, params[4], params[5])
        return out.squeeze(-1)

    def parameter_list(self) -> List[torch.nn.Parameter]:
        return [
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc3.weight,
            self.fc3.bias,
        ]

    @staticmethod
    def clone_parameters(params: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        return [p.clone().detach().requires_grad_(True) for p in params]
