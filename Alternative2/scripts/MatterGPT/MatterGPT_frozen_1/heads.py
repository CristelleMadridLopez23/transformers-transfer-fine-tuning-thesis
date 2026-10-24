"""Small prediction heads used with frozen MatterGPT features."""
from __future__ import annotations

import torch
from torch import nn


class MLP8(nn.Module):
    """Historical scalar MLP head, 768 -> 8 -> 1 with GELU/dropout."""
    def __init__(self, dimension: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, 8), nn.GELU(), nn.Dropout(dropout), nn.Linear(8, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map one feature vector per row to a scalar."""
        return self.network(values).squeeze(-1)


class TwoHead(nn.Module):
    """Separate composition-mean and centered structural-residual heads."""
    def __init__(self, dimension: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.mean_head = MLP8(dimension, dropout)
        self.residual_head = MLP8(dimension, dropout)

    def forward(self, composition_features: torch.Tensor, structural_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean, zero-centered triplet residuals, and reconstructed energy."""
        # One composition vector per triplet, three structure vectors per triplet.
        means = self.mean_head(composition_features)
        raw = self.residual_head(structural_features)
        residual = raw - raw.mean(dim=1, keepdim=True)
        return means, residual, means[:, None] + residual
