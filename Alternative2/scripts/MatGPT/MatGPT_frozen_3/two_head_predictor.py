#!/usr/bin/env python3
"""Arquitectura MLP de dos componentes aplicada a features congeladas."""
from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    """MLP oculta configurable con GELU y dropout."""
    def __init__(self, input_dim: int = 768, width: int = 8, dropout: float = 0.1) -> None:
        """Construye una proyección escalar con cuello de botella.

        Args:
            input_dim: Dimensión fija de embeddings MatGPT.
            width: Neuronas en el cuello de botella.
            dropout: Probabilidad entre cero y uno.
        """
        super().__init__()
        if input_dim < 1 or width < 1 or not 0 <= dropout < 1:
            raise ValueError("Dimensión/ancho/dropout inválido.")
        self.layers = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predice escalar por vector de entrada."""
        return self.layers(values).squeeze(-1)


class TwoHeadPredictor(nn.Module):
    """Predice nivel común desde h_mean y residuo individual desde h_end."""
    def __init__(self, width: int = 8, dropout: float = 0.1) -> None:
        """Inicializa cabeza composicional y estructural independientes."""
        super().__init__()
        self.composition = SmallMLP(width=width, dropout=dropout)
        self.residual = SmallMLP(width=width, dropout=dropout)

    def forward(self, h_mean_triplet: torch.Tensor, h_end_triplet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve μ, residuo centrado y energía reconstruida por triplete."""
        mu = self.composition(h_mean_triplet)
        raw = self.residual(h_end_triplet)
        delta = raw - raw.mean(dim=1, keepdim=True)
        return mu, delta, mu[:, None] + delta


def trainable_parameter_count(module: nn.Module) -> int:
    """Cuenta los parámetros entrenables de las cabezas."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
