#!/usr/bin/env python3
"""Pérdidas de media y residuo compatibles con el H2 histórico."""
from __future__ import annotations

import torch
from torch import nn


def two_component_smooth_l1(mu_pred: torch.Tensor, delta_pred: torch.Tensor,
                             mu_true: torch.Tensor, delta_true: torch.Tensor,
                             scale_mu: float, scale_delta: float, beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Devuelve pérdida composicional, residual y su suma con escalas train-only."""
    loss_mu = nn.functional.smooth_l1_loss((mu_pred - mu_true) / scale_mu,
        torch.zeros_like(mu_true), beta=beta, reduction="mean")
    loss_delta = nn.functional.smooth_l1_loss((delta_pred - delta_true) / scale_delta,
        torch.zeros_like(delta_true), beta=beta, reduction="mean")
    return loss_mu, loss_delta, loss_mu + loss_delta
