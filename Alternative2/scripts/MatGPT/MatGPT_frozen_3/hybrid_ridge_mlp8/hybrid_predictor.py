#!/usr/bin/env python3
"""Cabeza Ridge composicional y cabeza MLP8 estructural para H3."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


class StructuralMLP8(nn.Module):
    """MLP histórica 768→8→1 para energía residual por polimorfo."""
    def __init__(self, input_dim: int = 768, dropout: float = 0.1) -> None:
        """Inicializa proyección con GELU y dropout.

        Args:
            input_dim: Dimensión de los embeddings finales de MatGPT.
            dropout: Probabilidad de dropout histórica H2.
        """
        super().__init__()
        if input_dim != 768 or not 0 <= dropout < 1:
            raise ValueError("La MLP8 espera dimensión 768 y dropout en [0,1).")
        self.network = nn.Sequential(nn.Linear(input_dim, 8), nn.GELU(), nn.Dropout(dropout), nn.Linear(8, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predice residuos centrados para tensor [composición, 3, 768]."""
        if embeddings.ndim != 3 or embeddings.shape[1:] != (3, 768):
            raise ValueError("Se esperaba [n_composiciones, 3, 768].")
        raw = self.network(embeddings).squeeze(-1)
        return raw - raw.mean(dim=1, keepdim=True)


def fit_ridge_by_validation(x_train: np.ndarray, y_train: np.ndarray,
                            x_valid: np.ndarray, y_valid: np.ndarray,
                            alphas: tuple[float, ...]) -> tuple[float, float, np.ndarray, list[dict[str, float]]]:
    """Selecciona alpha por MAE interno usando scaler ajustado en inner-train.

    Args:
        x_train: Media de embeddings por composición inner-train.
        y_train: Energía DFT media por composición inner-train.
        x_valid: Media de embeddings por composición inner-validation.
        y_valid: Energías medias inner-validation.
        alphas: Grilla fijada antes de evaluar outer folds.
    Returns:
        Alpha, MAE y predicción del mejor modelo, más tabla de todos los alpha.
    """
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid) or not len(alphas):
        raise ValueError("Shapes o grilla Ridge inválidos.")
    best: tuple[float, float, np.ndarray] | None = None
    scores: list[dict[str, float]] = []
    for alpha in alphas:
        pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), fit_intercept=True,
                                                                              solver="lsqr", tol=1e-8))])
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_valid)
        score = float(np.mean(np.abs(pred - y_valid)))
        scores.append({"alpha": float(alpha), "inner_validation_mae_ev_atom": score})
        candidate = (score, float(alpha), pred)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    assert best is not None
    return best[1], best[0], best[2], scores


def make_ridge(alpha: float) -> Pipeline:
    """Construye pipeline final de estandarización y Ridge."""
    if alpha <= 0:
        raise ValueError("alpha debe ser positivo.")
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), fit_intercept=True,
                                                                      solver="lsqr", tol=1e-8))])


def centered_smooth_l1(pred: torch.Tensor, target: torch.Tensor, scale: float,
                       beta: float = 1.0) -> torch.Tensor:
    """Pérdida SmoothL1 normalizada con centrado polimórfico y escala train-only."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 3 or scale <= 0:
        raise ValueError("Pérdida residual espera tripletes alineados y escala positiva.")
    centered = pred - pred.mean(dim=1, keepdim=True)
    return nn.functional.smooth_l1_loss(centered / scale, target / scale, beta=beta, reduction="mean")
