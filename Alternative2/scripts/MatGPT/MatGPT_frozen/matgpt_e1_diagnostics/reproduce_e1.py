#!/usr/bin/env python3
"""Reproduce las predicciones históricas E1 sin reentrenar la cabeza."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, E1_EMBED, E1_REPORT, DiagnosticError, json_write, log, logger_for, sha256_file, load_inputs

LOGGER = logger_for("matgpt_e1_diagnostics")


class HistoricalE1Head(nn.Module):
    """Recrea los nombres y la arquitectura exactos de la MLP histórica E1."""

    def __init__(self, width: int, dropout: float) -> None:
        """Construye la MLP histórica E1 con los nombres de parámetros originales.

        Args:
            width: Ancho oculto documentado en el checkpoint E1.
            dropout: Tasa de dropout guardada en la configuración histórica.

        Returns:
            None.
        """
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(768, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predice energía normalizada por estructura.

        Args:
            values: Tensor con embeddings de forma `[batch, 768]`.

        Returns:
            Tensor de energías normalizadas con forma `[batch]`.
        """
        return self.regressor(values).squeeze(-1)


def main() -> None:
    """Carga el checkpoint histórico, compara IDs y exporta la auditoría."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=E1_REPORT / "dataset.csv")
    parser.add_argument("--embeddings", type=Path, default=E1_EMBED / "matgpt_uncond_last_token.npz")
    parser.add_argument("--metadata", type=Path, default=E1_EMBED / "embedding_metadata.csv")
    parser.add_argument("--audit", type=Path, default=E1_EMBED / "embedding_audit.json")
    parser.add_argument("--checkpoint", type=Path, default=E1_REPORT / "best_mlp.pt")
    parser.add_argument("--historical-predictions", type=Path, default=ALT / "reports" / "matgpt_e1_v2_eval_v2" / "predictions.csv")
    parser.add_argument("--out", type=Path, default=ALT / "reports" / "matgpt_e1_diagnostics" / "e1_reproduction.json")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        data, vectors, audit = load_inputs(args.dataset, args.embeddings, args.metadata, args.audit)
        if not args.checkpoint.is_file() or not args.historical_predictions.is_file():
            raise DiagnosticError("Falta checkpoint MLP E1 o CSV histórico de predicciones.")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("backbone_checkpoint_sha256") != audit["checkpoint_sha256"]:
            raise DiagnosticError("La cabeza histórica refiere un backbone diferente.")
        width = int(payload["training_config"]["regressor"]["hidden_dim"])
        model = HistoricalE1Head(width, float(payload["training_config"]["regressor"]["dropout"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            reproduced = model(torch.from_numpy(vectors)).numpy() * float(payload["target_std_train_ev_per_atom"]) + float(payload["target_mean_train_ev_per_atom"])
        historic = pd.read_csv(args.historical_predictions, low_memory=False)
        if historic.structure_id.astype(str).tolist() != data.structure_id.astype(str).tolist():
            raise DiagnosticError("El CSV histórico no tiene exactamente los IDs y el orden de E1.")
        if "energy_pred_ev_atom" not in historic:
            raise DiagnosticError("Falta energy_pred_ev_atom en las predicciones históricas.")
        expected = historic.energy_pred_ev_atom.to_numpy(float)
        errors = np.abs(reproduced - expected)
        max_error = float(errors.max())
        if max_error > args.tolerance:
            raise DiagnosticError(f"Reproducción E1 excede tolerancia: max_abs_error={max_error:.8g} eV/átomo")
        result = {"status": "PASS", "checkpoint_sha256": sha256_file(args.checkpoint),
                  "embedding_cache_sha256": sha256_file(args.embeddings), "dataset_sha256": sha256_file(args.dataset),
                  "historical_predictions_sha256": sha256_file(args.historical_predictions), "n_structures": len(data),
                  "max_abs_error_ev_atom": max_error, "mean_abs_error_ev_atom": float(errors.mean()),
                  "tolerance_ev_atom": args.tolerance, "test_labels_used_for_selection": False,
                  "checkpoint_path": str(args.checkpoint), "historical_predictions_path": str(args.historical_predictions)}
        json_write(args.out, result)
        log(LOGGER, logging.INFO, "e1_reproduction_passed", rows=len(data), max_abs_error_ev_atom=max_error, report=str(args.out))
    except (DiagnosticError, OSError, ValueError, RuntimeError, KeyError) as exc:
        log(LOGGER, logging.ERROR, "e1_reproduction_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
