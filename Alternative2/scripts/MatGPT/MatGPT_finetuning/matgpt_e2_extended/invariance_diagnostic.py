#!/usr/bin/env python3
"""Measure prediction spread over one fixed bank of verified static SLICES views."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
E2_DIR = ALT / "scripts/MatGPT_finetuning/matgpt_e2"
DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
INFO = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
VARIANTS = ALT / "reports/MatGPT_finetuning/E2-Aug1/variants_development_only_v2/variant_manifest.csv"
REPORTS = ALT / "reports/MatGPT_finetuning"
LOG = logging.getLogger("matgpt_e2.invariance")


def load_e2a() -> Any:
    """Import the exact E2-A model and <end> forward implementation."""
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location("matgpt_e2a_invariance", E2_DIR / "finetune.py")
    if spec is None or spec.loader is None:
        raise ImportError("No se puede importar el E2-A ejecutado.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_bank(e2a: Any, model: torch.nn.Module, head: torch.nn.Module, sequences: list[list[int]], pad: int,
                 device: torch.device, scale: float, mean: float) -> np.ndarray:
    """Predict a batch without truncation using the actual end-token hidden state."""
    width = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        batch[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    with torch.no_grad():
        result = head(e2a.forward_hidden(model, batch, pad, trainable=True)) * scale + mean
    return result.cpu().numpy()


def run(output: Path, models: list[str]) -> None:
    """Evaluate per-structure spread over the same four development-only views.

    Args:
        output: New directory for invariance results.
        models: Any of ``E2-A-partial_ft``, ``E2-Aug1``, and ``E2-Aug2``.
    Returns:
        None; writes CSV, JSON, Markdown, and a PNG figure.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"No se sobrescriben diagnósticos existentes: {output}")
    e2a = load_e2a()
    data = pd.read_csv(DATA, low_memory=False)
    folds = pd.read_csv(FOLDS)
    dev = data[data.split.isin(["train", "val"])].copy().reset_index(drop=True)
    group_map = folds.drop_duplicates("structure_id").set_index("structure_id").composition_group
    dev["composition_group"] = dev.structure_id.astype(str).map(group_map)
    variants = pd.read_csv(VARIANTS, low_memory=False)
    variants = variants[variants.accepted.astype(bool)].copy()
    info = json.loads(INFO.read_text(encoding="utf-8"))
    stoi = info["stoi"]
    view_map: dict[str, list[list[int]]] = {}
    for structure_id, group in variants.groupby("structure_id", sort=False):
        group = group.sort_values("variant_id")
        sequences = []
        for row in group.itertuples(index=False):
            tokens = str(row.variant_slices).split()
            sequences.append([int(stoi["<start>"]), *[int(stoi[token]) for token in tokens], int(stoi["<end>"])])
        if len(sequences) != 4:
            raise RuntimeError(f"Se requieren cuatro vistas fijas por estructura: {structure_id}")
        view_map[str(structure_id)] = sequences
    device = torch.device("cpu")
    result_rows: list[dict[str, Any]] = []
    for model_name in models:
        for fold in range(5):
            outer_ids = set(folds.loc[folds.outer_fold.eq(fold) & folds.outer_role.eq("outer_validation"), "structure_id"].astype(str))
            fold_data = dev[dev.structure_id.astype(str).isin(outer_ids)].copy()
            if len(fold_data) != 42:
                raise RuntimeError(f"Outer fold {fold} no tiene 42 estructuras estáticas.")
            for seed in (42, 123, 2026):
                if model_name == "E2-A-partial_ft":
                    checkpoint_path = REPORTS / "matgpt_e2/runs_endtoken_v2/cpu/partial_ft" / f"fold_{fold}_seed_{seed}.pt"
                else:
                    checkpoint_path = REPORTS / model_name / "development_only_v2/cv/cpu/runs" / f"fold_{fold}_seed_{seed}" / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model, loaded_stoi = e2a.setup()
                if loaded_stoi != stoi:
                    raise RuntimeError("El tokenizer del diagnóstico no coincide con el checkpoint.")
                if model_name == "E2-A-partial_ft":
                    model.load_state_dict(checkpoint["model"], strict=True)
                else:
                    model.transformer_decoder.layers[-1].load_state_dict(checkpoint["last_block"], strict=True)
                for parameter in model.parameters():
                    parameter.requires_grad = False
                head = e2a.Head()
                head.load_state_dict(checkpoint["head"], strict=True)
                model.eval()
                head.eval()
                predictions_by_view: list[np.ndarray] = []
                ids = fold_data.structure_id.astype(str).tolist()
                for view_index in range(4):
                    sequences = [view_map[structure_id][view_index] for structure_id in ids]
                    predictions_by_view.append(predict_bank(e2a, model, head, sequences, int(stoi["<pad>"]), device,
                                                            float(checkpoint.get("target_scale", checkpoint.get("target_scale_ev_atom"))),
                                                            float(checkpoint.get("target_mean", checkpoint.get("target_mean_ev_atom")))))
                matrix = np.stack(predictions_by_view, axis=1)
                spread = matrix.std(axis=1, ddof=0) * 1000.0
                span = (matrix.max(axis=1) - matrix.min(axis=1)) * 1000.0
                for i, row in enumerate(fold_data.itertuples(index=False)):
                    result_rows.append({"model": model_name, "fold": fold, "seed": seed,
                                        "structure_id": str(row.structure_id), "composition_group": str(row.composition_group),
                                        "sigma_inv_mev_atom": float(spread[i]), "view_range_mev_atom": float(span[i]),
                                        "n_views": 4, "all_views_graph_verified": True})
                LOG.info("invariance_run_complete model=%s fold=%d seed=%d", model_name, fold, seed)
    result = pd.DataFrame(result_rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "invariance_per_structure.csv", index=False)
    summary = result.groupby("model").sigma_inv_mev_atom.agg(
        mean_mev_atom="mean", median_mev_atom="median", p90_mev_atom=lambda x: float(np.quantile(x, .9)),
        max_mev_atom="max", observations="count").reset_index()
    summary.to_csv(output / "invariance_summary.csv", index=False)
    payload = {"status": "PASS", "models": models, "n_oof_structure_seed_rows_per_model": 630,
               "views_per_structure": 4, "test_historical_used": False,
               "definition": "sigma_inv = population standard deviation across four fixed graph-equivalent SLICES views, meV/atom",
               "note": "630 rows include each development structure once per seed; seeds are not independent structures."}
    (output / "invariance_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    colors = {name: color for name, color in zip(models, ("#8055A6", "#2878B5", "#E07A24"))}
    fig, ax = plt.subplots(figsize=(9, 5.6))
    vals = [result.loc[result.model.eq(name), "sigma_inv_mev_atom"].to_numpy() for name in models]
    artists = ax.boxplot(vals, labels=models, patch_artist=True, showfliers=True)
    for patch, name in zip(artists["boxes"], models):
        patch.set_facecolor(colors[name])
        patch.set_alpha(.7)
    ax.set_ylabel("Desviación estándar entre vistas, σ_inv (meV/átomo)")
    ax.set_title("Energía predicha ante cuatro serializaciones equivalentes")
    ax.grid(axis="y", alpha=.25)
    fig.text(.01, .01, "Cada observación = una estructura outer de desarrollo en una semilla (N=630 por modelo); sin test histórico.\n"
             "Cajas: Q1–Q3, línea: mediana, bigotes: 1.5×IQR; los puntos fuera de bigotes se muestran.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "invariance_spread.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Diagnóstico de invariancia de serialización", "",
             "Predicciones outer de desarrollo sobre las mismas cuatro vistas SLICES, verificadas con equivalencia de grafo. No se usa el test.", "",
             "| Modelo | Media σ_inv (meV/átomo) | Mediana | P90 | Máximo | N (estructura×semilla) |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.mean_mev_atom:.3f} | {row.median_mev_atom:.3f} | {row.p90_mev_atom:.3f} | {row.max_mev_atom:.3f} | {int(row.observations)} |")
    lines += ["", "Las semillas no se cuentan como composiciones nuevas. Un σ_inv pequeño no demuestra mejor ranking; se interpreta junto con las métricas estáticas OOF.", ""]
    (output / "invariance_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Parse models and output directory, then compute fixed-bank invariance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=["E2-A-partial_ft", "E2-Aug1", "E2-Aug2"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.out, args.model)


if __name__ == "__main__":
    main()
