#!/usr/bin/env python3
"""Entrena tres cabezas congeladas en 5 folds agrupados y 3 semillas (45 fits)."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import CANDIDATES, DATASET, OUT, SEEDS, E13Error, get_logger, load_fold_assignments, load_inputs, log, sha256_file, write_json

LOGGER = get_logger("matgpt_e1_3_train")
MAX_EPOCHS = 200
PATIENCE = 20
COMPOSITIONS_PER_BATCH = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


class Head(nn.Module):
    """MLP de E1.2: 768 → dimensión oculta → 1 con GELU y dropout 0.1."""

    def __init__(self, width: int) -> None:
        """Inicializa solo los parámetros pequeños de la cabeza."""
        super().__init__()
        if width not in {8, 32}:
            raise E13Error(f"Dimensión de cabeza no permitida: {width}")
        self.layers = nn.Sequential(nn.Linear(768, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Devuelve energía normalizada por estructura."""
        return self.layers(values).squeeze(-1)


def seed_all(seed: int) -> None:
    """Fija fuentes aleatorias Python/NumPy/PyTorch para una ejecución."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def composition_batches(data: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Baraja composiciones y conserva juntos todos sus polimorfos disponibles."""
    groups = [np.asarray(group.index, dtype=int) for _, group in data.loc[indices].groupby("composition_group", sort=True)]
    order = rng.permutation(len(groups))
    return [np.concatenate([groups[i] for i in order[start:start + COMPOSITIONS_PER_BATCH]])
            for start in range(0, len(order), COMPOSITIONS_PER_BATCH)]


def pairwise_loss(prediction: torch.Tensor, energy: torch.Tensor, formulas: list[str], scale: float) -> torch.Tensor:
    """Promedia Huber de pares DFT/ML dentro de cada composición del lote."""
    losses: list[torch.Tensor] = []
    for formula in sorted(set(formulas)):
        positions = [i for i, item in enumerate(formulas) if item == formula]
        if len(positions) < 2:
            continue
        i, j = torch.triu_indices(len(positions), len(positions), offset=1, device=prediction.device)
        pred, truth = prediction[positions], energy[positions]
        residual = ((pred[i] - pred[j]) - (truth[i] - truth[j])) / scale
        losses.append(nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def objective(prediction: torch.Tensor, energy: torch.Tensor, formulas: list[str], target_scale: float,
              pair_scale: float, pair_lambda: float) -> torch.Tensor:
    """Replica E1.2 Huber absoluta normalizada más loss pairwise ponderada."""
    absolute = nn.functional.smooth_l1_loss((prediction - energy) / target_scale,
                                             torch.zeros_like(energy), beta=1.0)
    return absolute + pair_lambda * pairwise_loss(prediction, energy, formulas, pair_scale)


def train_one(data: pd.DataFrame, vectors_np: np.ndarray, candidate_name: str, fold: int, seed: int,
              train_ids: np.ndarray, inner_val_ids: np.ndarray, outer_ids: np.ndarray,
              output: Path, input_hashes: dict[str, str]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Ajusta un candidato solo con inner-train y evalúa el fold exterior una vez."""
    spec = CANDIDATES[candidate_name]
    seed_all(seed)
    # Copia explícita a tensores ordinarios: los caches se produjeron bajo inference_mode.
    x = torch.from_numpy(np.array(vectors_np, dtype=np.float32, copy=True)).clone()
    if torch.is_inference(x) or not torch.isfinite(x).all():
        raise E13Error("El tensor de embeddings debe ser ordinario y finito antes de autograd.")
    y_np = data.energy_dft_ev_atom.to_numpy(dtype=np.float32)
    if not np.isfinite(y_np).all():
        raise E13Error("Etiqueta DFT no finita.")
    y = torch.from_numpy(np.array(y_np, copy=True)).clone()
    # Ninguna media, desviación o escala incluye inner-val, outer-val ni histórico test.
    target_mean = float(y_np[train_ids].mean())
    target_scale = max(float(y_np[train_ids].std()), 1e-6)
    pair_differences: list[float] = []
    for _, group in data.iloc[train_ids].groupby("composition_group", sort=True):
        values = group.energy_dft_ev_atom.to_numpy(float)
        pair_differences.extend(float(values[i] - values[j]) for i in range(len(values)) for j in range(i + 1, len(values)))
    pair_scale = max(float(np.sqrt(np.mean(np.square(pair_differences)))) if pair_differences else 0.0, 0.001)
    formulas = data.composition_group.astype(str).tolist()
    model = Head(int(spec["width"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise E13Error("La cabeza debe tener todos sus parámetros entrenables.")
    optimizer_rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_state, waited = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for batch_ids in composition_batches(data, train_ids, optimizer_rng):
            batch_formulas = [formulas[i] for i in batch_ids]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[batch_ids]) * target_scale + target_mean
            loss = objective(prediction, y[batch_ids], batch_formulas, target_scale, pair_scale, float(spec["pairwise_lambda"]))
            if not torch.isfinite(loss):
                raise E13Error(f"Loss no finita durante train: {candidate_name}/fold{fold}/seed{seed}/epoch{epoch}.")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            inner_prediction = model(x[inner_val_ids]) * target_scale + target_mean
            val_loss = float(objective(inner_prediction, y[inner_val_ids], [formulas[i] for i in inner_val_ids],
                                       target_scale, pair_scale, float(spec["pairwise_lambda"])))
        if not np.isfinite(val_loss):
            raise E13Error(f"Inner validation loss no finita: {candidate_name}/fold{fold}/seed{seed}.")
        history.append({"configuration": candidate_name, "fold": fold, "seed": seed, "epoch": epoch,
            "train_batch_objective_mean": float(np.mean(train_losses)), "inner_validation_objective": val_loss,
            "inner_train_compositions": int(data.iloc[train_ids].composition_group.nunique()),
            "inner_validation_compositions": int(data.iloc[inner_val_ids].composition_group.nunique()),
            "target_mean_inner_train_ev_atom": target_mean, "target_scale_inner_train_ev_atom": target_scale,
            "pair_scale_inner_train_ev_atom": pair_scale})
        # Igual que E1.2: selecciona el mínimo objetivo normalizado de validation.
        if val_loss < best_loss - 1e-12:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if waited >= PATIENCE:
            break
    if best_state is None:
        raise E13Error("Early stopping no produjo checkpoint seleccionado.")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    checkpoint_path = output / "checkpoints" / candidate_name / f"fold_{fold}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": best_state, "candidate": candidate_name, "fold": fold, "seed": seed,
        "pooling": spec["pooling"], "hidden_dim": spec["width"], "pairwise_lambda": spec["pairwise_lambda"],
        "dropout": 0.1, "best_epoch": best_epoch, "best_inner_validation_objective": best_loss,
        "target_mean_inner_train_ev_atom": target_mean, "target_scale_inner_train_ev_atom": target_scale,
        "pair_scale_inner_train_ev_atom": pair_scale, "dataset_sha256": input_hashes["dataset"],
        "embedding_sha256": input_hashes[spec["pooling"]], "backbone_sha256": input_hashes["checkpoint"],
        "backbone_frozen": True, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "training_config": {"optimizer": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_compositions": COMPOSITIONS_PER_BATCH,
            "absolute_huber_beta": 1.0, "pairwise_huber_beta": 1.0,
            "pairwise_loss_averaging": "mean per composition, only inner-train batches",
            "checkpoint_selection": "minimum inner-validation objective", "device": "cpu"}}
    torch.save(payload, checkpoint_path)
    # Solo ahora se evalúa el outer-validation, una vez fijado el epoch.
    with torch.inference_mode():
        outer_prediction = (model(x[outer_ids]) * target_scale + target_mean).cpu().numpy().astype(float)
    if not np.isfinite(outer_prediction).all():
        raise E13Error(f"Outer predictions no finitas para {candidate_name}/fold{fold}/seed{seed}.")
    pred = data.iloc[outer_ids][["structure_id", "composition_group", "polymorph_family", "energy_dft_ev_atom", "reduced_formula"]].copy()
    pred["configuration"] = candidate_name
    pred["fold"] = fold
    pred["seed"] = seed
    pred["pooling"] = spec["pooling"]
    pred["head"] = spec["head"]
    pred["pairwise_lambda"] = spec["pairwise_lambda"]
    pred["best_epoch"] = best_epoch
    pred["energy_pred_ev_atom"] = outer_prediction
    pred["error_ev_atom"] = pred.energy_pred_ev_atom - pred.energy_dft_ev_atom
    pred["delta_e_dft_mev_atom"] = pred.energy_dft_ev_atom.groupby(pred.composition_group).transform(lambda v: (v-v.min())*1000)
    pred["delta_e_pred_mev_atom"] = pred.energy_pred_ev_atom.groupby(pred.composition_group).transform(lambda v: (v-v.min())*1000)
    pred["historical_split"] = data.iloc[outer_ids].split.to_numpy()
    return history, pred


def run(out_dir: Path) -> None:
    """Ejecuta las 45 cabezas y preserva fragmentos por ejecución."""
    if not out_dir.is_dir() or not (out_dir / "fold_assignments.csv").is_file():
        raise E13Error("Corre prepare_group_folds.py antes de entrenar.")
    if (out_dir / "oof_predictions.csv").exists() or (out_dir / "checkpoints").exists() or (out_dir / "prediction_fragments").exists():
        raise E13Error("El destino ya tiene artefactos de entrenamiento; conserva esa corrida y prepara un --out-dir nuevo.")
    data, caches, provenance = load_inputs()
    assignments = load_fold_assignments(out_dir / "fold_assignments.csv")
    if set(assignments.structure_id.astype(str)) != set(data.structure_id.astype(str)):
        raise E13Error("Fold assignments no cubre exactamente los IDs actuales.")
    if (assignments.historical_split.eq("test") & assignments.outer_role.ne("historical_test_excluded")).any():
        raise E13Error("El test histórico no está excluido de todos los folds.")
    # Quita las filas test antes de formar tensores/targets; no entran en fit ni evaluación.
    dev_mask = data.split.isin(["train", "val"]).to_numpy()
    data = data.loc[dev_mask].reset_index(drop=True)
    caches = {key: matrix[dev_mask].copy() for key, matrix in caches.items()}
    assignments = assignments[assignments.historical_split.isin(["train", "val"])].copy().reset_index(drop=True)
    if len(data) != 210 or set(data.split.astype(str)) != {"train", "val"}:
        raise E13Error("La matriz de entrenamiento debe contener solo 210 filas de desarrollo, sin test.")
    hashes = {"dataset": sha256_file(DATASET), "checkpoint": sha256_file(__import__("common").CHECKPOINT),
              "last_token": provenance["embed_audits"]["last_token"]["cache_sha256"],
              "masked_mean": provenance["embed_audits"]["masked_mean"]["cache_sha256"]}
    histories: list[dict[str, Any]] = []
    for candidate_name, spec in CANDIDATES.items():
        matrix = caches[spec["pooling"]]
        for fold in range(5):
            fold_assign = assignments[assignments.outer_fold.eq(fold)].set_index("structure_id")
            train_ids = np.flatnonzero(data.structure_id.astype(str).map(fold_assign.inner_role).eq("inner_train").to_numpy())
            inner_val_ids = np.flatnonzero(data.structure_id.astype(str).map(fold_assign.inner_role).eq("inner_validation").to_numpy())
            outer_ids = np.flatnonzero(data.structure_id.astype(str).map(fold_assign.outer_role).eq("outer_validation").to_numpy())
            train_groups = set(data.iloc[train_ids].composition_group)
            inner_val_groups = set(data.iloc[inner_val_ids].composition_group)
            outer_groups = set(data.iloc[outer_ids].composition_group)
            if train_groups & inner_val_groups or train_groups & outer_groups or inner_val_groups & outer_groups:
                raise E13Error(f"Fuga de composición entre inner-train/inner-val/outer-val en fold {fold}.")
            if (len(train_groups), len(inner_val_groups), len(outer_groups)) != (45, 11, 14):
                raise E13Error(f"Conteos de grupos incorrectos en fold {fold}: train={len(train_groups)}, inner-val={len(inner_val_groups)}, outer-val={len(outer_groups)}.")
            for seed in SEEDS:
                history, predictions = train_one(data, matrix, candidate_name, fold, seed, train_ids, inner_val_ids,
                                                  outer_ids, out_dir, hashes)
                histories.extend(history)
                pred_dir = out_dir / "prediction_fragments"
                pred_dir.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(pred_dir / f"{candidate_name}_fold_{fold}_seed_{seed}.csv", index=False)
                history_dir = out_dir / "training_history_fragments"
                history_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(history_dir / f"{candidate_name}_fold_{fold}_seed_{seed}.csv", index=False)
                log(LOGGER, logging.INFO, "e13_run_complete", configuration=candidate_name, fold=fold, seed=seed,
                    best_epoch=min(history, key=lambda row: row["inner_validation_objective"])["epoch"] if history else None,
                    outer_rows=len(predictions), test_evaluated=False)
    # Solo se fusionan cuando las 45 ejecuciones terminaron.
    pred_parts = sorted((out_dir / "prediction_fragments").glob("*.csv"))
    if len(pred_parts) != 45:
        raise E13Error(f"Se requieren 45 fragmentos OOF, hallados {len(pred_parts)}.")
    pd.concat([pd.read_csv(path) for path in pred_parts], ignore_index=True).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(out_dir / "training_history.csv", index=False)
    run_records = []
    for candidate_name in CANDIDATES:
        for fold in range(5):
            for seed in SEEDS:
                paths = [out_dir / "checkpoints" / candidate_name / f"fold_{fold}_seed_{seed}.pt",
                         out_dir / "prediction_fragments" / f"{candidate_name}_fold_{fold}_seed_{seed}.csv",
                         out_dir / "training_history_fragments" / f"{candidate_name}_fold_{fold}_seed_{seed}.csv"]
                if not all(path.is_file() for path in paths):
                    raise E13Error(f"Falta archivo de una ejecución {candidate_name}/fold{fold}/seed{seed}.")
                run_records.append({"configuration": candidate_name, "fold": fold, "seed": seed,
                    "checkpoint": str(paths[0].relative_to(out_dir)), "checkpoint_sha256": sha256_file(paths[0]),
                    "outer_predictions": str(paths[1].relative_to(out_dir)), "outer_predictions_sha256": sha256_file(paths[1]),
                    "inner_training_history": str(paths[2].relative_to(out_dir)), "inner_training_history_sha256": sha256_file(paths[2])})
    manifest = {"status": "PASS", "completed_fits": 45, "candidates": list(CANDIDATES), "folds": 5,
        "seeds": list(SEEDS), "test_evaluated": False, "input_hashes": hashes,
        "oof_sha256": sha256_file(out_dir / "oof_predictions.csv"), "checkpoint_count": len(list((out_dir / "checkpoints").rglob("*.pt"))),
        "runs": run_records,
        "training_script_sha256": sha256_file(Path(__file__)), "backbone_trainable_parameters": 0}
    write_json(out_dir / "training_manifest.json", manifest)
    log(LOGGER, logging.INFO, "e13_training_complete", fits=45, oof_rows=len(pd.read_csv(out_dir / "oof_predictions.csv")),
        test_evaluated=False, out_dir=out_dir.resolve())


def main() -> None:
    """Valida la ruta de salida y lanza el conjunto fijo de ejecuciones E1.3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="E1.3 usa cabezas pequeñas sobre cachés locales; solo CPU está habilitado.")
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except (E13Error, OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        log(LOGGER, logging.ERROR, "e13_training_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
