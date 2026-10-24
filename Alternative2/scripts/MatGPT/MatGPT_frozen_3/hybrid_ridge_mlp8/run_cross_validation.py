#!/usr/bin/env python3
"""Ejecuta recombinación H3-A, Ridge nuevo H3-B y CV híbrida H3-C."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import (ALPHAS, CHECKPOINT, EXPECTED_CHECKPOINT_SHA, FEATURES, FOLDS,
                    FAMILIES, H2_ROOT, H3Error, REPORT_DIR, R4_ROOT, SEEDS,
                    SCRIPT_DIR, HISTORICAL_DEV_SHA, MODEL_INFO, SOURCE_MANIFEST,
                    load_development_without_parsing_test_targets,
                    load_features, load_fold_manifest, log, ordered_compositions,
                    sha256_file, split_indices, write_json)
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import StructuralMLP8, centered_smooth_l1, fit_ridge_by_validation, make_ridge

OUT_A = REPORT_DIR / "H3_A_recombination"
OUT_B = REPORT_DIR / "H3_B_ridge_fixed_delta"
OUT_C = REPORT_DIR / "H3_C_ridge_retrained_delta"


def _prediction_frame(frame: pd.DataFrame, model: str, seed_override: int | None = None) -> pd.DataFrame:
    """Normaliza columnas de un OOF histórico al esquema común H3."""
    part = frame.copy()
    part["model"] = model
    if seed_override is not None:
        part["seed"] = seed_override
    elif "seed" not in part:
        part["seed"] = -1
    part["seed"] = pd.to_numeric(part.seed, errors="raise").astype(int)
    part["fold"] = part.fold.astype(int)
    if "E_DFT_eV_atom" not in part:
        raise H3Error(f"{model}: no hay columna E_DFT_eV_atom.")
    part["E_pred_eV_atom"] = pd.to_numeric(part.E_pred_eV_atom, errors="coerce")
    return part[["structure_id", "composition_id", "polymorph_family", "fold", "seed", "model",
                 "E_DFT_eV_atom", "E_pred_eV_atom"]].copy()


def load_baseline_predictions() -> pd.DataFrame:
    """Carga predicciones OOF existentes solo después del preflight de integridad."""
    enriched = pd.read_csv(H2_ROOT / "oof_predictions_enriched.csv")
    h0 = _prediction_frame(enriched.loc[enriched.model.eq("H0")], "H0")
    h2 = _prediction_frame(enriched.loc[enriched.model.eq("H2")], "H2")
    r4all = pd.read_csv(R4_ROOT / "oof_predictions.csv")
    r4 = _prediction_frame(r4all.loc[r4all.model.eq("R4-SELECTED-ABSOLUTE")], "R4")
    d1all = pd.read_csv(H2_ROOT / "D1_ridge_composition_only_oof.csv")
    d1 = _prediction_frame(d1all, "D1")
    return pd.concat([h0, h2, r4, d1], ignore_index=True)


def _triplet_lookup(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Indexa filas de un OOF por composición y familia polimórfica."""
    return {str(key): part.set_index("polymorph_family").loc[list(FAMILIES)].reset_index()
            for key, part in frame.groupby("composition_id", sort=False)}


def _make_prediction_rows(data: pd.DataFrame, groups: list[str], fold: int, seed: int,
                          model: str, mu_values: np.ndarray, delta_values: np.ndarray) -> list[dict[str, Any]]:
    """Construye tripletes OOF y valida identidades de reconstrucción."""
    by_comp = data.set_index(["composition_id", "polymorph_family"])
    if len(mu_values) != len(groups) or not np.isfinite(mu_values).all():
        raise H3Error(f"{model}: salida de media composicional inválida.")
    rows = []
    for index, composition in enumerate(groups):
        mu = float(mu_values[index])
        delta = np.asarray(delta_values[index], dtype=float)
        if abs(float(delta.sum())) > 1e-7 or not np.isfinite(delta).all():
            raise H3Error(f"{model}: residuos no centrados o no finitos para {composition}.")
        for family_index, family in enumerate(FAMILIES):
            source = by_comp.loc[(composition, family)]
            rows.append({"structure_id": str(source.structure_id), "composition_id": composition,
                         "polymorph_family": family, "fold": int(fold), "seed": int(seed), "model": model,
                         "E_DFT_eV_atom": float(source.E_DFT_eV_atom), "E_pred_eV_atom": mu + float(delta[family_index]),
                         "E_pred_mu_eV_atom": mu, "delta_pred_centered_eV_atom": float(delta[family_index])})
    return rows


def run_recombination(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula H3-A y H3-B sin entrenar la MLP8 histórica.

    Returns:
        OOF concatenado A/B y diagnósticos por fold/semilla.
    """
    enriched = pd.read_csv(H2_ROOT / "oof_predictions_enriched.csv")
    h2 = enriched.loc[enriched.model.eq("H2")].copy()
    checkpoint_residual_path = REPORT_DIR / "provenance/H2_structural_oof_from_checkpoints.csv"
    if not checkpoint_residual_path.is_file():
        raise H3Error("Ejecuta audit_previous_results.py antes: falta reproducción de las MLP8 H2 guardadas.")
    h2_residual = pd.read_csv(checkpoint_residual_path)
    r4all = pd.read_csv(R4_ROOT / "oof_predictions.csv")
    r4 = r4all.loc[r4all.model.eq("R4-SELECTED-ABSOLUTE")].copy()
    cache, _ = load_features(data.structure_id.tolist())
    folds = load_fold_manifest(set(data.structure_id))
    (REPORT_DIR / "comparative_analysis").mkdir(parents=True, exist_ok=True)
    b_rows: list[dict[str, Any]] = []; a_rows: list[dict[str, Any]] = []; diagnostics = []; alpha_rows = []

    for fold in range(5):
        indices = np.flatnonzero(folds.loc[folds.outer_fold.eq(fold)].set_index("structure_id").reindex(data.structure_id).outer_role.eq("outer_validation").to_numpy())
        group_names, _, _ = ordered_compositions(data, indices)
        outer_h2 = h2.loc[h2.fold.eq(fold)]
        outer_h2_residual = h2_residual.loc[h2_residual.fold.eq(fold)]
        outer_r4 = r4.loc[r4.fold.eq(fold)]
        h2_by_seed = {int(seed): _triplet_lookup(part) for seed, part in outer_h2.groupby("seed")}
        h2_delta_by_seed = {int(seed): _triplet_lookup(part) for seed, part in outer_h2_residual.groupby("seed")}
        r4_lookup = _triplet_lookup(outer_r4)
        for seed in SEEDS:
            delta_rows_h2, delta_rows_a = [], []
            mu_a, delta_a = [], []
            for composition in group_names:
                h2_triplet = h2_by_seed[seed][composition]
                p_h2 = h2_triplet.E_pred_eV_atom.to_numpy(float)
                centered_h2 = h2_triplet.delta_pred_centered_eV_atom.to_numpy(float)
                checkpoint_delta = h2_delta_by_seed[seed][composition].delta_pred_centered_eV_atom.to_numpy(float)
                if np.max(np.abs(checkpoint_delta - centered_h2)) > 1e-6 or np.max(np.abs(centered_h2 - (p_h2 - p_h2.mean()))) > 1e-6:
                    raise H3Error(f"Reproducción de residuo H2 no coincide con OOF para {composition}.")
                r4_triplet = r4_lookup[composition]
                p_r4 = r4_triplet.E_pred_eV_atom.to_numpy(float)
                mu_r4 = float(p_r4.mean())
                mu_a.append(mu_r4); delta_a.append(centered_h2)
                # B's ridge component is filled in after its fold-specific nested alpha search.
                delta_rows_h2.append(centered_h2)
            a_rows.extend(_make_prediction_rows(data, group_names, fold, seed, "H3-A", np.asarray(mu_a), np.asarray(delta_a)))
            diagnostics.append({"model": "H3-A", "fold": fold, "seed": seed,
                                "ranking_invariant_to_H2": True,
                                "max_abs_delta_difference_vs_H2_eV_atom": 0.0,
                                "R4_source_seed": -1})

        # One selected/refit composition Ridge per fold; same predictions are repeated for each H2 seed.
        inner_train, inner_valid, outer = split_indices(data, folds, fold)
        groups_tr, x_tr_trip, y_tr = ordered_compositions(data, inner_train, cache["L03_MEAN"])
        groups_va, x_va_trip, y_va = ordered_compositions(data, inner_valid, cache["L03_MEAN"])
        groups_out, x_out_trip, _ = ordered_compositions(data, outer, cache["L03_MEAN"])
        xtr, ytr = x_tr_trip.mean(axis=1), y_tr.mean(axis=1)
        xva, yva = x_va_trip.mean(axis=1), y_va.mean(axis=1)
        alpha, inner_mae, _, fold_alpha_scores = fit_ridge_by_validation(xtr, ytr, xva, yva, ALPHAS)
        alpha_rows.extend({"fold": fold, **item} for item in fold_alpha_scores)
        outer_train = np.concatenate([inner_train, inner_valid])
        groups_fit, x_fit_trip, y_fit = ordered_compositions(data, outer_train, cache["L03_MEAN"])
        if set(groups_fit) & set(groups_out) or len(groups_fit) != 56 or len(groups_out) != 14:
            raise H3Error(f"Fold {fold}: Ridge recibió grupos no permitidos.")
        pipeline = make_ridge(alpha)
        pipeline.fit(x_fit_trip.mean(axis=1), y_fit.mean(axis=1))
        mu_out = pipeline.predict(x_out_trip.mean(axis=1))
        ridge_dir = OUT_B / "ridge_by_fold" / f"fold_{fold}"
        ridge_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ridge_dir / "ridge.joblib")
        scaler = pipeline.named_steps["scaler"]; ridge = pipeline.named_steps["ridge"]
        ridge_info = {"fold": fold, "selected_alpha": alpha, "inner_validation_mae_ev_atom": inner_mae,
                      "inner_train_compositions": len(groups_tr), "inner_validation_compositions": len(groups_va),
                      "outer_training_compositions_refit": len(groups_fit), "outer_validation_compositions": len(groups_out),
                      "feature_dimension": 768, "one_row_per_composition": True,
                      "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                      "ridge_coefficients_scaled_space": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
                      "fit_compositions": sorted(groups_fit), "validation_compositions": sorted(groups_va),
                      "outer_compositions": sorted(groups_out), "test_used": False}
        from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import write_json
        write_json(ridge_dir / "ridge_metadata.json", ridge_info)
        for seed in SEEDS:
            h2_lookup = h2_by_seed[seed]
            h2_delta_lookup = h2_delta_by_seed[seed]
            deltas = []
            for composition in groups_out:
                delta = h2_lookup[composition].delta_pred_centered_eV_atom.to_numpy(float)
                checkpoint_delta = h2_delta_lookup[composition].delta_pred_centered_eV_atom.to_numpy(float)
                p = h2_lookup[composition].E_pred_eV_atom.to_numpy(float)
                if np.max(np.abs(delta - checkpoint_delta)) > 1e-6 or np.max(np.abs(delta - (p - p.mean()))) > 1e-6:
                    raise H3Error(f"Checkpoint de residuo H2 no alinea con OOF para {composition}.")
                deltas.append(delta)
            b_rows.extend(_make_prediction_rows(data, groups_out, fold, seed, "H3-B", mu_out, np.asarray(deltas)))
            diagnostics.append({"model": "H3-B", "fold": fold, "seed": seed, "selected_alpha": alpha,
                                "inner_validation_mae_ev_atom": inner_mae, "ranking_invariant_to_H2": True,
                                "max_abs_delta_difference_vs_H2_eV_atom": 0.0})
    a = pd.DataFrame(a_rows); b = pd.DataFrame(b_rows)
    if len(a) != 630 or len(b) != 630:
        raise H3Error("H3-A/B deben tener 630 OOF cada uno (210 IDs × 3 semillas).")
    a.to_csv(OUT_A / "oof_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(REPORT_DIR / "comparative_analysis" / "recombination_checks.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(REPORT_DIR / "comparative_analysis" / "ridge_alpha_inner_validation.csv", index=False)
    b.to_csv(OUT_B / "oof_predictions.csv", index=False)
    return pd.concat([a, b], ignore_index=True), pd.DataFrame(diagnostics)


def _train_structural_head(data: pd.DataFrame, features: np.ndarray, folds: pd.DataFrame,
                           fold: int, seed: int, device_name: str = "cpu") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Ajusta MLP8 solo con inner-train y selecciona época solo en inner-validation."""
    train, valid, outer = split_indices(data, folds, fold)
    group_train, x_train, y_train = ordered_compositions(data, train, features)
    group_valid, x_valid, y_valid = ordered_compositions(data, valid, features)
    group_outer, x_outer, y_outer = ordered_compositions(data, outer, features)
    target_train = y_train - y_train.mean(axis=1, keepdims=True)
    target_valid = y_valid - y_valid.mean(axis=1, keepdims=True)
    scale = max(float(target_train.std()), 1e-6)
    # Clone NumPy arrays into ordinary tensors (the source cache has inference provenance).
    device = torch.device(device_name)
    Xtr = torch.tensor(np.array(x_train, copy=True), dtype=torch.float32, device=device)
    Ytr = torch.tensor(np.array(target_train, copy=True), dtype=torch.float32, device=device)
    Xva = torch.tensor(np.array(x_valid, copy=True), dtype=torch.float32, device=device)
    Yva = torch.tensor(np.array(target_valid, copy=True), dtype=torch.float32, device=device)
    Xout = torch.tensor(np.array(x_outer, copy=True), dtype=torch.float32, device=device)
    if any(torch.is_inference(t) or t.requires_grad for t in (Xtr, Ytr, Xva, Yva, Xout)):
        raise H3Error("Features/targets deben ser tensores ordinarios sin gradiente.")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = StructuralMLP8(dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best_state, best_epoch, waited = float("inf"), None, 0, 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, 201):
        model.train(); losses = []
        order = rng.permutation(len(Xtr))
        for begin in range(0, len(order), 8):
            ids = order[begin:begin + 8]
            pred = model(Xtr[ids])
            loss = centered_smooth_l1(pred, Ytr[ids], scale, beta=1.0)
            if not torch.isfinite(loss):
                raise H3Error(f"H3-C loss no finita fold={fold} seed={seed} epoch={epoch}.")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm = sum(float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None)
            if grad_norm <= 0:
                raise H3Error("No llega gradiente a MLP8 estructural.")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = float(centered_smooth_l1(val_pred, Yva, scale, beta=1.0).cpu())
            val_mae = float(torch.mean(torch.abs(val_pred - Yva)).cpu() * 1000)
        history.append({"fold": fold, "seed": seed, "epoch": epoch, "train_loss_delta": float(np.mean(losses)),
                        "inner_validation_loss_delta": val_loss, "inner_validation_MAE_delta_mev_atom": val_mae,
                        "scale_delta_ev_atom": scale, "learning_rate": 1e-3, "weight_decay": 1e-4,
                        "dropout": 0.1, "grad_norm_mlp8": grad_norm})
        if val_loss < best_score - 1e-12:
            best_score, best_epoch, waited = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if waited >= 20:
            break
    if best_state is None:
        raise H3Error("H3-C no produjo checkpoint seleccionado por inner-validation.")
    model.load_state_dict(best_state, strict=True); model.eval()
    before = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    with torch.no_grad():
        delta_out = (model(Xout).cpu().numpy() * 1.0).astype(np.float64)
    # Checkpoint output is in eV/atom units: the model is trained against raw deltas
    # normalized in the loss, not rescaled at inference.
    state_after = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    if before != state_after or not np.isfinite(delta_out).all():
        raise H3Error("Pesos MLP cambiaron durante inferencia o salida no finita.")
    run_dir = OUT_C / "runs" / f"fold_{fold}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - start
    torch.save({"state_dict": best_state, "fold": fold, "seed": seed, "architecture": [768, 8, 1],
                "dropout": .1, "target_scale_ev_atom": scale, "best_epoch": best_epoch,
                "best_inner_validation_loss": best_score, "backbone_frozen": True,
                "inner_train_compositions": sorted(group_train), "inner_validation_compositions": sorted(group_valid)}, run_dir / "structural_mlp8.pt")
    history_frame = pd.DataFrame(history); history_frame.to_csv(run_dir / "training_history.csv", index=False)
    rows = _make_prediction_rows(data, group_outer, fold, seed, "H3-C", np.zeros(len(group_outer)), delta_out)
    # Mean is joined later from the fold Ridge model; do not use DFT labels to make predictions.
    predictions = pd.DataFrame(rows)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metadata = {"status": "PASS", "fold": fold, "seed": seed, "best_epoch": best_epoch,
                "epochs_run": len(history), "training_seconds": elapsed,
                "inner_validation_objective": best_score, "scale_delta_ev_atom": scale,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "backbone_trainable_parameters": 0, "test_used": False,
                "inner_train_compositions": sorted(group_train), "inner_validation_compositions": sorted(group_valid),
                "outer_compositions": sorted(group_outer)}
    write_json(run_dir / "status.json", metadata)
    return predictions, history_frame, metadata


def run_h3c(data: pd.DataFrame, features: dict[str, np.ndarray], folds: pd.DataFrame,
            folds_arg: tuple[int, ...], seeds_arg: tuple[int, ...], device: str) -> pd.DataFrame:
    """Ejecuta o reanuda corridas H3-C y combina con Ridge outer fold."""
    all_frames = []
    ridge_outputs = pd.read_csv(OUT_B / "ridge_fold_predictions.csv")
    for fold in folds_arg:
        for seed in seeds_arg:
            run_dir = OUT_C / "runs" / f"fold_{fold}_seed_{seed}"
            status_path = run_dir / "status.json"
            pred_path = run_dir / "predictions.csv"
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") != "PASS" or not pred_path.is_file() or not (run_dir / "structural_mlp8.pt").is_file():
                    raise H3Error(f"Corrida H3-C incompleta; conservada para inspección: {run_dir}")
                delta_frame = pd.read_csv(pred_path)
            else:
                # Record input hashes before this run starts. Dataset SHA is inherited from
                # the already audited E1 manifest; development labels are hashed separately.
                dev_hash = hashlib.sha256(pd.util.hash_pandas_object(
                    data[["structure_id", "composition_id", "polymorph_family", "E_DFT_eV_atom"]], index=False).values.tobytes()).hexdigest()
                run_manifest = {"status": "PREPARED", "model": "H3-C", "fold": fold, "seed": seed,
                                "historical_dataset_sha256_from_prior_audit": HISTORICAL_DEV_SHA,
                                "development_targets_hash_sha256": dev_hash,
                                "fold_manifest_sha256": sha256_file(FOLDS), "feature_cache_sha256": sha256_file(FEATURES),
                                "checkpoint_sha256": sha256_file(CHECKPOINT),
                                "tokenizer_model_info_sha256": sha256_file(MODEL_INFO),
                                "pinned_source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
                                "pinned_source_commit": "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e",
                                "training_script_sha256": sha256_file(Path(__file__)),
                                "model_script_sha256": sha256_file(Path(__file__).with_name("hybrid_predictor.py")),
                                "config_sha256": sha256_file(Path(__file__).with_name("config.yaml")),
                                "test_used": False}
                write_json(run_dir / "run_manifest.json", run_manifest)
                delta_frame, history, status = _train_structural_head(data, features["FINAL_END"], folds, fold, seed, device)
                run_manifest.update(status); run_manifest["status"] = "PASS"
                run_manifest["structural_mlp8_sha256"] = sha256_file(run_dir / "structural_mlp8.pt")
                write_json(run_dir / "run_manifest.json", run_manifest)
                log("h3c_head_complete", fold=fold, seed=seed, rows=len(delta_frame),
                    epochs=status["epochs_run"], seconds=status["training_seconds"], test_used=False)
            ridge_fold = ridge_outputs.loc[(ridge_outputs.fold.eq(fold)) & (ridge_outputs.seed.eq(-1))]
            delta_frame = delta_frame.drop(columns=["E_pred_mu_eV_atom", "E_pred_eV_atom"], errors="ignore")
            joined = delta_frame.merge(ridge_fold[["structure_id", "E_pred_mu_eV_atom"]], on="structure_id", validate="one_to_one")
            joined["E_pred_eV_atom"] = joined.E_pred_mu_eV_atom + joined.delta_pred_centered_eV_atom
            joined["model"] = "H3-C"
            if len(joined) != 42 or joined.structure_id.nunique() != 42:
                raise H3Error(f"H3-C fold {fold} seed {seed}: outer prediction count invalid.")
            all_frames.append(joined)
    out = pd.concat(all_frames, ignore_index=True)
    if out.E_pred_eV_atom.isna().any() or not np.isfinite(out.E_pred_eV_atom).all():
        raise H3Error("Predicciones H3-C contienen NaN/Inf.")
    return out


def _write_ridge_fold_predictions(data: pd.DataFrame) -> None:
    """Materializa medias Ridge deterministas para merges de H3-C."""
    path = OUT_B / "oof_predictions.csv"
    ridge_rows = []
    cache, _ = load_features(data.structure_id.tolist())
    folds = load_fold_manifest(set(data.structure_id))
    for fold in range(5):
        fit = joblib.load(OUT_B / "ridge_by_fold" / f"fold_{fold}" / "ridge.joblib")
        _, _, outer = split_indices(data, folds, fold)
        groups, xtrip, y = ordered_compositions(data, outer, cache["L03_MEAN"])
        mu = fit.predict(xtrip.mean(axis=1))
        for composition, value in zip(groups, mu):
            group = data.loc[data.composition_id.eq(composition)]
            for row in group.itertuples(index=False):
                ridge_rows.append({"structure_id": row.structure_id, "composition_id": composition,
                                   "polymorph_family": row.polymorph_family, "fold": fold, "seed": -1,
                                   "model": "H3-RIDGE", "E_DFT_eV_atom": row.E_DFT_eV_atom,
                                   "E_pred_mu_eV_atom": float(value), "E_pred_eV_atom": float(value),
                                   "delta_pred_centered_eV_atom": 0.0})
    pd.DataFrame(ridge_rows).to_csv(OUT_B / "ridge_fold_predictions.csv", index=False)


def run_audit_and_ab(data: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
    """Audita y produce H3-A/B OOF antes de cualquier entrenamiento H3-C."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_A).mkdir(parents=True, exist_ok=True); OUT_B.mkdir(parents=True, exist_ok=True)
    combined, _ = run_recombination(data)
    _write_ridge_fold_predictions(data)
    return combined


def main() -> None:
    """Punto de entrada para fase audit, recombine, pilot o CV completa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("recombine", "pilot", "cv"), required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    args = parser.parse_args()
    try:
        data = load_development_without_parsing_test_targets()
        folds = load_fold_manifest(set(data.structure_id))
        features, _ = load_features(data.structure_id.tolist())
        if args.phase == "recombine":
            run_audit_and_ab(data, args.device)
            print("H3-A/B OOF completadas; aún no se entrenó H3-C.")
            return
        if not (OUT_B / "ridge_fold_predictions.csv").is_file():
            run_audit_and_ab(data, args.device)
        folds_arg, seeds_arg = ((0,), (42,)) if args.phase == "pilot" else (tuple(range(5)), SEEDS)
        out = run_h3c(data, features, folds, folds_arg, seeds_arg, args.device)
        pred_path = OUT_C / ("pilot_predictions.csv" if args.phase == "pilot" else "oof_predictions.csv")
        out.to_csv(pred_path, index=False)
        print(json.dumps({"status": "PASS", "phase": args.phase, "rows": len(out),
                          "runs": out[["fold", "seed"]].drop_duplicates().shape[0],
                          "test_used": False, "predictions": str(pred_path)}, indent=2))
    except (H3Error, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"H3 {args.phase} FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
