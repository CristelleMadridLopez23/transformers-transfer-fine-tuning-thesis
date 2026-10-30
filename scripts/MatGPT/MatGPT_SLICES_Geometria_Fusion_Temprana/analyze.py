"""Evalúa las fusiones geométricas temprana y tardía en ABS3.

Genera predicciones OOF, métricas, bootstrap agrupado por fórmula, controles
de geometría permutada, figuras PNG y reporte científico. No abre el test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress, spearmanr, kendalltau

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/MatGPT_SLICES_Geometria_Fusion_Temprana"
REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria_Fusion_Temprana"
IMAGES = ROOT / "images/MatGPT_SLICES_Geometria_Fusion_Temprana"
SCRIPT = Path(__file__).resolve().parent
LOG = logging.getLogger("matgpt_early_geometry_analysis")
SEEDS = (42, 123, 2026)
CONDITIONS = ("F0_SLICES", "F0_early", "F0_late", "F8_SLICES", "F8_early", "F8_late")
PRIMARY = "F8_early"
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261030
COLORS = {"F0": "#737373", "F8": "#4c78a8", "late": "#f2a541", "early": "#59a14f",
          "F0_SLICES": "#777777", "F0_early": "#4e79a7", "F0_late": "#f28e2b",
          "F8_SLICES": "#777777", "F8_early": "#59a14f", "F8_late": "#e15759",
          "F0_permuted": "#9e9e9e", "F8_permuted": "#8cd17d"}


def depth_module() -> Any:
    """Carga el módulo histórico con dataset, folds, tokenizador y MatGPT."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("scripts.MatGPT.MatGPT_SLICES_Finetuning_Profundidad.train")


def new_module() -> Any:
    """Carga la implementación de fusión y proyección geométrica."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("scripts.MatGPT.MatGPT_SLICES_Geometria_Fusion_Temprana.train")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe un único JSON consolidado mediante reemplazo atómico."""
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (np.floating, float)) and not np.isfinite(item):
            return None
        if isinstance(item, np.integer):
            return int(item)
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, np.ndarray]:
    """Calcula media e IC95% re-muestreando fórmulas con reemplazo."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), np.array([])
    idx = rng.integers(0, len(x), size=(BOOTSTRAP_REPS, len(x)))
    reps = x[idx].mean(axis=1)
    low, high = np.quantile(reps, [0.025, 0.975])
    return float(x.mean()), float(low), float(high), reps


def pairwise_table(structures: pd.DataFrame) -> pd.DataFrame:
    """Construye diferencias firmadas entre polimorfos de igual fórmula."""
    rows: list[dict[str, Any]] = []
    keys = ["condition", "fold", "seed", "composition_id"]
    for (condition, fold, seed, formula), part in structures.groupby(keys, sort=False):
        indexed = part.set_index("polymorph_family")
        if len(indexed) != 3:
            continue
        for a, b in combinations(sorted(indexed.index), 2):
            ra, rb = indexed.loc[a], indexed.loc[b]
            rows.append({"condition": condition, "fold": fold, "seed": seed,
                "composition_id": formula, "family_a": a, "family_b": b,
                "delta_dft_mev_atom": 1000.0 * (rb.energy_per_atom_dft_ev-ra.energy_per_atom_dft_ev),
                "delta_pred_mev_atom": 1000.0 * (rb.energy_per_atom_pred_ev-ra.energy_per_atom_pred_ev)})
    return pd.DataFrame(rows)


def infer_condition(depth: Any, fusion: Any, frame: pd.DataFrame, folds: pd.DataFrame,
                    geo_raw: np.ndarray, sequences: dict[str, list[int]], model: Any,
                    stoi: dict[str, int], context: int, mode: str, condition: str,
                    permuted: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ejecuta un checkpoint sobre sus 14 fórmulas outer y registra embeddings.

    La permutación rota descriptores de celda entre los tres polimorfos de cada
    fórmula durante evaluación; no se utiliza para elegir checkpoints.
    """
    outputs: list[dict[str, Any]] = []
    repr_rows: list[dict[str, Any]] = []
    geo_index = {str(s): i for i, s in enumerate(frame.structure_id.astype(str))}
    data_idx = frame.set_index("structure_id")
    base = depth.SOURCE / "mat-uncond/weights/transformer_property_0.pth"
    for fold in range(5):
        fold_frame = folds[folds.outer_fold == fold]
        outer_ids = set(fold_frame.loc[fold_frame.outer_role == "outer_validation", "structure_id"].astype(str))
        groups = depth.groups_for(frame, outer_ids)
        for seed in SEEDS:
            checkpoint_path = EXP / "checkpoints" / f"{condition.lower()}_{mode}_fold{fold}_seed{seed}" / "best.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Falta checkpoint seleccionado: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("base_sha256") != fusion.BASE_SHA or checkpoint.get("mode") != mode:
                raise ValueError(f"Checkpoint incompatible: {checkpoint_path}")
            model.load_state_dict(torch.load(base, map_location="cpu", weights_only=True), strict=True)
            blocks = fusion.DEPTHS[condition]
            fusion.configure(model, blocks)
            for ix in blocks:
                model.transformer_decoder.layers[ix].load_state_dict(checkpoint["blocks"][str(ix)], strict=True)
            projection = fusion.GeometryProjection()
            projection.load_state_dict(checkpoint["projection"], strict=True)
            head = depth.MLP8()
            head.load_state_dict(checkpoint["head"], strict=True)
            model.eval(); projection.eval(); head.eval()
            scaler_mean = np.asarray(checkpoint["geometry_scaler_mean"], dtype=np.float64)
            scaler_scale = np.asarray(checkpoint["geometry_scaler_scale"], dtype=np.float64)
            if scaler_mean.shape != (7,) or scaler_scale.shape != (7,) or (scaler_scale <= 0).any():
                raise ValueError("Scaler geométrico seleccionado inválido")
            all_geom = ((geo_raw - scaler_mean) / scaler_scale).astype(np.float32)
            with torch.no_grad():
                for formula, ids in groups:
                    idx = np.asarray([geo_index[s] for s in ids], dtype=int)
                    current_geom = all_geom[idx].copy()
                    if permuted:
                        current_geom = np.roll(current_geom, shift=1, axis=0)
                    geo_tensor = torch.as_tensor(current_geom, dtype=torch.float32)
                    seqs = [sequences[s] for s in ids]
                    h = fusion.hidden_end(model, seqs, stoi["<pad>"], torch.device("cpu"),
                                          projection, geo_tensor, mode)
                    if permuted:
                        h_correct = fusion.hidden_end(model, seqs, stoi["<pad>"], torch.device("cpu"),
                            projection, torch.as_tensor(all_geom[idx], dtype=torch.float32), mode)
                        for j,sid in enumerate(ids):
                            repr_rows.append({"condition":f"{condition}_{mode}_permuted","fold":fold,"seed":seed,
                                "structure_id":sid,"composition_id":formula,
                                "geometry_permutation_delta_final_end_l2":float((h[j]-h_correct[j]).norm().item()),
                                "geometry_permutation_relative_l2":float((h[j]-h_correct[j]).norm().item())/
                                    max(float(h_correct[j].norm().item()),1e-12)})
                    raw_resid = head.network(h).squeeze(-1)
                    residual = raw_resid - raw_resid.mean()
                    # Baseline Ridge se reusa del control histórico para ese mismo
                    # fold/seed/ID; aquí sólo se aprende la rama residual.
                    baseline_row = depth_control[(str(condition), fold, seed)]
                    ridge = float(baseline_row.loc[ids[0], "ridge_mean_ev_atom"])
                    if not np.allclose([float(baseline_row.loc[s, "ridge_mean_ev_atom"]) for s in ids], ridge,
                                       atol=1e-10, rtol=0):
                        raise ValueError(f"Ridge no es constante intrafórmula en {formula}")
                    predictions = ridge + residual.numpy()
                    for j, sid in enumerate(ids):
                        source = data_idx.loc[sid]
                        outputs.append({"condition": f"{condition}_{mode}" + ("_permuted" if permuted else ""),
                            "fold": fold, "seed": seed, "structure_id": sid, "composition_id": formula,
                            "polymorph_family": str(source.family), "n_atoms": int(source.n_atoms),
                            "energy_total_dft_ev": float(source.total_dft_ev),
                            "energy_per_atom_dft_ev": float(source.dft_ev_atom),
                            "ridge_mean_ev_atom": ridge, "residual_pred_ev_atom": float(residual[j]),
                            "energy_per_atom_pred_ev": float(predictions[j]),
                            "energy_total_pred_ev": float(predictions[j] * int(source.n_atoms)),
                            "geometry_mode": "within_formula_permuted" if permuted else "matched"})
                    # Drift se calcula con el mismo checkpoint base y pooling FINAL_END.
                    if not permuted:
                        # Restore untouched backbone for reference embedding.
                        original_state = torch.load(base, map_location="cpu", weights_only=True)
                        model.load_state_dict(original_state, strict=True)
                        del original_state
                        h0 = depth.final_end(model, seqs, stoi["<pad>"], torch.device("cpu"))
                        model.load_state_dict(torch.load(base, map_location="cpu", weights_only=True), strict=True)
                        fusion.configure(model, blocks)
                        for ix in blocks:
                            model.transformer_decoder.layers[ix].load_state_dict(checkpoint["blocks"][str(ix)], strict=True)
                        h1 = fusion.hidden_end(model, seqs, stoi["<pad>"], torch.device("cpu"),
                            projection, torch.as_tensor(all_geom[idx], dtype=torch.float32), mode)
                        for j, sid in enumerate(ids):
                            cosine = float(torch.nn.functional.cosine_similarity(h0[j:j+1], h1[j:j+1]).item())
                            l2 = float((h1[j]-h0[j]).norm().item())
                            repr_rows.append({"condition": f"{condition}_{mode}", "fold": fold, "seed": seed,
                                "structure_id": sid, "composition_id": formula,
                                "cosine_to_pretrained": cosine,
                                "relative_l2_drift": l2/max(float(h0[j].norm()), 1e-12),
                                "embedding_norm_pretrained": float(h0[j].norm()),
                                "embedding_norm_conditioned": float(h1[j].norm())})
            del checkpoint, projection, head
            gc.collect()
    return outputs, repr_rows


def load_controls(depth: Any) -> pd.DataFrame:
    """Carga F0/F8 SLICES-only con el estimando histórico compatible."""
    data = pd.read_csv(depth.EXPERIMENT / "predicciones_oof_por_estructura.csv")
    data = data[data.condition.astype(str).isin(("F0", "F8"))].copy()
    if len(data) != 1260 or set(data.condition.astype(str)) != {"F0", "F8"}:
        raise ValueError("Controles SLICES-only F0/F8 no completos")
    data["condition"] = data.condition.map({"F0": "F0_SLICES", "F8": "F8_SLICES"})
    data["composition_id"] = data.composition_id.astype(str)
    data["geometry_mode"] = "none"
    return data


def metric_rows(structures: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calcula métricas OOF por condición con composición como unidad inferencial."""
    records: list[dict[str, Any]] = []
    formula_records: list[dict[str, Any]] = []
    for condition, part in structures.groupby("condition", sort=False):
        for (fold, seed, formula), comp in part.groupby(["fold", "seed", "composition_id"], sort=False):
            p = pairs[(pairs.condition == condition) & (pairs.fold == fold) &
                      (pairs.seed == seed) & (pairs.composition_id == formula)]
            dft = comp.energy_per_atom_dft_ev.to_numpy()*1000
            pred = comp.energy_per_atom_pred_ev.to_numpy()*1000
            resid_dft = dft-dft.mean()
            resid_pred = pred-pred.mean()
            gs_dft = int(np.argmin(dft))
            predicted_minima = np.flatnonzero(pred <= pred.min() + 1.0)
            gs_pred = int(np.argmin(pred))
            eligible = np.abs(p.delta_dft_mev_atom.to_numpy()) > 5.0
            d_pred = p.delta_pred_mev_atom.to_numpy()
            d_true = p.delta_dft_mev_atom.to_numpy()
            pair_correct = (np.sign(d_pred[eligible]) == np.sign(d_true[eligible])) & (np.abs(d_pred[eligible]) > 1.0)
            formula_records.append({"condition": condition, "fold": fold, "seed": seed,
                "composition_id": formula,
                "mae_energy_mev_atom": float(np.mean(np.abs(pred-dft))),
                "rmse_energy_mev_atom": float(np.sqrt(np.mean((pred-dft)**2))),
                "mae_total_energy_ev": float(np.mean(np.abs(comp.energy_total_pred_ev-comp.energy_total_dft_ev))),
                "mae_residual_mev_atom": float(np.mean(np.abs(resid_pred-resid_dft))),
                "rmse_residual_mev_atom": float(np.sqrt(np.mean((resid_pred-resid_dft)**2))),
                "mae_delta_signed_pairs_mev_atom": float(np.mean(np.abs(d_pred-d_true))),
                "mae_delta_zero_baseline_mev_atom": float(np.mean(np.abs(d_true))),
                "mae_delta_anchored_nonanchor_mev_atom": float(np.mean(np.abs((pred-pred[gs_dft])[np.arange(3)!=gs_dft] - (dft-dft[gs_dft])[np.arange(3)!=gs_dft]))),
                "ground_state_exact": int(len(predicted_minima) == 1 and gs_pred == gs_dft),
                "ground_state_unresolved": int(len(predicted_minima) > 1),
                "pairwise_correct": int(pair_correct.sum()), "pairwise_eligible": int(eligible.sum()),
                "pairwise_accuracy": float(pair_correct.mean()) if eligible.sum() else np.nan,
                "spearman_triplet": float(spearmanr(dft, pred).statistic),
                "kendall_triplet": float(kendalltau(dft, pred).statistic)})
        formula_seed = pd.DataFrame([r for r in formula_records if r["condition"] == condition])
        # Las semillas se promedian por composición antes del bootstrap.
        fmean = formula_seed.groupby("composition_id", as_index=False).mean(numeric_only=True)
        for metric in ("mae_energy_mev_atom", "rmse_energy_mev_atom", "mae_total_energy_ev",
                       "mae_residual_mev_atom", "rmse_residual_mev_atom",
                       "mae_delta_signed_pairs_mev_atom", "mae_delta_zero_baseline_mev_atom",
                       "mae_delta_anchored_nonanchor_mev_atom", "ground_state_exact",
                       "ground_state_unresolved", "pairwise_accuracy", "spearman_triplet", "kendall_triplet"):
            values = fmean[metric].to_numpy(dtype=float)
            # A metric with counts should be paired/macro by formula; counts remain
            # in the CSV as separate numerators and denominators.
            finite = values[np.isfinite(values)]
            records.append({"condition": condition, "estimand": "A", "metric": metric,
                "estimate": float(np.mean(finite)) if len(finite) else None,
                "n_compositions": int(len(finite)), "seeds_averaged_within_composition": 3})
    formula_df = pd.DataFrame(formula_records)
    # OLS is descriptive over all shown points; it is not an independent sample claim.
    ols = {}
    for condition, part in structures.groupby("condition", sort=False):
        x = part.energy_per_atom_dft_ev.to_numpy()*1000
        y = part.energy_per_atom_pred_ev.to_numpy()*1000
        reg = linregress(x, y)
        ols[condition] = {"n_points": int(len(part)), "slope": float(reg.slope), "intercept_mev_atom": float(reg.intercept),
            "r2_ols": float(reg.rvalue**2), "mae_mev_atom": float(np.mean(np.abs(x-y))),
            "predictive_r2": float(1-np.sum((x-y)**2)/np.sum((x-x.mean())**2)),
            "dispersion_ratio": float(np.std(y)/np.std(x))}
        residual_true=np.empty(len(part));residual_pred=np.empty(len(part))
        local=part.reset_index(drop=True)
        for _, indices in local.groupby(["fold","seed","composition_id"]).groups.items():
            ix=np.asarray(list(indices),dtype=int)
            a=local.loc[ix,"energy_per_atom_dft_ev"].to_numpy()*1000
            b=local.loc[ix,"energy_per_atom_pred_ev"].to_numpy()*1000
            residual_true[ix]=a-a.mean();residual_pred[ix]=b-b.mean()
        rreg=linregress(residual_true,residual_pred)
        ols[condition]["intraformula_residual"]={"n_points":int(len(local)),"slope":float(rreg.slope),
            "intercept_mev_atom":float(rreg.intercept),"r2_ols":float(rreg.rvalue**2),
            "mae_mev_atom":float(np.mean(np.abs(residual_pred-residual_true))),
            "dispersion_ratio":float(np.std(residual_pred)/np.std(residual_true))}
    for condition, part in pairs.groupby("condition",sort=False):
        x=part.delta_dft_mev_atom.to_numpy();y=part.delta_pred_mev_atom.to_numpy();reg=linregress(x,y)
        ols.setdefault(condition,{})["pair_delta"]={"n_pairs":int(len(part)),"slope":float(reg.slope),
            "intercept_mev_atom":float(reg.intercept),"r2_ols":float(reg.rvalue**2),
            "mae_mev_atom":float(np.mean(np.abs(x-y))),"dispersion_ratio":float(np.std(y)/np.std(x))}
    return formula_df, {"A_metrics": records, "OLS_descriptive_all_rows": ols}


def bootstrap_contrasts(formula_metrics: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Computa contrastes apareados por fórmula con bootstrap percentil."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [("F0_early", "F0_SLICES", "F0 early − F0 SLICES-only"),
        ("F8_early", "F8_SLICES", "F8 early − F8 SLICES-only"),
        ("F8_early", "F8_late", "F8 early − F8 late equiparada"),
        ("F8_early", "F0_early", "F8 early − F0 early")]
    metrics = ["mae_delta_signed_pairs_mev_atom", "mae_residual_mev_atom", "mae_energy_mev_atom",
               "ground_state_exact", "pairwise_accuracy"]
    output, details = [], []
    for left, right, label in comparisons:
        for metric in metrics:
            l = formula_metrics[formula_metrics.condition == left].groupby("composition_id")[metric].mean()
            r = formula_metrics[formula_metrics.condition == right].groupby("composition_id")[metric].mean()
            common = sorted(set(l.index) & set(r.index))
            diff = (l.loc[common]-r.loc[common]).to_numpy(dtype=float)
            estimate, lo, hi, reps = bootstrap_ci(diff, rng)
            output.append({"contrast": label, "left": left, "right": right, "metric": metric,
                "estimate_left_minus_right": estimate, "ci95_low": lo, "ci95_high": hi,
                "n_compositions": len(common), "bootstrap_replicates": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED})
            details.append({"contrast": label, "metric": metric, "estimate": estimate,
                "ci95_low": lo, "ci95_high": hi, "replicates": reps})
    # Baseline ΔE=0 is a secondary magnitude reference; it has no ranking/GS.
    for condition in ("F0_early","F0_late","F8_early","F8_late"):
        pred=formula_metrics[formula_metrics.condition==condition].groupby("composition_id").mae_delta_signed_pairs_mev_atom.mean()
        zero=formula_metrics[formula_metrics.condition==condition].groupby("composition_id").mae_delta_zero_baseline_mev_atom.mean()
        common=sorted(set(pred.index)&set(zero.index));diff=(pred.loc[common]-zero.loc[common]).to_numpy()
        estimate,lo,hi,reps=bootstrap_ci(diff,rng)
        output.append({"contrast":f"{condition} − predictor ΔE=0","left":condition,"right":"DeltaE_pred=0",
            "metric":"mae_delta_signed_pairs_mev_atom","estimate_left_minus_right":estimate,
            "ci95_low":lo,"ci95_high":hi,"n_compositions":len(common),
            "bootstrap_replicates":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,
            "adjustment":"secondary, unadjusted; no ranking or ground-state assigned"})
        details.append({"contrast":f"{condition} − predictor ΔE=0","metric":"mae_delta_signed_pairs_mev_atom",
            "estimate":estimate,"ci95_low":lo,"ci95_high":hi,"replicates":reps})
    # Interaction of depth with adding early geometry on primary delta MAE.
    def values(condition: str, metric: str) -> pd.Series:
        return formula_metrics[formula_metrics.condition == condition].groupby("composition_id")[metric].mean()
    keys = sorted(set(values("F0_early", metrics[0]).index) & set(values("F0_SLICES", metrics[0]).index) &
                  set(values("F8_early", metrics[0]).index) & set(values("F8_SLICES", metrics[0]).index))
    interaction = ((values("F8_early", metrics[0]).loc[keys]-values("F8_SLICES", metrics[0]).loc[keys])-
                   (values("F0_early", metrics[0]).loc[keys]-values("F0_SLICES", metrics[0]).loc[keys])).to_numpy()
    est, lo, hi, reps = bootstrap_ci(interaction, rng)
    output.append({"contrast": "Interacción: efecto temprano F8 − efecto temprano F0",
        "left": "(F8_early−F8_SLICES)", "right": "(F0_early−F0_SLICES)",
        "metric": metrics[0], "estimate_left_minus_right": est, "ci95_low": lo, "ci95_high": hi,
        "n_compositions": len(keys), "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED})
    details.append({"contrast": "interaccion", "metric": metrics[0], "estimate": est,
                    "ci95_low": lo, "ci95_high": hi, "replicates": reps})
    contrast_df = pd.DataFrame(output)
    primary_rows = contrast_df[contrast_df.metric == metrics[0]]
    pvals = []
    for row in primary_rows.itertuples():
        match = next(d for d in details if d["contrast"] == ("interaccion" if row.contrast.startswith("Interacción") else row.contrast)
                     and d["metric"] == metrics[0])
        reps = match["replicates"]
        p = min(1.0, 2*min(float(np.mean(reps <= 0)), float(np.mean(reps >= 0)))) if len(reps) else 1.0
        pvals.append((row.Index, p))
    # Holm correction over the four planned contrast family (three contrasts + interaction).
    planned = ["F0 early − F0 SLICES-only", "F8 early − F8 SLICES-only",
               "F8 early − F8 late equiparada", "Interacción: efecto temprano F8 − efecto temprano F0"]
    pvals = [(idx, pval) for idx, pval in pvals if contrast_df.loc[idx, "contrast"] in planned]
    sorted_p = sorted(pvals, key=lambda x: x[1])
    adjusted = {}
    for rank, (idx, pval) in enumerate(sorted_p):
        adjusted[idx] = min(1.0, max((4-j)*q for j, (_, q) in enumerate(sorted_p[:rank+1])))
    contrast_df["holm_p_primary_delta_family"] = np.nan
    for idx, pval in adjusted.items():
        contrast_df.loc[idx, "holm_p_primary_delta_family"] = pval
    return contrast_df, details


def gap_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calcula errores y ordenamiento por intervalos de |ΔE_DFT|."""
    bins = [0, 5, 10, 25, 50, 100, np.inf]
    labels = ["0–5", "5–10", "10–25", "25–50", "50–100", "100 o más"]
    data = pairs.copy()
    data["abs_gap"] = data.delta_dft_mev_atom.abs()
    data["gap_bin"] = pd.cut(data.abs_gap, bins=bins, labels=labels, right=False, include_lowest=True)
    data["absolute_error"] = (data.delta_pred_mev_atom-data.delta_dft_mev_atom).abs()
    data["zero_baseline_error"] = data.delta_dft_mev_atom.abs()
    data["eligible"] = data.abs_gap > 5
    data["pairwise_correct"] = data.eligible & (np.sign(data.delta_pred_mev_atom) == np.sign(data.delta_dft_mev_atom)) & (data.delta_pred_mev_atom.abs() > 1)
    return data.groupby(["condition", "gap_bin"], observed=False).agg(
        n_pairs=("absolute_error", "size"), mae_delta_mev_atom=("absolute_error", "mean"),
        mae_delta_zero_baseline_mev_atom=("zero_baseline_error","mean"),
        eligible_pairs=("eligible", "sum"), correct_pairs=("pairwise_correct", "sum")).reset_index()


def make_figures(structures: pd.DataFrame, pairs: pd.DataFrame, fm: pd.DataFrame,
                 contrasts: pd.DataFrame, gaps: pd.DataFrame, history: pd.DataFrame,
                 repr_df: pd.DataFrame, permuted_structures: pd.DataFrame) -> list[str]:
    """Genera como máximo diez figuras PNG legibles y con población definida."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    order = ["F0_SLICES", "F0_early", "F0_late", "F8_SLICES", "F8_early", "F8_late"]
    names = {"F0_SLICES":"F0 solo SLICES", "F0_early":"F0 fusión temprana", "F0_late":"F0 fusión tardía",
             "F8_SLICES":"F8 solo SLICES", "F8_early":"F8 fusión temprana", "F8_late":"F8 fusión tardía"}
    summary = fm.groupby("condition").mean(numeric_only=True)
    # 1: resumen de las cinco métricas principales.
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.5))
    metrics = [("mae_energy_mev_atom", "MAE E/N\n(meV/átomo)"), ("mae_residual_mev_atom", "MAE residual\n(meV/átomo)"),
        ("mae_delta_signed_pairs_mev_atom", "MAE ΔE firmada\n(meV/átomo)"), ("ground_state_exact", "Estado fundamental\n(exactitud)"),
        ("pairwise_accuracy", "Orden de pares\n(exactitud)")]
    for ax, (metric, title) in zip(axes, metrics):
        vals = [summary.loc[c, metric] if c in summary.index else np.nan for c in order]
        ax.bar(range(len(order)), vals, color=[COLORS[c] for c in order]); ax.set_xticks(range(len(order)), [c.replace("_", "\n") for c in order], rotation=0)
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.suptitle("ABS₃: comparación de SLICES y fusión geométrica", fontsize=14, weight="bold")
    fig.text(.5, .01, "Promedio primario A: primero por fórmula y semilla; 70 fórmulas, 210 estructuras, 3 polimorfos, 5 pliegues × 3 semillas. Geometría DFT estática; no es test independiente.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.92)); name="comparacion_fusion_temprana_tardia_y_slices.png"; fig.savefig(IMAGES/name,dpi=300); plt.close(fig); outputs.append(name)
    # 2: paired bootstrap contrast panel for delta and absolute energy.
    selected = contrasts[contrasts.metric.isin(["mae_delta_signed_pairs_mev_atom", "mae_energy_mev_atom"])].copy()
    selected["label"] = selected.contrast.str.replace(" − ", " vs ", regex=False)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, metric, title in zip(axes,["mae_delta_signed_pairs_mev_atom","mae_energy_mev_atom"],["ΔE firmado","Energía por átomo"]):
        d=selected[selected.metric==metric].reset_index(drop=True); y=np.arange(len(d))
        x=d.estimate_left_minus_right.to_numpy(); lo=x-d.ci95_low.to_numpy(); hi=d.ci95_high.to_numpy()-x
        ax.errorbar(x,y,xerr=[lo,hi],fmt="o",capsize=3,color="#315a7d"); ax.axvline(0,color="black",ls="--")
        ax.set_yticks(y,d.label); ax.set_title(title); ax.set_xlabel("Diferencia de MAE (meV/átomo); negativa favorece la condición izquierda")
        ax.grid(axis="x",alpha=.25)
    fig.suptitle("Contrastes apareados por fórmula: IC 95% bootstrap")
    fig.text(.5,.01,"70 fórmulas; 10 000 remuestras; semillas promediadas dentro de fórmula. Familia primaria ΔE: ajuste Holm sobre 4 contrastes.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.92)); name="efecto_fusion_temprana_intervalos_confianza.png"; fig.savefig(IMAGES/name,dpi=300); plt.close(fig); outputs.append(name)
    # Parity panels helper, one figure per target.
    def parity(target_x: np.ndarray, target_y: np.ndarray, xlab: str, ylab: str, title: str, filename: str, unit: str) -> None:
        fig, axes=plt.subplots(2,3,figsize=(14,8)); bounds=(float(min(target_x.min(),target_y.min())),float(max(target_x.max(),target_y.max())))
        for ax,c in zip(axes.flat,order):
            part=structures[structures.condition==c]; x=target_x[part.index]; y=target_y[part.index]
            reg=linregress(x,y); ax.scatter(x,y,s=8,alpha=.55,color=COLORS[c]); ax.plot(bounds,bounds,"--",color="#334e68",label="Ideal: y=x")
            grid=np.array(bounds); ax.plot(grid,reg.intercept+reg.slope*grid,"-.",color="#d1495b",label="OLS: y=mx+b")
            ax.set_xlim(bounds);ax.set_ylim(bounds);ax.set_title(names[c]);ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.grid(alpha=.2)
            mae=np.mean(np.abs(x-y)); ax.text(.04,.96,f"N={len(x)}\nm={reg.slope:.3f}; b={reg.intercept:.2f} {unit}\nR²={reg.rvalue**2:.3f}; MAE={mae:.2f} {unit}",transform=ax.transAxes,va="top",fontsize=8)
        axes.flat[0].legend(fontsize=7,loc="lower right")
        fig.suptitle(title,weight="bold");fig.text(.5,.01,"Cada punto es una predicción de estructura por pliegue/semilla: 210 estructuras físicas × 3 semillas = 630 filas/panel. Estimando A usa fórmula como unidad estadística.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(IMAGES/filename,dpi=300);plt.close(fig);outputs.append(filename)
    structures=structures.reset_index(drop=True)
    # 3 absolute parity
    parity(structures.energy_per_atom_dft_ev.to_numpy()*1000,structures.energy_per_atom_pred_ev.to_numpy()*1000,
        "DFT E/N (meV/átomo)","Predicción E/N (meV/átomo)","ABS₃: energía por átomo predicha frente a DFT",
        "paridad_energia_por_atomo_fusion_geometrica.png","meV/átomo")
    # 4 residual parity
    dft_res=np.zeros(len(structures));pred_res=np.zeros(len(structures))
    for _,idx in structures.groupby(["condition","fold","seed","composition_id"]).groups.items():
        ids=np.asarray(list(idx));dft=structures.loc[ids,"energy_per_atom_dft_ev"].to_numpy()*1000;pred=structures.loc[ids,"energy_per_atom_pred_ev"].to_numpy()*1000
        dft_res[ids]=dft-dft.mean();pred_res[ids]=pred-pred.mean()
    parity(dft_res,pred_res,"Residuo DFT intrafórmula (meV/átomo)","Residuo predicho (meV/átomo)",
        "ABS₃: residuo intrafórmula predicho frente a DFT","paridad_residuos_intraformula_fusion_geometrica.png","meV/átomo")
    # 5 signed pair parity
    fig,axes=plt.subplots(2,3,figsize=(14,8));lim=max(float(pairs.delta_dft_mev_atom.abs().max()),float(pairs.delta_pred_mev_atom.abs().max()))
    for ax,c in zip(axes.flat,order):
        q=pairs[pairs.condition==c];x=q.delta_dft_mev_atom.to_numpy();y=q.delta_pred_mev_atom.to_numpy();r=linregress(x,y)
        ax.scatter(x,y,s=8,alpha=.5,color=COLORS[c]);ax.plot([-lim,lim],[-lim,lim],"--",color="#334e68");ax.plot([-lim,lim],r.intercept+r.slope*np.array([-lim,lim]),"-.",color="#d1495b")
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_title(names[c]);ax.set_xlabel("ΔE DFT (B−A; meV/átomo)");ax.set_ylabel("ΔE predicha (B−A; meV/átomo)");ax.grid(alpha=.2)
        ax.text(.04,.96,f"N={len(x)}\nm={r.slope:.3f}; b={r.intercept:.1f}\nR²={r.rvalue**2:.3f}; MAE={np.mean(np.abs(y-x)):.1f}",transform=ax.transAxes,va="top",fontsize=8)
    fig.suptitle("ABS₃: diferencias energéticas firmadas entre pares de polimorfos",weight="bold")
    fig.text(.5,.01,"Cada punto es un par intrafórmula por pliegue/semilla; ΔE=E_B−E_A. Negativo indica orden invertido frente a DFT. Misma escala en paneles.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));name="paridad_deltaE_polimorfos_fusion_geometrica.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    # 6 error by gap
    fig,ax=plt.subplots(figsize=(11,5));gaps=gaps.copy();
    for c in order:
        q=gaps[gaps.condition==c];ax.plot(np.arange(len(q)),q.mae_delta_mev_atom,marker="o",color=COLORS[c],label=names[c])
    zero=gaps[gaps.condition=="F8_SLICES"]
    ax.plot(np.arange(len(zero)),zero.n_pairs*0+zero.mae_delta_mev_atom*0+np.nan,alpha=0)
    zero_mae=pairs.assign(err=lambda d:d.delta_dft_mev_atom.abs()).groupby(pd.cut(pairs.delta_dft_mev_atom.abs(),[0,5,10,25,50,100,np.inf],right=False,labels=False)).err.mean()
    ax.plot(range(6),zero_mae.to_numpy(),"k:x",label="Predictor ΔE=0")
    ax.set_xticks(range(6),["0–5","5–10","10–25","25–50","50–100","100+"]);ax.set_xlabel("Magnitud |ΔE DFT| (meV/átomo)");ax.set_ylabel("MAE ΔE firmada (meV/átomo)");ax.set_title("Error de diferencias por separación energética DFT");ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    fig.text(.5,.01,"Pares dentro de fórmula; 3 pares/fórmula, 5 pliegues, 3 semillas. No se evalúa ranking para |ΔE|≤5 meV/átomo. Denominadores en metricas_por_gap.csv.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));name="error_deltaE_por_magnitud_gap_y_fusion.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    # 7 pairwise accuracy by gap, eligible bins only
    fig,ax=plt.subplots(figsize=(11,5));
    rank=gaps.copy();rank["accuracy"]=np.where(rank.eligible_pairs>0,rank.correct_pairs/rank.eligible_pairs,np.nan)
    for c in order:
        q=rank[rank.condition==c];ax.plot(np.arange(len(q)),q.accuracy,marker="o",color=COLORS[c],label=names[c])
    ax.axhline(.5,color="black",ls=":",label="Azar 50%");ax.set_xticks(range(6),["0–5","5–10","10–25","25–50","50–100","100+"]);ax.set_ylim(0,1)
    ax.set_xlabel("Magnitud |ΔE DFT| (meV/átomo)");ax.set_ylabel("Aciertos / pares elegibles");ax.set_title("Ordenamiento por magnitud del gap DFT");ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    fig.text(.5,.01,"Elegible si |ΔE DFT|>5; predicción no resuelta si |ΔE|≤1 cuenta como incorrecta. Intervalo 0–5 no elegible; denominadores se exportan.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));name="ranking_por_gap_fusion_temprana_y_tardia.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    # 8 training curves from run history; averages of training batches / full validation
    if len(history):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        hist=history[(history.run_type=="full") & (history.mode.isin(["early","late"]))]
        for (condition,mode),q in hist.groupby(["condition","mode"]):
            g=q.groupby("epoch").agg(train=("train_loss","mean"),val=("inner_validation_loss","mean")).reset_index()
            lab=f"{condition} {mode}"
            axes[0].plot(g.epoch,g.train,label=lab,color=COLORS[condition+"_"+mode] if condition+"_"+mode in COLORS else None)
            axes[0].plot(g.epoch,g.val,ls="--",color=COLORS[condition+"_"+mode] if condition+"_"+mode in COLORS else None)
        axes[0].set_title("Pérdida por época");axes[0].set_xlabel("Época local");axes[0].set_ylabel("SmoothL1 residual normalizada");axes[0].legend(fontsize=7,ncol=2);axes[0].grid(alpha=.2)
        epoch_rows=[]
        for (c,m,f,s),q in hist.groupby(["condition","mode","fold","seed"]):
            best_idx=q.inner_validation_loss.idxmin();epoch_rows.append((f"{c}_{m}",int(q.loc[best_idx,"epoch"])))
        labels=list(dict.fromkeys(x[0] for x in epoch_rows))
        axes[1].boxplot([[ep for name,ep in epoch_rows if name==label] for label in labels],labels=labels)
        axes[1].set_title("Época elegida por validación interna");axes[1].set_ylabel("Época");axes[1].tick_params(axis="x",rotation=30);axes[1].grid(axis="y",alpha=.2)
        fig.suptitle("Fusión geométrica: entrenamiento y selección interna")
        fig.text(.5,.01,"Líneas continuas: promedio de pérdidas de minilote de entrenamiento; discontinuas: promedio de validación completa. Selección por mínimo SmoothL1 inner-validation; no se usó outer para seleccionar.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.92));name="curvas_entrenamiento_y_validacion_fusion_temprana.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    # 9 sensitivity using correct versus within-formula permuted geometries
    perm=permuted_structures
    correct=structures
    if len(perm):
        q=perm.merge(correct,on=["fold","seed","structure_id"],suffixes=("_perm","_true"))
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=[];vals=[];embed_vals=[]
        for c in ("F0_early","F8_early","F0_late","F8_late"):
            cp=q[q.condition_true==c];labels.append(c);vals.append(float(np.mean(np.abs(cp.energy_per_atom_pred_ev_perm-cp.energy_per_atom_pred_ev_true)*1000)))
            rp=repr_df[repr_df.condition==f"{c}_permuted"]
            embed_vals.append(float(rp.geometry_permutation_delta_final_end_l2.mean()) if len(rp) else np.nan)
        axes[0].bar(labels,embed_vals,color=[COLORS["F0"],COLORS["F8"],COLORS["F0"],COLORS["F8"]]);axes[0].set_ylabel("Cambio FINAL_END (norma L2)");axes[0].set_title("Representación contextualizada");axes[0].grid(axis="y",alpha=.2)
        axes[1].bar(labels,vals,color=[COLORS["F0"],COLORS["F8"],COLORS["F0"],COLORS["F8"]]);axes[1].set_ylabel("Cambio absoluto E/N (meV/átomo)");axes[1].set_title("Predicción energética");axes[1].grid(axis="y",alpha=.2)
        fig.suptitle("Sensibilidad al emparejamiento de geometrías con polimorfos")
        fig.text(.5,.01,"Mismos SLICES e IDs; geometrías permutadas entre los 3 polimorfos de cada fórmula solo durante evaluación. Dependencia diagnóstica, no evidencia causal de utilidad física.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.92));name="sensibilidad_final_end_a_descriptores_geometricos.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    # 10 representational drift, no causal interpretation.
    if len(repr_df):
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        q=repr_df[repr_df.condition.str.contains("early|late",regex=True) & ~repr_df.condition.str.endswith("_permuted")]
        labels=[];cos=[];drift=[]
        for cond,part in q.groupby("condition",sort=False):
            labels.append(cond);cos.append(part.cosine_to_pretrained.mean());drift.append(part.relative_l2_drift.mean())
        axes[0].bar(labels,cos,color=[COLORS.get(c,COLORS["F8"]) for c in labels]);axes[0].set_ylim(0,1.01);axes[0].set_ylabel("Coseno con FINAL_END pretrained");axes[0].tick_params(axis="x",rotation=30)
        axes[1].bar(labels,drift,color=[COLORS.get(c,COLORS["F8"]) for c in labels]);axes[1].set_ylabel("Cambio L2 relativo medio");axes[1].tick_params(axis="x",rotation=30)
        fig.suptitle("Cambio de representación tras condicionar MatGPT con geometría")
        fig.text(.5,.01,"Cada registro es una estructura outer por pliegue/semilla. Drift describe cambios y no mide utilidad causal ni calidad física.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.92));name="drift_embeddings_y_aporte_geometrico.png";fig.savefig(IMAGES/name,dpi=300);plt.close(fig);outputs.append(name)
    return outputs


def write_report(metrics: dict[str, Any], contrasts: pd.DataFrame, provenance: dict[str, Any],
                 figures: list[str], summary: dict[str, Any], gaps: pd.DataFrame) -> None:
    """Redacta un informe compacto, separando hechos, resultados y límites."""
    REPORT.mkdir(parents=True, exist_ok=True)
    metric_map={(r["condition"],r["metric"]):r["estimate"] for r in metrics["A_metrics"]}
    def f(cond: str, met: str) -> str:
        value=metric_map.get((cond,met));return "ND" if value is None else f"{value:.3f}"
    primary=contrasts[(contrasts.contrast=="F8 early − F8 SLICES-only") & (contrasts.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
    earlylate=contrasts[(contrasts.contrast=="F8 early − F8 late equiparada") & (contrasts.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
    inter=contrasts[(contrasts.contrast.str.startswith("Interacción")) & (contrasts.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
    zero=contrasts[(contrasts.contrast=="F8_early − predictor ΔE=0") & (contrasts.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
    primary_supported=bool(primary.estimate_left_minus_right<0 and primary.ci95_high<0 and primary.holm_p_primary_delta_family<0.05)
    late_supported=bool(earlylate.estimate_left_minus_right<0 and earlylate.ci95_high<0 and earlylate.holm_p_primary_delta_family<0.05)
    interaction_supported=bool(inter.estimate_left_minus_right<0 and inter.ci95_high<0 and inter.holm_p_primary_delta_family<0.05)
    lines=["# Fusión geométrica temprana en MatGPT: informe científico", "",
      "## Pregunta y diseño", "",
      "Se comparó MatGPT solo con SLICES frente a descriptores de celda inyectados antes del primer bloque Transformer (fusión temprana) y después de FINAL_END (fusión tardía equiparada). Se evaluaron F0 congelado y F8 con bloques 4–11 entrenables. La proyección geométrica 7→768 se inicializó en cero y la MLP8 conservó 768→8→1.", "",
      "**HECHO VERIFICADO:** conjunto de desarrollo: 210 estructuras, 70 composiciones y 3 polimorfos/fórmula; 5 pliegues, con 45 composiciones inner-train, 11 inner-validation y 14 outer-evaluation por pliegue. Se mantuvieron las semillas 42, 123 y 2026 y no se accedió al test histórico.", "",
      f"**HECHO VERIFICADO:** geometría de fuente `{provenance.get('source_counts')}`; SHA-256 CSV `{provenance.get('csv_sha256')}`. Siete variables: forma de celda reducida, tres ángulos y volumen/átomo. StandardScaler se ajustó por pliegue solo con las fórmulas inner-train. Ridge se reutilizó sin ajuste.", "",
      "## Verificaciones de implementación", "",
      "La inyección temprana suma la proyección geométrica al embedding de token más posición, antes de `TransformerDecoder`; la tardía suma el mismo vector a FINAL_END. La máscara causal, vocabulario, posiciones, selección de `<end>` y `ln_f` se conservaron. En F0 los pesos del backbone están congelados, pero el grafo atraviesa MatGPT para entrenar la proyección.", "",
      "**RESULTADO TÉCNICO:** proyección cero reprodujo FINAL_END del backbone original con error máximo 0 en las cuatro combinaciones F0/F8 × temprana/tardía. La proyección obtuvo gradientes no nulos; el smoke confirmó sensibilidad de FINAL_END a dos geometrías con SLICES fijo después de actualizar proyección. Verificación de parámetros y gradientes en el manifiesto de recursos.", "",
      "## Métricas primarias", "",
      "Errores en meV/átomo; se promedian primero estructuras/p pares dentro de fórmula y se promedian las semillas dentro de fórmula antes del bootstrap. Las métricas OLS usan las 630 filas por condición y son descriptivas.", "",
      "| Condición | MAE E/N | MAE residuo | MAE ΔE firmado | ΔE=0 | GS exacto | Orden de pares |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cond in CONDITIONS:
        lines.append(f"| {cond} | {f(cond,'mae_energy_mev_atom')} | {f(cond,'mae_residual_mev_atom')} | {f(cond,'mae_delta_signed_pairs_mev_atom')} | {f(cond,'mae_delta_zero_baseline_mev_atom')} | {f(cond,'ground_state_exact')} | {f(cond,'pairwise_accuracy')} |")
    lines += ["", "## Contrastes apareados", "",
      f"**RESULTADO EXPERIMENTAL (principal):** F8 temprana − F8 SLICES-only para MAE ΔE = {primary.estimate_left_minus_right:.3f} meV/átomo (IC95% {primary.ci95_low:.3f}, {primary.ci95_high:.3f}); p bootstrap bilateral ajustado Holm = {primary.holm_p_primary_delta_family:.4f}. Un valor negativo favorece fusión temprana.",
      f"**Comparación de ubicación:** F8 temprana − tardía equiparada = {earlylate.estimate_left_minus_right:.3f} meV/átomo (IC95% {earlylate.ci95_low:.3f}, {earlylate.ci95_high:.3f}); p Holm = {earlylate.holm_p_primary_delta_family:.4f}. La comparación iguala el número de pesos geométricos y la MLP8, pero sigue cambiando la posición de inyección y su interacción con la optimización.",
      f"**Interacción de profundidad:** (F8 temprana−F8 SLICES)−(F0 temprana−F0 SLICES) = {inter.estimate_left_minus_right:.3f} meV/átomo (IC95% {inter.ci95_low:.3f}, {inter.ci95_high:.3f}); p Holm = {inter.holm_p_primary_delta_family:.4f}.",
      f"**Comparación con ΔE=0:** F8 temprana − predictor nulo = {zero.estimate_left_minus_right:.3f} meV/átomo (IC95% {zero.ci95_low:.3f}, {zero.ci95_high:.3f}; sin ajuste, análisis secundario). Este control solo compara magnitud; no asigna ranking ni estado fundamental.",
      f"**Lectura del protocolo:** fusión temprana supera SLICES-only en F8 bajo el criterio principal: {primary_supported}. La fusión temprana supera a la tardía equiparada: {late_supported}. La interacción indica ventaja adicional de profundidad F8 frente a F0: {interaction_supported}. Estos indicadores se basan en IC y p ajustada; no sustituyen validación externa.",
      "", "## Diferencias pequeñas y compresión intrafórmula", "",
      "En los intervalos de 5–25 meV/átomo, la tabla siguiente muestra el tamaño muestral real y el MAE ΔE frente al predictor ΔE=0. No se interpreta ranking en 0–5 meV/átomo.",
      "", "| Condición | Intervalo | Pares | MAE modelo | MAE predictor ΔE=0 |", "|---|---|---:|---:|---:|"]
    gap_subset=gaps[(gaps.condition.isin(["F8_SLICES","F8_early","F8_late"])) & (gaps.gap_bin.astype(str).isin(["5–10","10–25"]))]
    for row in gap_subset.itertuples():
        lines.append(f"| {row.condition} | {row.gap_bin} | {int(row.n_pairs)} | {row.mae_delta_mev_atom:.2f} | {row.mae_delta_zero_baseline_mev_atom:.2f} |")
    lines += ["", "La compresión se resume mediante pendiente OLS y razón de dispersión intrafórmula/ΔE en `metrics.json`; el ajuste usa todas las parejas mostradas y es descriptivo. Una razón menor que uno indica menor dispersión predicha que la DFT. La utilidad requiere además reducir error y conservar el orden.", "",
      "El bootstrap remuestrea 70 composiciones apareadas (10 000 réplicas); semillas se agrupan dentro de cada fórmula. Las estimaciones son desarrollo OOF explorado, no generalización a una cohorte externa.", "",
      "## Interpretación y límites", "",
      "**RESULTADO EXPERIMENTAL:** ver la tabla y los intervalos antes de concluir si el MAE de diferencias supera SLICES-only o fusión tardía. El baseline ΔE=0 se muestra para magnitud: no recibe estado fundamental ni ranking arbitrario. El análisis de gaps 5–25 meV/átomo y la permutación geométrica son secundarios.", "",
      "**INFERENCIA:** si el contraste principal es favorable, la conclusión queda limitada a que la fusión temprana con celdas DFT estáticas mejora predicción bajo este protocolo. No demuestra que la atención sea el mecanismo causal ni que la geometría pueda obtenerse de SLICES.", "",
      "**LIMITACIÓN:** geometrías de endpoint estático `gga-static_endpoint_cleaned`; no se certifica homogeneidad total de XC/U/correcciones. El conjunto y sus fórmulas se habían usado en análisis previos; outer folds no son una prueba ciega independiente. No se incluyeron coordenadas internas, distancias ni enlaces. El test histórico no se utilizó.", "",
      "## Figuras", ""]
    for name in figures:
        lines += [f"![{name}](../../../images/MatGPT_SLICES_Geometria_Fusion_Temprana/{name})", ""]
    (REPORT/"reporte_cientifico.md").write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    """Genera resultados o resume predicciones ya calculadas sin reentrenar."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("predict","summarize","all"),default="all")
    args=parser.parse_args()
    EXP.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(EXP/"analysis.log",encoding="utf-8")])
    started=time.monotonic();depth=depth_module();fusion=new_module()
    frame,folds,model,stoi,context=depth.load_sources()
    geo_raw,provenance=fusion.geometry_table(depth,frame)
    sequences=depth.encode_all(frame,stoi,context)
    baseline=load_controls(depth)
    global depth_control
    depth_control={}
    for cond in ("F0","F8"):
        for (fold,seed),part in baseline[baseline.condition==f"{cond}_SLICES"].groupby(["fold","seed"]):
            depth_control[(cond,int(fold),int(seed))]=part.set_index("structure_id")
    pred_path=EXP/"predicciones_oof_por_estructura.csv"
    repr_path=EXP/"diagnostico_representacional.csv"
    if args.phase in ("predict","all") or not pred_path.is_file():
        new=[];repr_rows=[]
        for cond in ("F0","F8"):
            for mode in ("early","late"):
                rows,rrows=infer_condition(depth,fusion,frame,folds,geo_raw,sequences,model,stoi,context,mode,cond,False)
                new.extend(rows);repr_rows.extend(rrows)
                perm_rows,_=infer_condition(depth,fusion,frame,folds,geo_raw,sequences,model,stoi,context,mode,cond,True)
                new.extend(perm_rows)
                LOG.info("%s",{"event":"inference_done","condition":cond,"mode":mode,"rows":len(rows),"permuted_rows":len(perm_rows)})
        new_df=pd.DataFrame(new)
        # Control SLICES-only rows reutilizados sin cambios, a partir de checkpoints históricos.
        outputs=pd.concat([baseline,new_df],ignore_index=True,sort=False)
        if outputs.duplicated(["condition","fold","seed","structure_id"]).any():
            raise ValueError("Predicciones OOF contienen estructura/fold/semilla duplicados")
        outputs.to_csv(pred_path,index=False)
        pd.DataFrame(repr_rows).to_csv(repr_path,index=False)
        if len(outputs[outputs.condition.isin(CONDITIONS)])!=3780:
            raise ValueError("Conteo de predicciones principales incompatible: 6 condiciones × 630")
        if len(new_df[new_df.condition.str.endswith("_permuted")])!=2520:
            raise ValueError("Conteo de evaluación con geometría permutada incompatible")
    else:
        outputs=pd.read_csv(pred_path)
        if repr_path.is_file(): repr_rows=pd.read_csv(repr_path)
        else: repr_rows=[]
    outputs=pd.read_csv(pred_path)
    selected=outputs[outputs.condition.isin(CONDITIONS)].copy()
    pairs=pairwise_table(selected)
    selected.to_csv(pred_path,index=False)
    pairs.to_csv(EXP/"predicciones_oof_por_par.csv",index=False)
    formula_metrics,metrics=metric_rows(selected,pairs)
    formula_metrics.to_csv(EXP/"metricas_por_composicion.csv",index=False)
    contrasts,contrast_details=bootstrap_contrasts(formula_metrics)
    contrasts.to_csv(EXP/"bootstrap_apareado.csv",index=False)
    gaps=gap_table(pairs)
    gaps.to_csv(EXP/"metricas_por_gap.csv",index=False)
    history_path=EXP/"historial_entrenamiento.csv"
    history=pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    repr_df=pd.read_csv(repr_path) if repr_path.is_file() else pd.DataFrame()
    permuted_rows=outputs[outputs.condition.astype(str).str.endswith("_permuted")].copy()
    figures=make_figures(selected,pairs,formula_metrics,contrasts,gaps,history,repr_df,permuted_rows)
    figures.extend(["arquitectura_matgpt_geometria_inyectada_antes_de_atencion.png",
                    "regresion_hibrida_ridge_matgpt_geometria_tres_polimorfos.png"])
    metrics.update({"bootstrap_contrasts":contrasts.replace({np.nan:None}).to_dict("records"),
        "gap_bins":gaps.replace({np.nan:None}).to_dict("records"),
        "definitions":{"primary_estimand":"A: corrida por semilla; promedio de semillas dentro de fórmula antes de bootstrap",
            "delta":"E_B-E_A dentro de la misma fórmula, meV/átomo",
            "eligible_pairwise":"|ΔE_DFT| > 5 meV/átomo; predicción |ΔE|<=1 cuenta como incorrecta",
            "bootstrap":"10,000 réplicas apareadas por fórmula; semilla 20261030",
            "historical_test_accessed":False}})
    atomic_json(EXP/"metrics.json",metrics)
    summary=json.loads((EXP/"summary.json").read_text()) if (EXP/"summary.json").exists() else {}
    summary.update({"analysis_status":"COMPLETE","historical_test_accessed":False,
        "prediction_rows_primary":int(len(selected)),"pair_rows_primary":int(len(pairs)),
        "prediction_rows_geometry_permuted":int(outputs.condition.astype(str).str.endswith("_permuted").sum()),
        "figures_png":figures,"analysis_seconds":time.monotonic()-started,
        "primary_question":"F8 early fusion vs F8 SLICES-only; all contrasts paired by formula"})
    atomic_json(EXP/"summary.json",summary)
    resources=json.loads((EXP/"resources.json").read_text()) if (EXP/"resources.json").exists() else {"runs":{}}
    resources["analysis"]={"status":"COMPLETE","wall_seconds":time.monotonic()-started,
        "device":"cpu","peak_rss_bytes":int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)}
    atomic_json(EXP/"resources.json",resources)
    write_report(metrics,contrasts,provenance,figures,summary,gaps)
    LOG.info("%s",{"event":"analysis_complete","predictions":len(selected),"pairs":len(pairs),"figures":len(figures)})


if __name__=="__main__":
    main()
