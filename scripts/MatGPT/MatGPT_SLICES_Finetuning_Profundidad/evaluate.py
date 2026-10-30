"""Reconstruye OOF y métricas agrupadas por composición tras finalizar F2/F4/F8."""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
REPORT = EXPERIMENT
RIDGE_DIR = ROOT / "reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models"
FEATURES = ROOT / "reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/composition_features.csv"
F0_PRED = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/predicciones_outer_por_estructura.csv"
LOG = logging.getLogger("matgpt_depth_eval")
SEEDS = (42, 123, 2026)
CONDITIONS = ("F0", "F2", "F4", "F8")
BINS = ((0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, float("inf")))


def metrics_from_predictions(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Calcula errores energéticos, residuos, pares, ranking y grupos de brecha."""
    by_composition: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for (condition, fold, seed, formula), group in predictions.groupby(
            ["condition", "fold", "seed", "composition_id"], sort=True):
        group = group.sort_values("polymorph_family")
        dft = group.energy_per_atom_dft_ev.to_numpy(float)
        predicted = group.energy_per_atom_pred_ev.to_numpy(float)
        true_residual = dft - dft.mean()
        pred_residual = predicted - predicted.mean()
        gs = int(np.argmin(dft))
        pred_min = float(predicted.min())
        gs_pred = np.flatnonzero(np.isclose(predicted, pred_min, atol=1e-10, rtol=0))
        exact_gs = int(len(gs_pred) == 1 and int(gs_pred[0]) == gs)
        dft_gaps = (dft - dft[gs]) * 1000
        pred_gaps = (predicted - dft[gs]) * 1000
        pair_errors: list[float] = []
        pair_rows: list[dict[str, Any]] = []
        for i, j in itertools.combinations(range(3), 2):
            dt = float((dft[j] - dft[i]) * 1000)
            dp = float((predicted[j] - predicted[i]) * 1000)
            eligible = abs(dt) > 5.0
            resolved = abs(dp) > 1.0
            correct = bool(resolved and np.sign(dt) == np.sign(dp)) if eligible else None
            pair_errors.append(abs(dp - dt))
            item = group.iloc[i]
            pair_rows.append({"condition": condition, "fold": int(fold), "seed": int(seed),
                              "composition_id": formula,
                              "structure_id_a": group.iloc[i].structure_id,
                              "structure_id_b": group.iloc[j].structure_id,
                              "family_a": group.iloc[i].polymorph_family,
                              "family_b": group.iloc[j].polymorph_family,
                              "delta_dft_mev_atom": dt, "delta_pred_mev_atom": dp,
                              "abs_delta_dft_mev_atom": abs(dt), "abs_error_mev_atom": abs(dp-dt),
                              "eligible_pair_gt_5_mev": eligible, "prediction_resolved_gt_1_mev": resolved,
                              "pairwise_correct": float(correct) if correct is not None else np.nan,
                              "delta_zero_baseline_mev_atom": 0.0})
        pairs.extend(pair_rows)
        tau_s = spearmanr(dft, predicted).statistic if np.std(predicted) > 0 else np.nan
        tau_k = kendalltau(dft, predicted).statistic if np.std(predicted) > 0 else np.nan
        sorted_dft = np.sort(dft)
        gs_eligible = bool((sorted_dft[1] - sorted_dft[0]) * 1000 > 1.0)
        by_composition.append({"condition": condition, "fold": int(fold), "seed": int(seed),
            "composition_id": formula, "mae_energy_ev_atom": float(np.mean(np.abs(predicted-dft))),
            "mse_energy_ev_atom": float(np.mean((predicted-dft)**2)),
            "rmse_energy_ev_atom": float(np.sqrt(np.mean((predicted-dft)**2))),
            "mae_energy_total_ev": float(np.mean(np.abs(group.energy_total_pred_ev.to_numpy(float)-group.energy_total_dft_ev.to_numpy(float)))),
            "energy_bias_ev_atom": float(np.mean(predicted-dft)),
            "dft_mean_ev_atom": float(np.mean(dft)), "predicted_mean_ev_atom": float(np.mean(predicted)),
            "mean_energy_error_ev_atom": float(np.mean(predicted)-np.mean(dft)),
            "mae_residual_mev_atom": float(np.mean(np.abs(pred_residual-true_residual))*1000),
            "rmse_residual_mev_atom": float(np.sqrt(np.mean((pred_residual-true_residual)**2))*1000),
            "mae_delta_signed_pairs_mev_atom": float(np.mean(pair_errors)),
            "mae_delta_zero_baseline_mev_atom": float(np.mean([abs(x["delta_dft_mev_atom"]) for x in pair_rows])),
            "mae_delta_anchored_dft_gs_mev_atom": float(np.mean(np.abs(pred_gaps-dft_gaps))),
            "ground_state_exact": exact_gs, "ground_state_unresolved": int(len(gs_pred) != 1),
            "ground_state_eligible_gt_1_mev": int(gs_eligible),
            "ground_state_correct_eligible": int(gs_eligible and exact_gs == 1),
            "eligible_pairs": sum(bool(x["eligible_pair_gt_5_mev"]) for x in pair_rows),
            "correct_eligible_pairs": sum(x["pairwise_correct"] == 1.0 for x in pair_rows),
            "unresolved_eligible_pairs": sum(bool(x["eligible_pair_gt_5_mev"] and not x["prediction_resolved_gt_1_mev"]) for x in pair_rows),
            "spearman_triplet": float(tau_s), "kendall_triplet": float(tau_k),
            "dispersion_ratio_absolute": float(np.std(predicted, ddof=0)/np.std(dft, ddof=0)) if np.std(dft)>0 else np.nan,
            "dispersion_ratio_intracomposition": float(np.std(pred_residual)/np.std(true_residual)) if np.std(true_residual)>0 else np.nan,
            "slope_residual": float(np.polyfit(true_residual, pred_residual, 1)[0]) if np.std(true_residual)>0 else np.nan,
            "true_residual": true_residual.tolist(), "pred_residual": pred_residual.tolist(),
            "true_gap_anchored": dft_gaps.tolist(), "pred_gap_anchored": pred_gaps.tolist()})
    comp = pd.DataFrame(by_composition)
    pair_frame = pd.DataFrame(pairs)
    # Semillas son repeticiones del mismo material: se promedian dentro de fórmula.
    aggregate_rows: list[dict[str, Any]] = []
    for (condition, formula), group in comp.groupby(["condition", "composition_id"], sort=True):
        row = group.select_dtypes(include=[np.number]).mean().to_dict()
        for column in ("true_residual", "pred_residual", "true_gap_anchored", "pred_gap_anchored"):
            row[column] = np.mean(np.stack(group[column].map(np.asarray).to_list()), axis=0).tolist()
        row.update({"condition": condition, "composition_id": formula})
        aggregate_rows.append(row)
    aggregated = pd.DataFrame(aggregate_rows)
    summary: dict[str, Any] = {"unit": "formula (70 unique compositions); mean prediction across 3 training seeds",
        "by_condition": {}, "by_fold_seed": {}, "gap_bins": {}, "bootstrap": {}}
    for condition, data in aggregated.groupby("condition"):
        pred = predictions[predictions.condition == condition]
        averaged = pred.groupby(["composition_id", "structure_id"], as_index=False).agg(
            energy_per_atom_dft_ev=("energy_per_atom_dft_ev", "first"),
            energy_per_atom_pred_ev=("energy_per_atom_pred_ev", "mean"),
            energy_total_dft_ev=("energy_total_dft_ev", "first"), n_atoms=("n_atoms", "first"))
        x = averaged.energy_per_atom_dft_ev.to_numpy(float); y = averaged.energy_per_atom_pred_ev.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        r2_ols = 1 - np.sum((y-(slope*x+intercept))**2)/max(np.sum((y-y.mean())**2), 1e-30)
        r2_predictive = 1 - np.sum((y-x)**2)/max(np.sum((x-x.mean())**2), 1e-30)
        true_res = np.concatenate(data.true_residual.map(np.asarray).to_list())
        pred_res = np.concatenate(data.pred_residual.map(np.asarray).to_list())
        tgap = np.concatenate(data.true_gap_anchored.map(np.asarray).to_list())
        pgap = np.concatenate(data.pred_gap_anchored.map(np.asarray).to_list())
        pairs_avg = pair_frame[pair_frame.condition == condition].groupby(
            ["composition_id", "structure_id_a", "structure_id_b"], as_index=False).mean(numeric_only=True)
        pairs_avg["abs_error_mev_atom"]=(pairs_avg.delta_pred_mev_atom-pairs_avg.delta_dft_mev_atom).abs()
        pairs_avg["prediction_resolved_gt_1_mev"]=(pairs_avg.delta_pred_mev_atom.abs()>1.0)
        pairs_avg["pairwise_correct"]=((np.sign(pairs_avg.delta_pred_mev_atom)==np.sign(pairs_avg.delta_dft_mev_atom)) &
                                         pairs_avg.prediction_resolved_gt_1_mev).astype(float)
        eligible = pairs_avg[pairs_avg.eligible_pair_gt_5_mev > 0.5]
        resolved = eligible[eligible.prediction_resolved_gt_1_mev > 0.5]
        ensemble_gs_correct=[]; ensemble_gs_eligible=[]; ensemble_gs_unresolved=[]
        condition_predictions=predictions[predictions.condition==condition]
        ensemble_structures=condition_predictions.groupby(["composition_id","structure_id"],as_index=False).agg(
            dft=("energy_per_atom_dft_ev","first"),pred=("energy_per_atom_pred_ev","mean"))
        for _,formula_group in ensemble_structures.groupby("composition_id"):
            truth=formula_group.dft.to_numpy(float); predicted=formula_group.pred.to_numpy(float)
            gs=int(np.argmin(truth)); minimum=float(np.min(predicted))
            selected=np.flatnonzero(np.isclose(predicted,minimum,atol=1e-10,rtol=0))
            ensemble_gs_correct.append(int(len(selected)==1 and int(selected[0])==gs))
            ensemble_gs_eligible.append(int((np.sort(truth)[1]-np.sort(truth)[0])*1000>1.0))
            ensemble_gs_unresolved.append(int(len(selected)!=1))
        ensemble_gs_eligible_count=int(sum(ensemble_gs_eligible))
        physical_pairs=int(len(pairs_avg))
        pair_correct=int(eligible.pairwise_correct.sum()) if len(eligible) else 0
        summary["by_condition"][condition] = {
            "n_structures": 210, "n_compositions": 70, "n_pairs": 210,
            "mae_energy_mev_atom": float(np.mean(np.abs(y-x))*1000),
            "rmse_energy_mev_atom": float(np.sqrt(np.mean((y-x)**2))*1000),
            "bias_energy_mev_atom": float(np.mean(y-x)*1000),
            "mae_energy_total_ev": float(np.mean(np.abs(averaged.energy_total_dft_ev.to_numpy(float)-averaged.n_atoms*y))),
            "mae_composition_mean_mev_atom": float(data.mean_energy_error_ev_atom.abs().mean()*1000),
            "mae_residual_mev_atom": float(np.mean(np.abs(pred_res-true_res))*1000),
            "rmse_residual_mev_atom": float(np.sqrt(np.mean((pred_res-true_res)**2))*1000),
            "slope_absolute": float(slope), "intercept_absolute_ev_atom": float(intercept),
            "r2_ols_absolute": float(r2_ols), "r2_predictive_absolute": float(r2_predictive),
            "dispersion_ratio_absolute": float(np.std(y)/np.std(x)),
            "slope_intracomposition": float(np.polyfit(true_res, pred_res, 1)[0]),
            "dispersion_ratio_intracomposition": float(np.std(pred_res)/np.std(true_res)),
            "mae_delta_signed_pairs_mev_atom": float(pairs_avg.abs_error_mev_atom.mean()),
            "mae_delta_zero_baseline_mev_atom": float(pairs_avg.delta_dft_mev_atom.abs().mean()),
            "mae_delta_anchored_dft_gs_mev_atom": float(np.mean(np.abs(pgap-tgap))),
            "ground_state_accuracy_exact": float(np.mean(ensemble_gs_correct)),
            "ground_state_denominator": int(len(ensemble_gs_correct)),
            "ground_state_eligible_gt_1_mev": ensemble_gs_eligible_count,
            "ground_state_accuracy_eligible": float(sum(v for v,e in zip(ensemble_gs_correct,ensemble_gs_eligible) if e)/ensemble_gs_eligible_count)
                if ensemble_gs_eligible_count else None,
            "ground_state_unresolved_compositions": int(sum(ensemble_gs_unresolved)),
            "pairwise_accuracy_eligible": float(pair_correct/len(eligible)) if len(eligible) else None,
            "pairwise_accuracy_among_resolved": float(resolved.pairwise_correct.mean()) if len(resolved) else None,
            "pairwise_correct": pair_correct,
            "pairwise_eligible_denominator": int(len(eligible)),
            "pairwise_resolved_denominator": int(len(resolved)),
            "pairwise_unresolved": int(len(eligible)-len(resolved)),
            "spearman_triplet_mean": float(data.spearman_triplet.mean()),
            "kendall_triplet_mean": float(data.kendall_triplet.mean())}
    for (condition, fold, seed), group in comp.groupby(["condition", "fold", "seed"]):
        summary["by_fold_seed"][f"{condition}_fold{fold}_seed{seed}"] = {
            "n_compositions": int(len(group)),
            "mae_energy_mev_atom": float(group.mae_energy_ev_atom.mean()*1000),
            "mae_residual_mev_atom": float(group.mae_residual_mev_atom.mean()),
            "mae_delta_mev_atom": float(group.mae_delta_signed_pairs_mev_atom.mean()),
            "ground_state_accuracy": float(group.ground_state_exact.mean()),
            "pairwise_correct": int(group.correct_eligible_pairs.sum()),
            "pairwise_eligible": int(group.eligible_pairs.sum())}
    for condition in CONDITIONS:
        group = pair_frame[pair_frame.condition == condition]
        physical=group.groupby(["composition_id","structure_id_a","structure_id_b"],as_index=False).mean(numeric_only=True)
        physical["abs_error_mev_atom"]=(physical.delta_pred_mev_atom-physical.delta_dft_mev_atom).abs()
        physical["prediction_resolved_gt_1_mev"]=(physical.delta_pred_mev_atom.abs()>1.0)
        physical["pairwise_correct"]=((np.sign(physical.delta_pred_mev_atom)==np.sign(physical.delta_dft_mev_atom)) &
                                        physical.prediction_resolved_gt_1_mev).astype(float)
        for low, high in BINS:
            label = f"{low}-{high if np.isfinite(high) else 'más'}"
            selected = physical[(physical.abs_delta_dft_mev_atom >= low) & (physical.abs_delta_dft_mev_atom < high)]
            eligible = selected[selected.eligible_pair_gt_5_mev]
            resolved = eligible[eligible.prediction_resolved_gt_1_mev]
            summary["gap_bins"].setdefault(condition, {})[label] = {
                "n_pairs": int(len(selected)), "mae_delta_mev_atom": float(selected.abs_error_mev_atom.mean()) if len(selected) else None,
                "eligible_pairs_gt_5_mev": int(len(eligible)),
                "resolved_pairs": int(len(resolved)),
                "pairwise_accuracy_resolved": float(resolved.pairwise_correct.mean()) if len(resolved) else None,
                "zero_baseline_mae_mev_atom": float(selected.abs_delta_dft_mev_atom.mean()) if len(selected) else None}
    return summary, comp, pair_frame


def bootstrap_contrasts(predictions: pd.DataFrame, repetitions: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Bootstrap por fórmula tras promediar las predicciones de las semillas."""
    rng = np.random.default_rng(seed)
    averaged=predictions.groupby(["condition","composition_id","structure_id"],as_index=False).agg(
        dft=("energy_per_atom_dft_ev","first"),pred=("energy_per_atom_pred_ev","mean"))
    rows=[]
    for (condition,formula),group in averaged.groupby(["condition","composition_id"],sort=True):
        truth=group.dft.to_numpy(float); predicted=group.pred.to_numpy(float)
        truth_res=truth-truth.mean(); pred_res=predicted-predicted.mean()
        delta_errors=[]; zero_errors=[]
        for i,j in itertools.combinations(range(3),2):
            delta=(predicted[j]-predicted[i]-truth[j]+truth[i])*1000
            delta_errors.append(abs(delta)); zero_errors.append(abs((truth[j]-truth[i])*1000))
        rows.append({"condition":condition,"composition_id":formula,
            "mae_energy_ev_atom":float(np.mean(np.abs(predicted-truth))),
            "mae_residual_mev_atom":float(np.mean(np.abs(pred_res-truth_res))*1000),
            "mae_delta_signed_pairs_mev_atom":float(np.mean(delta_errors)),
            "mae_delta_zero_baseline_mev_atom":float(np.mean(zero_errors))})
    agg=pd.DataFrame(rows)
    metric_map = {"MAE energía absoluta (meV/átomo)": ("mae_energy_ev_atom", 1000),
                  "MAE residuo intrafórmula (meV/átomo)": ("mae_residual_mev_atom", 1),
                  "MAE ΔE (meV/átomo)": ("mae_delta_signed_pairs_mev_atom", 1)}
    output: dict[str, Any] = {}
    for metric_name, (column, scale) in metric_map.items():
        pivot = agg.pivot(index="composition_id", columns="condition", values=column)
        for left, right in (("F2", "F0"), ("F4", "F0"), ("F8", "F0"), ("F4", "F2"), ("F8", "F4")):
            difference = (pivot[left]-pivot[right]).to_numpy(float)*scale
            sampled = rng.integers(0, len(difference), size=(repetitions, len(difference)))
            values = difference[sampled].mean(axis=1)
            output[f"{left} − {right} | {metric_name}"] = {"mean_difference": float(difference.mean()),
                "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)),
                "bootstrap_repetitions": repetitions, "groups": int(len(difference)),
                "interpretation": "negative favors left model for error"}
    for condition in CONDITIONS:
        data=agg[agg.condition==condition]
        delta=(data.mae_delta_signed_pairs_mev_atom-data.mae_delta_zero_baseline_mev_atom).to_numpy(float)
        sampled=rng.integers(0,len(delta),size=(repetitions,len(delta)))
        values=delta[sampled].mean(axis=1)
        output[f"{condition} − ΔE=0 | MAE de diferencias (meV/átomo)"]={
            "mean_difference":float(delta.mean()),"ci95_low":float(np.quantile(values,.025)),
            "ci95_high":float(np.quantile(values,.975)),"bootstrap_repetitions":repetitions,
            "groups":int(len(delta)),"interpretation":"negative favors the model over zero predictor"}
    return output


def representation_diagnostics(frame: pd.DataFrame, folds: pd.DataFrame, model: Any,
                               stoi: dict[str, int], block_size: int) -> dict[str, Any]:
    """Compara FINAL_END original y adaptado en las 56 fórmulas de outer-train del fold 0."""
    import train as tr
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    encoded = {str(row.structure_id): encode_sequence(str(row.slices), stoi, block_size)
               for row in frame.itertuples()}
    ids = sorted(folds[(folds.outer_fold == 0) & (folds.outer_role == "outer_train")].structure_id)
    formulas = frame.set_index("structure_id").loc[ids].formula.to_dict()
    device = torch.device("cpu")

    def batch_embeddings() -> np.ndarray:
        """Extrae embeddings por lotes de ocho, sin targets ni actualizaciones."""
        outputs=[]
        with torch.inference_mode():
            for start in range(0,len(ids),8):
                part=ids[start:start+8]
                outputs.append(tr.final_end(model,[encoded[sid] for sid in part],stoi["<pad>"],device).cpu().numpy())
        return np.concatenate(outputs)

    original_state=torch.load(ROOT/"models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
                              map_location="cpu",weights_only=True)
    model.load_state_dict(original_state,strict=True); del original_state
    model.eval(); original=batch_embeddings()
    result={"n_structures":len(ids),"n_compositions":len(set(formulas.values())),
            "evaluation_scope":"outer-train del fold 0; solamente diagnóstico, sin etiquetas usadas",
            "by_condition":{}}
    for condition in ("F2","F4","F8"):
        checkpoint=torch.load(REPORT/"checkpoints"/f"{condition.lower()}_fold0_seed42"/"best.pt",
                              map_location="cpu",weights_only=False)
        model.load_state_dict(torch.load(ROOT/"models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
                                         map_location="cpu",weights_only=True),strict=True)
        tr.configure_trainable(model,tuple(checkpoint["blocks"]))
        head=tr.MLP8(); tr.load_module_states(model,head,checkpoint["model"],tuple(checkpoint["blocks"]))
        model.eval(); adapted=batch_embeddings()
        norms=np.linalg.norm(original,axis=1); new_norm=np.linalg.norm(adapted,axis=1)
        cosine=np.sum(original*adapted,axis=1)/np.maximum(norms*new_norm,1e-12)
        drift=np.linalg.norm(adapted-original,axis=1)/np.maximum(norms,1e-12)
        centered=adapted-adapted.mean(axis=0,keepdims=True)
        singular=np.linalg.svd(centered,compute_uv=False)
        eigen=singular**2; probs=eigen/eigen.sum() if eigen.sum()>0 else eigen
        effective_rank=float(np.exp(-np.sum(probs[probs>0]*np.log(probs[probs>0])))) if np.any(probs>0) else 0.0
        lookup={sid:i for i,sid in enumerate(ids)}
        within=[]
        for formula in sorted(set(formulas.values())):
            index=[lookup[sid] for sid in ids if formulas[sid]==formula]
            distances=[]
            for i,j in itertools.combinations(index,2):
                distances.append(float(np.linalg.norm(adapted[i]-adapted[j])/
                    max((new_norm[i]+new_norm[j])/2,1e-12)))
            within.append(float(np.mean(distances)))
        result["by_condition"][condition]={"mean_cosine_to_original":float(cosine.mean()),
            "std_cosine_to_original":float(cosine.std()),"mean_relative_l2_drift":float(drift.mean()),
            "std_relative_l2_drift":float(drift.std()),"p05_relative_l2_drift":float(np.quantile(drift,.05)),
            "p95_relative_l2_drift":float(np.quantile(drift,.95)),"mean_embedding_norm":float(new_norm.mean()),
            "mean_variance_per_dimension":float(np.var(adapted,axis=0).mean()),
            "effective_rank_entropy":effective_rank,"mean_normalized_intraformula_distance":float(np.mean(within)),
            "n_structures":len(ids),"n_compositions":len(set(formulas.values()))}
        del head
    return result


def predict_adapted(frame: pd.DataFrame, folds: pd.DataFrame, model: Any,
                    stoi: dict[str, int], block_size: int) -> pd.DataFrame:
    """Evalúa seleccionados en cada pliegue sobre sus fórmulas outer retenidas."""
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    encoded = {str(row.structure_id): encode_sequence(str(row.slices), stoi, block_size)
               for row in frame.itertuples()}
    features = pd.read_csv(FEATURES).set_index("composition_id")
    feature_cols = list(features.columns)
    lookup = frame.set_index("structure_id")
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        held = folds[(folds.outer_fold == fold) & (folds.outer_role == "outer_validation")]
        formulas = sorted(held.composition_group.unique())
        ridge = joblib.load(RIDGE_DIR / f"ridge_composicional_fold_{fold}.joblib")
        means = {formula: float(ridge.predict(features.loc[[formula], feature_cols])[0]) for formula in formulas}
        for seed in SEEDS:
            for condition in ("F2", "F4", "F8"):
                run = REPORT / "checkpoints" / f"{condition.lower()}_fold{fold}_seed{seed}"
                if not (run / "complete.flag").exists():
                    raise RuntimeError(f"Aún no finaliza {condition}, pliegue {fold}, semilla {seed}")
                checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                if checkpoint.get("base_sha256") != "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414":
                    raise ValueError("Checkpoint adaptado referencia otro MatGPT")
                blocks = tuple(checkpoint["blocks"])
                from scripts.MatGPT.MatGPT_SLICES_Finetuning_Profundidad.train import (
                    MLP8, load_module_states, configure_trainable, final_end)
                base_state = torch.load(ROOT / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
                                        map_location="cpu", weights_only=True)
                model.load_state_dict(base_state, strict=True)
                del base_state
                configure_trainable(model, blocks)
                head = MLP8()
                load_module_states(model, head, checkpoint["model"], blocks)
                model.eval(); head.eval()
                for formula in formulas:
                    ids = [str(held.loc[held.composition_group == formula, "structure_id"].iloc[0])]
                    group_ids = [str(lookup.loc[sid].name) for sid in ids]
                    # Las tres estructuras se ordenan con la regla usada al entrenar.
                    subset = lookup.loc[held[held.composition_group == formula].structure_id]
                    ordered = [str(subset[subset.family == family].index[0]) for family in ("dist_perovskite", "hexagonal", "needle")]
                    with torch.inference_mode():
                        emb = final_end(model, [encoded[sid] for sid in ordered], stoi["<pad>"], torch.device("cpu"))
                        residual = head(emb).numpy()
                    for sid, value in zip(ordered, residual):
                        item = lookup.loc[sid]
                        e_pred = means[formula] + float(value)
                        rows.append({"condition": condition, "fold": fold, "seed": seed,
                            "structure_id": sid, "composition_id": formula,
                            "polymorph_family": item.family, "n_atoms": int(item.n_atoms),
                            "energy_total_dft_ev": float(item.total_dft_ev),
                            "energy_per_atom_dft_ev": float(item.dft_ev_atom),
                            "ridge_mean_ev_atom": means[formula], "residual_pred_ev_atom": float(value),
                            "energy_per_atom_pred_ev": e_pred, "energy_total_pred_ev": e_pred*int(item.n_atoms)})
    return pd.DataFrame(rows)


def main() -> None:
    """Genera los únicos CSV de predicciones/métricas y `metrics.json`."""
    import train as tr
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(2)
    frame, folds, model, stoi, block_size = tr.load_sources()
    prediction_path = REPORT / "predicciones_oof_por_estructura.csv"
    predictions: pd.DataFrame | None = None
    if prediction_path.exists():
        cached = pd.read_csv(prediction_path)
        if (len(cached) == 2520 and set(cached.condition) == set(CONDITIONS) and
                cached.groupby("condition").structure_id.nunique().to_dict() == {c: 210 for c in CONDITIONS}):
            predictions = cached
            LOG.info("Reutilizo las 2,520 predicciones OOF ya calculadas y guardadas.")
    previous_metrics_path=REPORT/"metrics.json"
    if previous_metrics_path.exists():
        previous_metrics=json.loads(previous_metrics_path.read_text())
        representation=previous_metrics.get("representation_diagnostics")
    else:
        representation=None
    if representation is None:
        representation = representation_diagnostics(frame, folds, model, stoi, block_size)
    if predictions is None:
        adapted = predict_adapted(frame, folds, model, stoi, block_size)
        old = pd.read_csv(F0_PRED)
        old = old[old.condition == "frozen"].copy()
        features = pd.read_csv(FEATURES).set_index("composition_id")
        ridge_cache: dict[tuple[int, str], float] = {}
        f0_rows: list[dict[str, Any]] = []
        for row in old.itertuples():
            key = (int(row.fold), str(row.composition_id))
            if key not in ridge_cache:
                pipeline = joblib.load(RIDGE_DIR / f"ridge_composicional_fold_{row.fold}.joblib")
                ridge_cache[key] = float(pipeline.predict(features.loc[[row.composition_id], list(features.columns)])[0])
            e_pred = ridge_cache[key] + float(row.residual_ev_atom)
            f0_rows.append({"condition": "F0", "fold": int(row.fold), "seed": int(row.seed),
                "structure_id": row.structure_id, "composition_id": row.composition_id,
                "polymorph_family": row.polymorph_family, "n_atoms": int(row.n_atoms),
                "energy_total_dft_ev": float(row.energy_total_dft_ev),
                "energy_per_atom_dft_ev": float(row.energy_per_atom_dft_ev),
                "ridge_mean_ev_atom": ridge_cache[key], "residual_pred_ev_atom": float(row.residual_ev_atom),
                "energy_per_atom_pred_ev": e_pred, "energy_total_pred_ev": e_pred*int(row.n_atoms)})
        predictions = pd.concat([pd.DataFrame(f0_rows), adapted], ignore_index=True)
    if (len(predictions) != 2520 or set(predictions.condition) != set(CONDITIONS) or
            predictions.groupby("condition").structure_id.nunique().to_dict() != {c: 210 for c in CONDITIONS}):
        raise ValueError("Predicciones incompletas o IDs OOF incompatibles")
    # Las etiquetas, el fold y los IDs deben coincidir para las cuatro condiciones.
    for _, group in predictions.groupby(["fold", "seed", "composition_id"]):
        if len(group) != 12 or group.structure_id.nunique() != 3:
            raise ValueError("Controles no apareados o estructura duplicada en OOF")
    prediction_path = REPORT / "predicciones_oof_por_estructura.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    summary, composition_metrics, pair_predictions = metrics_from_predictions(predictions)
    composition_metrics.drop(columns=["true_residual", "pred_residual", "true_gap_anchored", "pred_gap_anchored"]).to_csv(
        REPORT / "metricas_por_composicion.csv", index=False)
    pair_predictions.to_csv(REPORT / "predicciones_oof_por_par.csv", index=False)
    full_history = pd.read_csv(REPORT / "historial_entrenamiento.csv")
    full_history[full_history.run_type == "full"].to_csv(REPORT / "historial_entrenamiento.csv", index=False)
    summary["bootstrap"] = bootstrap_contrasts(predictions)
    summary["f0_source"] = str(F0_PRED.relative_to(ROOT))
    summary["f0_reused_without_retraining"] = True
    summary["statistical_unit"] = "70 composiciones; semillas promediadas por composición antes de bootstrap"
    summary["representation_diagnostics"] = representation
    def clean(value: Any) -> Any:
        """Convierte escalares no finitos a null para JSON estricto."""
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.integer,)):
            return int(value)
        return value
    (REPORT / "metrics.json").write_text(json.dumps(clean(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    LOG.info("Evaluación OOF terminada: %s", {k: v["mae_energy_mev_atom"] for k, v in summary["by_condition"].items()})


if __name__ == "__main__":
    main()
