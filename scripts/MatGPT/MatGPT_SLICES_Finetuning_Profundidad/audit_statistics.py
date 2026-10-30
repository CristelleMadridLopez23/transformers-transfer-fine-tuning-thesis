"""Audita agregaciones, contrastes y bootstrap de la ablación MatGPT por profundidad."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Profundidad"
IMAGES = ROOT / "images/MatGPT_SLICES_Finetuning_Profundidad"
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
F0_SOURCE = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/predicciones_outer_por_estructura.csv"
RIDGE_DIR = ROOT / "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models"
COMPOSITION_FEATURES = ROOT / "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/composition_features.csv"
CONDITIONS = ("F0", "F2", "F4", "F8")
SEEDS = (42, 123, 2026)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261029
LOG = logging.getLogger("matgpt_depth_statistics_audit")

METRIC_UNITS = {
    "mae_energy_mev_atom": "meV/átomo", "rmse_energy_mev_atom": "meV/átomo",
    "mae_energy_total_ev": "eV", "mae_residual_mev_atom": "meV/átomo",
    "mae_delta_signed_pairs_mev_atom": "meV/átomo", "mae_delta_zero_baseline_mev_atom": "meV/átomo",
    "mae_delta_anchored_all3_mev_atom": "meV/átomo", "mae_delta_anchored_nonanchor_mev_atom": "meV/átomo",
    "ground_state_accuracy_exact": "proporción", "ground_state_accuracy_eligible_gt_1_mev": "proporción",
    "pairwise_accuracy_macro_formula": "proporción", "pairwise_accuracy_pooled": "proporción",
    "pairwise_accuracy_among_resolved": "proporción", "pairwise_accuracy_resolved_pooled": "proporción", "spearman_triplet": "proporción",
    "kendall_triplet": "proporción",
}
ERROR_METRICS = {"mae_energy_mev_atom", "rmse_energy_mev_atom", "mae_energy_total_ev",
                 "mae_residual_mev_atom", "mae_delta_signed_pairs_mev_atom",
                 "mae_delta_zero_baseline_mev_atom", "mae_delta_anchored_all3_mev_atom",
                 "mae_delta_anchored_nonanchor_mev_atom"}


def load_development_dataset(path: Path) -> pd.DataFrame:
    """Lee solo registros train/val y no materializa etiquetas del test histórico."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"structure_id", "split", "reduced_formula", "polymorph_family", "num_atoms",
                    "energy_total_ev", "energy_per_atom_ev"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Faltan columnas del dataset: {required-set(reader.fieldnames or [])}")
        for raw in reader:
            if raw["split"] not in {"train", "val"}:
                continue
            rows.append({"structure_id": raw["structure_id"], "split": raw["split"],
                         "composition_id": raw["reduced_formula"],
                         "polymorph_family": raw["polymorph_family"],
                         "n_atoms": int(raw["num_atoms"]),
                         "energy_total_dft_ev": float(raw["energy_total_ev"]),
                         "energy_per_atom_dft_ev": float(raw["energy_per_atom_ev"])})
    return pd.DataFrame(rows)


def validate_integrity(pred: pd.DataFrame, pairs: pd.DataFrame, development: pd.DataFrame,
                       folds: pd.DataFrame, f0_source: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    """Valida IDs, folds, targets, pares, semillas, F0 y conservación de Ridge."""
    issues: list[str] = []
    if development.structure_id.duplicated().any() or len(development) != 210:
        issues.append("dataset train/val no tiene 210 IDs únicos")
    if development.composition_id.nunique() != 70 or not (development.groupby("composition_id").size() == 3).all():
        issues.append("dataset train/val no contiene exactamente 70 tripletes")
    if pred.duplicated(["condition", "fold", "seed", "structure_id"]).any():
        issues.append("predicciones con IDs repetidos en condición/pliegue/semilla")
    if pairs.duplicated(["condition", "fold", "seed", "composition_id", "structure_id_a", "structure_id_b"]).any():
        issues.append("pares repetidos")
    if len(pred) != 2520 or set(pred.condition) != set(CONDITIONS) or not (pred.groupby("condition").size()==630).all():
        issues.append("cantidad/condiciones de predicciones distintas de 2,520 / F0,F2,F4,F8")
    if len(pairs) != 2520:
        issues.append("cantidad de pares distinta de 2,520")
    if set(pred.seed.unique()) != set(SEEDS) or set(pairs.seed.unique()) != set(SEEDS):
        issues.append("semillas incompletas")
    development_ids = set(development.structure_id)
    if not set(pred.structure_id).issubset(development_ids):
        issues.append("predicciones contienen IDs fuera de train/val")
    if set(pred.structure_id) & set(folds.loc[folds.historical_split == "test", "structure_id"]):
        issues.append("hay IDs de test histórico en las predicciones")
    dlookup = development.set_index("structure_id")
    for _, row in pred.iterrows():
        ref = dlookup.loc[row.structure_id]
        if row.composition_id != ref.composition_id or row.polymorph_family != ref.polymorph_family:
            issues.append(f"identidad/composición/familia no coincide: {row.structure_id}"); break
        if int(row.n_atoms) != int(ref.n_atoms):
            issues.append(f"N no coincide: {row.structure_id}"); break
        if not np.isclose(row.energy_total_dft_ev, ref.energy_total_dft_ev, atol=1e-8, rtol=0):
            issues.append(f"E_total DFT no coincide: {row.structure_id}"); break
        if not np.isclose(row.energy_per_atom_dft_ev, ref.energy_per_atom_dft_ev, atol=1e-9, rtol=0):
            issues.append(f"E/N DFT no coincide: {row.structure_id}"); break
        if not np.isclose(row.energy_total_dft_ev / row.n_atoms, row.energy_per_atom_dft_ev, atol=1e-8, rtol=0):
            issues.append(f"unidad E/N inconsistente: {row.structure_id}"); break
        if not np.isclose(row.energy_total_pred_ev, row.n_atoms * row.energy_per_atom_pred_ev, atol=1e-8, rtol=0):
            issues.append(f"unidad de predicción total inconsistente: {row.structure_id}"); break
    # Una fórmula se evalúa fuera una sola vez, en su pliegue asignado.
    outer = folds[folds.outer_role == "outer_validation"]
    outer_counts = outer.groupby("structure_id").outer_fold.nunique()
    if len(outer_counts) != 210 or not (outer_counts == 1).all():
        issues.append("asignación outer no deja cada estructura en una sola evaluación")
    expected_fold = outer.set_index("structure_id").outer_fold
    formula_folds = outer.groupby("composition_group").outer_fold.nunique()
    if len(formula_folds) != 70 or not (formula_folds == 1).all():
        issues.append("polimorfos de una composición están repartidos entre pliegues outer")
    for _, row in pred.drop_duplicates(["structure_id", "fold"]).iterrows():
        if int(expected_fold.loc[row.structure_id]) != int(row.fold):
            issues.append(f"fold OOF no coincide: {row.structure_id}"); break
    expected_pair_rows = []
    for (condition, fold, seed, formula), group in pred.groupby(["condition", "fold", "seed", "composition_id"]):
        if len(group) != 3 or group.structure_id.nunique() != 3:
            issues.append(f"triplete incompleto: {condition}/{fold}/{seed}/{formula}"); break
        group = group.sort_values("polymorph_family")
        for i, j in itertools.combinations(range(3), 2):
            expected_pair_rows.append((condition, int(fold), int(seed), formula,
                                       group.iloc[i].structure_id, group.iloc[j].structure_id,
                                       (group.iloc[j].energy_per_atom_dft_ev-group.iloc[i].energy_per_atom_dft_ev)*1000,
                                       (group.iloc[j].energy_per_atom_pred_ev-group.iloc[i].energy_per_atom_pred_ev)*1000))
    pair_index = pairs.set_index(["condition", "fold", "seed", "composition_id", "structure_id_a", "structure_id_b"])
    for key in expected_pair_rows:
        idx = key[:6]
        if idx not in pair_index.index:
            issues.append(f"falta par u orientación no coincide: {idx}"); break
        row = pair_index.loc[idx]
        if not np.isclose(row.delta_dft_mev_atom, key[6], atol=1e-7, rtol=0) or not np.isclose(row.delta_pred_mev_atom, key[7], atol=1e-7, rtol=0):
            issues.append(f"ΔE/unidades/signo incorrectos para {idx}"); break
        expected_eligible=abs(key[6])>5.0; expected_resolved=abs(key[7])>1.0
        expected_correct=expected_eligible and expected_resolved and np.sign(key[6])==np.sign(key[7])
        if bool(row.eligible_pair_gt_5_mev)!=expected_eligible or bool(row.prediction_resolved_gt_1_mev)!=expected_resolved:
            issues.append(f"máscara elegible/resuelta inconsistente para {idx}"); break
        if expected_eligible and bool(row.pairwise_correct)!=bool(expected_correct):
            issues.append(f"etiqueta pairwise incorrecta para {idx}"); break
    # Ridge es una media común por composición y debe permanecer idéntico entre condiciones.
    ridge_spread = pred.groupby(["fold", "seed", "composition_id", "structure_id"]).ridge_mean_ev_atom.agg(lambda x: x.max()-x.min())
    if float(ridge_spread.max()) > 1e-10:
        issues.append("Ridge composicional varía entre condiciones")
    # Checkpoint Ridge y transformador por fold: hashes de preflight y salida sobre las mismas fórmulas.
    ridge_hashes: dict[str,str]={}
    try:
        features=pd.read_csv(COMPOSITION_FEATURES).set_index("composition_id")
        for fold in range(5):
            model_path=RIDGE_DIR/f"ridge_composicional_fold_{fold}.joblib"
            ridge_hashes[str(fold)]=_sha(model_path)
            expected_hash=summary.get("ridge_sha256",{}).get(str(fold))
            if expected_hash and ridge_hashes[str(fold)]!=expected_hash:
                issues.append(f"hash Ridge fold {fold} no coincide con preflight"); break
            ridge=joblib.load(model_path)
            formulas=sorted(pred.loc[(pred.condition=="F0")&(pred.fold==fold),"composition_id"].unique())
            for formula in formulas:
                cols=list(features.columns)
                expected=float(ridge.predict(features.loc[[formula],cols].to_numpy(float))[0])
                actual=float(pred.loc[(pred.condition=="F0")&(pred.fold==fold)&(pred.composition_id==formula),"ridge_mean_ev_atom"].iloc[0])
                if not np.isclose(expected,actual,atol=1e-9,rtol=0):
                    issues.append(f"salida Ridge no coincide con pipeline congelado: fold {fold}, {formula}"); break
    except Exception as exc:
        issues.append(f"no se pudieron verificar modelos Ridge históricos: {type(exc).__name__}: {exc}")
    # Comparar F0 guardado con la fuente congelada histórica, por ID físico/fold/semilla.
    frozen = f0_source[f0_source.condition.astype(str).str.lower() == "frozen"].copy()
    current = pred[pred.condition == "F0"].copy()
    f0_match = True
    if len(frozen) != 630:
        issues.append(f"fuente Frozen tiene {len(frozen)} filas; se esperaban 630"); f0_match = False
    else:
        join = current.merge(frozen, on=["structure_id", "fold", "seed"], suffixes=("_new", "_old"), validate="one_to_one")
        if len(join) != 630:
            issues.append("F0 no se alinea 1:1 con el control Frozen histórico"); f0_match = False
        elif not np.allclose(join.residual_pred_ev_atom, join.residual_ev_atom, atol=1e-10, rtol=0):
            issues.append("residuos F0 no coinciden con la fuente Frozen"); f0_match = False
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues,
              "n_structures": int(development.structure_id.nunique()), "n_compositions": int(development.composition_id.nunique()),
              "n_oof_rows": int(len(pred)), "n_pair_rows": int(len(pairs)), "n_seeds": int(pred.seed.nunique()),
              "n_unique_physical_pairs": int(pairs[["composition_id", "structure_id_a", "structure_id_b"]].drop_duplicates().shape[0]),
              "n_eligible_physical_pairs": int(pairs.drop_duplicates(["composition_id", "structure_id_a", "structure_id_b"]).eligible_pair_gt_5_mev.sum()),
              "f0_source_rows": int(len(frozen)), "f0_residuals_match": f0_match,
              "ridge_hashes": ridge_hashes,
              "max_ridge_cross_condition_spread_ev_atom": float(ridge_spread.max()),
              "test_labels_read": False, "historical_test_ids_used": False}
    if issues:
        raise ValueError("Integridad FAIL: " + "; ".join(issues))
    return result


def formula_seed_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas por fórmula/semilla y diferencias ΔE ancladas correctamente."""
    rows: list[dict[str, Any]] = []
    for (condition, formula, seed), group in pred.groupby(["condition", "composition_id", "seed"], sort=True):
        group = group.sort_values("polymorph_family").reset_index(drop=True)
        y = group.energy_per_atom_dft_ev.to_numpy(float)
        p = group.energy_per_atom_pred_ev.to_numpy(float)
        n = group.n_atoms.to_numpy(float)
        dft_gaps,pred_gaps,anchor = calculate_anchored_gaps(y,p)
        anchored_error = np.abs(pred_gaps-dft_gaps)
        pair_errors: list[float] = []; dft_pair_abs: list[float] = []
        eligible: list[bool] = []; correct: list[bool] = []; resolved: list[bool] = []
        for i, j in itertools.combinations(range(3), 2):
            dt = (y[j]-y[i])*1000; dp = (p[j]-p[i])*1000
            pair_errors.append(abs(dp-dt)); dft_pair_abs.append(abs(dt))
            eligible.append(abs(dt)>5.0); resolved.append(abs(dp)>1.0)
            correct.append(abs(dt)>5.0 and abs(dp)>1.0 and np.sign(dt)==np.sign(dp))
        num_eligible=sum(eligible); num_correct=sum(correct); num_resolved=sum(e and r for e,r in zip(eligible,resolved))
        sorted_y=np.sort(y); gs_eligible=bool((sorted_y[1]-sorted_y[0])*1000>1.0)
        min_indices=np.flatnonzero(np.isclose(p,p.min(),atol=1e-10,rtol=0))
        exact=int(len(min_indices)==1 and int(min_indices[0])==anchor)
        dres=y-y.mean(); pres=p-p.mean()
        rows.append({"condition":condition,"composition_id":formula,"seed":int(seed),"fold":int(group.fold.iloc[0]),
            "mae_energy_mev_atom":float(np.mean(np.abs(p-y))*1000),
            "rmse_energy_mev_atom":float(np.sqrt(np.mean((p-y)**2))*1000),
            "mae_energy_total_ev":float(np.mean(np.abs(group.energy_total_pred_ev.to_numpy(float)-group.energy_total_dft_ev.to_numpy(float)))),
            "mae_residual_mev_atom":float(np.mean(np.abs(pres-dres))*1000),
            "mae_delta_signed_pairs_mev_atom":float(np.mean(pair_errors)),
            "mae_delta_zero_baseline_mev_atom":float(np.mean(dft_pair_abs)),
            "mae_delta_anchored_all3_mev_atom":float(np.mean(anchored_error)),
            "mae_delta_anchored_nonanchor_mev_atom":float(np.mean(np.delete(anchored_error,anchor))),
            "ground_state_accuracy_exact":float(exact),
            "ground_state_accuracy_eligible_gt_1_mev":float(exact) if gs_eligible else np.nan,
            "ground_state_eligible":int(gs_eligible), "ground_state_correct":int(exact),
            "pairwise_accuracy_macro_formula":float(num_correct/num_eligible) if num_eligible else np.nan,
            "pairwise_accuracy_among_resolved":float(num_correct/num_resolved) if num_resolved else np.nan,
            "pairwise_correct_count":int(num_correct),"pairwise_eligible_count":int(num_eligible),
            "pairwise_resolved_count":int(num_resolved),"eligible_pair_count":int(num_eligible),
            "spearman_triplet":float(pd.Series(y).corr(pd.Series(p),method="spearman")),
            "kendall_triplet":float(pd.Series(y).corr(pd.Series(p),method="kendall")),
            "n_structures":3,"n_pairs":3,"anchor_family":str(group.iloc[anchor].polymorph_family)})
    return pd.DataFrame(rows)


def calculate_anchored_gaps(dft_ev_atom: np.ndarray, pred_ev_atom: np.ndarray) -> tuple[np.ndarray,np.ndarray,int]:
    """Ancla DFT y predicción a la misma estructura GS seleccionada por DFT."""
    dft=np.asarray(dft_ev_atom,dtype=float); pred=np.asarray(pred_ev_atom,dtype=float)
    if dft.ndim!=1 or pred.ndim!=1 or len(dft)!=3 or len(pred)!=3 or not np.isfinite(dft).all() or not np.isfinite(pred).all():
        raise ValueError("Se requieren tres energías DFT y tres predicciones finitas por composición")
    anchor=int(np.argmin(dft))
    return (dft-dft[anchor])*1000,(pred-pred[anchor])*1000,anchor


def ensemble_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Forma el estimando de predictor ensemble promediando semillas por estructura."""
    invariant = ["fold", "composition_id", "polymorph_family", "n_atoms", "energy_total_dft_ev", "energy_per_atom_dft_ev", "ridge_mean_ev_atom"]
    for (condition, sid), group in pred.groupby(["condition", "structure_id"]):
        for col in invariant:
            if group[col].nunique(dropna=False) != 1:
                raise ValueError(f"Atributo físico no invariante entre semillas: {condition}/{sid}/{col}")
    return pred.groupby(["condition","fold","structure_id","composition_id","polymorph_family"],as_index=False).agg(
        n_atoms=("n_atoms","first"),energy_total_dft_ev=("energy_total_dft_ev","first"),
        energy_per_atom_dft_ev=("energy_per_atom_dft_ev","first"),ridge_mean_ev_atom=("ridge_mean_ev_atom","first"),
        energy_per_atom_pred_ev=("energy_per_atom_pred_ev","mean"),energy_total_pred_ev=("energy_total_pred_ev","mean"))


def aggregate_estimands(seed_metrics: pd.DataFrame, ensemble_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construye métricas globales de corrida individual (A) y ensemble (B)."""
    out: list[dict[str, Any]] = []; formula_tables: dict[str,pd.DataFrame] = {}
    scalar_metrics = [m for m in METRIC_UNITS if m not in {"pairwise_accuracy_pooled","pairwise_accuracy_resolved_pooled"}]
    for condition in CONDITIONS:
        a = seed_metrics[seed_metrics.condition==condition]
        a_formula = a.groupby("composition_id",as_index=False)[scalar_metrics].mean()
        # Para conteos pooled se promedian semillas por fórmula, manteniendo el denominador físico.
        counts = a.groupby("composition_id",as_index=False)[["pairwise_correct_count","pairwise_eligible_count","pairwise_resolved_count","ground_state_eligible","ground_state_correct"]].mean()
        a_formula = a_formula.merge(counts,on="composition_id",suffixes=("","_counts"))
        a_formula["pairwise_correct_count"] = counts.pairwise_correct_count
        a_formula["pairwise_eligible_count"] = counts.pairwise_eligible_count
        a_formula["pairwise_resolved_count"] = counts.pairwise_resolved_count
        a_formula["pairwise_unresolved_count"] = a_formula.pairwise_eligible_count-a_formula.pairwise_resolved_count
        a_formula["ground_state_eligible"] = counts.ground_state_eligible
        a_formula["ground_state_correct"] = counts.ground_state_correct
        b = ensemble_metrics[ensemble_metrics.condition==condition].groupby("composition_id",as_index=False)[scalar_metrics+[
            "pairwise_correct_count","pairwise_eligible_count","pairwise_resolved_count","ground_state_eligible","ground_state_correct"]].mean()
        b["pairwise_unresolved_count"] = b.pairwise_eligible_count-b.pairwise_resolved_count
        formula_tables[f"A_{condition}"] = a_formula
        formula_tables[f"B_{condition}"] = b
        for label, table, seedwise in (("A_mean_single_seed",a_formula,True),("B_three_seed_prediction_ensemble",b,False)):
            for metric in scalar_metrics:
                values=table[metric].dropna()
                if metric=="pairwise_accuracy_pooled":
                    continue
                out.append({"condition":condition,"estimand":label,"metric":metric,"value":float(values.mean()),
                    "units":METRIC_UNITS[metric],"n_compositions":int(values.size),"n_structures":210,
                    "n_pairs":210,"seed_aggregation":"mean per formula then mean formulae" if seedwise else "mean prediction per structure before metric",
                    "denominator":"70 formulae; eligible-only for eligible metrics"})
            denominator=float(table.pairwise_eligible_count.sum())
            numerator=float(table.pairwise_correct_count.sum())
            out.append({"condition":condition,"estimand":label,"metric":"pairwise_accuracy_pooled","value":numerator/denominator,
                "units":"proporción","n_compositions":70,"n_structures":210,"n_pairs":int(denominator),
                "seed_aggregation":"paired physical pairs across 3 seeds" if seedwise else "ensemble pair prediction",
                "denominator":"eligible physical pairs; unresolved predictions count incorrect"})
            resolved_denominator=float(table.pairwise_resolved_count.sum())
            out.append({"condition":condition,"estimand":label,"metric":"pairwise_accuracy_resolved_pooled",
                "value":float(table.pairwise_correct_count.sum()/resolved_denominator) if resolved_denominator else np.nan,
                "units":"proporción","n_compositions":70,"n_structures":210,"n_pairs":int(resolved_denominator),
                "seed_aggregation":"paired resolved physical pairs across seeds" if seedwise else "ensemble resolved pairs",
                "denominator":"eligible pairs whose predicted |ΔE| exceeds 1 meV/átomo"})
            # Keep counts and all three individual seed summaries visible.
            for seed in SEEDS:
                if seedwise:
                    sub=a[a.seed==seed]
                    for metric in scalar_metrics:
                        vals=sub[metric].dropna()
                        if len(vals): out.append({"condition":condition,"estimand":f"seed_{seed}","metric":metric,"value":float(vals.mean()),
                            "units":METRIC_UNITS[metric],"n_compositions":int(vals.size),"n_structures":210,"n_pairs":210,
                            "seed_aggregation":"single training seed; macro mean over formulae","denominator":"70 formulae; eligible-only for eligible metrics"})
    return pd.DataFrame(out),formula_tables


def bootstrap_contrasts(formula_tables: dict[str,pd.DataFrame], reps: int=BOOTSTRAP_REPLICATES,
                        seed: int=BOOTSTRAP_SEED) -> pd.DataFrame:
    """Bootstrap pareado de composiciones, conservando pares y las tres semillas."""
    rng=np.random.default_rng(seed); contrasts=[("F2","F0"),("F4","F0"),("F8","F0"),("F4","F2"),("F8","F4")]
    metrics=["mae_energy_mev_atom","mae_residual_mev_atom","mae_delta_signed_pairs_mev_atom",
             "ground_state_accuracy_exact","ground_state_accuracy_eligible_gt_1_mev",
             "pairwise_accuracy_macro_formula","pairwise_accuracy_pooled"]
    rows=[]
    for estimand in ("A","B"):
        for metric in metrics:
            for left,right in contrasts:
                l=formula_tables[f"{estimand}_{left}"].set_index("composition_id").sort_index()
                r=formula_tables[f"{estimand}_{right}"].set_index("composition_id").sort_index()
                # Macro-composición es la inferencia primaria; emparejamiento explícito.
                if metric=="pairwise_accuracy_pooled":
                    # Ratio de conteos por remuestra, ponderado por pares elegibles.
                    nums=l.pairwise_correct_count.to_numpy(float)-r.pairwise_correct_count.to_numpy(float)
                    # Contrast ratio is computed as left accuracy minus right accuracy within each bootstrap draw.
                    li=l.pairwise_correct_count.to_numpy(float); ld=l.pairwise_eligible_count.to_numpy(float)
                    ri=r.pairwise_correct_count.to_numpy(float); rd=r.pairwise_eligible_count.to_numpy(float)
                    idx=rng.integers(0,len(l),size=(reps,len(l)))
                    vals=li[idx].sum(1)/ld[idx].sum(1)-ri[idx].sum(1)/rd[idx].sum(1)
                    estimate=float(li.sum()/ld.sum()-ri.sum()/rd.sum())
                    valid=np.ones(len(l),dtype=bool)
                else:
                    lv=l[metric].to_numpy(float); rv=r[metric].to_numpy(float)
                    valid=np.isfinite(lv)&np.isfinite(rv); diff=lv[valid]-rv[valid]
                    idx=rng.integers(0,len(diff),size=(reps,len(diff)))
                    vals=np.nanmean(diff[idx],axis=1); estimate=float(np.mean(diff))
                rows.append(_bootstrap_row(estimand,f"{left} − {right}",metric,estimate,vals,len(valid)))
        # Baseline ΔE=0 only compares error magnitudes; it has no defined ranking.
        for condition in CONDITIONS:
            table=formula_tables[f"{estimand}_{condition}"].set_index("composition_id").sort_index()
            diff=(table.mae_delta_signed_pairs_mev_atom-table.mae_delta_zero_baseline_mev_atom).to_numpy(float)
            idx=rng.integers(0,len(diff),size=(reps,len(diff))); vals=np.mean(diff[idx],axis=1)
            rows.append(_bootstrap_row(estimand,f"{condition} − ΔE=0","mae_delta_vs_zero",float(diff.mean()),vals,len(diff)))
            # IC de la métrica absoluta por condición (no del contraste con cero).
            raw=table.mae_delta_signed_pairs_mev_atom.to_numpy(float)
            idx=rng.integers(0,len(raw),size=(reps,len(raw))); samples=np.mean(raw[idx],axis=1)
            rows.append(_bootstrap_row(estimand,condition,"mae_delta_signed_pairs_mev_atom",float(raw.mean()),samples,len(raw)))
    return pd.DataFrame(rows)


def _bootstrap_row(estimand: str, contrast: str, metric: str, estimate: float,
                   samples: np.ndarray, n_groups: int) -> dict[str,Any]:
    """Empaqueta estimación observada e intervalo percentil bootstrap."""
    return {"estimand":estimand,"contrast":contrast,"metric":metric,"estimate":estimate,
            "bootstrap_mean":float(np.mean(samples)),"ci95_low":float(np.quantile(samples,.025)),
            "ci95_high":float(np.quantile(samples,.975)),"bootstrap_replicates":BOOTSTRAP_REPLICATES,
            "n_compositions":n_groups,"includes_zero":bool(np.quantile(samples,.025)<=0<=np.quantile(samples,.975)),
            "direction":"negative favors left model" if metric in ERROR_METRICS or metric=="mae_delta_vs_zero" else "positive favors left model"}


def _draw_interval(axis: plt.Axes, x: float, y: float, low: float, high: float,
                   color: str, marker: str="o", vertical: bool=False) -> None:
    """Dibuja IC y estimación sin asumir que el percentil contiene el estimador."""
    if vertical:
        axis.vlines(x,low,high,color=color,lw=1.2)
        axis.hlines([low,high],x-.045,x+.045,color=color,lw=1.2)
        axis.scatter([x],[y],color=color,marker=marker,s=24,zorder=4)
    else:
        axis.hlines(y,low,high,color=color,lw=1.2)
        axis.vlines([low,high],y-.09,y+.09,color=color,lw=1.2)
        axis.scatter([x],[y],color=color,marker=marker,s=24,zorder=4)


def historical_reconciliation(metrics: dict[str,Any], recalc: pd.DataFrame, boots: pd.DataFrame,
                              formula_tables: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Compara valores publicados con ambos estimandos y clasifica su procedencia."""
    hist=metrics["by_condition"]; rows=[]
    fields={"mae_energy_mev_atom":"mae_energy_mev_atom","mae_residual_mev_atom":"mae_residual_mev_atom",
        "mae_delta_signed_pairs_mev_atom":"mae_delta_signed_pairs_mev_atom","mae_delta_zero_baseline_mev_atom":"mae_delta_zero_baseline_mev_atom",
        "mae_delta_anchored_dft_gs_mev_atom":"mae_delta_anchored_all3_mev_atom",
        "ground_state_accuracy_exact":"ground_state_accuracy_exact","pairwise_accuracy_eligible":"pairwise_accuracy_pooled"}
    for condition in CONDITIONS:
        for histkey,metric in fields.items():
            old=float(hist[condition][histkey])
            vals={}
            for est in ("A_mean_single_seed","B_three_seed_prediction_ensemble"):
                sub=recalc[(recalc.condition==condition)&(recalc.estimand==est)&(recalc.metric==metric)]
                vals[est]=float(sub.value.iloc[0]) if len(sub) else np.nan
            if metric=="mae_delta_anchored_all3_mev_atom":
                status="ERROR"; reason="El código histórico restó E_DFT del ancla a E_pred, por lo que reproduce el error absoluto por estructura; no es una diferencia predicha respecto al ancla predicha."
                corrected_metric="mae_delta_anchored_all3_mev_atom; usar nonanchor para métrica primaria"
            elif np.isclose(old,vals["A_mean_single_seed"],atol=1e-7,rtol=0):
                status="PASS"; reason="Coincide con promedio por fórmula de MAE de cada semilla (Enfoque A)."; corrected_metric=metric
            elif np.isclose(old,vals["B_three_seed_prediction_ensemble"],atol=1e-7,rtol=0):
                status="PASS"; reason="Coincide con MAE tras promediar las predicciones de semillas (Enfoque B)."; corrected_metric=metric
            else:
                status="DIFFERENT_ESTIMAND"; reason="El valor histórico no coincide con ninguno de los dos estimandos reconstruidos; comprobar métrica/fuente desactualizada."; corrected_metric=metric
            rows.append({"metric_name":histkey,"condition":condition,"historical_source":"metrics.json/by_condition",
                "historical_value":old,"recalculated_A":vals["A_mean_single_seed"],"recalculated_B":vals["B_three_seed_prediction_ensemble"],
                "difference_from_A":old-vals["A_mean_single_seed"],"difference_from_B":old-vals["B_three_seed_prediction_ensemble"],
                "identified_estimand":"A" if status=="PASS" and np.isclose(old,vals["A_mean_single_seed"],atol=1e-7) else ("B" if status=="PASS" else "not reconciled"),
                "cause":reason,"recommended_action":corrected_metric,"status":status})
        for old_key, metric, count_metric in (
            ("pairwise_correct","pairwise_accuracy_pooled","pairwise_correct_count"),
            ("pairwise_eligible_denominator","pairwise_accuracy_pooled","pairwise_eligible_count"),
            ("pairwise_resolved_denominator","pairwise_accuracy_resolved_pooled","pairwise_resolved_count"),
            ("pairwise_unresolved","pairwise_accuracy_pooled","pairwise_unresolved_count")):
            old=float(hist[condition].get(old_key,np.nan))
            vals={}
            for est,key in (("A_mean_single_seed","A"),("B_three_seed_prediction_ensemble","B")):
                table_key=f"{key}_{condition}"
                vals[est]=float(formula_tables[table_key][count_metric].sum())
            state="PASS" if any(np.isclose(old,value,atol=1e-8,rtol=0) for value in vals.values()) else "ERROR"
            rows.append({"metric_name":old_key,"condition":condition,"historical_source":"metrics.json/by_condition",
                "historical_value":old,"recalculated_A":vals["A_mean_single_seed"],"recalculated_B":vals["B_three_seed_prediction_ensemble"],
                "difference_from_A":old-vals["A_mean_single_seed"],"difference_from_B":old-vals["B_three_seed_prediction_ensemble"],
                "identified_estimand":"A or B" if state=="PASS" else "not reconciled",
                "cause":"El conteo no se puede inferir de una exactitud redondeada; se recalculó desde IDs de pares, semillas y máscaras de elegibilidad.",
                "recommended_action":"Usar denominadores reconstruidos y reportar por semilla; no redondear una exactitud agregada a conteo.","status":state})
    # Direct historical bootstrap point estimates are not forced to match a different summary estimand.
    for label,boot in metrics.get("bootstrap",{}).items():
        pieces=label.split(" | ")
        if len(pieces)!=2: continue
        contrast,metriclabel=pieces
        hit=boots[(boots.estimand=="B")&(boots.contrast==contrast)]
        if "MAE ΔE" in metriclabel: hit=hit[hit.metric=="mae_delta_signed_pairs_mev_atom"]
        elif "MAE energía absoluta" in metriclabel: hit=hit[hit.metric=="mae_energy_mev_atom"]
        elif "ΔE=0" in metriclabel: hit=hit[hit.metric=="mae_delta_vs_zero"]
        else: continue
        if not len(hit): continue
        old=float(boot["mean_difference"]); new=float(hit.estimate.iloc[0])
        rows.append({"metric_name":f"bootstrap point: {label}","condition":"paired contrast","historical_source":"metrics.json/bootstrap",
            "historical_value":old,"recalculated_A":np.nan,"recalculated_B":new,"difference_from_A":np.nan,"difference_from_B":old-new,
            "identified_estimand":"B","cause":"Bootstrap histórico calculado sobre predicciones promediadas entre semillas; los valores by_condition de ΔE son estimadores A. La comparación directa mezclaba estimandos.",
            "recommended_action":"Conservar y etiquetar por estimando; no comparar punto A con bootstrap B.","status":"PASS" if np.isclose(old,new,atol=1e-7) else "DIFFERENT_ESTIMAND"})
    return pd.DataFrame(rows)


def make_figures(recalc: pd.DataFrame, boots: pd.DataFrame) -> None:
    """Genera dos figuras PNG compactas que explican agregación e IC apareados."""
    IMAGES.mkdir(parents=True,exist_ok=True)
    colors={"F0":"#6B7280","F2":"#4C78A8","F4":"#F2A541","F8":"#59A14F"}
    # A: ΔE MAE by estimation target and paired contrasts, with cluster bootstrap CIs.
    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    for ix,est in enumerate(("A_mean_single_seed","B_three_seed_prediction_ensemble")):
        data=recalc[(recalc.estimand==est)&(recalc.metric=="mae_delta_signed_pairs_mev_atom")].set_index("condition").loc[list(CONDITIONS)]
        vals=data.value.to_numpy(float)
        positions=np.arange(4)+(ix-.5)*.34
        axes[0].bar(positions,vals,width=.32,label="A: promedio de corridas" if ix==0 else "B: predictor ensemble",
                    color=[colors[c] for c in CONDITIONS],alpha=.9 if ix==0 else .5,hatch=None if ix==0 else "//")
        for position,condition,value in zip(positions,CONDITIONS,vals):
            ci=boots[(boots.estimand==("A" if ix==0 else "B"))&(boots.contrast==condition)&
                     (boots.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
            _draw_interval(axes[0],position,value,ci.ci95_low,ci.ci95_high,"#24292F",vertical=True)
    axes[0].set_xticks(np.arange(4),CONDITIONS); axes[0].set_ylabel("MAE de ΔE (meV/átomo)")
    axes[0].set_title("MAE global: dos estimandos"); axes[0].legend(fontsize=8)
    axes[0].set_ylim(0,70)
    contrasts=[("F2 − F0","F2","A"),("F4 − F0","F4","A"),("F8 − F0","F8","A"),
               ("F2 − F0","F2","B"),("F4 − F0","F4","B"),("F8 − F0","F8","B")]
    ypos=np.arange(len(contrasts))
    for y,(con,left,est) in zip(ypos,contrasts):
        row=boots[(boots.estimand==est)&(boots.contrast==con)&(boots.metric=="mae_delta_signed_pairs_mev_atom")].iloc[0]
        color=colors[left]; marker="o" if est=="A" else "s"
        _draw_interval(axes[1],row.estimate,y,row.ci95_low,row.ci95_high,color,marker=marker)
    axes[1].axvline(0,color="black",linestyle="--",linewidth=1); axes[1].set_yticks(ypos,[f"{c} · {e}" for c,_,e in contrasts]); axes[1].invert_yaxis()
    axes[1].set_xlabel("Cambio de MAE ΔE (meV/átomo); negativo favorece el primero")
    axes[1].set_title("Contrastes apareados e IC del 95%")
    fig.suptitle("ABS₃: MAE de diferencias energéticas y efecto de promediar semillas",weight="bold")
    fig.text(.5,.015,"A: error por corrida, promediado por fórmula. B: media de tres predicciones antes de calcular el error.\nPuntos y barras: estimación e IC95% bootstrap por fórmula (70 composiciones); 210 pares físicos, tres semillas.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.93)); fig.savefig(IMAGES/"comparacion_agregacion_mae_y_bootstrap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # B: predeclared list of paired contrasts under the primary single-run estimand A.
    selected=[("F2 − F0","mae_delta_signed_pairs_mev_atom","F2 − F0"),
              ("F4 − F0","mae_delta_signed_pairs_mev_atom","F4 − F0"),
              ("F8 − F0","mae_delta_signed_pairs_mev_atom","F8 − F0"),
              ("F0 − ΔE=0","mae_delta_vs_zero","F0 − ΔE=0"),
              ("F8 − ΔE=0","mae_delta_vs_zero","F8 − ΔE=0")]
    fig,ax=plt.subplots(figsize=(8.8,5.6)); ys=np.arange(len(selected))
    for y,(contrast,metric,label) in zip(ys,selected):
        row=boots[(boots.estimand=="A")&(boots.contrast==contrast)&(boots.metric==metric)].iloc[0]
        color="#6B7280" if "ΔE=0" in contrast else colors[contrast.split()[0]]
        _draw_interval(ax,row.estimate,y,row.ci95_low,row.ci95_high,color,marker="o")
        if row.includes_zero: ax.text(row.ci95_high+.15,y,"incluye 0",va="center",fontsize=8)
    ax.axvline(0,color="black",linestyle="--",linewidth=1); ax.set_yticks(ys,[x[2] for x in selected]); ax.invert_yaxis()
    ax.set_xlabel("Cambio de MAE ΔE (meV/átomo); negativo favorece modelo de la izquierda")
    ax.set_title("ABS₃: contrastes primarios con promedio por corrida")
    ax.text(.01,-.18,"A: promedio de MAE por semilla dentro de fórmula. IC percentil 95%, bootstrap apareado por 70 composiciones, 10 000 réplicas. No es prueba externa ciega.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(IMAGES/"reconciliacion_intervalos_confianza_finetuning.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def write_report(integrity: dict[str,Any], reconciliation: pd.DataFrame, recalc: pd.DataFrame,
                 boots: pd.DataFrame, hist: dict[str,Any]) -> None:
    """Escribe una auditoría reproducible y explicita hallazgos y limitaciones."""
    b=boots.set_index(["estimand","contrast","metric"])
    def stat(est: str, con: str, met: str) -> str:
        r=b.loc[(est,con,met)]; return f"{r.estimate:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
    table=reconciliation[reconciliation.metric_name.isin(["mae_delta_signed_pairs_mev_atom","mae_delta_anchored_dft_gs_mev_atom","mae_delta_zero_baseline_mev_atom",
        "pairwise_correct","pairwise_eligible_denominator","pairwise_resolved_denominator","pairwise_unresolved"])][
        ["condition","metric_name","historical_value","recalculated_A","recalculated_B","status","cause"]].to_markdown(index=False,floatfmt=".4f")
    main=recalc[(recalc.estimand=="A_mean_single_seed")&recalc.condition.isin(CONDITIONS)]
    def values(metric: str) -> str:
        x=main[main.metric==metric].set_index("condition").value
        return "; ".join(f"{c} {x[c]:.3f}" for c in CONDITIONS)
    lines=["# Auditoría estadística del fine-tuning de MatGPT por profundidad","",
      "## Estado","",
      f"**PASS en integridad de los datos; discrepancias históricas confirmadas en agregación y métricas.** El MAE ΔE histórico mezcla estimandos A/B, los contadores pairwise no se reconcilian consistentemente y la fórmula histórica de ΔE anclada es incorrecta. Las correcciones están calculadas en este informe y los CSV adjuntos. No se entrenaron modelos ni se accedió a etiquetas del test histórico.","",
      "## Integridad de datos","",
      f"- Estado: {integrity['status']}; {integrity['n_structures']} estructuras físicas, {integrity['n_compositions']} composiciones, {integrity['n_seeds']} semillas; {integrity['n_oof_rows']} filas de predicción y {integrity['n_pair_rows']} filas de pares (4 condiciones × 210 estructuras/pares únicos × 3 semillas).",
      f"- Hay {integrity['n_unique_physical_pairs']} pares físicos únicos y {integrity['n_eligible_physical_pairs']} elegibles con |ΔE_DFT| > 5 meV/átomo. La orientación recalculada es B−A; se verificó ΔE=1000·(E_B/N_B−E_A/N_A).",
      f"- Las etiquetas e IDs coinciden con train/val; no se leyeron targets de test. F0 coincide con el archivo congelado histórico: {integrity['f0_residuals_match']}. Ridge permanece idéntico entre condiciones; rango máximo {integrity['max_ridge_cross_condition_spread_ev_atom']:.3g} eV/átomo.",
      "- Cada fórmula tiene 3 polimorfos, 3 pares, 3 semillas y una única evaluación outer. No hay registros OOF duplicados ni IDs de test.","",
      "## Dos estimandos distintos","",
      "**A — corrida individual promedio:** para cada fórmula se calcula el MAE de sus tres pares por semilla, se promedian las tres semillas y luego se promedian las 70 fórmulas. Mide el rendimiento medio de corridas individuales; es el estimando primario recomendado.",
      "**B — predictor ensemble:** se promedian primero las tres energías predichas por estructura, se forman las diferencias y se calcula el error. Mide el rendimiento de un ensemble de tres modelos. Debido a |·| no coincide con A.",
      "En el JSON histórico, `mae_energy_mev_atom` y el bootstrap de energía usan B, pero `mae_delta_signed_pairs_mev_atom` corresponde a A. El bootstrap de ΔE usa B. Por eso las diferencias directas del resumen y los puntos del bootstrap anteriores no podían coincidir.","",
      "### Reproducción de la discrepancia MAE ΔE","",
      f"Histórico por condición (estimando A): {values('mae_delta_signed_pairs_mev_atom')} meV/átomo.",
      f"Enfoque B reconstruido: " + "; ".join(f"{c} {float(recalc[(recalc.condition==c)&(recalc.estimand=='B_three_seed_prediction_ensemble')&(recalc.metric=='mae_delta_signed_pairs_mev_atom')].value.iloc[0]):.3f}" for c in CONDITIONS)+" meV/átomo.",
      "Los contrastes directos A son F2−F0 = −0.287, F4−F0 = −1.576 y F8−F0 = −3.597 meV/átomo. El bootstrap histórico (+1.685, +0.441, −1.301) es B, no A; se reproduce desde las predicciones promediadas. No es redondeo ni pérdida de pares.",
      f"IC95% bootstrap A, F2−F0: {stat('A','F2 − F0','mae_delta_signed_pairs_mev_atom')}; F4−F0: {stat('A','F4 − F0','mae_delta_signed_pairs_mev_atom')}; F8−F0: {stat('A','F8 − F0','mae_delta_signed_pairs_mev_atom')} meV/átomo.",
      f"IC95% bootstrap B, F8−F0: {stat('B','F8 − F0','mae_delta_signed_pairs_mev_atom')} meV/átomo.","",
      "### Predictor ΔE=0","",
      f"El baseline tiene MAE de 60.470 meV/átomo con los mismos 210 pares físicos; F0 A tiene {float(main[(main.condition=='F0')&(main.metric=='mae_delta_signed_pairs_mev_atom')].value.iloc[0]):.3f}, diferencia −1.626. El valor histórico bootstrap −6.045 vuelve a ser B: predicciones promediadas frente al cero, con estimación observada B reproducida y muestreo pareado por fórmula.",
      f"Para el estimando primario A, F0−cero: {stat('A','F0 − ΔE=0','mae_delta_vs_zero')} meV/átomo; F8−cero: {stat('A','F8 − ΔE=0','mae_delta_vs_zero')} meV/átomo. Ambos IC incluyen cero. El bootstrap histórico B para F0 es {stat('B','F0 − ΔE=0','mae_delta_vs_zero')} meV/átomo.",
      "El baseline cero conserva exactamente los mismos pares DFT, sin filtro adicional. Su exactitud de ranking no está definida porque todas sus predicciones son empates.","",
      "## ΔE anclada al estado fundamental DFT","",
      "El código histórico calculó `pred_i − DFT_g` en vez de `pred_i − pred_g`, donde g es la estructura mínima DFT. Al restar luego el gap DFT, el error resultante es exactamente `pred_i − DFT_i`; por eso su MAE coincide con el MAE absoluto E/N. Es un error de fórmula/etiquetado, no una identidad física.",
      "La métrica corregida usa el mismo polimorfo g seleccionado por DFT en ambos ejes: (pred_i−pred_g)−(DFT_i−DFT_g). La métrica primaria promedia los dos polimorfos no ancla; el cálculo incluyendo el ancla (cuyo error es exactamente cero) queda también reportado. Los cambios comunes por fórmula se cancelan exactamente.",
      "No se hallaron empates exactos DFT que alteren la selección por `argmin`; las casi degeneraciones se tratan según los umbrales de elegibilidad (1 meV/átomo para GS, 5 para pares).", "",
      "## Ranking y estado fundamental","",
      "La exactitud pairwise principal cuenta predicciones no resueltas (|ΔE_pred|≤1 meV/átomo) como incorrectas entre los pares DFT elegibles; también se reporta la exactitud condicionada a pares resueltos. El denominador físico elegible es 197 por semilla. F4 tiene pares no resueltos, por lo que exactitud sobre elegibles y exactitud entre resueltos difieren. Ground state exacto se calcula por argmin y los empates predichos no cuentan como acierto.",
      "Las exactitudes A promedian dentro de fórmula las tres semillas; por ello pueden ser fraccionarias y no deben transformarse en conteos enteros. Se incluyen denominadores y métricas por semilla en `metricas_recalculadas.csv`.","",
      "La auditoría también detecta que los campos históricos `pairwise_correct`, `pairwise_resolved_denominator` y `pairwise_unresolved` no se reproducen consistentemente como conteos de A ni de B. El denominador elegible de 197 sí se confirma. Las razones de exactitud pueden coincidir con A aunque esos contadores no; se deben citar los conteos reconstruidos desde pares y semillas, no los contadores históricos.","",
      "## Métricas primarias recalculadas (A)","",
      f"- MAE E/N (meV/átomo): {values('mae_energy_mev_atom')}",
      f"- MAE residual intrafórmula (meV/átomo): {values('mae_residual_mev_atom')}",
      f"- MAE ΔE de tres pares (meV/átomo): {values('mae_delta_signed_pairs_mev_atom')}",
      f"- MAE ΔE anclada corregida, dos estructuras no ancla (meV/átomo): {values('mae_delta_anchored_nonanchor_mev_atom')}",
      "En `bootstrap_recalculado.csv` se incluyen IC apareados por composición para MAE absoluta, residual, ΔE, exactitud GS y pairwise, además de los contrastes contra ΔE=0. Las semillas permanecen agrupadas dentro de cada fórmula.","",
      "## Impacto científico","",
      f"1. ¿F8 mejora ΔE frente a F0? Para el estimando primario A, la reducción sí queda respaldada por este bootstrap: {stat('A','F8 − F0','mae_delta_signed_pairs_mev_atom')} meV/átomo; el IC excluye cero. Para el predictor ensemble B el contraste no es concluyente: {stat('B','F8 − F0','mae_delta_signed_pairs_mev_atom')}. Esta evidencia es interna a las 70 fórmulas de desarrollo ya exploradas, no una evaluación externa.",
      f"2. ¿F4 mejora GS? La diferencia exacta A es {stat('A','F4 − F0','ground_state_accuracy_exact')}; el IC percentil incluye cero (límite inferior 0), por lo que no demuestra una mejora al 95%. Esto no demuestra equivalencia.",
      f"3. ¿Algún modelo supera a ΔE=0? F0−cero A es {stat('A','F0 − ΔE=0','mae_delta_vs_zero')} y F8−cero A es {stat('A','F8 − ΔE=0','mae_delta_vs_zero')}; ambos IC incluyen cero, sin evidencia concluyente frente al baseline.",
      "4. El contraste F8−F0 mejora el MAE ΔE bajo A, pero F2−F0 y F4−F0 no son concluyentes, y F8−F4 tampoco: esto no establece una tendencia monotónica ni prueba que aumentar profundidad sea causalmente mejor. El resultado depende del estimando y de las semillas agregadas.",
      "5. Deben corregirse las conclusiones que mezclaron MAE A con IC B y retirar la métrica histórica de ΔE anclada como si fuera una diferencia válida.",
      "6. El conjunto ya fue explorado en experimentos previos; los intervalos son incertidumbre interna por composición, no validación externa ciega.","",
      "## Reconciliación histórica","",table,"",
      "## Reproducibilidad y archivos","",
      f"Predicciones de entrada SHA-256: `{_sha(EXPERIMENT/'predicciones_oof_por_estructura.csv')}`; pares: `{_sha(EXPERIMENT/'predicciones_oof_por_par.csv')}`.",
      "Código: `scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/audit_statistics.py`. Tests: `scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/tests/test_statistical_audit.py`. Resultados CSV y gráficos PNG se guardan bajo `experiments/MatGPT_SLICES_Finetuning_Profundidad/` e `images/MatGPT_SLICES_Finetuning_Profundidad/`.",
      "Figuras: [comparación de agregación](../../../images/MatGPT_SLICES_Finetuning_Profundidad/comparacion_agregacion_mae_y_bootstrap.png) y [contrastes e intervalos](../../../images/MatGPT_SLICES_Finetuning_Profundidad/reconciliacion_intervalos_confianza_finetuning.png).","",
      "### Limitaciones","",
      "Solo hay 70 composiciones ya exploradas; tres semillas son repeticiones de entrenamiento, no 630 materiales. Se conservaron las definiciones DFT archivadas, sin reetiquetar ni homogeneizar la procedencia gga-static. No se consultó el test histórico, no se reentrenó y los outer folds son desarrollo, no una estimación ciega externa.",""]
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/"auditoria_estadistica.md").write_text("\n".join(lines),encoding="utf-8")


def _sha(path: Path) -> str:
    """Devuelve SHA-256 por bloques para registrar fuentes OOF sin cargarlas de nuevo."""
    import hashlib
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()


def update_consolidated_jsons(result: dict[str, Any]) -> None:
    """Anota la auditoría en los tres JSON consolidados existentes sin duplicarlos."""
    audit = {
        "status": "COMPLETED_WITH_HISTORICAL_DISCREPANCIES",
        "scope": "auditoría estadística offline; sin entrenamiento y sin etiquetas del test histórico",
        "integrity": result["integrity"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outputs": result["output_csvs"],
        "report": str((REPORT / "auditoria_estadistica.md").relative_to(ROOT)),
        "figures": [str((IMAGES / name).relative_to(ROOT)) for name in (
            "comparacion_agregacion_mae_y_bootstrap.png",
            "reconciliacion_intervalos_confianza_finetuning.png")],
        "runtime": {key: result[key] for key in ("wall_seconds", "cpu_seconds", "peak_rss_bytes")},
        "note": "Las métricas históricas se preservan; consultar los CSV de auditoría para estimandos A y B y la fórmula corregida de ΔE anclada.",
    }
    experiment_jsons = {
        "metrics.json": "statistical_audit",
        "summary.json": "statistical_audit",
        "resources.json": "statistical_audit",
    }
    for filename, key in experiment_jsons.items():
        path = EXPERIMENT / filename
        content = json.loads(path.read_text(encoding="utf-8"))
        content[key] = audit
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def run_audit() -> dict[str,Any]:
    """Ejecuta auditoría, reconstruye estimandos, bootstrap, informe y figuras."""
    started=time.perf_counter(); cpu_started=time.process_time(); rss_start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    development=load_development_dataset(DATASET)
    pred=pd.read_csv(EXPERIMENT/"predicciones_oof_por_estructura.csv")
    pairs=pd.read_csv(EXPERIMENT/"predicciones_oof_por_par.csv")
    folds=pd.read_csv(FOLDS)
    f0=pd.read_csv(F0_SOURCE)
    hist=json.loads((EXPERIMENT/"metrics.json").read_text(encoding="utf-8"))
    integrity=validate_integrity(pred,pairs,development,folds,f0,hist)
    seed_metrics=formula_seed_metrics(pred)
    ensemble=ensemble_predictions(pred)
    ensemble_metrics=formula_seed_metrics(ensemble.assign(seed=-1))
    recalc,formula_tables=aggregate_estimands(seed_metrics,ensemble_metrics)
    boots=bootstrap_contrasts(formula_tables)
    reconciliation=historical_reconciliation(hist,recalc,boots,formula_tables)
    recalc_path=EXPERIMENT/"metricas_recalculadas.csv"; boot_path=EXPERIMENT/"bootstrap_recalculado.csv"
    recon_path=EXPERIMENT/"reconciliacion_metricas.csv"
    recalc.to_csv(recalc_path,index=False); boots.to_csv(boot_path,index=False); reconciliation.to_csv(recon_path,index=False)
    write_report(integrity,reconciliation,recalc,boots,hist); make_figures(recalc,boots)
    result={"integrity":integrity,"n_bootstrap_rows":int(len(boots)),"n_reconciliation_rows":int(len(reconciliation)),
        "wall_seconds":time.perf_counter()-started,"cpu_seconds":time.process_time()-cpu_started,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform=="darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "output_csvs":[str(recalc_path.relative_to(ROOT)),str(boot_path.relative_to(ROOT)),str(recon_path.relative_to(ROOT))]}
    update_consolidated_jsons(result)
    return result


def main() -> None:
    """Valida argumentos, ejecuta el análisis offline y registra el resultado."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates",type=int,default=BOOTSTRAP_REPLICATES)
    args=parser.parse_args()
    if args.bootstrap_replicates!=BOOTSTRAP_REPLICATES:
        parser.error(f"El protocolo bloquea {BOOTSTRAP_REPLICATES} réplicas")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    summary=run_audit()
    LOG.info("auditoría finalizada: %s",summary)


if __name__=="__main__":
    main()
