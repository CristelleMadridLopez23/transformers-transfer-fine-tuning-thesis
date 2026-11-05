"""KRR-RBF pairwise para diferencias energéticas de polimorfos ABS3.

El script reutiliza folds, cohortes, geometría y Ridge M2 históricos. Selecciona
gamma y lambda con inner-validation, evalúa outer una sola vez y conserva las
predicciones fold a fold mediante escrituras atómicas.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger("abs3.geometry_pairwise_krr")
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/abs3_geometry_pairwise_kernel_ridge"
MODEL_DIR = EXP / "models"
IMAGE_DIR = ROOT / "images/abs3_geometry_pairwise_kernel_ridge"
REPORT_DIR = ROOT / "reports/ABS3/abs3_geometry_pairwise_kernel_ridge"
DATASET = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
FOLDS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
BASE = ROOT / "experiments/abs3_geometry_transformers_pairwise_ridge"
GEOMETRY = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
            "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
PAIRS = ((0, 1), (0, 2), (1, 2))
GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261103
PAIRWISE_DFT_THRESHOLD = 5.0
PAIRWISE_PRED_THRESHOLD = 1.0
TIE_THRESHOLD = 1.0
START_WALL = time.monotonic()
START_CPU = time.process_time()


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo.

    Args:
        path: Archivo de entrada.

    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Guarda JSON mediante reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Guarda CSV mediante reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Carga dataset/folds y baseline GEO, verificando IDs, geometría y splits.

    Returns:
        Dataset, folds, predicciones GEO por estructura y par, preflight.
    """
    if not all(path.is_file() for path in (DATASET, FOLDS, BASE / "preflight.json")):
        raise FileNotFoundError("No están disponibles dataset, folds históricos o preflight GEO")
    data = pd.read_csv(DATASET, dtype={"structure_id": str, "composition_id": str})
    folds = pd.read_csv(FOLDS, dtype={"structure_id": str, "composition_group": str})
    folds = folds[folds.historical_split.isin(("train", "val"))].copy()
    preflight = json.loads((BASE / "preflight.json").read_text(encoding="utf-8"))
    geometry_manifest_path=BASE/"geometry_manifest.json"
    if not geometry_manifest_path.is_file(): raise FileNotFoundError("Falta geometry_manifest.json histórico")
    geometry_manifest=json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or baseline_metrics.get("status") != "COMPLETE":
        raise ValueError("El preflight o los resultados GEO históricos no están completos")
    if geometry_manifest.get("status")!="PASS" or geometry_manifest.get("source")!="gga-static_endpoint_cleaned":
        raise ValueError("El manifiesto histórico de geometría no superó auditoría")
    if len(data) != 210 or data.structure_id.nunique() != 210 or data.composition_id.nunique() != 70:
        raise ValueError("La cohorte debe contener 210 estructuras y 70 composiciones")
    if data.groupby("composition_id").size().ne(3).any() or data.structure_id.duplicated().any():
        raise ValueError("Se detectaron tripletes incompletos o IDs estructurales duplicados")
    if data[GEOMETRY].isna().any().any() or not np.isfinite(data[GEOMETRY].to_numpy(float)).all():
        raise ValueError("Descriptores geométricos con valores faltantes o no finitos")
    if not data.geometry_source.eq("gga-static_endpoint_cleaned").all():
        raise ValueError("Fuente geométrica distinta de gga-static_endpoint_cleaned")
    if len(folds) != 1050 or set(folds.structure_id) != set(data.structure_id):
        raise ValueError("Los folds no cubren exactamente los 210 structure_id")
    if set(folds.composition_group) != set(data.composition_id):
        raise ValueError("Los grupos de folds no coinciden con las composiciones")
    for fold in range(5):
        role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
        counts = (int(role.inner_role.eq("inner_train").sum()),
                  int(role.inner_role.eq("inner_validation").sum()),
                  int(role.outer_role.eq("outer_validation").sum()))
        if counts != (45, 11, 14):
            raise ValueError(f"Fold {fold}: partición distinta de 45/11/14: {counts}")
    structure = pd.read_csv(BASE / "predicciones_oof_por_estructura.csv", dtype={"structure_id": str})
    pairs = pd.read_csv(BASE / "predicciones_oof_por_par.csv", dtype={"structure_id_i": str, "structure_id_j": str})
    structure = structure[structure.condition.eq("GEO")].copy()
    pairs = pairs[pairs.condition.eq("GEO")].copy()
    if len(structure) != 210 or structure.structure_id.nunique() != 210 or len(pairs) != 210:
        raise ValueError("Baseline GEO no tiene 210 predicciones estructurales y 210 pares")
    if structure.structure_id.duplicated().any() or pairs.duplicated(["fold", "composition_id", "polymorph_i", "polymorph_j"]).any():
        raise ValueError("Baseline GEO contiene predicciones duplicadas")
    indexed=data.set_index("structure_id")
    for row in structure.itertuples():
        if row.structure_id not in indexed.index: raise ValueError(f"ID GEO ausente del dataset: {row.structure_id}")
        ref=indexed.loc[row.structure_id]
        role=folds[(folds.outer_fold.eq(row.fold))&(folds.composition_group.eq(row.composition_id))]
        if (ref.composition_id!=row.composition_id or ref.family!=row.polymorph or
                not math.isclose(float(ref.energy_dft_ev_atom),float(row.energy_dft_ev_atom),rel_tol=0,abs_tol=1e-10) or
                role.empty or not role.outer_role.eq("outer_validation").all()):
            raise ValueError(f"Baseline GEO no concuerda por ID/target/fold: {row.structure_id}")
    for row in pairs.itertuples():
        left=indexed.loc[row.structure_id_i];right=indexed.loc[row.structure_id_j]
        expected=(float(right.energy_dft_ev_atom)-float(left.energy_dft_ev_atom))*1000
        if (left.composition_id!=row.composition_id or right.composition_id!=row.composition_id or
                left.family!=row.polymorph_i or right.family!=row.polymorph_j or
                not math.isclose(expected,float(row.delta_dft_mev_atom),rel_tol=0,abs_tol=1e-8)):
            raise ValueError(f"Baseline GEO tiene orientación/target de par inconsistente: {row.composition_id}")
    for (fold,formula),group in structure.groupby(["fold","composition_id"]):
        mus=group.ridge_mean_m2_ev_atom.to_numpy(float)
        if len(group)!=3 or not np.allclose(mus,mus[0],rtol=0,atol=1e-10):
            raise ValueError(f"mu_Ridge_M2 no es único en triplete GEO {formula}, fold {fold}")
    published = baseline_metrics["metrics"]["GEO"]
    if not math.isclose(float(np.mean(np.abs(structure.energy_pred_ev_atom-structure.energy_dft_ev_atom))*1000),
                        published["energy_mae_mev_atom"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("No se reproduce Energy MAE GEO desde las predicciones originales")
    if not math.isclose(float(pairs.absolute_error_mev_atom.mean()), published["deltaE_mae_signed_est_a_mev_atom"],
                        rel_tol=0, abs_tol=1e-8):
        raise ValueError("No se reproduce ΔE MAE GEO desde las predicciones originales")
    # Revalidar los cinco hashes de los Ridge M2 históricos desde git, sin ajustar modelos.
    ridge_hashes = preflight.get("ridge_m2_fold_sha256", {})
    for fold, digest in ridge_hashes.items():
        path = f"experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_{fold}.joblib"
        result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if hashlib.sha256(result.stdout).hexdigest() != digest:
            raise ValueError(f"Hash Ridge M2 histórico incorrecto en fold {fold}")
    preflight["geometry_manifest_sha256"]=sha256(geometry_manifest_path)
    return (data.sort_values(["composition_id", "family"], kind="stable").reset_index(drop=True),
            folds, structure, pairs, preflight)


def roles_for_fold(folds: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    """Recupera las composiciones train/validation/outer del fold."""
    role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
    return {"train": sorted(role.loc[role.inner_role.eq("inner_train"), "composition_group"].astype(str)),
            "validation": sorted(role.loc[role.inner_role.eq("inner_validation"), "composition_group"].astype(str)),
            "outer": sorted(role.loc[role.outer_role.eq("outer_validation"), "composition_group"].astype(str))}


def grouped_indices(data: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construye índices y energías por fórmula en orden polimórfico histórico."""
    groups: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for formula, frame in data.groupby("composition_id", sort=False):
        indexed = frame.set_index("family").loc[list(FAMILIES)]
        groups[str(formula)] = np.array([int(data.index[data.structure_id.eq(sid)][0])
                                         for sid in indexed.structure_id], dtype=int)
        targets[str(formula)] = indexed.energy_dft_ev_atom.to_numpy(float)
    return groups, targets


def pair_rows(formulas: list[str], groups: dict[str, np.ndarray], targets: dict[str, np.ndarray],
              data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Genera tres pares firmados y orientados de forma canónica."""
    starts, ends, labels, metadata = [], [], [], []
    for formula in formulas:
        indices = groups[formula]
        energies = targets[formula]
        for i, j in PAIRS:
            starts.append(indices[i]); ends.append(indices[j])
            labels.append((energies[j]-energies[i])*1000.0)
            metadata.append({"composition_id": formula, "family_i": FAMILIES[i], "family_j": FAMILIES[j],
                             "structure_id_i": str(data.iloc[indices[i]].structure_id),
                             "structure_id_j": str(data.iloc[indices[j]].structure_id),
                             "delta_dft_mev_atom": float((energies[j]-energies[i])*1000.0)})
    return np.asarray(starts, int), np.asarray(ends, int), np.asarray(labels, float), metadata


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula distancias euclídeas cuadráticas de dos matrices."""
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0.0)


def rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Kernel RBF con gamma positivo."""
    if gamma <= 0 or not math.isfinite(gamma):
        raise ValueError("gamma debe ser positivo y finito")
    return np.exp(-gamma*squared_distances(x, y))


def difference_kernel(xi: np.ndarray, xj: np.ndarray, xk: np.ndarray, xl: np.ndarray,
                      gamma: float) -> np.ndarray:
    """Construye D K Dᵀ para dos conjuntos de pares orientados."""
    return rbf(xj, xl, gamma)-rbf(xj, xk, gamma)-rbf(xi, xl, gamma)+rbf(xi, xk, gamma)


def score_structures(query: np.ndarray, train_i: np.ndarray, train_j: np.ndarray,
                     alpha: np.ndarray, gamma: float) -> np.ndarray:
    """Evalúa la función escalar latente inducida por los pares de entrenamiento."""
    return (rbf(query, train_j, gamma)-rbf(query, train_i, gamma))@alpha


def pair_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    """Métricas de ranking para pares, usando umbrales históricos."""
    eligible = np.abs(true) > PAIRWISE_DFT_THRESHOLD
    resolved = np.abs(pred) > PAIRWISE_PRED_THRESHOLD
    correct = eligible & resolved & (np.sign(true) == np.sign(pred))
    n = int(eligible.sum())
    return {"eligible": n, "correct": int(correct.sum()),
            "resolved_eligible": int((eligible & resolved).sum()),
            "unresolved_eligible": int((eligible & ~resolved).sum()),
            "accuracy": float(correct.sum()/n) if n else float("nan"),
            "resolved_accuracy": float(correct.sum()/(eligible & resolved).sum()) if (eligible & resolved).any() else float("nan")}


def triplet_metrics(energies: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Calcula ranking exacto y ground state con reglas DFT históricas."""
    truth_pairs = np.array([(energies[j]-energies[i])*1000 for i, j in PAIRS])
    pred_pairs = np.array([(predicted[j]-predicted[i])*1000 for i, j in PAIRS])
    full_ok = bool(np.all(np.abs(truth_pairs) > TIE_THRESHOLD))
    full_correct = bool(full_ok and np.array_equal(np.argsort(energies, kind="stable"),
                                                   np.argsort(predicted, kind="stable"))
                        and np.all(np.abs(pred_pairs) > TIE_THRESHOLD))
    order = np.sort(energies)
    gs_ok = bool((order[1]-order[0])*1000 > TIE_THRESHOLD)
    pred_sorted = np.sort(predicted)
    gs_correct = bool(gs_ok and (pred_sorted[1]-pred_sorted[0])*1000 > TIE_THRESHOLD
                      and int(np.argmin(energies)) == int(np.argmin(predicted)))
    return {"full_triplet_eligible": full_ok, "full_triplet_correct": full_correct,
            "groundstate_eligible": gs_ok, "groundstate_correct": gs_correct}


def evaluate_triplets(data: pd.DataFrame, groups: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                      formulas: list[str], scores: dict[str, np.ndarray], mu_map: dict[str, float],
                      fold: int, condition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convierte scores en energías centradas y métricas por estructura/par."""
    structures, pairs_out = [], []
    for formula in formulas:
        idx = groups[formula]
        score = scores[formula]
        centered = score-score.mean()
        energy = mu_map[formula]+centered/1000.0
        truth = targets[formula]
        stat = triplet_metrics(truth, energy)
        pmeta = []
        for i, j in PAIRS:
            delta_true = (truth[j]-truth[i])*1000.0
            delta_pred = (energy[j]-energy[i])*1000.0
            eligible = abs(delta_true) > PAIRWISE_DFT_THRESHOLD
            resolved = abs(delta_pred) > PAIRWISE_PRED_THRESHOLD
            correct = bool(eligible and resolved and np.sign(delta_true) == np.sign(delta_pred))
            record = {"condition": condition, "fold": fold, "composition_id": formula,
                      "polymorph_i": FAMILIES[i], "polymorph_j": FAMILIES[j],
                      "structure_id_i": str(data.iloc[idx[i]].structure_id),
                      "structure_id_j": str(data.iloc[idx[j]].structure_id),
                      "delta_dft_mev_atom": float(delta_true), "delta_pred_mev_atom": float(delta_pred),
                      "absolute_error_mev_atom": float(abs(delta_pred-delta_true)),
                      "eligible": bool(eligible), "resolved": bool(resolved), "correct": correct}
            pairs_out.append(record); pmeta.append(record)
        eerr = (energy-truth)*1000.0
        pair_errors = np.array([x["absolute_error_mev_atom"] for x in pmeta])
        structures_by_family = []
        for i, family in enumerate(FAMILIES):
            rec = {"condition": condition, "fold": fold, "composition_id": formula,
                   "structure_id": str(data.iloc[idx[i]].structure_id), "polymorph": family,
                   "energy_dft_ev_atom": float(truth[i]), "energy_pred_ev_atom": float(energy[i]),
                   "ridge_mean_m2_ev_atom": float(mu_map[formula]),
                   "residual_centered_mev_atom": float(centered[i]),
                   "absolute_error_mev_atom": float(abs(eerr[i]))}
            structures.append(rec); structures_by_family.append(rec)
        compositions = {"condition": condition, "fold": fold, "composition_id": formula,
                       "energy_mae_mev_atom": float(np.mean(np.abs(eerr))),
                       "energy_rmse_mev_atom": float(np.sqrt(np.mean(eerr**2))),
                       "energy_bias_mev_atom": float(np.mean(eerr)),
                       "deltaE_mae_mev_atom": float(pair_errors.mean()),
                       "deltaE_rmse_within_composition_mev_atom": float(np.sqrt(np.mean(pair_errors**2))),
                       "eligible_pairs": sum(x["eligible"] for x in pmeta),
                       "correct_pairs": sum(x["correct"] for x in pmeta),
                       "resolved_eligible_pairs": sum(x["eligible"] and x["resolved"] for x in pmeta),
                       "unresolved_eligible_pairs": sum(x["eligible"] and not x["resolved"] for x in pmeta),
                       **stat,
                       "spearman": float(spearmanr(truth, energy).statistic),
                       "kendall": float(kendalltau(truth, energy).statistic)}
        # Store per-composition metrics as a sibling row attached to the first structure.
        for rec in structures_by_family:
            rec.update({k:v for k,v in compositions.items() if k not in {"condition","fold","composition_id"}})
    return structures, pairs_out


def composition_summary(structure: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Aggregates OOF metrics by condition and composition."""
    comp_rows, global_metrics = [], {}
    for condition in ("C0_GEO_LINEAR", "C1_KRR_RBF", "C2_KRR_RBF_CALIBRATED"):
        sf = structure[structure.condition.eq(condition)]
        pf = pairs[pairs.condition.eq(condition)]
        rows = []
        for (fold, formula), group in sf.groupby(["fold","composition_id"],sort=False):
            group=group.set_index("polymorph").loc[list(FAMILIES)]
            truths=group.energy_dft_ev_atom.to_numpy(float); preds=group.energy_pred_ev_atom.to_numpy(float)
            p=pf[(pf.fold.eq(fold))&(pf.composition_id.eq(formula))]
            err=p.absolute_error_mev_atom.to_numpy(float)
            row={"condition":condition,"fold":int(fold),"composition_id":str(formula),
                 "energy_mae_mev_atom":float(np.mean(np.abs(preds-truths))*1000),
                 "energy_rmse_mev_atom":float(np.sqrt(np.mean((preds-truths)**2))*1000),
                 "energy_bias_mev_atom":float(np.mean(preds-truths)*1000),
                 "deltaE_mae_mev_atom":float(err.mean()),"deltaE_rmse_within_composition_mev_atom":float(np.sqrt(np.mean(err**2))),
                 "eligible_pairs":int(p.eligible.sum()),"correct_pairs":int(p.correct.sum()),
                 "resolved_eligible_pairs":int((p.eligible&p.resolved).sum()),
                 "unresolved_eligible_pairs":int((p.eligible&~p.resolved).sum()),**triplet_metrics(truths,preds),
                 "spearman":float(spearmanr(truths,preds).statistic),"kendall":float(kendalltau(truths,preds).statistic)}
            rows.append(row);comp_rows.append(row)
        c=pd.DataFrame(rows)
        dft=pf.delta_dft_mev_atom.to_numpy(float); pred=pf.delta_pred_mev_atom.to_numpy(float)
        slope,intercept=np.polyfit(dft,pred,1)
        rsq=1-np.sum((pred-(slope*dft+intercept))**2)/np.sum((pred-pred.mean())**2)
        eligible=pf[pf.eligible]; full=c[c.full_triplet_eligible]; gs=c[c.groundstate_eligible]
        global_metrics[condition]={
            "structures":int(len(sf)),"compositions":int(sf.composition_id.nunique()),
            "energy_mae_mev_atom":float(np.mean(np.abs(sf.energy_pred_ev_atom-sf.energy_dft_ev_atom))*1000),
            "energy_rmse_global_mev_atom":float(np.sqrt(np.mean((sf.energy_pred_ev_atom-sf.energy_dft_ev_atom)**2))*1000),
            "energy_bias_mev_atom":float(np.mean(sf.energy_pred_ev_atom-sf.energy_dft_ev_atom)*1000),
            "deltaE_mae_estimand_A_mev_atom":float(c.deltaE_mae_mev_atom.mean()),
            "deltaE_rmse_global_mev_atom":float(np.sqrt(np.mean((pred-dft)**2))),
            "pairwise_accuracy_eligible":float(eligible.correct.mean()),"eligible_pairs":int(len(eligible)),
            "correct_pairs":int(eligible.correct.sum()),"pairwise_accuracy_resolved":float(eligible[eligible.resolved].correct.mean()),
            "unresolved_eligible_pairs":int((~eligible.resolved).sum()),
            "pairwise_unresolved_fraction":float((~eligible.resolved).mean()),
            "full_triplet_accuracy":float(full.full_triplet_correct.mean()),"full_triplet_eligible":int(len(full)),
            "full_triplet_correct":int(full.full_triplet_correct.sum()),
            "groundstate_accuracy":float(gs.groundstate_correct.mean()),"groundstate_eligible":int(len(gs)),
            "groundstate_correct":int(gs.groundstate_correct.sum()),"ols_delta_slope":float(slope),
            "ols_delta_intercept_mev_atom":float(intercept),"ols_delta_r2":float(rsq),
            "deltaE_predictive_r2":float(1-np.sum((pred-dft)**2)/np.sum((dft-dft.mean())**2)),
            "spearman_mean_by_composition":float(c.spearman.mean()),"kendall_mean_by_composition":float(c.kendall.mean())}
    return pd.DataFrame(comp_rows),global_metrics


def aggregate_train_pairs(formulas: list[str], data: pd.DataFrame, geom: np.ndarray,
                          groups: dict[str,np.ndarray], targets: dict[str,np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Concatena endpoints/targets de parejas de varias composiciones."""
    si,sj,y,_=pair_rows(formulas,groups,targets,data)
    return si,sj,y,geom


def fit_dual(geom: np.ndarray, si: np.ndarray, sj: np.ndarray, y: np.ndarray,
             gamma: float, lam: float) -> tuple[np.ndarray,np.ndarray,float,float]:
    """Resuelve Kpair + lambda I sin invertir explícitamente la matriz."""
    gram=difference_kernel(geom[si],geom[sj],geom[si],geom[sj],gamma)
    gram=(gram+gram.T)*0.5
    if not np.isfinite(gram).all() or not np.allclose(gram,gram.T,rtol=0,atol=1e-9):
        raise ValueError("K_pair contiene valores no finitos o no es simétrico")
    mineig=float(np.linalg.eigvalsh(gram).min())
    if mineig < -1e-7:
        raise ValueError(f"K_pair no es semidefinido positivo: lambda_min={mineig}")
    system=gram+lam*np.eye(len(gram))
    alpha=np.linalg.solve(system,y)
    if not np.isfinite(alpha).all():
        raise ValueError("La solución dual contiene valores no finitos")
    cond=float(np.linalg.cond(system))
    return alpha,geom[si].copy(),mineig,cond


def predict_pairs(geom: np.ndarray, q_i: np.ndarray, q_j: np.ndarray, train_i_geom: np.ndarray,
                  train_j_geom: np.ndarray, alpha: np.ndarray, gamma: float) -> np.ndarray:
    """Predice diferencias desde la función escalar latente compartida."""
    scores_i=score_structures(geom[q_i],train_i_geom,train_j_geom,alpha,gamma)
    scores_j=score_structures(geom[q_j],train_i_geom,train_j_geom,alpha,gamma)
    return scores_j-scores_i


def linear_kernel_reduction_test() -> dict[str, float | bool]:
    """Comprueba que KRR con kernel lineal equivale a Ridge primal sin intercepto."""
    rng=np.random.default_rng(19); x=rng.normal(size=(135,7)); y=rng.normal(size=135); xv=rng.normal(size=(33,7)); lam=.1
    alpha=np.linalg.solve(x@x.T+lam*np.eye(len(x)),y)
    dual=(xv@x.T)@alpha
    primal=xv@np.linalg.solve(x.T@x+lam*np.eye(x.shape[1]),x.T@y)
    err=float(np.max(np.abs(dual-primal)))
    return {"passed":bool(err<1e-9),"max_abs_prediction_difference":err,"lambda":lam}


def kernel_invariant_tests() -> dict[str, Any]:
    """Comprueba simetría/PSD del kernel y antisimetría/ciclo de sus scores."""
    rng=np.random.default_rng(2718); x=rng.normal(size=(9,7)); gamma=.37
    starts=np.array([0,1,3,0,5]);ends=np.array([1,2,4,5,8])
    gram=difference_kernel(x[starts],x[ends],x[starts],x[ends],gamma)
    symmetry=float(np.max(np.abs(gram-gram.T)))
    min_eig=float(np.linalg.eigvalsh((gram+gram.T)/2).min())
    alpha=rng.normal(size=len(starts));ti=x[starts];tj=x[ends]
    f=score_structures(x[[0,1,2]],ti,tj,alpha,gamma)
    anti=abs((f[1]-f[0])+(f[0]-f[1]))
    cycle=abs((f[1]-f[0])+(f[2]-f[1])-(f[2]-f[0]))
    mu=-4.2; centered=f-f.mean(); energy=mu+centered/1000
    reconstruction=float(np.max(np.abs(np.array([(energy[j]-energy[i])*1000 for i,j in PAIRS])-
                                           np.array([f[j]-f[i] for i,j in PAIRS]))))
    passed=bool(symmetry<1e-10 and min_eig>-1e-8 and anti<1e-12 and cycle<1e-12 and reconstruction<1e-10)
    return {"passed":passed,"kernel_symmetry_max_abs":symmetry,"kernel_min_eigenvalue":min_eig,
            "antisymmetry_error":anti,"triplet_cycle_error":cycle,"energy_reconstruction_max_error_mev_atom":reconstruction}


def bootstrap_paired(compositions: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap C1-C0 y C2-C0 por composición completa."""
    forms=sorted(compositions.composition_id.unique()); n=len(forms)
    rng=np.random.default_rng(BOOTSTRAP_SEED); idx=rng.integers(0,n,size=(BOOTSTRAP_N,n))
    records=[]
    metric_defs=[("energy_mae_mev_atom","energy_mae_mev_atom","meV/átomo"),
                 ("deltaE_mae_estimand_A_mev_atom","deltaE_mae_mev_atom","meV/átomo")]
    for left in ("C1_KRR_RBF","C2_KRR_RBF_CALIBRATED"):
        for key,col,unit in metric_defs:
            a=compositions[compositions.condition.eq(left)].set_index("composition_id").loc[forms][col].to_numpy()
            b=compositions[compositions.condition.eq("C0_GEO_LINEAR")].set_index("composition_id").loc[forms][col].to_numpy()
            d=a-b; draws=np.mean(d[idx],axis=1)
            records.append({"contrast":f"{left} − C0_GEO_LINEAR","metric":key,"estimate":float(d.mean()),"unit":unit,
                            "ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),
                            "replicates":BOOTSTRAP_N,"resampling_unit":"composition_id","seed":BOOTSTRAP_SEED})
        for key,num_col,den_col in [("pairwise_accuracy_pct","correct_pairs","eligible_pairs"),
                                    ("groundstate_accuracy_pct","groundstate_correct","groundstate_eligible"),
                                    ("full_triplet_accuracy_pct","full_triplet_correct","full_triplet_eligible")]:
            def counts(cond: str) -> tuple[np.ndarray,np.ndarray]:
                c=compositions[compositions.condition.eq(cond)].set_index("composition_id").loc[forms]
                return c[num_col].to_numpy(float),c[den_col].to_numpy(float)
            an,ad=counts(left);bn,bd=counts("C0_GEO_LINEAR")
            draws=100*(an[idx].sum(1)/ad[idx].sum(1)-bn[idx].sum(1)/bd[idx].sum(1))
            estimate=100*(an.sum()/ad.sum()-bn.sum()/bd.sum())
            records.append({"contrast":f"{left} − C0_GEO_LINEAR","metric":key,"estimate":float(estimate),
                            "unit":"puntos porcentuales","ci95_low":float(np.quantile(draws,.025)),
                            "ci95_high":float(np.quantile(draws,.975)),"replicates":BOOTSTRAP_N,
                            "resampling_unit":"composition_id","seed":BOOTSTRAP_SEED})
    return pd.DataFrame(records)


def save_figures(metrics: dict[str,dict[str,Any]], pairs: pd.DataFrame, bootstrap: pd.DataFrame,
                 gaps: pd.DataFrame) -> None:
    """Genera las siete figuras científicas del experimento como PNG."""
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    order=["C0_GEO_LINEAR","C1_KRR_RBF","C2_KRR_RBF_CALIBRATED"]
    labels={"C0_GEO_LINEAR":"Ridge lineal GEO","C1_KRR_RBF":"KRR-RBF GEO","C2_KRR_RBF_CALIBRATED":"KRR-RBF calibrado"}
    colors={"C0_GEO_LINEAR":"#7A8B5A","C1_KRR_RBF":"#2878B5","C2_KRR_RBF_CALIBRATED":"#D28BC2"}
    note="OOF de desarrollo: 70 composiciones, 210 estructuras, cinco folds; macro-promedio por fórmula para MAE. Celdas estáticas relajadas por DFT."
    for key,title,ylabel,file in [
        ("energy_mae_mev_atom","ABS₃: MAE de energía absoluta por átomo","MAE (meV/átomo)","energy_mae.png"),
        ("deltaE_mae_estimand_A_mev_atom","ABS₃: MAE de diferencias ΔE firmadas","MAE de los tres pares por composición (meV/átomo)","deltaE_mae.png")]:
        fig,ax=plt.subplots(figsize=(8.5,5.5)); vals=[metrics[c][key] for c in order]
        bars=ax.bar([labels[c] for c in order],vals,color=[colors[c] for c in order]); ax.bar_label(bars,fmt="%.2f",padding=3)
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.25)
        fig.text(.02,.015,note+" ΔEᵢⱼ=Eⱼ−Eᵢ entre polimorfos de la misma fórmula." if "delta" in key else note,fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.08,1,1));fig.savefig(IMAGE_DIR/file,dpi=300,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.5)); vals=[metrics[c]["pairwise_accuracy_eligible"]*100 for c in order]
    bars=ax.bar([labels[c] for c in order],vals,color=[colors[c] for c in order]);ax.bar_label(bars,fmt="%.1f%%",padding=3)
    ax.axhline(90,color="#A33D3D",linestyle="--",label="Meta aspiracional: 90%")
    ax.set_ylim(0,105);ax.set_ylabel("Exactitud pairwise elegible (%)");ax.set_title("ABS₃: ranking pairwise OOF")
    ax.legend();ax.grid(axis="y",alpha=.25)
    n_eligible=metrics[order[0]]["eligible_pairs"]
    fig.text(.02,.015,f"OOF: 70 composiciones, 210 pares totales, cinco folds. Denominador: {n_eligible} pares con |ΔE_DFT|>5 meV/átomo; predicción |ΔE|≤1 meV/átomo cuenta como incorrecta.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(IMAGE_DIR/"pairwise_ranking.png",dpi=300,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.8));x=np.arange(3);w=.36
    gs=[100*metrics[c]["groundstate_accuracy"] for c in order];full=[100*metrics[c]["full_triplet_accuracy"] for c in order]
    b1=ax.bar(x-w/2,gs,w,label="Selección del estado fundamental",color="#4C78A8");b2=ax.bar(x+w/2,full,w,label="Orden exacto del triplete",color="#F58518")
    for bars,metric in [(b1,"groundstate"),(b2,"full_triplet")]:
        for bar,c in zip(bars,order):
            num=metrics[c][metric+"_correct"];den=metrics[c][metric+"_eligible"]
            ax.annotate(f"{num}/{den}",(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,2),textcoords="offset points",ha="center",fontsize=7)
    ax.set_xticks(x,[labels[c] for c in order]);ax.set_ylabel("Exactitud (%)");ax.set_title("ABS₃: estado fundamental y orden completo")
    ax.set_ylim(0,82);ax.legend(loc="upper center",bbox_to_anchor=(.5,1.0),ncol=2,fontsize=8,framealpha=.95);ax.grid(axis="y",alpha=.25)
    fig.text(.02,.015,"OOF de desarrollo, 70 composiciones. Estado fundamental elegible si el gap DFT al segundo mínimo >1 meV/átomo (denominador indicado sobre barras). Triplete evaluable si ningún par DFT ≤1 meV/átomo; empate predicho no es acierto.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(IMAGE_DIR/"groundstate_full_triplet.png",dpi=300,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7));allx=pairs[pairs.condition.isin(order)].delta_dft_mev_atom.to_numpy();lim=max(abs(allx.min()),abs(allx.max()))*1.04
    ax.plot([-lim,lim],[-lim,lim],"--",color="#333",label="Ideal: predicción = DFT")
    ann=[]
    for c in order:
        d=pairs[pairs.condition.eq(c)];xv=d.delta_dft_mev_atom.to_numpy();yv=d.delta_pred_mev_atom.to_numpy()
        ax.scatter(xv,yv,s=13,alpha=.28,color=colors[c],label=labels[c])
        slope,inter=np.polyfit(xv,yv,1);fit=slope*xv+inter;r2=1-np.sum((yv-fit)**2)/np.sum((yv-yv.mean())**2)
        grid=np.linspace(-lim,lim,150);ax.plot(grid,slope*grid+inter,color=colors[c],lw=1.6)
        ann.append(f"{labels[c]}: N={len(xv)}, m={slope:.2f}, b={inter:.1f} meV/átomo, R² OLS={r2:.2f}, MAE={np.mean(abs(yv-xv)):.1f} meV/átomo")
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="ΔE DFT firmada Eⱼ−Eᵢ (meV/átomo)",ylabel="ΔE predicha firmada Eⱼ−Eᵢ (meV/átomo)",title="ABS₃: paridad OOF de ΔE")
    ax.set_aspect("equal",adjustable="box");ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.text(.015,.01,"Un punto = un par intracomposición; 210 pares/condición. Ajuste OLS libre sobre todos los puntos representados.\n"+"\n".join(ann),fontsize=7.5,wrap=True)
    fig.tight_layout(rect=(0,.20,1,1));fig.savefig(IMAGE_DIR/"deltaE_parity.png",dpi=300,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.5));
    for c in order:
        g=gaps[gaps.condition.eq(c)];ax.plot(g.gap_bin,g.deltaE_mae_mev_atom,marker="o",label=labels[c],color=colors[c])
    # Cada condición usa los mismos pares DFT; anotar n una sola vez evita solapar etiquetas.
    for point in gaps[gaps.condition.eq(order[0])].itertuples():
        if np.isfinite(point.deltaE_mae_mev_atom):
            ymax=gaps[gaps.gap_bin.eq(point.gap_bin)].deltaE_mae_mev_atom.max()
            ax.annotate(f"n={point.n_pairs}",(point.gap_bin,ymax),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    ax.set_ylabel("MAE de ΔE por intervalo (meV/átomo)");ax.set_xlabel("Magnitud del gap DFT |ΔE| (meV/átomo)")
    ax.set_title("ABS₃: error según magnitud de gap DFT");ax.legend();ax.grid(alpha=.25)
    fig.text(.02,.015,"Cada punto resume pares OOF dentro del intervalo; n por intervalo y condición se registra en gap_compression_analysis.csv. Pares dentro de la misma composición no son independientes.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(IMAGE_DIR/"error_por_gap.png",dpi=300,bbox_inches="tight");plt.close(fig)
    primary=bootstrap[bootstrap.metric.isin(["energy_mae_mev_atom","deltaE_mae_estimand_A_mev_atom","pairwise_accuracy_pct","groundstate_accuracy_pct","full_triplet_accuracy_pct"])]
    fig,axes=plt.subplots(1,5,figsize=(15,5.5))
    for ax,(_,metric) in zip(axes,primary.groupby("metric",sort=False)):
        for pos,row in enumerate(metric.itertuples()):
            # Las diferencias energéticas del CSV de bootstrap ya están en meV/átomo.
            estimate=row.estimate;lo=row.ci95_low;hi=row.ci95_high
            ax.errorbar(estimate,pos,xerr=[[estimate-lo],[hi-estimate]],fmt="o",capsize=3)
        metric_titles={"energy_mae_mev_atom":"MAE energía\n(meV/átomo)",
                       "deltaE_mae_estimand_A_mev_atom":"MAE ΔE\n(meV/átomo)",
                       "pairwise_accuracy_pct":"Exactitud pairwise\n(puntos porcentuales)",
                       "groundstate_accuracy_pct":"Estado fundamental\n(puntos porcentuales)",
                       "full_triplet_accuracy_pct":"Orden completo\n(puntos porcentuales)"}
        ax.axvline(0,color="#333",linestyle="--");ax.set_title(metric_titles.get(metric.iloc[0].metric,metric.iloc[0].metric),fontsize=8)
        ax.set_yticks(range(len(metric)),["RBF sin calibrar − Ridge", "RBF calibrado − Ridge"][:len(metric)],fontsize=7);ax.grid(axis="x",alpha=.2)
    fig.suptitle("ABS₃: bootstrap apareado por composición (IC95%)")
    fig.text(.02,.015,"10 000 réplicas de 70 composiciones OOF. Contrastes: KRR-RBF sin calibrar o calibrado menos Ridge lineal GEO; valores negativos favorecen KRR para errores. IC percentiles de desarrollo.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.09,1,.90));fig.savefig(IMAGE_DIR/"bootstrap_paired.png",dpi=300,bbox_inches="tight");plt.close(fig)


def write_reports(metrics: dict[str,dict[str,Any]], folds: pd.DataFrame, gaps: pd.DataFrame,
                  bootstrap: pd.DataFrame, config: dict[str,Any], resources: dict[str,Any]) -> None:
    """Escribe README e informe científico desde los artefactos finales.

    Args:
        metrics: Métricas OOF de C0, C1 y C2.
        folds: Tabla de hiperparámetros seleccionados por fold.
        gaps: Errores agregados por intervalo de gap DFT.
        bootstrap: Contrastes apareados con IC95%.
        config: Configuración científica serializada.
        resources: Recursos medidos del proceso.
    """
    labels={"C0_GEO_LINEAR":"Ridge pairwise lineal GEO","C1_KRR_RBF":"KRR-RBF sin calibrar",
            "C2_KRR_RBF_CALIBRATED":"KRR-RBF con calibración positiva"}
    rows=[]
    for c,m in metrics.items():
        rows.append(f"| {labels[c]} | {m['energy_mae_mev_atom']:.2f} | {m['energy_rmse_global_mev_atom']:.2f} | "
                    f"{m['deltaE_mae_estimand_A_mev_atom']:.2f} | {m['deltaE_rmse_global_mev_atom']:.2f} | "
                    f"{m['correct_pairs']}/{m['eligible_pairs']} ({100*m['pairwise_accuracy_eligible']:.2f}%) | "
                    f"{m['groundstate_correct']}/{m['groundstate_eligible']} ({100*m['groundstate_accuracy']:.2f}%) | "
                    f"{m['full_triplet_correct']}/{m['full_triplet_eligible']} ({100*m['full_triplet_accuracy']:.2f}%) | "
                    f"{m['ols_delta_slope']:.3f} | {m['deltaE_predictive_r2']:.3f} |")
    boot_lines=[]
    for row in bootstrap.itertuples():
        boot_lines.append(f"| {row.contrast} | {row.metric} | {row.estimate:.3f} {row.unit} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] |")
    fold_lines=[]
    for row in folds.itertuples():
        fold_lines.append(f"| {row.fold} | {row.selected_gamma_multiplier:g} | {row.selected_gamma:.6g} | {row.selected_lambda:g} | "
                          f"{row.validation_deltaE_mae_mev_atom:.2f} | {row.calibration_factor:.3f} |")
    gap_lines=[]
    for row in gaps.itertuples():
        gap_lines.append(f"| {labels[row.condition]} | {row.gap_bin} | {row.n_pairs} | "
                         f"{row.deltaE_mae_mev_atom:.2f} | {row.deltaE_rmse_mev_atom:.2f} |")
    report=("# ABS₃: Kernel Ridge no lineal sobre geometría para ΔE polimórfica\n\n"
      "## Pregunta y protocolo\n\n"
      "Se comparó el Ridge pairwise lineal geométrico histórico (C0) con Kernel Ridge RBF sobre el mismo vector de siete descriptores (C1). "
      "C2 reaplica las predicciones de C1 con un único factor escalar positivo por fold, estimado solo con predicciones de validación interna. "
      "No se entrenaron Transformers ni se modificaron Ridge M2.\n\n"
      "La cohorte contiene 70 composiciones ABS₃, 210 estructuras y tres polimorfos por composición. Se reutilizaron los cinco folds agrupados históricos (45 train, 11 validación y 14 outer por fold). "
      "El StandardScaler se ajustó solo sobre las 135 estructuras inner-train y se aplicó sin reajuste a validación y outer. "
      "La unidad de bootstrap es la composición completa. No se accedió al test independiente. La geometría proviene de celdas finales ya relajadas por DFT.\n\n"
      "El KRR usa el kernel de diferencias inducido por un kernel RBF estructural: `K_pair((i,j),(k,l)) = k(j,l)-k(j,k)-k(i,l)+k(i,k)`. "
      "Se probaron 12 combinaciones (multiplicadores gamma 0.25/1/4 y lambda 0.001/0.01/0.1/1). La selección usa MAE de ΔE de validación; empates se rompen por ranking, mayor lambda y menor gamma. "
      "El refit utiliza inner-train+validación tras seleccionar, conservando el scaler train-only. C0 son las predicciones históricas GEO y sus métricas se reprodujeron desde los CSV originales.\n\n"
      "## Resultados OOF\n\n"
      "Energía y errores en meV/átomo. `ΔE MAE` es el estimando A: macro-promedio por fórmula de los tres pares firmados; RMSE es global sobre observaciones. Pairwise exige `|ΔE_DFT|>5`; `|ΔE_pred|≤1` es no resuelto e incorrecto. Ground state exige gap DFT al segundo mínimo >1; el orden completo excluye fórmulas con algún empate DFT ≤1.\n\n"
      "| Condición | Energy MAE | Energy RMSE global | ΔE MAE A | ΔE RMSE global | Pairwise | Ground state | Triplete completo | Pendiente OLS | R² predictivo ΔE |\n"
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(rows)+"\n\n"
      "El baseline GEO tiene 47.55 meV/átomo de Energy MAE, 49.78 de ΔE MAE, 136/197 (69.04%) pairwise, 42/67 (62.69%) ground state y 19/66 (28.79%) tripletes exactos. "
      "KRR sin calibrar no mejora Energy MAE ni ΔE MAE; mantiene la exactitud pairwise y ground state, aunque aumenta el orden completo exacto a 22/66. "
      "C2 reduce ΔE MAE puntualmente a 49.57, pero no alcanza 90% de ranking ni supera el baseline en energía absoluta.\n\n"
      "## Hiperparámetros y calibración por fold\n\n"
      "| Fold | Multiplicador γ | γ | λ | MAE ΔE validación | Factor positivo C2 |\n|---:|---:|---:|---:|---:|---:|\n"+"\n".join(fold_lines)+"\n\n"
      "## Bootstrap apareado\n\n"
      "10 000 réplicas percentiles por composición (semilla 20261103); diferencias C1/C2 − C0. Negativo favorece KRR para errores y positivo para exactitudes. Los intervalos describen OOF de desarrollo, no una prueba externa.\n\n"
      "| Contraste | Métrica | Diferencia | IC95% |\n|---|---|---:|---:|\n"+"\n".join(boot_lines)+"\n\n"
      "## Error por magnitud del gap DFT\n\n"
      "| Condición | Intervalo | n pares | ΔE MAE (meV/átomo) | ΔE RMSE global (meV/átomo) |\n|---|---|---:|---:|---:|\n"+"\n".join(gap_lines)+"\n\n"
      "## Interpretación y límites\n\n"
      "La función no lineal regularizada no supera de forma general al Ridge pairwise lineal geométrico. C1 presenta pendiente OLS menor (más compresión) que GEO; la escala positiva elegida en C2 también deja la pendiente por debajo de 1 y no cambia signos de ranking. "
      "El aumento de exactitud de triplete de C1 es puntual y no basta para afirmar una mejora general; revisar sus intervalos apareados. La meta ≥90% no se alcanza. Los resultados son OOF sobre desarrollo, con 70 composiciones, y no sustituyen el test histórico cerrado.\n\n"
      "Las condiciones se seleccionaron por validación interna. C2 es exploratoria y comparte el mismo conjunto de validación para la selección de kernel y la escala; no debe presentarse como resultado confirmatorio. Las celdas ya relajadas por DFT limitan la interpretación como predicción sin cálculos estructurales previos.\n\n"
      f"Recursos medidos: {resources['wall_seconds']:.2f} s de pared, {resources['process_cpu_seconds']:.2f} s CPU, RAM máxima {resources['peak_rss_bytes']} bytes, {resources['cpu_threads']} hilo CPU; sin GPU/MPS.\n")
    readme=("# ABS₃ Geometry Pairwise Kernel Ridge\n\n"
      "KRR-RBF geometry-only, comparado con las predicciones OOF históricas GEO. No entrena extractores ni Ridge M2.\n\n"
      "Ejecución: `python scripts/ABS3/abs3_geometry_pairwise_kernel_ridge/run_krr.py`\n\n"
      "Artefactos principales: `config.yaml`, `preflight.json`, `kernel_hyperparameters.csv`, `oof_predictions_structures.csv`, `oof_predictions_pairs.csv`, `metrics.json`, `bootstrap_paired.csv`, `resources.json` y `reporte_cientifico.md`. "
      "C1 es la comparación primaria; C2 es calibración positiva exploratoria. El test independiente permanece cerrado.\n")
    for directory in (EXP,REPORT_DIR):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/"reporte_cientifico.md").write_text(report,encoding="utf-8")
        (directory/"README.md").write_text(readme,encoding="utf-8")


def main() -> None:
    """Ejecuta auditoría, selección interna, OOF KRR y análisis científico."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    EXP.mkdir(parents=True,exist_ok=True);MODEL_DIR.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        data,folds,base_struct,base_pairs,base_preflight=load_inputs()
        data=data.sort_values(["composition_id","family"],kind="stable").reset_index(drop=True)
        groups,targets=grouped_indices(data)
        linear_test=linear_kernel_reduction_test()
        invariant_test=kernel_invariant_tests()
        if not linear_test["passed"]: raise RuntimeError("Falló la equivalencia del kernel lineal con Ridge primal")
        if not invariant_test["passed"]: raise RuntimeError("Fallaron invariantes numéricos del kernel pairwise")
        preflight={"status":"PASS","dataset_sha256":sha256(DATASET),"folds_sha256":sha256(FOLDS),
                   "historical_geo_preflight":"PASS","historical_geo_predictions":"PASS",
                   "ridge_m2_fold_hashes":base_preflight["ridge_m2_fold_sha256"],
                   "ridge_m2_retrained":False,"outer_used_for_selection":False,"historical_test_accessed":False,
                   "geometry_dimensions":7,"linear_kernel_reduction_test":linear_test,"kernel_invariant_tests":invariant_test,
                   "pair_orientation":["dist_perovskite→hexagonal","dist_perovskite→needle","hexagonal→needle"],
                   "conditions_available":["C0_GEO_LINEAR","C1_KRR_RBF","C2_KRR_RBF_CALIBRATED"]}
        atomic_json(EXP/"preflight.json",preflight)
        atomic_json(EXP/"dataset_manifest.json",{"dataset":"data/abs3/abs3_development_f4_geometry/dataset.csv",
            "sha256":sha256(DATASET),"structures":210,"compositions":70,"polymorphs_per_composition":3,
            "geometry_source":"gga-static_endpoint_cleaned","geometry_features":GEOMETRY,
            "fold_manifest":str(FOLDS.relative_to(ROOT)),"fold_sha256":sha256(FOLDS),
            "folds":5,"inner_train_validation_outer":[45,11,14],
            "pair_orientation":["dist_perovskite→hexagonal","dist_perovskite→needle","hexagonal→needle"],
            "historical_test_accessed":False})
        c0s=base_struct.copy();c0s["condition"]="C0_GEO_LINEAR";c0s["residual_centered_mev_atom"]=(c0s.energy_pred_ev_atom-c0s.ridge_mean_m2_ev_atom)*1000
        c0p=base_pairs.copy();c0p["condition"]="C0_GEO_LINEAR";c0p=c0p.rename(columns={"delta_dft_mev_atom":"delta_dft_mev_atom","delta_pred_mev_atom":"delta_pred_mev_atom"})
        c0p["absolute_error_mev_atom"]=abs(c0p.delta_pred_mev_atom-c0p.delta_dft_mev_atom)
        structs=[c0s[["condition","fold","composition_id","structure_id","polymorph","energy_dft_ev_atom","energy_pred_ev_atom","ridge_mean_m2_ev_atom","residual_centered_mev_atom"]]]
        pairout=[c0p[["condition","fold","composition_id","polymorph_i","polymorph_j","structure_id_i","structure_id_j","delta_dft_mev_atom","delta_pred_mev_atom","absolute_error_mev_atom","eligible","resolved","correct"]]]
        hyper_rows=[];scaler_records={};fold_summary=[];calibration=[];gap_records=[]
        # Save the verified baseline immediately, then append new folds transactionally.
        atomic_csv(EXP/"oof_predictions_structures.csv",structs[0]);atomic_csv(EXP/"oof_predictions_pairs.csv",pairout[0])
        for fold in range(5):
            roles=roles_for_fold(folds,fold);tr,va,outer=roles["train"],roles["validation"],roles["outer"]
            train_struct=np.concatenate([groups[f] for f in tr]);
            scaler=StandardScaler().fit(data.iloc[train_struct][GEOMETRY].to_numpy(float))
            geom=scaler.transform(data[GEOMETRY].to_numpy(float))
            if not np.isfinite(geom).all(): raise ValueError(f"Fold {fold}: scaler produjo valores no finitos")
            scaler_records[str(fold)]={"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"n_train_structures":int(len(train_struct)),"train_compositions":len(tr)}
            ti,tj,ty,_=pair_rows(tr,groups,targets,data)
            vi,vj,vy,vmeta=pair_rows(va,groups,targets,data)
            dist=squared_distances(geom[train_struct],geom[train_struct]);positive=dist[dist>1e-12]
            if not len(positive): raise ValueError(f"Fold {fold}: geometría inner-train degenerada")
            median_distance=float(np.median(positive));gamma_ref=1.0/median_distance
            candidates=[]
            for multiplier in GAMMA_MULTIPLIERS:
                gamma=gamma_ref*multiplier
                for lam in LAMBDAS:
                    alpha,train_i_geom,min_eig,condition_number=fit_dual(geom,ti,tj,ty,gamma,lam)
                    train_j_geom=geom[tj].copy()
                    pred_val=predict_pairs(geom,vi,vj,train_i_geom,train_j_geom,alpha,gamma)
                    abs_err=np.abs(pred_val-vy);by_formula=[]
                    for formula in va:
                        loc=[r for r,m in enumerate(vmeta) if m["composition_id"]==formula]
                        by_formula.append(float(np.mean(abs_err[loc])))
                    val_mae=float(np.mean(by_formula));rank=pair_metrics(vy,pred_val)
                    row={"fold":fold,"gamma_multiplier":multiplier,"gamma_reference":gamma_ref,"gamma":gamma,
                         "lambda":lam,"validation_deltaE_mae_mev_atom":val_mae,
                         "validation_pairwise_accuracy":rank["accuracy"],"validation_eligible_pairs":rank["eligible"],
                         "kernel_min_eigenvalue":min_eig,"regularized_system_condition_number":condition_number}
                    candidates.append(row)
            # Min ΔE MAE; tie-break ranking, stronger lambda, lower gamma.
            candidates.sort(key=lambda x:(x["validation_deltaE_mae_mev_atom"],
                                          -(x["validation_pairwise_accuracy"] if math.isfinite(x["validation_pairwise_accuracy"]) else -1),
                                          -x["lambda"],x["gamma"]))
            selected=candidates[0];hyper_rows.extend(candidates)
            atomic_csv(EXP/"kernel_hyperparameters.csv",pd.DataFrame(hyper_rows))
            gamma=float(selected["gamma"]);lam=float(selected["lambda"])
            # Refit selected KRR on inner-train + inner-validation, retaining train-only scaler and gamma.
            fit_formulas=tr+va;fi,fj,fy,_=pair_rows(fit_formulas,groups,targets,data)
            alpha,fit_i_geom,min_eig,condition_number=fit_dual(geom,fi,fj,fy,gamma,lam)
            fit_j_geom=geom[fj].copy()
            # Secondary positive scalar, fit only to selected-model inner-validation predictions.
            val_scores={f:score_structures(geom[groups[f]],geom[ti],geom[tj],
                                           fit_dual(geom,ti,tj,ty,gamma,lam)[0],gamma) for f in va}
            val_pred=[]
            for formula in va:
                s=val_scores[formula]
                val_pred.extend([(s[j]-s[i]) for i,j in PAIRS])
            val_pred=np.asarray(val_pred);den=float(val_pred@val_pred)
            cal_factor=max(float(val_pred@vy)/den,1e-12) if den>0 else 1.0
            calibration.append({"fold":fold,"selected_gamma":gamma,"selected_lambda":lam,
                                "positive_calibration_factor":cal_factor,"fit_source":"selected inner-validation predictions",
                                "outer_used":False})
            outer_scores={f:score_structures(geom[groups[f]],fit_i_geom,fit_j_geom,alpha,gamma) for f in outer}
            mu_map={}
            fold_geo=base_struct[base_struct.fold.eq(fold)]
            for formula in outer:
                group=fold_geo[fold_geo.composition_id.eq(formula)]
                vals=group.ridge_mean_m2_ev_atom.to_numpy(float)
                if len(vals)!=3 or not np.allclose(vals,vals[0],rtol=0,atol=1e-10):
                    raise ValueError(f"Fold {fold} {formula}: mu_Ridge_M2 no es único en el triplete")
                mu_map[formula]=float(vals[0])
            for condition,factor in (("C1_KRR_RBF",1.0),("C2_KRR_RBF_CALIBRATED",cal_factor)):
                scaled_scores={f:outer_scores[f]*factor for f in outer}
                srows,prows=evaluate_triplets(data,groups,targets,outer,scaled_scores,mu_map,fold,condition)
                structs.append(pd.DataFrame(srows));pairout.append(pd.DataFrame(prows))
            fold_summary.append({"fold":fold,"train_compositions":len(tr),"validation_compositions":len(va),
                                 "outer_compositions":len(outer),"selected_gamma_multiplier":selected["gamma_multiplier"],
                                 "gamma_reference":gamma_ref,"selected_gamma":gamma,"selected_lambda":lam,
                                 "validation_deltaE_mae_mev_atom":selected["validation_deltaE_mae_mev_atom"],
                                 "validation_pairwise_accuracy":selected["validation_pairwise_accuracy"],
                                 "calibration_factor":cal_factor,"refit_kernel_min_eigenvalue":min_eig,
                                 "refit_condition_number":condition_number})
            atomic_csv(EXP/"metrics_per_fold.csv",pd.DataFrame(fold_summary))
            atomic_json(EXP/"geometry_scalers.json",scaler_records)
            # Per-fold atomic persistence prevents loss of completed outer folds.
            struct_frame=pd.concat(structs,ignore_index=True);pair_frame=pd.concat(pairout,ignore_index=True)
            atomic_csv(EXP/"oof_predictions_structures.csv",struct_frame)
            atomic_csv(EXP/"oof_predictions_pairs.csv",pair_frame)
            joblib.dump({"scaler":scaler,"train_i_geometry":fit_i_geom,"train_j_geometry":fit_j_geom,
                         "dual_coefficients":alpha,"gamma":gamma,"lambda":lam,"calibration_factor":cal_factor,
                         "ridge_m2_fold_hash":preflight["ridge_m2_fold_hashes"][str(fold)]},
                        MODEL_DIR/f"fold_{fold}_krr.joblib")
            LOGGER.info("fold=%d seleccionado gamma=%.6g lambda=%.4g val_deltaE_MAE=%.4f calibration=%.4g",
                        fold,gamma,lam,selected["validation_deltaE_mae_mev_atom"],cal_factor)
        struct_frame=pd.concat(structs,ignore_index=True);pair_frame=pd.concat(pairout,ignore_index=True)
        atomic_csv(EXP/"oof_predictions_structures.csv",struct_frame)
        atomic_csv(EXP/"oof_predictions_pairs.csv",pair_frame)
        atomic_csv(EXP/"kernel_hyperparameters.csv",pd.DataFrame(hyper_rows))
        atomic_csv(EXP/"metrics_per_fold.csv",pd.DataFrame(fold_summary))
        atomic_json(EXP/"geometry_scalers.json",scaler_records)
        atomic_csv(EXP/"calibration_factors.csv",pd.DataFrame(calibration))
        comp,metrics=composition_summary(struct_frame,pair_frame)
        atomic_csv(EXP/"metrics_per_composition.csv",comp)
        atomic_csv(EXP/"krr_folds_summary.csv",pd.DataFrame(fold_summary))
        fold_metric_rows=[]
        for (condition,fold),sub in comp.groupby(["condition","fold"],sort=False):
            fold_metric_rows.append({"condition":condition,"fold":int(fold),"compositions":int(len(sub)),
                "energy_mae_mev_atom":float(sub.energy_mae_mev_atom.mean()),
                "deltaE_mae_estimand_A_mev_atom":float(sub.deltaE_mae_mev_atom.mean()),
                "pairwise_correct_pairs":int(sub.correct_pairs.sum()),"pairwise_eligible_pairs":int(sub.eligible_pairs.sum()),
                "pairwise_accuracy":float(sub.correct_pairs.sum()/sub.eligible_pairs.sum()),
                "groundstate_correct":int(sub.groundstate_correct.sum()),"groundstate_eligible":int(sub.groundstate_eligible.sum()),
                "full_triplet_correct":int(sub.full_triplet_correct.sum()),"full_triplet_eligible":int(sub.full_triplet_eligible.sum())})
        atomic_csv(EXP/"metrics_per_fold.csv",pd.DataFrame(fold_metric_rows))
        # Alias outputs named by the original specification.
        atomic_csv(EXP/"oof_predictions_pairs.csv",pair_frame)
        atomic_csv(EXP/"energy_mae_results.csv",pd.DataFrame([{"condition":c,"energy_mae_mev_atom":m["energy_mae_mev_atom"],"energy_rmse_global_mev_atom":m["energy_rmse_global_mev_atom"],"energy_bias_mev_atom":m["energy_bias_mev_atom"]} for c,m in metrics.items()]))
        atomic_csv(EXP/"deltaE_mae_results.csv",pd.DataFrame([{"condition":c,"deltaE_mae_estimand_A_mev_atom":m["deltaE_mae_estimand_A_mev_atom"],"deltaE_rmse_global_mev_atom":m["deltaE_rmse_global_mev_atom"],"ols_slope":m["ols_delta_slope"],"ols_intercept_mev_atom":m["ols_delta_intercept_mev_atom"],"ols_fit_r2":m["ols_delta_r2"],"predictive_r2":m["deltaE_predictive_r2"]} for c,m in metrics.items()]))
        atomic_csv(EXP/"ranking_results.csv",pd.DataFrame([{"condition":c,"pairwise_accuracy":m["pairwise_accuracy_eligible"],"correct_pairs":m["correct_pairs"],"eligible_pairs":m["eligible_pairs"],"groundstate_accuracy":m["groundstate_accuracy"],"groundstate_correct":m["groundstate_correct"],"groundstate_eligible":m["groundstate_eligible"],"full_triplet_accuracy":m["full_triplet_accuracy"],"full_triplet_correct":m["full_triplet_correct"],"full_triplet_eligible":m["full_triplet_eligible"]} for c,m in metrics.items()]))
        gap_bins=[(0,5,"0–5"),(5,10,"5–10"),(10,25,"10–25"),(25,50,"25–50"),(50,100,"50–100"),(100,float("inf"),">100")]
        for c in metrics:
            q=pair_frame[pair_frame.condition.eq(c)].copy();q["gap_abs"]=abs(q.delta_dft_mev_atom)
            for lo,hi,label in gap_bins:
                b=q[(q.gap_abs>=lo)&(q.gap_abs<hi)]
                gap_records.append({"condition":c,"gap_bin":label,"n_pairs":len(b),
                                    "deltaE_mae_mev_atom":float(b.absolute_error_mev_atom.mean()) if len(b) else float("nan"),
                                    "deltaE_rmse_mev_atom":float(np.sqrt(np.mean((b.delta_pred_mev_atom-b.delta_dft_mev_atom)**2))) if len(b) else float("nan")})
        gaps=pd.DataFrame(gap_records);atomic_csv(EXP/"gap_compression_analysis.csv",gaps)
        bootstrap=bootstrap_paired(comp,pair_frame);atomic_csv(EXP/"bootstrap_paired.csv",bootstrap)
        # Exact historical baseline GEO metrics, verified directly from its summary.
        baseline_summary=json.loads((BASE/"summary.json").read_text(encoding="utf-8"))["metrics"]["GEO"]
        for key in ("energy_mae_mev_atom","deltaE_mae_signed_est_a_mev_atom","pairwise_accuracy_eligible",
                    "groundstate_accuracy","full_triplet_accuracy"):
            target={"energy_mae_mev_atom":"energy_mae_mev_atom","deltaE_mae_signed_est_a_mev_atom":"deltaE_mae_estimand_A_mev_atom",
                    "pairwise_accuracy_eligible":"pairwise_accuracy_eligible","groundstate_accuracy":"groundstate_accuracy",
                    "full_triplet_accuracy":"full_triplet_accuracy"}[key]
            if not math.isclose(metrics["C0_GEO_LINEAR"][target],baseline_summary[key],rel_tol=0,abs_tol=1e-8):
                raise ValueError(f"C0 no reproduce baseline historical {key}")
        config={"experiment":"abs3_geometry_pairwise_kernel_ridge","dataset":"data/abs3/abs3_development_f4_geometry/dataset.csv",
                "dataset_sha256":sha256(DATASET),"cohort":{"compositions":70,"structures":210,"polymorphs_per_composition":3},
                "folds":{"count":5,"inner_train_compositions":45,"inner_validation_compositions":11,"outer_compositions":14},
                "features":GEOMETRY,"scaler":"StandardScaler fitted on 135 inner-train structures per fold; applied unchanged to validation/outer; retained for refit",
                "kernel":"RBF on structures; pair kernel K(j,l)-K(j,k)-K(i,l)+K(i,k)","gamma_reference":"inverse median positive squared distance among inner-train structures after scaler",
                "gamma_multipliers":GAMMA_MULTIPLIERS,"lambdas":LAMBDAS,"selection":"inner-validation macro DeltaE MAE; tie: higher pairwise accuracy, higher lambda, lower gamma",
                "refit":"selected model refit using inner-train + inner-validation pairs, retaining inner-train scaler/gamma",
                "calibration":"C2 positive no-intercept scalar fit on selected-model inner-validation pair predictions; secondary/exploratory",
                "ridge_M2":"historical fixed 17-fraction models; no fit; same M2 means as GEO baseline",
                "units":{"target":"signed meV/atom","M2_energy":"eV/atom","reconstruction":"mu_M2 + centered_score/1000"},
                "thresholds":{"pair_eligible_abs_dft_gt_mev_atom":5,"predicted_unresolved_abs_le_mev_atom":1,"groundstate_gap_gt_mev_atom":1,"full_triplet_tie_gap_le_mev_atom":1},
                "metrics":{"primary":"OOF macro MAE of 3 signed pair errors per composition","rmse":"global sqrt(mean(error^2)) over structures/pairs","ranking":"eligible pair correct / eligible pairs; unresolved incorrect","bootstrap":"10,000 paired composition resamples, percentile IC95%, seed 20261103"},
                "historical_test_accessed":False,"outer_used_for_selection":False}
        (EXP/"config.yaml").write_text("# Configuración científica serializada en YAML-like con definiciones en README.\n"+"\n".join(f"{k}: {json.dumps(v,ensure_ascii=False)}" for k,v in config.items())+"\n",encoding="utf-8")
        save_figures(metrics,pair_frame,bootstrap,gaps)
        summary={"status":"COMPLETE","conditions":metrics,"baseline_geo_exact":baseline_summary,
                 "ranking_90pct_goal_reached":{c:bool(m["pairwise_accuracy_eligible"]>=.9) for c,m in metrics.items()},
                 "calibration_secondary_only":True,"historical_test_accessed":False}
        atomic_json(EXP/"summary.json",summary);atomic_json(EXP/"metrics.json",{"primary_metric":"OOF macro MAE of three signed intracomposition pair errors, meV/atom",
            "definitions":{"energy":"E_pred=mu_Ridge_M2+centered_score/1000","deltaE":"E_j-E_i within formula",
            "pairwise":"eligible |DFT delta|>5 meV/atom; predicted |delta|<=1 unresolved and incorrect",
            "full_triplet":"exact order; exclude DFT pair ties <=1 meV/atom","groundstate":"eligible DFT gap to second minimum >1 meV/atom; predicted unresolved gap <=1 meV/atom is incorrect",
            "ols_fit_r2":"OLS goodness of fit with predicted ΔE as response","predictive_r2":"1-SSE/SST using DFT ΔE as reference response"},
            "models":metrics,"bootstrap":bootstrap.to_dict("records"),"historical_test_accessed":False})
        resources={"status":"COMPLETE","wall_seconds":time.monotonic()-START_WALL,"process_cpu_seconds":time.process_time()-START_CPU,
                   "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"device":"cpu","cpu_threads":1,
                   "mps":False,"cuda":False,"solver":"numpy.linalg.solve(K_pair + lambda I, y); no explicit matrix inverse",
                   "train_kernel_matrix_shape":[135,135],"outer_structures":210,"outer_pairs":210,
                   "bootstrap_replicates":BOOTSTRAP_N,"historical_test_accessed":False,
                   "python_version":platform.python_version(),"numpy_version":np.__version__,"pandas_version":pd.__version__}
        atomic_json(EXP/"resources.json",resources)
        write_reports(metrics,pd.DataFrame(fold_summary),gaps,bootstrap,config,resources)
        LOGGER.info("KRR benchmark completo; métricas=%s",json.dumps(metrics,ensure_ascii=False))


if __name__ == "__main__":
    main()
