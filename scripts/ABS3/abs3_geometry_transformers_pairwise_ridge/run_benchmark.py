"""ABS3 benchmark de geometría y embeddings congelados con Ridge pairwise.

Selecciona alpha solo con inner-validation, ajusta la transformación final con
inner-train más inner-validation y predice cada composición en su fold outer.
El Transformer nunca se carga para entrenamiento ni recibe gradientes.
"""
from __future__ import annotations

import argparse
import csv
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
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger("abs3.pairwise_ridge")
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/abs3_geometry_transformers_pairwise_ridge"
IMAGE_DIR = ROOT / "images/abs3_geometry_transformers_pairwise_ridge"
REPORT_DIR = ROOT / "reports/ABS3/abs3_geometry_transformers_pairwise_ridge"
DATASET = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
FOLDS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
EMBED_ROOT = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71/embeddings"
FROZEN_OOF = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71/predicciones_oof_por_estructura.csv"
M2_OOF = ROOT / "experiments/matgpt_f0_stoichiometric_ridge_m2_comparison/predicciones_oof_por_estructura.csv"
HIST_MATGPT_M2 = ROOT / "experiments/matgpt_f0_stoichiometric_ridge_m2_comparison/metrics.json"
HIST_MATTER_GEOM = ROOT / "experiments/mattergpt_abs3_geometry_late_fusion/metrics.json"
GEOMETRY = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
            "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
ELEMENTS = ("Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr")
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
CONDITIONS = ("GEO", "MAT", "MAT-G", "MATTER", "MATTER-G")
ACTIVE_CONDITIONS = CONDITIONS
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RIDGE_HASHES = {
    0: "99d5114f567450f46322cd39e021405bc4ba9fd396fadeb07ecb62f7a204f3d9",
    1: "48bc05a164936d4c2ef104baa60086a95388121eb95624003aad07388ddcfde1",
    2: "664fc440ce0a699fa5ac5e62f9167e3359682ada5bb63838323cb7a5a1efcf0a",
    3: "e26bae8b47c98b86731c283bcbd69732f6395dea7e66491bad82f08e53005e49",
    4: "680e1266483f838fbc962bed8b0d5458c3e7f1e059cf30a83dafdbf76c56408d",
}
RIDGE_GIT_TEMPLATE = "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_{}.joblib"
CHECKPOINTS = {
    "MAT": ("models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
            "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"),
    "MATTER": ("models/MatterGPT/pytorch_model.pt",
               "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"),
}
STARTED = time.monotonic()
PROCESS_STARTED = time.process_time()


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo por bloques.

    Args:
        path: Archivo de entrada.

    Returns:
        Hash hexadecimal del archivo.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON con reemplazo atómico.

    Args:
        path: Ruta de salida.
        payload: Contenido serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]] | pd.DataFrame) -> None:
    """Escribe CSV atómico, preservando resultados completos previos.

    Args:
        path: Ruta canónica.
        records: Filas o tabla de salida.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def csv_sha256(path: Path) -> str:
    """Hash de una ruta regular usada en manifiestos."""
    return sha256(path) if path.is_file() else "MISSING"


def load_dataset_and_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y valida cohortes, claves explícitas, geometría y particiones.

    Returns:
        Dataset ordenado por composición y familia, más asignaciones históricas.
    """
    data = pd.read_csv(DATASET, dtype={"structure_id": str, "composition_id": str}, low_memory=False)
    folds = pd.read_csv(FOLDS, dtype={"structure_id": str, "composition_group": str})
    folds = folds[folds.historical_split.isin(("train", "val"))].copy()
    if len(data) != 210 or data.structure_id.nunique() != 210 or data.composition_id.nunique() != 70:
        raise ValueError("Dataset no contiene exactamente 210 estructuras y 70 composiciones")
    if data.groupby("composition_id").size().ne(3).any() or data.structure_id.duplicated().any():
        raise ValueError("Hay estructuras duplicadas o tripletes incompletos")
    if not data.SLICES.astype(str).equals(data.slices_original.astype(str)):
        raise ValueError("SLICES difiere de la representación original")
    if data[GEOMETRY].isna().any().any() or not np.isfinite(data[GEOMETRY].to_numpy(float)).all():
        raise ValueError("Geometría con NaN o valores no finitos")
    if not data.geometry_source.eq("gga-static_endpoint_cleaned").all():
        raise ValueError("La fuente geométrica no coincide con gga-static_endpoint_cleaned")
    if len(folds) != 1050 or folds.structure_id.nunique() != 210:
        raise ValueError(f"Manifiesto de folds inesperado: {len(folds)} filas")
    if set(data.structure_id) != set(folds.structure_id):
        raise ValueError("Los structure_id del dataset y los folds no coinciden")
    if set(data.composition_id) != set(folds.composition_group):
        raise ValueError("composition_id no coincide con composition_group")
    for fold in range(5):
        role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
        counts = (int(role.inner_role.eq("inner_train").sum()),
                  int(role.inner_role.eq("inner_validation").sum()),
                  int(role.outer_role.eq("outer_validation").sum()))
        if counts != (45, 11, 14):
            raise ValueError(f"Fold {fold} no conserva tamaños 45/11/14: {counts}")
        if role.composition_group.duplicated().any():
            raise ValueError(f"Composición repetida en roles del fold {fold}")
    return data.sort_values(["composition_id", "family"], kind="stable").reset_index(drop=True), folds


def load_features(data: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Carga cachés 768d verificadas, conservando GEO como rama independiente.

    Args:
        data: Dataset canónico.

    Returns:
        Matrices por extractor, manifiesto de cachés y estado de checkpoints.
    """
    manifest_path = EMBED_ROOT / "embedding_manifest.json"
    if not manifest_path.is_file():
        return {}, {"status": "MISSING", "path": str(manifest_path)}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    checkpoint_state: dict[str, Any] = {}
    for name in ("MAT", "MATTER"):
        key = name.lower()
        record = manifest.get("matgpt" if name == "MAT" else "mattergpt", {})
        cache = EMBED_ROOT / ("matgpt_original_final_end.npz" if name == "MAT" else "mattergpt_original_final_end.npz")
        checkpoint_rel, expected = CHECKPOINTS[name]
        checkpoint_path = ROOT / checkpoint_rel
        actual_checkpoint = sha256(checkpoint_path) if checkpoint_path.is_file() else "MISSING"
        checkpoint_state[name] = {"path": checkpoint_rel, "expected_sha256": expected,
                                  "actual_sha256": actual_checkpoint,
                                  "status": "PASS" if actual_checkpoint == expected else "MISSING_OR_HASH_MISMATCH"}
        if not cache.is_file() or record.get("sha256") != sha256(cache) or record.get("checkpoint_sha256") != expected:
            details[name] = {"status": "CACHE_MISSING_OR_HASH_MISMATCH", "path": str(cache)}
            continue
        with np.load(cache, allow_pickle=False) as archive:
            ids = archive["structure_id"].astype(str)
            values = archive["embeddings"].astype(np.float64)
        if values.shape != (210, 768) or len(set(ids)) != 210 or not np.isfinite(values).all():
            details[name] = {"status": "INVALID_CACHE", "shape": list(values.shape)}
            continue
        lookup = {sid: i for i, sid in enumerate(ids)}
        if set(lookup) != set(data.structure_id.astype(str)):
            details[name] = {"status": "ID_MISMATCH"}
            continue
        features[name] = values[[lookup[sid] for sid in data.structure_id.astype(str)]]
        details[name] = {"status": "PASS", "path": str(cache.relative_to(ROOT)),
                         "sha256": sha256(cache), "shape": list(values.shape),
                         "checkpoint_sha256": expected,
                         "property_vector": [1.0, 2.0] if name == "MATTER" else None,
                         "feature_definition": "FINAL_END(768)" if name == "MAT" else "terminal ln_f contextualized(768)"}
    return features, {"status": "PASS" if len(features) == 2 else "PARTIAL",
                      "embedding_manifest_sha256": sha256(manifest_path), "extractors": details}, checkpoint_state


def ridge_m2_oof_means(data: pd.DataFrame) -> tuple[dict[tuple[int, str], float], dict[str, str]]:
    """Valida hashes de M2 y recupera medias composicionales outer históricas.

    Args:
        data: Cohorte ABS3.

    Returns:
        Mapa (fold, fórmula) a μ y hashes SHA de los cinco pipelines.
    """
    if not FROZEN_OOF.is_file() or not M2_OOF.is_file():
        raise FileNotFoundError("Faltan predicciones históricas con μ_Ridge M2 verificado")
    hashes: dict[str, str] = {}
    for fold, expected in RIDGE_HASHES.items():
        git_path = RIDGE_GIT_TEMPLATE.format(fold)
        result = subprocess.run(["git", "show", f"HEAD:{git_path}"], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        digest = hashlib.sha256(result.stdout).hexdigest()
        if digest != expected:
            raise ValueError(f"Ridge M2 fold {fold} hash incorrecto: {digest}")
        model = joblib.load(__import__("io").BytesIO(result.stdout))
        if int(getattr(model, "n_features_in_", -1)) != 17 or not hasattr(model, "predict"):
            raise ValueError(f"El Ridge M2 fold {fold} no es el pipeline de 17 fracciones")
        hashes[str(fold)] = digest
    frame = pd.read_csv(M2_OOF, dtype={"structure_id": str, "composition_id": str})
    frame = frame[frame.model.eq("RETRAINED_M2_C0") & frame.seed.eq(42)]
    required = {"fold", "composition_id", "structure_id", "ridge_mean_m2_ev_atom", "energy_dft_ev_atom"}
    if not required.issubset(frame.columns):
        raise ValueError("Predicciones de referencia no incluyen mu Ridge M2 y claves")
    if frame.structure_id.nunique() != 210 or set(frame.structure_id) != set(data.structure_id):
        raise ValueError("Referencia M2 no cubre exactamente 210 estructuras outer")
    means: dict[tuple[int, str], float] = {}
    for (fold, formula), group in frame.groupby(["fold", "composition_id"]):
        vals = group.ridge_mean_m2_ev_atom.to_numpy(float)
        if len(group) != 3 or not np.allclose(vals, vals[0], rtol=0, atol=1e-10):
            raise ValueError(f"μ_M2 no es único dentro del triplete {formula}/fold{fold}")
        means[(int(fold), str(formula))] = float(vals[0])
    if len(means) != 70:
        raise ValueError(f"Se esperaban 70 medias M2 OOF formula-fold, hay {len(means)}")
    frozen = pd.read_csv(FROZEN_OOF, dtype={"structure_id": str, "composition_id": str})
    frozen = frozen[frozen.model.isin(("MatGPT_F0", "MatterGPT_F0")) & frozen.seed.eq(42)]
    if "ridge_mean_ev_atom" not in frozen:
        raise ValueError("Predicciones frozen no contienen μ composicional")
    for row in frozen.itertuples():
        key = (int(row.fold), str(row.composition_id))
        if key in means and abs(float(row.ridge_mean_ev_atom) - means[key]) > 1e-10:
            raise ValueError(f"μ Ridge M2 difiere entre OOF M2 y extractores frozen para {key}")
    return means, hashes


def role_formulas(folds: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    """Devuelve composición asignada a cada rol del fold."""
    role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
    return {"train": sorted(role.loc[role.inner_role.eq("inner_train"), "composition_group"].astype(str)),
            "validation": sorted(role.loc[role.inner_role.eq("inner_validation"), "composition_group"].astype(str)),
            "outer": sorted(role.loc[role.outer_role.eq("outer_validation"), "composition_group"].astype(str))}


def make_triplets(data: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    """Construye estructuras de cada fórmula en el orden de familia definido.

    Args:
        data: Dataset ordenado.

    Returns:
        Índices por fórmula y targets DFT por átomo en eV.
    """
    groups: dict[str, list[int]] = {}
    targets: dict[str, np.ndarray] = {}
    for formula, group in data.groupby("composition_id", sort=False):
        group = group.set_index("family").loc[list(FAMILIES)]
        groups[str(formula)] = group.index.map({family: data.index[data.structure_id.eq(group.loc[family].structure_id)][0]
                                               for family in FAMILIES}).tolist()
        targets[str(formula)] = group.energy_dft_ev_atom.to_numpy(float)
    return groups, targets


def pair_features(formulas: list[str], features: np.ndarray, data: pd.DataFrame,
                  groups: dict[str, list[int]], targets: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Crea tres diferencias firmadas por fórmula, con orientación canónica.

    Args:
        formulas: Composiciones incluidas.
        features: Matriz por estructura ya estandarizada.
        data: Dataset unido por structure_id.
        groups: Índices por composición.
        targets: Energías DFT por átomo.

    Returns:
        Diferencias X, ΔE objetivo en eV/átomo y claves de los pares.
    """
    x_rows, y_rows, meta = [], [], []
    for formula in formulas:
        indices = groups[formula]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            x_rows.append(features[indices[j]] - features[indices[i]])
            y_rows.append(targets[formula][j] - targets[formula][i])
            meta.append({"composition_id": formula, "family_i": FAMILIES[i], "family_j": FAMILIES[j],
                         "structure_id_i": str(data.iloc[indices[i]].structure_id),
                         "structure_id_j": str(data.iloc[indices[j]].structure_id),
                         "delta_dft_ev_atom": targets[formula][j] - targets[formula][i]})
    return np.asarray(x_rows), np.asarray(y_rows), meta


def rank_stats(dft: np.ndarray, pred: np.ndarray) -> dict[str, float | int | bool]:
    """Métricas de pares, ranking completo y selección del ground state.

    Args:
        dft: Tres energías por átomo para un triplete.
        pred: Tres energías predichas por átomo.

    Returns:
        Conteos y banderas con los umbrales predefinidos.
    """
    pair_truth, pair_prediction = [], []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair_truth.append((dft[j] - dft[i]) * 1000)
        pair_prediction.append((pred[j] - pred[i]) * 1000)
    pair_truth_a, pair_pred_a = np.asarray(pair_truth), np.asarray(pair_prediction)
    eligible = np.abs(pair_truth_a) > 5.0
    resolved = np.abs(pair_pred_a) > 1.0
    correct = eligible & resolved & (np.sign(pair_truth_a) == np.sign(pair_pred_a))
    dft_sorted = np.sort(dft)
    no_dft_ties = bool(np.all(np.abs(pair_truth_a) > 1.0))
    dft_order = np.argsort(dft, kind="stable")
    pred_order = np.argsort(pred, kind="stable")
    pred_top_resolved = bool((np.sort(pred)[1] - np.sort(pred)[0]) * 1000 > 1.0)
    ground_eligible = bool((dft_sorted[1] - dft_sorted[0]) * 1000 > 1.0)
    return {"eligible_pair_count": int(eligible.sum()), "correct_pair_count": int(correct.sum()),
            "resolved_eligible_pair_count": int((eligible & resolved).sum()),
            "unresolved_eligible_pair_count": int((eligible & ~resolved).sum()),
            "full_triplet_eligible": no_dft_ties,
            "full_triplet_correct": bool(no_dft_ties and np.array_equal(dft_order, pred_order)
                                          and np.all(np.abs(pair_pred_a) > 1.0)),
            "groundstate_eligible": ground_eligible,
            "groundstate_correct": bool(ground_eligible and pred_top_resolved and int(np.argmin(dft)) == int(np.argmin(pred)))}


def compute_metrics(structure: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calcula métricas OOF globales, por fórmula, fold y polimorfo.

    Args:
        structure: Predicciones por estructura.
        pairs: Predicciones por par.

    Returns:
        Métricas de composición, fold, familia y resumen global.
    """
    compositions, folds, polymorphs, global_metrics = [], [], [], {}
    for condition in ACTIVE_CONDITIONS:
        frame = structure[structure.condition.eq(condition)]
        pair_frame = pairs[pairs.condition.eq(condition)]
        by_formula = []
        for (fold, formula), group in frame.groupby(["fold", "composition_id"]):
            group = group.set_index("polymorph").loc[list(FAMILIES)].reset_index()
            e_dft = group.energy_dft_ev_atom.to_numpy(float)
            e_pred = group.energy_pred_ev_atom.to_numpy(float)
            energy_error = e_pred - e_dft
            pgroup = pair_frame[(pair_frame.fold == fold) & (pair_frame.composition_id == formula)]
            p_err = pgroup.absolute_error_mev_atom.to_numpy(float)
            stat = rank_stats(e_dft, e_pred)
            comp = {"condition": condition, "fold": int(fold), "composition_id": str(formula),
                    "energy_mae_mev_atom": float(np.mean(np.abs(energy_error)) * 1000),
                    "energy_rmse_mev_atom": float(np.sqrt(np.mean(energy_error**2)) * 1000),
                    "energy_bias_mev_atom": float(np.mean(energy_error) * 1000),
                    "deltaE_mae_mev_atom": float(np.mean(p_err)),
                    "deltaE_rmse_within_formula_mev_atom": float(np.sqrt(np.mean(p_err**2))),
                    "eligible_pairs": stat["eligible_pair_count"], "correct_pairs": stat["correct_pair_count"],
                    "resolved_eligible_pairs": stat["resolved_eligible_pair_count"],
                    "unresolved_eligible_pairs": stat["unresolved_eligible_pair_count"],
                    "full_triplet_eligible": stat["full_triplet_eligible"],
                    "full_triplet_correct": stat["full_triplet_correct"],
                    "groundstate_eligible": stat["groundstate_eligible"],
                    "groundstate_correct": stat["groundstate_correct"],
                    "spearman": float(spearmanr(e_dft, e_pred).statistic),
                    "kendall": float(kendalltau(e_dft, e_pred).statistic)}
            by_formula.append(comp)
            compositions.append(comp)
            for _, row in group.iterrows():
                polymorphs.append({"condition": condition, "fold": int(fold), "composition_id": str(formula),
                    "family": row.polymorph, "absolute_energy_error_mev_atom": abs(float(row.energy_pred_ev_atom-row.energy_dft_ev_atom))*1000})
        comp_df = pd.DataFrame(by_formula)
        pf = pair_frame
        dft_delta = pf.delta_dft_mev_atom.to_numpy(float)
        pred_delta = pf.delta_pred_mev_atom.to_numpy(float)
        errors = pred_delta - dft_delta
        eligible = pf.eligible.to_numpy(bool)
        resolved = pf.resolved.to_numpy(bool)
        slopes = np.polyfit(dft_delta, pred_delta, 1) if len(dft_delta) > 1 else (float("nan"), float("nan"))
        r2 = 1.0 - float(np.sum(errors**2) / np.sum((dft_delta - dft_delta.mean())**2))
        # OOF ranking denominators are pair- or composition-level, never cross-formula.
        n_eligible = int(eligible.sum())
        total_correct = int(pf.loc[eligible, "correct"].sum())
        full_eligible = comp_df[comp_df.full_triplet_eligible]
        gs_eligible = comp_df[comp_df.groundstate_eligible]
        global_metrics[condition] = {
            "structures": int(len(frame)), "compositions": int(frame.composition_id.nunique()),
            "energy_mae_mev_atom": float(np.mean(np.abs(frame.energy_pred_ev_atom-frame.energy_dft_ev_atom))*1000),
            "energy_rmse_global_mev_atom": float(np.sqrt(np.mean((frame.energy_pred_ev_atom-frame.energy_dft_ev_atom)**2))*1000),
            "energy_bias_mev_atom": float(np.mean(frame.energy_pred_ev_atom-frame.energy_dft_ev_atom)*1000),
            "deltaE_mae_signed_est_a_mev_atom": float(comp_df.deltaE_mae_mev_atom.mean()),
            "deltaE_rmse_global_mev_atom": float(np.sqrt(np.mean(errors**2))),
            "pairwise_accuracy_eligible": total_correct/n_eligible if n_eligible else float("nan"),
            "pairwise_accuracy_resolved": float(pf.loc[eligible & resolved, "correct"].mean()) if (eligible & resolved).any() else float("nan"),
            "pairwise_unresolved_fraction": float((eligible & ~resolved).sum()/n_eligible) if n_eligible else float("nan"),
            "eligible_pairs": n_eligible, "correct_pairs": total_correct,
            "full_triplet_accuracy": float(full_eligible.full_triplet_correct.mean()) if len(full_eligible) else float("nan"),
            "full_triplet_eligible_compositions": int(len(full_eligible)),
            "full_triplet_correct_compositions": int(full_eligible.full_triplet_correct.sum()),
            "groundstate_accuracy": float(gs_eligible.groundstate_correct.mean()) if len(gs_eligible) else float("nan"),
            "groundstate_eligible_compositions": int(len(gs_eligible)),
            "groundstate_correct_compositions": int(gs_eligible.groundstate_correct.sum()),
            "spearman_mean_by_composition": float(comp_df.spearman.mean()),
            "kendall_mean_by_composition": float(comp_df.kendall.mean()),
            "ols_delta_slope": float(slopes[0]), "ols_delta_intercept_mev_atom": float(slopes[1]), "ols_delta_r2": r2}
        for fold in range(5):
            x = comp_df[comp_df.fold.eq(fold)]
            p = pf[pf.fold.eq(fold)]
            fold_eligible = p[p.eligible]
            folds.append({"condition": condition, "fold": fold, "compositions": int(len(x)),
                "energy_mae_mev_atom": float(x.energy_mae_mev_atom.mean()),
                "deltaE_mae_signed_est_a_mev_atom": float(x.deltaE_mae_mev_atom.mean()),
                "pairwise_accuracy_eligible": float(fold_eligible.correct.mean()) if len(fold_eligible) else float("nan"),
                "eligible_pairs": int(len(fold_eligible)),
                "groundstate_accuracy": float(x.loc[x.groundstate_eligible, "groundstate_correct"].mean()) if x.groundstate_eligible.any() else float("nan")})
    return pd.DataFrame(compositions), pd.DataFrame(folds), pd.DataFrame(polymorphs), global_metrics


def bootstrap_metrics(compositions: pd.DataFrame, pairs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Bootstrap apareado por fórmula, recalculando cocientes de exactitud.

    Args:
        compositions: Métricas a nivel de fórmula.
        pairs: Resultados por par con elegibilidad y acierto.
        n_boot: Réplicas bootstrap.
        seed: Semilla reproducible.

    Returns:
        Estimaciones e IC95% de los cinco contrastes predefinidos.
    """
    comparisons = [("MAT-G", "GEO"), ("MATTER-G", "GEO"), ("MAT-G", "MAT"),
                   ("MATTER-G", "MATTER"), ("MATTER-G", "MAT")]
    comparisons = [(left, right) for left, right in comparisons
                   if left in ACTIVE_CONDITIONS and right in ACTIVE_CONDITIONS]
    metrics = ("energy_mae_mev_atom", "deltaE_mae_mev_atom", "pairwise_accuracy_pct",
               "groundstate_accuracy_pct")
    formulas = sorted(compositions.composition_id.unique())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(formulas), size=(n_boot, len(formulas)))
    records = []
    for left, right in comparisons:
        a = compositions[compositions.condition.eq(left)].set_index("composition_id").loc[formulas]
        b = compositions[compositions.condition.eq(right)].set_index("composition_id").loc[formulas]
        for metric in metrics:
            if metric == "energy_mae_mev_atom":
                va, vb = a[metric].to_numpy(), b[metric].to_numpy()
                draws = np.mean(va[indices]-vb[indices], axis=1)
                estimate = float(np.mean(va-vb)); unit = "meV/átomo"
            elif metric == "deltaE_mae_mev_atom":
                va, vb = a[metric].to_numpy(), b[metric].to_numpy()
                draws = np.mean(va[indices]-vb[indices], axis=1)
                estimate = float(np.mean(va-vb)); unit = "meV/átomo"
            elif metric == "pairwise_accuracy_pct":
                numerator_a, denominator_a, numerator_b, denominator_b = [], [], [], []
                for condition in (left, right):
                    pf = pairs[pairs.condition.eq(condition)].set_index("composition_id").loc[formulas]
                    nums = pf.groupby(level=0).correct.sum().reindex(formulas).to_numpy(float)
                    dens = pf.groupby(level=0).eligible.sum().reindex(formulas).to_numpy(float)
                    if condition == left:
                        numerator_a, denominator_a = nums, dens
                    else:
                        numerator_b, denominator_b = nums, dens
                num_a = np.sum(numerator_a[indices], axis=1); den_a = np.sum(denominator_a[indices], axis=1)
                num_b = np.sum(numerator_b[indices], axis=1); den_b = np.sum(denominator_b[indices], axis=1)
                draws = 100 * (num_a/den_a - num_b/den_b)
                estimate = 100 * (numerator_a.sum()/denominator_a.sum() - numerator_b.sum()/denominator_b.sum())
                unit = "puntos porcentuales"
            else:
                def gs_vectors(condition: str) -> tuple[np.ndarray, np.ndarray]:
                    subset = compositions[compositions.condition.eq(condition)].set_index("composition_id").loc[formulas]
                    return (subset.groundstate_correct.to_numpy(float), subset.groundstate_eligible.to_numpy(float))
                na, da = gs_vectors(left); nb, db = gs_vectors(right)
                dra = np.sum(na[indices], axis=1); dsa = np.sum(da[indices], axis=1)
                drb = np.sum(nb[indices], axis=1); dsb = np.sum(db[indices], axis=1)
                draws = 100 * (dra/dsa - drb/dsb)
                estimate = 100 * (na.sum()/da.sum() - nb.sum()/db.sum())
                unit = "puntos porcentuales"
            records.append({"contrast": f"{left} − {right}", "metric": metric, "estimate": float(estimate),
                "unit": unit, "ci95_low": float(np.quantile(draws, .025)),
                "ci95_high": float(np.quantile(draws, .975)), "replicates": n_boot,
                "resampling_unit": "composition_id", "seed": seed})
    return pd.DataFrame(records)


def historical_controls() -> dict[str, Any]:
    """Carga resultados MLP8 históricos comparables y verifica valores publicados.

    Returns:
        Métricas MLP8 M2 de MatGPT+geometría y MatterGPT+geometría.
    """
    mat = json.loads(HIST_MATGPT_M2.read_text(encoding="utf-8"))
    matter = json.loads(HIST_MATTER_GEOM.read_text(encoding="utf-8"))
    matm = mat["metrics_estimand_A"]["RETRAINED_M2_C1"]
    mattm = matter["MatterGPT_F0_geometry"]
    return {
        "MatGPT_original_geometry_M2_MLP8": {"energy_mae_mev_atom": matm["mae_EN_mev_atom"],
            "deltaE_mae_signed_mev_atom": matm["mae_delta_signed_mev_atom"],
            "pairwise_accuracy": matm["pairwise_accuracy_eligible"], "source": str(HIST_MATGPT_M2.relative_to(ROOT))},
        "MatterGPT_original_geometry_M2_MLP8": {"energy_mae_mev_atom": mattm["mae_energy_ev_atom"]*1000,
            "deltaE_mae_signed_mev_atom": mattm["mae_delta_pairs_mev_atom"],
            "pairwise_accuracy": mattm["pairwise_accuracy_eligible_including_unresolved"],
            "source": str(HIST_MATTER_GEOM.relative_to(ROOT))}}


def generate_figures(structure: pd.DataFrame, pairs: pd.DataFrame, comps: pd.DataFrame,
                     metrics: dict[str, Any], bootstrap: pd.DataFrame) -> None:
    """Genera seis figuras científicas PNG en español, 300 dpi.

    Args:
        structure: Predicciones estructurales OOF.
        pairs: Predicciones pareadas OOF.
        comps: Métricas macro por fórmula.
        metrics: Métricas agregadas.
        bootstrap: Contrastes apareados con IC95%.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    order = list(ACTIVE_CONDITIONS)
    palette = {"GEO": "#7A8B5A", "MAT": "#2878B5", "MAT-G": "#63A6D8",
               "MATTER": "#8E5AA7", "MATTER-G": "#D28BC2"}
    labels = {"GEO": "Geometría", "MAT": "MatGPT", "MAT-G": "MatGPT + geometría",
              "MATTER": "MatterGPT", "MATTER-G": "MatterGPT + geometría"}
    note = "OOF: 70 composiciones ABS₃ (210 estructuras), 5 folds; cada composición se evalúa una vez. Error macro por composición; unidades por átomo."
    # 1: Energy MAE.
    vals = [metrics[c]["energy_mae_mev_atom"] for c in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars=ax.bar([labels[c] for c in order], vals, color=[palette[c] for c in order])
    ax.set_ylabel("MAE de energía absoluta (meV/átomo)"); ax.set_title("ABS₃: MAE de energía DFT por átomo")
    ax.bar_label(bars, fmt="%.1f", padding=3); ax.grid(axis="y", alpha=.25)
    fig.text(.02,.015,note,fontsize=8,wrap=True); fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(IMAGE_DIR/"energy_mae.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 2: pairwise DeltaE MAE.
    vals=[metrics[c]["deltaE_mae_signed_est_a_mev_atom"] for c in order]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar([labels[c] for c in order],vals,color=[palette[c] for c in order])
    ax.set_ylabel("MAE de ΔE firmado entre polimorfos (meV/átomo)")
    ax.set_title("ABS₃: error de diferencias energéticas intracomposición")
    ax.bar_label(bars,fmt="%.1f",padding=3); ax.grid(axis="y",alpha=.25)
    fig.text(.02,.015,note+" ΔEᵢⱼ = Eⱼ − Eᵢ, con ambos polimorfos de la misma composición.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(IMAGE_DIR/"deltaE_mae.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 3: pairwise ranking with 90% target.
    vals=[100*metrics[c]["pairwise_accuracy_eligible"] for c in order]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar([labels[c] for c in order],vals,color=[palette[c] for c in order])
    ax.axhline(90,color="#A33D3D",linestyle="--",linewidth=1.5,label="Meta descriptiva: 90%")
    ax.set_ylabel("Exactitud pairwise elegible (%)"); ax.set_ylim(0,105)
    ax.set_title("ABS₃: ranking pairwise de polimorfos")
    ax.bar_label(bars,fmt="%.1f%%",padding=3); ax.grid(axis="y",alpha=.25); ax.legend()
    fig.text(.02,.015,"OOF: 70 composiciones y 197 pares elegibles en total (|ΔE_DFT| > 5 meV/átomo). Predicción no resuelta si |ΔE_pred| ≤ 1 meV/átomo y cuenta incorrecta. Línea del 90% = objetivo, no criterio de validez.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(IMAGE_DIR/"pairwise_ranking.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 4: ground-state and exact full-triplet ranking.
    x=np.arange(len(order)); width=.36
    gs=[100*metrics[c]["groundstate_accuracy"] for c in order]
    full=[100*metrics[c]["full_triplet_accuracy"] for c in order]
    fig,ax=plt.subplots(figsize=(10,6)); bars_gs=ax.bar(x-width/2,gs,width,label="Ground state",color="#4C78A8")
    bars_full=ax.bar(x+width/2,full,width,label="Triplete completo",color="#F58518")
    for bar,cond in zip(bars_gs,order):
        ax.annotate(f"{metrics[cond]['groundstate_correct_compositions']}/{metrics[cond]['groundstate_eligible_compositions']}",
                    (bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=7)
    for bar,cond in zip(bars_full,order):
        ax.annotate(f"{metrics[cond]['full_triplet_correct_compositions']}/{metrics[cond]['full_triplet_eligible_compositions']}",
                    (bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords="offset points",ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x,[labels[c] for c in order]); ax.set_ylabel("Exactitud (%)")
    ax.set_title("ABS₃: selección del estado fundamental y orden completo")
    ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.text(.02,.015,"Etiquetas sobre barras = aciertos/elegibles por condición. Ground state elegible si el gap DFT al segundo mínimo >1 meV/átomo; predicción con gap superior ≤1 meV/átomo no cuenta como acierto. Triplete excluido si cualquier par DFT difiere ≤1 meV/átomo.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(IMAGE_DIR/"groundstate_fulltriplet.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 5: signed DeltaE parity with per-condition OLS.
    fig,ax=plt.subplots(figsize=(9,8)); allx=pairs.delta_dft_mev_atom.to_numpy(); lim=max(abs(allx.min()),abs(allx.max()))*1.04
    ax.plot([-lim,lim],[-lim,lim],color="#333333",linestyle="--",label="Ideal: predicción = DFT")
    annotations=[]
    for cond in order:
        sub=pairs[pairs.condition.eq(cond)]; xx=sub.delta_dft_mev_atom.to_numpy(); yy=sub.delta_pred_mev_atom.to_numpy()
        ax.scatter(xx,yy,s=15,alpha=.35,color=palette[cond],label=labels[cond])
        m,b=np.polyfit(xx,yy,1); yfit=m*xx+b; r2=1-np.sum((yy-yfit)**2)/np.sum((yy-yy.mean())**2)
        grid=np.linspace(-lim,lim,200)
        ax.plot(grid,m*grid+b,color=palette[cond],linewidth=1.5,alpha=.9)
        annotations.append(f"{labels[cond]}: N={len(xx)}, m={m:.2f}, b={b:.1f} meV/átomo, R²={r2:.2f}, MAE={np.mean(abs(yy-xx)):.1f} meV/átomo")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect("equal",adjustable="box")
    ax.set_xlabel("ΔE DFT firmada, Eⱼ−Eᵢ (meV/átomo)"); ax.set_ylabel("ΔE predicha firmada, Eⱼ−Eᵢ (meV/átomo)")
    ax.set_title("ABS₃: paridad de diferencias energéticas OOF")
    ax.grid(alpha=.2); ax.legend(loc="upper left",fontsize=8)
    fig.text(.02,.012,"Punto = un par de polimorfos dentro de una composición; 70 composiciones y 210 pares (3 por composición) en cada condición OOF. Líneas de color = ajuste OLS con intercepto libre, calculado con todos los puntos del color:\n"+"\n".join(annotations),fontsize=7.4,wrap=True)
    fig.tight_layout(rect=(0,.20,1,1)); fig.savefig(IMAGE_DIR/"deltaE_parity.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 6: paired bootstrap vs GEO, three metrics with native units.
    chosen=bootstrap[bootstrap.contrast.isin(["MAT-G − GEO","MATTER-G − GEO"])]
    fig,axes=plt.subplots(1,3,figsize=(14,5.8))
    panels=[("deltaE_mae_mev_atom","ΔE MAE (meV/átomo)"),
            ("energy_mae_mev_atom","Energy MAE (meV/átomo)"),
            ("pairwise_accuracy_pct","Exactitud pairwise (p.p.)")]
    for ax,(metric,title) in zip(axes,panels):
        rows=chosen[chosen.metric.eq(metric)].reset_index(drop=True)
        y=np.arange(len(rows)); center=rows.estimate.to_numpy(float)
        lo=center-rows.ci95_low.to_numpy(float); hi=rows.ci95_high.to_numpy(float)-center
        ax.errorbar(center,y,xerr=np.vstack([lo,hi]),fmt="o",capsize=4,color="#315A7D")
        ax.axvline(0,color="#555555",linestyle="--",linewidth=1)
        ax.set_yticks(y,rows.contrast.str.replace(" − GEO","",regex=False)); ax.set_title(title)
        ax.grid(axis="x",alpha=.25)
    fig.suptitle("ABS₃: diferencias apareadas frente a geometría sola (IC95% bootstrap por composición)")
    fig.text(.02,.015,"70 fórmulas remuestreadas en 10 000 réplicas; diferencia = modelo híbrido − GEO. Valores negativos favorecen híbrido en MAE; positivos favorecen híbrido en ranking.",fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.07,1,.92)); fig.savefig(IMAGE_DIR/"bootstrap_vs_geometry.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    """Ejecuta preflight, CV agrupada y consolidación científica."""
    global ACTIVE_CONDITIONS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--bootstrap",type=int,default=10000)
    args=parser.parse_args()
    if args.threads!=1:
        parser.error("Para no interferir con F4, este experimento se limita a un hilo CPU")
    if args.bootstrap<1000:
        parser.error("Use al menos 1000 réplicas bootstrap")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    EXP.mkdir(parents=True,exist_ok=True); IMAGE_DIR.mkdir(parents=True,exist_ok=True); REPORT_DIR.mkdir(parents=True,exist_ok=True)
    threadpool_limits(limits=1)
    status={"status":"RUNNING","started_local":time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "device":"cpu","cpu_threads":1,"mps":False,"cuda":False,"historical_test_accessed":False}
    write_json(EXP/"resources.json",status)
    data,folds=load_dataset_and_folds()
    features,embedding_manifest,checkpoint_state=load_features(data)
    ridge_means,ridge_hashes=ridge_m2_oof_means(data)
    available={"GEO":True,
        "MAT":"MAT" in features and checkpoint_state.get("MAT",{}).get("status")=="PASS",
        "MAT-G":"MAT" in features and checkpoint_state.get("MAT",{}).get("status")=="PASS",
        "MATTER":"MATTER" in features and checkpoint_state.get("MATTER",{}).get("status")=="PASS",
        "MATTER-G":"MATTER" in features and checkpoint_state.get("MATTER",{}).get("status")=="PASS"}
    ACTIVE_CONDITIONS=tuple(c for c in CONDITIONS if available[c])
    if not available["MATTER"] or not available["MAT"]:
        LOGGER.warning("Se ejecutarán únicamente condiciones con cachés verificadas: %s",available)
    data=data.copy(); data["_row"]=np.arange(len(data)); groups,targets=make_triplets(data)
    geometry=data[GEOMETRY].to_numpy(float)
    feature_rows=[]; structure_rows=[]; pair_rows=[]; alpha_rows=[]
    config=("experiment: abs3_geometry_transformers_pairwise_ridge\n"
        f"dataset: {DATASET.relative_to(ROOT)}\ndataset_sha256: {sha256(DATASET)}\n"
        "cohort: {structures: 210, compositions: 70, polymorphs_per_composition: 3}\n"
        "splits: {folds: 5, inner_train_compositions: 45, inner_validation_compositions: 11, outer_evaluation_compositions: 14}\n"
        "conditions: [GEO, MAT, MAT-G, MATTER, MATTER-G]\n"
        "ridge_pairwise: {type: Ridge, fit_intercept: false, solver: lsqr, tol: 1.0e-10, alphas: [0.01, 0.1, 1, 10, 100, 1000]}\n"
        "alpha_selection: inner_validation_deltaE_MAE_then_pairwise_accuracy\n"
        "final_fit: inner_train_plus_validation_after_alpha_selection\n"
        "M2: fixed_historical_17_fraction_models; no_fit; outer_mu_reused_from_verified_M2_OOF\n"
        "rank_thresholds: {eligible_abs_dft_delta_gt_meV_atom: 5, prediction_unresolved_if_abs_delta_le_meV_atom: 1}\n"
        "full_triplet: exclude_formula_if_any_dft_pair_gap_le_1_meV_atom; predicted_tie_le_1_counts_incorrect\n"
        "random_seeds: none; pairwise_Ridge_is_deterministic\n"
        "historical_test_accessed: false\n")
    (EXP/"config.yaml").write_text(config,encoding="utf-8")
    manifest={"status":"PASS","dataset_sha256":sha256(DATASET),"rows":len(data),
        "structure_ids":data.structure_id.nunique(),"compositions":data.composition_id.nunique(),
        "targets_column":"energy_dft_ev_atom","slices_column":"SLICES","fold_manifest_sha256":sha256(FOLDS),
        "splits_per_fold":{"inner_train":45,"inner_validation":11,"outer_evaluation":14},
        "historical_test_accessed":False}
    write_json(EXP/"dataset_manifest.json",manifest)
    write_json(EXP/"embedding_manifest.json",embedding_manifest)
    geom_manifest={"status":"PASS","source":"gga-static_endpoint_cleaned","rows":len(data),
        "structure_id_join":"exact dataset structure_id; no positional join","columns":GEOMETRY,
        "all_finite":bool(np.isfinite(geometry).all()),"scaler":"StandardScaler fit per fold on inner-train for alpha selection; refit on inner-train+validation for final OOF fit"}
    write_json(EXP/"geometry_manifest.json",geom_manifest)
    rows_pred:dict[str,np.ndarray]={}
    preflight={"status":"PASS","dataset_sha256":sha256(DATASET),"folds_sha256":sha256(FOLDS),
        "embedding_manifest":embedding_manifest,"checkpoint_hashes":checkpoint_state,
        "ridge_m2_fold_sha256":ridge_hashes,"ridge_m2_mu_rows":len(ridge_means),
        "conditions_available":available,"geometry_rows":len(data),"threads":1,
        "F4_process_modified":False,"historical_test_accessed":False}
    write_json(EXP/"preflight.json",preflight)
    write_json(EXP/"resources.json",{**status,"wall_seconds":0,"cpu_seconds":0,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"solver":"scikit-learn Ridge lsqr, tol=1e-10"})
    for fold in range(5):
        roles=role_formulas(folds,fold)
        structure_by_cond:dict[str,list[dict[str,Any]]]={}
        pair_by_cond:dict[str,list[dict[str,Any]]]={}
        for condition in CONDITIONS:
            if not available[condition]:
                continue
            source_name={"GEO":None,"MAT":"MAT","MAT-G":"MAT","MATTER":"MATTER","MATTER-G":"MATTER"}[condition]
            raw_parts=[geometry] if source_name is None else [features[source_name]]
            if condition in ("MAT-G","MATTER-G"):
                raw_parts.append(geometry)
            raw=np.concatenate(raw_parts,axis=1)
            # Alpha selection scaler uses inner-train only; validation is transformed, never fitted.
            idx_train=np.concatenate([groups[f] for f in roles["train"]])
            scaler=StandardScaler().fit(raw[idx_train])
            scaled=scaler.transform(raw)
            Xtr,ytr,_=pair_features(roles["train"],scaled,data,groups,targets)
            Xval,yval,valmeta=pair_features(roles["validation"],scaled,data,groups,targets)
            candidates=[]
            for alpha in ALPHAS:
                model=Ridge(alpha=alpha,fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000)
                model.fit(Xtr,ytr)
                pred=model.predict(Xval)
                per_comp=[]
                for formula in roles["validation"]:
                    loc=[i for i,item in enumerate(valmeta) if item["composition_id"]==formula]
                    per_comp.append(float(np.mean(np.abs((pred[loc]-yval[loc])*1000))))
                dft_mev=yval*1000; pred_mev=pred*1000
                eligible=np.abs(dft_mev)>5; resolved=np.abs(pred_mev)>1
                rank=float(np.mean((np.sign(pred_mev[eligible])==np.sign(dft_mev[eligible])) & resolved[eligible])) if eligible.any() else 0.0
                cand={"condition":condition,"fold":fold,"alpha":alpha,
                    "inner_validation_deltaE_mae_mev_atom":float(np.mean(per_comp)),
                    "inner_validation_pairwise_accuracy":rank,"eligible_pairs":int(eligible.sum())}
                candidates.append(cand)
            best=min(candidates,key=lambda r:(r["inner_validation_deltaE_mae_mev_atom"],-r["inner_validation_pairwise_accuracy"],r["alpha"]))
            for item in candidates:
                item["selected"]=(item["alpha"]==best["alpha"])
                alpha_rows.append(item)
            # Refit feature scaler and Ridge on inner-train + validation only.
            fit_formulas=roles["train"]+roles["validation"]
            fit_idx=np.concatenate([groups[f] for f in fit_formulas])
            final_scaler=StandardScaler().fit(raw[fit_idx])
            final_features=final_scaler.transform(raw)
            Xfit,yfit,_=pair_features(fit_formulas,final_features,data,groups,targets)
            final_model=Ridge(alpha=float(best["alpha"]),fit_intercept=False,solver="lsqr",tol=1e-10,max_iter=10000)
            final_model.fit(Xfit,yfit)
            # Algebraic diagnostics on each triplet: scores imply antisymmetry and cycle consistency.
            outer_struct=[]; outer_pair=[]
            for formula in roles["outer"]:
                idx=groups[formula]
                scores=final_model.predict(final_features[idx])
                delta=scores-float(scores.mean())
                mu=ridge_means[(fold,formula)]
                energies=mu+delta
                if abs(float(delta.sum()))>1e-10:
                    raise AssertionError(f"Triplete no centrado: {formula}")
                for k,family in enumerate(FAMILIES):
                    row=data.iloc[idx[k]]
                    outer_struct.append({"condition":condition,"fold":fold,"composition_id":formula,
                        "structure_id":str(row.structure_id),"polymorph":family,
                        "energy_dft_ev_atom":float(targets[formula][k]),"energy_pred_ev_atom":float(energies[k]),
                        "ridge_mean_m2_ev_atom":mu,"residual_centered_ev_atom":float(delta[k]),
                        "selected_alpha":float(best["alpha"])})
                for i,j in ((0,1),(0,2),(1,2)):
                    dft_delta=(targets[formula][j]-targets[formula][i])*1000
                    pred_delta=(scores[j]-scores[i])*1000
                    reverse=(scores[i]-scores[j])*1000
                    if abs(pred_delta+reverse)>1e-10:
                        raise AssertionError("Pairwise Ridge no conserva antisimetría")
                    outer_pair.append({"condition":condition,"fold":fold,"composition_id":formula,
                        "polymorph_i":FAMILIES[i],"polymorph_j":FAMILIES[j],
                        "structure_id_i":str(data.iloc[idx[i]].structure_id),"structure_id_j":str(data.iloc[idx[j]].structure_id),
                        "delta_dft_mev_atom":dft_delta,"delta_pred_mev_atom":pred_delta,
                        "absolute_error_mev_atom":abs(pred_delta-dft_delta),"eligible":abs(dft_delta)>5,
                        "resolved":abs(pred_delta)>1,"correct":bool(abs(pred_delta)>1 and np.sign(pred_delta)==np.sign(dft_delta)),
                        "selected_alpha":float(best["alpha"])})
                # 3-cycle check: Δ01 + Δ12 = Δ02.
                d01=(scores[1]-scores[0]); d12=(scores[2]-scores[1]); d02=(scores[2]-scores[0])
                if abs(float(d01+d12-d02))>1e-10:
                    raise AssertionError("Pairwise Ridge no conserva consistencia de ciclo")
            structure_by_cond[condition]=outer_struct; pair_by_cond[condition]=outer_pair
            structure_rows.extend(outer_struct); pair_rows.extend(outer_pair)
            LOGGER.info("fold=%s condition=%s alpha=%s val_deltaE_MAE=%.4f",fold,condition,best["alpha"],best["inner_validation_deltaE_mae_mev_atom"])
        # Incrementally commit all condition predictions for this outer fold.
        write_csv(EXP/"predicciones_oof_por_estructura.csv",structure_rows)
        write_csv(EXP/"predicciones_oof_por_par.csv",pair_rows)
        write_csv(EXP/"ridge_alpha_selection.csv",alpha_rows)
        write_json(EXP/"resources.json",{**status,"stage":"OOF","completed_folds":fold+1,
            "wall_seconds":time.monotonic()-STARTED,"cpu_seconds":time.process_time()-PROCESS_STARTED,
            "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"device":"cpu","cpu_threads":1,
            "mps":False,"cuda":False,"historical_test_accessed":False})
    structure=pd.DataFrame(structure_rows); pairs=pd.DataFrame(pair_rows)
    expected=len(ACTIVE_CONDITIONS)*210
    if len(structure)!=expected or structure.groupby("condition").structure_id.nunique().ne(210).any():
        raise ValueError(f"OOF estructural incompleto: {len(structure)} filas; esperado {expected}")
    if len(pairs)!=len(ACTIVE_CONDITIONS)*210:
        raise ValueError(f"OOF de pares incompleto: {len(pairs)}; esperado {len(ACTIVE_CONDITIONS)*210}")
    comps,fold_metrics,poly_metrics,metrics=compute_metrics(structure,pairs)
    bootstrap=bootstrap_metrics(comps,pairs,args.bootstrap,20261103)
    write_csv(EXP/"metricas_por_composicion.csv",comps)
    write_csv(EXP/"metricas_por_fold.csv",fold_metrics)
    write_csv(EXP/"energy_mae_results.csv",pd.DataFrame([{"condition":c,"energy_mae_mev_atom":metrics[c]["energy_mae_mev_atom"],"energy_rmse_global_mev_atom":metrics[c]["energy_rmse_global_mev_atom"],"energy_bias_mev_atom":metrics[c]["energy_bias_mev_atom"]} for c in ACTIVE_CONDITIONS]))
    write_csv(EXP/"deltaE_mae_results.csv",pd.DataFrame([{"condition":c,"deltaE_mae_signed_est_a_mev_atom":metrics[c]["deltaE_mae_signed_est_a_mev_atom"],"deltaE_rmse_global_mev_atom":metrics[c]["deltaE_rmse_global_mev_atom"],"ols_delta_slope":metrics[c]["ols_delta_slope"],"ols_delta_intercept_mev_atom":metrics[c]["ols_delta_intercept_mev_atom"],"ols_delta_r2":metrics[c]["ols_delta_r2"]} for c in ACTIVE_CONDITIONS]))
    write_csv(EXP/"polymorph_ranking_results.csv",pd.DataFrame([{"condition":c,"pairwise_accuracy":metrics[c]["pairwise_accuracy_eligible"],"pairwise_accuracy_resolved":metrics[c]["pairwise_accuracy_resolved"],"unresolved_fraction":metrics[c]["pairwise_unresolved_fraction"],"eligible_pairs":metrics[c]["eligible_pairs"],"correct_pairs":metrics[c]["correct_pairs"],"goal_90pct_met_descriptively":metrics[c]["pairwise_accuracy_eligible"]>=.90} for c in ACTIVE_CONDITIONS]))
    write_csv(EXP/"groundstate_results.csv",pd.DataFrame([{"condition":c,"groundstate_accuracy":metrics[c]["groundstate_accuracy"],"eligible_compositions":metrics[c]["groundstate_eligible_compositions"],"correct_compositions":metrics[c]["groundstate_correct_compositions"],"full_triplet_accuracy":metrics[c]["full_triplet_accuracy"],"full_triplet_eligible_compositions":metrics[c]["full_triplet_eligible_compositions"],"full_triplet_correct_compositions":metrics[c]["full_triplet_correct_compositions"]} for c in ACTIVE_CONDITIONS]))
    write_csv(EXP/"metricas_por_polimorfo.csv",poly_metrics)
    write_csv(EXP/"bootstrap_apareado.csv",bootstrap)
    controls=historical_controls()
    write_json(EXP/"metrics.json",{"primary_metric":"macro MAE of 3 signed intracomposition pairs per composition, meV/atom",
        "definitions":{"energy_prediction":"E_pred_i = mu_Ridge_M2(c) + (w·z_i − mean_j(w·z_j))",
            "ridge_pairwise":"Ridge(fit_intercept=False) on z_j−z_i and E_j−E_i",
            "pairwise_threshold":"eligible if |ΔE_DFT|>5 meV/atom; unresolved if |ΔE_pred|<=1 meV/atom and incorrect",
            "full_triplet":"exact total order; exclude if any DFT pair gap <=1 meV/atom; predicted pair gap <=1 is unresolved/incorrect",
            "ground_state":"eligible if DFT second−first gap>1 meV/atom; predicted top gap<=1 is not a hit",
            "aggregation":"one deterministic OOF prediction/composition; pairs nested inside composition"},
        "models":metrics,"bootstrap":bootstrap.to_dict("records"),"historical_mlp8_controls":controls,"historical_test_accessed":False})
    write_json(EXP/"summary.json",{"status":"COMPLETE","conditions":ACTIVE_CONDITIONS,"metrics":metrics,
        "ranking_90pct_goal_reached":{c:metrics[c]["pairwise_accuracy_eligible"]>=.90 for c in ACTIVE_CONDITIONS},
        "historical_mlp8_controls":controls,"test_split_accessed":False})
    if set(ACTIVE_CONDITIONS)==set(CONDITIONS):
        generate_figures(structure,pairs,comps,metrics,bootstrap)
    resources={"status":"COMPLETE","wall_seconds":time.monotonic()-STARTED,
        "process_cpu_seconds":time.process_time()-PROCESS_STARTED,
        "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device":"cpu","cpu_threads":1,"mps":False,"cuda":False,
        "solver":"scikit-learn Ridge LSQR, tol=1e-10, max_iter=10000",
        "matrices":{"feature_dimensions":{"GEO":7,"MAT":768,"MAT-G":775,"MATTER":768,"MATTER-G":775},
            "outer_structure_rows":len(structure),"outer_pair_rows":len(pairs)},
        "bootstrap_replicates":args.bootstrap,"historical_test_accessed":False,
        "python_version":platform.python_version(),"numpy_version":np.__version__,"pandas_version":pd.__version__,
        "scikit_learn_version":__import__("sklearn").__version__}
    write_json(EXP/"resources.json",resources)
    report=["# ABS₃: geometría y Transformers con Ridge pairwise","","## Cohorte y protocolo","",
        f"Dataset SHA-256: `{sha256(DATASET)}`. Se evaluaron 70 composiciones (210 estructuras, tres polimorfos), cinco folds agrupados y un OOF determinista por composición. Cada fold tuvo 45 fórmulas de inner-train, 11 de inner-validation y 14 outer. El test histórico no se abrió.","",
        "Ridge M2 histórico, con 17 fracciones estequiométricas, permaneció fijo; sus predicciones outer verificadas anclan E_pred. El Ridge pairwise no tiene intercepto y aprende ΔE firmada sobre z_j−z_i. α se seleccionó por ΔE MAE en validación, con exactitud pairwise como desempate; la normalización final se ajustó con inner-train más inner-validation, sin outer.","",
        "Los dos extractores permanecieron congelados. Se reutilizaron embeddings FINAL_END existentes y se incorporaron los siete descriptores de `gga-static_endpoint_cleaned` después de la extracción. Las celdas ya están relajadas por DFT, por lo que estas condiciones híbridas reciben geometría posterior a relajación.","",
        "## Resultados principales","",pd.DataFrame([{"Condición":c,"Energy MAE (meV/átomo)":metrics[c]["energy_mae_mev_atom"],"ΔE MAE (meV/átomo)":metrics[c]["deltaE_mae_signed_est_a_mev_atom"],"Pairwise (%)":100*metrics[c]["pairwise_accuracy_eligible"],"Triplete exacto (%)":100*metrics[c]["full_triplet_accuracy"],"Ground state (%)":100*metrics[c]["groundstate_accuracy"]} for c in ACTIVE_CONDITIONS]).to_markdown(index=False),"",
        "### Objetivo de ranking del 90%","", "El 90% es una meta descriptiva, no una regla de validez ni de selección. Pairwise se calcula en pares con |ΔE_DFT|>5 meV/átomo; predicciones con |ΔE_pred|≤1 meV/átomo se consideran no resueltas e incorrectas.","",
        "## Contrastes apareados","",bootstrap.to_markdown(index=False),"",
        "## Controles MLP8 históricos","",json.dumps(controls,indent=2,ensure_ascii=False),"",
        "Los resultados históricos se reportan aparte: sus protocolos/cabezas no son el Ridge pairwise nuevo. Solo se comparan como referencias cuando comparten Ridge M2, targets, cohortes y estimando documentados.","",
        "## Limitaciones","","La evaluación es OOF sobre el conjunto de desarrollo y no sustituye el test independiente cerrado. Ridge es lineal sobre diferencias y, algebraicamente, equivale a una puntuación lineal por estructura. Un ranking alto no garantiza calibración de magnitudes. La geometría proviene de estructuras ya relajadas por DFT. No se utilizaron semillas duplicadas para el Ridge determinista.",""]
    text="\n".join(report)
    (EXP/"reporte_cientifico.md").write_text(text,encoding="utf-8")
    (REPORT_DIR/"reporte_cientifico.md").write_text(text,encoding="utf-8")
    readme=("# ABS3 Geometry + Frozen Transformers Pairwise Ridge\n\n"
        "Ejecutar desde la raíz: `python scripts/ABS3/abs3_geometry_transformers_pairwise_ridge/run_benchmark.py --threads 1 --bootstrap 10000`. "
        "Se reutilizan cachés originales verificadas. Ridge M2 queda fijo. Consulte `preflight.json`, `metrics.json`, `resources.json` y `reporte_cientifico.md`.\n")
    (EXP/"README.md").write_text(readme,encoding="utf-8")
    LOGGER.info("Benchmark completo: %s",json.dumps(metrics,ensure_ascii=False))


if __name__ == "__main__":
    main()
