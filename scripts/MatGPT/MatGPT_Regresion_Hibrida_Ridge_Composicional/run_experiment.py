"""Ajusta cinco Ridge composicionales y combina sus medias con residuos ya guardados.

No invoca MatGPT, no entrena MLP8/LoRA y no lee el test histórico. El único
ajuste nuevo es StandardScaler + Ridge por cada fold externo.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common as cm

LOG = logging.getLogger("matgpt_hybrid_composition")


def formula_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula un único target DFT medio por fórmula en eV/átomo.

    Args:
        data: Estructuras train/val con energías por átomo verificadas.
    Returns:
        Tabla de 70 fórmulas, media DFT y tamaño del triplete.
    """
    result = data.groupby("composition_id", as_index=False).agg(
        mean_energy_dft_ev_atom=("energy_per_atom_dft_ev", "mean"), n_structures=("structure_id", "nunique"))
    if len(result) != 70 or result.n_structures.ne(3).any():
        raise ValueError("Cada target composicional requiere tres polimorfos DFT")
    return result


def formula_roles(folds: pd.DataFrame, fold: int) -> dict[str, set[str]]:
    """Obtiene fórmulas únicas para ajuste, selección y evaluación externa.

    Args:
        folds: Asignaciones históricas por estructura y fold.
        fold: Fold externo de 0 a 4.
    Returns:
        Sets de fórmulas inner-train, inner-validation, outer-train y outer-test.
    """
    part = folds.loc[folds.outer_fold.eq(fold)]
    unique = part.drop_duplicates("structure_id")
    result = {
        "inner_train": set(part.loc[part.inner_role.eq("inner_train"), "composition_group"]),
        "inner_validation": set(part.loc[part.inner_role.eq("inner_validation"), "composition_group"]),
        "outer_train": set(unique.loc[unique.outer_role.eq("outer_train"), "composition_group"]),
        "outer_test": set(unique.loc[unique.outer_role.eq("outer_validation"), "composition_group"]),
    }
    if tuple(map(len, (result["inner_train"], result["inner_validation"],
                       result["outer_train"], result["outer_test"]))) != (45, 11, 56, 14):
        raise ValueError(f"Fold {fold}: conteos de composición distintos de 45/11/56/14")
    if (result["inner_train"] & result["inner_validation"] or
            result["outer_train"] & result["outer_test"] or
            result["inner_train"] | result["inner_validation"] != result["outer_train"]):
        raise ValueError(f"Fold {fold}: particiones de fórmula solapadas o incompatibles")
    return result


def fit_composition_ridges(data: pd.DataFrame, folds: pd.DataFrame, feature_table: pd.DataFrame,
                           output: Path) -> tuple[dict[int, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Selecciona alpha en 45/11 fórmulas y guarda cinco refits 56-fórmula.

    Args:
        data: Dataset de 210 estructuras de desarrollo.
        folds: Manifiesto agrupado por fórmula.
        feature_table: Matriz estequiométrica 70×17.
        output: Directorio de artefactos del experimento.
    Returns:
        Diccionario fold→Pipeline, tabla de selección y grid de validación.
    """
    targets = formula_targets(data).set_index("composition_id")
    features = feature_table.set_index("composition_id").loc[targets.index, cm.ELEMENTS]
    X = features.to_numpy(dtype=np.float64)
    y = targets.mean_energy_dft_ev_atom.to_numpy(dtype=np.float64)
    formulas = targets.index.to_numpy(dtype=str)
    x_by = {formula: X[index] for index, formula in enumerate(formulas)}
    y_by = targets.mean_energy_dft_ev_atom.to_dict()
    models: dict[int, Pipeline] = {}
    selections: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        roles = formula_roles(folds, fold)
        train_formulas = sorted(roles["inner_train"])
        validation_formulas = sorted(roles["inner_validation"])
        outer_train_formulas = sorted(roles["outer_train"])
        x_train = np.stack([x_by[item] for item in train_formulas])
        y_train = np.asarray([y_by[item] for item in train_formulas])
        x_valid = np.stack([x_by[item] for item in validation_formulas])
        y_valid = np.asarray([y_by[item] for item in validation_formulas])
        best_alpha: float | None = None
        best_mae = float("inf")
        for alpha in cm.ALPHAS:
            candidate = Pipeline([("scaler", StandardScaler()),
                                  ("ridge", Ridge(alpha=alpha, fit_intercept=True,
                                                   solver="lsqr", tol=1e-8))])
            candidate.fit(x_train, y_train)
            predictions = candidate.predict(x_valid)
            validation_mae = float(mean_absolute_error(y_valid, predictions))
            validation_rows.append({"fold": fold, "alpha": alpha,
                                    "inner_train_compositions": 45,
                                    "inner_validation_compositions": 11,
                                    "validation_mae_ev_atom": validation_mae})
            if validation_mae < best_mae:
                best_alpha, best_mae = float(alpha), validation_mae
        if best_alpha is None:
            raise RuntimeError(f"No se seleccionó alpha para el fold {fold}")
        x_outer = np.stack([x_by[item] for item in outer_train_formulas])
        y_outer = np.asarray([y_by[item] for item in outer_train_formulas])
        final = Pipeline([("scaler", StandardScaler()),
                          ("ridge", Ridge(alpha=best_alpha, fit_intercept=True,
                                           solver="lsqr", tol=1e-8))])
        final.fit(x_outer, y_outer)
        path = model_dir / f"ridge_composicional_fold_{fold}.joblib"
        joblib.dump(final, path)
        models[fold] = final
        selections.append({"fold": fold, "selected_alpha": best_alpha,
                           "inner_validation_mae_ev_atom": best_mae,
                           "ridge_features": 17, "inner_train_compositions": 45,
                           "inner_validation_compositions": 11, "outer_train_compositions": 56,
                           "outer_test_compositions": 14,
                           "scaler_fit_role": "inner_train_for_selection; outer_train_for_saved_pipeline",
                           "solver": "lsqr", "tolerance": 1e-8,
                           "intercept": True, "pipeline_path": str(path.relative_to(cm.ROOT))})
        LOG.info(json.dumps({"event": "ridge_fold_complete", "fold": fold,
                             "selected_alpha": best_alpha, "validation_mae_ev_atom": best_mae,
                             "n_fit_outer": len(outer_train_formulas)}, ensure_ascii=False))
    selection_frame = pd.DataFrame(selections)
    grid_frame = pd.DataFrame(validation_rows)
    # The existing composition-only control is an independent implementation check.
    existing = pd.read_csv(cm.COMPOSITION_CONTROL)
    old_alphas = pd.read_csv(cm.ROOT / "reports/MatGPT/MatGPT_H3C_Final/controls/composition_seed_42/fold_records.csv")
    old_alpha_by_fold = old_alphas.set_index("fold").selected_alpha.astype(float).to_dict()
    if any(float(selection_frame.set_index("fold").loc[fold, "selected_alpha"]) != old_alpha_by_fold[fold]
           for fold in range(5)):
        raise ValueError("Selección de alpha no reproduce el control composicional histórico")
    dft_formula = formula_targets(data).set_index("composition_id").mean_energy_dft_ev_atom
    feature_index = feature_table.set_index("composition_id")
    differences: list[float] = []
    for fold, model in models.items():
        held = formula_roles(folds, fold)["outer_test"]
        table = data[data.composition_id.isin(held)].drop_duplicates("composition_id").set_index("composition_id")
        predicted = model.predict(feature_index.loc[sorted(held), cm.ELEMENTS].to_numpy(dtype=float))
        pred_by = dict(zip(sorted(held), predicted))
        for row in existing.loc[existing.fold.eq(fold)].itertuples():
            if row.composition_id not in pred_by:
                raise ValueError("Control composicional histórico usa una fórmula externa incorrecta")
            differences.append(abs(float(row.ridge_mean_ev_atom) - float(pred_by[row.composition_id])))
    if max(differences, default=float("inf")) > 1e-5:
        raise ValueError(f"Nuevo Ridge no reproduce el control composicional existente: max={max(differences):g}")
    selection_frame["historical_control_alpha_match"] = True
    selection_frame["max_abs_difference_existing_composition_control_ev_atom"] = max(differences)
    return models, selection_frame, grid_frame


def make_predictions(data: pd.DataFrame, folds: pd.DataFrame, source: pd.DataFrame,
                     models: dict[int, Pipeline], feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crea M1–M4 y composición sola usando residuos existentes sin alterarlos.

    Args:
        data: Referencia DFT de desarrollo.
        folds: Asignaciones externas históricas.
        source: Predicciones Frozen/LoRA ya existentes.
        models: Los cinco Ridge composicionales guardados.
        feature_table: Fracciones elementales por fórmula.
    Returns:
        Predicciones por estructura, control composicional y medias por fórmula.
    """
    lookup = data.set_index("structure_id")
    feat = feature_table.set_index("composition_id")
    mean_by_fold_formula: dict[tuple[int, str], float] = {}
    embedding_means: dict[tuple[int, str], float] = {}
    composition_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    composition_only_rows: list[dict[str, Any]] = []
    source_groups = source.groupby(["condition", "fold", "seed", "composition_id"], sort=False)
    for fold, model in models.items():
        held = sorted(formula_roles(folds, fold)["outer_test"])
        features = feat.loc[held, cm.ELEMENTS].to_numpy(dtype=np.float64)
        means = model.predict(features)
        for formula, mean in zip(held, means):
            mean_by_fold_formula[(fold, formula)] = float(mean)
        for formula in held:
            source_formula = source[(source.fold == fold) & (source.composition_id == formula)]
            if source_formula.empty:
                raise ValueError(f"Faltan residuos guardados para {formula}, fold {fold}")
            embedding_values = source_formula.ridge_mean_ev_atom.to_numpy(dtype=float)
            if np.ptp(embedding_values) > 1e-6:
                raise ValueError(f"Ridge de embeddings cambia con semilla/residual para {formula}")
            embedding_means[(fold, formula)] = float(embedding_values.mean())
            target = lookup.loc[source_formula.structure_id.iloc[0]]
            composition_rows.append({"fold": fold, "composition_id": formula,
                                     "n_structures": 3,
                                     "mean_energy_dft_ev_atom": float(source_formula.energy_per_atom_dft_ev.mean()),
                                     "mean_energy_embedding_ridge_ev_atom": embedding_means[(fold, formula)],
                                     "mean_energy_composition_ridge_ev_atom": mean_by_fold_formula[(fold, formula)],
                                     **{element: float(value) for element, value in zip(cm.ELEMENTS,
                                              cm.stoichiometric_fractions(formula))},
                                     "n_atoms_first_structure": int(target.n_atoms)})
            for _, row in source_formula.iterrows():
                old_mean = float(row.ridge_mean_ev_atom)
                residual = float(row.residual_ev_atom)
                model_id = "M1" if row.condition == "frozen" else "M3"
                record = {"model_id": model_id, "mean_branch": "ridge_embeddings",
                          "residual_branch": "frozen_mlp8" if row.condition == "frozen" else "lora_mlp8",
                          "fold": fold, "seed": int(row.seed), "structure_id": row.structure_id,
                          "composition_id": formula, "polymorph_family": row.polymorph_family,
                          "n_atoms": int(row.n_atoms),
                          "energy_total_dft_ev": float(row.energy_total_dft_ev),
                          "energy_per_atom_dft_ev": float(row.energy_per_atom_dft_ev),
                          "mean_energy_pred_ev_atom": old_mean,
                          "residual_pred_ev_atom": residual,
                          "energy_per_atom_pred_ev": float(row.energy_per_atom_pred_ev),
                          "energy_total_pred_ev": float(row.energy_total_pred_ev)}
                if abs(record["energy_per_atom_pred_ev"] - (old_mean + residual)) > 1e-6:
                    raise ValueError(f"Suma Ridge/residuo no coincide con salida guardada: {row.structure_id}")
                structure_rows.append(record)
                comp_mean = mean_by_fold_formula[(fold, formula)]
                new_id = "M2" if row.condition == "frozen" else "M4"
                new_pred = comp_mean + residual
                structure_rows.append({**record, "model_id": new_id, "mean_branch": "ridge_composicion",
                                       "mean_energy_pred_ev_atom": comp_mean,
                                       "energy_per_atom_pred_ev": new_pred,
                                       "energy_total_pred_ev": new_pred * int(row.n_atoms)})
        for formula in held:
            comp_mean = mean_by_fold_formula[(fold, formula)]
            rows = data.loc[data.composition_id.eq(formula)]
            for row in rows.itertuples():
                composition_only_rows.append({"model_id": "composition_only", "fold": fold,
                                              "structure_id": row.structure_id,
                                              "composition_id": formula,
                                              "polymorph_family": row.polymorph_family,
                                              "n_atoms": int(row.n_atoms),
                                              "energy_total_dft_ev": float(row.energy_total_dft_ev),
                                              "energy_per_atom_dft_ev": float(row.energy_per_atom_dft_ev),
                                              "mean_energy_pred_ev_atom": comp_mean,
                                              "residual_pred_ev_atom": 0.0,
                                              "energy_per_atom_pred_ev": comp_mean,
                                              "energy_total_pred_ev": comp_mean * int(row.n_atoms)})
    predictions = pd.DataFrame(structure_rows)
    baseline = pd.DataFrame(composition_only_rows)
    composition_predictions = pd.DataFrame(composition_rows)
    if (len(predictions) != 2520 or predictions.structure_id.nunique() != 210 or
            predictions.groupby("model_id").size().to_dict() != {"M1": 630, "M2": 630, "M3": 630, "M4": 630}):
        raise ValueError("Predicciones M1–M4 requieren 630 filas por modelo sobre los mismos 210 IDs")
    if len(baseline) != 210 or baseline.structure_id.nunique() != 210:
        raise ValueError("Control composición-only requiere 210 estructuras, una predicción por estructura")
    if len(composition_predictions) != 70 or composition_predictions.composition_id.nunique() != 70:
        raise ValueError("La tabla de medias debe cubrir las 70 composiciones OOF exactamente una vez")
    return predictions, baseline, composition_predictions


def write_split_table(folds: pd.DataFrame, output: Path) -> None:
    """Exporta asignaciones por fórmula sin tocar ni reconstruir los folds.

    Args:
        folds: Manifiesto histórico de 1050 filas.
        output: Directorio de reportes.
    Returns:
        None; escribe `outer_split_assignments.csv`.
    """
    rows = []
    for fold, part in folds.groupby("outer_fold", sort=True):
        for formula, group in part.groupby("composition_group", sort=True):
            roles = set(group.inner_role)
            outer_role = set(group.outer_role)
            if len(roles) != 1 or len(outer_role) != 1 or len(group) != 3:
                raise ValueError(f"Fórmula dividida entre roles: {formula}, fold {fold}")
            role = "outer_test" if outer_role == {"outer_validation"} else next(iter(roles))
            rows.append({"fold": int(fold), "composition_id": formula, "role": role,
                         "n_structures": len(group), "structure_ids": ";".join(sorted(group.structure_id))})
    pd.DataFrame(rows).to_csv(output / "outer_split_assignments.csv", index=False)


def run(output: Path = cm.REPORT) -> dict[str, Any]:
    """Ejecuta auditoría, Ridge, combinación de residuos y guarda CSV/joblib.

    Args:
        output: Carpeta canónica del experimento.
    Returns:
        Resumen pequeño para logging; el detalle vive en CSV y Markdown.
    """
    allowed_measurement_files = {"resources.json", "run.log"}
    if output.exists() and any(path.name not in allowed_measurement_files for path in output.iterdir()):
        raise FileExistsError(f"La salida ya contiene resultados; se rehúsa sobrescribir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data, folds, source, audit = cm.load_and_audit_sources()
    features = cm.formula_feature_table(data)
    features.to_csv(output / "composition_features.csv", index=False)
    write_split_table(folds, output)
    (output / "auditoria_ridge_composicional.md").write_text(
        "# Auditoría Ridge composicional\n\n"
        "## Fuentes y controles verificados\n\n" +
        "| Control | Resultado |\n|---|---|\n" +
        f"| Dataset train/val | {audit['source_counts']['structures']} estructuras, 70 fórmulas |\n" +
        f"| Asignaciones históricas | {audit['source_counts']['fold_assignments']} filas; 45/11/56/14 fórmulas por pliegue |\n" +
        f"| Predicciones Frozen/LoRA existentes | {audit['source_counts']['C_predictions']} filas; hash verificado |\n" +
        f"| Replay H3-C Frozen | PASS; diferencia máxima {audit['frozen_prediction_max_abs_difference_from_h3c_ev_atom']:.3g} eV/átomo |\n" +
        "| IDs, N, composición y energías | Coincidencia exacta con las etiquetas del desarrollo |\n" +
        "| Target del Ridge nuevo | Media de E_DFT/N de los tres polimorfos, eV/átomo |\n" +
        "| Datos del test histórico | No leídos ni usados |\n\n" +
        "## Características y alpha\n\n" +
        "Vector de 17 fracciones estequiométricas en el orden fijo Ba, Ca, Cd, Cu, Ge, Hf, Mg, Nb, Pb, S, Si, Sn, Sr, Ti, V, Zn y Zr. " +
        "No entran IDs, etiquetas DFT, geometría, topología ni propiedades calculadas. " +
        "Cada polimorfo recibe el mismo vector; las fracciones se comprueban en `composition_features.csv`.\n\n" +
        "El StandardScaler de selección se ajusta solo en las 45 fórmulas inner-train; cada alpha se puntúa sobre 11 fórmulas inner-validation. " +
        "Después se crea un pipeline independiente por pliegue y se reajusta en las 56 fórmulas outer-train. " +
        "Grid alpha: `[0.01, 0.1, 1, 10, 100, 1000]`, según el control composicional existente. " +
        "Solver `lsqr`, tolerancia 1e-8, intercepto habilitado. Resultados en `ridge_validation_grid.csv` y `ridge_selection.csv`.\n\n" +
        "## Procedencia\n\n" +
        f"Dataset SHA-256: `{audit['dataset_sha256']}`; folds: `{audit['folds_sha256']}`; " +
        f"predicciones Frozen/LoRA: `{audit['frozen_lora_predictions_sha256']}`; " +
        f"reproducción H3-C: `{audit['h3c_replayed_predictions_sha256']}`; " +
        f"control composicional previo: `{audit['existing_composition_control_sha256']}`.\n",
        encoding="utf-8")
    models, selection, validation = fit_composition_ridges(data, folds, features, output)
    selection.to_csv(output / "ridge_selection.csv", index=False)
    validation.to_csv(output / "ridge_validation_grid.csv", index=False)
    predictions, baseline, composition_predictions = make_predictions(data, folds, source, models, features)
    predictions.to_csv(output / "oof_predictions_by_structure.csv", index=False)
    baseline.to_csv(output / "composition_only_predictions.csv", index=False)
    composition_predictions.to_csv(output / "oof_mean_energy_by_composition.csv", index=False)
    # Copiar los IDs/targets/predicciones históricas usadas como controles sin reescribir las fuentes.
    frozen_ids = source.loc[source.condition.eq("frozen"), ["structure_id", "fold", "seed", "energy_per_atom_pred_ev"]].copy()
    frozen_ids.to_csv(output / "source_frozen_predictions_used.csv", index=False)
    summary = {"status": "PASS", "newly_fitted_models": 5, "newly_trained_components": ["Ridge composicional"],
               "MLP8_or_LoRA_training": False, "transformer_inference": False,
               "rows_M1_to_M4": len(predictions), "composition_only_rows": len(baseline),
               "physical_structures": 210, "compositions": 70, "historical_test_used": False,
               "selected_alpha_by_fold": dict(zip(selection.fold.astype(str), selection.selected_alpha.astype(float)))}
    LOG.info(json.dumps({"event": "model_predictions_complete", **summary}, ensure_ascii=False))
    return summary


def main() -> None:
    """CLI con salida configurable y bloqueo de sobrescritura.

    Returns:
        None; devuelve código cero cuando completan auditoría y ajuste.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cm.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.output.resolve())


if __name__ == "__main__":
    main()
