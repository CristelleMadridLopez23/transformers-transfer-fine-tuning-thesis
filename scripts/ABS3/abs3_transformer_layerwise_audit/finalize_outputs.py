"""Finaliza informe, bootstrap energético y recursos desde artefactos OOF guardados.

Este script permite cerrar el paquete sin repetir el ajuste de probes ni las
permutaciones si la etapa de presentación se interrumpe después de guardar CSV.
"""
from __future__ import annotations

import json
import logging
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from run_layerwise import EXP, IMG, REPORT, ROOT, BOOT_N, BOOT_SEED, atomic_csv, atomic_json, build_report, make_figures

LOG = logging.getLogger("abs3.layerwise.finalize")


def bootstrap_energy(comp: pd.DataFrame) -> pd.DataFrame:
    """Calcula bootstrap apareado BEST_LAYER−FINAL para MAE energético.

    Args:
        comp: Métricas OOF por fórmula, incluidas filas FINAL y BEST_LAYER.
    Returns:
        Una fila de IC95% por modelo/probe.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOT_SEED)
    selected = comp[comp.evaluation_role.isin(["FINAL", "BEST_LAYER"])]
    for (model, probe), group in selected.groupby(["model", "probe"], sort=False):
        wide = group.pivot(index="formula", columns="evaluation_role", values="energy_mae_mev_atom").dropna()
        if len(wide) != 70:
            raise ValueError(f"Bootstrap requiere 70 fórmulas para {model}/{probe}; recibió {len(wide)}")
        delta = (wide["BEST_LAYER"] - wide["FINAL"]).to_numpy(float)
        samples = rng.integers(0, len(delta), size=(BOOT_N, len(delta)))
        draws = delta[samples].mean(axis=1)
        rows.append({"contrast": f"{model} BEST_LAYER − FINAL {probe}", "metric": "energy_mae_mev_atom",
                     "estimate": float(delta.mean()), "ci95_low": float(np.quantile(draws, .025)),
                     "ci95_high": float(np.quantile(draws, .975)), "replicates": BOOT_N,
                     "seed": BOOT_SEED, "unit": "composition"})
    return pd.DataFrame(rows)


def selected_metrics(comp: pd.DataFrame, pairs: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas OOF de las filas FINAL y BEST_LAYER.

    Args:
        comp: Métricas por composición/rol.
        pairs: Predicciones OOF firmadas por par.
    Returns:
        Métricas agregadas por modelo, probe y procedimiento.
    """
    rows: list[dict[str, Any]] = []
    for (model, probe, role), group in comp[comp.evaluation_role.isin(["FINAL", "BEST_LAYER"])].groupby(
            ["model", "probe", "evaluation_role"], sort=False):
        p = pairs[(pairs.model.eq(model)) & (pairs.probe.eq(probe)) & (pairs.evaluation_role.eq(role))]
        s = structures[(structures.model.eq(model)) & (structures.probe.eq(probe)) & (structures.evaluation_role.eq(role))]
        x = p.delta_dft_mev_atom.to_numpy(float)
        y = p.delta_pred_mev_atom.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        eligible = int(p.eligible_pair.astype(bool).sum())
        rows.append({"model": model, "probe": probe, "layer": role, "folds": int(group.fold.nunique()),
                     "structures": int(group.formula.nunique() * 3), "compositions": int(group.formula.nunique()),
                     "pairs": len(p), "energy_mae_mev_atom": float(np.abs(s.energy_error_mev_atom).mean()),
                     "energy_rmse_global_mev_atom": float(np.sqrt(np.mean(s.energy_error_mev_atom.to_numpy(float) ** 2))),
                     "deltaE_mae_estimand_A_mev_atom": float(group.deltaE_mae_estimand_A_mev_atom.mean()),
                     "deltaE_rmse_global_mev_atom": float(np.sqrt(np.mean((y - x) ** 2)),),
                     "pairwise_correct": int(p.loc[p.eligible_pair.astype(bool), "pair_correct"].astype(bool).sum()),
                     "pairwise_eligible": eligible,
                     "pairwise_accuracy": float(p.loc[p.eligible_pair.astype(bool), "pair_correct"].astype(bool).mean()),
                     "unresolved_eligible": int((p.eligible_pair.astype(bool) & ~p.prediction_resolved.astype(bool)).sum()),
                     "groundstate_correct": int(group.groundstate_correct.sum()),
                     "groundstate_eligible": int(group.groundstate_eligible.sum()),
                     "groundstate_accuracy": float(group.groundstate_correct.sum() / group.groundstate_eligible.sum()),
                     "full_triplet_correct": int(group.full_triplet_correct.sum()),
                     "full_triplet_eligible": int(group.full_triplet_eligible.sum()),
                     "full_triplet_accuracy": float(group.full_triplet_correct.sum() / group.full_triplet_eligible.sum()),
                     "ols_slope": float(slope), "predictive_r2": float(r2_score(x, y)),
                     "ols_intercept_mev_atom": float(intercept), "ols_r2": float(r2_score(x, y))})
    return pd.DataFrame(rows)


def main() -> None:
    """Valida artefactos guardados y escribe los entregables canónicos."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    required = ["layer_energy_results.csv", "layer_selection_per_fold.csv", "bootstrap_paired.csv",
                "metrics_per_composition.csv", "conditional_embedding_value.csv", "conditional_value_oof_pairs.csv",
                "conditional_value_oof_structures.csv", "permutation_controls.csv", "token_alignment_audit.json"]
    missing = [f for f in required if not (EXP / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos OOF para finalizar: {missing}")
    metrics = pd.read_csv(EXP / "layer_energy_results.csv")
    selections = pd.read_csv(EXP / "layer_selection_per_fold.csv")
    boot = pd.read_csv(EXP / "bootstrap_paired.csv")
    comp = pd.read_csv(EXP / "metrics_per_composition.csv")
    structures = pd.read_csv(EXP / "oof_predictions_structures.csv")
    conditional = pd.read_csv(EXP / "conditional_embedding_value.csv")
    cp = pd.read_csv(EXP / "conditional_value_oof_pairs.csv")
    cs = pd.read_csv(EXP / "conditional_value_oof_structures.csv")
    perms = pd.read_csv(EXP / "permutation_controls.csv")
    pairs = pd.read_csv(EXP / "oof_predictions_pairs.csv")
    derived = selected_metrics(comp, pairs, structures)
    metrics = pd.concat([metrics[~metrics.layer.isin(["FINAL", "BEST_LAYER"])], derived], ignore_index=True)
    atomic_csv(EXP / "layer_energy_results.csv", metrics)
    atomic_csv(EXP / "layer_ridge_results.csv", metrics[metrics.probe.eq("Ridge")])
    atomic_csv(EXP / "layer_krr_results.csv", metrics[metrics.probe.eq("KRR")])
    # Sustituye una etiqueta interna provisional por el nombre del probe real.
    for filename in ["oof_predictions_pairs.csv", "oof_predictions_structures.csv"]:
        frame = pd.read_csv(EXP / filename)
        if "condition" in frame.columns:
            frame["condition"] = "PAIRWISE_" + frame["probe"].astype(str).str.upper()
            atomic_csv(EXP / filename, frame)
    energy_boot = bootstrap_energy(comp)
    boot = pd.concat([boot, energy_boot], ignore_index=True)
    boot = boot.drop_duplicates(["contrast", "metric"], keep="last").reset_index(drop=True)
    atomic_csv(EXP / "bootstrap_paired.csv", boot)
    variance = pd.read_csv(EXP / "within_between_variance.csv")
    geometry = pd.read_csv(EXP / "layer_geometry_probe.csv")
    inner = pd.read_csv(EXP / "layer_energy_inner_validation.csv")
    make_figures(pd.read_csv(EXP / "layer_separability.csv"), variance, geometry, inner,
                 selections, metrics, pairs, conditional, boot)
    alignment = json.loads((EXP / "token_alignment_audit.json").read_text(encoding="utf-8"))
    report = build_report(metrics, selections, boot, conditional, alignment, perms)
    (EXP / "reporte_cientifico.md").write_text(report, encoding="utf-8")
    (REPORT / "reporte_cientifico.md").write_text(report, encoding="utf-8")
    readme = ("# Auditoría layerwise ABS₃\n\nAuditoría OOF de estados L0–L12 y representación FINAL de MatGPT y MatterGPT originales. "
              "La selección por capa usa inner-validation; el test independiente permanece cerrado. "
              "Reproducir con `python scripts/ABS3/abs3_transformer_layerwise_audit/extract_layers.py`, "
              "`python scripts/ABS3/abs3_transformer_layerwise_audit/run_layerwise.py` y, si solo falta cerrar el informe, "
              "`python scripts/ABS3/abs3_transformer_layerwise_audit/finalize_outputs.py`.\n")
    for path in [EXP / "README.md", REPORT / "README.md"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme, encoding="utf-8")
    config = """# Protocolo fijado — auditoría de representaciones por capa ABS₃
experiment: abs3_transformer_layerwise_audit
cohort:
  compositions: 70
  structures: 210
  polymorphs_per_composition: 3
  outer_folds: 5
  inner_train_formulas: 45
  inner_validation_formulas: 11
  outer_formulas_per_fold: 14
  split_unit: composition_id
extractors:
  MatGPT: {checkpoint_sha256: 4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414, frozen: true, hidden_dimension: 768}
  MatterGPT: {checkpoint_sha256: 8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21, frozen: true, hidden_dimension: 768, property_vector: [1.0, 2.0]}
extraction:
  states: [L0_INPUT, L1_BLOCK_0, L2_BLOCK_1, L3_BLOCK_2, L4_BLOCK_3, L5_BLOCK_4, L6_BLOCK_5, L7_BLOCK_6, L8_BLOCK_7, L9_BLOCK_8, L10_BLOCK_9, L11_BLOCK_10, L12_BLOCK_11, L12_PRE_LN_F, FINAL_POST_LN_F]
  token_policy: last_valid_structural_terminal_token
  dtype: float32_cache
  final_cache_match: exact_numeric_match_against_historical
probes:
  pairwise_ridge:
    intercept: false
    alpha_grid: [0.01, 0.1, 1, 10, 100, 1000]
  pairwise_rbf_kernel_ridge:
    gamma_multipliers: [0.25, 1, 4]
    lambda_grid: [0.001, 0.01, 0.1, 1]
  geometry_additive_kernel:
    eta_grid: [0, 0.1, 1, 10]
  selection: minimum_inner_validation_signed_pair_deltaE_MAE
  tie_break: [higher_validation_pairwise_accuracy, greater_regularization, shallower_layer]
  final_refit: inner_train_only_45_formulas
geometry:
  source: gga-static_endpoint_cleaned
  descriptors: [a_reduced_shape, b_reduced_shape, c_reduced_shape, alpha_reduced_deg, beta_reduced_deg, gamma_reduced_deg, volume_per_atom_ang3]
  scaler_fit: inner_train_only
targets:
  pair_delta: signed_1000_times_Ej_minus_Ei_eV_per_atom
  primary_estimand: macro_average_formula_mean_of_three_absolute_pair_errors_then_mean_over_70_formulas
  pairwise_eligible_threshold_meV_atom: 5
  unresolved_prediction_threshold_meV_atom: 1
  ground_state_dft_gap_threshold_meV_atom: 1
ridge_m2:
  frozen: true
  source: experiments/mattergpt_abs3_frozen_ridge_mlp8/predicciones_oof_por_estructura.csv
  role: absolute_energy_composition_anchor_only
statistics:
  bootstrap_replicates: 10000
  bootstrap_seed: 20261103
  permutation_replicates_per_model_probe_fold_layer_role: 100
  permutation_seed: 20261103
safety:
  test_accessed: false
  transformer_weights_modified: false
"""
    (EXP / "config.yaml").write_text(config, encoding="utf-8")
    selected_metric_rows = metrics[metrics.layer.isin(["FINAL", "BEST_LAYER"])].copy()
    summary = {"status": "COMPLETE_WITH_LIMITATIONS", "cohort": {"compositions": 70, "structures": 210,
               "polymorphs_per_composition": 3, "folds": 5,
               "sizes": {"inner_train": 45, "inner_validation": 11, "outer": 14}},
               "layer_selection_protocol": "Por fold/model/probe: mínimo MAE ΔE en inner-validation; desempates por ranking, regularización y menor profundidad; refit con inner-train.",
               "test_accessed": False, "backbones_modified": False,
               "ranking_90pct_reached": bool(metrics.pairwise_accuracy.max() >= .90),
               "selected_layers": json.loads(selections.to_json(orient="records")),
               "terminal_reproduction": "FINAL coincide exactamente con las cachés históricas; ver token_alignment_audit.json.",
               "selected_procedure_metrics": json.loads(selected_metric_rows.to_json(orient="records")),
               "limitations": ["BEST_LAYER se selecciona por validación; su OOF es evaluación de desarrollo CV, no test externo.",
                              "Los probes de todas las capas son diagnósticos descriptivos; solo BEST_LAYER estima el procedimiento de selección.",
                              "Ridge M2 composicional permanece fijo para anclar energía absoluta.",
                              "Los kernels aditivos y contrastes OOF son exploratorios."]}
    atomic_json(EXP / "summary.json", summary)
    # El análisis previo terminó entre los timestamps de los artefactos iniciales y de permutación;
    # conservar duración exacta de extracción y separar el tiempo medido de esta finalización.
    extract_resources = json.loads((EXP / "extract_resources.json").read_text(encoding="utf-8"))
    finalizer_seconds = time.perf_counter() - started
    first_phase = os.path.getmtime(EXP / "within_between_variance.csv")
    last_phase = os.path.getmtime(EXP / "permutation_controls.csv")
    phase_elapsed_lower_bound = max(0.0, last_phase - first_phase)
    resource_obj = {"status": "analysis_outputs_present_finalization_completed", "device": "cpu", "cpu_threads": 1,
                    "test_accessed": False, "backbones_modified": False,
                    "extraction_wall_seconds": extract_resources["wall_seconds"],
                    "finalization_wall_seconds": finalizer_seconds,
                    "layerwise_analysis_wall_seconds": None,
                    "layerwise_analysis_elapsed_lower_bound_seconds": phase_elapsed_lower_bound,
                    "layerwise_analysis_wall_note": "El proceso principal no dejó resources.json. El intervalo entre el primer artefacto de fase analítica y el CSV final de permutaciones es una cota inferior basada en mtime, no el tiempo exacto del proceso.",
                    "peak_rss_bytes_finalizer": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "permutations_per_model_probe_fold_layer": int(perms.permutation.max() + 1),
                    "bootstrap_replicates": BOOT_N, "bootstrap_seed": BOOT_SEED,
                    "extraction_cpu_threads": extract_resources.get("cpu_threads", 1)}
    atomic_json(EXP / "resources.json", resource_obj)
    LOG.info("Paquete finalizado: métricas=%d selecciones=%d permutaciones=%d", len(metrics), len(selections), len(perms))


if __name__ == "__main__":
    main()
