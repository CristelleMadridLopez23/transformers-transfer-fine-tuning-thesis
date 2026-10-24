#!/usr/bin/env python3
"""Resume energías, fuerzas, tensiones y diferencias polimórficas del JSON DFT ABS3."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Alternative2" / "01_data" / "raw_data" / "results(1).json"
OUT = ROOT / "Alternative2" / "reports" / "MatGPT_finetuning" / "dft_raw_results_study_v1"
LOGGER = logging.getLogger("dft_rawdata_study")
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
STRESS_KBAR_TO_GPA = 0.1


class StudyError(RuntimeError):
    """Error de estructura o integridad que impide el análisis."""


def canonical_formula(raw: str) -> str:
    """Reduce una fórmula plana a clave química independiente del orden de elementos."""
    text = str(raw).replace(" ", "")
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", text))
    if not text or "".join(m.group(0) for m in matches) != text:
        raise StudyError(f"Fórmula no interpretable sin adivinar: {raw!r}")
    amounts: dict[str, Fraction] = defaultdict(Fraction)
    for match in matches:
        element, number = match.groups()
        amounts[element] += Fraction(number) if number else Fraction(1)
    denominator = math.lcm(*(x.denominator for x in amounts.values()))
    integers = {e: int(x * denominator) for e, x in amounts.items()}
    divisor = math.gcd(*integers.values())
    integers = {e: n // divisor for e, n in integers.items()}
    return "".join(e + (str(n) if n != 1 else "") for e, n in sorted(integers.items()))


def numeric_array(value: Any, expected_ndim: int, field: str) -> np.ndarray | None:
    """Devuelve un array float finito con dimensionalidad verificada o None."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        LOGGER.warning("field_not_numeric field=%s", field)
        return None
    if array.ndim != expected_ndim or not np.isfinite(array).all():
        LOGGER.warning("field_invalid_shape_or_nonfinite field=%s shape=%s", field, array.shape)
        return None
    return array


def extract_record(key: str, record: dict[str, Any]) -> dict[str, Any]:
    """Extrae identidad, DFT energies, fuerzas y stress de una entrada raw."""
    match = re.search(r"--(fm|nm)--gga-(static|relax)$", key)
    family = next((name for name in FAMILIES if f"_{name}--" in key), None)
    if not match or family is None:
        raise StudyError(f"No se puede extraer familia/estado/cálculo de {key!r}")
    result = record.get("results") or {}
    raw_formula = str(result.get("formula") or key.split("--", 1)[0])
    energy = result.get("E_per_at")
    try:
        energy = float(energy)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"E_per_at ausente/no numérico en {key}") from exc
    if not np.isfinite(energy):
        raise StudyError(f"E_per_at no finito en {key}")
    structure = record.get("structure") or {}
    n_atoms = len(structure.get("sites", []))
    forces = numeric_array(record.get("forces"), 2, "forces")
    stress = numeric_array(record.get("stress"), 2, "stress")
    if forces is not None and (forces.shape[1] != 3 or (n_atoms and forces.shape[0] != n_atoms)):
        forces = None
    if stress is not None and stress.shape != (3, 3):
        stress = None
    incar = (record.get("meta") or {}).get("incar") or {}
    return {
        "calculation_id": key,
        "raw_formula": raw_formula,
        "composition_group": canonical_formula(raw_formula),
        "polymorph_family": family,
        "magnetic_state": match.group(1),
        "calculation_type": match.group(2),
        "energy_dft_ev_atom": energy,
        "energy_relax_field_ev_atom": result.get("E_relax"),
        "converged": result.get("convergence"),
        "n_atoms": n_atoms,
        "trajectory_frames": len(record.get("trajectory") or []),
        "force_mae_to_zero_ev_angstrom": float(np.abs(forces).mean()) if forces is not None else np.nan,
        "force_rmse_to_zero_ev_angstrom": float(np.sqrt(np.square(forces).mean())) if forces is not None else np.nan,
        "force_mean_atom_norm_ev_angstrom": float(np.linalg.norm(forces, axis=1).mean()) if forces is not None else np.nan,
        "force_p95_atom_norm_ev_angstrom": float(np.quantile(np.linalg.norm(forces, axis=1), .95)) if forces is not None else np.nan,
        "force_max_atom_norm_ev_angstrom": float(np.linalg.norm(forces, axis=1).max()) if forces is not None else np.nan,
        "stress_mae_to_zero_gpa": float(np.abs(stress).mean() * STRESS_KBAR_TO_GPA) if stress is not None else np.nan,
        "stress_rmse_to_zero_gpa": float(np.sqrt(np.square(stress).mean()) * STRESS_KBAR_TO_GPA) if stress is not None else np.nan,
        "stress_max_abs_component_gpa": float(np.abs(stress).max() * STRESS_KBAR_TO_GPA) if stress is not None else np.nan,
        "hydrostatic_stress_gpa": float(np.trace(stress) / 3 * STRESS_KBAR_TO_GPA) if stress is not None else np.nan,
        "incar_gga": incar.get("GGA"),
        "incar_encut_ev": incar.get("ENCUT"),
        "incar_ediff_ev": incar.get("EDIFF"),
        "incar_ediffg_ev_angstrom": incar.get("EDIFFG"),
        "incar_isif": incar.get("ISIF"),
        "incar_nsw": incar.get("NSW"),
    }


def paired_energy_tables(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula consistencia static-relax y diferencias relativas DFT por composición."""
    pair_keys = ["composition_group", "magnetic_state", "polymorph_family"]
    energy = records.pivot_table(index=pair_keys, columns="calculation_type", values="energy_dft_ev_atom", aggfunc="first").reset_index()
    if not {"static", "relax"}.issubset(energy.columns):
        raise StudyError("No hay datos static y relax para comparar.")
    paired = energy.dropna(subset=["static", "relax"]).copy()
    paired["static_minus_relax_mev_atom"] = (paired["static"] - paired["relax"]) * 1000
    paired["abs_static_relax_difference_mev_atom"] = paired.static_minus_relax_mev_atom.abs()
    paired["delta_e_static_mev_atom"] = paired.groupby("magnetic_state").static.transform(lambda x: (x - x.min()) * 1000)
    paired["delta_e_relax_mev_atom"] = paired.groupby("magnetic_state").relax.transform(lambda x: (x - x.min()) * 1000)
    # Recompute minima only within the same composition and spin state.
    paired["delta_e_static_mev_atom"] = paired.groupby(["composition_group", "magnetic_state"]).static.transform(lambda x: (x - x.min()) * 1000)
    paired["delta_e_relax_mev_atom"] = paired.groupby(["composition_group", "magnetic_state"]).relax.transform(lambda x: (x - x.min()) * 1000)
    paired["abs_delta_e_protocol_difference_mev_atom"] = (paired.delta_e_static_mev_atom - paired.delta_e_relax_mev_atom).abs()
    rows = []
    for (formula, spin), group in records.groupby(["composition_group", "magnetic_state"]):
        for calc_type in ("static", "relax"):
            part = group[group.calculation_type.eq(calc_type)]
            if set(part.polymorph_family) != set(FAMILIES) or part.polymorph_family.duplicated().any():
                continue
            minimum = part.energy_dft_ev_atom.min()
            for row in part.itertuples():
                rows.append({"composition_group": formula, "magnetic_state": spin, "calculation_type": calc_type,
                    "polymorph_family": row.polymorph_family, "energy_dft_ev_atom": row.energy_dft_ev_atom,
                    "delta_e_dft_mev_atom": (row.energy_dft_ev_atom - minimum) * 1000,
                    "is_dft_ground_state": bool(np.isclose(row.energy_dft_ev_atom, minimum, atol=1e-12))})
    return paired, pd.DataFrame(rows)


def metric_summary(values: pd.Series, unit: str) -> dict[str, Any]:
    """Resume distribución con media, mediana, percentiles, máximo y unidad."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(x):
        return {"n": 0, "unit": unit}
    return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p95": float(np.quantile(x, .95)), "max": float(np.max(x)), "unit": unit}


def main() -> None:
    """Audita el JSON real y guarda tablas, metodología y conclusiones en Markdown."""
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUT.exists() and any(OUT.iterdir()):
        raise StudyError(f"El destino ya contiene archivos; conserva resultados y usa una revisión nueva: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    LOGGER.info("loading_json path=%s size_bytes=%d", SOURCE, SOURCE.stat().st_size)
    with SOURCE.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = [extract_record(key, value) for key, value in payload.items()]
    records = pd.DataFrame(rows)
    paired, deltas = paired_energy_tables(records)
    records.to_csv(OUT / "record_level_metrics.csv", index=False)
    paired.to_csv(OUT / "static_relax_energy_consistency.csv", index=False)
    deltas.to_csv(OUT / "dft_polymorph_delta_e.csv", index=False)

    summary: dict[str, Any] = {
        "source": str(SOURCE.resolve()), "source_bytes": SOURCE.stat().st_size,
        "n_records": int(len(records)), "n_compositions": int(records.composition_group.nunique()),
        "counts_by_calculation": {str(k): int(v) for k, v in records.calculation_type.value_counts().items()},
        "counts_by_convergence": {str(k): int(v) for k, v in records.converged.value_counts(dropna=False).items()},
        "unique_magnetic_states": sorted(records.magnetic_state.unique().tolist()),
        "polymorph_families": sorted(records.polymorph_family.unique().tolist()),
        "force_residual_component_mae_to_zero": {}, "force_residual_component_rmse_to_zero": {},
        "stress_residual_component_mae_to_zero": {}, "stress_residual_component_rmse_to_zero": {},
        "energy_dft_ev_atom": metric_summary(records.energy_dft_ev_atom, "eV/átomo"),
        "paired_static_relax_abs_energy_difference": metric_summary(paired.abs_static_relax_difference_mev_atom, "meV/átomo"),
        "paired_static_relax_signed_energy_difference": metric_summary(paired.static_minus_relax_mev_atom, "meV/átomo"),
        "static_relax_relative_delta_e_difference": metric_summary(paired.abs_delta_e_protocol_difference_mev_atom, "meV/átomo"),
        "dft_polymorph_delta_e": {}, "dft_ground_state_family_counts": {},
        "method_settings": {},
    }
    for calc_type, group in records.groupby("calculation_type"):
        summary["force_residual_component_mae_to_zero"][calc_type] = metric_summary(group.force_mae_to_zero_ev_angstrom, "eV/Å; media por cálculo")
        summary["force_residual_component_rmse_to_zero"][calc_type] = metric_summary(group.force_rmse_to_zero_ev_angstrom, "eV/Å; RMS por cálculo")
        summary["stress_residual_component_mae_to_zero"][calc_type] = metric_summary(group.stress_mae_to_zero_gpa, "GPa; media de 9 componentes por cálculo")
        summary["stress_residual_component_rmse_to_zero"][calc_type] = metric_summary(group.stress_rmse_to_zero_gpa, "GPa; RMS de 9 componentes por cálculo")
        summary["method_settings"][calc_type] = {col: sorted(group[col].dropna().astype(str).unique().tolist()) for col in ("incar_gga", "incar_encut_ev", "incar_ediff_ev", "incar_ediffg_ev_angstrom", "incar_isif", "incar_nsw")}
    for calc_type, group in deltas.groupby("calculation_type"):
        summary["dft_polymorph_delta_e"][calc_type] = metric_summary(group.delta_e_dft_mev_atom, "meV/átomo sobre mínimo DFT de composición")
        ground = group[group.is_dft_ground_state].groupby("polymorph_family").composition_group.nunique()
        summary["dft_ground_state_family_counts"][calc_type] = {str(k): int(v) for k, v in ground.items()}
    summary["residual_threshold_counts"] = {}
    for calc_type, group in records.groupby("calculation_type"):
        summary["residual_threshold_counts"][calc_type] = {
            "max_atom_force_gt_0.03_ev_angstrom": int(group.force_max_atom_norm_ev_angstrom.gt(.03).sum()),
            "max_atom_force_gt_0.05_ev_angstrom": int(group.force_max_atom_norm_ev_angstrom.gt(.05).sum()),
            "max_abs_stress_component_gt_0.10_gpa": int(group.stress_max_abs_component_gpa.gt(.10).sum()),
            "max_abs_stress_component_gt_0.25_gpa": int(group.stress_max_abs_component_gpa.gt(.25).sum()),
            "stress_tensor_rmse_gt_0.25_gpa": int(group.stress_rmse_to_zero_gpa.gt(.25).sum()),
        }
    ground_matches = 0
    n_composition_groups = 0
    pair_order_matches: list[bool] = []
    for _, group in paired.groupby(["composition_group", "magnetic_state"]):
        if len(group) != 3:
            continue
        n_composition_groups += 1
        static_min = set(group.loc[group.static.eq(group.static.min()), "polymorph_family"])
        relax_min = set(group.loc[group.relax.eq(group.relax.min()), "polymorph_family"])
        ground_matches += int(bool(static_min & relax_min))
        static_rank = group.static.rank(method="average").to_numpy()
        relax_rank = group.relax.rank(method="average").to_numpy()
        for i in range(3):
            for j in range(i + 1, 3):
                pair_order_matches.append(bool(np.sign(static_rank[i] - static_rank[j]) == np.sign(relax_rank[i] - relax_rank[j])))
    summary["static_relax_ground_family_overlap"] = {"matching_compositions": ground_matches, "n_compositions": n_composition_groups, "fraction": ground_matches / n_composition_groups if n_composition_groups else None}
    summary["static_relax_pair_order_agreement"] = {"matching_pairs": int(sum(pair_order_matches)), "n_pairs": len(pair_order_matches), "fraction": float(np.mean(pair_order_matches)) if pair_order_matches else None}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    energy_consistency = summary["paired_static_relax_abs_energy_difference"]
    relative_consistency = summary["static_relax_relative_delta_e_difference"]
    lines = [
        "# Estudio DFT de `results(1).json` — ABS₃", "",
        f"Archivo fuente: `{SOURCE}`. Registros: **{len(records)}**; composiciones químicas: **{records.composition_group.nunique()}**.", "",
        "## Alcance y lectura correcta", "",
        "El JSON contiene resultados DFT, no predicciones de un MLIP. Por tanto, Energy MAE y Relative polymorph ΔE error del MLIP contra DFT **no se pueden calcular de este archivo por sí solo**. Aquí se informa la consistencia entre cálculos DFT `static` y `relax`, más los residuos de fuerza/tensión de las estructuras finales respecto a cero. Estos valores no son errores de un MLIP.", "",
        "Los valores de fuerza son componentes DFT finales, resumidos frente al objetivo de equilibrio `F=0`; las cifras son medias/RMS por estructura y su distribución entre estructuras. El stress se resume por tensor frente a cero y se convierte de kbar a GPa. VASP reporta fuerzas en eV/Å y stress en kB/kbar en `vasprun.xml` ([VASP: vasprun.xml](https://vasp.at/wiki/Vasprun.xml)); `ISIF` determina si se calcula stress y qué grados de libertad se relajan ([VASP: ISIF](https://vasp.at/wiki/ISIF)).", "",
        "## Resumen frente a las bandas de referencia", "",
        "| Magnitud de la tabla | Resultado del JSON DFT | Comparabilidad con el umbral MLIP |",
        "|---|---:|---|",
        f"| Energy MAE | No disponible como error MLIP–DFT. Diferencia absoluta DFT `static`–`relax`: media **{energy_consistency.get('mean', float('nan')):.3f} meV/átomo**, mediana **{energy_consistency.get('median', float('nan')):.3f}**, P95 **{energy_consistency.get('p95', float('nan')):.3f}**, máximo **{energy_consistency.get('max', float('nan')):.3f}** (n={energy_consistency.get('n',0)} pares). | Consistencia entre energías DFT emparejadas; no es Energy MAE de MLIP. |",
        f"| Force MAE / RMSE | Se reporta el residuo de componentes DFT a cero por `static` y `relax`; ver tabla completa abajo. | No es MAE/RMSE de fuerzas de MLIP frente a DFT. |",
        f"| Stress MAE / RMSE | Se reporta residuo tensorial DFT a cero en GPa; ver tabla completa abajo. | No es error de stress de MLIP frente a DFT. |",
        f"| Relative polymorph ΔE error | No disponible como error MLIP–DFT. Diferencia absoluta entre ΔE DFT `static` y `relax`: media **{relative_consistency.get('mean', float('nan')):.3f} meV/átomo**, mediana **{relative_consistency.get('median', float('nan')):.3f}**, P95 **{relative_consistency.get('p95', float('nan')):.3f}**, máximo **{relative_consistency.get('max', float('nan')):.3f}** (n={relative_consistency.get('n',0)} polimorfos emparejados). | Estabilidad de ΔE entre dos salidas DFT; no accuracy de ranking de MLIP. |", "",
        "## Residuos DFT finales por tipo de cálculo", "",
        "MAE/RMSE aquí significa componente frente a cero, calculado en cada tensor/fuerza final y luego resumido por estructura. No se comparan fuerzas ni tensiones predichas por un potencial.", "",
        "| Cálculo | Registros con fuerza | Force MAE a 0 (media, eV/Å) | Force RMSE a 0 (media RMS, eV/Å) | Media del P95 de norma atómica (eV/Å) | Registros con stress | Stress MAE a 0 (media, GPa) | Stress RMSE a 0 (media RMS, GPa) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for calc_type in ("relax", "static"):
        group = records[records.calculation_type.eq(calc_type)]
        lines.append(f"| {calc_type} | {group.force_mae_to_zero_ev_angstrom.notna().sum()} | {group.force_mae_to_zero_ev_angstrom.mean():.5f} | {group.force_rmse_to_zero_ev_angstrom.mean():.5f} | {group.force_p95_atom_norm_ev_angstrom.mean():.5f} | {group.stress_mae_to_zero_gpa.notna().sum()} | {group.stress_mae_to_zero_gpa.mean():.5f} | {group.stress_rmse_to_zero_gpa.mean():.5f} |")
    lines.extend(["", "Distribución entre los 240 cálculos de cada tipo, en orden **media / mediana / P95 / máximo**:", "", "| Cálculo | Force MAE a 0 (eV/Å) | Force RMSE a 0 (eV/Å) | P95 norma atómica (eV/Å) | Stress MAE a 0 (GPa) | Stress RMSE a 0 (GPa) | Máx. componente stress (GPa) |", "|---|---|---|---|---|---|---|"])
    distribution_columns = [
        ("force_mae_to_zero_ev_angstrom", "force_rmse_to_zero_ev_angstrom", "force_p95_atom_norm_ev_angstrom", "stress_mae_to_zero_gpa", "stress_rmse_to_zero_gpa", "stress_max_abs_component_gpa")]
    for calc_type in ("relax", "static"):
        group = records[records.calculation_type.eq(calc_type)]
        cells = []
        for col in distribution_columns[0]:
            x = group[col].dropna()
            cells.append(" / ".join(f"{v:.5f}" for v in (x.mean(), x.median(), x.quantile(.95), x.max())))
        lines.append("| " + calc_type + " | " + " | ".join(cells) + " |")
    t = summary["residual_threshold_counts"]
    lines.append("")
    lines.append(f"Como contexto de relajación, relax excede norma máxima atómica 0.03 eV/Å: **{t['relax']['max_atom_force_gt_0.03_ev_angstrom']}/240**; static excede 0.03: **{t['static']['max_atom_force_gt_0.03_ev_angstrom']}/240** y 0.05: **{t['static']['max_atom_force_gt_0.05_ev_angstrom']}/240. Stress componente >0.25 GPa: relax **{t['relax']['max_abs_stress_component_gt_0.25_gpa']}/240**, static **{t['static']['max_abs_stress_component_gt_0.25_gpa']}/240**. Estos son residuos DFT, no errores de potencial.")
    lines.extend(["", "## Rangos polimórficos DFT", "", "ΔE se calcula para cada composición y estado magnético como `E_familia − min(E_dist_perovskite, E_hexagonal, E_needle)`. La tabla completa está en `dft_polymorph_delta_e.csv`.", "", "| Cálculo | Polimorfos en composiciones completas | ΔE DFT media (meV/átomo) | Mediana | P95 | Máximo | Ground states por familia |", "|---|---:|---:|---:|---:|---:|---|"])
    for calc_type in ("static", "relax"):
        group = deltas[deltas.calculation_type.eq(calc_type)]
        family_counts = summary["dft_ground_state_family_counts"].get(calc_type, {})
        lines.append(f"| {calc_type} | {len(group)} | {group.delta_e_dft_mev_atom.mean():.3f} | {group.delta_e_dft_mev_atom.median():.3f} | {group.delta_e_dft_mev_atom.quantile(.95):.3f} | {group.delta_e_dft_mev_atom.max():.3f} | `{family_counts}` |")
    lines.append("")
    lines.append(f"Concordancia de polimorfo mínimo DFT static/relax: **{summary['static_relax_ground_family_overlap']['matching_compositions']}/{summary['static_relax_ground_family_overlap']['n_compositions']}** composiciones ({100*summary['static_relax_ground_family_overlap']['fraction']:.1f}%). Orden por pares concordante: **{summary['static_relax_pair_order_agreement']['matching_pairs']}/{summary['static_relax_pair_order_agreement']['n_pairs']}** ({100*summary['static_relax_pair_order_agreement']['fraction']:.2f}%).")
    lines.extend(["", "## Parámetros DFT y limitaciones", "", f"Convergencia registrada: `{summary['counts_by_convergence']}`. Parámetros INCAR observados por cálculo: `{json.dumps(summary['method_settings'], ensure_ascii=False)}`.", "", "Los registros `relax` usan `ISIF=3` y `EDIFFG` como criterio reportado; los `static` tienen `NSW=0`. Las fuerzas/tensiones top-level son solo el último resultado almacenado, no la distribución completa de cada trayectoria. La diferencia `static`–`relax` incluye cualquier cambio de geometría/procedimiento y no debe interpretarse como incertidumbre formal de DFT. Para validar las bandas Energy/Force/Stress MAE de un MLIP hacen falta sus predicciones sobre exactamente estas estructuras y los mismos labels DFT; para ΔE hace falta además comparar pares de polimorfos por composición.", "", "## Artefactos", "", "- `record_level_metrics.csv`: una fila por cálculo DFT.", "- `static_relax_energy_consistency.csv`: comparación emparejada por composición, spin y polimorfo.", "- `dft_polymorph_delta_e.csv`: ΔE de cada polimorfo respecto al mínimo DFT de su composición.", "- `summary.json`: resumen machine-readable.", ""])
    (OUT / "DFT_raw_results_study.md").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("study_complete records=%d compositions=%d output=%s", len(records), records.composition_group.nunique(), OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
