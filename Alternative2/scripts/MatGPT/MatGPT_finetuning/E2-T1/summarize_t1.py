#!/usr/bin/env python3
"""Write a concise, provenance-aware E2-T1 report from completed audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(report_dir: Path) -> Path:
    """Build a Markdown summary from the machine-readable T1 decision.

    Args:
        report_dir: E2-T1 report directory containing decision.json.
    Returns:
        Path to E2_T1_report.md.
    """
    decision = json.loads((report_dir / "decision.json").read_text(encoding="utf-8"))
    pbe = decision.get("pbe_metadata", {})
    compat = decision.get("static_relax_metadata_compatibility", {})
    status = decision.get("status", "BLOCKED")
    lines = [
        "# E2-T1 — auditoría de trayectorias de relajación", "",
        f"**Estado: {status}.** Decisión T2: **{decision.get('t2_training_gate', status)}**. "
        "La auditoría excluye test histórico antes de reconstruir y tokenizar frames.", "",
        "## Cobertura y procedencia", "",
        f"- Frames relax en fuente total: {decision.get('n_raw_relax_frames_all_partitions', 'no informado'):,}.",
        f"- Frames development auditados: {decision.get('n_development_relax_frames_audited', 'no informado'):,}.",
        f"- Frames históricos de test excluidos antes de conversión: {decision.get('n_historical_test_relax_frames_excluded_before_structure_conversion', 'no informado'):,}.",
        f"- Trayectorias de desarrollo: {decision.get('n_development_relax_tasks', decision.get('n_relax_tasks', 'no informado'))}.",
        f"- Secuencias de IDs únicas: {decision.get('n_unique_token_sequences', 'no informado')} / {decision.get('n_development_relax_frames_audited', 'no informado')} "
        f"(ratio {decision.get('unique_token_ratio', float('nan')):.4f}).",
        f"- Equivalencia de grafo reserializada: {decision.get('graph_reencode_equivalent', 0)} / {decision.get('graph_reencode_checked', 0)}.",
        f"- Energía raw/cache: máximo |Δ| {decision.get('raw_cache_max_abs_energy_diff_ev_atom', float('nan')):.3g} eV/átomo "
        f"en {decision.get('n_cached_frames_matched_to_raw', 0)} frames coincidentes.",
        f"- Metadatos PBE en relax: {pbe.get('all_pbe_without_u', 'no auditado')}; pareo static/relax: {compat.get('status', 'pendiente')}.",
        f"- Static/relax difieren en ISMEAR en {compat.get('static_relax_ismear_differences', 'no auditado')} de {compat.get('development_static_relax_pairs', 0)} pares; el export no expone descomposición independiente de correcciones. No se combinan energías absolutas.", "",
        "## Colisiones y candidatos", "",
        f"- Grupos de IDs repetidos: {decision.get('n_collision_groups', 0)}; con rango ≥1 meV/átomo: {decision.get('n_collision_groups_range_ge_1mev_atom', 0)}.",
        f"- Rango máximo de colisión: {decision.get('max_collision_range_mev_atom', 0):.3f} meV/átomo.",
        f"- MAE mínima media dentro de grupos repetidos (mediana como predictor): {decision.get('mean_collision_min_mae_mev_atom', 0):.3f} meV/átomo.",
        f"- Candidatos de secuencia única en desarrollo: {decision.get('development_trajectory_candidates', 0)}.",
        f"- Composiciones elegibles por fold: `{json.dumps(decision.get('development_compositions_with_candidates_by_fold', {}), sort_keys=True)}`.",
        f"- Fracción de grupos repetidos con rango ≥1 meV/átomo: {decision.get('collision_groups_range_ge_1mev_fraction', 0):.1%}; umbral operativo de proyecto: 50%.",
        "La cota de colisión corresponde a un predictor determinista que recibe solo IDs; no es un error de optimización. Las diferencias de energía se calculan dentro de cada relajación.", "",
        "## Límites", "",
        "T1 es una auditoría de datos y representabilidad, no una evaluación de rendimiento. Fuerzas y estrés se conservan como descriptores de los frames; no se usan como pérdidas del modelo textual. Los cambios geométricos que no cambian el grafo SLICES no son observables para MatGPT.",
        "El test histórico se excluyó antes de conversión, selección y entrenamiento; tampoco se calcularon métricas supervisadas de test.", "",
        "Artefactos: `frame_tokenization.csv`, `trajectory_manifest.csv`, `collision_summary.csv`, `trajectory_statistics.csv`, `selection_candidates.csv`, `metadata_compatibility.json` y `figures/`.", "",
    ]
    path = report_dir / "E2_T1_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Parse report directory and create the Markdown summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.report_dir))


if __name__ == "__main__":
    main()
