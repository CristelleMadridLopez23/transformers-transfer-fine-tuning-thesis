#!/usr/bin/env python3
"""Genera figuras PNG desde estadísticas del CSV real y OOF de development."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR, AuditError

MODEL_COLORS = {"H0": "#365f91", "H2": "#e1812c", "R3-DROP30": "#4d9b74", "R4-SELECTED-ABSOLUTE": "#b45587"}
MODEL_LABELS = {"H0": "H0", "H2": "H2", "R3-DROP30": "R3-DROP30", "R4-SELECTED-ABSOLUTE": "R4-SELECTED-ABSOLUTE"}


def generate_figures(out_dir: Path) -> list[Path]:
    """Crea figuras de frecuencia y asociación exploratoria.

    Args:
        out_dir: Carpeta de figuras de esta auditoría.

    Returns:
        Rutas de PNG guardadas.
    """
    stats = REPORT_DIR / "corpus_statistics"
    required = [stats / "corpus_schema.json", stats / "element_frequencies.csv",
                stats / "chalcogenide_statistics.json", stats / "abs3_counts_by_reconstructed_split.csv",
                REPORT_DIR / "domain_overlap/abs3_composition_overlap.csv",
                REPORT_DIR / "error_correlations/development_exposure_error_rows.csv",
                REPORT_DIR / "error_correlations/spearman_bootstrap.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AuditError(f"Faltan datos verificados para producir figuras: {missing}")
    import json
    schema = json.loads(required[0].read_text(encoding="utf-8"))
    chalc = json.loads(required[2].read_text(encoding="utf-8"))
    n_rows = int(schema["rows"])
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    # Element tokens: denominator is all occurrences, not materials.
    elements = pd.read_csv(required[1]).head(20).sort_values("atom_occurrences")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(elements.element, elements.atom_occurrences, color="#557a95")
    ax.set(title="Elementos atómicos más frecuentes en el corpus completo", xlabel="Ocurrencias de tokens atómicos", ylabel="Elemento")
    ax.text(.01, -.18, f"N = {n_rows:,} registros SLICES. Top 20 por ocurrencias; el denominador del eje x es el número de tokens atómicos.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "element_frequencies.png"))

    # S/Se/Te categories overlap by design.
    labels = ["Contiene S", "Contiene Se", "Contiene Te", "Contiene S/Se/Te"]
    values = [chalc["rows_with_S"], chalc["rows_with_Se"], chalc["rows_with_Te"], chalc["rows_with_any_S_Se_Te"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=["#a88b45", "#507c9b", "#7d6c9e", "#4d9b74"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set(title="Presencia de calcógenos en el corpus completo", ylabel="Registros SLICES", ylim=(0, max(values) * 1.18))
    ax.text(.01, -.19, f"N = {n_rows:,} registros válidos. Categorías S, Se y Te pueden superponerse; ‘cualquiera’ es la unión.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "chalcogenide_presence.png"))

    # ABX3 record counts by reconstructed partition.
    abx = pd.read_csv(required[3])
    families = ["ABS3", "ABSe3", "ABTe3"]
    splits = ["full", "train_candidate", "validation"]
    full_stats = json.loads((stats / "abs3_corpus_statistics.json").read_text(encoding="utf-8"))
    full_counts = {"ABS3": full_stats["ABS3_rows"], "ABSe3": full_stats["ABSe3_rows"], "ABTe3": full_stats["ABTe3_rows"]}
    matrix = np.array([[full_counts[fam] if split == "full" else int(abx.loc[(abx.family == fam) & (abx.split_candidate == split), "material_rows"].sum())
                        for split in splits] for fam in families])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(families)); width = .24
    for idx, split in enumerate(splits):
        bars = ax.bar(x + (idx - 1) * width, matrix[:, idx], width, label={"full": "CSV completo", "train_candidate": "Train candidato", "validation": "Validación candidata"}[split])
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=8)
    ax.set(xticks=x, xticklabels=families, ylabel="Registros SLICES", title="Estequiometrías ternarias ABX₃ por ámbito")
    ax.legend(frameon=True)
    ax.text(.01, -.2, f"N total: CSV {n_rows:,}; train candidato 275,879; validación candidata 30,654. Split torch.randperm(seed=20260910); no acredita exposición histórica.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "abx3_counts_by_scope.png"))

    # Composition overlap tiers; four mutually exclusive categories.
    overlap = pd.read_csv(required[4])
    dev = overlap[overlap.historical_split.isin(["train", "val"])].copy()
    dev["category"] = np.select(
        [dev.exact_match_train_candidate, dev.exact_match_validation_candidate,
         (~dev.exact_match_full_corpus) & dev.same_elements_other_stoichiometry_rows.gt(0)],
        ["Exacta en train candidato", "Exacta en validación candidata", "Sin exacta; mismos elementos, otra proporción"],
        default="Sin coincidencia exacta o de elementos")
    order = ["Exacta en train candidato", "Exacta en validación candidata", "Sin exacta; mismos elementos, otra proporción", "Sin coincidencia exacta o de elementos"]
    counts = dev.category.value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(order[::-1], counts.reindex(order[::-1]), color="#557a95")
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set(title="Solapamiento composicional con las 70 fórmulas de desarrollo", xlabel="Composiciones (fórmulas reducidas)", xlim=(0, max(counts) * 1.2))
    ax.text(.01, -.17, "El test histórico no aparece en esta figura. Train candidato usa el split aleatorio reconstruido; coincidencia de elementos no implica misma estequiometría ni estructura.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "development_composition_overlap.png"))

    # Structural size and token length are sequence descriptors only.
    records = pd.read_csv(stats / "composition_records.csv", usecols=["n_sites", "slices_length_tokens"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(records.n_sites, bins=35, color="#557a95", edgecolor="white")
    axes[0].set(xlabel="Sitios atómicos por registro", ylabel="Registros", title="Tamaño de celda codificada")
    axes[1].hist(records.slices_length_tokens, bins=40, color="#4d9b74", edgecolor="white")
    axes[1].set(xlabel="Tokens SLICES por registro", ylabel="Registros", title="Longitud de secuencia")
    fig.suptitle("Distribución de tamaño y longitud en el corpus completo")
    fig.text(.05, .005, f"N = {n_rows:,}; histogramas de conteos: 35 bins para sitios y 40 para longitud. Longitud incluye sitios y tokens de aristas; no mide similitud geométrica.", fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, .94))
    made.append(_save(fig, out_dir / "cell_size_and_slices_length.png"))

    # Four development-only scatter panels; share the exact-candidate exposure x.
    errors = pd.read_csv(required[5])
    corr = pd.read_csv(required[6])
    models = ["H0", "H2", "R3-DROP30", "R4-SELECTED-ABSOLUTE"]
    metrics = [("MAE_absolute_mev_atom", "MAE absoluta (meV/átomo)"),
               ("MAE_delta_GS_anchored_mev_atom", "MAE ΔE anclada al GS DFT (meV/átomo)")]
    for metric, ylabel in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, models):
            part = errors[errors.model.eq(model)]
            stats_row = corr[(corr.model == model) & (corr.exposure_measure == "train_candidate_rows_exact_formula") & (corr.outcome == metric)]
            ax.scatter(part.train_candidate_rows_exact_formula, part[metric], s=26, alpha=.78,
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=.3)
            rho = float(stats_row.spearman_rho.iloc[0]) if len(stats_row) else np.nan
            lo = float(stats_row.bootstrap_95_ci_low.iloc[0]) if len(stats_row) else np.nan
            hi = float(stats_row.bootstrap_95_ci_high.iloc[0]) if len(stats_row) else np.nan
            ax.set_title(f"{MODEL_LABELS[model]} | ρ={rho:.2f} [{lo:.2f}, {hi:.2f}] | N={len(part)}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.22)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in axes[-1]: ax.set_xlabel("Filas exactas por fórmula en train candidato")
        fig.suptitle(f"Exposición composicional candidata frente a {ylabel}")
        fig.text(.05, .005, "Un punto = una composición de desarrollo. Intervalo 95% bootstrap por composición, 2,000 réplicas; asociación exploratoria, no causal. Split candidato, no exposición histórica demostrada.", fontsize=8.5)
        fig.tight_layout(rect=(0, .05, 1, .94))
        made.append(_save(fig, out_dir / ("exposure_vs_absolute_error.png" if "absolute" in metric else "exposure_vs_delta_error.png")))

    # Ranking outcomes: rows are models, columns are GS and pairwise metrics.
    ranking_metrics = [("GS_correct", "Ground-state accuracy (fraction)"),
                       ("pairwise_accuracy", "Pairwise accuracy (fraction)")]
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), sharex=True, sharey=True)
    for row_index, model in enumerate(models):
        for col_index, (outcome, label) in enumerate(ranking_metrics):
            ax = axes[row_index, col_index]
            part = errors[errors.model.eq(model)]
            stats_row = corr[(corr.model == model) & (corr.exposure_measure == "train_candidate_rows_exact_formula") & (corr.outcome == outcome)]
            ax.scatter(part.train_candidate_rows_exact_formula, part[outcome], s=25, alpha=.78,
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=.3)
            rho = float(stats_row.spearman_rho.iloc[0]) if len(stats_row) else np.nan
            lo = float(stats_row.bootstrap_95_ci_low.iloc[0]) if len(stats_row) else np.nan
            hi = float(stats_row.bootstrap_95_ci_high.iloc[0]) if len(stats_row) else np.nan
            ax.set_title(f"{model} | ρ={rho:.2f} [{lo:.2f}, {hi:.2f}] | N=70")
            ax.set_ylabel(label); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.22)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[-1]: ax.set_xlabel("Filas exactas por fórmula en train candidato")
    fig.suptitle("Exposición composicional candidata frente al ranking OOF", y=.995)
    fig.text(.05, .005, "Un punto = una composición de desarrollo. Accuracy es proporción [0,1]; IC 95% Spearman bootstrap, 2,000 remuestras. Asociación exploratoria, no causal.", fontsize=8.5)
    fig.tight_layout(rect=(0, .02, 1, .98))
    made.append(_save(fig, out_dir / "exposure_vs_ranking.png"))

    # Exact ABS3 formula frequencies in candidate training rows.
    # The audit table contains all three ABX3 chalcogen families. Keep this
    # figure strictly scoped to sulfides so the counts match the ABS3 label.
    abs3 = pd.read_csv(stats / "abs3_formulas.csv")
    abs3 = abs3[abs3.family.eq("ABS3")].copy()
    assignments = pd.read_csv(stats / "reconstructed_split_assignments.csv", usecols=["row_index", "split_candidate"])
    abs3 = abs3.merge(assignments, on="row_index", validate="one_to_one")
    train_abs3 = abs3[abs3.split_candidate.eq("train_candidate")]
    formula_keys = abs3.formula_reduced_key.drop_duplicates().tolist()
    freq = train_abs3.groupby("formula_reduced_key").size().reindex(formula_keys, fill_value=0).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(freq.index, freq.values, color="#a88b45")
    ax.set(title="Registros ABS₃ exactos por fórmula en train candidato", xlabel="Registros SLICES por fórmula", ylabel="Fórmula reducida")
    ax.text(.01, -.13, f"N = {len(freq)} fórmulas ABS₃ únicas en el CSV completo; train candidato contiene {len(train_abs3)} filas. Split candidato; composición no identifica estructura perovskita.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "abs3_formula_frequency_train_candidate.png"))

    # Symmetric A/B co-occurrence; formula order does not determine A and B roles.
    pairs = train_abs3[["element_a", "element_b"]].copy()
    elements_abs = sorted(set(pairs.element_a) | set(pairs.element_b))
    matrix_pair = pd.DataFrame(0, index=elements_abs, columns=elements_abs)
    for row in pairs.itertuples(index=False):
        matrix_pair.loc[row.element_a, row.element_b] += 1
        matrix_pair.loc[row.element_b, row.element_a] += 1
    fig, ax = plt.subplots(figsize=(max(8, .45 * len(elements_abs)), max(7, .42 * len(elements_abs))))
    mesh = ax.imshow(matrix_pair.to_numpy(), cmap="magma", interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(elements_abs)), elements_abs, rotation=90)
    ax.set_yticks(range(len(elements_abs)), elements_abs)
    ax.set(title="Pares A/B en ABS₃ del train candidato", xlabel="Elemento del par A/B (orden simétrico)", ylabel="Elemento del par A/B (orden simétrico)")
    colorbar = fig.colorbar(mesh, ax=ax); colorbar.set_label("Registros ABS₃ por par (conteo)")
    ax.text(0, -.18, f"N = {len(train_abs3)} registros ABS₃; par A/B no ordenado y matriz simétrica. Split candidato.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "abs3_A_B_cooccurrence.png"))

    # Seen/unseen boxplots use one OOF value per development formula.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    box_data: list[np.ndarray] = []; labels: list[str] = []; positions: list[float] = []; colors: list[str] = []
    x_position = 1.0
    for model in models:
        part = errors[errors.model.eq(model)]
        exact = part.train_candidate_rows_exact_formula.gt(0)
        for seen_flag, label in [(True, "Match exacto"), (False, "Sin match")]:
            values = part.loc[exact.eq(seen_flag), "MAE_absolute_mev_atom"].dropna().to_numpy()
            box_data.append(values); labels.append(f"{model}\n{label}\nN={len(values)}")
            positions.append(x_position); colors.append(MODEL_COLORS[model]); x_position += 1
        x_position += .5
    boxes = ax.boxplot(box_data, positions=positions, widths=.65, patch_artist=True, showfliers=True,
                       medianprops={"color": "black", "linewidth": 1.4})
    for box, color in zip(boxes["boxes"], colors): box.set_facecolor(color); box.set_alpha(.62)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set(ylabel="MAE absoluta OOF (meV/átomo)", title="Error absoluto por coincidencia de fórmula en train candidato")
    ax.grid(axis="y", alpha=.22)
    ax.text(.01, -.31, "Caja = Q1–Q3; línea = mediana; bigotes hasta 1.5×IQR; puntos = valores extremos. Cada punto pertenece a una composición de development; exposición candidata.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    made.append(_save(fig, out_dir / "seen_unseen_absolute_error.png"))

    return made


def _save(fig: plt.Figure, path: Path) -> Path:
    """Guarda y cierra una figura exclusivamente en PNG.

    Args:
        fig: Figura Matplotlib.
        path: Destino PNG.

    Returns:
        Ruta guardada.
    """
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def main() -> None:
    """Genera figuras desde tablas auditadas y reporta sus rutas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR / "figures")
    args = parser.parse_args()
    try:
        for path in generate_figures(args.out_dir):
            print(path)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
