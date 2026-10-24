#!/usr/bin/env python3
"""Audita geometría de embeddings y similitud intra/intercomposición E1.2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_diagnostics.common import ALT, E1_EMBED, E1_REPORT, DiagnosticError, json_write, load_inputs, log, logger_for, sha256_file

LOGGER = logger_for("matgpt_e1_diagnostics")


def diagnostics(data: pd.DataFrame, vectors: np.ndarray, pooling: str, inter_samples: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calcula distancias intra e intercomposición y relación con energía DFT.

    Args:
        data: Dataset en orden idéntico al caché.
        vectors: Embeddings finitos [n, 768].
        pooling: Nombre del método de agregación.
        inter_samples: Número máximo de pares intercomposición muestreados.
        seed: Semilla de muestreo.

    Returns:
        Tabla de pares calculados y resumen cuantitativo.
    """
    cosine = cosine_similarity(vectors)
    euclidean = euclidean_distances(vectors)
    compositions = data.composition.astype(str).to_numpy()
    energies = data.energy_dft_ev_atom.to_numpy(float)
    lengths = data.matgpt_slices.astype(str).map(lambda value: len(value.split())).to_numpy() if "matgpt_slices" in data else np.full(len(data), np.nan)
    norms = np.linalg.norm(vectors, axis=1)
    intra: list[dict[str, Any]] = []
    inter_candidates: list[tuple[int, int]] = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            same = compositions[i] == compositions[j]
            if same:
                intra.append({"pooling": pooling, "pair_scope": "intra_composition", "structure_id_i": str(data.iloc[i].structure_id),
                              "structure_id_j": str(data.iloc[j].structure_id), "composition_i": compositions[i], "composition_j": compositions[j],
                              "family_i": str(data.iloc[i].polymorph_family), "family_j": str(data.iloc[j].polymorph_family),
                              "cosine_similarity": float(cosine[i, j]), "euclidean_distance": float(euclidean[i, j]),
                              "absolute_dft_energy_difference_mev_atom": abs(energies[i] - energies[j]) * 1000,
                              "sequence_length_difference": abs(int(lengths[i]) - int(lengths[j])) if np.isfinite(lengths[i]) and np.isfinite(lengths[j]) else None})
            else:
                inter_candidates.append((i, j))
    rng = np.random.default_rng(seed)
    if len(inter_candidates) > inter_samples:
        selected = rng.choice(len(inter_candidates), size=inter_samples, replace=False)
        inter_candidates = [inter_candidates[int(index)] for index in selected]
    for i, j in inter_candidates:
        intra.append({"pooling": pooling, "pair_scope": "inter_composition_sample", "structure_id_i": str(data.iloc[i].structure_id),
                      "structure_id_j": str(data.iloc[j].structure_id), "composition_i": compositions[i], "composition_j": compositions[j],
                      "family_i": str(data.iloc[i].polymorph_family), "family_j": str(data.iloc[j].polymorph_family),
                      "cosine_similarity": float(cosine[i, j]), "euclidean_distance": float(euclidean[i, j]),
                      "absolute_dft_energy_difference_mev_atom": None,
                      "sequence_length_difference": abs(int(lengths[i]) - int(lengths[j])) if np.isfinite(lengths[i]) and np.isfinite(lengths[j]) else None})
    pairs = pd.DataFrame(intra)
    within = pairs[pairs.pair_scope.eq("intra_composition")]
    between = pairs[pairs.pair_scope.eq("inter_composition_sample")]
    energy_corr_cos = spearmanr(within.cosine_similarity, within.absolute_dft_energy_difference_mev_atom) if len(within) >= 3 else None
    energy_corr_euc = spearmanr(within.euclidean_distance, within.absolute_dft_energy_difference_mev_atom) if len(within) >= 3 else None
    length_corr = spearmanr(pairs.euclidean_distance, pairs.sequence_length_difference) if len(pairs) >= 3 and pairs.sequence_length_difference.notna().all() else None
    norm_corr = spearmanr(norms, lengths) if len(data) >= 3 and np.isfinite(lengths).all() else None
    duplicates = []
    if "matgpt_slices" in data:
        for sequence, group in data.groupby("matgpt_slices", sort=False):
            if group.structure_id.nunique() > 1:
                duplicates.append({"sequence": str(sequence), "structure_ids": group.structure_id.astype(str).tolist(),
                                   "compositions": group.composition.astype(str).tolist(),
                                   "families": group.polymorph_family.astype(str).tolist(),
                                   "energy_ev_atom": group.energy_dft_ev_atom.astype(float).tolist()})
    summary = {"pooling": pooling, "n_structures": len(data), "n_compositions": int(data.composition.nunique()),
               "n_within_composition_pairs": len(within), "n_inter_composition_sampled_pairs": len(between),
               "intra_cosine_mean": float(within.cosine_similarity.mean()) if len(within) else None,
               "intra_cosine_median": float(within.cosine_similarity.median()) if len(within) else None,
               "inter_cosine_mean": float(between.cosine_similarity.mean()) if len(between) else None,
               "inter_cosine_median": float(between.cosine_similarity.median()) if len(between) else None,
               "intra_euclidean_mean": float(within.euclidean_distance.mean()) if len(within) else None,
               "inter_euclidean_mean": float(between.euclidean_distance.mean()) if len(between) else None,
               "embedding_norm_mean": float(norms.mean()), "embedding_norm_median": float(np.median(norms)),
               "embedding_norm_std": float(norms.std()),
               "spearman_embedding_norm_vs_sequence_length": float(norm_corr.statistic) if norm_corr is not None and np.isfinite(norm_corr.statistic) else None,
               "spearman_cosine_vs_abs_dft_delta": float(energy_corr_cos.statistic) if energy_corr_cos is not None and np.isfinite(energy_corr_cos.statistic) else None,
               "spearman_euclidean_vs_abs_dft_delta": float(energy_corr_euc.statistic) if energy_corr_euc is not None and np.isfinite(energy_corr_euc.statistic) else None,
               "spearman_pair_distance_vs_sequence_length_difference": float(length_corr.statistic) if length_corr is not None and np.isfinite(length_corr.statistic) else None,
               "n_duplicate_slices_groups": len(duplicates), "duplicate_slices_groups": duplicates,
               "inter_pair_sampling_seed": seed, "inter_pair_sampling_max": inter_samples,
               "interpretation_limit": "Similitud y distancias son descripciones geométricas; por sí solas no prueban información física ni orden energético."}
    return pairs, summary


def main() -> None:
    """Valida dos cachés y exporta análisis comparativo de geometría."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=E1_REPORT / "dataset.csv")
    parser.add_argument("--last-embeddings", type=Path, default=E1_EMBED / "matgpt_uncond_last_token.npz")
    parser.add_argument("--last-metadata", type=Path, default=E1_EMBED / "embedding_metadata.csv")
    parser.add_argument("--last-audit", type=Path, default=E1_EMBED / "embedding_audit.json")
    parser.add_argument("--mean-embeddings", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "matgpt_uncond_masked_mean.npz")
    parser.add_argument("--mean-metadata", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "embedding_metadata.csv")
    parser.add_argument("--mean-audit", type=Path, default=ALT / "data" / "embeddings" / "matgpt_e1_2_masked_mean_v1" / "embedding_audit.json")
    parser.add_argument("--inter-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ALT / "reports" / "matgpt_e1_2")
    args = parser.parse_args()
    try:
        last_data, last, _ = load_inputs(args.dataset, args.last_embeddings, args.last_metadata, args.last_audit)
        mean_data, mean, _ = load_inputs(args.dataset, args.mean_embeddings, args.mean_metadata, args.mean_audit)
        if last_data.structure_id.astype(str).tolist() != mean_data.structure_id.astype(str).tolist():
            raise DiagnosticError("Orden de IDs distinto entre pooling last-token y mean.")
        if args.inter_pairs < 1:
            raise DiagnosticError("--inter-pairs debe ser positivo.")
        out = args.out_dir.resolve()
        if (out / "embedding_distances.csv").exists() or (out / "embedding_summary.json").exists():
            raise DiagnosticError(f"Ya existe diagnóstico en {out}; usa versión nueva.")
        out.mkdir(parents=True, exist_ok=True)
        last_pairs, last_summary = diagnostics(last_data, last, "last_token", args.inter_pairs, args.seed)
        mean_pairs, mean_summary = diagnostics(mean_data, mean, "masked_mean", args.inter_pairs, args.seed)
        pairs = pd.concat([last_pairs, mean_pairs], ignore_index=True)
        pairs.to_csv(out / "embedding_distances.csv", index=False)
        json_write(out / "embedding_summary.json", {"last_token": last_summary, "masked_mean": mean_summary,
                    "paired_cache_ids_equal": True, "last_embedding_sha256": sha256_file(args.last_embeddings),
                    "mean_embedding_sha256": sha256_file(args.mean_embeddings)})
        log(LOGGER, logging.INFO, "embedding_diagnostics_complete", intra_pairs_per_pooling=last_summary["n_within_composition_pairs"],
            inter_pairs_per_pooling=last_summary["n_inter_composition_sampled_pairs"], report=str(out))
    except (DiagnosticError, OSError, ValueError, RuntimeError, KeyError) as exc:
        log(LOGGER, logging.ERROR, "embedding_diagnostics_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
