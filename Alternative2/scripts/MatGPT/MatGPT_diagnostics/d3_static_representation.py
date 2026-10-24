#!/usr/bin/env python3
"""Compara distinción estática de SLICES, tokens, embeddings y geometría."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from pymatgen.core import Element

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development, sha256, write_json

LOG = logging.getLogger("matgpt_diagnostics.d3")
SOURCE = ALT / "models/MatGPT-source/mat-concat"
CHECKPOINT = SOURCE / "mat-uncond/weights/transformer_property_0.pth"
INFO = CHECKPOINT.with_name("model_info_property_0.json")
EXPECTED_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
E2_SCRIPT = Path(__file__).resolve().parents[1] / "MatGPT_finetuning/matgpt_e2/finetune.py"


def import_model_helpers():
    """Importa las rutinas de carga/forward ya auditadas de E2-A."""
    sys.path.insert(0, str(E2_SCRIPT.parent))
    import finetune  # type: ignore[import-not-found]
    return finetune


def periodic_geometry(row: pd.Series) -> dict[str, float]:
    """Calcula descriptores invariantes al orden y rotación desde celda/posiciones.

    Distancias con imagen periódica mínima, volumen por átomo y coordinación media
    con cortes fijos. Los cortes son descriptores geométricos auxiliares, no labels.
    """
    cell = np.asarray(json.loads(row.cell_json), dtype=float)
    arrays = json.loads(row.arrays_json)
    positions = np.asarray(arrays["positions"], dtype=float)
    numbers = np.asarray(arrays["numbers"], dtype=int)
    if cell.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Geometría inválida: {row.structure_id}")
    if len(positions) != len(numbers) or not np.isfinite(positions).all():
        raise ValueError(f"Posiciones/especies no compatibles: {row.structure_id}")
    inv_cell = np.linalg.inv(cell)
    distances: list[float] = []
    counts = {cutoff: 0 for cutoff in (2.5, 3.0, 3.5, 4.0)}
    pair_species_distances: dict[str, list[float]] = {}
    # Las imágenes -1,0,1 en cada eje bastan para celdas periódicas razonables de este set.
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    for i in range(len(positions)):
        for j in range(len(positions)):
            frac = (positions[j] - positions[i]) @ inv_cell
            candidate_frac = frac[None, :] + shifts
            if i == j:
                candidate_frac = candidate_frac[np.any(shifts != 0, axis=1)]
            candidate_cart = candidate_frac @ cell
            distance = float(np.linalg.norm(candidate_cart, axis=1).min())
            if distance <= 6.0:
                distances.append(distance)
                pair_key = "_".join(sorted((Element.from_Z(int(numbers[i])).symbol, Element.from_Z(int(numbers[j])).symbol)))
                pair_species_distances.setdefault(pair_key, []).append(distance)
            for cutoff in counts:
                if distance < cutoff:
                    counts[cutoff] += 1
    if not distances:
        raise ValueError(f"No se encontraron distancias periódicas para {row.structure_id}")
    d = np.asarray(distances)
    volume = float(abs(np.linalg.det(cell)))
    mass_amu = float(sum(float(Element.from_Z(int(z)).atomic_mass) for z in numbers))
    out = {"structure_id": str(row.structure_id), "composition_group": str(row.composition_group),
           "reduced_formula": str(row.reduced_formula), "polymorph_family": str(row.polymorph_family),
           "n_atoms": int(len(numbers)), "volume_per_atom_ang3": volume / len(numbers),
           "density_g_cm3": float(mass_amu * 1.66053906660 / volume),
           "pair_distance_count_le_6A": int(len(d)), "pair_distance_mean_angstrom": float(d.mean()),
           "pair_distance_std_angstrom": float(d.std()), "pair_distance_q10_angstrom": float(np.quantile(d, .10)),
           "pair_distance_median_angstrom": float(np.median(d)), "pair_distance_q90_angstrom": float(np.quantile(d, .90))}
    out.update({f"pair_count_below_{cutoff:.1f}A_per_atom": float(value / len(numbers)) for cutoff, value in counts.items()})
    for pair_key, pair_values in pair_species_distances.items():
        values = np.asarray(pair_values, dtype=float)
        out[f"pair_{pair_key}_mean_angstrom"] = float(values.mean())
        out[f"pair_{pair_key}_std_angstrom"] = float(values.std())
        out[f"pair_{pair_key}_count_per_atom"] = float(len(values) / len(numbers))
    return out


def embed(data: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extrae end y masked-mean congelados del checkpoint original MatGPT."""
    if not CHECKPOINT.is_file() or not INFO.is_file():
        raise FileNotFoundError("Falta checkpoint o model_info unconditional de MatGPT.")
    checkpoint_sha = sha256(CHECKPOINT)
    if checkpoint_sha != EXPECTED_SHA:
        raise RuntimeError(f"Checkpoint SHA inesperado: {checkpoint_sha}")
    helper = import_model_helpers()
    model, stoi = helper.setup()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("El backbone congelado contiene parámetros entrenables.")
    sequences = [helper.encode(value, stoi, device) for value in data.matgpt_slices.astype(str)]
    max_len = int(json.loads(INFO.read_text())["block_size"])
    if max(map(len, sequences)) > max_len:
        raise ValueError("Una secuencia excede block_size del checkpoint.")
    pad, end, start = stoi["<pad>"], stoi["<end>"], stoi["<start>"]
    max_width = max(map(len, sequences))
    ids = torch.full((len(sequences), max_width), pad, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    if not torch.isfinite(model.token_embedding_table(ids)).all():
        raise RuntimeError("Embedding de tokens contiene NaN/Inf.")
    with torch.inference_mode():
        batch_size = 8
        end_vectors, mean_vectors = [], []
        for lo in range(0, len(ids), batch_size):
            xb = ids[lo:lo+batch_size]
            b, t = xb.shape
            x = model.token_embedding_table(xb) + model.position_embedding_table(torch.arange(t, device=device))
            x = x.permute(1, 0, 2)
            mask = model.generate_square_subsequent_mask(t).to(device)
            memory = torch.zeros((t, b, model.n_embd), device=device)
            for layer in model.transformer_decoder.layers:
                x = layer(x, memory, tgt_mask=mask)
            hidden = model.ln_f(x.permute(1, 0, 2))
            tokens = xb.eq(end)
            if not torch.all(tokens.sum(dim=1).eq(1)):
                raise RuntimeError("Token <end> no es único en cada SLICES.")
            positions = tokens.long().argmax(dim=1)
            end_vectors.append(hidden[torch.arange(b, device=device), positions].cpu().numpy())
            valid = xb.ne(pad) & xb.ne(start) & xb.ne(end)
            if not torch.all(valid.any(dim=1)):
                raise RuntimeError("SLICES sin token estructural para masked mean.")
            mean_vectors.append((hidden * valid.unsqueeze(-1)).sum(dim=1).div(valid.sum(dim=1, keepdim=True)).cpu().numpy())
    end_vec = np.concatenate(end_vectors); mean_vec = np.concatenate(mean_vectors)
    # Auditoría de padding derecho: compara el end vector más corto solo vs con padding del lote.
    short_idx = int(np.argmin([len(x) for x in sequences]))
    short = torch.tensor([sequences[short_idx] + [pad] * (max_width - len(sequences[short_idx]))], device=device)
    single_width = len(sequences[short_idx])
    with torch.inference_mode():
        def one_forward(xb: torch.Tensor) -> torch.Tensor:
            b, t = xb.shape
            x = model.token_embedding_table(xb) + model.position_embedding_table(torch.arange(t, device=device))
            x = x.permute(1, 0, 2); mask = model.generate_square_subsequent_mask(t).to(device)
            mem = torch.zeros((t, b, model.n_embd), device=device)
            for layer in model.transformer_decoder.layers: x = layer(x, mem, tgt_mask=mask)
            h = model.ln_f(x.permute(1, 0, 2))
            return h[0, single_width-1]
        padded_h = one_forward(short)
        unpadded_h = one_forward(torch.tensor([sequences[short_idx]], device=device))
        padding_error = float((padded_h-unpadded_h).abs().max().cpu())
    if padding_error > 1e-5:
        raise RuntimeError(f"Embedding <end> depende del padding: error={padding_error:.3g}")
    rows = []
    for i, row in data.reset_index(drop=True).iterrows():
        record = {"structure_id": str(row.structure_id), "composition_group": str(row.composition_group),
                  "reduced_formula": str(row.reduced_formula), "polymorph_family": str(row.polymorph_family),
                  "energy_per_atom_ev": float(row.energy_per_atom_ev), "sequence_sha256": hashlib.sha256(str(row.matgpt_slices).encode()).hexdigest(),
                  "end_embedding_sha256": hashlib.sha256(end_vec[i].tobytes()).hexdigest()}
        record.update({f"end_{j:03d}": float(x) for j, x in enumerate(end_vec[i])})
        record.update({f"mean_{j:03d}": float(x) for j, x in enumerate(mean_vec[i])})
        rows.append(record)
    return pd.DataFrame(rows), {"checkpoint_sha256": checkpoint_sha, "strict_load": True,
                                "backbone_trainable_parameters": 0, "poolings": ["end", "masked_mean"],
                                "padding_end_max_abs_error": padding_error, "device": str(device),
                                "block_size": max_len, "embedding_dimension": int(end_vec.shape[1])}


def pair_rows(data: pd.DataFrame, embeddings: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Construye pares dentro de composición para representaciones y energía."""
    merged = data.merge(embeddings, on=["structure_id", "composition_group", "reduced_formula", "polymorph_family", "energy_per_atom_ev"], validate="one_to_one")
    merged = merged.merge(geometry, on=["structure_id", "composition_group", "reduced_formula", "polymorph_family"], validate="one_to_one")
    rows = []
    geom_cols = [c for c in geometry.columns if c.endswith("angstrom") or c.endswith("ang3")
                 or c.startswith("pair_count_") or c.endswith("count_per_atom")]
    for composition, group in merged.groupby("composition_group"):
        records = list(group.to_dict("records"))
        for i, j in itertools.combinations(range(len(records)), 2):
            a, b = records[i], records[j]
            item: dict[str, object] = {"composition_group": composition,
                "formula": a["reduced_formula"], "structure_id_a": a["structure_id"], "structure_id_b": b["structure_id"],
                "family_a": a["polymorph_family"], "family_b": b["polymorph_family"],
                "dft_abs_difference_mev_atom": abs(a["energy_per_atom_ev"]-b["energy_per_atom_ev"])*1000,
                "slices_string_equal": a["sequence_sha256"] == b["sequence_sha256"],
                "end_embedding_cosine": float(np.dot([a[f"end_{k:03d}"] for k in range(768)], [b[f"end_{k:03d}"] for k in range(768)]) /
                    (np.linalg.norm([a[f"end_{k:03d}"] for k in range(768)])*np.linalg.norm([b[f"end_{k:03d}"] for k in range(768)])+1e-12)),
                "end_embedding_euclidean": float(np.linalg.norm(np.array([a[f"end_{k:03d}"] for k in range(768)])-np.array([b[f"end_{k:03d}"] for k in range(768)]))),
                "mean_embedding_cosine": float(np.dot([a[f"mean_{k:03d}"] for k in range(768)], [b[f"mean_{k:03d}"] for k in range(768)]) /
                    (np.linalg.norm([a[f"mean_{k:03d}"] for k in range(768)])*np.linalg.norm([b[f"mean_{k:03d}"] for k in range(768)])+1e-12)),
                "mean_embedding_euclidean": float(np.linalg.norm(np.array([a[f"mean_{k:03d}"] for k in range(768)])-np.array([b[f"mean_{k:03d}"] for k in range(768)])))}
            item["input_ids_equal"] = a["input_ids_sha256"] == b["input_ids_sha256"]
            geom_diff = []
            missing_geometry = 0
            available_geometry = 0
            for col in geom_cols:
                av, bv = float(a[col]), float(b[col])
                if np.isfinite(av) or np.isfinite(bv):
                    available_geometry += 1
                if np.isfinite(av) and np.isfinite(bv):
                    item[f"absdiff_{col}"] = abs(av-bv)
                    geom_diff.append(abs(av-bv))
                else:
                    item[f"absdiff_{col}"] = np.nan
                    missing_geometry += 1
            item["geometry_descriptor_l1"] = float(np.mean(geom_diff)) if geom_diff else np.nan
            item["geometry_shared_feature_fraction"] = float(len(geom_diff)/max(1,available_geometry))
            item["geometry_missing_feature_count"] = int(missing_geometry)
            rows.append(item)
    return pd.DataFrame(rows)


def run() -> None:
    """Genera auditorías de cadena, tokens, embeddings y descriptores geométricos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output", type=Path, default=REPORT / "D3")
    args = parser.parse_args()
    data, _, meta = load_development()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "mps" if args.device == "auto" and torch.backends.mps.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    for required in ("matgpt_slices", "arrays_json", "cell_json", "topology_sha256"):
        if required not in data:
            raise ValueError(f"Falta columna de D3: {required}")
    # Token ID hash uses the exact unconditional vocabulary, including start/end and no energy tokens.
    helper = import_model_helpers(); stoi = json.loads(INFO.read_text())["stoi"]
    token_hashes = []
    for value in data.matgpt_slices.astype(str):
        token_ids = helper.encode(value, stoi, device)
        token_hashes.append(hashlib.sha256(np.asarray(token_ids, dtype=np.int64).tobytes()).hexdigest())
    data = data.copy(); data["input_ids_sha256"] = token_hashes
    embeddings, emb_meta = embed(data, device)
    geometry = pd.DataFrame([periodic_geometry(row) for _, row in data.iterrows()])
    args.output.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(args.output / "frozen_embeddings.csv", index=False)
    geometry.to_csv(args.output / "geometry_descriptors.csv", index=False)
    data[["structure_id", "composition_group", "reduced_formula", "polymorph_family", "energy_per_atom_ev", "matgpt_slices", "input_ids_sha256", "topology_sha256", "strategy3_graph_equivalent"]].to_csv(args.output / "static_slices_inventory.csv", index=False)
    pairs = pair_rows(data, embeddings, geometry)
    pairs.to_csv(args.output / "embedding_pair_distances.csv", index=False)
    # SLICES/input collision report summarizes three physical structures per formula group.
    collision_rows = []
    for comp, group in data.groupby("composition_group"):
        ids_eq = group.input_ids_sha256.nunique(); seq_eq = group.matgpt_slices.astype(str).nunique()
        topo_eq = group.topology_sha256.nunique(dropna=False)
        collision_rows.append({"composition_group": comp, "reduced_formula": group.reduced_formula.iloc[0],
                               "n_structures": len(group), "n_slices_strings": seq_eq, "n_input_id_sequences": ids_eq,
                               "n_topology_hashes": topo_eq, "all_same_slices": seq_eq == 1,
                               "all_same_input_ids": ids_eq == 1, "all_same_topology_hash": topo_eq == 1,
                               "dft_energy_range_mev_atom": float(group.energy_per_atom_ev.max()-group.energy_per_atom_ev.min())*1000,
                               "graph_equivalence_flags_all_true": bool(group.strategy3_graph_equivalent.fillna(False).all())})
    collisions = pd.DataFrame(collision_rows); collisions.to_csv(args.output / "collision_analysis.csv", index=False)
    correlations = {}
    for col in ("end_embedding_euclidean", "end_embedding_cosine", "mean_embedding_euclidean", "mean_embedding_cosine", "geometry_descriptor_l1"):
        rho = spearmanr(pairs[col], pairs.dft_abs_difference_mev_atom)
        correlations[col] = {"spearman_rho": float(rho.statistic), "p_value_exploratory": float(rho.pvalue), "n_pairs": len(pairs)}
    summary = {"dataset": meta, "embedding_audit": emb_meta,
               "n_compositions": int(len(collisions)), "n_pairs": int(len(pairs)),
               "slices_distinct_counts": {str(k): int(v) for k,v in collisions.n_slices_strings.value_counts().sort_index().items()},
               "input_ids_distinct_counts": {str(k): int(v) for k,v in collisions.n_input_id_sequences.value_counts().sort_index().items()},
               "same_slices_but_dft_range_ge_1mev": int(((collisions.all_same_slices) & (collisions.dft_energy_range_mev_atom >= 1)).sum()),
               "same_input_ids_but_dft_range_ge_1mev": int(((collisions.all_same_input_ids) & (collisions.dft_energy_range_mev_atom >= 1)).sum()),
               "spearman_vs_abs_dft_pair_gap": correlations, "static_topology_claim": "topology_sha256 is reported as provided by the dataset; no independent graph-isomorphism proof was run",
               "geometric_cutoffs_angstrom": [2.5, 3.0, 3.5, 4.0, 6.0], "historical_test_used": False}
    write_json(args.output / "d3_summary.json", summary)
    LOG.info("d3_complete structures=%d compositions=%d pairs=%d device=%s", len(data), len(collisions), len(pairs), device)


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
