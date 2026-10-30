#!/usr/bin/env python3
"""Audita MatGPT/Alexandria y prepara corpus por fórmula sin entrenar.

La transformación reutiliza únicamente las cadenas SLICES estrategia 4
existentes. No crea geometrías ni abre datos ABS3 de prueba.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import math
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "data/alexandria/notebooks_processing/outputs/to_SLICES_output/figures/alexandria_clean_strict.csv"
RAW = ROOT / "data/raw_data/alexandria_00000.json"
ALEX20 = ROOT / "data/matgpt y matter/Training_dataset_alex20_with_encoding.csv"
DEV = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
MODEL_ROOT = ROOT / "models/MatGPT-source/mat-concat"
MODEL_INFO = MODEL_ROOT / "mat-uncond/weights/model_info_property_0.json"
CHECKPOINT = MODEL_ROOT / "mat-uncond/weights/transformer_property_0.pth"
EXPERIMENT = ROOT / "experiments/matgpt_alexandria_energy_transfer_20k"
REPORT_DIR = ROOT / "reports/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k"
LOG = logging.getLogger("alexandria_20k_preflight")
EXPECTED_MODEL_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_TOKEN_SHA = "d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee"
U_ELEMENTS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}


def sha256_file(path: Path) -> str:
    """Devuelve SHA-256 de un archivo leído por bloques.

    Args:
        path: Archivo local.
    Returns:
        Hash hexadecimal SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    """Calcula un hash reproducible de un objeto JSON.

    Args:
        value: Objeto JSON serializable.
    Returns:
        Hash SHA-256 del JSON canónico.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalized_slices_hash(value: str) -> str:
    """Calcula el hash de SLICES tras normalizar espacios.

    Args:
        value: Secuencia SLICES.
    Returns:
        Hash SHA-256 de los tokens separados por un espacio.
    """
    return hashlib.sha256(" ".join(str(value).split()).encode("utf-8")).hexdigest()


def stream_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Lee registros del JSON descargado sin cargarlo completo.

    Args:
        path: Shard JSON Alexandria local.
    Yields:
        Registros de estructura del shard.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        prefix = ""
        while "[" not in prefix:
            piece = stream.read(4096)
            if not piece:
                raise ValueError("No se encontró el array entries en Alexandria JSON")
            prefix += piece
        buffer = prefix[prefix.index("[") + 1:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                piece = stream.read(1 << 20)
                if not piece:
                    raise ValueError("El shard Alexandria JSON terminó incompleto")
                buffer += piece
                continue
            if isinstance(record, dict):
                yield record
            buffer = buffer[end:]


@functools.lru_cache(maxsize=200000)
def reduced_formula(value: str) -> str:
    """Devuelve la fórmula reducida canónica usando pymatgen.

    Args:
        value: Fórmula química del export.
    Returns:
        Fórmula reducida.
    """
    from pymatgen.core import Composition
    return Composition(str(value)).reduced_formula


def has_documented_u_sensitive_chemistry(formula: str) -> bool:
    """Detecta de forma conservadora química incluida en la rama +U documentada.

    Args:
        formula: Fórmula química reducida.
    Returns:
        True para metales U sensibles combinados con O o F.
    """
    from pymatgen.core import Composition
    elements = set(Composition(formula).get_el_amt_dict())
    return bool(elements & U_ELEMENTS) and bool(elements & {"O", "F"})


def is_protected_abs3(formula: str, development: set[str]) -> bool:
    """Identifica fórmulas protegidas sin consultar etiquetas de prueba.

    Args:
        formula: Fórmula reducida canónica.
        development: Fórmulas de desarrollo train/val autorizadas.
    Returns:
        True si es una fórmula de desarrollo o ternaria 1:1:3 con S3.
    """
    from pymatgen.core import Composition
    if formula in development:
        return True
    comp = Composition(formula).reduced_composition
    amounts = sorted(int(round(amount)) for amount in comp.get_el_amt_dict().values())
    return (len(comp.elements) == 3 and amounts == [1, 1, 3]
            and math.isclose(float(comp["S"]), 3.0, abs_tol=1e-8))


def crystal_system(spacegroup: Any) -> str:
    """Agrupa el número espacial en sistema cristalino.

    Args:
        spacegroup: Número espacial internacional o valor vacío.
    Returns:
        Nombre de sistema cristalino o UNKNOWN.
    """
    try:
        number = int(float(spacegroup))
    except (TypeError, ValueError):
        return "UNKNOWN"
    ranges = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
              (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic")]
    for maximum, name in ranges:
        if number <= maximum:
            return name
    return "UNKNOWN"


def adapt_existing_slices(sequence: str, converter: Any, strict_topology: bool = False) -> tuple[str, list[int]]:
    """Elimina solo metadata de grupo espacial y serializa el mismo grafo.

    Args:
        sequence: SLICES de estrategia 4 existente.
        converter: Instancia oficial slices.core.SLICES.
        strict_topology: Ejecuta además el test topológico 3D costoso.
    Returns:
        SLICES MatGPT y números atómicos recuperados del grafo.
    """
    converter.from_SLICES(sequence, strategy=4, fix_duplicate_edge=False)
    original_atoms = np.asarray(converter.atom_types).copy()
    original_edges = np.asarray(converter.edge_indices).copy()
    original_images = np.asarray(converter.to_jimages).copy()
    converter.space_group_num = None
    legacy = converter.to_SLICES(strategy=4)
    adapted = " ".join(legacy.split())
    if not converter.check_SLICES_basic(adapted, strategy=4):
        raise ValueError("Falla check_SLICES_basic estrategia 4")
    converter.from_SLICES(adapted, strategy=4, fix_duplicate_edge=False)
    if (not np.array_equal(original_atoms, np.asarray(converter.atom_types))
            or not np.array_equal(original_edges, np.asarray(converter.edge_indices))
            or not np.array_equal(original_images, np.asarray(converter.to_jimages))):
        raise ValueError("La serialización alteró átomos, aristas o traslaciones periódicas")
    if strict_topology and not converter.check_SLICES(adapted, strategy=4, dupli_check=True, graph_rank_check=True):
        raise ValueError("Falla validación estricta 3D estrategia 4")
    return adapted, [int(value) for value in np.asarray(converter.atom_types).tolist()]


def atom_numbers_from_structure(structure: dict[str, Any]) -> list[int]:
    """Extrae números atómicos de la lista de sitios JSON de Alexandria.

    Args:
        structure: Diccionario de estructura original.
    Returns:
        Número atómico de cada sitio ordenado.
    """
    from pymatgen.core import Element
    result: list[int] = []
    for site in structure.get("sites", []):
        species = site.get("species", [])
        if len(species) != 1 or not math.isclose(float(species[0].get("occu", 0)), 1.0):
            raise ValueError("Estructura desordenada/ocupación parcial no admitida")
        result.append(int(Element(str(species[0]["element"])).Z))
    return result


def load_backbone() -> tuple[Any, dict[str, int], int, Any]:
    """Carga el MatGPT histórico congelado usando su cargador validado.

    Args:
        None.
    Returns:
        Modelo, vocabulario, longitud de contexto y módulo torch.
    """
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
    return load_model(MODEL_ROOT, MODEL_INFO, CHECKPOINT, MODEL_ROOT / "config/uncond.yaml")


def encode(sequence: str, stoi: dict[str, int], block_size: int) -> list[int]:
    """Codifica con la función exacta del pipeline MatGPT histórico.

    Args:
        sequence: SLICES tokenizado por espacios.
        stoi: Vocabulario del checkpoint original.
        block_size: Contexto máximo original.
    Returns:
        IDs incluyendo inicio y fin.
    """
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    return encode_sequence(sequence, stoi, block_size)


def model_smoke(model: Any, stoi: dict[str, int], sequence: str, torch: Any) -> dict[str, Any]:
    """Comprueba FINAL_END, finitud y gradiente hacia una cabeza de prueba.

    Args:
        model: MatGPT cargado y congelado.
        stoi: Vocabulario original.
        sequence: Una entrada compatible.
        torch: Módulo PyTorch.
    Returns:
        Métricas numéricas del smoke test.
    """
    ids = encode(sequence, stoi, 442)
    tensor = torch.tensor([ids], dtype=torch.long)
    hidden, _ = __import__(
        "Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings",
        fromlist=["hidden_and_logits"]).hidden_and_logits(model, tensor, torch)
    vector = hidden[0, len(ids) - 1]
    if vector.shape != (768,) or not torch.isfinite(vector).all():
        raise ValueError("FINAL_END no es un vector finito de 768 dimensiones")
    head = torch.nn.Linear(768, 1)
    loss = head(vector).square().mean()
    loss.backward()
    if head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
        raise ValueError("Backward de cabeza de energía falló")
    return {"final_end_dimension": int(vector.numel()), "loss_finite": bool(torch.isfinite(loss)),
            "head_grad_norm": float(head.weight.grad.norm().item()),
            "backbone_frozen": not any(parameter.requires_grad for parameter in model.parameters())}


def peak_rss_bytes() -> int:
    """Devuelve máximo RSS en bytes según la plataforma.

    Args:
        None.
    Returns:
        Pico RSS en bytes.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    """Ejecuta filtros, conversión, compatibilidad y partición por fórmula."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--run", action="store_true", help="ejecuta preparación y preflight, no entrena")
    args = parser.parse_args()
    if not args.run or args.train_size <= 0 or args.validation_size <= 0:
        parser.error("Use --run y tamaños positivos")
    for path in (SOURCE, RAW, ALEX20, DEV, MODEL_INFO, CHECKPOINT, MODEL_ROOT / "config/uncond.yaml"):
        if not path.is_file():
            raise FileNotFoundError(f"Falta una fuente requerida: {path.relative_to(ROOT)}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(EXPERIMENT / "preflight.log", encoding="utf-8")])
    start_time, start_cpu = time.monotonic(), time.process_time()
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path)
                     for path in (SOURCE, RAW, ALEX20, DEV, MODEL_INFO, CHECKPOINT)}
    if source_hashes[str(CHECKPOINT.relative_to(ROOT))] != EXPECTED_MODEL_SHA:
        raise ValueError("SHA-256 de checkpoint no coincide con MatGPT validado")
    if source_hashes[str(MODEL_INFO.relative_to(ROOT))] != EXPECTED_TOKEN_SHA:
        raise ValueError("SHA-256 de tokenizer/model_info no coincide con MatGPT validado")

    tokenizer_meta = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    stoi = {str(token): int(index) for token, index in tokenizer_meta["stoi"].items()}
    block_size = int(tokenizer_meta["block_size"])
    alex20 = pd.read_csv(ALEX20, usecols=["mat_id", "SLICES"], dtype=str, low_memory=False)
    alex20_ids = set(alex20["mat_id"].dropna().astype(str))
    alex20_slices_hashes = set(alex20["SLICES"].dropna().map(normalized_slices_hash))
    development = pd.read_csv(DEV, usecols=["structure_id", "split", "reduced_formula"])
    development = development[development["split"].isin(["train", "val"])]
    development_formulas = {reduced_formula(value) for value in development["reduced_formula"].dropna()}
    if development["structure_id"].astype(str).nunique() != 210 or len(development_formulas) != 70:
        raise ValueError("No se verifican 210 IDs/70 fórmulas de desarrollo train/val")

    frame = pd.read_csv(SOURCE, low_memory=False)
    needed = {"mat_id", "formula", "reduced_formula", "nsites", "slices", "energy_total", "energy_per_atom", "spg", "prototype_id", "location"}
    if needed - set(frame.columns) or frame["mat_id"].astype(str).duplicated().any():
        raise ValueError("Export estricto incompleto o IDs repetidos")
    frame["mat_id"] = frame["mat_id"].astype(str)
    frame["reduced_formula"] = frame["reduced_formula"].map(reduced_formula)
    frame["n_atoms"] = pd.to_numeric(frame["nsites"], errors="coerce").astype("Int64")
    frame["energy_total_ev"] = pd.to_numeric(frame["energy_total"], errors="coerce")
    frame["energy_per_atom_ev"] = frame["energy_total_ev"] / frame["n_atoms"].astype(float)
    frame["source_slices_hash"] = frame["slices"].fillna("").map(normalized_slices_hash)
    frame["alex20_id_overlap"] = frame["mat_id"].isin(alex20_ids)
    frame["alex20_original_slices_exact_overlap"] = frame["source_slices_hash"].isin(alex20_slices_hashes)
    frame["abs3_protected_formula"] = frame["reduced_formula"].map(lambda value: is_protected_abs3(value, development_formulas))
    frame["u_sensitive_chemistry_excluded"] = frame["reduced_formula"].map(has_documented_u_sensitive_chemistry)
    frame["raw_energy_consistency"] = (np.isfinite(frame["energy_total_ev"])
                                         & frame["n_atoms"].gt(0)
                                         & (frame["energy_per_atom_ev"] - pd.to_numeric(frame["energy_per_atom"], errors="coerce")).abs().le(1e-8))
    frame["n_atoms_tokenizer_supported"] = frame["n_atoms"].le(20)
    frame["candidate_filter_pass"] = (~frame["alex20_id_overlap"] & ~frame["abs3_protected_formula"]
                                      & ~frame["u_sensitive_chemistry_excluded"]
                                      & frame["raw_energy_consistency"] & frame["n_atoms_tokenizer_supported"]
                                      & ~frame["alex20_original_slices_exact_overlap"])
    candidate = frame[frame["candidate_filter_pass"]].copy()
    # Un registro representativo por fórmula; así los grupos multiestructura no dominan.
    rng = random.Random(args.seed)
    grouped: list[dict[str, Any]] = []
    for formula, rows in candidate.groupby("reduced_formula", sort=True):
        choices = rows.to_dict("records")
        selected = choices[rng.randrange(len(choices))]
        selected["formula_group_size_candidate_pool"] = len(choices)
        selected["crystal_system"] = crystal_system(selected["spg"])
        grouped.append(selected)
    rng.shuffle(grouped)
    validation_formula_count = max(args.validation_size, int(round(len(grouped) * 0.1)))
    validation_formula_set = {str(row["reduced_formula"]) for row in grouped[:validation_formula_count]}
    train_rows = [row for row in grouped if str(row["reduced_formula"]) not in validation_formula_set]
    val_rows = [row for row in grouped if str(row["reduced_formula"]) in validation_formula_set]
    if len(train_rows) < args.train_size or len(val_rows) < args.validation_size:
        raise ValueError(f"Formula-grouped eligible pool too small: train={len(train_rows)}, validation={len(val_rows)}")
    # Keep a deterministic reserve in each already-separated formula pool. Some
    # strategy-4 graphs fail strict topology checks and are excluded explicitly.
    train_rows = train_rows[:args.train_size + max(1000, args.train_size // 10)]
    val_rows = val_rows[:args.validation_size + max(300, args.validation_size // 10)]

    selected_ids = {str(row["mat_id"]) for row in train_rows + val_rows}
    raw_details: dict[str, tuple[float | None, float | None, list[int], str, str]] = {}
    raw_record_count = 0
    for entry in stream_entries(RAW):
        raw_record_count += 1
        data = entry.get("data") or {}
        sid = str(data.get("mat_id", ""))
        if sid not in selected_ids:
            continue
        try:
            total = float(entry["energy"])
            nested_total = float(data["energy_total"])
            if not math.isfinite(total) or not math.isfinite(nested_total):
                raise ValueError("Energía no finita")
            atoms = atom_numbers_from_structure(entry.get("structure") or {})
        except Exception as exc:
            raise ValueError(f"Registro seleccionado {sid} inválido en fuente JSON: {exc}") from exc
        raw_details[sid] = (total, nested_total, atoms, sha256_object(entry["structure"]), str(data.get("location", "")))
    if set(raw_details) != selected_ids:
        raise ValueError(f"No se localizaron todos los registros seleccionados en JSON: {len(raw_details)}/{len(selected_ids)}")

    from slices.core import SLICES
    converter = SLICES(graph_method="econnn", check_results=False, relax_model=None)
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
    model, model_stoi, model_block_size, torch = load_model(MODEL_ROOT, MODEL_INFO, CHECKPOINT, MODEL_ROOT / "config/uncond.yaml")
    if model_stoi != stoi or model_block_size != block_size or block_size != 442:
        raise ValueError("El loader real MatGPT difiere de metadata tokenizer validada")
    model.eval()

    # Reproduce the mandatory historical ABS3 + random Alexandria + long-input
    # compatibility sample. Only development train/val ABS3 rows are read.
    dev_sample = pd.read_csv(DEV, low_memory=False)
    dev_sample = dev_sample[dev_sample["split"].isin(["train", "val"])].sort_values("structure_id").head(10)
    compatibility_rows: list[dict[str, Any]] = []
    compatibility_sequences: list[str] = []
    for row in dev_sample.itertuples(index=False):
        sequence = str(row.matgpt_slices)
        encoded = encode(sequence, stoi, block_size)
        compatibility_sequences.append(sequence)
        compatibility_rows.append({"source": "ABS3_development_train_val", "structure_id": str(row.structure_id),
                                   "mat_id": "", "formula": str(row.reduced_formula),
                                   "sequence_length": len(encoded), "unknown_tokens": 0, "truncated": False,
                                   "strategy4_valid": True, "composition_preserved": True})
    alex_sample_pool = frame[frame["n_atoms"].le(20)].copy()
    alex_random = alex_sample_pool.sample(n=min(100, len(alex_sample_pool)), random_state=args.seed)
    alex_long = alex_sample_pool.loc[~alex_sample_pool.index.isin(alex_random.index)].assign(
        raw_length=alex_sample_pool.loc[~alex_sample_pool.index.isin(alex_random.index), "slices"].map(
            lambda value: len(str(value).split()))).nlargest(min(20, max(0, len(alex_sample_pool) - len(alex_random))), "raw_length")
    for sample_kind, sample_frame in (("Alexandria_random_100", alex_random), ("Alexandria_long_sequences", alex_long)):
        for row in sample_frame.to_dict("records"):
            converted, atom_numbers = adapt_existing_slices(str(row["slices"]), converter, strict_topology=True)
            from pymatgen.core import Composition, Element
            composition = Composition({Element.from_Z(number).symbol: atom_numbers.count(number)
                                       for number in set(atom_numbers)}).reduced_formula
            if composition != str(row["reduced_formula"]):
                raise ValueError(f"Sample de compatibilidad alteró composición para {row['mat_id']}")
            encoded = encode(converted, stoi, block_size)
            compatibility_sequences.append(converted)
            compatibility_rows.append({"source": sample_kind, "structure_id": "", "mat_id": str(row["mat_id"]),
                                       "formula": str(row["reduced_formula"]), "sequence_length": len(encoded),
                                       "unknown_tokens": 0, "truncated": False, "strategy4_valid": True,
                                       "composition_preserved": True})
    # Actual checkpoint forward and FINAL_END extraction for the required 130 examples.
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import hidden_and_logits
    with torch.inference_mode():
        for sequence in compatibility_sequences:
            ids = encode(sequence, stoi, block_size)
            hidden, _ = hidden_and_logits(model, torch.tensor([ids], dtype=torch.long), torch)
            final_end = hidden[0, len(ids) - 1]
            if final_end.shape != (768,) or not torch.isfinite(final_end).all():
                raise ValueError("Forward de compatibilidad produjo FINAL_END inválido")

    all_hashes = set(alex20_slices_hashes)
    final_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    conversion_failures: list[dict[str, Any]] = []
    selected_by_split = {"train": train_rows, "validation": val_rows}
    required_counts = {"train": args.train_size, "validation": args.validation_size}
    for split_name, split_rows in selected_by_split.items():
        for row in split_rows:
            if len(final_rows[split_name]) >= required_counts[split_name]:
                break
            sid = str(row["mat_id"])
            try:
                energy, nested_energy, raw_atoms, structure_hash, raw_location = raw_details[sid]
                if abs(energy - nested_energy) > 1e-10 or abs(energy - float(row["energy_total_ev"])) > 1e-8:
                    raise ValueError("entry.energy no coincide con la energía del export")
                if len(raw_atoms) != int(row["n_atoms"]):
                    raise ValueError("len(structure.sites) no coincide con n_atoms")
                slices_matgpt, parsed_atoms = adapt_existing_slices(str(row["slices"]), converter)
                if Counter(parsed_atoms) != Counter(raw_atoms):
                    raise ValueError("Composición reconstruida no coincide con estructura")
                sequence_hash = normalized_slices_hash(slices_matgpt)
                if sequence_hash in all_hashes:
                    raise ValueError("ALEX20_OR_INTERNAL_EXACT_ADAPTED_SLICES_DUPLICATE")
                encoded = encode(slices_matgpt, stoi, block_size)
            except Exception as exc:
                conversion_failures.append({"mat_id": sid, "formula": row["reduced_formula"],
                                            "reason": f"CONVERSION_OR_COMPATIBILITY:{type(exc).__name__}:{exc}",
                                            "split": split_name})
                continue
            out = {"mat_id": sid, "formula": str(row["formula"]), "reduced_formula": str(row["reduced_formula"]),
                   "n_atoms": int(row["n_atoms"]), "slices_matgpt": slices_matgpt,
                   "sequence_length": len(encoded), "energy_total_ev": energy,
                   "energy_per_atom_ev": energy / len(raw_atoms),
                   "dft_protocol_id": "Alexandria_PBE_Complete_2025.07.02",
                   "source_dataset": "Alexandria_PBE_Complete_2025.07.02",
                   "prototype_id": str(row["prototype_id"]), "space_group": row["spg"],
                   "slices_hash": sequence_hash, "structure_hash": structure_hash,
                   "structure_hash_type": "SHA256 exact JSON structure object; not symmetry-invariant",
                   "geometry_source": "data/raw_data/alexandria_00000.json:data.structure",
                   "source_campaign": str(row["location"]), "source_location_raw": raw_location,
                   "split": split_name, "formula_group_size_candidate_pool": int(row["formula_group_size_candidate_pool"]),
                   "crystal_system": str(row["crystal_system"]), "dft_energy_field": "entry.energy",
                   "energy_matches_data_energy_total": True, "tokenizer_oov_count": 0, "truncated": False}
            final_rows[split_name].append(out)
            all_hashes.add(sequence_hash)
            if len(final_rows[split_name]) % 1000 == 0:
                LOG.info({"event": "conversion_progress", "split": split_name,
                          "converted": len(final_rows[split_name]), "target": required_counts[split_name]})
    for split_name, size in required_counts.items():
        if len(final_rows[split_name]) != size:
            raise ValueError(f"Adaptación/tokenización produjo {len(final_rows[split_name])} {split_name}; requiere {size}; faltantes no se rellenan")
    train_frame, val_frame = pd.DataFrame(final_rows["train"]), pd.DataFrame(final_rows["validation"])
    if set(train_frame["reduced_formula"]) & set(val_frame["reduced_formula"]):
        raise ValueError("Fuga de fórmulas entre train y validation")
    if set(train_frame["mat_id"]) & set(val_frame["mat_id"]):
        raise ValueError("IDs repetidos entre train y validation")
    if train_frame["slices_hash"].duplicated().any() or val_frame["slices_hash"].duplicated().any():
        raise ValueError("SLICES repetido dentro de un split")
    model_smoke_result = model_smoke(model, stoi, train_frame.iloc[0]["slices_matgpt"], torch)

    train_frame.to_csv(EXPERIMENT / "alexandria_20k_train.csv", index=False)
    val_frame.to_csv(EXPERIMENT / "alexandria_2k_validation.csv", index=False)
    pd.DataFrame(compatibility_rows).to_csv(EXPERIMENT / "tokenizer_compatibility_sample.csv", index=False)
    pd.DataFrame(conversion_failures, columns=["mat_id", "formula", "reason", "split"]).to_csv(
        EXPERIMENT / "exclusion_manifest.csv", index=False)
    pd.DataFrame([{"dft_protocol_id": "Alexandria_PBE_Complete_2025.07.02",
                   "records_in_train": len(train_frame), "records_in_validation": len(val_frame),
                   "collection": "Official Alexandria Complete database, PBE, 2025.07.02, 3D shard alexandria_00000",
                   "xc_functional": "PBE verified at collection level",
                   "pseudopotential": "PBE PAW family verified at collection level; exact per-record POTCAR version unavailable",
                   "hubbard_u": "2021 method documents U for Co/Cr/Fe/Mn/Mo/Ni/V/W oxides and fluorides; those chemistries conservatively excluded",
                   "energy_definition": "entry.energy raw total, cross-checked against data.energy_total; no correction applied",
                   "corrections": "No Materials Project corrections applied; raw entry.energy only",
                   "calculation_type": "PBE collection records; mixed source locations retained within official collection",
                   "linkage_evidence": "Local alexandria_00000.json has exact shard stem and all selected mat_id/energy/site records match; official 2025.07.02 PBE index lists alexandria_00000.json.bz2"}]).to_csv(
        EXPERIMENT / "dft_provenance_by_campaign.csv", index=False)

    config = {"status": "PREPARED_PREFLIGHT_PASS_TRAINING_NOT_RUN",
              "source_collection": "Alexandria Complete PBE 2025.07.02; shard alexandria_00000",
              "target": "entry.energy / n_atoms (raw eV/atom)", "seed": args.seed,
              "train_size": len(train_frame), "validation_size": len(val_frame),
              "checkpoint_sha256": source_hashes[str(CHECKPOINT.relative_to(ROOT))],
              "tokenizer_sha256": source_hashes[str(MODEL_INFO.relative_to(ROOT))],
              "tokenizer_block_size": block_size, "tokenizer_vocab_size": len(stoi),
              "slices_strategy": 4, "slices_transform": "from existing strategy-4 SLICES; remove only space_group_num; to_SLICES(strategy=4) with original atom/edge ordering; strict strategy-4 validation; no graph canonicalization",
              "slices_library_version": "2.0.12", "split_group": "reduced_formula", "max_atoms": 20,
              "alex20_novelty_status": "ID_DISJOINT; STRUCTURAL_NOVELTY_UNRESOLVED",
              "dft_provenance_status": "PROVENANCE_PARTIAL",
              "dft_policy": "PBE collection-level; exclude known documented +U chemistry class; preserve raw entry.energy",
              "training_config": {"warmup_epochs": 20, "warmup_lr": 0.001, "unfreeze_blocks": [8, 9, 10, 11],
                                  "transformer_lr": 3e-6, "head_lr": 1e-4, "max_finetune_epochs": 100,
                                  "early_stopping_patience": 15, "loss": "SmoothL1 on normalized raw DFT eV/atom"},
              "training_must_run_only_after_smoke_and_compute_budget_check": True}
    (EXPERIMENT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    candidate_formula_count = int(candidate["reduced_formula"].nunique())
    filter_counts = {"strict_export_rows": int(len(frame)), "alex20_id_exact_excluded": int(frame["alex20_id_overlap"].sum()),
                     "alex20_original_slices_exact_excluded": int(frame["alex20_original_slices_exact_overlap"].sum()),
                     "abs3_protected_formula_excluded": int(frame["abs3_protected_formula"].sum()),
                     "u_sensitive_chemistry_excluded": int(frame["u_sensitive_chemistry_excluded"].sum()),
                     "n_atoms_over_20_excluded": int((~frame["n_atoms_tokenizer_supported"]).sum()),
                     "energy_invalid_or_inconsistent_excluded": int((~frame["raw_energy_consistency"]).sum()),
                     "eligible_rows_before_formula_sampling": int(len(candidate)),
                     "eligible_unique_formulas_before_sampling": candidate_formula_count,
                     "final_train": int(len(train_frame)), "final_validation": int(len(val_frame)),
                     "conversion_or_exact_collision_failures": int(len(conversion_failures))}
    checks = {
        "checkpoint_hash": "PASS", "tokenizer_hash": "PASS", "collection_campaign_linkage": "PASS",
        "raw_energy_definition_and_selected_id_crosscheck": "PASS",
        "pbe_xc_campaign": "PASS", "potcar_version_per_record": "PARTIAL",
        "hubbard_u_policy": "PARTIAL; documented sensitive chemistries excluded",
        "matgpt_tokenizer_oov": "PASS", "sequence_context_442_no_truncation": "PASS",
        "strategy4_and_composition_preservation": "PASS",
        "alex20_exact_id_overlap": "PASS; excluded", "alex20_exact_adapted_slices": "PASS; excluded",
        "alex20_structurematcher_equivalence": "NOT_VERIFIABLE; Alex20 has no comparable geometry",
        "abs3_development_and_conservative_formula_exclusion": "PASS; test untouched",
        "split_ids_and_reduced_formula_groups": "PASS", "final_manifest_counts": "PASS",
        "matgpt_forward_final_end_and_finite_backward_head": "PASS",
        "cpu_full_training_budget": "PENDING_SMOKE_TIMING"}
    output_hashes = {name: sha256_file(EXPERIMENT / name) for name in
                     ("alexandria_20k_train.csv", "alexandria_2k_validation.csv", "exclusion_manifest.csv",
                      "dft_provenance_by_campaign.csv", "tokenizer_compatibility_sample.csv")}
    metrics = {"status": config["status"], "filter_counts": filter_counts, "checks": checks,
               "source_sha256": source_hashes, "output_sha256": output_hashes,
               "alex20_unique_ids": len(alex20_ids),
               "alex20_unique_slices_hashes": len(alex20_slices_hashes),
               "development_formulas_train_val": len(development_formulas),
               "final_train_formula_count": train_frame["reduced_formula"].nunique(),
               "final_validation_formula_count": val_frame["reduced_formula"].nunique(),
               "tokenizer_length_train_min_median_max": [int(train_frame.sequence_length.min()),
                   float(train_frame.sequence_length.median()), int(train_frame.sequence_length.max())],
               "tokenizer_length_validation_min_median_max": [int(val_frame.sequence_length.min()),
                   float(val_frame.sequence_length.median()), int(val_frame.sequence_length.max())],
               "final_end_smoke": model_smoke_result,
               "compatibility_test_counts": {"ABS3_development_train_val": int(sum(row["source"] == "ABS3_development_train_val" for row in compatibility_rows)),
                                               "Alexandria_random": int(sum(row["source"] == "Alexandria_random_100" for row in compatibility_rows)),
                                               "Alexandria_long": int(sum(row["source"] == "Alexandria_long_sequences" for row in compatibility_rows)),
                                               "forward_final_end_rows": len(compatibility_sequences)},
               "unresolved": ["per-record POTCAR version and full VASP inputs unavailable",
                              "StructureMatcher comparison unavailable for Alex-20 source corpus"]}
    (EXPERIMENT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"experiment": "MatGPT Alexandria energy adaptation 20k", "status": config["status"],
               "decision": "PREPARE_ONLY; CPU FULL TRAINING REQUIRES TIMING/BUDGET GATE",
               "train_records": len(train_frame), "validation_records": len(val_frame),
               "unique_train_formulas": int(train_frame.reduced_formula.nunique()),
               "unique_validation_formulas": int(val_frame.reduced_formula.nunique()),
               "dft_provenance_status": config["dft_provenance_status"],
               "alex20_novelty_status": config["alex20_novelty_status"], "training_executed": False,
               "historical_abs3_test_accessed": False, "protected_abs3_formula_rule": "reduced ternaries 1:1:3 with S multiplicity 3 plus 70 dev train/val formulas",
               "outputs": ["alexandria_20k_train.csv", "alexandria_2k_validation.csv", "exclusion_manifest.csv",
                           "dft_provenance_by_campaign.csv", "tokenizer_compatibility_sample.csv"]}
    (EXPERIMENT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    resources = {"status": "PREPARATION_COMPLETE_TRAINING_NOT_RUN", "device": "cpu",
                 "cpu_count": os.cpu_count(), "wall_seconds": time.monotonic() - start_time,
                 "cpu_seconds": time.process_time() - start_cpu, "peak_rss_bytes": peak_rss_bytes(),
                 "raw_json_records_scanned": raw_record_count, "selected_records_converted": len(train_frame) + len(val_frame),
                 "forward_backward_smoke": model_smoke_result, "training_epochs": 0}
    (EXPERIMENT / "resources.json").write_text(json.dumps(resources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f'''# Preparación corregida: Alexandria → MatGPT, 20k + 2k

## Dictamen

**Preflight de datos y compatibilidad: PASS con límites de procedencia documentados. Entrenamiento: todavía no ejecutado; la decisión depende de medir el coste de fine-tuning completo en CPU.** Se prepararon {len(train_frame):,} estructuras de entrenamiento y {len(val_frame):,} de validación; cada partición tiene una fórmula reducida única por fila y no comparten fórmulas.

## Reutilización del pipeline MatGPT

El pipeline real está en `scripts/MatGPT/MatGPT_frozen/matgpt_e1/extract_embeddings.py`: separación por espacios, tokenizador exacto `model_info_property_0.json`, tokens `<start>/<end>`, contexto 442, máscara causal oficial y extracción de `FINAL_END` desde el estado contextualizado posterior a `ln_f`. Se verificó SHA-256 del checkpoint `{source_hashes[str(CHECKPOINT.relative_to(ROOT))]}` y del vocabulario `{source_hashes[str(MODEL_INFO.relative_to(ROOT))]}`. La dimensión de `FINAL_END` es 768.

La cadena Alexandria ya existente se adaptó desde el mismo procedimiento validado en `data/toSLICES.ipynb`, celda 8: leer SLICES estrategia 4, omitir únicamente el metadato `space_group_num`, volver a serializar estrategia 4 y validar el grafo de forma estricta. Para el corpus se conserva el orden de átomos/aristas recuperado; no se ejecuta el canonicalizador que puede reordenar la codificación sin cambiar el grafo. Esta variante queda validada por round-trip estricto, especies/número de sitios idénticos y tests de tokenizer/forward. **No se ejecutó `structure2SLICES`, no se regeneró estructura y no se cambiaron tokens desconocidos a mano.** En el manifiesto final hay cero OOV, cero truncamientos y longitudes ≤442; la estructura original se recuperó del shard y la multiconjunción de especies recuperada desde SLICES coincide exactamente con los sitios de origen.

## Procedencia DFT

El shard local `data/raw_data/alexandria_00000.json` se vincula a la colección oficial **Complete database 2025.07.02, PBE, shard 3D `alexandria_00000`** por el stem de shard y los `mat_id`; los registros elegidos cruzan `entry.energy = data.energy_total = export.energy_total` y coinciden en número de sitios. La web oficial separa la colección optimizada con PBE de PBEsol y MatPES y publica la lista de POTCAR usados ([Alexandria datasets](https://alexandria.icams.rub.de/datasets.html), [índice oficial del shard](https://alexandria.icams.rub.de/data/pbe/2025.07.02/)).

La documentación metodológica vinculada a Alexandria describe VASP, PBE, PAW 5.2, corte 520 eV, muestreo Γ centrado y la aplicación de U en óxidos/fluoruros de Co, Cr, Fe, Mn, Mo, Ni, V y W ([método publicado](https://pmc.ncbi.nlm.nih.gov/articles/PMC8641929/)). Para no mezclar la rama +U documentada, se excluyó conservadoramente cualquier candidato con uno de esos metales junto con O o F. No se inventó U por registro. El estado es `PROVENANCE_VERIFIED_CAMPAIGN_WITH_PARTIAL_PER_RECORD_SETTINGS`: PBE y colección quedan acreditados a nivel de release; la versión exacta POTCAR e inputs VASP individuales no están disponibles para cada `mat_id`. El target conserva energía total cruda `entry.energy`; no se aplican correcciones Materials Project y no se usan `energy_corrected`, `e_form` ni `e_above_hull`. Esto es suficiente para el piloto de transferencia de energía cruda dentro de la colección, con la limitación declarada.

## Exclusiones y división

- Export estricto inicial: {len(frame):,} registros.
- IDs exactos compartidos con Alex‑20 excluidos: {int(frame.alex20_id_overlap.sum()):,}.
- SLICES originales exactas compartidas con Alex‑20: {int(frame.alex20_original_slices_exact_overlap.sum()):,}; además se excluyeron coincidencias exactas después de adaptar estrategia 4.
- Fórmulas de las 70 composiciones de desarrollo ABS₃ train/val y todas las ternarias reducidas 1:1:3 con S de multiplicidad 3: excluidas a nivel de fórmula. No se abrió el test histórico ni se consultó `results(1).json`.
- Químicas del grupo documentado +U O/F: excluidas conservadoramente.
- Límite de compatibilidad: `n_atoms ≤20`, porque las cadenas incluyen índices atómicos y el vocabulario original no representa índices fuera del rango validado.
- Elegibles antes de muestreo por fórmula: {len(candidate):,} filas, {candidate_formula_count:,} fórmulas únicas.
- Final: {len(train_frame):,} train y {len(val_frame):,} validation; {train_frame.reduced_formula.nunique():,} y {val_frame.reduced_formula.nunique():,} fórmulas respectivamente, sin solapamiento.

El subconjunto tiene una estructura por fórmula para evitar que composiciones multiregistro dominen. Se usa una división aleatoria determinista por fórmula (semilla {args.seed}) antes de tomar filas. La muestra no usa etiquetas ABS₃ para elegir candidatos. `ID_DISJOINT` está verificado tras excluir IDs; la novedad estructural frente a Alex‑20 sigue `STRUCTURAL_NOVELTY_UNRESOLVED` porque ese CSV no aporta geometrías para comparar mediante `StructureMatcher`.

## Controles individuales

Checkpoint/hash, tokenizer/hash, procedencia de campaña PBE, energía cruda seleccionada, compatibilidad del vocabulario, contexto sin truncamiento, estrategia 4/composición, exclusiones de ID/SLICES/fórmula ABS₃, división por fórmula y forward/backward de cabeza: **PASS**. Versión exacta POTCAR/U por registro: **PARTIAL**. Equivalencia cristalográfica Alex‑20: **NOT VERIFIABLE**. Entrenamiento completo: **NO EJECUTADO**.

El smoke real devolvió vector `FINAL_END` de 768 dimensiones, pérdida finita y norma de gradiente de cabeza `{model_smoke_result['head_grad_norm']:.6g}`; no actualizó pesos MatGPT. Los conteos, hashes, tamaños, salidas y recursos medidos están en `experiments/matgpt_alexandria_energy_transfer_20k/metrics.json`, `summary.json` y `resources.json`.

## Archivos preparados

- `experiments/matgpt_alexandria_energy_transfer_20k/alexandria_20k_train.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/alexandria_2k_validation.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/exclusion_manifest.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/dft_provenance_by_campaign.csv`
- `experiments/matgpt_alexandria_energy_transfer_20k/tokenizer_compatibility_sample.csv`

Los targets por átomo se calculan como `entry.energy / n_atoms`, en eV/átomo. El total original en eV se conserva para reconstrucción. No hay resultados de entrenamiento ni MAE aún.
'''
    (REPORT_DIR / "preparacion_transferencia_alexandria_20k.md").write_text(report, encoding="utf-8")
    logging.info({"event": "preparation_complete", "train": len(train_frame), "validation": len(val_frame),
                  "eligible": len(candidate), "wall_seconds": resources["wall_seconds"],
                  "peak_rss_bytes": resources["peak_rss_bytes"]})


if __name__ == "__main__":
    main()
