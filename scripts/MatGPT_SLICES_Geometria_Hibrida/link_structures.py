"""Enlaza exactamente los 210 IDs ABS3 de desarrollo con su EXTXYZ original.

Solo analiza los frames indicados por ``source_index`` de train/val. Los demás
frames se saltan como texto; no se parsean etiquetas ni geometrías de test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from pymatgen.core import Composition, Lattice, Structure

LOG = logging.getLogger("abs3_geometry_link")
EXPECTED_SOURCE_SHA256 = "a31f436fd6f44be4f85c5110f8d5ebe37718a5b28eec5453425d6a9724271de4"
SOURCE_LABEL = "gga-static_endpoint_cleaned"


def sha256(path: Path) -> str:
    """Devuelve SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def development_rows(dataset: Path) -> dict[int, dict[str, Any]]:
    """Lee solo los campos geométricos de filas train/val con ID verificable."""
    selected: dict[int, dict[str, Any]] = {}
    ids: set[str] = set()
    with dataset.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in ("train", "val"):
                continue
            index = int(row["source_index"])
            sid = row["structure_id"]
            if index < 0 or index in selected or sid in ids:
                raise ValueError(f"Índice o ID duplicado/inválido: {index}, {sid}")
            ids.add(sid)
            selected[index] = {key: row[key] for key in (
                "structure_id", "source_index", "source_key", "config_id",
                "reduced_formula", "polymorph_family", "num_atoms", "source_file",
                "cell_json", "pbc_json", "arrays_json", "info_json", "split")}
    if len(selected) != 210 or len(ids) != 210:
        raise ValueError(f"Se esperaban 210 IDs físicos de desarrollo; hay {len(selected)}")
    paths = {row["source_file"] for row in selected.values()}
    if paths != {"data/only_hex_new/configurations_only_hex_new.extxyz"}:
        raise ValueError(f"Procedencia heterogénea: {paths}")
    return selected


def frames_by_index(source: Path, wanted: set[int]) -> dict[int, str]:
    """Recorre límites EXTXYZ y conserva texto solo de los índices autorizados."""
    frames: dict[int, str] = {}
    with source.open(encoding="utf-8") as stream:
        index = 0
        while True:
            first = stream.readline()
            if not first:
                break
            try:
                count = int(first.strip())
            except ValueError as exc:
                raise ValueError(f"Frame EXTXYZ {index} sin tamaño válido") from exc
            if count <= 0:
                raise ValueError(f"Frame EXTXYZ {index} vacío")
            comment = stream.readline()
            if not comment:
                raise ValueError(f"Comentario ausente en frame {index}")
            if index in wanted:
                atom_lines = [stream.readline() for _ in range(count)]
                if any(not line for line in atom_lines):
                    raise ValueError(f"Frame {index} truncado")
                frames[index] = first + comment + "".join(atom_lines)
            else:
                # No interpretar el comentario ni ninguna etiqueta de frames ajenos.
                for _ in range(count):
                    if not stream.readline():
                        raise ValueError(f"Frame {index} truncado")
            index += 1
    if set(frames) != wanted:
        raise ValueError(f"Faltan índices originales: {sorted(wanted-set(frames))}")
    LOG.info(json.dumps({"event": "frames_selected", "selected": len(frames), "total_frames_scanned": index}))
    return frames


def checked_structure(text: str, row: dict[str, Any], source_index: int) -> Structure:
    """Valida ID, celda, posiciones, elementos y composición antes de exportar."""
    atoms = read(io.StringIO(text), format="extxyz")
    sid = row["structure_id"]
    info = atoms.info
    if str(info.get("config_id")) != sid or str(info.get("source_key")) != row["source_key"]:
        raise ValueError(f"ID/source_key distintos en frame {source_index}: {sid}")
    expected_info = json.loads(row["info_json"])
    for key in ("polymorph", "calculation", "data_origin", "frame_index", "ionic_step", "converged"):
        if str(info.get(key)) != str(expected_info.get(key)):
            raise ValueError(f"Metadato {key} distinto: {sid}")
    if info.get("calculation") != "static" or info.get("data_origin") != "static":
        raise ValueError(f"Frame no estático: {sid}")
    if info.get("endpoint_role") != "static_ground_truth" or not bool(info.get("endpoint_cleaning_applied")):
        raise ValueError(f"Procedencia de endpoint no verificada: {sid}")
    if len(atoms) != int(row["num_atoms"]) or not bool(np.all(atoms.pbc)):
        raise ValueError(f"N/pbc incompatibles: {sid}")
    if not np.array_equal(atoms.pbc, np.asarray(json.loads(row["pbc_json"]), dtype=bool)):
        raise ValueError(f"PBC distinta al CSV: {sid}")
    arrays = json.loads(row["arrays_json"])
    if not np.array_equal(atoms.numbers, np.asarray(arrays["numbers"], dtype=int)):
        raise ValueError(f"Especies/orden distintos: {sid}")
    if not np.allclose(atoms.positions, np.asarray(arrays["positions"], dtype=float), rtol=0, atol=1e-7):
        raise ValueError(f"Posiciones distintas: {sid}")
    if not np.allclose(np.asarray(atoms.cell), np.asarray(json.loads(row["cell_json"]), dtype=float), rtol=0, atol=1e-7):
        raise ValueError(f"Celda distinta: {sid}")
    structure = Structure(Lattice(np.asarray(atoms.cell)), atoms.get_chemical_symbols(),
                          atoms.get_scaled_positions(wrap=False), coords_are_cartesian=False,
                          to_unit_cell=False)
    if len(structure) != int(row["num_atoms"]):
        raise ValueError(f"N de pymatgen distinto: {sid}")
    if structure.composition.reduced_composition != Composition(row["reduced_formula"]).reduced_composition:
        raise ValueError(f"Composición distinta: {sid}")
    return structure


def link(dataset: Path, source: Path, output: Path) -> dict[str, Any]:
    """Crea JSONL alineado, sin energías, más manifiesto de hashes/procedencia."""
    dataset, source, output = dataset.resolve(), source.resolve(), output.resolve()
    if not dataset.is_file() or not source.is_file():
        raise FileNotFoundError("Falta CSV de desarrollo o EXTXYZ fuente")
    if output.exists():
        raise FileExistsError(f"Se rechaza sobrescribir JSONL: {output}")
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Hash EXTXYZ distinto del manifiesto: {source_hash}")
    selected = development_rows(dataset)
    frames = frames_by_index(source, set(selected))
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    formulas: set[str] = set()
    for index in sorted(selected):
        row = selected[index]
        structure = checked_structure(frames[index], row, index)
        formulas.add(row["reduced_formula"])
        records.append({"structure_id": row["structure_id"], "geometry_source": SOURCE_LABEL,
                        "source_file": str(source), "source_index": index,
                        "source_sha256": source_hash, "structure": structure.as_dict()})
    if len(records) != 210 or len(formulas) != 70:
        raise ValueError(f"Cobertura inesperada: {len(records)} IDs, {len(formulas)} fórmulas")
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"status": "PASS", "records": 210, "compositions": 70,
                "geometry_source": SOURCE_LABEL, "source_file": str(source),
                "source_sha256": source_hash, "dataset_sha256": sha256(dataset),
                "jsonl_sha256": sha256(output), "selected_source_indices": sorted(selected),
                "identity_checks": ["config_id", "source_key", "source_index", "atom_count", "cell",
                                    "pbc", "atomic_numbers", "positions", "reduced_formula", "endpoint_role"],
                "historical_test_energies_read": False, "energy_fields_exported": False,
                "provenance_limit": "static endpoint_cleaned; relaxation history and XC/U policy not certified per structure"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    LOG.info(json.dumps({"event": "linked", "status": "PASS", "records": 210,
                         "jsonl_sha256": manifest["jsonl_sha256"]}))
    return manifest


def main() -> None:
    """Valida argumentos y ejecuta enlace exacto sobre desarrollo ABS3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    link(args.dataset, args.source, args.output)


if __name__ == "__main__":
    main()
