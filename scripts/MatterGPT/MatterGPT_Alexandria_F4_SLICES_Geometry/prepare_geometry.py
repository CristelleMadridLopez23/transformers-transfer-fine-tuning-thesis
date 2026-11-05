#!/usr/bin/env python3
"""Vincula geometría cristalográfica Alexandria por mat_id para el F4 MatterGPT.

Lee las entradas ComputedStructureEntry originales y aplica sin cambios la
función histórica de geometría de celda primitiva/Niggli usada en ABS₃.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
SOURCE_RAW = ROOT / "data/raw_data/alexandria_00000.json"
TRAIN_CSV = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/alexandria_matter_compatible_train.csv"
VAL_CSV = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/alexandria_matter_compatible_validation.csv"
OUT = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry"
GEOM_COLS = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
LOG = logging.getLogger("alexandria_geometry_prepare")


def sha_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo en bloques."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(obj: Any) -> str:
    """Calcula el hash canónico igual al usado por el manifiesto Alexandria."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_module(path: Path, name: str) -> Any:
    """Carga módulo local por ruta sin ejecutar main."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar módulo requerido {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_targets() -> pd.DataFrame:
    """Carga solo las 17,245/1,718 estructuras del split MatterGPT ya fijado."""
    train = pd.read_csv(TRAIN_CSV, dtype={"mat_id": str, "structure_hash": str, "slices_hash": str})
    val = pd.read_csv(VAL_CSV, dtype={"mat_id": str, "structure_hash": str, "slices_hash": str})
    if (len(train), len(val)) != (17245, 1718):
        raise ValueError(f"Cohorte no corresponde al split anterior: {len(train)}/{len(val)}")
    if train.mat_id.duplicated().any() or val.mat_id.duplicated().any() or set(train.mat_id) & set(val.mat_id):
        raise ValueError("mat_id duplicado o compartido entre train/validation")
    if set(train.split) != {"train"} or set(val.split) != {"validation"}:
        raise ValueError("Los CSV no conservan las particiones previas")
    train = train.copy(); val = val.copy()
    train["alexandria_split"] = "train"; val["alexandria_split"] = "validation"
    data = pd.concat([train, val], ignore_index=True)
    if len(data) != 18963 or data.mat_id.nunique() != 18963:
        raise ValueError("Se requieren exactamente 18,963 IDs Alexandria únicos")
    return data


def atomic_json(path: Path, value: Any) -> None:
    """Guarda JSON con reemplazo atómico."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def run() -> dict[str, Any]:
    """Audita IDs y extrae siete descriptores desde las celdas originales."""
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "geometry_manifest.csv").exists():
        raise FileExistsError(f"No se sobrescribe geometría previa: {OUT/'geometry_manifest.csv'}")
    data = load_targets()
    prepare = load_module(ROOT / "scripts/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/prepare_provisional.py", "alex_geometry_source_functions")
    geometry = load_module(ROOT / "scripts/MatGPT_SLICES_Geometria_Hibrida/exportar_geometria.py", "historical_geometry_descriptor")
    from pymatgen.core import Structure

    by_id = {str(row.mat_id): row for row in data.itertuples(index=False)}
    found: set[str] = set(); output: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    raw_sha = sha_file(SOURCE_RAW)
    LOG.info("Geometría de %d IDs; escaneo por mat_id en %s", len(by_id), SOURCE_RAW)
    for rec_no, entry in enumerate(prepare.stream_entries(SOURCE_RAW), start=1):
        meta = entry.get("data") or {}; sid = str(meta.get("mat_id", ""))
        if sid not in by_id:
            continue
        if sid in found:
            failures.append({"mat_id": sid, "split": by_id[sid].alexandria_split, "reason": "mat_id duplicado en shard JSON"})
            continue
        row = by_id[sid]
        try:
            struct_dict = entry.get("structure")
            if not isinstance(struct_dict, dict):
                raise ValueError("entry.structure no es un objeto JSON")
            actual_hash = sha_object(struct_dict)
            if actual_hash != str(row.structure_hash):
                raise ValueError(f"hash de estructura no coincide: {actual_hash} != {row.structure_hash}")
            atoms = prepare.atom_numbers_from_structure(struct_dict)
            structure = Structure.from_dict(struct_dict)
            if len(structure) != int(row.n_atoms) or len(atoms) != int(row.n_atoms):
                raise ValueError(f"número de átomos distinto {len(structure)} != {row.n_atoms}")
            entry_energy = float(entry["energy"])
            nested_energy = float(meta["energy_total"])
            expected_energy = float(row.energy_total_ev)
            if max(abs(entry_energy - nested_energy), abs(entry_energy - expected_energy)) > 1e-8:
                raise ValueError(f"energía total no coincide con entry.energy ({entry_energy}, {expected_energy})")
            ep = entry_energy / len(structure)
            if abs(ep - float(row.energy_per_atom_ev)) > 1e-10:
                raise ValueError(f"E/N no coincide: {ep} != {row.energy_per_atom_ev}")
            if str(meta.get("location", "")) != str(row.source_location_raw):
                raise ValueError("source_location_raw distinto para el mat_id")
            desc = geometry.structure_descriptor(structure)
            vals = np.asarray([desc[c] for c in GEOM_COLS], dtype=float)
            if not np.isfinite(vals).all() or (vals[[0,1,2,6]] <= 0).any():
                raise ValueError("Descriptor no finito o longitud/volumen no positivo")
            output.append({"mat_id": sid, "split": row.alexandria_split, "formula": row.formula,
                "reduced_formula": row.reduced_formula, "n_atoms": len(structure), "energy_total_ev": entry_energy,
                "energy_per_atom_ev": ep, "slices_matgpt": row.slices_matgpt, "slices_hash": row.slices_hash,
                "structure_hash": actual_hash, "geometry_source": "ComputedStructureEntry.entry.structure",
                "source_path": "data/raw_data/alexandria_00000.json", "source_sha256": raw_sha,
                "source_location": str(meta.get("location", "")), **desc})
            found.add(sid)
        except Exception as exc:
            failures.append({"mat_id": sid, "split": row.alexandria_split, "reason": f"{type(exc).__name__}:{exc}"})
        if rec_no % 100000 == 0:
            LOG.info("JSON scan records=%d matched=%d/%d", rec_no, len(found), len(by_id))

    missing = sorted(set(by_id) - found)
    for sid in missing:
        failures.append({"mat_id": sid, "split": by_id[sid].alexandria_split, "reason": "mat_id no encontrado en shard JSON"})
    pd.DataFrame(failures, columns=["mat_id", "split", "reason"]).to_csv(OUT / "geometry_missing_or_invalid.csv", index=False)
    if failures or len(output) != 18963:
        atomic_json(OUT / "geometry_audit.json", {"status": "FAIL", "expected": 18963, "valid": len(output),
            "missing_or_invalid": len(failures), "failure_csv": str((OUT / "geometry_missing_or_invalid.csv").relative_to(ROOT)),
            "source_sha256": raw_sha, "historical_test_accessed": False})
        raise ValueError(f"No se generó geometría completa: válidas={len(output)}, fallas={len(failures)}; revisar geometry_missing_or_invalid.csv")

    frame = pd.DataFrame(output)
    order = pd.Categorical(frame.split, categories=["train", "validation"], ordered=True)
    frame = frame.assign(_split_order=order).sort_values(["_split_order", "mat_id"]).drop(columns="_split_order").reset_index(drop=True)
    if frame.groupby("split").size().to_dict() != {"train":17245,"validation":1718}:
        raise ValueError("Conteos de geometría por split no coinciden")
    scaler = StandardScaler().fit(frame.loc[frame.split.eq("train"), GEOM_COLS].to_numpy(float))
    frame[[f"scaled_{c}" for c in GEOM_COLS]] = scaler.transform(frame[GEOM_COLS].to_numpy(float))
    geom_path = OUT / "geometry_manifest.csv"; frame.to_csv(geom_path, index=False)
    scaler_path = OUT / "alexandria_geometry_scaler.joblib"; joblib.dump(scaler, scaler_path)
    scaler_meta = {"scaler": "sklearn StandardScaler", "fit_rows": 17245, "fit_split": "train only",
                   "columns": GEOM_COLS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(), "validation_fit": False}
    atomic_json(OUT / "geometry_scaler.json", scaler_meta)
    manifest = {"status":"PASS","rows":len(frame),"train_rows":17245,"validation_rows":1718,
        "matched_by":"mat_id","one_to_one":True,"all_structure_hashes_equal":True,"all_energy_targets_equal":True,
        "source_json":str(SOURCE_RAW.relative_to(ROOT)),"source_sha256":raw_sha,
        "train_csv_sha256":sha_file(TRAIN_CSV),"validation_csv_sha256":sha_file(VAL_CSV),
        "geometry_manifest_sha256":sha_file(geom_path),"scaler_sha256":sha_file(scaler_path),
        "geometry_definition":"reused structure_descriptor() from MatGPT_SLICES_Geometria_Hibrida/exportar_geometria.py; pymatgen primitive tolerance 0.01, Niggli tol 1e-5; a,b,c / cube_root(primitive volume); angles reduced cell; original cell volume per atom",
        "scaler":scaler_meta,"invalid_rows":0,"historical_test_accessed":False,"elapsed_wall_seconds":time.time()-started}
    atomic_json(OUT / "geometry_audit.json", manifest)
    return manifest


def main() -> int:
    """Valida argumentos locales, escribe auditoría y propaga fallos descriptivos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output_dir.resolve() != OUT.resolve():
        raise ValueError("La salida está fijada a la carpeta del experimento indicada")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
