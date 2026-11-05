"""Audita fuentes, particiones y cachés para el experimento F0/F1/F2 ABS₃.

La auditoría lee únicamente el dataset de desarrollo, verifica hashes de los
checkpoints y Ridge M2 históricos, y comprueba la integridad de embeddings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("abs3_original_supervised_ft_f0_f1_f2.preflight")
EXPECTED = {
    "MatGPT": ("models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
               "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"),
    "MatterGPT": ("models/MatterGPT/pytorch_model.pt",
                  "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"),
}
RIDGE_HASHES = (
    "99d5114f567450f46322cd39e021405bc4ba9fd396fadeb07ecb62f7a204f3d9",
    "48bc05a164936d4c2ef104baa60086a95388121eb95624003aad07388ddcfde1",
    "664fc440ce0a699fa5ac5e62f9167e3359682ada5bb63838323cb7a5a1efcf0a",
    "e26bae8b47c98b86731c283bcbd69732f6395dea7e66491bad82f08e53005e49",
    "680e1266483f838fbc962bed8b0d5458c3e7f1e059cf30a83dafdbf76c56408d",
)
RIDGE_GIT = "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_{}.joblib"
MATGPT_CACHE = Path("experiments/02_complementarios/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz")
MATTER_CACHE = Path("experiments/01_principales/mattergpt_abs3_frozen_ridge_mlp8/embeddings_mattergpt.npz")


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 en bloques.

    Args:
        path: Archivo que se resumirá.
    Returns:
        Hash SHA-256 hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_live_extractors() -> dict[str, Any]:
    """Carga estrictamente ambos backbones y compara las cachés con extracción nueva.

    Returns:
        Auditoría de carga y error numérico máximo por extractor.
    """
    import sys
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from scripts.ABS3.abs3_matgpt_mattergpt_original_f4_71 import extract_embeddings as extractor

    frame = extractor.load_dataset()
    results: dict[str, Any] = {}
    for model_name in ("MatGPT", "MatterGPT"):
        model_source = extractor.MATGPT_SOURCE if model_name == "MatGPT" else extractor.MATTER_DIR
        if str(model_source) not in sys.path:
            sys.path.insert(0, str(model_source))
        model, tokenizer, context, load_audit = (
            extractor.load_matgpt() if model_name == "MatGPT" else extractor.load_mattergpt()
        )
        if load_audit.get("missing_keys") or load_audit.get("unexpected_keys"):
            raise ValueError(f"Carga estricta incompleta para {model_name}: {load_audit}")
        sequences = extractor.encode(frame, tokenizer, context, matter=model_name == "MatterGPT")
        if model_name == "MatGPT":
            fresh = extractor.extract_matgpt(model, sequences, tokenizer["<pad>"], batch_size=8)
            cache_path = ROOT / MATGPT_CACHE
            cache_key, id_key = "FINAL_END", "structure_id"
        else:
            fresh = extractor.extract_mattergpt(model, sequences, batch_size=8)
            cache_path = ROOT / MATTER_CACHE
            cache_key, id_key = "embeddings", "structure_ids"
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_ids = archive[id_key].astype(str)
            cached_values = archive[cache_key]
        if set(cached_ids) != set(frame.structure_id.astype(str)):
            raise ValueError(f"Los IDs de caché no coinciden con dataset para {model_name}")
        position = {structure_id: index for index, structure_id in enumerate(cached_ids)}
        order = [position[structure_id] for structure_id in frame.structure_id.astype(str)]
        difference = np.abs(fresh - cached_values[order])
        max_abs = float(difference.max())
        rms = float(np.sqrt(np.mean(np.square(fresh - cached_values[order]))))
        # Tolerance allows serialization precision while detecting a different
        # checkpoint, terminal-token policy, tokenizer or conditioning vector.
        status = max_abs <= 2e-5
        results[model_name] = {
            "strict_load": "PASS", "checkpoint_sha256": load_audit["checkpoint_sha256"],
            "tokenizer_sha256": load_audit["tokenizer_sha256"],
            "cache_sha256": sha256_file(cache_path), "max_abs_difference": max_abs,
            "rmse_difference": rms, "tolerance": 2e-5,
            "property_vector": load_audit.get("property_vector"),
            "status": "PASS" if status else "FAIL",
        }
        del model
        if not status:
            raise ValueError(f"La extracción FINAL no reproduce la caché {model_name}: max_abs={max_abs}")
    return results


def run_preflight(verify_extractors: bool = True) -> dict[str, Any]:
    """Verifica checkpoints, cohortes, folds, cachés y artefactos Ridge.

    Returns:
        Diccionario de auditoría con hechos observados y estado general.
    """
    checks: dict[str, Any] = {}
    for name, (relative, expected_hash) in EXPECTED.items():
        path = ROOT / relative
        observed = sha256_file(path) if path.is_file() else None
        checks[f"checkpoint_{name}"] = {
            "path": relative, "exists": path.is_file(), "sha256": observed,
            "expected_sha256": expected_hash, "status": "PASS" if observed == expected_hash else "FAIL",
        }

    data_path = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
    folds_path = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
    dataset_hash = sha256_file(data_path)
    with folds_path.open(newline="", encoding="utf-8-sig") as stream:
        fold_rows = list(csv.DictReader(stream))
    development_ids = {row["structure_id"] for row in fold_rows
                       if row.get("historical_split") in {"train", "val"}}
    development: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    with data_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            # The current file contains only 210 development records. Filter by
            # fold-manifest IDs before reading any target into the audit table.
            if row.get("structure_id") not in development_ids:
                continue
            if row.get("split") != "development":
                raise ValueError(f"Fila de desarrollo con split inesperado: {row.get('structure_id')}")
            development.append(row)
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    ids = {row["structure_id"] for row in development}
    formulas: dict[str, list[dict[str, str]]] = {}
    for row in development:
        formulas.setdefault(row["reduced_formula"], []).append(row)
    fold_audit: dict[str, Any] = {}
    for fold in range(5):
        selected = [row for row in fold_rows if row.get("historical_split") in {"train", "val"}
                    and int(row["outer_fold"]) == fold]
        roles: dict[str, dict[str, str]] = {}
        for row in selected:
            roles[row["composition_group"]] = row
        counts = {
            "inner_train": sum(row.get("inner_role") == "inner_train" for row in roles.values()),
            "inner_validation": sum(row.get("inner_role") == "inner_validation" for row in roles.values()),
            "outer_evaluation": sum(row.get("outer_role") == "outer_validation" for row in roles.values()),
        }
        fold_audit[str(fold)] = {**counts, "status": "PASS" if tuple(counts.values()) == (45, 11, 14) else "FAIL"}

    checks["dataset_and_splits"] = {
        "dataset_path": str(data_path.relative_to(ROOT)), "dataset_sha256": dataset_hash,
        "fold_manifest_path": str(folds_path.relative_to(ROOT)), "fold_manifest_sha256": sha256_file(folds_path),
        "development_rows": len(development), "unique_structure_ids": len(ids),
        "unique_formulas": len(formulas), "structures_per_formula": sorted({len(rows) for rows in formulas.values()}),
        "observed_split_values": split_counts, "folds": fold_audit,
        "status": "PASS" if len(ids) == 210 and len(formulas) == 70 and all(len(v) == 3 for v in formulas.values())
                  and all(v["status"] == "PASS" for v in fold_audit.values()) else "FAIL",
    }

    checks["embedding_caches"] = {}
    for model_name, relative, feature_key, ids_key in (
        ("MatGPT", MATGPT_CACHE, "FINAL_END", "structure_id"),
        ("MatterGPT", MATTER_CACHE, "embeddings", "structure_ids"),
    ):
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as archive:
            embeddings = archive[feature_key]
            cache_ids = archive[ids_key].astype(str)
            property_vector = archive["property_vector"].tolist() if "property_vector" in archive.files else None
        checks["embedding_caches"][model_name] = {
            "path": str(relative), "sha256": sha256_file(path), "shape": list(embeddings.shape),
            "finite": bool(np.isfinite(embeddings).all()), "unique_ids": len(set(cache_ids)) == 210,
            "same_development_ids": set(cache_ids) == ids,
            "property_vector": property_vector,
            "status": "PASS" if embeddings.shape == (210, 768) and np.isfinite(embeddings).all()
                      and len(set(cache_ids)) == 210 and set(cache_ids) == ids
                      and (model_name != "MatterGPT" or property_vector == [1.0, 2.0]) else "FAIL",
        }

    checks["ridge_m2"] = {}
    for fold, expected_hash in enumerate(RIDGE_HASHES):
        result = subprocess.run(["git", "show", f"HEAD:{RIDGE_GIT.format(fold)}"], cwd=ROOT,
                                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        observed_hash = hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None
        checks["ridge_m2"][str(fold)] = {
            "source": f"HEAD:{RIDGE_GIT.format(fold)}", "sha256": observed_hash,
            "expected_sha256": expected_hash,
            "status": "PASS" if result.returncode == 0 and observed_hash == expected_hash else "FAIL",
        }

    checks["test_independent_accessed"] = False
    checks["live_extractor_validation"] = verify_live_extractors() if verify_extractors else "SKIPPED"
    checks["active_process_visibility"] = "BLOCKED_BY_OS_PERMISSION"
    all_statuses = [value["status"] for key, value in checks.items()
                    if isinstance(value, dict) and "status" in value]
    all_statuses += [value["status"] for value in checks["embedding_caches"].values()]
    all_statuses += [value["status"] for value in checks["ridge_m2"].values()]
    live = checks["live_extractor_validation"]
    if isinstance(live, dict):
        all_statuses += [value["status"] for value in live.values()]
    else:
        all_statuses.append("FAIL")
    checks["overall_status"] = "PASS" if all(status == "PASS" for status in all_statuses) else "FAIL"
    return checks


def main() -> None:
    """Interpreta argumentos, guarda resultados y registra duración/recursos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/abs3_original_supervised_ft_f0_f1_f2/preflight.json")
    parser.add_argument("--skip-live-extractors", action="store_true",
                        help="Omite la recarga estricta y equivalencia numérica (solo para inspección rápida).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()
    payload = run_preflight(verify_extractors=not args.skip_live_extractors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    resources = {
        "phase": "read_only_preflight", "wall_seconds": time.monotonic() - started,
        "process_cpu_seconds": time.process_time(), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
        "cpu_count_logical": os.cpu_count(), "torch_threads": 1,
        "platform": platform.platform(), "python": platform.python_version(),
        "device_training_available": "CPU only; MPS unavailable",
        "process_table": "not readable in this environment; ps denied by OS permission",
        "heavy_training_started": False, "test_independent_accessed": False,
    }
    (args.output.parent / "resources.json").write_text(json.dumps(resources, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("preflight=%s output=%s", payload["overall_status"], args.output)
    if payload["overall_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
