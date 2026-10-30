"""Recupera dependencias H3-C faltantes desde Git en un repositorio aislado.

No modifica las rutas históricas borradas ni reentrena Ridge. Solo copia blobs
exactos de HEAD y enlaza datos/checkpoint que aún existen en el árbol activo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

LOG = logging.getLogger("geometry_recover_inputs")


def sha256(path: Path) -> str:
    """Calcula SHA-256 de archivo recuperado o enlazado."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover(root: Path, stage: Path) -> dict[str, object]:
    """Crea rutas esperadas por preflight en stage, con hashes verificables."""
    root, stage = root.resolve(), stage.resolve()
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError(f"Staging no vacío: {stage}")
    report = root / "reports/MatGPT/MatGPT_H3C_Final/reproduction/reproduction_manifest.json"
    manifest = json.loads(report.read_text())
    if manifest.get("status") != "PASS":
        raise ValueError("Reproducción H3-C no verificada")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    fold_rel = "reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
    ridge_rel = "reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_B_ridge_fixed_delta/ridge_by_fold"
    recovered: dict[str, str] = {}
    for rel in [fold_rel] + [f"{ridge_rel}/fold_{fold}/ridge.joblib" for fold in range(5)]:
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(["git", "show", f"{head}:{rel}"], cwd=root)
        target.write_bytes(data)
        recovered[rel] = sha256(target)
    expected_fold = manifest["source_hashes"][fold_rel]
    if recovered[fold_rel] != expected_fold:
        raise ValueError("Fold recuperado no coincide con reproducción histórica")
    linked: dict[str, str] = {}
    for rel in (
        "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv",
        "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth",
        "reports/MatGPT/MatGPT_H3C_Final/reproduction/replayed_predictions.csv",
    ):
        source = root / rel
        if not source.is_file():
            raise FileNotFoundError(f"Dependencia activa ausente: {source}")
        destination = stage / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
        linked[rel] = sha256(destination)
    for rel in list(manifest["source_hashes"]):
        if rel in linked and linked[rel] != manifest["source_hashes"][rel]:
            raise ValueError(f"Hash de fuente activa cambió: {rel}")
    result: dict[str, object] = {"status": "PARTIAL_PENDING_CACHE", "git_head": head,
                                 "fold_sha256": expected_fold, "recovered": recovered,
                                 "linked": linked, "original_deleted_paths_untouched": True,
                                 "historical_test_accessed": False}
    (stage / "recovery_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    LOG.info(json.dumps({"event": "h3c_dependencies_recovered", "files": len(recovered), "status": result["status"]}))
    return result


def main() -> None:
    """Analiza CLI y crea staging aislado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    recover(args.repo_root, args.staging_root)


if __name__ == "__main__":
    main()
