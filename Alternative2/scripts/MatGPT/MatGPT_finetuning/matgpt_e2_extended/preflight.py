#!/usr/bin/env python3
"""Audit the pinned MatGPT E2 inputs without reading the historical test labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
INFO = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
CHECKPOINT = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
SOURCE_MANIFEST = ALT / "models/MatGPT-source.download-manifest.json"
OUT = ALT / "reports/MatGPT_finetuning/matgpt_e2_extended/preflight"
EXPECTED_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
EXPECTED_COMMIT = "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e"
LOG = logging.getLogger("matgpt_e2_extended.preflight")


def sha256(path: Path) -> str:
    """Return SHA-256 for a file.

    Args:
        path: File to hash.
    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict[str, Any]:
    """Validate source identities, folds, vocabulary compatibility, and exclusions.

    Args:
        None.
    Returns:
        Machine-readable audit results. Raises on a critical integrity failure.
    """
    for path in (DATA, FOLDS, INFO, CHECKPOINT, SOURCE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Falta entrada requerida: {path}")
    checkpoint_sha = sha256(CHECKPOINT)
    if checkpoint_sha != EXPECTED_SHA:
        raise RuntimeError(f"SHA-256 inesperado para checkpoint: {checkpoint_sha}")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("resolved_commit") != EXPECTED_COMMIT:
        raise RuntimeError("El código local de MatGPT no coincide con el commit E0 fijado.")
    info = json.loads(INFO.read_text(encoding="utf-8"))
    stoi = info["stoi"]
    data = pd.read_csv(DATA, low_memory=False)
    folds = pd.read_csv(FOLDS)
    if len(data) != 234 or data.structure_id.astype(str).duplicated().any():
        raise RuntimeError("Dataset estático E2-A debe contener 234 IDs únicos.")
    if not data.calculation_type.astype(str).eq("static").all():
        raise RuntimeError("La entrada de entrenamiento incluye registros no estáticos.")
    for column in ("structure_id", "matgpt_slices", "energy_per_atom_ev", "split"):
        if column not in data:
            raise RuntimeError(f"Falta columna de dataset: {column}")
    required_fold_columns = {"structure_id", "composition_group", "historical_split", "outer_fold", "outer_role", "inner_role"}
    if not required_fold_columns.issubset(folds.columns):
        raise RuntimeError("La asignación de folds no contiene los campos requeridos.")
    if folds.groupby("structure_id").composition_group.nunique().max() != 1:
        raise RuntimeError("Una estructura aparece asociada a más de una composición en folds.")
    group_map = folds.drop_duplicates("structure_id").set_index("structure_id").composition_group
    data["composition_group"] = data.structure_id.astype(str).map(group_map)
    if data.composition_group.isna().any():
        raise RuntimeError("No se pudo mapear la composición desde la asignación E1.3.")
    unknown = sorted({token for seq in data.matgpt_slices.astype(str)
                      for token in seq.split() if token not in stoi})
    if unknown:
        raise RuntimeError(f"SLICES histórico contiene tokens no presentes en vocabulario: {unknown[:12]}")
    if len(stoi) != 172 or int(info.get("block_size", -1)) != 442:
        raise RuntimeError("Metadatos del tokenizer/checkpoint no coinciden con auditoría E0 (172/442).")

    # The audited E2-A sequence uses strategy-4 compact edge labels, but omits
    # the optional space-group prefix. The current API adds a prefix that the
    # pinned checkpoint vocabulary cannot tokenize.
    strategy4_prefix_probe = {"o", "y", "j", "OOO"}
    strategy4_prefix_missing = sorted(strategy4_prefix_probe - set(stoi))
    dev = data[data.split.isin(["train", "val"])].copy()
    test_ids = set(data.loc[data.split.eq("test"), "structure_id"].astype(str))
    development_ids = set(dev.structure_id.astype(str))
    if len(dev) != 210 or len(test_ids) != 24:
        raise RuntimeError("El split histórico no coincide con 210 de desarrollo / 24 de test.")
    fold_dev = folds[folds.historical_split.isin(["train", "val"])].copy()
    fold_test_ids = set(folds.loc[folds.historical_split.eq("test"), "structure_id"].astype(str))
    if fold_test_ids & development_ids or fold_test_ids != test_ids:
        raise RuntimeError("Folds no conservan la exclusión exacta del test histórico.")
    expected_per_fold = set(data.loc[data.split.isin(["train", "val"]), "structure_id"].astype(str))
    fold_checks = []
    for fold_id, frame in fold_dev.groupby("outer_fold"):
        ids = set(frame.structure_id.astype(str))
        if ids != expected_per_fold or len(frame) != 210:
            raise RuntimeError(f"Fold {fold_id} no cubre los mismos 210 IDs de desarrollo.")
        roles = frame.outer_role.value_counts().to_dict()
        if int(roles.get("outer_validation", 0)) != 42:
            raise RuntimeError(f"Fold {fold_id}: outer validation no tiene 42 estructuras.")
        grouped_roles = frame.groupby("composition_group").outer_role.nunique()
        if grouped_roles.max() != 1:
            raise RuntimeError(f"Fold {fold_id}: una composición cruza la partición externa.")
        fold_checks.append({"fold": int(fold_id), "rows": len(frame), "outer_validation": 42})
    if len(fold_checks) != 5:
        raise RuntimeError("Se esperaban exactamente cinco folds históricos.")

    result: dict[str, Any] = {
        "status": "PASS_WITH_PROTOCOL_NOTE",
        "checkpoint_sha256": checkpoint_sha,
        "source_commit": source.get("resolved_commit"),
        "source_files_sha256": {
            "dataset.csv": sha256(DATA),
            "fold_assignments.csv": sha256(FOLDS),
            "model_info_property_0.json": sha256(INFO),
        },
        "vocabulary_size": len(stoi),
        "block_size": int(info["block_size"]),
        "historical_sequences_tokenizable": True,
        "strategy4_prefix_tokens_missing_from_vocab": strategy4_prefix_missing,
        "audited_e2a_representation": "strategy-4 compact periodic edge labels, without the optional space-group prefix",
        "protocol_resolution": "Use strategy 4 and preserve the historical no-prefix input. The API's optional space-group prefix is omitted because its symbols are absent from the pinned vocabulary; the periodic graph is verified after omission.",
        "development_rows": len(dev),
        "development_compositions": int(dev.composition_group.nunique()),
        "historical_test_rows_excluded": len(test_ids),
        "historical_test_compositions_excluded": int(data.loc[data.split.eq("test"), "composition_group"].nunique()),
        "folds": fold_checks,
        "device_probe": None,
    }
    return result


def main() -> None:
    """Parse arguments, run the preflight, and save its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT, help="Directorio nuevo del preflight.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = audit()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        LOG.info("preflight_complete status=%s dev=%s folds=%s optional_sg_prefix_supported=%s", report["status"], report["development_rows"], len(report["folds"]), not report["strategy4_prefix_tokens_missing_from_vocab"])
    except Exception as exc:
        LOG.exception("preflight_failed error=%s", exc)
        raise


if __name__ == "__main__":
    main()
