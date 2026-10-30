"""Coteja un cache MatGPT reconstruido con evidencia H3-C histórica guardada."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("geometry_cache_audit")
EXPECTED_OLD = "2f790a30ba0fa3cd961c12de82b65febf0ae651b1f217efd5df49c8c5a9cfb62"


def digest(path: Path) -> str:
    """SHA-256 por bloques."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path, stage: Path) -> dict[str, object]:
    """Compara distancias por par y medias Ridge sin ajustar parámetros."""
    root, stage = root.resolve(), stage.resolve()
    rel = "reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
    path = stage / rel
    saved = root / "reports/MatGPT/MatGPT_H3C_Final/analysis/embedding_pairs.csv"
    replay = root / "reports/MatGPT/MatGPT_H3C_Final/reproduction/replayed_predictions.csv"
    pair_table = pd.read_csv(saved)
    pair_table = pair_table[pair_table.backbone == "pretrained"]
    predictions = pd.read_csv(replay)
    predictions = predictions[predictions.seed == 42]
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["structure_id"].astype(str).tolist()
        if len(ids) != 210 or len(set(ids)) != 210:
            raise ValueError("IDs del cache reconstruido incompletos")
        lookup = {sid: i for i, sid in enumerate(ids)}
        features = {name: cache[name].astype(np.float64) for name in ("L03_MEAN", "FINAL_END")}
    differences: dict[str, dict[str, float]] = {}
    for name, vectors in features.items():
        standard = (vectors - vectors.mean(axis=0)) / np.maximum(vectors.std(axis=0), 1e-6)
        sub = pair_table[pair_table.feature == name]
        if len(sub) != 210:
            raise ValueError(f"Pares históricos incompletos: {name}")
        errors: list[list[float]] = [[], [], []]
        for row in sub.itertuples():
            a, b = lookup[row.structure_a], lookup[row.structure_b]
            cosine = float(np.dot(vectors[a], vectors[b]) / (np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b])))
            raw = float(np.linalg.norm(vectors[a] - vectors[b]))
            scaled = float(np.linalg.norm(standard[a] - standard[b]))
            for bucket, value, reference in zip(errors, (cosine, raw, scaled),
                                                (row.cosine_similarity, row.euclidean_raw, row.euclidean_standardized)):
                bucket.append(abs(value - reference))
        differences[name] = {"cosine": max(errors[0]), "euclidean_raw": max(errors[1]),
                             "euclidean_standardized": max(errors[2])}
    ridge_errors = []
    ridge_root = stage / "reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_B_ridge_fixed_delta/ridge_by_fold"
    for fold in range(5):
        ridge = joblib.load(ridge_root / f"fold_{fold}/ridge.joblib")
        subset = predictions[predictions.fold == fold]
        if len(subset) != 42:
            raise ValueError(f"Predicciones fold {fold} incompletas")
        for _, group in subset.groupby("composition_id"):
            mean = features["L03_MEAN"][[lookup[sid] for sid in group.structure_id]].astype(np.float32).mean(axis=0)
            pred = float(ridge.predict(mean.reshape(1, -1))[0])
            ridge_errors.append(abs(pred - float(group.ridge_mean_ev_atom.iloc[0])))
    max_ridge = max(ridge_errors)
    maxima = [value for part in differences.values() for value in part.values()]
    status = "PASS_NUMERIC" if max_ridge < 1e-5 and max(maxima) < 1e-4 else "FAIL"
    result: dict[str, object] = {"status": status, "cache_sha256": digest(path),
                                 "historical_cache_sha256": EXPECTED_OLD,
                                 "bitwise_identical": digest(path) == EXPECTED_OLD,
                                 "max_pair_feature_differences": differences,
                                 "max_ridge_mean_difference_ev_atom": max_ridge,
                                 "reference_pair_table_sha256": digest(saved),
                                 "reference_replayed_predictions_sha256": digest(replay),
                                 "note": "Paridad numérica de distancias y rama Ridge; no equivalencia bit a bit del archivo NPZ.",
                                 "historical_test_accessed": False}
    audit_file = path.with_suffix(".numeric_audit.json")
    audit_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    LOG.info(json.dumps({"event": "cache_numeric_audit", "status": status, "max_ridge": max_ridge}))
    if status != "PASS_NUMERIC":
        raise ValueError("Cache reconstruido no pasó comparación numérica H3-C")
    recovery_file = stage / "recovery_manifest.json"
    recovery = json.loads(recovery_file.read_text())
    recovery["status"] = "COMPLETE_NUMERIC"
    recovery["cache_sha256"] = result["cache_sha256"]
    recovery["cache_numeric_audit_sha256"] = digest(audit_file)
    recovery_file.write_text(json.dumps(recovery, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    """Valida CLI y ejecuta cotejo de solo lectura sobre referencia histórica."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit(args.repo_root, args.staging_root)


if __name__ == "__main__":
    main()
