#!/usr/bin/env python3
"""Audita checkpoint, features, folds y compatibilidad de OOF para H3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import (CHECKPOINT, EXPECTED_CHECKPOINT_SHA, EXPECTED_COMMIT, FEATURES,
                    FOLDS, H2_ROOT, H3Error, R4_ROOT, REPORT_DIR, ALT, ROOT,
                    load_development_without_parsing_test_targets, load_features,
                    load_fold_manifest, sha256_file, write_json)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.two_head_predictor import TwoHeadPredictor


def audit() -> dict[str, Any]:
    """Comprueba que cada OOF cubra desarrollo con folds y etiquetas idénticos.

    Returns:
        Resumen JSON de preflight sin cargar etiquetas test.
    """
    if not CHECKPOINT.is_file() or sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise H3Error("El checkpoint original no coincide con el SHA E0 esperado.")
    source_manifest_path = ALT / "models/MatGPT-source.download-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    commit = str(source_manifest.get("resolved_commit", ""))
    if commit != EXPECTED_COMMIT:
        raise H3Error(f"Manifiesto de código MatGPT en {commit}, esperado {EXPECTED_COMMIT}.")
    source_files = {item["path"]: item["sha256"] for item in source_manifest.get("files", [])}
    for relative in ("mat-concat/train.py", "mat-concat/utils/dataset.py", "mat-concat/utils/models.py", "mat-concat/config/uncond.yaml"):
        path = ALT / "models/MatGPT-source" / relative
        if relative not in source_files or not path.is_file() or sha256_file(path) != source_files[relative]:
            raise H3Error(f"Código/config MatGPT no coincide con el manifiesto fijado: {relative}.")

    data = load_development_without_parsing_test_targets()
    folds = load_fold_manifest(set(data.structure_id))
    features, feature_manifest = load_features(data.structure_id.tolist())
    feature_index = {name: idx for idx, name in enumerate(data.structure_id)}
    if sha256_file(FEATURES) != feature_manifest.get("cache_file_sha256", sha256_file(FEATURES)):
        raise H3Error("Hash declarado del cache de features no coincide.")

    h2 = pd.read_csv(H2_ROOT / "oof_predictions_enriched.csv")
    h0 = h2.loc[h2.model.eq("H0")].copy()
    h2 = h2.loc[h2.model.eq("H2")].copy()
    r4 = pd.read_csv(R4_ROOT / "oof_predictions.csv")
    r4 = r4.loc[r4.model.eq("R4-SELECTED-ABSOLUTE")].copy()
    d1 = pd.read_csv(H2_ROOT / "D1_ridge_composition_only_oof.csv")

    expected = set(data.structure_id)
    def check_oof(frame: pd.DataFrame, label: str, expected_runs: set[tuple[int, int]] | None = None) -> None:
        """Verifica cobertura completa sin duplicados y sin IDs fuera de dev."""
        ids = set(frame.structure_id.astype(str))
        if ids - expected:
            raise H3Error(f"{label} contiene IDs fuera de desarrollo (posible test).")
        if frame.duplicated(["structure_id", "fold"] + (["seed"] if "seed" in frame else [])).any():
            raise H3Error(f"{label} contiene claves OOF duplicadas.")
        if "seed" in frame and expected_runs is not None:
            observed = set(zip(frame.fold.astype(int), frame.seed.astype(int)))
            if observed != expected_runs or len(frame) != 42 * len(expected_runs):
                raise H3Error(f"{label} no cubre exactamente todas las corridas esperadas.")
        elif len(frame) != 210 or frame.structure_id.nunique() != 210:
            raise H3Error(f"{label} debe contener una predicción determinista por cada uno de 210 IDs OOF.")
        if frame.historical_split.eq("test").any() if "historical_split" in frame else False:
            raise H3Error(f"{label} marca filas de test.")

    expected_runs = {(fold, seed) for fold in range(5) for seed in (42, 123, 2026)}
    check_oof(h0, "H0", expected_runs)
    check_oof(h2, "H2", expected_runs)
    check_oof(r4, "R4 selected absolute")
    check_oof(d1, "D1")
    if set(h0.structure_id) != expected or set(h2.structure_id) != expected:
        raise H3Error("H0/H2 no contienen cada ID de desarrollo.")

    # Reconcile structure IDs, DFT labels, fold mapping, composition, and families.
    reference = data.set_index("structure_id")
    fold_by_id = folds.loc[folds.outer_fold.eq(0)].set_index("structure_id")
    for name, frame in (("H0", h0), ("H2", h2), ("R4", r4), ("D1", d1)):
        for row in frame.itertuples(index=False):
            sid = str(row.structure_id)
            ref = reference.loc[sid]
            if str(row.composition_id) != ref.composition_id or str(row.polymorph_family) != ref.polymorph_family:
                raise H3Error(f"{name}: metadatos estructurales no alineados para {sid}.")
            y_col = "E_DFT_eV_atom" if "E_DFT_eV_atom" in frame else "E_DFT_eV_atom"
            if abs(float(getattr(row, y_col)) - float(ref.E_DFT_eV_atom)) > 1e-9:
                raise H3Error(f"{name}: etiqueta DFT no coincide para {sid}.")
            expected_fold = int(folds.loc[(folds.structure_id.eq(sid)) & (folds.outer_fold.eq(int(row.fold))), "outer_role"].eq("outer_validation").all())
            if expected_fold != 1:
                raise H3Error(f"{name}: fila no está en outer validation correspondiente ({sid}).")
    for frame, name in ((h0, "H0"), (h2, "H2")):
        for fold, seed in expected_runs:
            sub = frame.loc[frame.fold.eq(fold) & frame.seed.eq(seed)]
            expected_ids = set(folds.loc[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation"), "structure_id"])
            if set(sub.structure_id) != expected_ids or len(sub) != 42:
                raise H3Error(f"{name}: outer fold {fold}/seed {seed} no alinea exactamente.")
    for frame, name in ((r4, "R4"), (d1, "D1")):
        for fold in range(5):
            sub = frame.loc[frame.fold.eq(fold)]
            expected_ids = set(folds.loc[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation"), "structure_id"])
            if set(sub.structure_id) != expected_ids:
                raise H3Error(f"{name}: outer fold {fold} no alinea exactamente.")

    h2_runs = H2_ROOT / "runs/H2"
    h2_checkpoints = sorted(h2_runs.glob("fold_*_seed_*/head.pt"))
    if len(h2_checkpoints) != 15:
        raise H3Error(f"Se esperaban 15 checkpoints H2, encontrados {len(h2_checkpoints)}.")
    h2_provenance = [json.loads(path.with_name("run_provenance.json").read_text(encoding="utf-8")) for path in h2_checkpoints]
    if any(item.get("test_used") is not False or item.get("status") != "PASS" for item in h2_provenance):
        raise H3Error("Algún checkpoint H2 no está certificado PASS/test-free.")
    # Re-run only the frozen-feature residual head from each saved H2 checkpoint.
    # This does not read outer DFT labels to predict; labels are used only afterward
    # to compare with the archived OOF residual column.
    checkpoint_residual_rows = []
    oof_h2 = h2.set_index(["structure_id", "fold", "seed"])
    family_order = {family: index for index, family in enumerate(("dist_perovskite", "hexagonal", "needle"))}
    feature_by_id = {sid: index for index, sid in enumerate(data.structure_id)}
    for fold in range(5):
        outer_ids = set(folds.loc[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation"), "structure_id"])
        outer = data.loc[data.structure_id.isin(outer_ids)].copy()
        outer["_family_rank"] = outer.polymorph_family.map(family_order)
        outer = outer.sort_values(["composition_id", "_family_rank"], kind="stable")
        ids = outer.structure_id.astype(str).tolist()
        x = torch.tensor(np.array(features["FINAL_END"][[feature_by_id[sid] for sid in ids]], copy=True), dtype=torch.float32)
        x = x.reshape(-1, 3, 768)
        for seed in (42, 123, 2026):
            checkpoint = h2_runs / f"fold_{fold}_seed_{seed}" / "head.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = TwoHeadPredictor(width=8, dropout=.1)
            model.load_state_dict(payload["state_dict"], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise H3Error("La cabeza histórica H2 se esperaba solo para inferencia.")
            scale = float(payload["scales"]["scale_struct_ev_atom"])
            with torch.no_grad():
                raw = model.residual(x)
                delta = (raw - raw.mean(dim=1, keepdim=True)).cpu().numpy() * scale
            if not np.isfinite(delta).all() or np.max(np.abs(delta.sum(axis=1))) > 1e-7:
                raise H3Error(f"H2 residual no centrado/fino fold={fold} seed={seed}.")
            for comp_index, composition in enumerate(outer.composition_id.iloc[::3].astype(str)):
                for family_index, family in enumerate(("dist_perovskite", "hexagonal", "needle")):
                    row = outer.iloc[comp_index * 3 + family_index]
                    old = oof_h2.loc[(row.structure_id, fold, seed), "delta_pred_centered_eV_atom"]
                    error = abs(float(delta[comp_index, family_index]) - float(old))
                    if error > 1e-6:
                        raise H3Error(f"H2 checkpoint no reproduce residual OOF para {row.structure_id}, error={error}.")
                    checkpoint_residual_rows.append({"structure_id": row.structure_id, "composition_id": composition,
                        "polymorph_family": family, "fold": fold, "seed": seed,
                        "delta_pred_centered_eV_atom": float(delta[comp_index, family_index]),
                        "max_abs_difference_vs_archived_oof_eV_atom": error})
    checkpoint_residual_path = REPORT_DIR / "provenance" / "H2_structural_oof_from_checkpoints.csv"
    pd.DataFrame(checkpoint_residual_rows).to_csv(checkpoint_residual_path, index=False)
    audit_path = REPORT_DIR / "provenance" / "preflight.json"
    summary = {
        "status": "PASS", "checkpoint_sha256": sha256_file(CHECKPOINT), "source_commit": commit,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "pinned_source_file_hashes_verified": ["mat-concat/train.py", "mat-concat/utils/dataset.py", "mat-concat/utils/models.py", "mat-concat/config/uncond.yaml"],
        "development_rows": len(data), "development_compositions": data.composition_id.nunique(),
        "historical_test_labels_parsed": False, "historical_test_ids_loaded_into_features": False,
        "fold_manifest_sha256": sha256_file(FOLDS), "feature_cache_sha256": sha256_file(FEATURES),
        "feature_manifest_status": feature_manifest.get("status"), "feature_count": len(features),
        "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        "H0_oof_rows": len(h0), "H2_oof_rows": len(h2), "R4_oof_rows": len(r4), "D1_oof_rows": len(d1),
        "H2_structural_checkpoints": len(h2_checkpoints),
        "H2_checkpoint_residual_reproductions": len(checkpoint_residual_rows),
        "H2_checkpoint_residual_max_abs_error_ev_atom": float(pd.DataFrame(checkpoint_residual_rows).max_abs_difference_vs_archived_oof_eV_atom.max()),
        "H2_feature_provenance_compatibility": "All 15 H2 structural MLP8 checkpoints were re-run on the validated FINAL_END feature cache; outputs match archived OOF residuals within 1e-6 eV/atom.",
        "R4_seed_policy": "one deterministic outer prediction per structure/fold (seed=-1); reused as common component for each H2 seed in A.",
        "test_used": False,
    }
    write_json(audit_path, summary)
    return summary


def main() -> None:
    """Ejecuta auditoría inicial sin iniciar entrenamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        print(json.dumps(audit(), indent=2, ensure_ascii=False))
    except (H3Error, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"H3 preflight FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
