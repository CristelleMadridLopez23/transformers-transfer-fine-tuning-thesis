#!/usr/bin/env python3
"""Audita entradas E1 y congela folds externos e inner holdouts por composición."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import GroupKFold

from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import (ALT, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CANDIDATES, CHECKPOINT, DATASET, EXPECTED_CHECKPOINT_SHA,
                    EXPECTED_COMMIT, LAST_DIR, LAST_EXTRACTOR, LOAD_AUDIT, MEAN_DIR, MEAN_EXTRACTOR, OUT, SEEDS, SOURCE,
                    SOURCE_MANIFEST, TIE_TOLERANCE_EV, E13Error, get_logger, load_inputs, log, sha256_file, write_json)

LOGGER = get_logger("matgpt_e1_3_prepare")
INNER_SEED = 13072026


def strict_backbone_audit() -> dict[str, Any]:
    """Carga el decoder oficial strict=True, congela parámetros y verifica congelamiento.

    Returns:
        Hash, arquitectura y número de parámetros congelados.
    """
    import importlib.util
    import sys
    import yaml

    source_path = str(SOURCE.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from utils.models import ModifiedTransformerDecoder

    info = json.loads(CHECKPOINT.with_name("model_info_property_0.json").read_text(encoding="utf-8"))
    stoi = {str(token): int(index) for token, index in info["stoi"].items()}
    config = yaml.safe_load((SOURCE / "config" / "uncond.yaml").read_text(encoding="utf-8"))
    mc, tc = config["model"], config["training"]
    if len(stoi) != 172 or int(info["block_size"]) != 442 or (mc.get("n_embd"), mc.get("n_layer"), mc.get("n_head")) != (768, 12, 12):
        raise E13Error("Metadatos/arquitectura no corresponden a E0 (vocab=172, block=442, hidden=768, layers/heads=12).")
    if tc.get("conditioning_mode") != "unconditional" or int(tc.get("num_properties", -1)) != 0:
        raise E13Error("La configuración local no es unconditional.")
    model = ModifiedTransformerDecoder(vocab_size=len(stoi), n_embd=768, n_head=12, n_layer=12, block_size=442,
        dropout=float(mc["dropout"]), stoi=stoi, num_properties=0, conditioning_mode="unconditional").cpu()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise E13Error(f"El backbone conserva parámetros entrenables: {trainable[:10]}")
    return {"strict_load": True, "checkpoint_sha256": sha256_file(CHECKPOINT), "source_commit": EXPECTED_COMMIT,
            "vocab_size": len(stoi), "block_size": int(info["block_size"]), "hidden_size": 768,
            "n_layer": 12, "n_head": 12, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": 0, "frozen": True}


def build_assignments(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Asigna GroupKFold a development y holdout interior fijo a cada fold.

    Args:
        data: Dataset E1 con columna `composition_group` validada.

    Returns:
        Una fila por structure_id, mapa de rol/fold y resumen de tamaños.
    """
    dev = data[data.split.isin(["train", "val"])].copy().reset_index(drop=True)
    test = data[data.split.eq("test")].copy().reset_index(drop=True)
    group_kfold = GroupKFold(n_splits=5)
    outer_by_group: dict[str, int] = {}
    for fold, (_, valid_idx) in enumerate(group_kfold.split(dev, groups=dev.composition_group), start=0):
        for group in dev.iloc[valid_idx].composition_group.unique():
            if group in outer_by_group:
                raise E13Error(f"Grupo asignado a más de un outer fold: {group}")
            outer_by_group[str(group)] = fold
    if len(outer_by_group) != 70:
        raise E13Error(f"GroupKFold asignó {len(outer_by_group)} grupos de desarrollo, esperado 70.")
    records: list[dict[str, Any]] = []
    fold_summary: dict[str, Any] = {}
    for fold in range(5):
        valid_groups = {group for group, assigned in outer_by_group.items() if assigned == fold}
        train_groups = set(outer_by_group) - valid_groups
        rng = np.random.default_rng(INNER_SEED + fold)
        sorted_train_groups = np.asarray(sorted(train_groups), dtype=object)
        inner_count = max(1, int(round(len(sorted_train_groups) * 0.20)))
        inner_val_groups = set(rng.permutation(sorted_train_groups)[:inner_count].tolist())
        if inner_val_groups & valid_groups or inner_val_groups & set(outer_by_group).difference(train_groups):
            raise E13Error(f"Inner holdout comparte grupos con outer validation en fold {fold}.")
        train_rows = dev[dev.composition_group.isin(train_groups)]
        val_rows = dev[dev.composition_group.isin(valid_groups)]
        inner_rows = train_rows[train_rows.composition_group.isin(inner_val_groups)]
        inner_train_rows = train_rows[~train_rows.composition_group.isin(inner_val_groups)]
        if (len(val_rows), len(valid_groups), len(train_rows), len(train_groups), len(inner_rows), len(inner_val_groups)) != (42, 14, 168, 56, 33, inner_count):
            raise E13Error(f"Conteo inesperado en fold {fold}.")
        fold_summary[str(fold)] = {"outer_train_groups": len(train_groups), "outer_train_structures": len(train_rows),
            "outer_validation_groups": len(valid_groups), "outer_validation_structures": len(val_rows),
            "inner_train_groups": int(inner_train_rows.composition_group.nunique()), "inner_train_structures": len(inner_train_rows),
            "inner_validation_groups": len(inner_val_groups), "inner_validation_structures": len(inner_rows),
            "inner_seed": INNER_SEED + fold}
        # Una fila por estructura y fold: su rol inner cambia según qué outer
        # fold se esté evaluando. El test aparece una sola vez con fold=-1.
        for row in dev.itertuples(index=False):
            role = "outer_validation" if row.composition_group in valid_groups else "outer_train"
            inner_role = "not_applicable" if role == "outer_validation" else ("inner_validation" if row.composition_group in inner_val_groups else "inner_train")
            records.append({"structure_id": str(row.structure_id), "composition_group": row.composition_group,
                "historical_split": row.split, "outer_fold": fold, "outer_role": role, "inner_role": inner_role})
    for row in test.itertuples(index=False):
        records.append({"structure_id": str(row.structure_id), "composition_group": row.composition_group,
            "historical_split": "test", "outer_fold": -1, "outer_role": "historical_test_excluded", "inner_role": "not_applicable"})
    assignments = pd.DataFrame(records)
    if assignments.duplicated(["structure_id", "outer_fold"]).any():
        raise E13Error("La tabla de asignación contiene duplicados structure_id/fold.")
    dev_assignments = assignments[assignments.historical_split.isin(["train", "val"])]
    test_assignments = assignments[assignments.historical_split.eq("test")]
    if len(dev_assignments) != 210 * 5 or dev_assignments.groupby("structure_id").outer_fold.nunique().ne(5).any():
        raise E13Error("Cada estructura de desarrollo debe tener una fila en cada uno de los cinco folds.")
    if len(test_assignments) != 24 or test_assignments.structure_id.duplicated().any():
        raise E13Error("El test histórico debe aparecer una vez por estructura, siempre excluido.")
    return assignments, fold_summary


def run(out_dir: Path) -> None:
    """Valida fuentes, asigna grupos, y conserva hashes/configuración sin entrenar."""
    if out_dir.exists() and any(path.name != "preflight_audit.json" for path in out_dir.iterdir()):
        raise E13Error(f"El destino ya contiene artefactos ajenos al preflight; conserva el histórico y usa un directorio nuevo: {out_dir}")
    data, _, provenance = load_inputs()
    preflight_path = out_dir / "preflight_audit.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        model_audit = preflight.get("checkpoint", {})
        if preflight.get("status") != "PASS" or model_audit.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
            raise E13Error("Preflight ausente, incompleto o hecho con otro checkpoint; vuelve a ejecutar audit_cv.py.")
        for pooling, audit in provenance["embed_audits"].items():
            previous = preflight.get("embedding_audits", {}).get(pooling, {})
            if previous.get("cache_sha256") != audit.get("cache_sha256") or previous.get("metadata_sha256") != audit.get("metadata_sha256"):
                raise E13Error(f"El caché {pooling} cambió después del preflight; vuelve a auditar.")
        for key, hashes in provenance["e12_hashes"].items():
            if preflight.get("e12_hashes", {}).get(key) != hashes:
                raise E13Error(f"Resultados E1.2 {key} cambiaron después del preflight.")
    else:
        model_audit = strict_backbone_audit()
    assignments, folds = build_assignments(data)
    output = out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    # Conserva colisiones detectadas; no elimina filas ni estructura alguna.
    collision_mask = data.matgpt_slices.astype(str).duplicated(keep=False)
    collisions = data.loc[collision_mask, ["structure_id", "composition_group", "split", "polymorph_family", "matgpt_slices"]].copy()
    collisions.to_csv(output / "sequence_collisions.csv", index=False)
    config = {"experiment": "matgpt_e1_3", "candidates": CANDIDATES, "seeds": list(SEEDS),
        "outer_cv": {"method": "GroupKFold", "n_splits": 5, "groups": "canonical full chemical composition", "development": "historical train + val", "development_structures": 210, "development_compositions": 70, "historical_test_excluded_structures": 24, "historical_test_excluded_compositions": 8},
        "inner_validation": {"method": "deterministic seeded group holdout", "fraction": 0.20, "seed_base": INNER_SEED, "fold_seed": "seed_base + zero-based outer fold", "purpose": "early stopping only"},
        "training": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "max_epochs": 200, "patience": 20, "compositions_per_batch": 8, "absolute_loss": "smooth_l1_loss(beta=1.0) of normalized residual", "target_scale": "std and mean of inner-train targets only; std floor 1e-6", "pairwise_loss": "per-composition mean smooth_l1_loss(beta=1.0) of within-group pair residuals normalized by inner-train RMS pair difference; floor 0.001 eV/atom", "checkpoint_selection": "lowest inner-validation objective; no outer fold used for selection", "backbone_trainable_parameters": 0},
        "metrics": {"energy": "eV/atom", "relative_energy": "meV/atom", "tie_tolerance_ev_atom": TIE_TOLERANCE_EV, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED},
        "limits": ["E1 historical test is excluded entirely from E1.3.", "E1.1/E1.2 candidates were selected using the same eight historical validation compositions; E1.3 estimates sensitivity/robustness and does not remove selection bias.", "E1 historical test was previously inspected and is not blind.", "E1.2 geometry diagnostics included all 234 structures; their test-containing correlations are exploratory and did not select E1.3 folds.", "DFT labels are static GGA total energies per atom, not formation energies."]}
    (output / "config.yaml").write_text(__import__("yaml").safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    src_paths = [SOURCE / "config" / "uncond.yaml", SOURCE / "utils" / "models.py", SOURCE / "utils" / "dataset.py", LAST_EXTRACTOR, MEAN_EXTRACTOR,
                 SOURCE_MANIFEST, LOAD_AUDIT, CHECKPOINT, CHECKPOINT.with_name("model_info_property_0.json"), DATASET,
                 ALT / "reports" / "matgpt_e1_v2" / "split_manifest.json"]
    source_hashes = {str(path.relative_to(ALT) if path.is_relative_to(ALT) else path.name): sha256_file(path) for path in src_paths}
    source_hashes.update({f"{kind}_cache": provenance["embed_audits"][kind]["cache_sha256"] for kind in ("last_token", "masked_mean")})
    provenance_json = {"status": "PASS", "resolved_project_root": str(ALT.resolve()), "resolved_inputs": {"dataset": str(DATASET.resolve()), "historical_split_manifest": str((ALT / "reports/matgpt_e1_v2/split_manifest.json").resolve()), "last_token_cache": str((LAST_DIR / "matgpt_uncond_last_token.npz").resolve()), "masked_mean_cache": str((MEAN_DIR / "matgpt_uncond_masked_mean.npz").resolve()), "checkpoint": str(CHECKPOINT.resolve())},
        "source_hashes": source_hashes, "e0_strict_backbone_audit": model_audit, "e12_result_hashes": provenance["e12_hashes"],
        "embedding_audits": provenance["embed_audits"],
        "extractor_code_verification": provenance["extractor_verification"],
        "split_manifest": provenance["manifest"],
        "e13_pipeline_script_hashes": {path.name: sha256_file(path) for path in (Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("train_cv_heads.py"), Path(__file__).with_name("evaluate_cv.py"), Path(__file__).with_name("audit_cv.py"))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "folds": folds, "fold_assignments_sha256": sha256_file(output / "fold_assignments.csv"),
        "sequence_collision_rows": int(len(collisions)), "historical_test_metrics_used": False,
        "e1_2_all_234_geometry_correlations": "Exploratory only; never used in E1.3 candidate or fold selection."}
    write_json(output / "provenance.json", provenance_json)
    write_json(output / "integrity_checks.json", {"status": "PASS", "preflight": "PASS", "checkpoint": model_audit,
        "development_ids": 210, "development_compositions": 70, "historical_test_ids_excluded": 24,
        "historical_test_compositions_excluded": 8, "dev_test_id_overlap": 0, "dev_test_composition_overlap": 0,
        "fold_summary": folds, "collisions_logged": int(len(collisions)), "training_started": False})
    log(LOGGER, logging.INFO, "e13_folds_prepared", structures=210, compositions=70, outer_folds=5,
        inner_seed_base=INNER_SEED, collision_rows=len(collisions), out_dir=output)


def main() -> None:
    """Valida CLI y prepara el manifiesto congelado antes de entrenar cabezas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except (E13Error, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        log(LOGGER, logging.ERROR, "e13_prepare_failed", error=str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
