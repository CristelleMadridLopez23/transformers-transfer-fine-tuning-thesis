#!/usr/bin/env python3
"""Audit matched static/relax DFT metadata for development compositions only."""
from __future__ import annotations

import argparse
import json
import hashlib
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
RAW = ALT / "01_data/raw_data/results(1).json"
DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
OUT = ALT / "reports/MatGPT_finetuning/E2-T1"
CACHE = ALT / "reports/MatGPT_finetuning/matgpt_e2_t0/frame_tokenization.csv"
FOLDS = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
CHECKPOINT = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
AUDIT_SCRIPT = ALT / "scripts/MatGPT_finetuning/E2-T1/audit_relax_entries.py"
SHARED_SETTINGS = ("GGA", "LDAU", "ENCUT", "EDIFF", "ISPIN")
RELAX_SETTINGS = (*SHARED_SETTINGS, "ISMEAR", "SIGMA", "KSPACING", "ISIF", "NSW")


def file_sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def potcar_signature(meta: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Return element-to-pseudopotential names from one VASP metadata record."""
    potcars = meta.get("potcar", {}) or {}
    return {str(element): (str(value.get("name", "")), str(value.get("pp", "")))
            for element, value in potcars.items() if isinstance(value, dict)}


def audit(raw_path: Path, dataset_path: Path, out: Path) -> dict[str, Any]:
    """Compare DFT metadata and static labels to matching relax tasks in development.

    Args:
        raw_path: Raw Materials Project export.
        dataset_path: E1 static table with exact source_task IDs and split.
        out: Existing E2-T1 output directory.
    Returns:
        Audit payload describing metadata compatibility and limitations.
    """
    if not raw_path.is_file() or not dataset_path.is_file() or not (out / "decision.json").is_file():
        raise FileNotFoundError("Requiere raw data, dataset estático y decisión de estructura T1.")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    data = pd.read_csv(dataset_path, low_memory=False)
    dev = data[data.split.isin(["train", "val"])].copy()
    relax_incar_values: dict[str, set[str]] = {key: set() for key in ("GGA", "LDAU", "ENCUT", "EDIFF", "EDIFFG", "ISIF", "ISPIN", "ISMEAR", "SIGMA", "KSPACING", "NSW")}
    dev_potcars: set[str] = set()
    dev_metadata_missing = 0
    rows: list[dict[str, Any]] = []
    metadata_missing: list[str] = []
    for row in dev.itertuples(index=False):
        static_key = str(row.source_key)
        relax_key = static_key.replace("gga-static", "gga-relax")
        static_record, relax_record = raw.get(static_key), raw.get(relax_key)
        if static_record is None or relax_record is None:
            metadata_missing.append(static_key)
            continue
        smeta, rmeta = static_record.get("meta", {}), relax_record.get("meta", {})
        sinc, rinc = smeta.get("incar", {}) or {}, rmeta.get("incar", {}) or {}
        if not rinc or not (rmeta.get("potcar", {}) or {}):
            dev_metadata_missing += 1
        for key in relax_incar_values:
            relax_incar_values[key].add(str(rinc.get(key, "MISSING")))
        for element_data in (rmeta.get("potcar", {}) or {}).values():
            if isinstance(element_data, dict):
                dev_potcars.add(str(element_data.get("name", "MISSING")) + ":" + str(element_data.get("pp", "MISSING")))
        shared = {key: {"static": sinc.get(key), "relax": rinc.get(key),
                        "equal": sinc.get(key) == rinc.get(key)} for key in SHARED_SETTINGS}
        same_potcars = potcar_signature(smeta) == potcar_signature(rmeta)
        all_paw_pbe = bool(potcar_signature(smeta)) and all(name == "PAW_PBE" for name, _ in potcar_signature(smeta).values())
        static_energy = static_record.get("results", {}).get("E_per_at")
        cached_energy = float(row.energy_per_atom_ev)
        label_diff = abs(float(static_energy) - cached_energy) if static_energy is not None else float("inf")
        missing_or_diff = any(not value["equal"] for value in shared.values())
        rows.append({
            "structure_id": str(row.structure_id), "composition_group": str(row.composition_group) if hasattr(row, "composition_group") else str(row.reduced_formula),
            "source_static_task": static_key, "source_relax_task": relax_key,
            "historical_split": str(row.split), "shared_settings": shared,
            "shared_settings_equal": not missing_or_diff, "same_potcar_signature": same_potcars,
            "static_potcar_all_PAW_PBE": all_paw_pbe, "static_energy_per_atom_ev_raw": static_energy,
            "static_relax_pbe_no_u": (sinc.get("GGA") == "Pe" and rinc.get("GGA") == "Pe"
                                       and sinc.get("LDAU") is False and rinc.get("LDAU") is False),
            "static_relax_ismear": {"static": sinc.get("ISMEAR"), "relax": rinc.get("ISMEAR"),
                                     "same": sinc.get("ISMEAR") == rinc.get("ISMEAR")},
            "relax_method_settings": {key: rinc.get(key) for key in RELAX_SETTINGS},
            "static_energy_per_atom_ev_dataset": cached_energy, "static_label_abs_diff_ev_atom": label_diff,
            "raw_static_result_keys": sorted(static_record.get("results", {}).keys()),
            "raw_relax_result_keys": sorted(relax_record.get("results", {}).keys()),
        })
    mismatches = [row for row in rows if not row["shared_settings_equal"] or not row["same_potcar_signature"]
                  or not row["static_potcar_all_PAW_PBE"] or not row["static_relax_pbe_no_u"]
                  or row["static_label_abs_diff_ev_atom"] > 1e-7]
    method_setting_counts: dict[str, dict[str, int]] = {}
    for setting in RELAX_SETTINGS:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row["relax_method_settings"].get(setting))
            counts[value] = counts.get(value, 0) + 1
        method_setting_counts[setting] = counts
    payload = {
        "status": "PASS" if len(rows) == len(dev) and not mismatches else "BLOCKED",
        "development_static_relax_pairs": len(rows), "expected_development_structures": int(len(dev)),
        "metadata_missing_ids": metadata_missing, "mismatch_ids": [row["structure_id"] for row in mismatches],
        "shared_settings_checked": list(SHARED_SETTINGS), "all_shared_settings_equal": not any(not row["shared_settings_equal"] for row in rows),
        "relax_task_setting_value_counts": method_setting_counts,
        "static_relax_ismear_differences": int(sum(not row["static_relax_ismear"]["same"] for row in rows)),
        "static_relax_ismear_values": sorted({json.dumps(row["static_relax_ismear"], sort_keys=True) for row in rows}),
        "all_static_relax_potcars_match": all(row["same_potcar_signature"] for row in rows),
        "all_potcars_are_PAW_PBE": all(row["static_potcar_all_PAW_PBE"] for row in rows),
        "max_static_label_abs_diff_ev_atom": max((row["static_label_abs_diff_ev_atom"] for row in rows), default=None),
        "raw_result_exports_expose_energy_fields": sorted({key for row in rows for key in row["raw_static_result_keys"]}),
        "correction_metadata_note": "Los pares static/relax se cotejan en PBE, LDAU, ENCUT, EDIFF, ISPIN y PAW_PBE. ISMEAR/KSPACING y otros ajustes pueden diferir entre cálculos separados y se reportan por tarea; no se combinan energías absolutas. L_traj usa diferencias dentro de una única relajación, cuyos ajustes permanecen constantes por tarea. El export no expone una descomposición independiente de otras correcciones energéticas.",
        "historical_test_used": False,
    }
    dev_pbe = bool(relax_incar_values["GGA"] == {"Pe"} and relax_incar_values["LDAU"] == {"False"}
                   and dev_potcars and all(value.startswith("PAW_PBE:") for value in dev_potcars)
                   and dev_metadata_missing == 0)
    dev_relax_metadata = {
        "n_relax_tasks": int(len(dev)), "missing_incar_or_potcar": dev_metadata_missing,
        "incar_unique_values": {key: sorted(values) for key, values in relax_incar_values.items()},
        "potcar_identifiers": sorted(dev_potcars), "all_pbe_without_u": dev_pbe,
        "reference_check": "Metadatos solo de las 210 tareas relax enlazadas a train+validation; las tareas de test se excluyeron antes de inspección detallada.",
    }
    (out / "metadata_compatibility.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([{**row, "shared_settings": json.dumps(row["shared_settings"], sort_keys=True)} for row in rows]).to_csv(out / "metadata_compatibility.csv", index=False)
    decision_path = out / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["n_development_relax_tasks"] = int(len(dev))
    decision["n_relax_tasks"] = int(len(dev))
    decision["pbe_metadata"] = dev_relax_metadata
    decision["static_relax_metadata_compatibility"] = payload
    decision["test_historical_used_for_training_or_model_selection"] = False
    eligible_counts = decision.get("development_compositions_with_candidates_by_fold", {})
    min_eligible = min((int(value) for value in eligible_counts.values()), default=0)
    collision_groups = int(decision.get("n_collision_groups", 0))
    high_energy_collisions = int(decision.get("n_collision_groups_range_ge_1mev_atom", 0))
    collision_fraction = high_energy_collisions / collision_groups if collision_groups else 0.0
    if payload["status"] != "PASS" or not dev_pbe or not decision.get("all_frames_reencoded", False):
        gate = "BLOCKED"
    elif min_eligible < 10:
        gate = "LIMITED"
    elif collision_fraction > 0.5 or float(decision.get("unique_token_ratio", 0.0)) < 0.5:
        gate = "LIMITED"
    else:
        gate = "GO"
    decision["collision_groups_range_ge_1mev_fraction"] = collision_fraction
    decision["operational_gate_note"] = "Gate LIMITED if >50% of repeated-ID groups span >=1 meV/atom or global unique-ID ratio <0.5; this is a conservative project rule, not a physical threshold. LIMITED permits pilot only."
    decision["status"] = gate
    decision["t2_training_gate"] = gate
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "raw_sha256": file_sha256(raw_path), "dataset_sha256": file_sha256(dataset_path),
        "frame_cache_sha256": file_sha256(CACHE), "fold_assignments_sha256": file_sha256(FOLDS),
        "checkpoint_sha256": file_sha256(CHECKPOINT), "audit_script_sha256": file_sha256(AUDIT_SCRIPT),
        "python": platform.python_version(), "source_commit": "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e",
        "historical_test_structure_conversion": False,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Run the read-only provenance comparison and update the T1 decision record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.dataset, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
