#!/usr/bin/env python3
"""Audita cobertura y procedencia de las salidas CV D1–D3, sin leer métricas de test."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickletools
import zipfile
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development, sha256, write_json

LOG = logging.getLogger("matgpt_diagnostics.audit")
EXPECTED_CHECKPOINT = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"


def audit_d2(output: Path) -> dict[str, object]:
    """Verifica que CV D2 cubra cada estructura/fold/semilla una vez.

    Args:
        output: Directorio de salida CV D2.
    Returns:
        Diccionario JSON con verificaciones y recuentos observados.
    Raises:
        ValueError: Si hay faltantes, duplicados, fuga de test o artefactos inválidos.
    """
    data, folds, meta = load_development()
    pred = pd.read_csv(output / "oof_predictions_cv.csv")
    history = pd.read_csv(output / "training_history_cv.csv")
    expected_ids = set(data.structure_id.astype(str))
    configs = ["D2-CENTER", "D2-CENTER-ONLY"]
    seeds = [42, 123, 2026]
    if len(pred) != 1260 or pred.duplicated(["configuration", "seed", "structure_id"]).any():
        raise ValueError("OOF debe contener 1,260 filas únicas por configuración, semilla e ID.")
    if not set(pred.configuration).issubset(set(configs)) or set(pred.seed.astype(int)) != set(seeds):
        raise ValueError("Configuraciones o semillas inesperadas en D2 OOF.")
    for config in configs:
        for seed in seeds:
            subset = pred[(pred.configuration == config) & (pred.seed == seed)]
            if len(subset) != 210 or set(subset.structure_id.astype(str)) != expected_ids:
                raise ValueError(f"Cobertura OOF incompleta: {config}, seed={seed}.")
            if subset.groupby("structure_id").fold.nunique().ne(1).any():
                raise ValueError(f"Un ID tiene más de un fold: {config}, seed={seed}.")
            if subset.groupby("composition_group").fold.nunique().ne(1).any():
                raise ValueError(f"Fuga de composición entre folds: {config}, seed={seed}.")
            if subset.groupby("composition_group").size().ne(3).any():
                raise ValueError(f"Hay composiciones sin sus tres polimorfos: {config}, seed={seed}.")
    historical_test = pd.read_csv(ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv").query("split == 'test'")
    if set(pred.structure_id.astype(str)) & set(historical_test.structure_id.astype(str)):
        raise ValueError("OOF incluye estructuras del test histórico.")
    if set(pred.reduced_formula.astype(str)) & set(historical_test.reduced_formula.astype(str)):
        raise ValueError("OOF comparte composiciones con el test histórico.")
    if not pred.energy_pred_ev_atom.map(pd.notna).all() or pred.energy_pred_ev_atom.map(lambda x: abs(float(x)) == float("inf")).any():
        raise ValueError("Predicciones OOF contienen NaN/Inf.")
    expected_runs = {(c, f, s) for c in configs for f in range(5) for s in seeds}
    history_runs = set(map(tuple, history[["configuration", "fold", "seed"]].drop_duplicates().itertuples(index=False, name=None)))
    if history_runs != expected_runs:
        raise ValueError("El historial no contiene las 30 combinaciones configuración×fold×semilla.")
    checkpoints = sorted((output / "checkpoints").rglob("*.pt"))
    if len(checkpoints) != 30:
        raise ValueError(f"Se esperaban 30 checkpoints ligeros; encontrados {len(checkpoints)}.")
    for checkpoint in checkpoints:
        # Inspecciona el contenedor sin cargar torch: en este entorno importar
        # torch junto con numpy/sklearn puede abortar por runtimes OpenMP dobles.
        with zipfile.ZipFile(checkpoint) as archive:
            pickle_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
            if pickle_name is None:
                raise ValueError(f"No hay state_dict serializado en {checkpoint.name}.")
            serialized = archive.read(pickle_name)
        strings = {str(arg) for opcode, arg, _ in pickletools.genops(serialized)
                   if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}}
        if EXPECTED_CHECKPOINT not in strings:
            raise ValueError(f"Referencia de checkpoint no coincide en {checkpoint.name}.")
        if not {"last_block", "head"}.issubset(strings) or {"model", "backbone"} & strings:
            raise ValueError(f"El archivo no parece contener solo el último bloque y cabeza: {checkpoint.name}.")
    timing = history.groupby(["configuration", "fold", "seed"]).elapsed_seconds.max()
    return {
        "status": "PASS", "historical_test_used": False,
        "dataset_sha256": meta["data_sha256"], "fold_manifest_sha256": meta["folds_sha256"],
        "n_oof_rows": int(len(pred)), "n_structures_per_configuration_seed": 210,
        "n_compositions_per_configuration_seed": 70, "n_folds": 5, "seeds": seeds,
        "n_fits_completed": len(history_runs), "n_checkpoints": len(checkpoints),
        "checkpoint_sha256": EXPECTED_CHECKPOINT, "all_oof_finite": True,
        "all_composition_groups_stay_in_one_outer_fold": True,
        "test_ids_and_formulas_excluded": True,
        "sum_fit_wall_seconds": float(timing.sum()),
        "mean_fit_wall_seconds": float(timing.mean()),
        "current_trainer_source_sha256": sha256(Path(__file__).with_name("d2_centered_training.py")),
        "exact_trainer_source_hash_captured_at_run_start": False,
        "source_provenance_note": (
            "El proceso CV ya estaba iniciado cuando se añadieron al archivo del trainer campos de tiempo y una guarda de salida. "
            "La función objetivo, los folds, las semillas, el optimizador y la selección de checkpoint no cambiaron; "
            "el hash registrado identifica el archivo actual y no prueba el bytecode cargado al inicio."
        ),
        "fold_compositions": {str(int(f)): int(v) for f,v in folds.loc[folds.outer_role.eq("outer_validation")].groupby("outer_fold").composition_group.nunique().items()},
    }


def main() -> None:
    """Ejecuta auditoría final de cobertura y escribe su JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-cv", type=Path, default=REPORT / "D2/cv_v1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = audit_d2(args.d2_cv)
    destination = args.out or args.d2_cv / "integrity_checks_cv.json"
    write_json(destination, result)
    LOG.info("diagnostics_audit_passed fits=%d checkpoints=%d rows=%d output=%s",
             result["n_fits_completed"], result["n_checkpoints"], result["n_oof_rows"], destination)


if __name__ == "__main__":
    main()
