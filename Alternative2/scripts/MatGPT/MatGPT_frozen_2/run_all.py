#!/usr/bin/env python3
"""Orquesta H0/H1/H2 con fases reanudables y nunca accede al test histórico."""
from __future__ import annotations

import argparse
import logging
import sys
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import MODELS, REPORT_DIR, SEEDS, ExperimentError, log
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.prepare_experiment import run as preflight
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.extract_frozen_features import extract
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.train_grouped_cv import run_all as train
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.evaluate_energy_conventions import run as evaluate


def run_phase(phase: str, out_dir: Path) -> None:
    """Ejecuta las etapas pedidas, con smoke/pilot incluidos en las 45 corridas.

    Args:
        phase: preflight, features, tests, smoke, pilot, full o all.
        out_dir: Carpeta versionada del informe.
    """
    if phase == "all":
        for stage in ("preflight", "features", "tests", "smoke", "pilot", "full"):
            run_phase(stage, out_dir)
        return
    if phase == "preflight":
        preflight(out_dir)
    if phase == "features":
        extract(out_dir / "features", "cpu", 4)
    if phase == "tests":
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/"tests"))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise ExperimentError("Las pruebas matemáticas sintéticas no pasaron; entrenamiento detenido.")
    if phase == "smoke":
        # Las corridas smoke son las primeras tres corridas de la CV completa,
        # con el protocolo final; no se reinician ni se seleccionan distinto.
        train(out_dir, max_epochs=200, selection=MODELS, folds_to_run=(0,), seeds_to_run=(42,))
    if phase == "pilot":
        train(out_dir, max_epochs=200, selection=MODELS, folds_to_run=(0,), seeds_to_run=(123,2026))
    if phase == "full":
        train(out_dir, max_epochs=200)
        evaluate(out_dir, bootstrap_replicates=2000)


def main() -> None:
    """Valida fase y dispara la parte explícitamente solicitada."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("preflight","features","tests","smoke","pilot","full","all"),default="all")
    parser.add_argument("--out-dir",type=Path,default=REPORT_DIR)
    args=parser.parse_args()
    try:
        run_phase(args.phase,args.out_dir)
        log(logging.INFO,"phase_complete",phase=args.phase,out_dir=args.out_dir.resolve(),test=False)
    except (ExperimentError,OSError,ValueError,KeyError,RuntimeError,ImportError) as exc:
        log(logging.ERROR,"phase_failed",phase=args.phase,error=str(exc))
        raise SystemExit(2) from exc


if __name__=="__main__":
    main()
