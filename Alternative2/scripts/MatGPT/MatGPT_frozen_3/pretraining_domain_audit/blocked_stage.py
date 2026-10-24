#!/usr/bin/env python3
"""Comprueba prerrequisitos para análisis dependientes del corpus original."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR, configure_logging, write_json


def run_stage(stage: str, argv: Sequence[str] | None = None) -> int:
    """Revisa que exista salida válida de corpus antes de una fase posterior.

    Args:
        stage: Nombre descriptivo de la auditoría dependiente.
        argv: Argumentos opcionales de línea de comandos.

    Returns:
        Código de salida 0 si hay prerequisitos; 2 si falta el corpus auditado.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description=f"Preflight de la fase {stage}.")
    parser.add_argument("--corpus", type=Path, help="CSV exacto Training_dataset_alex20_with_encoding.csv")
    args = parser.parse_args(argv)
    schema = REPORT_DIR / "corpus_statistics" / "corpus_schema.json"
    ready = schema.is_file() and args.corpus is not None and args.corpus.is_file() and args.corpus.name == "Training_dataset_alex20_with_encoding.csv"
    result = {"stage": stage, "status": "READY_FOR_DATA_PHASE" if ready else "BLOCKED",
              "corpus_schema_path": str(schema), "corpus_path": str(args.corpus) if args.corpus else None,
              "reason": "Falta validar el corpus original; no se calcularon métricas ni se usó un sustituto." if not ready else "Prerequisitos mínimos presentes; ejecutar la fase completa solo después de inspeccionar el manifiesto."}
    write_json(REPORT_DIR / "provenance" / f"{stage}_preflight.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(run_stage("stage"))
