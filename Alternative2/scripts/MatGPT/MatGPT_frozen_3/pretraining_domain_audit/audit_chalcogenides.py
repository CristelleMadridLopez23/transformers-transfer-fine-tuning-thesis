#!/usr/bin/env python3
"""Muestra conteos empíricos S/Se/Te generados desde el corpus validado."""

from __future__ import annotations

import json

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR

def main() -> None:
    """Imprime resumen S/Se/Te; falla claramente si falta el corpus."""
    path = REPORT_DIR / "corpus_statistics" / "chalcogenide_statistics.json"
    if not path.is_file():
        raise SystemExit("BLOCKED: primero audita el corpus exacto con audit_corpus_schema.py.")
    print(path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
