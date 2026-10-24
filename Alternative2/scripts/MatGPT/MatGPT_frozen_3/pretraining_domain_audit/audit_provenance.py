#!/usr/bin/env python3
"""Ejecuta únicamente la verificación local de procedencia de MatGPT."""

from __future__ import annotations

import json

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import REPORT_DIR, audit_provenance, configure_logging, write_json


def main() -> None:
    """Verifica y guarda los hashes del checkpoint y fuentes."""
    configure_logging()
    result = audit_provenance()
    path = REPORT_DIR / "provenance" / "provenance.json"
    write_json(path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
