#!/usr/bin/env python3
"""Busca solo el CSV original nombrado por la configuración MatGPT."""

from __future__ import annotations

import json

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import ALT, REPORT_DIR, configure_logging, write_json


def main() -> None:
    """Registra coincidencias exactas dentro de Alternative2."""
    configure_logging()
    paths = sorted(ALT.rglob("Training_dataset_alex20_with_encoding.csv"))
    result = {"status": "FOUND" if paths else "BLOCKED", "matches": [str(p) for p in paths],
              "substitutes_used": [], "search_root": str(ALT)}
    write_json(REPORT_DIR / "provenance" / "corpus_location.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
