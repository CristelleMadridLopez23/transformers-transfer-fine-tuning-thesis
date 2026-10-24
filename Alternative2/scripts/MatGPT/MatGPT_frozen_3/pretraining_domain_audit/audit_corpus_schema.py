#!/usr/bin/env python3
"""Valida esquema del CSV original MatGPT indicado por ruta explícita."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import AuditError, audit_corpus, configure_logging


def main() -> None:
    """Audita las columnas y validez de SLICES del corpus."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit_corpus(args.corpus), indent=2, ensure_ascii=False))
    except (AuditError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
