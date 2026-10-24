#!/usr/bin/env python3
"""Prueba el parser de composición SLICES en registros aislados."""

from __future__ import annotations

import argparse
import json

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import AuditError, parse_slices_composition, reduced_formula


def main() -> None:
    """Extrae composición desde el bloque inicial de especies de una cadena."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices", help="Cadena SLICES completa entre comillas")
    args = parser.parse_args()
    try:
        counts = parse_slices_composition(args.slices)
    except AuditError as exc:
        parser.error(str(exc))
    print(json.dumps({"counts": counts, "formula_reduced_key": reduced_formula(counts)}, indent=2))


if __name__ == "__main__":
    main()
