#!/usr/bin/env python3
"""Parse a SMILES string with Open Babel and emit its atom/bond graph as JSON."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    return parser.parse_args()


def main() -> None:
    try:
        from openbabel import openbabel as ob
    except ImportError as exc:
        raise SystemExit(
            "Open Babel is missing. Install transferability/requirements.txt."
        ) from exc

    args = parse_args()
    conversion = ob.OBConversion()
    if not conversion.SetInFormat("smi"):
        raise SystemExit("Open Babel SMILES reader is unavailable")

    molecule = ob.OBMol()
    if not conversion.ReadString(molecule, args.smiles):
        raise SystemExit("Open Babel could not parse the supplied SMILES")

    result = {
        "atoms": [
            {
                "index": atom.GetIdx(),
                "element": ob.GetSymbol(atom.GetAtomicNum()),
            }
            for atom in ob.OBMolAtomIter(molecule)
        ],
        "bonds": [
            {
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx(),
                "order": bond.GetBondOrder(),
            }
            for bond in ob.OBMolBondIter(molecule)
        ],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
