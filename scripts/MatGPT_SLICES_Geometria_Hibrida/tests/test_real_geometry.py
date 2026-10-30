"""Pruebas de identidad de las 210 geometrías reales ABS3 exportadas."""
from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from pymatgen.core import Composition, Structure


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
JSONL = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl"


class RealGeometryTests(unittest.TestCase):
    """Bloquea correspondencia física, N y fuente homogénea sin abrir test."""

    def test_exact_ids_atoms_composition_and_source(self) -> None:
        """Cada ID de desarrollo aparece una vez con estructura original válida."""
        with DATASET.open(newline="", encoding="utf-8-sig") as stream:
            rows = {row["structure_id"]: row for row in csv.DictReader(stream)
                    if row["split"] in ("train", "val")}
        structures = [json.loads(line) for line in JSONL.read_text().splitlines() if line]
        self.assertEqual(len(rows), 210)
        self.assertEqual(len(structures), 210)
        self.assertEqual({row["structure_id"] for row in structures}, set(rows))
        self.assertEqual({row["geometry_source"] for row in structures}, {"gga-static_endpoint_cleaned"})
        for row in structures:
            structure = Structure.from_dict(row["structure"])
            reference = rows[row["structure_id"]]
            self.assertEqual(len(structure), int(reference["num_atoms"]))
            self.assertEqual(structure.composition.reduced_composition,
                             Composition(reference["reduced_formula"]).reduced_composition)
            self.assertEqual(int(row["source_index"]), int(reference["source_index"]))
            self.assertNotIn("energy", row)


if __name__ == "__main__":
    unittest.main()
