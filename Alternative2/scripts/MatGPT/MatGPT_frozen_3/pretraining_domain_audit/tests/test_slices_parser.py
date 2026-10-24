"""Pruebas sintéticas del parser SLICES, aisladas de resultados científicos."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.pretraining_domain_audit.domain_audit import AuditError, parse_slices_composition, reduced_formula


class SlicesParserTests(unittest.TestCase):
    """Pruebas del bloque de sitios y validación de aristas."""

    def test_counts_barium_titanium_sulfide_and_indices_above_nine(self) -> None:
        """Los índices de nodos de aristas no se interpretan como átomos."""
        counts = parse_slices_composition("Ba Ba Ba Ba Ti Ti Ti Ti S S S S S S S S S S S S 0 10 ooo 10 11 +o-")
        self.assertEqual(counts, {"Ba": 4, "Ti": 4, "S": 12})
        self.assertEqual(reduced_formula(counts), "BaTiS3")

    def test_reduced_formula_and_repeated_species(self) -> None:
        """La composición se reduce por máximo común divisor."""
        self.assertEqual(reduced_formula({"Ba": 2, "Ti": 2, "S": 6}), "BaTiS3")

    def test_required_composition_examples(self) -> None:
        """Parsea ejemplos sulfuros, seleniuro, binario, óxido y oxysulfide."""
        examples = {
            "Ba Ti S S S": {"Ba": 1, "Ti": 1, "S": 3},
            "Sr Zr S S S": {"Sr": 1, "Zr": 1, "S": 3},
            "Ba Zr Se Se Se": {"Ba": 1, "Zr": 1, "Se": 3},
            "Na Cl": {"Na": 1, "Cl": 1},
            "Ba Ti O O O": {"Ba": 1, "Ti": 1, "O": 3},
            "Ba Ti O S": {"Ba": 1, "Ti": 1, "O": 1, "S": 1},
        }
        for site_text, expected in examples.items():
            sites = site_text.split()
            actual = parse_slices_composition(f"{site_text} 0 1 ooo")
            self.assertEqual(actual, expected)
            self.assertEqual(sum(actual.values()), len(sites))

    def test_rejects_invalid_site_or_edge(self) -> None:
        """El parser no adivina ante sitios o aristas inválidos."""
        with self.assertRaises(AuditError):
            parse_slices_composition("Ba Ti 0 3 ooo")
        with self.assertRaises(AuditError):
            parse_slices_composition("Ba Ti S 0 2 nonsense")
        with self.assertRaises(AuditError):
            parse_slices_composition("Ba Ti S 0 3 ooo")


if __name__ == "__main__":
    unittest.main()
