"""Pruebas unitarias pequeñas para canonización de fórmulas del split por grupo."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1_3.common import E13Error, canonical_formula


class CanonicalFormulaTests(unittest.TestCase):
    """Comprueba que el orden escrito y múltiplos estequiométricos no dividen grupos."""

    def test_order_and_stoichiometric_multiple_share_group(self) -> None:
        """Las fórmulas equivalentes deben producir la misma clave química."""
        self.assertEqual(canonical_formula("BaGeS3"), canonical_formula("S6Ge2Ba2"))

    def test_unsupported_formula_is_rejected(self) -> None:
        """Paréntesis/cargas no admitidos detienen el preflight sin adivinar."""
        with self.assertRaises(E13Error):
            canonical_formula("Ba(GeS3)2")


if __name__ == "__main__":
    unittest.main()
