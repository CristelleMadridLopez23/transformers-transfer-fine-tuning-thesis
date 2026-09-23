import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_composition_smiles.py"
SPEC = importlib.util.spec_from_file_location("prepare_composition_smiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompositionSmilesTests(unittest.TestCase):
    def test_flat_formula_is_parsed_and_reduced(self):
        counts = MODULE.parse_formula("Ba4Ge4S12")
        self.assertEqual(MODULE.reduce_counts(counts), {"Ba": 1, "Ge": 1, "S": 3})

    def test_output_is_deterministic_and_disconnected(self):
        self.assertEqual(
            MODULE.composition_smiles({"S": 3, "Ge": 1, "Ba": 1}),
            "[Ba].[Ge].[S].[S].[S]",
        )

    def test_unsupported_formula_fails_loudly(self):
        with self.assertRaises(ValueError):
            MODULE.parse_formula("Ca(OH)2")

    def test_site_ratio_accepts_supercell(self):
        self.assertTrue(
            MODULE.same_ratio({"Ba": 1, "Ge": 1, "S": 3}, {"Ba": 4, "Ge": 4, "S": 12})
        )


if __name__ == "__main__":
    unittest.main()
