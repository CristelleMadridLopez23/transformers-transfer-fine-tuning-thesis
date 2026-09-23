import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_unit_cell_smiles.py"
SPEC = importlib.util.spec_from_file_location("prepare_unit_cell_smiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UnitCellSmilesTests(unittest.TestCase):
    def test_full_cell_stoichiometry_is_preserved(self):
        self.assertEqual(
            MODULE.disconnected_smiles({"Ba": 4, "Ge": 4, "S": 12}).count("["),
            20,
        )

    def test_disconnected_output_is_deterministic(self):
        self.assertEqual(
            MODULE.disconnected_smiles({"S": 3, "Ba": 1, "Ge": 1}),
            "[Ba].[Ge].[S].[S].[S]",
        )

    def test_structure_key_ignores_relax_static_suffix(self):
        relax = "Ba1Ge1S3--Ba1Ge1S3_hexagonal--nm--gga-relax"
        static = "Ba1Ge1S3--Ba1Ge1S3_hexagonal--nm--gga-static"
        *_, relax_key = MODULE.split_record_id(relax, "Ba1Ge1S3")
        *_, static_key = MODULE.split_record_id(static, "Ba1Ge1S3")
        self.assertEqual(relax_key, static_key)


if __name__ == "__main__":
    unittest.main()
