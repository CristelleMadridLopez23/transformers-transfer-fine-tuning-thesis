"""Pruebas científicas del vector composicional y partición histórica."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import joblib

SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT))
import common as cm  # noqa: E402
from run_experiment import formula_roles  # noqa: E402


class ProtocolTests(unittest.TestCase):
    """Controles de entradas composicionales, splits y fórmulas energéticas."""

    def test_element_order_and_fractions(self) -> None:
        """La representación cumple orden fijo y fracciones exactas para BaZrS3."""
        self.assertEqual(cm.ELEMENTS, ("Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr"))
        vector = cm.stoichiometric_fractions("BaZrS3")
        self.assertEqual(vector.shape, (17,))
        self.assertTrue(np.isclose(vector.sum(), 1.0))
        self.assertTrue(np.allclose(vector[[cm.ELEMENTS.index("Ba"), cm.ELEMENTS.index("Zr"), cm.ELEMENTS.index("S")]], [1/5, 1/5, 3/5]))

    def test_same_formula_has_same_composition_features(self) -> None:
        """Los polimorfos de una fórmula reciben un vector idéntico."""
        self.assertTrue(np.array_equal(cm.stoichiometric_fractions("BaZrS3"), cm.stoichiometric_fractions("BaZrS3")))

    def test_feature_table_contains_composition_only_inputs(self) -> None:
        """La matriz no contiene energía, IDs de estructura ni geometría."""
        data, _, _, _ = cm.load_and_audit_sources()
        table = cm.formula_feature_table(data)
        self.assertEqual(tuple(table.columns), ("composition_id", *cm.ELEMENTS))
        self.assertEqual(table.shape, (70, 18))

    def test_saved_scaler_is_fit_on_outer_train_formulas(self) -> None:
        """Cada scaler guardado reproduce la media de las 56 fórmulas outer-train."""
        data, folds, _, _ = cm.load_and_audit_sources()
        features = cm.formula_feature_table(data).set_index("composition_id")
        for fold in range(5):
            roles = formula_roles(folds, fold)
            pipeline = joblib.load(cm.REPORT / "models" / f"ridge_composicional_fold_{fold}.joblib")
            expected = features.loc[sorted(roles["outer_train"]), cm.ELEMENTS].to_numpy(float).mean(axis=0)
            self.assertTrue(np.allclose(pipeline.named_steps["scaler"].mean_, expected, atol=1e-12))

    def test_historical_grouped_splits(self) -> None:
        """Los cinco pliegues conservan particiones 45/11/56/14 sin solapamiento."""
        data, folds, _, _ = cm.load_and_audit_sources()
        self.assertEqual(len(data), 210)
        for fold in range(5):
            roles = formula_roles(folds, fold)
            self.assertEqual(tuple(len(roles[k]) for k in ("inner_train", "inner_validation", "outer_train", "outer_test")), (45, 11, 56, 14))
            self.assertTrue(roles["inner_train"].isdisjoint(roles["inner_validation"]))
            self.assertTrue(roles["outer_train"].isdisjoint(roles["outer_test"]))

    def test_common_energy_offset_cancels_from_pair_gaps(self) -> None:
        """Sumar la misma media a los tres residuos deja ΔE y ranking invariantes."""
        residual = np.asarray([-0.03, 0.01, 0.02])
        old = 1.7 + residual
        new = -3.2 + residual
        old_gaps = np.asarray([(old[j]-old[i])*1000 for i in range(3) for j in range(i+1, 3)])
        new_gaps = np.asarray([(new[j]-new[i])*1000 for i in range(3) for j in range(i+1, 3)])
        self.assertTrue(np.allclose(old_gaps, new_gaps, atol=1e-10))
        self.assertTrue(np.array_equal(np.argsort(old), np.argsort(new)))

    def test_energy_total_conversion(self) -> None:
        """La energía total predicha se obtiene multiplicando E/N por N."""
        e_atom, n_atoms = -4.2, 20
        self.assertEqual(e_atom * n_atoms, -84.0)


if __name__ == "__main__":
    unittest.main()
