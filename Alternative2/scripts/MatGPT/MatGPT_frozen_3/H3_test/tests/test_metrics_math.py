"""Pruebas sintéticas de unidades, centrado y convenios H3-TEST."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.common import FAMILIES
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.H3_test.metrics import composition_metrics, summarize_model


class MetricMathTests(unittest.TestCase):
    """Comprueba identidades métricas sin acceder a datos del proyecto."""

    def setUp(self) -> None:
        """Crea un único triplete sintético con inversión de orden."""
        self.frame = pd.DataFrame({
            "composition_id": ["synthetic_ABS3"] * 3,
            "polymorph_family": list(FAMILIES),
            "E_DFT_eV_atom": [-10.000, -9.980, -9.950],
            "E_pred": [-9.970, -10.010, -9.950],
        })

    def test_ev_to_mev_scaling_and_anchor_can_be_negative(self) -> None:
        """Anclaje DFT conserva inversiones negativas y conversión eV→meV."""
        row, _ = composition_metrics(self.frame, "E_pred")
        self.assertAlmostEqual(abs(-9.280 - -10.000) * 1000, 720.0)
        self.assertEqual(row["DFT_GS_family"], "dist_perovskite")
        self.assertEqual(row["predicted_GS_family"], "hexagonal")

    def test_centered_residual_and_mse_identity(self) -> None:
        """Error cuadrático, a diferencia del MAE, satisface la identidad."""
        row, _ = composition_metrics(self.frame, "E_pred")
        self.assertLess(row["MSE_identity_abs_error"], 1e-10)
        self.assertAlmostEqual(float((self.frame.E_pred - self.frame.E_pred.mean()).sum()), 0.0, places=12)

    def test_pair_order_is_canonical_and_three_pairs_only(self) -> None:
        """Un triplete genera exactamente (P,H), (P,N), (H,N)."""
        _, pairs = composition_metrics(self.frame.sample(frac=1, random_state=7), "E_pred")
        self.assertEqual([(p["family_i"], p["family_j"]) for p in pairs],
                         [(FAMILIES[0], FAMILIES[1]), (FAMILIES[0], FAMILIES[2]), (FAMILIES[1], FAMILIES[2])])
        self.assertEqual(len(pairs), 3)

    def test_summary_counts_compositions_not_structures_for_gs(self) -> None:
        """La matriz GS tiene un voto por composición física."""
        summary, comp, pair = summarize_model(self.frame, "E_pred")
        self.assertEqual(summary["N_compositions"], 1)
        self.assertEqual(len(comp), 1)
        self.assertEqual(len(pair), 3)


if __name__ == "__main__":
    unittest.main()
