"""Pruebas sintéticas de empates, anclajes y denominadores."""
from __future__ import annotations

import unittest
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.metrics import classify_pair, triplet_metrics


class MetricDenominatorTests(unittest.TestCase):
    """Comprueba reglas near-degenerate y orientación canónica de pares."""
    def frame(self, labels: list[float], pred: list[float]) -> pd.DataFrame:
        """Construye triplete de prueba en eV/átomo."""
        return pd.DataFrame({"composition_id": ["X"]*3,
            "polymorph_family": ["dist_perovskite", "hexagonal", "needle"], "fold": [0]*3,
            "E_DFT_eV_atom": labels, "E_pred_eV_atom": pred})

    def test_dft_gap_one_mev_excludes_gs(self) -> None:
        """Un gap DFT de 1 meV no es GS elegible."""
        result, _ = triplet_metrics(self.frame([-1, -.999, -.95], [-1, -.99, -.9]))
        self.assertFalse(result["GS_eligible"])

    def test_predicted_tie_is_abstention(self) -> None:
        """Mínimos predichos separados menos de 1 meV no reciben acierto."""
        result, _ = triplet_metrics(self.frame([-1, -.95, -.9], [-1, -.9995, -.9]))
        self.assertTrue(result["GS_predicted_abstention"])
        self.assertFalse(result["GS_correct"])

    def test_pair_orientation_and_tie(self) -> None:
        """Signo negativo significa i_lower; tolerancia incluye empate."""
        self.assertEqual(classify_pair(-2), "i_lower")
        self.assertEqual(classify_pair(0.5), "tie")
        self.assertEqual(classify_pair(2), "j_lower")

    def test_exactly_three_unordered_pairs(self) -> None:
        """Un triplete genera tres pares canónicos, no seis dirigidos."""
        _, pairs = triplet_metrics(self.frame([-1, -.9, -.8], [-1, -.9, -.8]))
        self.assertEqual(len(pairs), 3)
        self.assertEqual([(p["family_i"], p["family_j"]) for p in pairs],
            [("dist_perovskite", "hexagonal"), ("dist_perovskite", "needle"), ("hexagonal", "needle")])


if __name__ == "__main__":
    unittest.main()
