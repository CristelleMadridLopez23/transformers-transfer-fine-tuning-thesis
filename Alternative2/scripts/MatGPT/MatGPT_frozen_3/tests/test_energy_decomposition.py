"""Pruebas matemáticas sintéticas de separación μ+δ y MSE."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.metrics import triplet_metrics


class EnergyDecompositionTests(unittest.TestCase):
    """Verifica escalas, componentes y reconstrucción energética."""
    def frame(self) -> pd.DataFrame:
        """Crea un triplete sintético independiente de resultados científicos."""
        return pd.DataFrame({"composition_id": ["X"]*3, "polymorph_family": ["dist_perovskite", "hexagonal", "needle"],
            "fold": [0]*3, "E_DFT_eV_atom": [-1.0, -.99, -.97], "E_pred_eV_atom": [-.998, -.991, -.969]})

    def test_ev_to_mev(self) -> None:
        """0.72 eV equivale a 720 meV."""
        self.assertEqual(.72 * 1000, 720)

    def test_mse_identity(self) -> None:
        """El error cuadrático se separa en media y residuo centrado."""
        row, _ = triplet_metrics(self.frame())
        self.assertLess(row["MSE_identity_abs_error"], 1e-10)

    def test_reconstruction_sum_and_zero_mean(self) -> None:
        """Las desviaciones centradas reconstruyen energía y suman cero."""
        energies = np.array([-1.01, -0.99, -0.97])
        mu = float(energies.mean())
        delta = energies - mu
        self.assertAlmostEqual(float(delta.sum()), 0)
        self.assertTrue(np.allclose(mu + delta, energies))

    def test_mean_and_residual_mae_are_not_additive_claim(self) -> None:
        """La métrica registra componentes sin definir MAE como suma."""
        row, _ = triplet_metrics(self.frame())
        self.assertIn("MAE_mu_mev_atom", row)
        self.assertIn("MAE_delta_centered_mev_atom", row)
        self.assertNotAlmostEqual(row["MAE_absolute_mev_atom"], row["MAE_mu_mev_atom"] + row["MAE_delta_centered_mev_atom"])


if __name__ == "__main__":
    unittest.main()
