"""Pruebas sintéticas para identidades algebraicas H3, sin mezclar resultados reales."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.common import FAMILIES
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import StructuralMLP8
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.metrics import per_composition


class HybridAlgebraTests(unittest.TestCase):
    """Comprueba centrado, reconstrucción, cancelación y métricas con vectores pequeños."""

    def test_mlp8_triplets_are_centered_and_permutation_equivariant(self) -> None:
        """La MLP centra las tres predicciones y respeta permutaciones de candidatos."""
        torch.manual_seed(11)
        model = StructuralMLP8().eval()
        x = torch.randn(2, 3, 768)
        first = model(x)
        permutation = torch.tensor([2, 0, 1])
        second = model(x[:, permutation])
        restored = second[:, torch.argsort(permutation)]
        self.assertLess(float(first.sum(dim=1).abs().max().detach()), 1e-7)
        self.assertTrue(torch.allclose(first, restored, atol=1e-7, rtol=1e-7))

    def test_common_ridge_cancels_from_polymorph_differences(self) -> None:
        """Sumar la misma media de Ridge no cambia ningún contraste de polimorfos."""
        mu = -5.8
        delta = np.array([-0.02, 0.005, 0.015])
        energy = mu + delta
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual((energy[i] - energy[j]), delta[i] - delta[j], places=14)
        self.assertAlmostEqual(float(energy.mean()), mu, places=14)

    def test_h3a_b_have_h2_ranking_by_construction(self) -> None:
        """Reemplazar solo μ deja invariantes argmin y los tres pares orientados."""
        h2 = np.array([-5.1, -5.08, -5.2])
        delta = h2 - h2.mean()
        for mu in (-6.0, -4.5):
            rebuilt = mu + delta
            self.assertEqual(int(np.argmin(rebuilt)), int(np.argmin(h2)))
            for i, j in ((0, 1), (0, 2), (1, 2)):
                self.assertAlmostEqual(rebuilt[i] - rebuilt[j], h2[i] - h2[j], places=14)

    def test_metric_uses_dft_gs_anchor_and_canonical_pairs(self) -> None:
        """La evaluación conserva negativos anclados y orientación P-H/P-N/H-N."""
        rows = []
        truth = [-5.0, -4.99, -4.90]
        pred = [-4.95, -5.01, -4.92]
        for family, y, p in zip(FAMILIES, truth, pred):
            rows.append({"model": "synthetic", "composition_id": "X", "fold": 0, "seed": 1,
                         "polymorph_family": family, "E_DFT_eV_atom": y, "E_pred_eV_atom": p})
        metric, pairs = per_composition(pd.DataFrame(rows))
        self.assertEqual([(p["family_i"], p["family_j"]) for p in pairs],
                         [(FAMILIES[0], FAMILIES[1]), (FAMILIES[0], FAMILIES[2]), (FAMILIES[1], FAMILIES[2])])
        self.assertTrue(metric["MSE_identity_abs_error"] < 1e-10)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(np.isfinite(metric["MAE_delta_GS_anchored_mev_atom"]))


if __name__ == "__main__":
    unittest.main()
