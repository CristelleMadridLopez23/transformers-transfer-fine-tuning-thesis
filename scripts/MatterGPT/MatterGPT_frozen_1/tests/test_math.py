"""Synthetic unit tests for MatterGPT head algebra and shared energy metrics."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))
from heads import TwoHead
from energy_metrics import composition_metrics


class FrozenHeadMathTests(unittest.TestCase):
    """Check algebraic identities without mixing synthetic data into reports."""

    def test_centered_residuals_and_reconstruction(self) -> None:
        torch.manual_seed(7)
        model = TwoHead(16, dropout=0.0).eval()
        x_mean = torch.randn(4, 16)
        x_struct = torch.randn(4, 3, 16)
        mu, delta, pred = model(x_mean, x_struct)
        self.assertTrue(torch.allclose(delta.sum(dim=1), torch.zeros(4), atol=1e-7))
        self.assertTrue(torch.allclose(pred.mean(dim=1), mu, atol=1e-7))

    def test_permutation_equivariance(self) -> None:
        torch.manual_seed(12)
        model = TwoHead(16, dropout=0.0).eval()
        x_mean = torch.randn(1,16); x = torch.randn(1,3,16); perm = torch.tensor([2,0,1])
        pred = model(x_mean,x)[2]
        pred_perm = model(x_mean,x[:,perm])[2]
        inv = torch.argsort(perm)
        self.assertTrue(torch.allclose(pred,pred_perm[:,inv],atol=1e-7))

    def test_composition_shift_cancels_from_differences(self) -> None:
        residual = np.array([0.2,-0.1,-0.1]); mu = 5.4
        e = mu + residual
        self.assertTrue(np.allclose(e[:,None]-e[None,:],residual[:,None]-residual[None,:]))

    def test_metric_reports_physical_triplet_counts(self) -> None:
        data = pd.DataFrame({"composition_id":["A"]*3,"polymorph_family":["dist_perovskite","hexagonal","needle"],
                             "y":[-2.0,-1.99,-1.8],"p":[-2.0,-1.9,-1.7]})
        metric=composition_metrics(data,"y","p")
        self.assertEqual(metric["n_compositions"],1)
        self.assertEqual(metric["pairwise_eligible"],3)
        self.assertEqual(metric["gs_eligible"],1)


if __name__ == "__main__":
    unittest.main()
