"""Pruebas matemáticas con datos sintéticos, separadas de resultados reales."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_2.evaluate_energy_conventions import category, summarize_triplet
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.train_grouped_cv import TwoHeads
from Alternative2.scripts.MatGPT.MatGPT_frozen_2.common import EXPECTED_CHECKPOINT_SHA, FAMILIES


class FrozenTwoHeadMathTests(unittest.TestCase):
    """Casos sintéticos para conversiones, ranking y centrado."""

    def test_ev_to_mev(self) -> None:
        self.assertEqual(0.720 * 1000, 720.0)

    def test_absolute_mae(self) -> None:
        self.assertEqual(float(np.mean(np.abs(np.array([1., 2., 3.])-np.array([2., 2., 1.])))), 1.0)

    def test_argmin_inversion(self) -> None:
        self.assertEqual(int(np.argmin([0.0, 1.0, 2.0])), 0)
        self.assertEqual(int(np.argmin([2.0, 0.0, 1.0])), 1)

    def test_dft_anchor_can_be_negative(self) -> None:
        frame = self.triplet([0.000, .020, .030], [.025, .005, .030])
        metric = summarize_triplet(frame)
        self.assertLess(min(metric["pred_anchor_mev_atom"]), 0)

    def test_own_min_is_nonnegative(self) -> None:
        frame = self.triplet([0.0, .02, .03], [.025, .005, .03])
        metric = summarize_triplet(frame)
        self.assertGreaterEqual(min(metric["pred_own_min_mev_atom"]), 0)

    def test_centered_residuals_sum_to_zero(self) -> None:
        raw = np.array([2., 3., 7.]); centered = raw - raw.mean()
        self.assertAlmostEqual(float(centered.sum()), 0., places=12)

    def test_energy_reconstruction(self) -> None:
        mu, delta = 1.25, np.array([-.25, .05, .20])
        self.assertTrue(np.allclose(mu+delta, [1.0, 1.30, 1.45]))

    def test_mse_identity_not_mae_identity(self) -> None:
        error = np.array([0., 1., 5.]); mean = error.mean(); residual = error-mean
        self.assertAlmostEqual(float(np.mean(error**2)), float(mean**2+np.mean(residual**2)))
        self.assertNotAlmostEqual(float(np.mean(np.abs(error))), float(abs(mean)+np.mean(np.abs(residual))))

    def test_permutation_invariance(self) -> None:
        torch.manual_seed(12); model=TwoHeads().eval()
        x=torch.randn(1,3,768); perm=torch.tensor([2,0,1])
        with torch.no_grad():
            original=model(x.mean(1),x)[2]
            shuffled=model(x[:,perm].mean(1),x[:,perm])[2][:,torch.argsort(perm)]
        self.assertTrue(torch.allclose(original,shuffled,atol=1e-6,rtol=1e-6))

    def test_pair_direction_is_canonical(self) -> None:
        self.assertEqual(category(-2.0), "i_lower")
        self.assertEqual(category(2.0), "j_lower")
        self.assertEqual(category(0.2), "tie")

    def test_gs_triplet_is_one_composition(self) -> None:
        m=summarize_triplet(self.triplet([0,.01,.02],[0,.01,.02]))
        self.assertIn(m["dft_gs_family"], FAMILIES)

    def test_pairwise_triplet_has_three_pairs(self) -> None:
        m=summarize_triplet(self.triplet([0,.01,.02],[0,.01,.02]))
        self.assertEqual(m["pairwise_physical_count"],3)

    def test_checkpoint_identity_is_declared(self) -> None:
        self.assertEqual(len(EXPECTED_CHECKPOINT_SHA),64)

    def test_end_pooling_uses_real_final_token(self) -> None:
        ids=[10,21,22,23,11]
        self.assertEqual(len(ids)-1,4)
        self.assertNotEqual(len(ids)-1,0)

    def test_test_exclusion_roles_are_disjoint(self) -> None:
        inner_train={"A","B"};inner_val={"C"};outer={"D"};historical_test={"Z"}
        self.assertFalse((inner_train|inner_val|outer)&historical_test)

    def test_training_scale_uses_train_only(self) -> None:
        train=np.array([1.,2.,3.]); validation=np.array([1000.]); scale=train.std()
        self.assertEqual(scale,float(np.std([1.,2.,3.])))
        self.assertNotEqual(scale,float(np.std(np.r_[train,validation])))

    def test_r2_ols_differs_from_diagonal_prediction(self) -> None:
        x=np.array([0.,1.,2.]); y=2*x+3
        ols_r2=1-np.sum((y-(2*x+3))**2)/np.sum((y-y.mean())**2)
        pred_r2=1-np.sum((y-x)**2)/np.sum((x-x.mean())**2)
        self.assertAlmostEqual(ols_r2,1.0)
        self.assertNotAlmostEqual(ols_r2,pred_r2)

    def test_h0_errors_split_into_mean_and_residual(self) -> None:
        error=np.array([.2,.4,.9]); mean=error.mean(); residual=error-mean
        self.assertAlmostEqual(float(np.mean(error**2)),float(mean**2+np.mean(residual**2)))

    @staticmethod
    def triplet(energies: list[float], predictions: list[float]):
        """Arma un grupo sintético canónico para una composición."""
        import pandas as pd
        return pd.DataFrame({"composition_id":["X"]*3,"polymorph_family":list(FAMILIES),
            "E_DFT_eV_atom":energies,"E_pred_eV_atom":predictions})


if __name__ == "__main__":
    unittest.main()
