"""Pruebas reproducibles de inicialización y Ridge."""
from __future__ import annotations

import unittest
import numpy as np
import torch
from sklearn.linear_model import Ridge

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.two_head_predictor import TwoHeadPredictor


class DeterminismTests(unittest.TestCase):
    """Comprueba dos ejecuciones con semilla idéntica sobre datos sintéticos."""
    def test_torch_initialization_seed(self) -> None:
        """La misma semilla reproduce pesos de cabeza."""
        torch.manual_seed(42); a = TwoHeadPredictor()
        torch.manual_seed(42); b = TwoHeadPredictor()
        self.assertTrue(all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))

    def test_ridge_deterministic(self) -> None:
        """Ridge determinista produce predicciones idénticas en reentrenamiento."""
        rng = np.random.default_rng(4); x = rng.normal(size=(15, 6)); y = rng.normal(size=15)
        p1 = Ridge(alpha=1).fit(x, y).predict(x)
        p2 = Ridge(alpha=1).fit(x, y).predict(x)
        self.assertTrue(np.array_equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
