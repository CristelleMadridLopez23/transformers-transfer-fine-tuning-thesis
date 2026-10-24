"""Pruebas de no solapamiento de composición y desarrollo."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import ExperimentError, load_dev
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.data_and_folds import assert_no_overlap


class FoldLeakageTests(unittest.TestCase):
    """Comprueba exclusión de test y disjunción por fórmula."""
    def test_development_loader_excludes_historical_test(self) -> None:
        """La vista cargada solo conserva train y validation."""
        data = load_dev()
        self.assertEqual(len(data), 210)
        self.assertEqual(data.composition_id.nunique(), 70)
        self.assertTrue(set(data.split).issubset({"train", "val"}))

    def test_overlap_is_rejected(self) -> None:
        """Una fórmula compartida entre particiones es error crítico."""
        data = pd.DataFrame({"composition_id": ["A", "A", "B", "B", "C", "C"]})
        with self.assertRaises(ExperimentError):
            assert_no_overlap(data, np.array([0, 1]), np.array([1, 2]))

    def test_disjoint_groups_pass(self) -> None:
        """Conjuntos separados por composición pasan."""
        data = pd.DataFrame({"composition_id": ["A", "A", "B", "B", "C", "C"]})
        assert_no_overlap(data, np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))


if __name__ == "__main__":
    unittest.main()
