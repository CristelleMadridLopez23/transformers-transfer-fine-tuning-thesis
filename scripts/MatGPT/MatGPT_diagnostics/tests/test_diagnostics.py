"""Pruebas pequeñas de invariantes para D1–D3."""
from __future__ import annotations

import json
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.d1_composition_baselines import formula_features
from Alternative2.scripts.MatGPT.MatGPT_diagnostics.d2_centered_training import centered_loss, group_metrics
from Alternative2.scripts.MatGPT.MatGPT_diagnostics.d3_static_representation import periodic_geometry


class CenteredLossTests(unittest.TestCase):
    """Comprueba propiedades simples de la pérdida centrada."""

    def test_common_energy_offset_is_removed(self) -> None:
        target = torch.tensor([-4.0, -3.9, -4.2, -5.0, -5.1, -4.8], requires_grad=False)
        prediction = target + 6.25
        loss = centered_loss(prediction, target, ["A"]*3+["B"]*3, scale=0.1)
        self.assertLess(float(loss), 1e-6)

    def test_prediction_tie_is_not_ground_state_success(self) -> None:
        frame = pd.DataFrame({"energy_per_atom_ev":[-5.0,-4.99,-4.8],
                              "energy_pred_ev_atom":[-5.1,-5.1,-5.1]})
        result = group_metrics(frame)
        self.assertEqual(result["ground_state_eligible"], 1.0)
        self.assertEqual(result["ground_state_accuracy"], 0.0)
        self.assertEqual(result["predicted_tie"], 1.0)


class CompositionFeatureTests(unittest.TestCase):
    """Las features composicionales dependen de fórmula, no de identidad de fila."""

    def test_equivalent_formula_features_match(self) -> None:
        keys = sorted({"Ba", "Hf", "S"})
        a = formula_features("BaHfS3", keys)
        b = formula_features("Ba1Hf1S3", keys)
        self.assertEqual(a, b)


class GeometryInvariantTests(unittest.TestCase):
    """Distancias y volumen son invariantes a orden de sitios y rotación rígida."""

    def row(self, cell: np.ndarray, positions: np.ndarray, numbers: list[int]) -> pd.Series:
        return pd.Series({"cell_json":json.dumps(cell.tolist()),
                          "arrays_json":json.dumps({"positions":positions.tolist(),"numbers":numbers}),
                          "structure_id":"toy","composition_group":"toy","reduced_formula":"AB",
                          "polymorph_family":"test"})

    def test_order_and_rotation_invariance(self) -> None:
        cell = np.eye(3)*5.0
        positions = np.array([[0.1,0.2,0.3],[2.1,2.2,2.3],[4.5,0.1,0.2]])
        base = periodic_geometry(self.row(cell,positions,[56,72,16]))
        angle=0.37; rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rotated=periodic_geometry(self.row(cell@rotation,positions@rotation,[56,72,16]))
        reordered=periodic_geometry(self.row(cell,positions[[2,0,1]],[16,56,72]))
        for key in ("volume_per_atom_ang3","density_g_cm3","pair_distance_mean_angstrom",
                    "pair_distance_std_angstrom","pair_distance_median_angstrom","pair_count_below_3.0A_per_atom"):
            self.assertAlmostEqual(base[key],rotated[key],places=6)
            self.assertAlmostEqual(base[key],reordered[key],places=6)


if __name__ == "__main__":
    unittest.main()
