"""Pruebas de unidades, ancla DFT y empates para la evaluación del Experimento D."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
MODULE_PATH = ROOT / "scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad/analyze.py"
SPEC = importlib.util.spec_from_file_location("experiment_d_analysis", MODULE_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ANALYSIS)


class EvaluationMathTests(unittest.TestCase):
    """Verifica fórmulas puras sin generar resultados de experimento."""

    @staticmethod
    def triplet(predictions: list[float]) -> pd.DataFrame:
        """Crea un triplete mínimo para comprobar la aritmética de evaluación."""
        dft = [-5.0, -4.99, -4.80]
        return pd.DataFrame({
            "condition": ["F2_GEOMETRÍA"] * 3,
            "fold": [0] * 3,
            "seed": [42] * 3,
            "composition_id": ["BaZrS3"] * 3,
            "polymorph_family": ["a", "b", "c"],
            "structure_id": ["a", "b", "c"],
            "n_atoms": [5] * 3,
            "energy_total_dft_ev": [e * 5 for e in dft],
            "energy_per_atom_dft_ev": dft,
            "energy_total_pred_ev": [e * 5 for e in predictions],
            "energy_per_atom_pred_ev": predictions,
            "residual_pred_ev_atom": [e - sum(predictions) / 3 for e in predictions],
        })

    def test_dft_ground_state_anchor_uses_same_structure(self) -> None:
        """ΔE anclada usa el mismo índice GS DFT en ambos ejes."""
        comp, pairs = ANALYSIS.calculate_metrics(self.triplet([-4.99, -5.0, -4.8]))
        row = comp.iloc[0]
        self.assertAlmostEqual(row.mae_delta_anchored_nonanchor_mev_atom, 15.0, places=8)
        self.assertAlmostEqual(row.mae_delta_zero_baseline_mev_atom,
                               pairs.abs_delta_dft_mev_atom.mean(), places=8)
        self.assertEqual(int(row.pairwise_eligible_count), 3)

    def test_predicted_ground_state_tie_is_not_a_hit(self) -> None:
        """Un empate en el mínimo predicho queda sin resolver, aunque incluya al GS DFT."""
        comp, _ = ANALYSIS.calculate_metrics(self.triplet([-5.0, -5.0, -4.8]))
        self.assertEqual(int(comp.iloc[0].ground_state_exact), 0)
        self.assertEqual(int(comp.iloc[0].ground_state_unresolved), 1)


if __name__ == "__main__":
    unittest.main()
