"""Pruebas unitarias ligeras de invariantes de la ablación MatGPT."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / "train.py"
SPEC = importlib.util.spec_from_file_location("depth_train", MODULE)
depth = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(depth)


class DepthProtocolTests(unittest.TestCase):
    """Comprueba dimensiones y máscaras de congelación del protocolo."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga una sola instancia del modelo oficial para las pruebas."""
        cls.frame, cls.folds, cls.model, _, _ = depth.load_sources()

    def test_head_shape_and_parameter_count(self) -> None:
        """La cabeza mantiene 768→8→1 y 6,161 parámetros."""
        head = depth.MLP8()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 6161)
        self.assertEqual(head.network[0].in_features, 768)
        self.assertEqual(head.network[0].out_features, 8)
        self.assertEqual(head.network[-1].out_features, 1)

    def test_conditions_select_expected_blocks_only(self) -> None:
        """Embeddings, ln_f, lm_head y capas no elegidas quedan congelados."""
        for condition, expected in depth.CONDITIONS.items():
            parameters = depth.configure_trainable(self.model, expected)
            active = {i for i, block in enumerate(self.model.transformer_decoder.layers)
                      if any(p.requires_grad for p in block.parameters())}
            self.assertEqual(active, set(expected))
            self.assertEqual(sum(p.numel() for p in parameters),
                             sum(p.numel() for i in expected for p in self.model.transformer_decoder.layers[i].parameters()))
            self.assertFalse(any(p.requires_grad for p in self.model.token_embedding_table.parameters()))
            self.assertFalse(any(p.requires_grad for p in self.model.position_embedding_table.parameters()))
            self.assertFalse(any(p.requires_grad for p in self.model.ln_f.parameters()))
            self.assertFalse(any(p.requires_grad for p in self.model.lm_head.parameters()))

    def test_residual_centering(self) -> None:
        """La salida MLP8 suma cero en cada triplete."""
        prediction = depth.MLP8()(torch.randn(4, 3, 768))
        self.assertTrue(torch.allclose(prediction.sum(dim=1), torch.zeros(4), atol=1e-6))

    def test_physical_ids_targets_and_triplets(self) -> None:
        """Los 210 IDs tienen energías eV/átomo coherentes y tres polimorfos."""
        self.assertEqual(len(self.frame), 210)
        self.assertEqual(self.frame.structure_id.nunique(), 210)
        self.assertEqual(self.frame.formula.nunique(), 70)
        self.assertTrue((self.frame.n_atoms > 0).all())
        self.assertTrue(torch.isfinite(torch.tensor(self.frame.dft_ev_atom.to_numpy())).all())
        self.assertTrue((self.frame.groupby("formula").size() == 3).all())

    def test_composition_grouped_splits(self) -> None:
        """Ninguna fórmula cruza entrenamiento, validación y outer de cada pliegue."""
        lookup = self.frame.set_index("structure_id").formula
        for fold in range(5):
            part = self.folds[self.folds.outer_fold == fold]
            sets = [set(lookup.loc[part.loc[part.inner_role == role, "structure_id"]])
                    for role in ("inner_train", "inner_validation")]
            sets.append(set(lookup.loc[part.loc[part.outer_role == "outer_validation", "structure_id"]]))
            self.assertEqual(list(map(len, sets)), [45, 11, 14])
            self.assertTrue(all(not (sets[i] & sets[j]) for i in range(3) for j in range(i+1,3)))

    def test_energy_difference_antisymmetry_and_transitivity(self) -> None:
        """La aritmética intrafórmula conserva antisimetría y transitividad."""
        e = torch.tensor([-4.2, -4.1, -3.9])
        ab, bc, ac = e[1]-e[0], e[2]-e[1], e[2]-e[0]
        self.assertAlmostEqual(float(ab + (e[0]-e[1])), 0.0, places=7)
        self.assertAlmostEqual(float(ab + bc), float(ac), places=7)

    def test_ridge_feature_schema_contains_only_stoichiometry(self) -> None:
        """La matriz composicional consta del identificador y 17 fracciones."""
        import pandas as pd
        table = pd.read_csv(depth.FEATURES)
        self.assertEqual(len(table), 70)
        self.assertEqual(table.columns[0], "composition_id")
        self.assertEqual(len(table.columns)-1, 17)
        self.assertTrue(((table.iloc[:, 1:].to_numpy() >= 0) & (table.iloc[:, 1:].to_numpy() <= 1)).all())


if __name__ == "__main__":
    unittest.main()
