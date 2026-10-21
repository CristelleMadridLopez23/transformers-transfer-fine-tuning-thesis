"""Minimal unit tests with explicitly synthetic fixtures plus real checkpoint integrity.

Synthetic energies/features only test algorithms; no experimental metrics or
scientific embeddings are written. The real checkpoint is loaded strictly on CPU.
"""
from __future__ import annotations

import importlib
import unittest

import numpy as np
import pandas as pd
import torch

from utils import (adapt_slices, energy_per_atom, load_config, make_splits,
                   parse_slices, project_root, read_json, tokenize_lossless)

extractor = importlib.import_module("03_extract_embeddings")
training = importlib.import_module("04_train_regressors")
evaluation = importlib.import_module("05_evaluate_models")


class ExperimentTests(unittest.TestCase):
    """Verify scientific transformations and failure guards using small labeled fixtures."""

    def test_energy_units(self) -> None:
        """Synthetic known total must yield the exact per-atom unit conversion."""
        self.assertEqual(energy_per_atom(-100, 20), -5)
        for total, count in [(float("nan"), 20), (-1, 0), (-1, 1.5)]:
            with self.assertRaises(ValueError):
                energy_per_atom(total, count)

    def test_adapter_preserves_structural_units(self) -> None:
        """Synthetic strategy-4 graph must retain species, edge order and image vectors."""
        original = "Ba Ti S 0 1 +oo 1 2 o-o 2 0 oo+"
        result = adapt_slices(original)
        self.assertEqual(parse_slices(original, 4), parse_slices(result, 3))
        for malformed in ["Ba Ti 0 2 ooo", "Ba Ti 0 1 oo", "Ba Ti 0 1 xyz", "Ba Ti 0"]:
            with self.assertRaises(ValueError):
                adapt_slices(malformed)

    def test_tokenization_coverage_and_truncation(self) -> None:
        """Candidate scanner must reject omissions/truncation and preserve multi-digit indices."""
        vocab = {"Ba": 0, "Ti": 1, "o o o": 2, "1": 3, "0": 4, "[CLS]": 5, "[SEP]": 6}
        tokens, ids = tokenize_lossless("Ba Ti 10 1 o o o", vocab, 20)
        self.assertEqual(tokens, ["[CLS]", "Ba", "Ti", "1", "0", "1", "o o o", "[SEP]"])
        self.assertEqual(len(ids), 8)
        with self.assertRaises(ValueError):
            tokenize_lossless("Ba Xx", vocab, 20)
        with self.assertRaises(ValueError):
            tokenize_lossless("Ba Ti 10 1 o o o", vocab, 7)

    def test_grouped_splits(self) -> None:
        """Synthetic polymorphs of one composition must share split and validation fold."""
        frame = pd.DataFrame({"structure_id": [f"s{i}" for i in range(60)],
                              "reduced_formula": [f"c{i // 3}" for i in range(60)]})
        first = make_splits(frame, 42, 0.2, 4)
        pd.testing.assert_frame_equal(first, make_splits(frame, 42, 0.2, 4))
        self.assertTrue(first.groupby("reduced_formula").split.nunique().eq(1).all())
        self.assertTrue(first.groupby("reduced_formula").fold.nunique().eq(1).all())
        self.assertEqual(first.loc[first.split.eq("test"), "reduced_formula"].nunique(), 4)

    def test_masked_pooling(self) -> None:
        """Artificial large special/pad states must never contaminate structural mean."""
        hidden = torch.tensor([[[100., 100.], [2., 4.], [4., 8.], [200., 200.], [999., 999.]]])
        mean, cls = extractor.pool_hidden(hidden, torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([[0, 1, 1, 0, 0]]))
        torch.testing.assert_close(mean, torch.tensor([[3., 6.]]))
        torch.testing.assert_close(cls, torch.tensor([[100., 100.]]))

    def test_relative_energy_and_minimum(self) -> None:
        """Synthetic within-composition deltas, near ties and regret must use eV/atom."""
        frame = pd.DataFrame({"structure_id": ["a", "b", "c"], "reduced_formula": ["AB", "AB", "AB"],
                              "polymorph_family": ["dist_perovskite", "hexagonal", "needle"],
                              "energy_per_atom_ev": [-5., -4.9, -4.8], "prediction_ev_atom": [-4.7, -5., -4.8]})
        values, pairs = evaluation.composition_metrics(frame, 0.001)
        self.assertAlmostEqual(values["selection_regret_mev_atom"], 100)
        self.assertEqual(values["minimum_selection_accuracy"], 0)
        self.assertAlmostEqual(pairs[0]["true_delta_ev_atom"], 0.1)
        frame.loc[1, "energy_per_atom_ev"] = -4.9995
        self.assertEqual(evaluation.composition_metrics(frame, 0.001)[0]["minimum_selection_accuracy"], 1)
        self.assertEqual(evaluation.frequent_family(frame, 0.001), "dist_perovskite")

    def test_head_training_synthetic_only(self) -> None:
        """Tiny synthetic development-only fits check all heads and feature-scaler isolation."""
        config = load_config("experiments/exp01_mattext_frozen/config.yaml")
        config["models"]["mlp"].update(max_epochs=3, patience=2)
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20, 4))
        y = x[:, 0] - 2 * x[:, 1]
        groups = np.repeat(np.arange(10), 2)
        configurations = {"ridge": {"alpha": 1.}, "random_forest": {"n_estimators": 3, "max_depth": 2},
                          "mlp": {"learning_rate": 0.001, "weight_decay": 0.001}}
        torch.set_num_threads(1)
        for name, params in configurations.items():
            model = training.fit_predictor(name, x, y, groups, params, config)
            predicted = training.predict(model, x[:2] + 100)
            self.assertEqual(predicted.shape, (2,))
            self.assertTrue(np.isfinite(predicted).all())
            if name == "ridge":
                np.testing.assert_allclose(model.regressor_.steps[0][1].mean_, x.mean(axis=0))
            if name == "mlp":
                self.assertEqual(model["dimension"], 4)
                self.assertGreater(model["epochs"], 0)

    def test_real_encoder_frozen_shape_determinism(self) -> None:
        """Real MatText weights with explicitly synthetic IDs verify strict load and determinism."""
        torch.set_num_threads(1)
        model = extractor.load_frozen(project_root() / "model-pretrained/mattext-slices-2m")
        self.assertFalse(model.training)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 32962048)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 0)
        first = extractor.encode_batch(model, [[1, 2, 3], [1, 2, 4, 3]], 0, "cpu")
        second = extractor.encode_batch(model, [[1, 2, 3], [1, 2, 4, 3]], 0, "cpu")
        for a, b in zip(first, second):
            self.assertEqual(a.shape, (2, 512))
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, b)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


def main() -> None:
    """Run unittest suite; fixtures are synthetic and never represent experiment results."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
