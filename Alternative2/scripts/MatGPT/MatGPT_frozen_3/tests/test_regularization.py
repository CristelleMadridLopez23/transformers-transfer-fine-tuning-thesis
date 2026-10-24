"""Pruebas de capacidad y cambios aislados de regularización."""
from __future__ import annotations

import unittest

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.train_heads import SPECS
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.two_head_predictor import TwoHeadPredictor, trainable_parameter_count


class RegularizationTests(unittest.TestCase):
    """Comprueba conteo paramétrico y configuración de ablaciones."""
    def test_width_four_has_fewer_parameters(self) -> None:
        """El cuello 4 reduce a la mitad aproximadamente los parámetros de ancho 8."""
        self.assertLess(trainable_parameter_count(TwoHeadPredictor(4, .1)), trainable_parameter_count(TwoHeadPredictor(8, .1)))

    def test_dropout_ablation_changes_only_dropout(self) -> None:
        """R3-DROP conserva ancho, WD y paciencia del baseline."""
        base, drop = SPECS["R2-REF8"], SPECS["R3-DROP30"]
        self.assertEqual(base["width"], drop["width"])
        self.assertEqual(base["weight_decay"], drop["weight_decay"])
        self.assertEqual(base["patience"], drop["patience"])
        self.assertNotEqual(base["dropout"], drop["dropout"])

    def test_weight_decay_ablation_changes_only_weight_decay(self) -> None:
        """R3-WD conserva ancho, dropout y parada."""
        base, wd = SPECS["R2-REF8"], SPECS["R3-WD1E2"]
        self.assertEqual(base["width"], wd["width"])
        self.assertEqual(base["dropout"], wd["dropout"])
        self.assertEqual(base["patience"], wd["patience"])
        self.assertNotEqual(base["weight_decay"], wd["weight_decay"])


if __name__ == "__main__":
    unittest.main()
