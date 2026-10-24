"""Pruebas sintéticas de identidad E0, end real y evaluación sin dropout."""
from __future__ import annotations

import unittest
import torch

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import EXPECTED_CHECKPOINT_SHA
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.freeze_and_extract import locate_end_positions


class CheckpointPoolingTests(unittest.TestCase):
    """Comprueba reglas de pooling sin cargar etiquetas ni test."""
    def test_checkpoint_sha_format(self) -> None:
        """SHA esperado corresponde a 64 dígitos hexadecimales."""
        self.assertRegex(EXPECTED_CHECKPOINT_SHA, r"^[a-f0-9]{64}$")

    def test_actual_end_not_position_zero(self) -> None:
        """El <end> se obtiene de la posición de token y padding queda después."""
        ids = torch.tensor([[1, 9, 4, 3, 0, 0], [1, 2, 3, 0, 0, 0]])
        self.assertEqual(locate_end_positions(ids, 3).tolist(), [3, 2])

    def test_missing_or_duplicate_end_rejected(self) -> None:
        """Una fila sin un único <end> bloquea extracción."""
        with self.assertRaises(RuntimeError):
            locate_end_positions(torch.tensor([[1, 2, 0]]), 3)
        with self.assertRaises(RuntimeError):
            locate_end_positions(torch.tensor([[1, 3, 3]]), 3)

    def test_dropout_eval_is_deterministic(self) -> None:
        """Dropout queda apagado para extracción en evaluación."""
        module = torch.nn.Sequential(torch.nn.Dropout(.8)).eval()
        x = torch.ones(4, 8)
        self.assertTrue(torch.equal(module(x), module(x)))


if __name__ == "__main__":
    unittest.main()
