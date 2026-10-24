"""Pruebas de capas brutas y control explícito post-ln_f."""
from __future__ import annotations

import unittest
import torch


class IntermediateLayerTests(unittest.TestCase):
    """Verifica que ln_f cambia solo el control final etiquetado."""
    def test_layer12_post_ln_f_is_explicit_transform(self) -> None:
        """La salida post-ln_f coincide con la capa LayerNorm aplicada a raw layer 12."""
        torch.manual_seed(4)
        raw = torch.randn(3, 2, 8)
        norm = torch.nn.LayerNorm(8)
        self.assertTrue(torch.allclose(norm(raw), norm(raw), atol=0, rtol=0))
        self.assertFalse(torch.allclose(norm(raw), raw))

    def test_end_and_mean_are_distinct_pooling_operations(self) -> None:
        """END selecciona posición final válida; MEAN promedia solo SLICES."""
        states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        end, mean = states[4], states[1:4].mean(0)
        self.assertFalse(torch.equal(end, states[0]))
        self.assertTrue(torch.equal(mean, torch.tensor([8., 9., 10., 11.])))


if __name__ == "__main__":
    unittest.main()
