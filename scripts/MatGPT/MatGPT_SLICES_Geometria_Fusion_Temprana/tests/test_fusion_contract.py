"""Pruebas de contrato para proyección geométrica y arquitectura híbrida."""
from __future__ import annotations

import unittest

from scripts.MatGPT.MatGPT_SLICES_Geometria_Fusion_Temprana.train import GeometryProjection, MLP8
import torch


class FusionContractTests(unittest.TestCase):
    """Comprueba dimensiones, inicialización y gradientes de los módulos nuevos."""

    def test_projection_is_zero_and_trainable(self) -> None:
        """La proyección tiene 5,376 pesos cero y recibe derivadas finitas."""
        layer = GeometryProjection()
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 5376)
        self.assertEqual(torch.count_nonzero(layer.linear.weight).item(), 0)
        x = torch.randn(4, 7)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (4, 768))
        (out * torch.randn_like(out)).sum().backward()
        self.assertIsNotNone(layer.linear.weight.grad)
        self.assertTrue(torch.isfinite(layer.linear.weight.grad).all())
        self.assertGreater(float(layer.linear.weight.grad.norm()), 0.0)

    def test_mlp8_dimensions_and_centering(self) -> None:
        """La cabeza permanece 768→8→1 y centra cada triplete."""
        model = MLP8()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6161)
        result = model(torch.randn(5, 3, 768))
        self.assertEqual(tuple(result.shape), (5, 3))
        self.assertTrue(torch.allclose(result.sum(dim=1), torch.zeros(5), atol=1e-6))

    def test_projection_rejects_wrong_feature_width(self) -> None:
        """Entradas que no contienen los siete descriptores se rechazan."""
        with self.assertRaises(ValueError):
            GeometryProjection()(torch.zeros(2, 8))


if __name__ == "__main__":
    unittest.main()
