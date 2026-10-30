"""Pruebas estructurales para la cabeza híbrida de Experimento D."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT / "scripts/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad"))
from train import GeometryMLP8, init_matched_head  # noqa: E402
import torch  # noqa: E402


class GeometryHeadTests(unittest.TestCase):
    """Comprueba dimensiones, inicialización y propagación de gradiente geométrico."""

    def test_parameter_count_and_zero_initialized_geometry(self) -> None:
        """La rama geométrica debe iniciar idéntica a MLP8 sobre las mismas entradas."""
        class SlicesHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = torch.nn.Sequential(torch.nn.Linear(768, 8), torch.nn.GELU(),
                                                   torch.nn.Dropout(0.1), torch.nn.Linear(8, 1))
        torch.manual_seed(42)
        base = SlicesHead()
        torch.manual_seed(42)
        geometry, init_hash, _ = init_matched_head(42, SlicesHead())
        self.assertEqual(sum(p.numel() for p in geometry.parameters()), 6217)
        self.assertEqual(len(init_hash), 64)
        geometry.eval(); base.eval()
        x = torch.randn(3, 768); g = torch.randn(3, 7)
        with torch.no_grad():
            self.assertTrue(torch.allclose(base.network[3](base.network[2](base.network[1](base.network[0](x)))),
                                           geometry.network[3](geometry.network[2](geometry.network[1](geometry.network[0](torch.cat([x,g],-1))))),
                                           atol=1e-7, rtol=1e-6))

    def test_geometric_columns_receive_gradients(self) -> None:
        """Los 56 pesos asociados a geometría deben participar en el gradiente."""
        head = GeometryMLP8()
        with torch.no_grad(): head.network[0].weight[:, 768:].zero_()
        x = torch.randn(2, 3, 775); target = torch.randn(2, 3)
        loss = torch.nn.functional.smooth_l1_loss(head(x), target)
        loss.backward()
        grad = head.network[0].weight.grad[:, 768:]
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.any(grad != 0))


if __name__ == "__main__":
    unittest.main()
