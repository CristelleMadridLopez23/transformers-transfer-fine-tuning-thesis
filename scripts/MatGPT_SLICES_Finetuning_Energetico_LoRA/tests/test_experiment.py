"""Pruebas de integridad de LoRA Q/V y de la rama Ridge histórica."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))
import experiment as ex  # noqa: E402
from lora_matgpt import insert_qv_lora  # noqa: E402


class MatGPTLoRAIntegrity(unittest.TestCase):
    """Comprueba el protocolo real sin consultar etiquetas de outer ni test."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga una vez checkpoint y tokenizador originales en CPU."""
        torch.set_num_threads(2)
        cls.frame = ex.development()
        cls.folds = ex.assignments(cls.frame)
        cls.model, cls.stoi, cls.context = ex.model_and_tokens()
        cls.tokens = ex.encode(cls.frame, cls.stoi, cls.context)
        first = cls.folds[(cls.folds.outer_fold == 0) & (cls.folds.inner_role == "inner_train")]
        cls.triplet = ex.formula_groups(cls.frame, set(first.structure_id))[0][1]

    def test_a_data_and_target(self) -> None:
        """Valida 210 IDs físicos, fórmula agrupada, E_total/N y residuos."""
        self.assertEqual((len(self.frame), self.frame.formula.nunique()), (210, 70))
        self.assertEqual(len(self.triplet), 3)
        target = ex.residual_target(self.frame, [("train", self.triplet)])
        self.assertLess(abs(float(target.sum())), 2e-6)
        self.assertTrue(np.isfinite(target).all())

    def test_b_initial_equivalence_padding_and_l03(self) -> None:
        """Q/V inicial no cambia FINAL_END, padding ni bloque 3/Ridge."""
        model, _stoi, _context = ex.model_and_tokens()
        model.eval()
        capture: list[torch.Tensor] = []
        hook = model.transformer_decoder.layers[3].register_forward_hook(
            lambda _module, _args, output: capture.append(output.detach().clone()))
        with torch.inference_mode():
            before = ex.final_end(model, self.triplet, self.tokens, self.stoi["<pad>"])
        l03_before = capture[-1]
        adapters = insert_qv_lora(model)
        with torch.inference_mode():
            after = ex.final_end(model, self.triplet, self.tokens, self.stoi["<pad>"])
            l03_after = capture[-1]
            single = ex.final_end(model, [self.triplet[0]], self.tokens, self.stoi["<pad>"])
        hook.remove()
        self.assertLess(float((before - after).abs().max()), 1e-5)
        self.assertLess(float((after[0] - single[0]).abs().max()), 1e-5)
        self.assertLess(float((l03_before - l03_after).abs().max()), 1e-6)
        self.assertEqual(sum(p.numel() for a in adapters for p in a.parameters() if p.requires_grad), 24576)
        self.assertTrue(all(not p.requires_grad for p in model.transformer_decoder.layers[3].parameters()))

    def test_c_gradients_and_checkpoint(self) -> None:
        """La cabeza y LoRA reciben gradientes y los pesos base quedan intactos."""
        model, _stoi, _context = ex.model_and_tokens()
        adapters = insert_qv_lora(model)
        head = ex.MLP8()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 6161)
        original = model.transformer_decoder.layers[10].self_attn.base.in_proj_weight.detach().clone()
        head.train()
        for adapter in adapters:
            adapter.train(); adapter.base.eval()
        target = torch.from_numpy(ex.residual_target(self.frame, [("", self.triplet)])[0])
        prediction = head(ex.final_end(model, self.triplet, self.tokens, self.stoi["<pad>"]))
        loss = ex.loss_residual(prediction, target, float(target.std()))
        loss.backward()
        trainable = [p for adapter in adapters for p in adapter.parameters() if p.requires_grad]
        self.assertTrue(any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in trainable))
        self.assertTrue(any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in head.parameters()))
        self.assertTrue(torch.equal(original, model.transformer_decoder.layers[10].self_attn.base.in_proj_weight))
        self.assertTrue(all(p.grad is None for p in model.transformer_decoder.layers[3].parameters()))
        self.assertTrue(torch.isfinite(loss))

    def test_d_energy_and_pair_algebra(self) -> None:
        """Comprueba E/N, antisimetría, transitividad y mínimo por triplete."""
        atoms = np.array([20, 20, 20], dtype=float)
        total = np.array([-100., -99.9, -99.7])
        per_atom = total / atoms
        self.assertTrue(np.allclose(total, atoms * per_atom))
        delta = lambda a, b: 1000 * (per_atom[b] - per_atom[a])
        self.assertAlmostEqual(delta(0, 1), -delta(1, 0), places=12)
        self.assertAlmostEqual(delta(0, 1) + delta(1, 2), delta(0, 2), places=12)
        self.assertEqual(int(np.argmin(per_atom)), 0)

    def test_e_matched_initial_head(self) -> None:
        """La inicialización de la cabeza es idéntica entre condiciones apareadas."""
        torch.manual_seed(123)
        frozen = ex.MLP8()
        torch.manual_seed(123)
        adapted = ex.MLP8()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(frozen.parameters(), adapted.parameters())))

    def test_f_checkpoint_restoration_and_selection(self) -> None:
        """Comprueba que el checkpoint disponible restaura cabeza/adaptadores determinísticamente."""
        run = ex.REPORT / "runs/lora_fold0_seed42"
        checkpoint_path = run / "best.pt"
        if not checkpoint_path.is_file():
            self.skipTest("El primer checkpoint LoRA aún no está disponible")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        head = ex.MLP8(); head.load_state_dict(checkpoint["head"], strict=True); head.eval()
        model, _stoi, _context = ex.model_and_tokens()
        adapters = insert_qv_lora(model)
        for adapter, state in zip(adapters, checkpoint["adapters"]):
            adapter.load_state_dict(state, strict=False)
        model.eval()
        with torch.inference_mode():
            features_a = ex.final_end(model, self.triplet, self.tokens, self.stoi["<pad>"])
            pred_a = head(features_a)
            features_b = ex.final_end(model, self.triplet, self.tokens, self.stoi["<pad>"])
            pred_b = head(features_b)
        self.assertTrue(torch.equal(pred_a, pred_b))
        history = pd.read_csv(run / "history.csv")
        selected = json.loads((run / "summary.json").read_text())["best_epoch"]
        self.assertEqual(int(history.loc[history.inner_validation_loss.idxmin(), "epoch"]), selected)
        self.assertEqual(checkpoint["scale_ev_atom"], json.loads((run / "summary.json").read_text())["scale_ev_atom"])
        self.assertEqual(ex.sha256(ex.CHECKPOINT), ex.EXPECTED["checkpoint"])

    def test_g_no_formula_leakage(self) -> None:
        """Cada fórmula aparece en una sola partición de cada fold histórico."""
        formula = self.frame.set_index("structure_id").formula
        for fold in range(5):
            part = self.folds[self.folds.outer_fold == fold]
            sets = [set(formula.loc[part[part.inner_role == role].structure_id])
                    for role in ("inner_train", "inner_validation")]
            sets.append(set(formula.loc[part[part.outer_role == "outer_validation"].structure_id]))
            self.assertEqual(list(map(len, sets)), [45, 11, 14])
            self.assertTrue(all(not (sets[i] & sets[j]) for i, j in itertools.combinations(range(3), 2)))


if __name__ == "__main__":
    unittest.main()
