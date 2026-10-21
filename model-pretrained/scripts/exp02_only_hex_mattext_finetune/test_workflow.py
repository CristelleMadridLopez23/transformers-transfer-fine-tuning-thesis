"""Small technical tests for graph coverage, padding masks, grouped folds and failed gates.

Fixtures are explicitly synthetic; no experimental metrics or trained artifacts
are created. Real vocabulary/official implementation agreement is audited separately.
"""
from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import torch

from finetuning import collate, partition
from workflow import load_config, require_audit


class WorkflowTests(unittest.TestCase):
    """Check scientific failure guards and split/mask behavior on tiny synthetic fixtures."""

    def test_structural_zero_is_not_padding(self) -> None:
        """Synthetic structural ID zero must retain attention while inserted ID 150 has mask zero."""
        batch = collate([{"input_ids": [151, 0, 152], "energy_per_atom_ev": -1.},
                         {"input_ids": [151, 0, 42, 152], "energy_per_atom_ev": -2.}], 150)
        torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))
        self.assertEqual(batch["input_ids"][0, -1], 150)

    def test_no_silent_truncation(self) -> None:
        """Synthetic sequence above context length must raise rather than lose structural units."""
        with self.assertRaises(ValueError):
            collate([{"input_ids": [1] * 513, "energy_per_atom_ev": -1.}], 150)

    def test_grouped_cv_and_reserved_test(self) -> None:
        """Synthetic polymorph/frame groups must remain together in every train-only CV fold."""
        rows = [{"reduced_formula": f"c{i // 3}", "split": "train"} for i in range(60)]
        rows += [{"reduced_formula": "validation_only", "split": "val"},
                 {"reduced_formula": "test_only", "split": "test"}]
        validation_groups = []
        for fold in range(5):
            training, validation = partition(rows, fold, 42, 5)
            tg, vg = ({r["reduced_formula"] for r in group} for group in [training, validation])
            self.assertFalse(tg & vg)
            self.assertNotIn("test_only", tg | vg)
            self.assertNotIn("validation_only", tg | vg)
            validation_groups.extend(vg)
        self.assertEqual(len(validation_groups), len(set(validation_groups)))
        training, validation = partition(rows, -1, 42, 5)
        self.assertEqual({r["reduced_formula"] for r in validation}, {"validation_only"})

    def test_failed_provenance_stops_training(self) -> None:
        """Synthetic failed gate must stop before reading model weights or selecting an optimizer."""
        config = load_config("experiments/exp02_only_hex_mattext_finetune/config.json")
        with patch("workflow.read_json", return_value={"status": "failed", "blockers": ["synthetic missing evidence"]}):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                require_audit(config)

    def test_compact_syntax_omission_is_detected(self) -> None:
        """Synthetic compact image labels must be explicitly adapted before original-vocab tokenization."""
        helpers = importlib.import_module("01_audit_tokenizer").existing_graph_helpers()
        vocab = {"Ba": 1, "S": 2, "0": 3, "1": 4, "o o o": 5, "[CLS]": 6, "[SEP]": 7}
        text = "Ba S 0 1 ooo"
        with self.assertRaises(ValueError):
            helpers.tokenize_lossless(text, vocab, 512)
        tokens, ids = helpers.tokenize_lossless(helpers.adapt_slices(text), vocab, 512)
        self.assertEqual(tokens, ["[CLS]", "Ba", "S", "0", "1", "o o o", "[SEP]"])
        self.assertEqual(len(ids), len(tokens))


def main() -> None:
    """Run bounded unittest suite; return None with nonzero status for technical failures."""
    torch.set_num_threads(1)
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
