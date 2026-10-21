"""Regression checks for forensic decisions and persisted scientific evidence.

Tests validate gate semantics, independent source references and preservation of
structural zero under padding. No model, GPU or SLICES conversion is executed.
Integration assertions read real probe output rather than manufacturing results.
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from audit import decide, read, tensor_header
from gather import DEFAULT, ROOT


class CertificationTests(unittest.TestCase):
    """Check formal decisions and exact persisted historical encoding evidence."""

    def test_missing_link_is_partial(self) -> None:
        """Successful software alone cannot certify a run with missing provenance."""
        self.assertEqual(decide(True, True, True, False, True, []), "PARTIALLY_VERIFIED")

    def test_contradiction_blocks_even_with_link(self) -> None:
        """Unresolved material protocol contradictions override favorable evidence."""
        self.assertEqual(decide(True, True, True, True, True, ["PAD unresolved"]), "BLOCKED")

    def test_full_evidence_required(self) -> None:
        """All evidence conditions allow certification; failed identity blocks it."""
        self.assertEqual(decide(True, True, True, True, True, []), "CERTIFIED")
        self.assertEqual(decide(False, True, True, True, True, []), "BLOCKED")
        self.assertEqual(decide(True, False, True, True, True, []), "BLOCKED")

    def test_invalid_tensor_header_rejected(self) -> None:
        """Malformed safetensors cannot be accepted as checkpoint identity proof."""
        with tempfile.TemporaryDirectory(prefix="mattext_certification_") as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(struct.pack("<Q", 1_000_000))
            with self.assertRaises(ValueError):
                tensor_header(path)

    def test_official_historical_references(self) -> None:
        """Pre-upload source must match independent published IDs and original test tokens."""
        golden = read(ROOT / DEFAULT / "tokenizer_golden_tests.json")
        variant = golden["implementations"]["xtal_preupload"]["variants"]["documentation_default"]
        self.assertTrue(variant["tests"][0]["official_ids_match"])
        references = [row for row in variant["tests"] if "official_tokens_match" in row]
        self.assertEqual(len(references), 4)
        self.assertTrue(all(row["official_tokens_match"] for row in references))
        self.assertIsNone(variant["tests"][0]["official_mask_match"])

    def test_structural_zero_remains_attended(self) -> None:
        """Actual historical masks attend vector0 while inserted PAD150 is masked."""
        golden = read(ROOT / DEFAULT / "tokenizer_golden_tests.json")
        variant = golden["implementations"]["xtal_preupload"]["variants"]["archived_pretraining_specials"]
        row = next(item for item in variant["tests"] if item["name"] == "structural_zero_and_padding")
        self.assertEqual(len(row["input_ids"]), 512)
        self.assertTrue(all(mask == 1 for value, mask in zip(row["input_ids"], row["attention_mask"]) if value == 0))
        self.assertTrue(all(mask == 0 for value, mask in zip(row["input_ids"], row["attention_mask"]) if value == 150))
        self.assertEqual(variant["mlm_probe"]["status"], "passed")

    def test_real_samples_preserve_prior_ids(self) -> None:
        """New historical probes must agree exactly with persisted real project encodings."""
        golden = read(ROOT / DEFAULT / "tokenizer_golden_tests.json")
        tests = golden["implementations"]["xtal_preupload"]["variants"]["archived_pretraining_specials"]["tests"]
        real = [row for row in tests if "matches_prior_candidate" in row]
        self.assertGreaterEqual(len(real), 5)
        self.assertTrue(all(row["matches_prior_candidate"] for row in real))
        self.assertTrue(all(row["repeat_exact"] for row in tests))


def main() -> None:
    """Run deterministic forensic unit/integration tests with standard unittest output."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
