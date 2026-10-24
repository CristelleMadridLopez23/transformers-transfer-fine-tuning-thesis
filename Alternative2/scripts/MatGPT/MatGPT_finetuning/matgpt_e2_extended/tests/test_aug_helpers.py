"""Unit checks for deterministic, distinct SLICES-view selection."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Alternative2.scripts.MatGPT.MatGPT_finetuning.matgpt_e2_extended.train_aug import secondary_index, selection_index


class VariantSelectionTests(unittest.TestCase):
    """Check deterministic primary selection and a distinct secondary view."""

    def test_primary_view_is_deterministic_and_in_range(self) -> None:
        """The same run coordinates must always select a valid view index."""
        first = selection_index("structure-1", 2, 123, 7, 4)
        second = selection_index("structure-1", 2, 123, 7, 4)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 4)

    def test_secondary_view_is_distinct(self) -> None:
        """Aug2's secondary view must never repeat its primary view."""
        for primary in range(4):
            secondary = secondary_index("structure-2", 0, 42, 1, 4, primary)
            self.assertIn(secondary, range(4))
            self.assertNotEqual(secondary, primary)

    def test_single_view_cannot_support_aug2(self) -> None:
        """Missing a second verified serialization must fail closed."""
        with self.assertRaises(ValueError):
            secondary_index("structure-3", 0, 42, 1, 1, 0)


if __name__ == "__main__":
    unittest.main()
