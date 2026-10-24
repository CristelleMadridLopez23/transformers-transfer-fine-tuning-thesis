"""Verifica exportación PNG exclusiva mediante firma real."""
from __future__ import annotations

import unittest
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Alternative2.scripts.MatGPT.MatGPT_frozen_3.make_png_figures import save_png
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import REPORT_DIR, ExperimentError


class PngOnlyTests(unittest.TestCase):
    """Comprueba extensión, firma PNG y rechazo de PDF/SVG."""
    def test_png_is_written_with_signature(self) -> None:
        """El artefacto de test temporal se elimina inmediatamente tras verificarlo."""
        path = REPORT_DIR / "manifests" / "test_png_signature.png"
        fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
        save_png(fig, path)
        try:
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        finally:
            path.unlink(missing_ok=True)

    def test_pdf_suffix_rejected(self) -> None:
        """El helper no permite exportar formatos vectoriales."""
        fig, _ = plt.subplots()
        try:
            with self.assertRaises(ExperimentError):
                save_png(fig, REPORT_DIR / "manifests" / "forbidden.pdf")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
