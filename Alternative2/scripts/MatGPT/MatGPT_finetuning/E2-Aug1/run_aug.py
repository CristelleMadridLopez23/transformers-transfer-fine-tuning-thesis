#!/usr/bin/env python3
"""CLI entry point for E2-Aug1 (graph-preserving SLICES augmentation)."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Delegate to the shared, audited augmentation trainer."""
    target = Path(__file__).resolve().parents[1] / "matgpt_e2_extended" / "train_aug.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
