#!/usr/bin/env python3
"""Gate-checked CLI for E2-T2 trajectory-supervised partial fine-tuning."""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Delegate to the shared trainer, which enforces T1/Aug2 prerequisites."""
    target = Path(__file__).resolve().parents[1] / "matgpt_e2_extended" / "train_aug.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
