"""Ejecuta localmente las fases de la ablación MatGPT y registra su estado."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
SCRIPT_DIR = ROOT / "scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad"


def main() -> None:
    """Valida opciones y ejecuta entrenamiento opcional y productos finales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Ejecuta el entrenamiento completo antes de los análisis")
    parser.add_argument("--evaluate", action="store_true", help="Vuelve a reconstruir predicciones y métricas desde checkpoints")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads debe ser positivo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    commands: list[list[str]] = []
    if args.train:
        commands.append([sys.executable, str(SCRIPT_DIR / "train.py"), "--phase", "all",
                         "--device", args.device, "--num-threads", str(args.num_threads)])
    if args.evaluate:
        commands.append([sys.executable, str(SCRIPT_DIR / "evaluate.py")])
    commands.extend([sys.executable, str(SCRIPT_DIR / filename)] for filename in
                    ("plots.py", "diagrams.py", "write_report.py"))
    for command in commands:
        logging.info("local command: %s", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
