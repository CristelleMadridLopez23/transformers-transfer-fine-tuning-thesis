"""Ejecuta el ajuste composicional, análisis y figuras sin entrenar redes neuronales."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import analyze_results
import figures
import run_experiment
import common as cm

LOG = logging.getLogger("matgpt_hybrid_run")


def run(output: Path = cm.REPORT) -> dict[str, Any]:
    """Realiza el experimento completo en secuencia.

    Args:
        output: Directorio nuevo de resultados.
    Returns:
        Resumen de PASS y conteos.
    """
    allowed_measurement_files = {"resources.json", "run.log"}
    if output.exists() and any(path.name not in allowed_measurement_files for path in output.iterdir()):
        raise FileExistsError(f"La salida ya contiene archivos; no se sobrescribe: {output}")
    fitted = run_experiment.run(output)
    analyzed = analyze_results.analyze(output)
    plotted = figures.generate(output)
    readme = ("# Regresión híbrida con Ridge composicional\n\n"
        "Experimento OOF de desarrollo ABS₃: el Ridge de 17 fracciones estequiométricas predice la energía media por fórmula. "
        "Se combina en paralelo con residuos MLP8 Frozen o LoRA previamente guardados y centrados por fórmula. "
        "No se entrenaron MLP8/LoRA, no se ejecutó inferencia nueva de MatGPT y el test histórico permaneció cerrado.\n\n"
        "## Reproducción\n\n"
        "Desde la raíz del repositorio: `bash sh_files/MatGPT_Regresion_Hibrida_Ridge_Composicional/run.sh`. "
        "El wrapper registra tiempo, CPU y RSS máximo en `resources.json` y salida completa en `run.log`.\n\n"
        "## Archivos principales\n\n"
        "- `oof_predictions_by_structure.csv`: cuatro condiciones, IDs físicos, pliegues y semillas.\n"
        "- `composition_only_predictions.csv`: un valor por estructura, con residuo cero y sin adjudicar ranking entre empates.\n"
        "- `ridge_selection.csv`, `ridge_validation_grid.csv` y `models/`: selección interna y cinco modelos por pliegue.\n"
        "- `metrics_by_seed.csv`, `metrics_by_fold_seed.csv`, `composition_metrics_by_formula.csv`, `pair_predictions.csv`.\n"
        "- `paired_bootstrap_by_composition.csv`: IC95% apareado por fórmula.\n"
        "- `invariance_audit.csv`: prueba de igualdad de ΔE, ranking y residuo.\n"
        "- `figures/`: nueve figuras científicas y diagrama; PNG y SVG.\n\n"
        "## Interpretación\n\n"
        "Cambiar Ridge solo cambia el término constante por composición. ΔE y ranking deben conservarse. "
        "Una reducción del error absoluto mide una mejor estimación de la media composicional, no más información estructural de MatGPT.\n")
    (output / "README.md").write_text(readme, encoding="utf-8")
    result = {"status": "PASS", "fit": fitted, "analysis": analyzed, "figures": plotted}
    LOG.info("Experimento completado: %s", result)
    return result


def main() -> None:
    """Valida argumentos y ejecuta la secuencia reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cm.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.output.resolve())


if __name__ == "__main__":
    main()
