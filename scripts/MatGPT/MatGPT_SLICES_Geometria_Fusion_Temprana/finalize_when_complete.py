"""Espera el marcador de las 60 corridas y ejecuta evaluación final una vez.

Este pequeño supervisor evita iniciar inferencia antes de que todos los
checkpoints F0/F8 de fusión temprana/tardía estén completos.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").is_file())
EXPERIMENT=ROOT/"experiments/MatGPT_SLICES_Geometria_Fusion_Temprana"
SCRIPT=Path(__file__).resolve().parent
LOG=logging.getLogger("fusion_finalizer")


def main() -> None:
    """Monitorea el estado y ejecuta análisis/diagramas tras COMPLETE."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds",type=int,default=30)
    parser.add_argument("--timeout-hours",type=float,default=36.0)
    args=parser.parse_args()
    if args.poll_seconds<5 or args.timeout_hours<=0:
        parser.error("El periodo debe ser >=5 s y timeout positivo")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(EXPERIMENT/"finalization.log",encoding="utf-8")])
    deadline=time.monotonic()+args.timeout_hours*3600
    last_count=-1
    while time.monotonic()<deadline:
        summary_path=EXPERIMENT/"summary.json"
        resources_path=EXPERIMENT/"resources.json"
        if summary_path.is_file() and resources_path.is_file():
            summary=json.loads(summary_path.read_text());resources=json.loads(resources_path.read_text())
            count=int(resources.get("completed_runs",0))
            if count!=last_count:
                LOG.info("%s",{"event":"monitor","completed_runs":count,
                    "expected_runs":60,"training_status":summary.get("training_status")})
                last_count=count
            if summary.get("training_status")=="COMPLETE" and count==60:
                # Smoke runs are diagnostics, not part of the 60-run study.
                # Older smoke entries used RUNNING as a temporary status;
                # normalize them before freezing the consolidated resources.
                for name, row in resources.get("runs", {}).items():
                    if name.endswith("_smoke") and row.get("status") == "RUNNING":
                        row["status"] = "SMOKE"
                resources["active_runs"] = sum(
                    row.get("status") == "RUNNING" for row in resources.get("runs", {}).values())
                resources["completed_runs"] = sum(
                    row.get("status") == "COMPLETE" for row in resources.get("runs", {}).values())
                resources["status"] = "COMPLETE"
                resources_path.write_text(json.dumps(resources, indent=2, ensure_ascii=False) + "\n")
                for command in (
                    [sys.executable,str(SCRIPT/"analyze.py"),"--phase","all"],
                    [sys.executable,str(SCRIPT/"diagrams.py")]):
                    LOG.info("%s",{"event":"run_finalizer_command","command":command})
                    result=subprocess.run(command,cwd=ROOT,check=False)
                    if result.returncode:
                        raise RuntimeError(f"Comando final falló con código {result.returncode}: {command}")
                LOG.info("%s",{"event":"finalization_complete"})
                return
            if summary.get("training_status")=="PARTIAL":
                raise RuntimeError(f"Entrenamiento parcial: {count}/60; análisis final no iniciado")
        time.sleep(args.poll_seconds)
    raise TimeoutError("Se agotó el tiempo de espera del supervisor; consultar run.log y resources.json")


if __name__=="__main__":main()
