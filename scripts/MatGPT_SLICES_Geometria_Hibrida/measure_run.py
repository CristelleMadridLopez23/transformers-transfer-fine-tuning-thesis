"""Ejecuta una fase local y registra tiempo, RSS y código de salida en JSON."""
from __future__ import annotations

import argparse
import json
import logging
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("geometry_resources")


def execute(command: list[str], metrics: Path, log: Path) -> int:
    """Mide una corrida sin sobrescribir recursos/logs previos."""
    if not command or metrics.exists() or log.exists():
        raise ValueError("Comando vacío o métricas/log ya existentes")
    metrics.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"started_at_utc": start, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": elapsed, "user_cpu_seconds": after.ru_utime-before.ru_utime,
              "system_cpu_seconds": after.ru_stime-before.ru_stime,
              "peak_rss_bytes": after.ru_maxrss, "exit_code": proc.returncode,
              "command": command, "log": str(log)}
    metrics.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    LOG.info(json.dumps({"event": "run_complete", **result}, ensure_ascii=False))
    return proc.returncode


def main() -> None:
    """Valida CLI y ejecuta subprocess con captura de recursos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.exit(execute(command, args.metrics, args.log))


if __name__ == "__main__":
    main()
