#!/usr/bin/env python3
"""Registra el progreso de una corrida Alexandria ya iniciada, sin intervenir en ella.

Lee el CSV de historial y los manifiestos de estado/recursos, y escribe eventos
JSON Lines de solo anexado para poder seguir cada fase y época con marcas de tiempo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/matgpt_alexandria_energy_transfer_20k"
TRACE = EXPERIMENT / "training_trace.jsonl"
LOGGER = logging.getLogger("alexandria_training_trace")
TERMINAL = {"COMPLETED", "FAILED", "PARTIAL_INTERRUPTED", "STOPPED"}


def file_sha256(path: Path) -> str | None:
    """Devuelve el hash SHA-256 del archivo, o None si no existe."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Lee un objeto JSON, tolerando un reemplazo atómico concurrente."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_history(path: Path) -> list[dict[str, str]]:
    """Lee el historial CSV; un archivo a medio escribir se reintentará luego."""
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"stage", "epoch", "train_smoothl1", "validation_smoothl1",
                        "epoch_wall_seconds", "optimizer_steps"}
            if not required.issubset(set(reader.fieldnames or [])):
                return []
            rows = []
            for row in reader:
                # pandas rewrites the CSV in place after each epoch; ignore a
                # transient partial row until the next stable read.
                if not row or not required.issubset(row) or not row.get("stage"):
                    continue
                try:
                    int(row["epoch"])
                    float(row["train_smoothl1"])
                    float(row["validation_smoothl1"])
                    float(row["epoch_wall_seconds"])
                    int(row["optimizer_steps"])
                except (TypeError, ValueError):
                    continue
                rows.append(row)
            return rows
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def append_event(stream: Any, event: dict[str, Any]) -> None:
    """Anexa y fuerza un evento JSONL para preservar el rastro en disco."""
    event["observed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def existing_epochs(path: Path) -> set[tuple[str, str]]:
    """Evita duplicar épocas si el monitor se reinicia."""
    seen: set[tuple[str, str]] = set()
    if not path.is_file():
        return seen
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "epoch_observed":
                    try:
                        stage = str(event.get("stage"))
                        epoch = str(int(event.get("epoch")))
                    except (TypeError, ValueError):
                        continue
                    if stage in {"warmup_frozen", "finetune_f4"}:
                        seen.add((stage, epoch))
    except OSError:
        pass
    return seen


def trace(interval_seconds: float) -> None:
    """Observa archivos de estado hasta que la corrida alcance estado terminal."""
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Escribiendo traza append-only en %s", TRACE)
    history_path = EXPERIMENT / "historial_entrenamiento.csv"
    resources_path = EXPERIMENT / "resources.json"
    summary_path = EXPERIMENT / "summary.json"
    seen = existing_epochs(TRACE)
    sources = {
        "training_script_on_disk_sha256_at_monitor_attach": file_sha256(Path(__file__).with_name("train_energy.py")),
        "running_process_loaded_script_sha256": "NOT_CAPTURED; monitor attached after process start",
        "config_sha256": file_sha256(EXPERIMENT / "config.yaml"),
        "train_manifest_sha256": file_sha256(EXPERIMENT / "alexandria_20k_train.csv"),
        "validation_manifest_sha256": file_sha256(EXPERIMENT / "alexandria_2k_validation.csv"),
    }
    with TRACE.open("a", encoding="utf-8", buffering=1) as output:
        if TRACE.stat().st_size:
            prior_events = []
            try:
                with TRACE.open("r", encoding="utf-8") as existing:
                    for line in existing:
                        try:
                            prior_events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                prior_events = []
            bad_epoch_events = [event for event in prior_events
                                if event.get("event") == "epoch_observed"
                                and (event.get("stage") not in {"warmup_frozen", "finetune_f4"}
                                     or not str(event.get("epoch", "")).isdigit())]
            if bad_epoch_events and not any(event.get("event") == "trace_correction" for event in prior_events):
                append_event(output, {"event": "trace_correction", "invalidated_event_count": len(bad_epoch_events),
                                      "reason": "A transient partial CSV rewrite was observed; malformed row is not a training epoch."})
        append_event(output, {"event": "monitor_attached", "experiment": str(EXPERIMENT.relative_to(ROOT)),
                              "poll_interval_seconds": interval_seconds, "source_hashes": sources})
        last_status: str | None = None
        last_history_signature: tuple[int, int] | None = None
        stable_history_reads = 0
        last_heartbeat = time.monotonic()
        while True:
            resources = read_json(resources_path)
            summary = read_json(summary_path)
            status = str(resources.get("status") or summary.get("status") or "UNKNOWN")
            history_rows = read_history(history_path)
            if history_path.exists():
                stat = history_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
            else:
                signature = None

            if signature is not None and signature == last_history_signature:
                stable_history_reads += 1
            else:
                stable_history_reads = 0

            if status != last_status:
                append_event(output, {
                    "event": "status_observed", "status": status,
                    "completed_finetune_epochs": resources.get("completed_finetune_epochs"),
                    "best_epoch": resources.get("best_epoch"),
                    "peak_rss_bytes": resources.get("peak_rss_bytes"),
                    "resource_status": resources.get("status"), "summary_status": summary.get("status"),
                })
                last_status = status

            # Wait until a changed CSV survives a full polling interval before
            # emitting its rows; this avoids logging pandas' truncate/rewrite window.
            stable_rows = history_rows if stable_history_reads >= 1 else []
            for row in stable_rows:
                stage, epoch = str(row.get("stage", "")), str(row.get("epoch", ""))
                key = (stage, epoch)
                if not stage or not epoch or key in seen:
                    continue
                seen.add(key)
                append_event(output, {
                    "event": "epoch_observed", "stage": stage, "epoch": epoch,
                    "train_smoothl1": row.get("train_smoothl1"),
                    "validation_smoothl1": row.get("validation_smoothl1"),
                    "epoch_wall_seconds": row.get("epoch_wall_seconds"),
                    "optimizer_steps": row.get("optimizer_steps"),
                    "gradient_norm_mean": row.get("gradient_norm_mean"),
                    "best_epoch_so_far": resources.get("best_epoch"),
                    "peak_rss_bytes": resources.get("peak_rss_bytes"),
                    "history_rows": len(history_rows),
                })

            if signature != last_history_signature and signature is not None:
                append_event(output, {"event": "history_file_updated", "size_bytes": signature[1],
                                      "history_rows": len(history_rows),
                                      "last_stage": history_rows[-1].get("stage") if history_rows else None,
                                      "last_epoch": history_rows[-1].get("epoch") if history_rows else None})
                last_history_signature = signature

            if time.monotonic() - last_heartbeat >= 60.0:
                latest = history_rows[-1] if history_rows else {}
                append_event(output, {
                    "event": "monitor_heartbeat", "observed_status_file_value": status,
                    "latest_history_stage": latest.get("stage"),
                    "latest_history_epoch": latest.get("epoch"),
                    "history_file_age_seconds": (time.time() - history_path.stat().st_mtime)
                    if history_path.exists() else None,
                    "resources_file_age_seconds": (time.time() - resources_path.stat().st_mtime)
                    if resources_path.exists() else None,
                    "note": "Heartbeat confirms the monitor is active; RUNNING is read from status files, not an OS process probe.",
                })
                last_heartbeat = time.monotonic()

            if status in TERMINAL:
                append_event(output, {"event": "monitor_finished", "status": status,
                                      "completed_finetune_epochs": resources.get("completed_finetune_epochs"),
                                      "best_epoch": resources.get("best_epoch"),
                                      "total_wall_seconds": resources.get("total_wall_seconds"),
                                      "peak_rss_bytes": resources.get("peak_rss_bytes")})
                return
            time.sleep(interval_seconds)


def main() -> None:
    """Valida los argumentos y comienza el seguimiento de solo lectura."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Segundos entre lecturas de los archivos de progreso (default: 10).")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval debe ser positivo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        trace(args.interval)
    except KeyboardInterrupt:
        LOGGER.info("Monitor detenido manualmente; el entrenamiento no fue interrumpido.")


if __name__ == "__main__":
    main()
