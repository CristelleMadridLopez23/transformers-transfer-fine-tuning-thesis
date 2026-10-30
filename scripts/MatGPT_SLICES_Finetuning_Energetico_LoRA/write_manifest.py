"""Escribe manifest de ejecución, códigos, fuentes y salidas verificadas."""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import experiment as ex

LOG = logging.getLogger("matgpt_lora_manifest")


def build(output: Path) -> dict[str, object]:
    """Registra hashes de fuentes y estado exacto de cada ajuste."""
    summaries = []
    run_pattern = re.compile(r"(frozen|lora)_fold[0-4]_seed(42|123|2026)")
    for path in sorted((output / "runs").glob("*_fold*_seed*/summary.json")):
        if not run_pattern.fullmatch(path.parent.name):
            continue
        summary = json.loads(path.read_text())
        summary["summary_path"] = str(path.relative_to(ex.ROOT))
        summaries.append(summary)
    resources = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*_resources.json"))
                 if re.fullmatch(r"(frozen|lora)_fold[0-4]_seed(42|123|2026)_resources\.json", p.name)]
    complete = len(summaries) == 30 and all(row.get("status") == "COMPLETE" for row in summaries)
    preflight_path = output / "preflight/preflight.json"
    payload: dict[str, object] = {
        "experiment_id": "MatGPT_SLICES_Finetuning_Energetico_LoRA",
        "status": "COMPLETE" if complete else "RUNNING_OR_PARTIAL",
        "preflight_status": json.loads(preflight_path.read_text()).get("status") if preflight_path.is_file() else "MISSING",
        "protocol_config": str((ex.ROOT / "experiments/MatGPT_SLICES_Finetuning_Energetico_LoRA/config.yaml").relative_to(ex.ROOT)),
        "source_hashes_sha256": {str(path.relative_to(ex.ROOT)): ex.sha256(path) for path in (
            ex.CHECKPOINT, ex.MODEL_INFO, ex.DATASET, ex.FOLDS, ex.CACHE,
            ex.SOURCE / "utils/models.py", ex.SOURCE / "config/uncond.yaml",
            Path(__file__), Path(__file__).with_name("experiment.py"), Path(__file__).with_name("lora_matgpt.py"),
            Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("figures.py"),
            Path(__file__).with_name("diagrams.py"), Path(__file__).with_name("write_reports.py"),
            Path(__file__).parent / "tests/test_experiment.py",
            ex.ROOT / "sh_files/MatGPT_SLICES_Finetuning_Energetico_LoRA/run_all.sh",
            ex.ROOT / "sh_files/MatGPT_SLICES_Finetuning_Energetico_LoRA/finalize_when_complete.sh",
            ex.ROOT / "experiments/MatGPT_SLICES_Finetuning_Energetico_LoRA/config.yaml")},
        "run_count": len(summaries), "expected_runs": 30,
        "training_runs": summaries,
        "resources": {"recorded_runs": len(resources),
                      "wall_seconds_total": sum(float(row.get("wall_seconds", 0)) for row in resources),
                      "peak_rss_bytes_max": max((int(row.get("peak_rss_bytes", 0)) for row in resources), default=0)},
        "historical_test_used": False,
        "ridge_updated": False,
        "geometry_input_used": False,
        "pretrained_checkpoint_modified": False,
        "generated_outputs": []}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp"):
            payload["generated_outputs"].append({"path": str(path.relative_to(ex.ROOT)), "sha256": ex.sha256(path)})
    ex.atomic_json(output / "manifest.json", payload)
    if complete and len(resources) != 30:
        raise ValueError("30 corridas completas requieren 30 JSON de recursos")
    return payload


def main() -> None:
    """Valida el destino y escribe manifest JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build(args.output.resolve())
    LOG.info("%s", {"event": "manifest_written", "status": manifest["status"],
                   "run_count": manifest["run_count"]})
    print(json.dumps({"status": manifest["status"], "run_count": manifest["run_count"]}))


if __name__ == "__main__":
    main()
