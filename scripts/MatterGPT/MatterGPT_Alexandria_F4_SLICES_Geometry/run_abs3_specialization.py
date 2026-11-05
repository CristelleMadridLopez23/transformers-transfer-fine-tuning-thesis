#!/usr/bin/env python3
"""Transfiere MatterGPT F4 con geometría Alexandria a ABS₃ congelado y LP-FT.

Adapta el orquestador auditado LP-FT para consumir el nuevo checkpoint F4 con
geometría, generar D0 congelado con una MLP8 nueva y ejecutar D1–D4. Todos los
checkpoints se seleccionan por inner-validation; nunca lee test histórico.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
ENERGY = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry"
OUT = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry_abs3_lpft"
TRANSFER = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry_abs3_transfer"
LPFT_SCRIPT = ROOT / "scripts/MatterGPT/MatterGPT_Alexandria_ABS3_LPFT_Gradual_L2SP/run_lpft_abs3.py"
LOG = logging.getLogger("mattergpt_alexandria_geometry_abs3")


def import_module(path: Path, name: str) -> Any:
    """Carga módulo local por ruta sin ejecutar su punto de entrada."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checksum(path: Path) -> str:
    """Calcula SHA-256 de un archivo local por bloques."""
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_new_loader(module: Any, checkpoint: Path, digest: str) -> None:
    """Configura LP-FT para reconstruir el nuevo F4 desde el checkpoint base."""
    original_sha = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
    if checksum(checkpoint) != digest:
        raise ValueError("SHA del checkpoint Alexandria F4 con geometría cambió")
    if checksum(ROOT / "models/MatterGPT/pytorch_model.pt") != original_sha:
        raise ValueError("Checkpoint generativo MatterGPT base no coincide")
    summary = json.loads((ENERGY / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETED" or summary.get("checkpoint_sha256") != digest:
        raise ValueError("La corrida F4 con geometría no está completada o no coincide su hash")

    def load_f4_model(train_module: Any, device: torch.device):
        """Carga el F4 parcial fusionado sobre MatterGPT original con strict=True."""
        model, stoi, block_size = train_module.load_model()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("base_checkpoint_sha256") != original_sha or state.get("condition_vector") != [1.0, 2.0]:
            raise ValueError("Procedencia/condicionamiento del F4-geometry incorrectos")
        current = model.state_dict()
        expected = {k for k in current if any(k.startswith(f"blocks.{i}.") for i in range(8, 12))}
        partial = state.get("blocks_8_11", {})
        if set(partial) != expected:
            raise ValueError(f"Bloques F4 incompletos: missing={len(expected-set(partial))}, extra={len(set(partial)-expected)}")
        if any(current[k].shape != v.shape for k, v in partial.items()):
            raise ValueError("Dimensiones de bloques F4 con geometría no coinciden")
        merged = dict(current); merged.update(partial)
        report = model.load_state_dict(merged, strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise ValueError(f"Reconstrucción no estricta del backbone: {report}")
        model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        meta = {"checkpoint_sha256": digest, "base_sha256": original_sha,
                "selected_epoch": int(state["selected_epoch"]), "condition_vector": [1.0, 2.0],
                "architecture": {"layers": len(model.blocks), "embedding_dim": 768,
                                 "context_length": block_size, "tokenizer_size": len(stoi)},
                "updated_tensors": len(partial), "full_state_load_strict": True,
                "alexandria_geometry": True}
        return model, base_state, meta

    module.F4_PATH = checkpoint
    module.F4_SHA = digest
    module.load_f4_model = load_f4_model
    module.OUT = OUT
    module.IMAGES = ROOT / "images" / OUT.name
    module.EXP = OUT.name
    module.TRANSFER = TRANSFER
    module.BASELINE_CSV = TRANSFER / "predicciones_oof_por_estructura.csv"

    def compute_new_embeddings(self: Any) -> np.ndarray:
        """Extrae FINAL_END congelado desde el backbone F4 con geometría Alexandria."""
        values = self.train_module.extract_embeddings(self.model, self.sequences, 8, self.device)
        if values.shape != (210, 768) or not np.isfinite(values).all():
            raise ValueError("Embedding F4-geometry ABS₃ no es 210×768 finito")
        return values

    module.LPFTExperiment._load_validated_embedding_cache = compute_new_embeddings

    def train_frozen_d0(self: Any):
        """Entrena D0 congelado sobre cada fold, con el protocolo MLP8 histórico."""
        if checksum(checkpoint) != digest: raise ValueError("Checkpoint F4 alterado antes de D0")
        self.m2.OUT = OUT
        all_predictions = []; all_histories = []; records = []
        for fold in range(5):
            centered = self.data.energy_dft_ev_atom.to_numpy(float) - self.data.groupby("composition_id").energy_dft_ev_atom.transform("mean").to_numpy(float)
            roles = self.folds[self.folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
            formulas = set(roles.loc[roles.inner_role.eq("inner_train"), "composition_group"])
            indices = np.flatnonzero(self.data.composition_id.isin(formulas).to_numpy())
            scale = float(np.std(centered[indices], ddof=0))
            for seed in module.SEEDS:
                prediction, history, record, _ = self.m2.train_run(
                    "RETRAINED_M2_C1", fold, seed, self.data, self.folds,
                    self.features, self.geom_by_fold[fold], self.mu, scale)
                prediction["model"] = "C0"
                all_predictions.append(prediction)
                history["condition"] = "C0"
                all_histories.append(history)
                records.append({**record, "condition": "D0_T1_frozen", "history_rows": len(history)})
                LOG.info("D0 congelado fold=%d seed=%d best_epoch=%s", fold, seed, record["best_epoch"])
        predictions = pd.concat(all_predictions, ignore_index=True)
        if len(predictions) != 630 or predictions.structure_id.nunique() != 210:
            raise ValueError("D0 no produjo exactamente 630 filas OOF")
        _, _, pack = self.m2.metrics_for(predictions, self.mg)
        row = next(r for r in pack["rows"] if r["model"] == "C0" and r["estimand"] == "A")
        pd.DataFrame(records).to_csv(OUT / "d0_checkpoint_manifest.csv", index=False)
        predictions.to_csv(OUT / "d0_predicciones_oof_por_estructura.csv", index=False)
        pd.concat(all_histories, ignore_index=True).to_csv(OUT / "d0_historial_entrenamiento.csv", index=False)
        return predictions, row

    module.LPFTExperiment._baseline = train_frozen_d0


class GeometryLPFT:
    """Thin entry wrapper that calibrates L2-SP from inner-train gradients only."""
    def __init__(self, lpft_module: Any, device: torch.device) -> None:
        """Creates the configured LP-FT experiment and fixes its L2-SP scale."""
        self.module = lpft_module
        self.experiment = lpft_module.LPFTExperiment(device, preflight_only=False)
        self.lambda_info = self.calibrate_lambda()
        lpft_module.LAMBDA_SP = self.lambda_info["lambda_sp"]

    def calibrate_lambda(self) -> dict[str, Any]:
        """Sets lambda so SP/data gradient ratio is 1% on one inner-train triplet."""
        m = self.module; ex = self.experiment; fold = 0; blocks = {10, 11}
        formula = sorted(ex.groups)[0]; indices = ex.groups[formula]
        ids, pos = ex.train_module.make_batch(ex.sequences, indices)
        ids, pos = ids.to(ex.device), pos.to(ex.device)
        geom = torch.as_tensor(ex.geom_by_fold[fold][indices], dtype=torch.float32, device=ex.device)
        energies = ex.data.energy_dft_ev_atom.to_numpy(float)[indices]
        y = torch.as_tensor(energies - energies.mean(), dtype=torch.float32, device=ex.device)
        for p in ex.model.parameters(): p.requires_grad_(False)
        params = m.set_trainable(ex.model, blocks)
        head = ex.m2.Head1().to(ex.device)
        hidden = ex.train_module.contextualized_final_end(ex.model, ids, pos)
        raw = head(torch.cat([hidden, geom], dim=1))
        data_loss = m.residual_loss(raw, y, ex.scales[fold])
        sp_loss = m.l2sp_loss(ex.model, ex.reference, blocks)
        gd = torch.autograd.grad(data_loss, params, retain_graph=True, allow_unused=True)
        gs = torch.autograd.grad(sp_loss, params, allow_unused=True)
        norm_data = float(torch.sqrt(sum(g.float().pow(2).sum() for g in gd if g is not None)))
        norm_sp = float(torch.sqrt(sum(g.float().pow(2).sum() for g in gs if g is not None)))
        raw_ratio = norm_sp / max(norm_data, 1e-30)
        target_weighted_ratio = 0.01
        if not np.isfinite(raw_ratio) or raw_ratio <= 0: raise ValueError("No se pudo calibrar gradiente L2-SP con inner-train")
        value = float(np.clip(target_weighted_ratio / raw_ratio, 10.0, 1e6))
        achieved = value * raw_ratio
        if not np.isfinite(value): raise ValueError("Lambda L2-SP no finito")
        for p in ex.model.parameters(): p.requires_grad_(False)
        return {"lambda_sp": value, "pilot_fold": 0, "pilot_formula": formula,
                "pilot_split": "inner_train", "gradient_norm_supervised": norm_data,
                "gradient_norm_l2sp_raw": norm_sp, "raw_gradient_ratio": raw_ratio,
                "target_weighted_gradient_ratio": target_weighted_ratio,
                "calibrated_weighted_gradient_ratio": achieved,
                "clip_range": [10.0, 1000000.0], "uses_outer": False,
                "normalization": "per-block relative L2-SP from new Alexandria F4+geometry checkpoint"}

    def execute(self) -> None:
        """Runs baseline D0, four LP-FT conditions and adds provenance notes."""
        OUT.mkdir(parents=True, exist_ok=True)
        self.module.atomic_json(OUT / "l2sp_lambda_pilot.json", self.lambda_info)
        self.experiment.execute()
        self.write_transfer_comparison()
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        summary["baseline"] = "D0 trained frozen head on new Alexandria F4+geometry backbone; not historical G1-G"
        summary["condition_mapping"] = {"C0": "D0 frozen", "C1": "D1 F2 no L2-SP", "C2": "D2 F2 with L2-SP",
                                         "C3": "D3 F2→F4 no L2-SP", "C4": "D4 F2→F4 with L2-SP"}
        summary["l2sp_lambda_pilot"] = self.lambda_info
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config_path = OUT / "config.yaml"
        config_path.write_text(config_path.read_text(encoding="utf-8") +
            "\n# F4 base es MatterGPT Alexandria entrenado con geometría explícita.\n"
            f"new_alexandria_checkpoint_sha256: {self.module.F4_SHA}\n"
            f"l2sp_lambda_calibrated_inner_train: {self.lambda_info['lambda_sp']}\n"
            "l2sp_reference: new Alexandria F4 with SLICES plus geometry\n"
            "condition_mapping: {C0: D0_frozen, C1: D1_F2, C2: D2_F2_L2SP, C3: D3_F2_to_F4, C4: D4_F2_to_F4_L2SP}\n", encoding="utf-8")
        report_path = ROOT / "reports/MatterGPT" / OUT.name / "reporte_cientifico.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Alexandria F4 con geometría → especialización ABS₃\n\n"
            "Esta evaluación usa como backbone el nuevo checkpoint MatterGPT entrenado en Alexandria con SLICES y fusión tardía de siete descriptores cristalográficos. D0 es una cabeza residual ABS₃ nueva entrenada con el backbone congelado; no reutiliza el baseline G1-G como sustituto. D1–D4 comparan warmup y F2/F4 con la referencia L2-SP anclada al nuevo checkpoint.\n\n"
            f"SHA-256 F4 Alexandria+geometría: `{self.module.F4_SHA}`.\n\n"
            f"Piloto L2-SP solo inner-train: lambda={self.lambda_info['lambda_sp']:.6g}; relación ponderada de gradientes objetivo 1%; fuera del ajuste no se usaron resultados outer.\n\n"
            "Los CSV, métricas, historial y checkpoints quedan en el directorio `experiments/" + OUT.name + "`. Los identificadores C0–C4 del orquestador corresponden a D0–D4 según `config.yaml`. El test histórico independiente no se abrió.\n", encoding="utf-8")
        readme = ROOT / "experiments" / OUT.name / "README.md"
        readme.write_text("# Especialización ABS₃ desde Alexandria F4 con geometría\n\n"
            "Ejecutar después de completar `mattergpt_alexandria_f4_slices_geometry`:\n\n"
            "```bash\nOMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 "
            "/Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python "
            "scripts/MatterGPT/MatterGPT_Alexandria_F4_SLICES_Geometry/run_abs3_specialization.py --device cpu\n```\n\n"
            "D0 se entrena con nueva cabeza congelada. D1–D4 son F2/F2+L2-SP/F2→F4/F2→F4+L2-SP. El lambda se calibra antes de la evaluación outer con gradientes de una fórmula inner-train y se fija. G1-G histórico queda como referencia externa, separado.\n", encoding="utf-8")
        (OUT / "reporte_cientifico.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    def write_transfer_comparison(self) -> None:
        """Compara D0 con el G1-G histórico y calcula bootstrap por composición."""
        m = self.module; ex = self.experiment
        pred_path = OUT / "predicciones_oof_por_estructura.csv"
        pred = pd.read_csv(pred_path)
        if set(pred.model.unique()) != {"C0", "C1", "C2", "C3", "C4"}:
            raise ValueError("OOF LP-FT no contiene las cinco condiciones esperadas")
        old_path = TRANSFER / "predicciones_oof_por_estructura.csv"
        old = pd.read_csv(old_path)
        old = old[old.model.eq("G1-G")].copy()
        if len(old) != 630:
            raise ValueError("Referencia histórica G1-G no tiene 630 filas OOF")
        frozen = pred[pred.model.eq("C0")].copy(); frozen["model"] = "T1_new_Alexandria_geometry"
        old["model"] = "T0_historical_Alexandria_SLICES_only"
        _, _, pack = ex.m2.metrics_for(pd.concat([frozen, old], ignore_index=True), ex.mg)
        table = pd.DataFrame([r for r in pack["rows"] if r["estimand"] == "A"])
        table.to_csv(OUT / "t1_vs_t0_metrics.csv", index=False)
        fs = pack["formula_seed"]
        values = fs.groupby(["model", "composition_id"], as_index=False).mae_delta_signed_pairs_mev_atom.mean()
        pivot = values.pivot(index="composition_id", columns="model", values="mae_delta_signed_pairs_mev_atom")
        left = "T1_new_Alexandria_geometry"; right = "T0_historical_Alexandria_SLICES_only"
        diff = (pivot[left] - pivot[right]).to_numpy(float)
        rng = np.random.default_rng(20261102)
        draws = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        contrast = {"contrast": "T1 congelado nuevo − T0 congelado histórico", "estimand": "A",
                    "estimate_mev_atom": float(diff.mean()), "ci95_low_mev_atom": float(np.quantile(draws, .025)),
                    "ci95_high_mev_atom": float(np.quantile(draws, .975)), "n_compositions": int(len(diff)),
                    "replicates": 10000, "seed": 20261102, "interpretation": "Desarrollo OOF; el baseline histórico no comparte pesos de cabeza"}
        pd.DataFrame([contrast]).to_csv(OUT / "t1_vs_t0_bootstrap.csv", index=False)
        metrics_path = OUT / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["T1_frozen_vs_T0_historical"] = {"metrics_estimand_A": table.to_dict("records"), "paired_bootstrap": contrast}
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path = OUT / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["T1_frozen_vs_T0_historical_bootstrap"] = contrast
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    """Valida el F4 finalizado y lanza la evaluación congelada/LP-FT ABS₃."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    checkpoint = ENERGY / "checkpoints/mattergpt_alexandria_f4_geometry_best.pt"
    summary = ENERGY / "summary.json"
    if not checkpoint.is_file() or not summary.is_file():
        raise FileNotFoundError("Primero debe terminar Alexandria F4+geometría y guardar summary/checkpoint")
    digest = checksum(checkpoint)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "mps" and not torch.backends.mps.is_available(): parser.error("MPS no disponible")
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "training.log", encoding="utf-8")])
    lpft = import_module(LPFT_SCRIPT, "lpft_geometry_target")
    install_new_loader(lpft, checkpoint, digest)
    runner = GeometryLPFT(lpft, device)
    runner.execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
