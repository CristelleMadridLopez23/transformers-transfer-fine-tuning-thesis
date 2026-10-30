"""Entrena la ablación de profundidad MatGPT con trazabilidad y reanudación.

Las etiquetas son residuos centrados de energía DFT por átomo. El checkpoint
MatGPT original, SLICES, Ridge composicional y pliegues se mantienen intactos.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
REPORT = EXPERIMENT
SOURCE = ROOT / "models/MatGPT-source/mat-concat"
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
RIDGE_DIR = ROOT / "reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models"
FEATURES = ROOT / "reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/composition_features.csv"
F0_PRED = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/predicciones_outer_por_estructura.csv"
F0_RUNS = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/runs"
BASE_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
TOKEN_SHA = "d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee"
FOLD_SHA = "a5f00a89a41e4645a68d1463f26f9eca10a09bdfd1208193a87e541ff7cd5627"
LOG = logging.getLogger("matgpt_depth")
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
CONDITIONS = {"F2": (10, 11), "F4": (8, 9, 10, 11),
              "F8": (4, 5, 6, 7, 8, 9, 10, 11)}
SEEDS = (42, 123, 2026)


def sha256(path: Path) -> str:
    """Calcula hash SHA-256 en bloques de memoria acotados."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe JSON mediante reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    """Guarda checkpoint recuperable sin copiar el backbone completo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class MLP8(nn.Module):
    """Regresor residual validado: 768→8→1 con GELU y dropout 0.1."""

    def __init__(self) -> None:
        """Inicializa la topología fija de 6,161 parámetros."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, embedding: Tensor) -> Tensor:
        """Predice y centra residuos sobre los tres polimorfos de cada fórmula."""
        raw = self.network(embedding).squeeze(-1)
        return raw - raw.mean(dim=-1, keepdim=True)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int], int]:
    """Carga únicamente desarrollo train/val y valida fuentes históricas."""
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
    for path, expected in ((SOURCE / "mat-uncond/weights/transformer_property_0.pth", BASE_SHA),
                           (SOURCE / "mat-uncond/weights/model_info_property_0.json", TOKEN_SHA),
                           (FOLDS, FOLD_SHA)):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Fuente ausente o hash inesperado: {path}")
    records: list[dict[str, Any]] = []
    with DATASET.open(newline="", encoding="utf-8") as stream:
        for item in csv.DictReader(stream):
            if item["split"] not in ("train", "val"):
                continue
            n = int(item["num_atoms"])
            total = float(item["energy_total_ev"])
            per_atom = float(item["energy_per_atom_ev"])
            if n <= 0 or abs(total / n - per_atom) > 1e-8:
                raise ValueError(f"Energía DFT inconsistente: {item['structure_id']}")
            records.append({"structure_id": item["structure_id"], "formula": item["reduced_formula"],
                            "family": item["polymorph_family"], "n_atoms": n,
                            "total_dft_ev": total, "dft_ev_atom": per_atom,
                            "slices": item["matgpt_slices"]})
    frame = pd.DataFrame(records)
    if len(frame) != 210 or frame.structure_id.nunique() != 210 or frame.formula.nunique() != 70:
        raise ValueError("Dataset histórico no contiene 210 estructuras/70 fórmulas")
    for formula, part in frame.groupby("formula"):
        if len(part) != 3 or set(part.family) != set(FAMILIES):
            raise ValueError(f"Triplete incompleto: {formula}")
    folds = pd.read_csv(FOLDS)
    folds = folds[folds.historical_split.isin(("train", "val"))].copy()
    if len(folds) != 1050 or set(folds.structure_id) != set(frame.structure_id):
        raise ValueError("Asignaciones históricas no alinean con los 210 IDs de desarrollo")
    formula_map = frame.set_index("structure_id").formula
    for fold in range(5):
        part = folds[folds.outer_fold == fold]
        sets = [set(formula_map.loc[part.loc[part.inner_role == role, "structure_id"]])
                for role in ("inner_train", "inner_validation")]
        sets.append(set(formula_map.loc[part.loc[part.outer_role == "outer_validation", "structure_id"]]))
        if list(map(len, sets)) != [45, 11, 14] or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Separación por composición inválida en pliegue {fold}")
    model, stoi, block_size, _ = load_model(
        SOURCE, SOURCE / "mat-uncond/weights/model_info_property_0.json",
        SOURCE / "mat-uncond/weights/transformer_property_0.pth", SOURCE / "config/uncond.yaml")
    if sum(p.numel() for p in model.parameters()) != 114026668:
        raise ValueError("Conteo de parámetros MatGPT distinto al checkpoint validado")
    if len(model.transformer_decoder.layers) != 12 or model.n_embd != 768:
        raise ValueError("Arquitectura MatGPT distinta de 12 bloques × 768")
    return frame, folds, model, stoi, block_size


def groups_for(frame: pd.DataFrame, ids: set[str]) -> list[tuple[str, list[str]]]:
    """Devuelve tripletes completos en el orden estructural histórico."""
    selected = frame[frame.structure_id.isin(ids)]
    result: list[tuple[str, list[str]]] = []
    for formula, part in selected.groupby("formula", sort=True):
        indexed = part.set_index("family")
        if set(indexed.index) != set(FAMILIES):
            raise ValueError(f"El split divide el triplete de {formula}")
        result.append((str(formula), [str(indexed.loc[name, "structure_id"]) for name in FAMILIES]))
    if {sid for _, triplet in result for sid in triplet} != ids:
        raise ValueError("Los IDs de los tripletes no coinciden con el split")
    return result


def encode_all(frame: pd.DataFrame, stoi: dict[str, int], block_size: int) -> dict[str, list[int]]:
    """Codifica SLICES estándar con tokenizador y contexto originales."""
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    return {str(row.structure_id): encode_sequence(str(row.slices), stoi, block_size)
            for row in frame.itertuples()}


def final_end(model: Any, sequences: list[list[int]], pad_id: int, device: torch.device) -> Tensor:
    """Extrae el estado contextualizado post-ln_f del token <end>."""
    length = max(map(len, sequences))
    batch = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    positions = []
    for row, sequence in enumerate(sequences):
        batch[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        positions.append(len(sequence) - 1)
    x = model.token_embedding_table(batch)
    x = x + model.position_embedding_table(torch.arange(length, device=device))
    x = x.permute(1, 0, 2)
    mask = model.generate_square_subsequent_mask(length).to(device)
    memory = torch.zeros(length, len(sequences), model.n_embd, device=device)
    hidden = model.transformer_decoder(x, memory, tgt_mask=mask).permute(1, 0, 2)
    hidden = model.ln_f(hidden)
    return hidden[torch.arange(len(sequences), device=device), torch.tensor(positions, device=device)]


def target_matrix(frame: pd.DataFrame, groups: list[tuple[str, list[str]]]) -> np.ndarray:
    """Calcula energía DFT/átomo menos media de composición en eV/átomo."""
    energy = frame.set_index("structure_id").dft_ev_atom
    values = np.asarray([[energy.loc[sid] for sid in ids] for _, ids in groups], dtype=np.float32)
    return values - values.mean(axis=1, keepdims=True)


def configure_trainable(model: Any, block_ids: tuple[int, ...]) -> list[nn.Parameter]:
    """Congela todo MatGPT salvo los bloques originales solicitados."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    for index in block_ids:
        for parameter in model.transformer_decoder.layers[index].parameters():
            parameter.requires_grad_(True)
            selected.append(parameter)
    if any(p.requires_grad for p in model.token_embedding_table.parameters()) or any(
            p.requires_grad for p in model.position_embedding_table.parameters()) or any(
            p.requires_grad for p in model.ln_f.parameters()) or any(
            p.requires_grad for p in model.lm_head.parameters()):
        raise ValueError("Módulo que debía permanecer congelado quedó entrenable")
    return selected


def module_states(model: Any, head: MLP8, block_ids: tuple[int, ...]) -> dict[str, Any]:
    """Extrae solo estados de bloques entrenables y MLP8."""
    return {"blocks": {str(i): model.transformer_decoder.layers[i].state_dict() for i in block_ids},
            "head": head.state_dict()}


def load_module_states(model: Any, head: MLP8, state: dict[str, Any], block_ids: tuple[int, ...]) -> None:
    """Restaura pesos adaptados sobre un MatGPT original recién cargado."""
    for index in block_ids:
        model.transformer_decoder.layers[index].load_state_dict(state["blocks"][str(index)], strict=True)
    head.load_state_dict(state["head"], strict=True)


def device_from(name: str) -> torch.device:
    """Resuelve el dispositivo solicitado sin activar aceleradores implícitamente."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                            else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado pero no disponible")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS solicitado pero no disponible")
    return device


def rss_bytes() -> int:
    """Devuelve máximo RSS observado por el proceso en bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def train_one(frame: pd.DataFrame, folds: pd.DataFrame, model: Any, stoi: dict[str, int],
              block_size: int, condition: str, fold: int, seed: int,
              device: torch.device, smoke: bool = False) -> dict[str, Any]:
    """Entrena una corrida con selección inner, checkpoint atómico y pausa reanudable."""
    block_ids = CONDITIONS[condition]
    run_name = f"{condition.lower()}_fold{fold}_seed{seed}" + ("_smoke" if smoke else "")
    out = REPORT / "checkpoints" / run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.flag").exists():
        LOG.info("%s", {"event": "skip_complete", "run": run_name})
        return {"status": "COMPLETE", "run": run_name}
    # Cada condición/fold/semilla vuelve al checkpoint original; no se heredan
    # actualizaciones de la corrida anterior.
    checkpoint_path = SOURCE / "mat-uncond/weights/transformer_property_0.pth"
    original_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(original_state, strict=True)
    del original_state
    gc.collect()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    part = folds[folds.outer_fold == fold]
    train_ids = set(part[part.inner_role == "inner_train"].structure_id)
    valid_ids = set(part[part.inner_role == "inner_validation"].structure_id)
    train_groups = groups_for(frame, train_ids)
    valid_groups = groups_for(frame, valid_ids)
    if smoke:
        train_groups, valid_groups = train_groups[:2], valid_groups[:1]
    scales = float(target_matrix(frame, train_groups).std())
    if not np.isfinite(scales) or scales <= 0:
        raise ValueError("Desviación de residuos inner-train inválida")
    model.to(device)
    transformer_parameters = configure_trainable(model, block_ids)
    torch.manual_seed(seed)
    head = MLP8().to(device)
    if sum(p.numel() for p in head.parameters()) != 6161:
        raise ValueError("El tamaño MLP8 cambió")
    head_initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
    model.eval()
    for idx in block_ids:
        model.transformer_decoder.layers[idx].train()
    head.train()
    optimizer = torch.optim.AdamW([
        {"params": transformer_parameters, "lr": 1e-5, "weight_decay": 0.01},
        {"params": list(head.parameters()), "lr": 1e-3, "weight_decay": 1e-4}],
        betas=(0.9, 0.999))
    encoded = encode_all(frame, stoi, block_size)
    indexed = frame.set_index("structure_id")
    train_targets = target_matrix(frame, train_groups)
    val_targets = target_matrix(frame, valid_groups)
    max_epochs = 3 if smoke else 200
    patience_limit = 2 if smoke else 20
    best_loss = float("inf"); best_epoch = 0; patience = 0; steps = 0; start_epoch = 1
    history: list[dict[str, Any]] = []
    last_path = out / "last.pt"
    if last_path.exists() and not smoke:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("run_name") != run_name or saved.get("base_sha256") != BASE_SHA:
            raise ValueError(f"Checkpoint de reanudación incompatible: {run_name}")
        load_module_states(model, head, saved["model"], block_ids)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        start_epoch = int(saved["epoch"]) + 1
        best_loss = float(saved["best_loss"]); best_epoch = int(saved["best_epoch"])
        patience = int(saved["patience"]); steps = int(saved["optimizer_steps"])
        combined_history = REPORT / "historial_entrenamiento.csv"
        if combined_history.exists():
            previous = pd.read_csv(combined_history)
            history = previous[(previous.condition == condition) & (previous.fold == fold) &
                               (previous.seed == seed) & (previous.run_type == "full")].to_dict("records")
    started = time.monotonic()
    cpu_started = time.process_time()

    def embeddings(ids: list[str], grad: bool) -> Tensor:
        seqs = [encoded[sid] for sid in ids]
        if grad:
            return final_end(model, seqs, stoi["<pad>"], device)
        with torch.no_grad():
            return final_end(model, seqs, stoi["<pad>"], device)

    def validation_loss() -> float:
        model.eval(); head.eval()
        losses: list[float] = []
        with torch.no_grad():
            for index, (_, ids) in enumerate(valid_groups):
                x = embeddings(ids, False)
                y = torch.as_tensor(val_targets[index], dtype=torch.float32, device=device)
                losses.append(float(F.smooth_l1_loss(head(x) / scales, y / scales, beta=1.0)))
        model.eval(); head.train()
        for idx in block_ids:
            model.transformer_decoder.layers[idx].train()
        return float(np.mean(losses))

    for epoch in range(start_epoch, max_epochs + 1):
        model.eval()
        for idx in block_ids:
            model.transformer_decoder.layers[idx].train()
        head.train()
        order = rng.permutation(len(train_groups))
        batch_losses: list[float] = []
        grad_norm_last = 0.0
        for offset in range(0, len(order), 8):
            selected = order[offset:offset + 8]
            optimizer.zero_grad(set_to_none=True)
            formula_losses = []
            for index in selected:
                _, ids = train_groups[int(index)]
                x = embeddings(ids, True)
                y = torch.as_tensor(train_targets[int(index)], dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(head(x) / scales, y / scales, beta=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Pérdida no finita")
                formula_losses.append(loss)
            batch_loss = torch.stack(formula_losses).mean()
            batch_loss.backward()
            active = transformer_parameters + list(head.parameters())
            grads = [p.grad for p in active if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Gradientes inexistentes o no finitos")
            if not any(bool(torch.any(p.grad != 0)) for p in transformer_parameters if p.grad is not None):
                raise FloatingPointError(f"Bloques {block_ids} sin gradiente no nulo")
            grad_norm_last = float(torch.sqrt(torch.stack([g.detach().float().pow(2).sum() for g in grads]).sum()))
            optimizer.step(); steps += 1
            batch_losses.append(float(batch_loss.detach()))
        val = validation_loss()
        row = {"condition": condition, "fold": fold, "seed": seed, "epoch": epoch,
               "run_type": "smoke" if smoke else "full",
               "train_loss": float(np.mean(batch_losses)), "inner_validation_loss": val,
               "optimizer_steps": steps, "gradient_norm_last_batch": grad_norm_last,
               "elapsed_seconds": time.monotonic() - started, "rss_peak_bytes": rss_bytes()}
        history.append(row)
        combined_history = REPORT / "historial_entrenamiento.csv"
        combined_history.parent.mkdir(parents=True, exist_ok=True)
        exists = combined_history.exists()
        pd.DataFrame([row]).to_csv(combined_history, mode="a", header=not exists, index=False)
        if val < best_loss - 1e-12:
            best_loss, best_epoch, patience = val, epoch, 0
            atomic_torch(out / "best.pt", {"model": module_states(model, head, block_ids),
                        "condition": condition, "fold": fold, "seed": seed,
                        "blocks": block_ids, "scale_ev_atom": scales,
                        "base_sha256": BASE_SHA, "tokenizer_sha256": TOKEN_SHA,
                        "best_epoch": epoch, "best_inner_loss": val})
        else:
            patience += 1
        state = {"run_name": run_name, "base_sha256": BASE_SHA,
                 "model": module_states(model, head, block_ids), "optimizer": optimizer.state_dict(),
                 "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": rng.bit_generator.state,
                 "epoch": epoch, "optimizer_steps": steps, "best_loss": best_loss,
                 "best_epoch": best_epoch, "patience": patience}
        atomic_torch(last_path, state)
        LOG.info("%s", {"event": "epoch", **row, "best_epoch": best_epoch,
                         "best_validation_loss": best_loss})
        update_resource(run_name, "RUNNING", time.monotonic() - started, rss_bytes(), epoch, steps,
                        cpu_seconds=time.process_time() - cpu_started)
        if patience >= patience_limit:
            break
    result = {"status": "SMOKE" if smoke else "COMPLETE", "condition": condition,
              "fold": fold, "seed": seed, "blocks": list(block_ids),
              "trainable_transformer_parameters": sum(p.numel() for p in transformer_parameters),
              "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
              "head_initial_sha256": head_initial_hash, "best_epoch": best_epoch,
              "best_inner_loss": best_loss, "epochs": len(history), "optimizer_steps": steps,
              "target_scale_ev_atom": scales, "elapsed_seconds": time.monotonic() - started,
              "cpu_seconds": time.process_time() - cpu_started,
              "rss_peak_bytes": rss_bytes(), "device": str(device)}
    if not smoke:
        (out / "complete.flag").write_text("complete\n")
    update_resource(run_name, result["status"], result["elapsed_seconds"], result["rss_peak_bytes"],
                    result["epochs"], steps, result, result["cpu_seconds"])
    return result


def update_resource(run: str, status: str, seconds: float, rss: int, epochs: int,
                    steps: int, detail: dict[str, Any] | None = None,
                    cpu_seconds: float = 0.0) -> None:
    """Actualiza el único inventario de recursos consolidado."""
    path = REPORT / "resources.json"
    data = json.loads(path.read_text()) if path.exists() else {"status": "RUNNING", "runs": {}}
    data.setdefault("runs", {})[run] = {"status": status, "wall_seconds": seconds,
                                        "cpu_seconds": cpu_seconds, "peak_rss_bytes": rss, "epochs": epochs,
                                        "optimizer_steps": steps, **(detail or {})}
    data["completed_runs"] = sum(v.get("status") == "COMPLETE" for v in data["runs"].values())
    data["active_runs"] = sum(v.get("status") == "RUNNING" for v in data["runs"].values())
    data["wall_seconds_observed_total"] = sum(v.get("wall_seconds", 0) for v in data["runs"].values())
    data["cpu_seconds_observed_total"] = sum(v.get("cpu_seconds", 0) for v in data["runs"].values())
    data["peak_rss_bytes_observed"] = max((v.get("peak_rss_bytes", 0) for v in data["runs"].values()), default=0)
    atomic_json(path, data)


def main() -> None:
    """Valida fuentes, ejecuta smokes y/o lanza secuencialmente F2/F4/F8."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "all"), required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads debe ser positivo")
    REPORT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(REPORT / "run.log", encoding="utf-8")])
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    device = device_from(args.device)
    LOG.info("%s", {"event": "start", "phase": args.phase, "device": str(device),
                     "torch": torch.__version__, "numpy": np.__version__, "cpu_count": os.cpu_count()})
    frame, folds, model, stoi, block_size = load_sources()
    tokens = encode_all(frame, stoi, block_size)
    if len(frame) != 210 or frame.formula.nunique() != 70 or min(map(len, tokens.values())) < 3:
        raise ValueError("Dataset o tokenización de desarrollo inválidos")
    feature_table = pd.read_csv(FEATURES).set_index("composition_id")
    feature_columns = [column for column in feature_table.columns if column not in ("composition_id", "target_ev_atom")]
    if len(feature_columns) != 17:
        raise ValueError(f"Se esperaban 17 fracciones; hay {len(feature_columns)}")
    if args.phase == "preflight":
        ridge_hashes = {str(i): sha256(RIDGE_DIR / f"ridge_composicional_fold_{i}.joblib") for i in range(5)}
        f0 = pd.read_csv(F0_PRED)
        f0 = f0[f0.condition == "frozen"]
        if len(f0) != 630 or f0.structure_id.nunique() != 210:
            raise ValueError("F0 histórico no contiene 630 predicciones de las 210 estructuras")
        for run in (F0_RUNS / f"frozen_fold{f}_seed{s}" for f in range(5) for s in SEEDS):
            meta = json.loads((run / "summary.json").read_text())
            if meta.get("status") != "COMPLETE":
                raise ValueError(f"F0 no está completo: {run}")
        summary = {"status": "PREFLIGHT_PASS", "structures": len(frame), "compositions": 70,
                   "folds": 5, "fold_counts": {str(f): {role: int(folds[(folds.outer_fold == f) &
                     ((folds.inner_role == role) if role.startswith("inner") else (folds.outer_role == role))].composition_group.nunique())
                     for role in ("inner_train", "inner_validation", "outer_validation")} for f in range(5)},
                   "token_length_min": min(map(len, tokens.values())), "token_length_max": max(map(len, tokens.values())),
                   "checkpoint_sha256": BASE_SHA, "tokenizer_sha256": TOKEN_SHA,
                   "folds_sha256": FOLD_SHA, "ridge_sha256": ridge_hashes,
                   "ridge_refit": False, "f0_reused": True, "device": str(device),
                   "architecture": {"blocks": 12, "hidden": 768, "total_parameters": 114026668,
                                    "head": "Linear(768,8)-GELU-Dropout(0.1)-Linear(8,1)", "head_parameters": 6161}}
        atomic_json(REPORT / "summary.json", summary)
        LOG.info("%s", summary)
        return
    if args.phase == "smoke":
        for condition in CONDITIONS:
            train_one(frame, folds, model, stoi, block_size, condition, 0, 42, device, smoke=True)
        LOG.info("%s", {"event": "all_smokes_pass"})
        summary_path = REPORT / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["smoke_test"] = "PASS: F2, F4, F8 forward/backward/optimizer"
            atomic_json(summary_path, summary)
        return
    preflight_path = REPORT / "summary.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text()).get("status") not in ("PREFLIGHT_PASS", "RUNNING"):
        raise RuntimeError("Debe pasar --phase preflight y --phase smoke antes del entrenamiento completo")
    resource_path = REPORT / "resources.json"
    if not resource_path.exists():
        atomic_json(resource_path, {"status": "RUNNING", "device": str(device),
                    "cpu_count": os.cpu_count(), "num_threads": args.num_threads,
                    "conditions": {k: {"blocks": v, "folds": 5, "seeds": list(SEEDS)} for k, v in CONDITIONS.items()},
                    "f0_reused_runs": 15, "runs": {}})
    atomic_json(preflight_path, {**json.loads(preflight_path.read_text()), "status": "RUNNING",
                "planned_finetuning_runs": 45, "f0_reused_runs": 15})
    for condition in CONDITIONS:
        for fold in range(5):
            for seed in SEEDS:
                LOG.info("%s", {"event": "run_start", "condition": condition, "fold": fold, "seed": seed})
                train_one(frame, folds, model, stoi, block_size, condition, fold, seed, device)
    final = json.loads(preflight_path.read_text())
    final["status"] = "TRAINING_COMPLETE"
    atomic_json(preflight_path, final)
    LOG.info("%s", {"event": "all_runs_complete"})


if __name__ == "__main__":
    main()
