"""Experimento C: control congelado y LoRA Q/V con regresión H3-C bloqueada.

Solo se entrena el residuo centrado de energía DFT por átomo en desarrollo ABS3.
Los cinco Ridge históricos se cargan para predicción outer, nunca se ajustan aquí.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lora_matgpt import QVLoRAAttention, insert_qv_lora

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT"
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = STAGE / "MatGPT_frozen_2/v1/fold_assignments.csv"
CACHE = STAGE / "MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
RIDGE = STAGE / "MatGPT_frozen_3/hybrid_ridge_mlp8/H3_B_ridge_fixed_delta/ridge_by_fold"
SOURCE = ROOT / "models/MatGPT-source/mat-concat"
MODEL_INFO = SOURCE / "mat-uncond/weights/model_info_property_0.json"
CHECKPOINT = SOURCE / "mat-uncond/weights/transformer_property_0.pth"
REPORT = ROOT / "reports/MatGPT_SLICES_Finetuning_Energetico_LoRA"
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
EXPECTED = {"checkpoint": "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
            "tokenizer": "d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee",
            "folds": "a5f00a89a41e4645a68d1463f26f9eca10a09bdfd1208193a87e541ff7cd5627"}
LOG = logging.getLogger("matgpt_lora_c")


def sha256(path: Path) -> str:
    """Devuelve SHA-256 de un archivo en lectura secuencial."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Guarda JSON por reemplazo atómico dentro de la carpeta de resultados."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    """Guarda un checkpoint de adaptadores/cabeza sin pesos pretrained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def development() -> pd.DataFrame:
    """Lee solo filas train/val; no interpreta ni exporta targets históricos de test."""
    rows: list[dict[str, Any]] = []
    with DATASET.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            if record["split"] not in ("train", "val"):
                continue
            number = int(record["num_atoms"])
            total = float(record["energy_total_ev"])
            per_atom = float(record["energy_per_atom_ev"])
            if number <= 0 or abs(total / number - per_atom) > 1e-8:
                raise ValueError("Energía E_total/N inválida: " + record["structure_id"])
            rows.append({"structure_id": record["structure_id"],
                         "formula": record["reduced_formula"],
                         "family": record["polymorph_family"], "n_atoms": number,
                         "total_dft_ev": total, "dft_ev_atom": per_atom,
                         "slices": record["matgpt_slices"]})
    frame = pd.DataFrame(rows)
    if len(frame) != 210 or frame.structure_id.nunique() != 210 or frame.formula.nunique() != 70:
        raise ValueError("Dataset de desarrollo distinto de 210 estructuras/70 fórmulas")
    for formula, group in frame.groupby("formula"):
        if len(group) != 3 or set(group.family) != set(FAMILIES):
            raise ValueError("Triplete incompleto: " + str(formula))
    return frame


def assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida los cinco pliegues históricos y sus 45/11/14 fórmulas."""
    folds = pd.read_csv(FOLDS)
    folds = folds[folds.historical_split.isin(["train", "val"])].copy()
    if len(folds) != 1050 or set(folds.structure_id) != set(frame.structure_id):
        raise ValueError("Asignaciones históricas incompatibles")
    lookup = frame.set_index("structure_id").formula
    for fold in range(5):
        part = folds[folds.outer_fold == fold]
        groups = [set(lookup.loc[part[part.inner_role == role].structure_id])
                  for role in ("inner_train", "inner_validation")]
        groups.append(set(lookup.loc[part[part.outer_role == "outer_validation"].structure_id]))
        if list(map(len, groups)) != [45, 11, 14] or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Fuga o tamaños incorrectos en fold {fold}")
    return folds


def feature_cache(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Acepta solo caché histórico o reconstrucción numéricamente auditada."""
    digest = sha256(CACHE)
    audit_path = CACHE.with_suffix(".numeric_audit.json")
    if not audit_path.is_file():
        raise ValueError("Falta auditoría numérica del caché reconstruido")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS_NUMERIC" or audit.get("cache_sha256") != digest or float(audit.get("max_ridge_mean_difference_ev_atom", np.inf)) >= 1e-5:
        raise ValueError("Caché L03/FINAL_END no verificado")
    with np.load(CACHE, allow_pickle=False) as loaded:
        if loaded["structure_id"].astype(str).tolist() != frame.structure_id.tolist():
            raise ValueError("Orden de IDs del caché incompatible")
        values = {key: loaded[key].astype(np.float32) for key in ("L03_MEAN", "FINAL_END")}
    if any(value.shape != (210, 768) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Dimensiones o valores de caché inválidos")
    return values


def model_and_tokens() -> tuple[Any, dict[str, int], int]:
    """Carga MatGPT original estrictamente desde sus rutas verificadas."""
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
    model, stoi, block_size, _ = load_model(SOURCE, MODEL_INFO, CHECKPOINT, SOURCE / "config/uncond.yaml")
    return model, stoi, block_size


def encode(frame: pd.DataFrame, stoi: dict[str, int], block_size: int) -> dict[str, list[int]]:
    """Tokeniza SLICES estándar y rechaza desconocidos/truncamiento."""
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    return {row.structure_id: encode_sequence(row.slices, stoi, block_size) for row in frame.itertuples()}


def formula_groups(frame: pd.DataFrame, ids: set[str]) -> list[tuple[str, list[str]]]:
    """Devuelve tripletes completos en orden estructural histórico."""
    subset = frame[frame.structure_id.isin(ids)]
    groups = []
    for formula, part in subset.groupby("formula", sort=True):
        indexed = part.set_index("family")
        if set(indexed.index) != set(FAMILIES):
            raise ValueError("Partición corta un triplete: " + formula)
        groups.append((formula, [str(indexed.loc[family, "structure_id"]) for family in FAMILIES]))
    if set(sid for _, group in groups for sid in group) != ids:
        raise ValueError("IDs de tripletes desalineados")
    return groups


class MLP8(nn.Module):
    """Cabeza histórica 768→8→1, aplicada a cada estructura independientemente."""

    def __init__(self) -> None:
        """Inicializa exactamente dos capas lineales y GELU/Dropout 0.1."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, embedding: Tensor) -> Tensor:
        """Centra las tres salidas crudas por fórmula en eV/átomo."""
        raw = self.network(embedding).squeeze(-1)
        return raw - raw.mean(dim=-1, keepdim=True)


def final_end(model: Any, ids: list[str], tokens: dict[str, list[int]], pad_id: int) -> Tensor:
    """Extrae estado contextualizado real del <end>, con padding causal."""
    sequences = [tokens[sid] for sid in ids]
    length = max(map(len, sequences))
    batch = torch.full((len(ids), length), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        batch[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    x = model.token_embedding_table(batch)
    x = x + model.position_embedding_table(torch.arange(length))
    x = x.permute(1, 0, 2)
    mask = model.generate_square_subsequent_mask(length)
    memory = torch.zeros(length, len(ids), model.n_embd)
    decoded = model.transformer_decoder(x, memory, tgt_mask=mask).permute(1, 0, 2)
    hidden = model.ln_f(decoded)
    positions = torch.tensor([len(sequence) - 1 for sequence in sequences], dtype=torch.long)
    return hidden[torch.arange(len(ids)), positions]


def residual_target(frame: pd.DataFrame, groups: list[tuple[str, list[str]]]) -> np.ndarray:
    """Calcula residuos DFT por composición, solo para el split recibido."""
    energies = frame.set_index("structure_id").dft_ev_atom
    values = np.array([[energies.loc[sid] for sid in ids] for _, ids in groups], dtype=np.float32)
    return values - values.mean(axis=1, keepdims=True)


def loss_residual(prediction: Tensor, target: Tensor, scale: float) -> Tensor:
    """SmoothL1(beta=1) histórico sobre residuos divididos por escala train."""
    return F.smooth_l1_loss(prediction / scale, target / scale, beta=1.0)


def preflight(output: Path) -> dict[str, Any]:
    """Verifica fuentes, splits, caché, tokenización, Ridge e invariantes base."""
    for name, path in (("checkpoint", CHECKPOINT), ("tokenizer", MODEL_INFO), ("folds", FOLDS)):
        if not path.is_file() or sha256(path) != EXPECTED[name]:
            raise ValueError(f"SHA-256 o ausencia de {name}: {path}")
    frame = development()
    folds = assignments(frame)
    cache = feature_cache(frame)
    model, stoi, block_size = model_and_tokens()
    encoded = encode(frame, stoi, block_size)
    ridge_hashes = {}
    for fold in range(5):
        path = RIDGE / f"fold_{fold}/ridge.joblib"
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "predict"):
            raise ValueError("Ridge inválido: " + str(path))
        ridge_hashes[str(fold)] = sha256(path)
    if len(stoi) != 172 or block_size != 442 or any(len(ids) > block_size for ids in encoded.values()):
        raise ValueError("Tokenizador/contexto MatGPT incompatible")
    if sum(p.numel() for p in model.parameters()) != 114026668:
        raise ValueError("Conteo pretrained inesperado")
    result = {"status": "PASS", "structures": len(frame), "formulas": frame.formula.nunique(),
              "outer_folds": 5, "seeds": [42, 123, 2026],
              "token_lengths": {"min": min(map(len, encoded.values())), "max": max(map(len, encoded.values()))},
              "source_sha256": {"checkpoint": sha256(CHECKPOINT), "tokenizer": sha256(MODEL_INFO),
                                "dataset": sha256(DATASET), "folds": sha256(FOLDS),
                                "cache": sha256(CACHE), "ridge": ridge_hashes},
              "cache_numeric_audit": json.loads(CACHE.with_suffix(".numeric_audit.json").read_text()),
              "architecture": "Ridge(L03_MEAN) + centered MLP8(FINAL_END)",
              "ridge_frozen": True, "historical_test_used": False,
              "torch": torch.__version__, "numpy": np.__version__, "device": "cpu"}
    atomic_json(output / "preflight.json", result)
    LOG.info("%s", {"event": "preflight_pass", "structures": len(frame), "folds": folds.outer_fold.nunique()})
    return result


def run_train(condition: str, fold: int, seed: int, output: Path,
              max_epochs: int = 200, smoke: bool = False) -> dict[str, Any]:
    """Entrena C0/C1 con splits idénticos y selecciona por validación residual."""
    if condition not in ("frozen", "lora") or fold not in range(5) or seed not in (42, 123, 2026):
        raise ValueError("Condición/fold/seed fuera del protocolo bloqueado")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    frame = development(); folds = assignments(frame); cache = feature_cache(frame)
    model, stoi, block_size = model_and_tokens()
    tokens = encode(frame, stoi, block_size)
    part = folds[folds.outer_fold == fold]
    train_ids = set(part[part.inner_role == "inner_train"].structure_id)
    valid_ids = set(part[part.inner_role == "inner_validation"].structure_id)
    outer_ids = set(part[part.outer_role == "outer_validation"].structure_id)
    train_groups = formula_groups(frame, train_ids)
    valid_groups = formula_groups(frame, valid_ids)
    if smoke:
        train_groups, valid_groups = train_groups[:2], valid_groups[:1]
    scale = float(residual_target(frame, train_groups).std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Escala residual train inválida")
    # Reiniciar semilla inmediatamente antes de la cabeza asegura C0/C1 apareados.
    torch.manual_seed(seed)
    head = MLP8()
    adapters: list[QVLoRAAttention] = []
    if condition == "lora":
        adapters = insert_qv_lora(model)
    model.eval()
    for adapter in adapters:
        adapter.train()
        adapter.base.eval()
    if sum(p.numel() for p in head.parameters()) != 6161:
        raise ValueError("MLP8 histórica cambió de tamaño")
    trainable_lora = [p for adapter in adapters for p in adapter.parameters() if p.requires_grad]
    if condition == "lora" and sum(p.numel() for p in trainable_lora) != 24576:
        raise ValueError("LoRA Q/V cambió de tamaño")
    groups = [{"params": list(head.parameters()), "lr": 1e-3, "weight_decay": 1e-4}]
    if adapters:
        groups.append({"params": trainable_lora, "lr": 1e-4, "weight_decay": 1e-2})
    optimizer = torch.optim.AdamW(groups)
    cache_index = {sid: i for i, sid in enumerate(frame.structure_id)}

    def embedding(ids: list[str]) -> Tensor:
        """Obtiene FINAL_END congelado o actualizado, sin caché en C1."""
        if condition == "frozen":
            return torch.from_numpy(np.stack([cache["FINAL_END"][cache_index[sid]] for sid in ids]))
        return final_end(model, ids, tokens, stoi["<pad>"])

    def evaluate(groups_eval: list[tuple[str, list[str]]]) -> float:
        """Promedia SmoothL1 de tripletes completos de inner-validation."""
        head.eval(); model.eval()
        with torch.inference_mode():
            values = []
            for _, ids in groups_eval:
                target = torch.from_numpy(residual_target(frame, [("", ids)])[0])
                values.append(float(loss_residual(head(embedding(ids)), target, scale)))
        for adapter in adapters:
            adapter.train(); adapter.base.eval()
        return float(np.mean(values))

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Corrida existente; no sobrescribir: {output}")
    output.mkdir(parents=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf"); best_epoch = 0; patience = 0; steps = 0
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    epochs = min(max_epochs, 3) if smoke else max_epochs
    for epoch in range(1, epochs + 1):
        head.train()
        for adapter in adapters:
            adapter.train(); adapter.base.eval()
        order = rng.permutation(len(train_groups))
        losses = []
        for offset in range(0, len(order), 8):
            indices = order[offset:offset + 8]
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                _, ids = train_groups[int(index)]
                target = torch.from_numpy(residual_target(frame, [("", ids)])[0])
                prediction = head(embedding(ids))
                loss = loss_residual(prediction, target, scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Pérdida no finita")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            grads = [p.grad for p in list(head.parameters()) + trainable_lora if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Gradientes inexistentes/no finitos")
            if adapters and not any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in trainable_lora):
                raise ValueError("LoRA no recibe gradientes")
            grad_values = [p.grad.detach().float().pow(2).sum() for p in list(head.parameters()) + trainable_lora
                           if p.grad is not None]
            norm = float(torch.sqrt(torch.stack(grad_values).sum()))
            optimizer.step(); steps += 1
        validation = evaluate(valid_groups)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "inner_validation_loss": validation, "optimizer_steps": steps,
               "gradient_norm_last_batch": norm, "elapsed_seconds": time.monotonic() - start}
        history.append(row)
        LOG.info("%s", {"event": "epoch", "condition": condition, "fold": fold,
                       "seed": seed, **row})
        state = {"condition": condition, "fold": fold, "seed": seed,
                 "head": head.state_dict(),
                 "adapters": [{k: v for k, v in adapter.state_dict().items() if not k.startswith("base.")}
                              for adapter in adapters],
                 "optimizer": optimizer.state_dict(), "scale_ev_atom": scale,
                 "epoch": epoch, "optimizer_steps": steps,
                 "checkpoint_sha256": EXPECTED["checkpoint"],
                 "tokenizer_sha256": EXPECTED["tokenizer"],
                 "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": rng.bit_generator.state}
        atomic_torch(output / "last.pt", state)
        if validation < best_loss - 1e-12:
            best_loss, best_epoch, patience = validation, epoch, 0
            atomic_torch(output / "best.pt", state)
        else:
            patience += 1
        if patience >= 20:
            break
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    summary = {"status": "SMOKE" if smoke else "COMPLETE", "condition": condition,
               "fold": fold, "seed": seed, "best_epoch": best_epoch,
               "best_inner_loss": best_loss, "epochs": len(history), "optimizer_steps": steps,
               "scale_ev_atom": scale, "train_formulas": len(train_groups),
               "inner_validation_formulas": len(valid_groups),
               "outer_formulas_held_out": len(formula_groups(frame, outer_ids)),
               "elapsed_seconds": time.monotonic() - start,
               "lora_parameters": sum(p.numel() for p in trainable_lora),
               "head_parameters": sum(p.numel() for p in head.parameters())}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """Ejecuta preflight o una corrida aislada con argumentos validados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train", "smoke"))
    parser.add_argument("--condition", choices=("frozen", "lora"), default="lora")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.max_epochs <= 0 or args.max_epochs > 200:
        parser.error("--max-epochs debe estar entre 1 y 200")
    output = args.output or (REPORT / ("preflight" if args.phase == "preflight" else
                                 f"runs/{args.condition}_fold{args.fold}_seed{args.seed}" +
                                 ("_smoke" if args.phase == "smoke" else "")))
    if args.phase == "preflight":
        result = preflight(output)
    else:
        result = run_train(args.condition, args.fold, args.seed, output,
                           args.max_epochs, args.phase == "smoke")
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
