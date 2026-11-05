"""Entrena y evalúa las condiciones ABS₃ congeladas/F1/F2 emparejadas.

Reutiliza los lectores, tokenizadores y extractores históricos; solo los bloques
finales seleccionados y la cabeza MLP8 reciben gradientes en F1/F2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/abs3_original_supervised_ft_f0_f1_f2"
SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("abs3_original_supervised_ft_f0_f1_f2")
SEEDS = (42, 123, 2026)
MODELS = ("MatGPT", "MatterGPT")
MODALITIES = ("SLICES", "SLICES_GEOMETRY")
POLICIES = ("F0", "F1", "F2")
BACKBONE_LR_GRID = (1e-6, 3e-6, 1e-5)
GEOMETRY = ("a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
            "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3")
FAMILY_ORDER = ("dist_perovskite", "hexagonal", "needle")
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))
EPOCHS_WARMUP = 20
EPOCHS_CONTINUATION = 71
PATIENCE_WARMUP = 8
PATIENCE_CONTINUATION = 12
MIN_DELTA = 0.0
LR_WARMUP = 1e-3
LR_HEAD = 1e-4
WD_HEAD = 1e-4
WD_BACKBONE = 1e-2
BATCH_COMPOSITIONS = 8
THREADS = 3

HISTORY_COLUMNS = [
    "model", "modality", "condition", "fold", "seed", "backbone_lr", "epoch",
    "execution_phase", "train_minibatch_smoothl1", "inner_validation_smoothl1",
    "inner_validation_delta_mae_mev_atom", "inner_validation_pairwise_accuracy",
    "gradient_norm_backbone_last_batch", "gradient_norm_head_last_batch",
    "active_backbone_parameter_count",
]


class MLP8(nn.Module):
    """Cabeza histórica MLP8 con entrada de 768 o 775 valores."""

    def __init__(self, input_dim: int) -> None:
        """Construye la cabeza con dimensión verificada.

        Args:
            input_dim: 768 para SLICES o 775 para fusión geométrica.
        """
        super().__init__()
        if input_dim not in (768, 775):
            raise ValueError(f"Dimensión MLP8 no permitida: {input_dim}")
        self.net = nn.Sequential(nn.Linear(input_dim, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, values: Tensor) -> Tensor:
        """Predice un score escalar por estructura.

        Args:
            values: Tensor [n_estructuras, input_dim].
        Returns:
            Scores [n_estructuras].
        """
        return self.net(values).squeeze(-1)


def load_helper() -> Any:
    """Importa el experimento F4 solo para reutilizar funciones auditadas.

    Returns:
        Módulo de utilidades F4 históricas.
    """
    helper_path = ROOT / "scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py"
    spec = importlib.util.spec_from_file_location("abs3_original_f4_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar helper histórico: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Escribe JSON por reemplazo atómico.

    Args:
        path: Ruta de destino.
        data: Objeto serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def state_cpu(module: nn.Module) -> dict[str, Tensor]:
    """Copia el estado de un módulo a CPU para un checkpoint pequeño.

    Args:
        module: Módulo PyTorch.
    Returns:
        Diccionario de tensores clonados en CPU.
    """
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def stack_for(model: nn.Module, model_name: str) -> Any:
    """Devuelve la pila Transformer identificada por arquitectura.

    Args:
        model: Backbone cargado estrictamente.
        model_name: MatGPT o MatterGPT.
    Returns:
        Secuencia de 12 bloques.
    """
    return model.transformer_decoder.layers if model_name == "MatGPT" else model.blocks


def configure_policy(model: nn.Module, model_name: str, policy: str) -> list[nn.Parameter]:
    """Congela el backbone y habilita solo el último bloque o los dos últimos.

    Args:
        model: Backbone original.
        model_name: Nombre de arquitectura.
        policy: F0, F1 o F2.
    Returns:
        Parámetros habilitados en el backbone.
    """
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stack = stack_for(model, model_name)
    if len(stack) != 12:
        raise ValueError(f"{model_name} no tiene 12 bloques")
    indices = {"F0": (), "F1": (11,), "F2": (10, 11)}[policy]
    active: list[nn.Parameter] = []
    for index in indices:
        for parameter in stack[index].parameters():
            parameter.requires_grad_(True)
            active.append(parameter)
    expected = {"F0": 0, "F1": len(list(stack[11].parameters())),
                "F2": len(list(stack[10].parameters())) + len(list(stack[11].parameters()))}[policy]
    if len(active) != expected:
        raise RuntimeError(f"Parámetros desbloqueados inesperados para {model_name}/{policy}")
    return active


def set_model_train_mode(model: nn.Module, model_name: str, policy: str) -> None:
    """Mantiene backbone congelado en eval y activa dropout en bloques F1/F2.

    Args:
        model: Backbone Transformer.
        model_name: Arquitectura.
        policy: F0, F1 o F2.
    """
    model.eval()
    stack = stack_for(model, model_name)
    for index in {"F0": (), "F1": (11,), "F2": (10, 11)}[policy]:
        stack[index].train()


def formulas_for_fold(folds: pd.DataFrame, fold: int) -> tuple[list[str], list[str], list[str]]:
    """Obtiene listas de fórmulas por partición del fold.

    Args:
        folds: Manifiesto histórico solo de train/val.
        fold: Pliegue externo.
    Returns:
        Fórmulas de train, validation y outer.
    """
    selected = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
    train = sorted(selected.loc[selected.inner_role.eq("inner_train"), "composition_group"].astype(str))
    valid = sorted(selected.loc[selected.inner_role.eq("inner_validation"), "composition_group"].astype(str))
    outer = sorted(selected.loc[selected.outer_role.eq("outer_validation"), "composition_group"].astype(str))
    if (len(train), len(valid), len(outer)) != (45, 11, 14):
        raise ValueError(f"Fold {fold}: particiones inválidas {(len(train), len(valid), len(outer))}")
    return train, valid, outer


def pair_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Calcula MAE firmado de tres pares y ranking elegible por triplete.

    Args:
        scores: Tres salidas MLP8 en eV/átomo.
        targets: Tres energías DFT en eV/átomo.
    Returns:
        Métricas internas usadas para selección.
    """
    pred = np.asarray([scores[j] - scores[i] for i, j in PAIR_ORDER]) * 1000.0
    truth = np.asarray([targets[j] - targets[i] for i, j in PAIR_ORDER]) * 1000.0
    eligible = np.abs(truth) > 5.0
    unresolved = np.abs(pred) <= 1.0
    correct = (np.sign(pred) == np.sign(truth)) & ~unresolved
    return {"delta_mae": float(np.mean(np.abs(pred - truth))),
            "pairwise_accuracy": float(np.mean(correct[eligible])) if eligible.any() else float("nan"),
            "eligible_pairs": int(eligible.sum()), "unresolved_pairs": int((unresolved & eligible).sum())}


def make_features(helper: Any, model: nn.Module | None, model_name: str, ids: list[str],
                  row_by_id: dict[str, int], embeddings: np.ndarray, geom_scaled: np.ndarray,
                  modality: str, encoded: dict[str, list[int]] | None,
                  tokenizer: dict[str, int] | None, device: torch.device) -> Tensor:
    """Genera características cached (F0) o con autograd (F1/F2).

    Args:
        helper: Utilidades validadas de extracción.
        model: Backbone vivo; None indica usar caché congelada.
        model_name: MatGPT/MatterGPT.
        ids: Tres o más IDs estables.
        row_by_id: Índice dataset por structure_id.
        embeddings: Caché [210,768].
        geom_scaled: Geometría escalada por fold.
        modality: SLICES o SLICES_GEOMETRY.
        encoded: Token IDs por estructura para F1/F2.
        tokenizer: Vocabulario del extractor para F1/F2.
        device: CPU local.
    Returns:
        Características float32 por estructura.
    """
    rows = [row_by_id[sid] for sid in ids]
    if model is None:
        hidden = torch.as_tensor(embeddings[rows], dtype=torch.float32, device=device)
    else:
        if encoded is None or tokenizer is None:
            raise ValueError("F1/F2 requiere secuencias tokenizadas")
        hidden = helper.live_embeddings(model, model_name, ids, encoded, tokenizer, device)
    if modality == "SLICES_GEOMETRY":
        geometry = torch.as_tensor(geom_scaled[rows], dtype=torch.float32, device=device)
        return torch.cat((hidden, geometry), dim=-1)
    return hidden


def predict_validation(helper: Any, head: MLP8, model: nn.Module | None, model_name: str,
                      modality: str, formulas: list[str], groups: dict[str, list[int]],
                      targets: dict[str, np.ndarray], data: pd.DataFrame, row_by_id: dict[str, int],
                      embeddings: np.ndarray, geom_scaled: np.ndarray,
                      encoded: dict[str, list[int]] | None, tokenizer: dict[str, int] | None,
                      scale: float, device: torch.device) -> tuple[float, float, float]:
    """Calcula MAE ΔE, exactitud pairwise y SmoothL1 en validation completa.

    Args:
        helper: Módulo con `live_embeddings`.
        head: MLP8.
        model: Backbone o None para caché F0.
        model_name: Arquitectura.
        modality: Modalidad de entrada.
        formulas: Fórmulas completas de validation.
        groups: Índices de triplete por fórmula.
        targets: Energías DFT por fórmula.
        data: Tabla de desarrollo.
        row_by_id: Índice por structure_id.
        embeddings: Caché FINAL_END.
        geom_scaled: Variables geométricas escaladas.
        encoded: Entradas tokenizadas.
        tokenizer: Tokenizador.
        scale: Residual standard deviation fit only on inner-train.
        device: CPU.
    Returns:
        (MAE ΔE, pairwise accuracy, SmoothL1 de residuo normalizado).
    """
    head.eval()
    if model is not None:
        model.eval()
    errors: list[float] = []
    correct: list[bool] = []
    losses: list[float] = []
    with torch.no_grad():
        for formula in formulas:
            ids = [str(data.iloc[index].structure_id) for index in groups[formula]]
            x = make_features(helper, model, model_name, ids, row_by_id, embeddings, geom_scaled,
                              modality, encoded, tokenizer, device)
            scores = head(x).detach().cpu().numpy()
            metric = pair_metrics(scores, targets[formula])
            errors.append(metric["delta_mae"])
            pred_center = scores - scores.mean()
            true_center = targets[formula] - targets[formula].mean()
            losses.append(float(F.smooth_l1_loss(torch.as_tensor(pred_center / scale),
                                                 torch.as_tensor(true_center / scale), beta=1.0)))
            for i, j in PAIR_ORDER:
                dy = (targets[formula][j] - targets[formula][i]) * 1000
                dp = (scores[j] - scores[i]) * 1000
                if abs(dy) > 5:
                    correct.append(bool(np.sign(dp) == np.sign(dy) and abs(dp) > 1))
    return float(np.mean(errors)), float(np.mean(correct)) if correct else float("nan"), float(np.mean(losses))


def train_head_warmup(helper: Any, model_name: str, modality: str, fold: int, seed: int,
                      data: pd.DataFrame, groups: dict[str, list[int]], targets: dict[str, np.ndarray],
                      row_by_id: dict[str, int], embeddings: np.ndarray, geom_scaled: np.ndarray,
                      train: list[str], valid: list[str], scale: float, device: torch.device,
                      max_epochs: int, phase: str) -> tuple[MLP8, dict[str, Any]]:
    """Ejecuta warmup compartido de MLP8 y selecciona por MAE ΔE validation.

    Args:
        helper: Utilidades históricas.
        model_name: MatGPT/MatterGPT.
        modality: SLICES o SLICES_GEOMETRY.
        fold: Fold externo.
        seed: Semilla.
        data: Dataset ABS3.
        groups: Índices de tripletes.
        targets: Energías DFT.
        row_by_id: Índice de IDs.
        embeddings: Representaciones congeladas.
        geom_scaled: Geometría escalada.
        train: Fórmulas de train.
        valid: Fórmulas de validation.
        scale: Escala residual inner-train.
        device: CPU.
        max_epochs: Presupuesto de piloto o experimento.
    Returns:
        Cabeza restaurada al mejor checkpoint y resumen.
    """
    input_dim = 775 if modality == "SLICES_GEOMETRY" else 768
    torch.manual_seed(seed + 1009 * fold)
    np.random.seed(seed + 1009 * fold)
    random.seed(seed + 1009 * fold)
    head = MLP8(input_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR_WARMUP, weight_decay=WD_HEAD)
    rng = np.random.default_rng(seed + 1009 * fold)
    best_mae, best_pair, best_epoch, stale = math.inf, -math.inf, 0, 0
    best_state: dict[str, Tensor] | None = None
    histories: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = np.asarray(train)[rng.permutation(len(train))]
        losses: list[float] = []
        for start in range(0, len(order), BATCH_COMPOSITIONS):
            formulas = [str(item) for item in order[start:start + BATCH_COMPOSITIONS]]
            ids = [str(data.iloc[i].structure_id) for f in formulas for i in groups[f]]
            x = make_features(helper, None, model_name, ids, row_by_id, embeddings, geom_scaled,
                              modality, None, None, device)
            scores = head(x).view(len(formulas), 3)
            centered = scores - scores.mean(dim=1, keepdim=True)
            target = torch.as_tensor(np.stack([targets[f] - targets[f].mean() for f in formulas]),
                                     dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(centered / scale, target / scale, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Warmup produjo loss no finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError("Warmup MLP8 tiene gradiente ausente o no finito")
            optimizer.step()
            losses.append(float(loss.detach()))
        val_mae, val_pair, val_loss = predict_validation(helper, head, None, model_name, modality,
            valid, groups, targets, data, row_by_id, embeddings, geom_scaled, None, None, scale, device)
        histories.append({"model": model_name, "modality": modality, "condition": "WARMUP",
                          "fold": fold, "seed": seed, "epoch": epoch, "execution_phase": phase,
                          "train_minibatch_smoothl1": float(np.mean(losses)),
                          "inner_validation_smoothl1": val_loss,
                          "inner_validation_delta_mae_mev_atom": val_mae,
                          "inner_validation_pairwise_accuracy": val_pair})
        improved = val_mae < best_mae - 1e-12 or (abs(val_mae - best_mae) <= 1e-12 and val_pair > best_pair)
        if improved:
            best_mae, best_pair, best_epoch, stale = val_mae, val_pair, epoch, 0
            best_state = state_cpu(head)
        else:
            stale += 1
            if stale >= PATIENCE_WARMUP:
                break
    if best_state is None:
        raise RuntimeError("Warmup no produjo un checkpoint válido")
    head.load_state_dict(best_state, strict=True)
    path = EXP / "warmup" / phase / model_name / modality / f"fold_{fold}_seed_{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": best_state, "input_dim": input_dim, "model": model_name,
                "modality": modality, "fold": fold, "seed": seed, "best_epoch": best_epoch,
                "best_validation_delta_mae_mev_atom": best_mae, "state_sha256": tensor_hash(best_state)}, path)
    append_rows(EXP / f"training_history_{phase}.csv", histories, HISTORY_COLUMNS)
    return head, {"path": str(path.relative_to(ROOT)), "best_epoch": best_epoch,
                  "best_validation_delta_mae_mev_atom": best_mae, "sha256": file_hash(path)}


def tensor_hash(state: dict[str, Tensor]) -> str:
    """Calcula un SHA determinista de los tensores ordenados del estado.

    Args:
        state: Mapping de tensores.
    Returns:
        SHA-256.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Calcula SHA-256 de un archivo.

    Args:
        path: Archivo.
    Returns:
        SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Agrega filas CSV y escribe encabezado solo al crear el archivo.

    Args:
        path: CSV.
        rows: Filas homogéneas.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def fit_candidate(helper: Any, model_name: str, modality: str, policy: str, fold: int, seed: int,
                  backbone_lr: float, warm_head: MLP8, data: pd.DataFrame,
                  groups: dict[str, list[int]], targets: dict[str, np.ndarray], row_by_id: dict[str, int],
                  embeddings: np.ndarray, geom_scaled: np.ndarray, train: list[str], valid: list[str],
                  scale: float, device: torch.device, encoded: dict[str, list[int]],
                  tokenizer: dict[str, int], max_epochs: int, phase: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Entrena una continuación desde el mismo warmup para una tasa de backbone.

    Args:
        helper: Extractor histórico.
        model_name: MatGPT/MatterGPT.
        modality: SLICES o SLICES_GEOMETRY.
        policy: F0/F1/F2.
        fold: Fold.
        seed: Semilla.
        backbone_lr: LR candidato del backbone; ignorado para F0.
        warm_head: MLP8 del warmup compartido.
        data: Dataset.
        groups: Índices de triplete.
        targets: Energías DFT.
        row_by_id: Índice de estructuras.
        embeddings: Caché original.
        geom_scaled: Geometría escalada.
        train: Fórmulas train.
        valid: Fórmulas validation.
        scale: Escala de residuos train.
        device: CPU.
        encoded: Secuencias.
        tokenizer: Vocabulario.
        max_epochs: Épocas para piloto/full.
    Returns:
        Checkpoint de validación ganador y su historial de época.
    """
    model = None
    active: list[nn.Parameter] = []
    if policy != "F0":
        model, loaded, audit = helper.load_model_base(model_name, helper.import_extractor())
        if audit["checkpoint_sha256"] != helper.EXPECTED_BASE[model_name]:
            raise ValueError(f"Hash base cambió para {model_name}")
        active = configure_policy(model, model_name, policy)
        model.to(device)
    head = MLP8(775 if modality == "SLICES_GEOMETRY" else 768).to(device)
    head.load_state_dict(warm_head.state_dict(), strict=True)
    if policy == "F0":
        optimizer = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=WD_HEAD)
    else:
        optimizer = torch.optim.AdamW([
            {"params": active, "lr": backbone_lr, "weight_decay": WD_BACKBONE},
            {"params": list(head.parameters()), "lr": LR_HEAD, "weight_decay": WD_HEAD},
        ])
    stack = stack_for(model, model_name) if model is not None else None
    order_rng = np.random.default_rng(seed + 1009 * fold)
    torch.manual_seed(seed + 1009 * fold)
    best_mae, best_pair, best_epoch, stale = math.inf, -math.inf, 0, 0
    best_head: dict[str, Tensor] | None = None
    best_blocks: dict[str, dict[str, Tensor]] = {}
    history: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    candidate_started = time.monotonic()
    frozen_parameters_checked = False
    for epoch in range(1, max_epochs + 1):
        head.train()
        if model is not None:
            set_model_train_mode(model, model_name, policy)
        order = np.asarray(train)[order_rng.permutation(len(train))]
        batch_losses: list[float] = []
        grad_backbone = 0.0
        grad_head = 0.0
        for start in range(0, len(order), BATCH_COMPOSITIONS):
            formulas = [str(value) for value in order[start:start + BATCH_COMPOSITIONS]]
            ids = [str(data.iloc[index].structure_id) for formula in formulas for index in groups[formula]]
            features = make_features(helper, model, model_name, ids, row_by_id, embeddings, geom_scaled,
                                     modality, encoded if model is not None else None,
                                     tokenizer if model is not None else None, device)
            raw = head(features).view(len(formulas), 3)
            pred = raw - raw.mean(dim=1, keepdim=True)
            target = torch.as_tensor(np.stack([targets[f] - targets[f].mean() for f in formulas]),
                                     dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(pred / scale, target / scale, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss no finita: {model_name}/{modality}/{policy}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = active + list(head.parameters())
            gradients = [parameter.grad for parameter in params]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("Gradiente ausente o NaN/Inf")
            grad_backbone = float(torch.sqrt(torch.stack([p.grad.detach().float().pow(2).sum()
                for p in active]).sum()).item()) if active else 0.0
            grad_head = float(torch.sqrt(torch.stack([p.grad.detach().float().pow(2).sum()
                for p in head.parameters()]).sum()).item())
            if policy != "F0" and grad_backbone <= 0:
                raise FloatingPointError(f"No hay gradiente efectivo en bloques {policy}")
            before_frozen = None
            if model is not None and not frozen_parameters_checked:
                before_frozen = [p.detach().clone() for p in model.parameters() if not p.requires_grad]
            optimizer.step()
            if model is not None and before_frozen is not None:
                frozen = [p for p in model.parameters() if not p.requires_grad]
                if any(not torch.equal(p.detach(), old) for p, old in zip(frozen, before_frozen)):
                    raise RuntimeError("Cambió un parámetro congelado")
                frozen_parameters_checked = True
            batch_losses.append(float(loss.detach()))
        val_mae, val_pair, val_loss = predict_validation(helper, head, model, model_name, modality,
            valid, groups, targets, data, row_by_id, embeddings, geom_scaled,
            encoded if model is not None else None, tokenizer if model is not None else None, scale, device)
        row = {"model": model_name, "modality": modality, "condition": policy, "fold": fold,
               "seed": seed, "backbone_lr": backbone_lr if policy != "F0" else None,
               "epoch": epoch, "execution_phase": phase,
               "train_minibatch_smoothl1": float(np.mean(batch_losses)),
               "inner_validation_smoothl1": val_loss,
               "inner_validation_delta_mae_mev_atom": val_mae,
               "inner_validation_pairwise_accuracy": val_pair,
               "gradient_norm_backbone_last_batch": grad_backbone,
               "gradient_norm_head_last_batch": grad_head,
               "active_backbone_parameter_count": sum(p.numel() for p in active)}
        epoch_rows.append(row)
        append_rows(EXP / f"training_history_{phase}.csv", [row], HISTORY_COLUMNS)
        improved = val_mae < best_mae - 1e-12 or (abs(val_mae - best_mae) <= 1e-12 and val_pair > best_pair)
        if improved:
            best_mae, best_pair, best_epoch, stale = val_mae, val_pair, epoch, 0
            best_head = state_cpu(head)
            best_blocks = {str(i): state_cpu(stack[i]) for i in ({"F0": (), "F1": (11,), "F2": (10, 11)}[policy])} if stack is not None else {}
        else:
            stale += 1
            if stale >= PATIENCE_CONTINUATION:
                break
    if best_head is None:
        raise RuntimeError("Continuación no guardó checkpoint por inner-validation")
    branch_id = f"{phase}_{model_name}_{modality}_{policy}_fold{fold}_seed{seed}_lr{backbone_lr:g}"
    candidate_path = EXP / "continuation" / branch_id / "best.pt"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": best_head, "blocks": best_blocks, "best_epoch": best_epoch,
                "best_validation_delta_mae_mev_atom": best_mae,
                "best_validation_pairwise_accuracy": best_pair, "model": model_name,
                "modality": modality, "policy": policy, "fold": fold, "seed": seed,
                "backbone_lr": backbone_lr, "ridge_sha256": helper.RIDGE_HASHES[fold]}, candidate_path)
    record = {"candidate": branch_id, "model": model_name, "modality": modality,
              "policy": policy, "fold": fold, "seed": seed,
              "backbone_lr": backbone_lr if policy != "F0" else None,
              "best_epoch": best_epoch, "epochs_run": epoch,
              "best_validation_delta_mae_mev_atom": best_mae,
              "best_validation_pairwise_accuracy": best_pair,
              "checkpoint": str(candidate_path.relative_to(ROOT)),
              "checkpoint_sha256": file_hash(candidate_path),
              "active_backbone_parameters": sum(p.numel() for p in active),
              "head_parameters": sum(p.numel() for p in head.parameters()),
              "parameter_gradients_verified": True,
              "elapsed_seconds": time.monotonic() - candidate_started,
              "execution_phase": phase}
    del model
    return record, epoch_rows


def evaluate_outer(helper: Any, model_name: str, modality: str, policy: str, fold: int, seed: int,
                   candidate: dict[str, Any], data: pd.DataFrame, groups: dict[str, list[int]],
                   targets: dict[str, np.ndarray], row_by_id: dict[str, int], embeddings: np.ndarray,
                   geom_scaled: np.ndarray, encoded: dict[str, list[int]], tokenizer: dict[str, int],
                   ridge_means: dict[int, dict[str, float]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruye y guarda predicciones outer del checkpoint elegido por inner-val.

    Args:
        helper: Utilidades históricas.
        model_name: MatGPT/MatterGPT.
        modality: Modalidad de features.
        policy: F0/F1/F2.
        fold: Outer fold.
        seed: Semilla.
        candidate: Registro del mejor LR/continuación.
        data: Cohorte.
        groups: Tripletes.
        targets: Targets DFT.
        row_by_id: Índice IDs.
        embeddings: Caché original.
        geom_scaled: Geometría fold.
        encoded: SLICES tokenizadas.
        tokenizer: Vocabulario.
        ridge_means: Medias M2 por fold y fórmula.
        device: CPU.
    Returns:
        Filas de predicción estructural y pairwise.
    """
    saved = torch.load(ROOT / candidate["checkpoint"], map_location="cpu", weights_only=True)
    model = None
    if policy != "F0":
        model, _, _ = helper.load_model_base(model_name, helper.import_extractor())
        active = configure_policy(model, model_name, policy)
        stack = stack_for(model, model_name)
        for index, state in saved["blocks"].items():
            stack[int(index)].load_state_dict(state, strict=True)
        model.to(device).eval()
    head = MLP8(775 if modality == "SLICES_GEOMETRY" else 768).to(device)
    head.load_state_dict(saved["head"], strict=True)
    head.eval()
    _, _, outer = formulas_for_fold(helper.load_data()[1], fold)
    structure_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for formula in outer:
            ids = [str(data.iloc[index].structure_id) for index in groups[formula]]
            x = make_features(helper, model, model_name, ids, row_by_id, embeddings,
                              geom_scaled, modality, encoded if model is not None else None,
                              tokenizer if model is not None else None, device)
            # Promover el score antes de centrar evita que la suma de tres
            # residuos float32 exceda la tolerancia por redondeo (~1e-7 eV).
            raw = head(x).cpu().numpy().astype(np.float64)
            centered = raw - raw.mean(dtype=np.float64)
            mu = ridge_means[fold][formula]
            local: list[dict[str, Any]] = []
            for k, row_index in enumerate(groups[formula]):
                item = data.iloc[row_index]
                pred = mu + float(centered[k])
                record = {"condition": f"{model_name}_{modality}_{policy}", "model": model_name,
                    "modality": modality, "freeze_policy": policy, "fold": fold, "seed": seed,
                    "formula": formula, "structure_id": str(item.structure_id), "polymorph": str(item.family),
                    "E_DFT_eV_atom": float(item.energy_dft_ev_atom), "mu_M2_eV_atom": mu,
                    "raw_mlp_score": float(raw[k]), "centered_residual_eV_atom": float(centered[k]),
                    "E_pred_eV_atom": pred, "best_epoch": candidate["best_epoch"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"]}
                structure_rows.append(record)
                local.append(record)
            if abs(sum(row["centered_residual_eV_atom"] for row in local)) > 1e-7:
                raise ValueError(f"Residuo no centrado: {formula}")
            for i, j in PAIR_ORDER:
                truth = (local[j]["E_DFT_eV_atom"] - local[i]["E_DFT_eV_atom"]) * 1000
                prediction = (local[j]["E_pred_eV_atom"] - local[i]["E_pred_eV_atom"]) * 1000
                eligible = abs(truth) > 5
                unresolved = abs(prediction) <= 1
                pair_rows.append({"condition": f"{model_name}_{modality}_{policy}", "model": model_name,
                    "modality": modality, "freeze_policy": policy, "fold": fold, "seed": seed,
                    "formula": formula, "polymorph_i": local[i]["polymorph"],
                    "polymorph_j": local[j]["polymorph"], "structure_id_i": ids[i], "structure_id_j": ids[j],
                    "DeltaE_DFT_meV_atom": truth, "DeltaE_pred_meV_atom": prediction,
                    "DeltaE_abs_error_meV_atom": abs(prediction - truth),
                    "pairwise_eligible": eligible,
                    "pairwise_correct": bool(np.sign(prediction) == np.sign(truth) and not unresolved),
                    "unresolved": unresolved})
    del model
    return structure_rows, pair_rows


def run(args: argparse.Namespace) -> None:
    """Prepara datos, comparte warmups y ejecuta F0/F1/F2 con guardado incremental.

    Args:
        args: Parámetros CLI.
    """
    start_wall, start_cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")
    helper = load_helper()
    data, folds = helper.load_data()
    data = data.copy()
    data["_row_index"] = np.arange(len(data))
    groups, targets = helper.build_triplets(data)
    row_by_id = {str(row.structure_id): int(i) for i, row in data.iterrows()}
    all_ids = data.structure_id.astype(str).to_numpy()
    models_by_modality: dict[str, np.ndarray] = {}
    for model_name in MODELS:
        models_by_modality[model_name] = helper.load_embedding_cache(model_name, all_ids)
    ridge_models, ridge_means, ridge_hashes = helper.load_ridge_models(data)
    del ridge_models
    extractor = helper.import_extractor()
    token_data: dict[str, tuple[dict[str, list[int]], dict[str, int]]] = {}
    for model_name in MODELS:
        model, loaded, _ = helper.load_model_base(model_name, extractor)
        token_data[model_name] = (helper.encode_sequences(data, model_name, loaded["stoi"], loaded["context"]), loaded["stoi"])
        del model

    selected_models = MODELS if args.model == "all" else (args.model,)
    selected_modalities = MODALITIES if args.modality == "all" else (args.modality,)
    selected_policies = POLICIES
    atomic_json(EXP / "training_status.json", {"status": "RUNNING", "phase": args.phase,
        "started_unix": time.time(), "models": list(selected_models),
        "modalities": list(selected_modalities), "policies": list(selected_policies),
        "folds": list(range(5)) if args.phase == "full" else [0],
        "seeds": list(SEEDS) if args.phase == "full" else [42],
        "max_epochs_warmup": EPOCHS_WARMUP, "max_epochs_continuation": EPOCHS_CONTINUATION,
        "backbone_lr_grid": list(BACKBONE_LR_GRID), "device": "cpu",
        "torch_threads": THREADS, "mps_available": torch.backends.mps.is_available()})
    if args.phase == "pilot":
        selected_folds, selected_seeds = (0,), (42,)
    else:
        selected_folds, selected_seeds = tuple(range(5)), SEEDS
    records_path = EXP / f"run_records_{args.phase}.csv"
    structure_dir, pair_dir = EXP / "predictions" / "structures", EXP / "predictions" / "pairs"
    completed = set()
    completed_path = EXP / ("pilot_completed_runs.csv" if args.phase == "pilot" else "completed_runs.csv")
    if args.resume and completed_path.exists():
        with completed_path.open(newline="", encoding="utf-8") as stream:
            completed = {row["run_id"] for row in csv.DictReader(stream)}
    completed_rows: list[dict[str, str]] = []

    for model_name in selected_models:
        encoded, tokenizer = token_data[model_name]
        embeddings = models_by_modality[model_name]
        for modality in selected_modalities:
            for fold in selected_folds:
                train, valid, outer = formulas_for_fold(folds, fold)
                geom_scaler = StandardScaler().fit(data.loc[data.composition_id.isin(train), list(GEOMETRY)].to_numpy(float))
                geom_scaled = geom_scaler.transform(data[list(GEOMETRY)].to_numpy(float)).astype(np.float32)
                centered_train = np.concatenate([targets[f] - targets[f].mean() for f in train])
                scale = float(np.std(centered_train, ddof=0))
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Escala residual inválida fold={fold}")
                for seed in selected_seeds:
                    warm_id = f"{model_name}_{modality}_fold{fold}_seed{seed}"
                    warm_path = EXP / "warmup" / args.phase / model_name / modality / f"fold_{fold}_seed_{seed}.pt"
                    if warm_path.exists() and args.resume:
                        state = torch.load(warm_path, map_location=device, weights_only=True)
                        warm_head = MLP8(int(state["input_dim"])).to(device)
                        warm_head.load_state_dict(state["head"], strict=True)
                        warm_summary = {"path": str(warm_path.relative_to(ROOT)),
                                        "best_epoch": int(state["best_epoch"]),
                                        "best_validation_delta_mae_mev_atom": float(state["best_validation_delta_mae_mev_atom"]),
                                        "sha256": file_hash(warm_path)}
                    else:
                        warm_head, warm_summary = train_head_warmup(helper, model_name, modality,
                            fold, seed, data, groups, targets, row_by_id, embeddings, geom_scaled,
                            train, valid, scale, device, min(EPOCHS_WARMUP, args.max_epochs or EPOCHS_WARMUP), args.phase)
                    candidate_by_policy: dict[str, dict[str, Any]] = {}
                    for policy in selected_policies:
                        run_id = f"{model_name}_{modality}_{policy}_fold{fold}_seed{seed}"
                        if run_id in completed and args.phase != "pilot":
                            continue
                        lrs = (LR_HEAD,) if policy == "F0" else BACKBONE_LR_GRID
                        candidate_records: list[dict[str, Any]] = []
                        for lr in lrs:
                            record, epochs = fit_candidate(helper, model_name, modality, policy,
                                fold, seed, float(lr), warm_head, data, groups, targets, row_by_id,
                                embeddings, geom_scaled, train, valid, scale, device, encoded,
                                tokenizer, min(EPOCHS_CONTINUATION, args.max_epochs or EPOCHS_CONTINUATION), args.phase)
                            append_rows(EXP / f"candidate_records_{args.phase}.csv", [record])
                            candidate_records.append(record)
                        candidate_records.sort(key=lambda item: (-item["best_validation_pairwise_accuracy"],
                                                                  item["best_validation_delta_mae_mev_atom"],
                                                                  item["best_epoch"], item["backbone_lr"] or 0))
                        winner = min(candidate_records, key=lambda item: (item["best_validation_delta_mae_mev_atom"],
                            -item["best_validation_pairwise_accuracy"], item["best_epoch"], item["backbone_lr"] or 0))
                        candidate_by_policy[policy] = winner
                        run_record = {**winner, "run_id": run_id, "warmup_checkpoint": warm_summary["path"],
                            "execution_phase": args.phase,
                            "warmup_best_epoch": warm_summary["best_epoch"], "warmup_checkpoint_sha256": warm_summary["sha256"],
                            "geometry_scaler_mean": geom_scaler.mean_.tolist() if modality == "SLICES_GEOMETRY" else None,
                            "geometry_scaler_scale": geom_scaler.scale_.tolist() if modality == "SLICES_GEOMETRY" else None,
                            "residual_scale_ev_atom": scale, "ridge_sha256": ridge_hashes[fold],
                            "selected_by": "inner_validation_signed_pair_DeltaE_MAE"}
                        append_rows(records_path, [run_record])
                        if args.phase == "full":
                            struct_rows, pairs = evaluate_outer(helper, model_name, modality, policy,
                                fold, seed, winner, data, groups, targets, row_by_id, embeddings,
                                geom_scaled, encoded, tokenizer, ridge_means, device)
                            append_rows(structure_dir / f"{run_id}.csv", struct_rows)
                            append_rows(pair_dir / f"{run_id}.csv", pairs)
                        completed_rows.append({"run_id": run_id, "status": "COMPLETE"})
                        append_rows(completed_path, completed_rows[-1:])
                        atomic_json(EXP / "training_status.json", {"status": "RUNNING", "phase": args.phase,
                            "started_unix": time.time() - (time.monotonic() - start_wall),
                            "completed_final_runs": sum(1 for _ in completed_path.open(encoding="utf-8")) - 1,
                            "expected_final_runs": 180 if args.phase == "full" else 12,
                            "last_completed": run_id, "device": "cpu", "torch_threads": THREADS,
                            "max_epochs_continuation": EPOCHS_CONTINUATION,
                            "backbone_lr_grid": list(BACKBONE_LR_GRID)})
                    if args.phase == "pilot":
                        continue

    run_resources = {"phase": args.phase, "wall_seconds": time.monotonic() - start_wall,
        "process_cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
        "cpu_count_logical": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "mps_available": torch.backends.mps.is_available(),
        "historical_test_accessed": False}
    atomic_json(EXP / "resources_run.json", run_resources)
    atomic_json(EXP / "training_status.json", {"status": "COMPLETE", "phase": args.phase,
        "finished_unix": time.time(), "resources": run_resources})
    if args.phase == "pilot":
        atomic_json(EXP / "pilot_status.json", {"status": "COMPLETE", "scope": "one fold, seed 42, inner validation only",
            "models": list(selected_models), "modalities": list(selected_modalities),
            "outer_evaluation_accessed": False, "resources": run_resources})


def main() -> None:
    """Valida argumentos, fija recursos y ejecuta fase solicitada."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--modality", choices=("all", *MODALITIES), default="all")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override explícito para piloto técnico; no usar para resultados finales.")
    parser.add_argument("--resume", action="store_true", help="Omite corridas ya registradas COMPLETE.")
    args = parser.parse_args()
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs debe ser positivo")
    if args.phase == "full" and args.max_epochs is not None:
        parser.error("No se permite limitar épocas en fase full")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
