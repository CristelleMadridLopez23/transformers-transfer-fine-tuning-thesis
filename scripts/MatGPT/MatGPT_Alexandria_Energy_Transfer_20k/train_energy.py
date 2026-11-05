#!/usr/bin/env python3
"""Adapta MatGPT a energía cruda Alexandria mediante cabeza y bloques F4.

El programa consume únicamente los manifiestos preparados y selecciona el
checkpoint por SmoothL1 de validation Alexandria, sin usar etiquetas ABS3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ROOT = ROOT / "models/MatGPT-source/mat-concat"
CHECKPOINT = MODEL_ROOT / "mat-uncond/weights/transformer_property_0.pth"
MODEL_INFO = MODEL_ROOT / "mat-uncond/weights/model_info_property_0.json"
DATA_EXPERIMENT = ROOT / "experiments/matgpt_alexandria_energy_transfer_20k"
EXPERIMENT = DATA_EXPERIMENT
TRAIN_CSV = DATA_EXPERIMENT / "alexandria_20k_train.csv"
VALID_CSV = DATA_EXPERIMENT / "alexandria_2k_validation.csv"
LOG = logging.getLogger("matgpt_alexandria_training")
BASE_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"


class EnergyHead(nn.Module):
    """MLP8 energético fijo de MatGPT: 768→8→1."""

    def __init__(self) -> None:
        """Inicializa la cabeza con GELU y dropout 0.1."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, embedding: Tensor) -> Tensor:
        """Predice energía normalizada por átomo.

        Args:
            embedding: Embeddings FINAL_END con forma [lote,768].
        Returns:
            Energías normalizadas con forma [lote].
        """
        return self.network(embedding).squeeze(-1)


class _CpuLayerNormFunction(torch.autograd.Function):
    """Evaluate LayerNorm on CPU while preserving gradients to original-device parameters."""

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, weight: Tensor, bias: Tensor,
                normalized_shape: tuple[int, ...], eps: float) -> Tensor:
        """Run the standard LayerNorm equation and save CPU copies for its backward pass."""
        ctx.input_device = inputs.device
        ctx.weight_device = weight.device
        ctx.bias_device = bias.device
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        inputs_cpu, weight_cpu, bias_cpu = inputs.detach().cpu(), weight.detach().cpu(), bias.detach().cpu()
        ctx.save_for_backward(inputs_cpu, weight_cpu, bias_cpu)
        return F.layer_norm(inputs_cpu, normalized_shape, weight_cpu, bias_cpu, eps).to(inputs.device)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None]:
        """Use PyTorch's CPU LayerNorm backward, then return gradients to source devices."""
        inputs_cpu, weight_cpu, bias_cpu = ctx.saved_tensors
        with torch.enable_grad():
            inputs_var = inputs_cpu.detach().requires_grad_(True)
            weight_var = weight_cpu.detach().requires_grad_(True)
            bias_var = bias_cpu.detach().requires_grad_(True)
            outputs = F.layer_norm(inputs_var, ctx.normalized_shape, weight_var, bias_var, ctx.eps)
            grad_inputs, grad_weight, grad_bias = torch.autograd.grad(
                outputs, (inputs_var, weight_var, bias_var), grad_output.detach().cpu(),
                retain_graph=False, create_graph=False
            )
        return (grad_inputs.to(ctx.input_device), grad_weight.to(ctx.weight_device),
                grad_bias.to(ctx.bias_device), None, None)


class CpuBackwardLayerNorm(nn.Module):
    """LayerNorm with unchanged parameters and equation, using CPU forward/backward kernels."""

    def __init__(self, original: nn.LayerNorm) -> None:
        """Wrap an existing affine LayerNorm without changing its learned parameters."""
        super().__init__()
        if original.weight is None or original.bias is None:
            raise ValueError("F4 CPU LayerNorm fallback requires affine weight and bias")
        self.weight = original.weight
        self.bias = original.bias
        self.normalized_shape = tuple(original.normalized_shape)
        self.eps = float(original.eps)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the mathematically identical normalization through the CPU autograd function."""
        return _CpuLayerNormFunction.apply(inputs, self.weight, self.bias, self.normalized_shape, self.eps)


def install_mps_layernorm_fallback(model: Any) -> None:
    """Move only final-block norm1 computation to CPU to avoid the observed MPS dbias NaN.

    Args:
        model: Loaded MatGPT backbone already moved to MPS.
    Returns:
        None.
    """
    for block in model.transformer_decoder.layers[8:12]:
        block.norm1 = CpuBackwardLayerNorm(block.norm1)


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo local.

    Args:
        path: Archivo a verificar.
    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Escribe JSON sin dejar un archivo parcialmente escrito.

    Args:
        path: Destino.
        value: Diccionario serializable.
    Returns:
        None.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def peak_rss_bytes() -> int:
    """Devuelve RSS máximo observado en bytes.

    Args:
        None.
    Returns:
        Pico RSS en bytes.
    """
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_matgpt() -> tuple[Any, dict[str, int], int, Any]:
    """Carga estricta del checkpoint MatGPT original congelado inicialmente.

    Args:
        None.
    Returns:
        Modelo, vocabulario, contexto y módulo torch.
    """
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import load_model
    info_path = MODEL_ROOT / "mat-uncond/weights/model_info_property_0.json"
    return load_model(MODEL_ROOT, info_path, CHECKPOINT, MODEL_ROOT / "config/uncond.yaml")


def encode_rows(frame: pd.DataFrame, stoi: dict[str, int], block_size: int) -> list[list[int]]:
    """Codifica todas las filas con el tokenizer SLICES histórico.

    Args:
        frame: Manifiesto del split.
        stoi: Vocabulario exacto del checkpoint.
        block_size: Contexto máximo.
    Returns:
        Una lista de input IDs por estructura.
    """
    from Alternative2.scripts.MatGPT.MatGPT_frozen.matgpt_e1.extract_embeddings import encode_sequence
    encoded = [encode_sequence(str(value), stoi, block_size) for value in frame["slices_matgpt"]]
    if any(len(sequence) > block_size for sequence in encoded):
        raise ValueError("Manifiesto contiene truncamiento/contexto excedido")
    return encoded


def make_batch(sequences: list[list[int]], indices: np.ndarray, pad_id: int) -> tuple[Tensor, Tensor]:
    """Construye un minibatch con padding dinámico a la secuencia más larga.

    Args:
        sequences: Input IDs de todas las estructuras.
        indices: Índices de filas para el minibatch.
        pad_id: ID original del token de padding.
    Returns:
        Tensor input IDs y posiciones de token final.
    """
    selected = [sequences[int(index)] for index in indices]
    width = max(len(sequence) for sequence in selected)
    ids = torch.full((len(selected), width), pad_id, dtype=torch.long)
    for row, sequence in enumerate(selected):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return ids, torch.tensor([len(sequence) - 1 for sequence in selected], dtype=torch.long)


def final_end(model: Any, ids: Tensor, positions: Tensor) -> Tensor:
    """Obtiene el estado causal contextualizado de <end> después de ln_f.

    Args:
        model: MatGPT.
        ids: Input IDs rellenados a la derecha.
        positions: Posición real de <end> por fila.
    Returns:
        Tensor FINAL_END con forma [lote,768].
    """
    length, batch = ids.shape[1], ids.shape[0]
    device = ids.device
    hidden = model.token_embedding_table(ids) + model.position_embedding_table(torch.arange(length, device=device))
    hidden = hidden.permute(1, 0, 2)
    mask = model.generate_square_subsequent_mask(length).to(device)
    memory = torch.zeros((length, batch, model.n_embd), device=device)
    hidden = model.transformer_decoder(hidden, memory, tgt_mask=mask).permute(1, 0, 2)
    hidden = model.ln_f(hidden)
    row_ids = torch.arange(batch, device=device)
    return hidden[row_ids, positions.to(device)]


def extract_frozen_embeddings(model: Any, sequences: list[list[int]], pad_id: int,
                              batch_size: int, device: torch.device) -> np.ndarray:
    """Extrae embeddings congelados para el warmup económico de la cabeza.

    Args:
        model: MatGPT congelado.
        sequences: Input IDs por material.
        pad_id: Token de padding.
        batch_size: Tamaño de minibatch.
        device: Dispositivo.
    Returns:
        Matriz float32 [N,768].
    """
    result: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(sequences), batch_size):
            batch_indices = np.arange(begin, min(begin + batch_size, len(sequences)))
            ids, positions = make_batch(sequences, batch_indices, pad_id)
            vectors = final_end(model, ids.to(device), positions.to(device))
            result.extend(vectors.detach().cpu().numpy().astype(np.float32))
    return np.stack(result)


def predict_epoch(model: Any, head: EnergyHead, sequences: list[list[int]], targets: np.ndarray,
                  pad_id: int, batch_size: int, device: torch.device) -> tuple[float, np.ndarray]:
    """Calcula SmoothL1 completo y predicciones normalizadas sin gradientes.

    Args:
        model: MatGPT.
        head: Regresor energético.
        sequences: Input IDs.
        targets: Targets normalizados.
        pad_id: ID de padding.
        batch_size: Tamaño de minibatch.
        device: Dispositivo.
    Returns:
        SmoothL1 promedio por registro y vector de predicciones.
    """
    model.eval()
    head.eval()
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    with torch.inference_mode():
        for begin in range(0, len(sequences), batch_size):
            indices = np.arange(begin, min(begin + batch_size, len(sequences)))
            ids, positions = make_batch(sequences, indices, pad_id)
            pred = head(final_end(model, ids.to(device), positions.to(device)))
            target = torch.as_tensor(targets[indices], dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(pred, target, beta=1.0, reduction="none")
            losses.extend(loss.cpu().numpy().tolist())
            predictions.extend(pred.cpu().numpy().tolist())
    return float(np.mean(losses)), np.asarray(predictions, dtype=np.float64)


def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE, bias y R² para energía por átomo.

    Args:
        actual: Targets eV/átomo.
        predicted: Predicciones eV/átomo.
    Returns:
        Métricas físicas.
    """
    residual = predicted - actual
    ss_total = float(np.square(actual - actual.mean()).sum())
    return {"mae_ev_atom": float(np.abs(residual).mean()),
            "mae_mev_atom": float(np.abs(residual).mean() * 1000),
            "rmse_ev_atom": float(np.sqrt(np.square(residual).mean())),
            "bias_ev_atom": float(residual.mean()),
            "r2_predictive": float(1 - np.square(residual).sum() / ss_total) if ss_total > 0 else float("nan")}


def train(args: argparse.Namespace) -> None:
    """Ejecuta warmup y fine-tuning F4 guardando recursos por época."""
    if not TRAIN_CSV.is_file() or not VALID_CSV.is_file():
        raise FileNotFoundError("Faltan manifests; ejecuta prepare_provisional.py --run primero")
    if sha256(CHECKPOINT) != BASE_SHA:
        raise ValueError("El checkpoint original no coincide con el SHA-256 esperado")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    prior_resources_path = EXPERIMENT / "resources.json"
    prior_resources = json.loads(prior_resources_path.read_text(encoding="utf-8")) if prior_resources_path.is_file() else {}
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    train_frame, val_frame = pd.read_csv(TRAIN_CSV), pd.read_csv(VALID_CSV)
    if len(train_frame) != 20000 or len(val_frame) != 2000:
        raise ValueError("Los tamaños de los manifests no son 20k/2k")
    if set(train_frame.reduced_formula) & set(val_frame.reduced_formula):
        raise ValueError("Fuga de fórmula entre train y validation")
    LOG.info(json.dumps({"event": "training_started", "seed": args.seed,
                         "train_records": len(train_frame), "validation_records": len(val_frame),
                         "threads": args.threads, "batch_size": args.batch_size,
                         "max_finetune_epochs": args.max_epochs, "patience": args.patience,
                         "base_checkpoint_sha256": BASE_SHA,
                         "train_manifest_sha256": sha256(TRAIN_CSV),
                         "validation_manifest_sha256": sha256(VALID_CSV)}, sort_keys=True))
    target_train = train_frame.energy_per_atom_ev.to_numpy(dtype=np.float64)
    target_val = val_frame.energy_per_atom_ev.to_numpy(dtype=np.float64)
    target_mean, target_std = float(target_train.mean()), float(target_train.std(ddof=0))
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Desviación del target train no válida")
    y_train = ((target_train - target_mean) / target_std).astype(np.float32)
    y_val = ((target_val - target_mean) / target_std).astype(np.float32)
    model, stoi, context, torch_module = load_matgpt()
    if context != 442 or torch_module is not torch:
        raise ValueError("Checkpoint/tokenizer MatGPT no coincide con preflight")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA pero no hay una GPU visible para PyTorch")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Se solicitó MPS pero Metal no está disponible para PyTorch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    smoke = {"batch_size": 8, "padded_length": 230, "lengths": [61, 66, 82, 90, 113, 135, 173, 230],
             "threads_2_wall_seconds": 2.0841486249119043, "threads_2_cpu_seconds": 3.080585,
             "threads_2_peak_rss_bytes": 2332229632, "threads_8_wall_seconds": 1.9585209162905812,
             "threads_8_cpu_seconds": 4.066702, "threads_8_peak_rss_bytes": 2319810560,
             "blocks_8_11_gradient_nonzero": True, "loss_finite": True,
             "rough_finetune_train_only_days_at_100_epochs_threads2": 5.7}
    atomic_json(EXPERIMENT / "resources.json", {"status": "RUNNING", "device": str(device),
        "cpu_count": os.cpu_count(), "threads": args.threads, "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size, "seed": args.seed,
        "base_checkpoint_sha256": BASE_SHA, "started_wall_time_unix": time.time(),
        "training_epochs_max": {"warmup": 20, "fine_tune": args.max_epochs},
        "preparation_run": prior_resources, "smoke_benchmark": smoke,
        "peak_rss_bytes_so_far": peak_rss_bytes()})
    initial_summary = json.loads((EXPERIMENT / "summary.json").read_text(encoding="utf-8"))
    initial_summary.update({"status": "TRAINING", "training_executed": True,
                            "execution_device": str(device), "current_attempt_started_unix": time.time()})
    atomic_json(EXPERIMENT / "summary.json", initial_summary)
    model.to(device)
    train_sequences = encode_rows(train_frame, stoi, context)
    val_sequences = encode_rows(val_frame, stoi, context)
    if sum(parameter.numel() for parameter in model.parameters()) != 114026668:
        raise ValueError("Recuento de parámetros distinto al MatGPT verificado")
    pad_id = stoi["<pad>"]
    head = EnergyHead().to(device)
    trainable_params_head = sum(parameter.numel() for parameter in head.parameters())
    if trainable_params_head != 6161:
        raise ValueError(f"MLP8 cambió de tamaño: {trainable_params_head}")

    # Stage A: frozen MatGPT features and only the MLP8 are updated.
    warm_start = time.monotonic()
    LOG.info(json.dumps({"event": "frozen_feature_extraction_started",
                         "train_records": len(train_sequences), "validation_records": len(val_sequences)}, sort_keys=True))
    train_embeddings = extract_frozen_embeddings(model, train_sequences, pad_id, args.batch_size, device)
    val_embeddings = extract_frozen_embeddings(model, val_sequences, pad_id, args.batch_size, device)
    warm_feature_seconds = time.monotonic() - warm_start
    LOG.info(json.dumps({"event": "frozen_feature_extraction_completed",
                         "wall_seconds": warm_feature_seconds, "peak_rss_bytes": peak_rss_bytes()}, sort_keys=True))
    warm_optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    history_path = EXPERIMENT / "historial_entrenamiento.csv"
    for epoch in range(1, 21):
        epoch_start = time.monotonic()
        head.train()
        order = rng.permutation(len(train_sequences))
        losses: list[float] = []
        for begin in range(0, len(order), args.batch_size):
            indices = order[begin:begin + args.batch_size]
            x = torch.from_numpy(train_embeddings[indices]).to(device)
            y = torch.from_numpy(y_train[indices]).to(device)
            warm_optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(head(x), y, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Warmup loss no finita, época {epoch}")
            loss.backward()
            warm_optimizer.step()
            losses.append(float(loss.detach().item()))
        head.eval()
        with torch.inference_mode():
            warm_val_pred = head(torch.from_numpy(val_embeddings).to(device)).cpu().numpy()
        val_loss = float(F.smooth_l1_loss(torch.from_numpy(warm_val_pred.astype(np.float32)),
                                          torch.from_numpy(y_val), beta=1.0).item())
        epoch_seconds = time.monotonic() - epoch_start
        history.append({"stage": "warmup_frozen", "epoch": epoch, "train_smoothl1": float(np.mean(losses)),
                        "validation_smoothl1": val_loss, "epoch_wall_seconds": epoch_seconds,
                        "optimizer_steps": int(np.ceil(len(order) / args.batch_size)), "l2_sp": 0.0})
        pd.DataFrame(history).to_csv(history_path, index=False)
        LOG.info(json.dumps({"event": "epoch_completed", "stage": "warmup_frozen", "epoch": epoch,
                             "train_smoothl1": float(np.mean(losses)), "validation_smoothl1": val_loss,
                             "epoch_wall_seconds": epoch_seconds,
                             "optimizer_steps": int(np.ceil(len(order) / args.batch_size))}, sort_keys=True))
    del train_embeddings, val_embeddings

    if device.type == "mps":
        # Preserve the original LayerNorm weights and equation, but avoid the MPS
        # backward kernel that produced a non-finite norm1.bias gradient in F4.
        install_mps_layernorm_fallback(model)

    # Stage B: unfreeze only original blocks 8–11; embeddings 0–7, ln_f and LM head stay fixed.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.transformer_decoder.layers[8:12]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    if any(parameter.requires_grad for parameter in model.token_embedding_table.parameters()):
        raise ValueError("Token embeddings se descongelaron inesperadamente")
    head.train()
    transformer_params = [parameter for block in model.transformer_decoder.layers[8:12]
                          for parameter in block.parameters() if parameter.requires_grad]
    if not transformer_params or not all(parameter.requires_grad for parameter in transformer_params):
        raise ValueError("F4 no tiene los cuatro bloques esperados entrenables")
    optimizer = torch.optim.AdamW([
        {"params": transformer_params, "lr": 3e-6, "weight_decay": 0.01},
        {"params": list(head.parameters()), "lr": 1e-4, "weight_decay": 1e-4}],
    )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    ckpt_dir = EXPERIMENT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "matgpt_alexandria_f4_energy_best.pt"
    summary = json.loads((EXPERIMENT / "summary.json").read_text(encoding="utf-8"))
    summary.update({"status": "TRAINING", "training_executed": True, "training_config": {"warmup_epochs": 20,
        "fine_tuning_epochs_max": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size,
        "threads": args.threads, "unfrozen_blocks": [8, 9, 10, 11], "device": str(device)}})
    atomic_json(EXPERIMENT / "summary.json", summary)
    resource_record: dict[str, Any] = {"status": "RUNNING", "device": str(device), "threads": args.threads,
        "batch_size": args.batch_size, "warmup_epochs": 20, "finetune_epochs_max": 100,
        "base_checkpoint_sha256": BASE_SHA, "warmup_feature_extraction_seconds": warm_feature_seconds,
        "mps_layernorm_cpu_fallback_blocks": [8, 9, 10, 11] if device.type == "mps" else [],
        "preparation_run": prior_resources, "smoke_benchmark": smoke,
        "parameters_trainable": {"head": trainable_params_head,
        "blocks_8_11": int(sum(parameter.numel() for parameter in transformer_params))}}
    atomic_json(EXPERIMENT / "resources.json", resource_record)
    LOG.info(json.dumps({"event": "finetune_f4_started", "unfrozen_blocks": [8, 9, 10, 11],
                         "trainable_head_parameters": trainable_params_head,
                         "trainable_backbone_parameters": resource_record["parameters_trainable"]["blocks_8_11"],
                         "maximum_epochs": args.max_epochs, "patience": args.patience,
                         "mps_layernorm_cpu_fallback_blocks": resource_record["mps_layernorm_cpu_fallback_blocks"]}, sort_keys=True))
    try:
        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            head.train()
            order = rng.permutation(len(train_sequences))
            epoch_loss_sum, seen, grad_sum = 0.0, 0, 0.0
            for begin in range(0, len(order), args.batch_size):
                indices = order[begin:begin + args.batch_size]
                ids, positions = make_batch(train_sequences, indices, pad_id)
                ids, positions = ids.to(device), positions.to(device)
                target = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                pred = head(final_end(model, ids, positions))
                loss = F.smooth_l1_loss(pred, target, beta=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Fine-tune loss no finita en época {epoch}")
                loss.backward()
                named_trainable = [(name, parameter) for name, parameter in model.named_parameters()
                                   if parameter.requires_grad]
                named_trainable.extend((f"head.{name}", parameter) for name, parameter in head.named_parameters()
                                       if parameter.requires_grad)
                named_gradients = [(name, parameter.grad.detach()) for name, parameter in named_trainable
                                   if parameter.grad is not None]
                finite_flags = torch.stack([torch.isfinite(gradient).all()
                                            for _, gradient in named_gradients])
                if not bool(finite_flags.all()):
                    bad_names = [name for name, gradient in named_gradients
                                 if not bool(torch.isfinite(gradient).all())]
                    raise FloatingPointError(
                        f"Gradientes no finitos en época {epoch}, lote {begin // args.batch_size}: {bad_names}"
                    )
                # Compute the diagnostic norm over one contiguous vector. This avoids
                # backend-specific accumulation behavior when summing dozens of MPS norms.
                flat_gradient = torch.cat([gradient.reshape(-1) for _, gradient in named_gradients])
                grad_norm = torch.linalg.vector_norm(flat_gradient)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError(
                        f"Norma global no finita aunque cada gradiente es finito; época {epoch}, "
                        f"lote {begin // args.batch_size}"
                    )
                grad_sum += float(grad_norm.detach().item())
                optimizer.step()
                batch_size = len(indices)
                epoch_loss_sum += float(loss.detach().item()) * batch_size
                seen += batch_size
            train_loss = epoch_loss_sum / seen
            val_loss, val_prediction_norm = predict_epoch(model, head, val_sequences, y_val, pad_id,
                                                           args.validation_batch_size, device)
            epoch_seconds = time.monotonic() - epoch_start
            history.append({"stage": "finetune_f4", "epoch": epoch, "train_smoothl1": train_loss,
                            "validation_smoothl1": val_loss, "epoch_wall_seconds": epoch_seconds,
                            "optimizer_steps": int(np.ceil(len(order) / args.batch_size)),
                            "gradient_norm_mean": grad_sum / max(1, int(np.ceil(len(order) / args.batch_size))),
                            "l2_sp": 0.0})
            pd.DataFrame(history).to_csv(history_path, index=False)
            if val_loss < best_val:
                best_val, best_epoch, stale = val_loss, epoch, 0
                state = {"base_checkpoint_sha256": BASE_SHA, "tokenizer_sha256": sha256(MODEL_INFO),
                         "selected_epoch": epoch, "validation_smoothl1": val_loss,
                         "target_mean_ev_atom": target_mean, "target_std_ev_atom": target_std,
                         "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                         "blocks_8_11": {name: value.detach().cpu() for name, value in model.state_dict().items()
                                         if any(name.startswith(f"transformer_decoder.layers.{idx}.") for idx in range(8, 12))}}
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(state, temporary)
                os.replace(temporary, best_path)
            else:
                stale += 1
            resource_record.update({"status": "RUNNING", "completed_finetune_epochs": epoch,
                                    "best_epoch": best_epoch, "best_validation_smoothl1": best_val,
                                    "last_epoch_wall_seconds": epoch_seconds,
                                    "total_wall_seconds": time.monotonic() - start_wall,
                                    "total_cpu_seconds": time.process_time() - start_cpu,
                                    "peak_rss_bytes": peak_rss_bytes(),
                                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                                    "optimizer_steps": int(np.ceil(len(train_sequences) / args.batch_size) * (20 + epoch))})
            atomic_json(EXPERIMENT / "resources.json", resource_record)
            LOG.info(json.dumps({"event": "epoch_completed", "stage": "finetune_f4", "epoch": epoch,
                                 "train_smoothl1": train_loss, "validation_smoothl1": val_loss,
                                 "best_epoch": best_epoch, "epoch_seconds": epoch_seconds,
                                 "stale_epochs": stale, "gradient_norm_mean": grad_sum / max(1, int(np.ceil(len(order) / args.batch_size))),
                                 "optimizer_steps": int(np.ceil(len(order) / args.batch_size)),
                                 "peak_rss_bytes": peak_rss_bytes()}, sort_keys=True))
            if stale >= args.patience:
                break
    except KeyboardInterrupt:
        resource_record.update({"status": "PARTIAL_INTERRUPTED", "total_wall_seconds": time.monotonic() - start_wall,
                                "total_cpu_seconds": time.process_time() - start_cpu, "peak_rss_bytes": peak_rss_bytes(),
                                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0})
        atomic_json(EXPERIMENT / "resources.json", resource_record)
        summary = json.loads((EXPERIMENT / "summary.json").read_text(encoding="utf-8"))
        summary.update({"status": "PARTIAL_INTERRUPTED", "training_executed": True,
                        "best_epoch": best_epoch, "best_validation_smoothl1": best_val})
        atomic_json(EXPERIMENT / "summary.json", summary)
        raise
    except Exception as exc:
        resource_record.update({"status": "FAILED", "failure_type": type(exc).__name__,
                                "failure_message": str(exc), "failed_epoch": locals().get("epoch"),
                                "failed_batch_start": locals().get("begin"),
                                "total_wall_seconds": time.monotonic() - start_wall,
                                "total_cpu_seconds": time.process_time() - start_cpu,
                                "peak_rss_bytes": peak_rss_bytes()})
        atomic_json(EXPERIMENT / "resources.json", resource_record)
        summary = json.loads((EXPERIMENT / "summary.json").read_text(encoding="utf-8"))
        summary.update({"status": "FAILED", "failure_type": type(exc).__name__,
                        "failure_message": str(exc), "failed_epoch": locals().get("epoch"),
                        "failed_batch_start": locals().get("begin")})
        atomic_json(EXPERIMENT / "summary.json", summary)
        raise

    if best_epoch == 0 or not best_path.is_file():
        raise RuntimeError("No se guardó checkpoint seleccionado por validation")
    # Restore selected F4/head state and evaluate independent Alexandria validation once.
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["blocks_8_11"], strict=False)
    head.load_state_dict(best["head_state_dict"], strict=True)
    val_prediction_norm = predict_epoch(model, head, val_sequences, y_val, pad_id,
                                        args.validation_batch_size, device)[1]
    val_prediction = val_prediction_norm * target_std + target_mean
    metrics = compute_regression_metrics(target_val, val_prediction)
    # Composition-only baseline: standardized elemental fractions and fixed Ridge alpha=1.
    from pymatgen.core import Composition
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    all_elements = sorted({el.symbol for formula in train_frame.reduced_formula
                           for el in Composition(str(formula)).elements})
    element_index = {element: index for index, element in enumerate(all_elements)}
    def composition_matrix(values: pd.Series) -> np.ndarray:
        matrix = np.zeros((len(values), len(all_elements)), dtype=np.float64)
        for row, formula in enumerate(values.astype(str)):
            comp = Composition(formula).fractional_composition
            for element, amount in comp.get_el_amt_dict().items():
                if element in element_index:
                    matrix[row, element_index[element]] = float(amount)
        return matrix
    x_train, x_val = composition_matrix(train_frame.reduced_formula), composition_matrix(val_frame.reduced_formula)
    scaler = StandardScaler().fit(x_train)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), target_train)
    baseline_metrics = compute_regression_metrics(target_val, ridge.predict(scaler.transform(x_val)))
    metrics_payload = json.loads((EXPERIMENT / "metrics.json").read_text(encoding="utf-8"))
    metrics_payload["training"] = {"status": "COMPLETED", "selected_epoch": best_epoch,
        "best_validation_smoothl1_normalized": best_val, "prediction_metrics": metrics,
        "composition_ridge_alpha_1_baseline": baseline_metrics,
        "target_normalization": {"mean_ev_atom_train": target_mean, "std_ev_atom_train": target_std},
        "validation_rows": len(val_frame), "prediction_unit": "eV/atom",
        "checkpoint_sha256": sha256(best_path), "checkpoint_path": str(best_path.relative_to(ROOT))}
    atomic_json(EXPERIMENT / "metrics.json", metrics_payload)
    summary = json.loads((EXPERIMENT / "summary.json").read_text(encoding="utf-8"))
    summary.update({"status": "COMPLETED", "training_executed": True, "selected_epoch": best_epoch,
                   "best_validation_smoothl1_normalized": best_val, "validation_metrics": metrics,
                   "composition_ridge_alpha_1_baseline": baseline_metrics,
                   "checkpoint": str(best_path.relative_to(ROOT)), "checkpoint_sha256": sha256(best_path),
                   "target_mean_ev_atom_train": target_mean, "target_std_ev_atom_train": target_std})
    atomic_json(EXPERIMENT / "summary.json", summary)
    resource_record.update({"status": "COMPLETED", "best_epoch": best_epoch,
                            "total_wall_seconds": time.monotonic() - start_wall,
                            "total_cpu_seconds": time.process_time() - start_cpu,
                            "peak_rss_bytes": peak_rss_bytes(),
                            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                            "completed_finetune_epochs": len([x for x in history if x["stage"] == "finetune_f4"]),
                            "optimizer_steps": sum(int(row.get("optimizer_steps", 0)) for row in history),
                            "checkpoint_sha256": sha256(best_path)})
    atomic_json(EXPERIMENT / "resources.json", resource_record)
    LOG.info(json.dumps({"event": "training_complete", "best_epoch": best_epoch,
                         "validation_metrics": metrics,
                         "wall_seconds": resource_record["total_wall_seconds"]}, sort_keys=True))


def main() -> None:
    """Valida argumentos, prepara log y ejecuta fine-tuning energético."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Dispositivo de entrenamiento; auto prioriza CUDA, luego Metal y CPU")
    parser.add_argument("--data-dir", type=Path, default=DATA_EXPERIMENT,
                        help="Directorio con manifiestos de entrenamiento y validación")
    parser.add_argument("--output-dir", type=Path, default=DATA_EXPERIMENT,
                        help="Directorio de resultados; permite preservar corridas previas")
    args = parser.parse_args()
    if min(args.batch_size, args.validation_batch_size, args.threads, args.max_epochs, args.patience) <= 0:
        parser.error("Todos los tamaños y límites deben ser positivos")
    global EXPERIMENT, TRAIN_CSV, VALID_CSV
    data_dir = args.data_dir.resolve()
    EXPERIMENT = args.output_dir.resolve()
    TRAIN_CSV = data_dir / "alexandria_20k_train.csv"
    VALID_CSV = data_dir / "alexandria_2k_validation.csv"
    if not TRAIN_CSV.is_file() or not VALID_CSV.is_file():
        parser.error(f"No encuentro los manifiestos train/validation en {data_dir}")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if EXPERIMENT != data_dir:
        for filename in ("config.yaml", "metrics.json", "summary.json", "resources.json"):
            source, destination = data_dir / filename, EXPERIMENT / filename
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(EXPERIMENT / "training.log", encoding="utf-8")])
    train(args)


if __name__ == "__main__":
    main()
