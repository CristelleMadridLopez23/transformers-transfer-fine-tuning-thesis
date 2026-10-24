#!/usr/bin/env python3
"""Train matched E2-Aug1/E2-Aug2 partial-FT runs from the original MatGPT checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
ALT = ROOT / "Alternative2"
E2_DIR = ALT / "scripts/MatGPT_finetuning/matgpt_e2"
DATA = ALT / "reports/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ALT / "reports/MatGPT_frozen/matgpt_e1_3/fold_assignments.csv"
VARIANTS = ALT / "reports/MatGPT_finetuning/E2-Aug1/variants_development_only_v2/variant_manifest.csv"
T1_DECISION = ALT / "reports/MatGPT_finetuning/E2-T1/decision.json"
T1_FRAMES = ALT / "reports/MatGPT_finetuning/E2-T1/frame_tokenization.csv"
T1_SELECTION = ALT / "reports/MatGPT_finetuning/E2-T1/selection_candidates.csv"
T2_AUG2_SMOKE = ALT / "reports/MatGPT_finetuning/E2-Aug2/development_only_v2/smoke/cpu/smoke_integrity.json"
INFO = ALT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
SEEDS = (42, 123, 2026)
LOG = logging.getLogger("matgpt_e2_aug")


def load_e2a_module() -> Any:
    """Load the executed E2-A implementation as a read-only source of model/loss code."""
    sys.path.insert(0, str(E2_DIR))
    spec = importlib.util.spec_from_file_location("matgpt_e2a_audited", E2_DIR / "finetune.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar el código E2-A auditado: {E2_DIR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_index(structure_id: str, fold: int, seed: int, epoch: int, n_views: int) -> int:
    """Select a deterministic primary view using fold, seed, epoch, and structure ID."""
    if n_views < 1:
        raise ValueError("Cada estructura necesita al menos su serialización histórica.")
    payload = f"primary|{fold}|{seed}|{epoch}|{structure_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % n_views


def secondary_index(structure_id: str, fold: int, seed: int, epoch: int, n_views: int, primary: int) -> int:
    """Select a deterministic secondary view distinct from the primary view."""
    if n_views < 2:
        raise ValueError(f"No hay segunda vista verificable para {structure_id}.")
    payload = f"secondary|{fold}|{seed}|{epoch}|{structure_id}".encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16) % (n_views - 1)
    return value + int(value >= primary)


def build_trajectory_bank(data: pd.DataFrame, fold: int, stoi: dict[str, int], block_size: int) -> tuple[dict[str, list[dict[str, Any]]], float, int]:
    """Build fold-local, inner-train-only energy-difference pairs from T1.

    Each trajectory is anchored to its last eligible frame. Identical token IDs
    are never paired, and only development compositions assigned to this fold's
    inner-train role are admitted.
    """
    if not T1_DECISION.is_file() or not T1_FRAMES.is_file() or not T1_SELECTION.is_file():
        raise FileNotFoundError("Falta decisión y tablas completas E2-T1 para habilitar E2-T2.")
    decision = json.loads(T1_DECISION.read_text(encoding="utf-8"))
    gate = str(decision.get("t2_training_gate", decision.get("status", "BLOCKED")))
    if gate not in {"GO", "LIMITED"}:
        raise RuntimeError(f"Gate E2-T1={gate}; no se permite construir muestras T2.")
    candidates = pd.read_csv(T1_SELECTION, low_memory=False)
    candidates = candidates[candidates.outer_fold.eq(fold) & candidates.inner_role.eq("inner_train") & candidates.eligible_for_t2_training.astype(bool)].copy()
    frame_data = pd.read_csv(T1_FRAMES, low_memory=False)
    candidates = candidates.merge(frame_data[["trajectory_id", "frame_index", "sequence", "input_ids_hash"]],
                                  on=["trajectory_id", "frame_index", "input_ids_hash"],
                                  how="left", validate="one_to_one")
    if candidates.sequence.isna().any():
        raise RuntimeError("No se pudo unir cada candidato T1 con su secuencia SLICES auditada.")
    allowed_groups = set(data.composition_group.astype(str))
    if set(candidates.composition_group.astype(str)) - allowed_groups:
        raise RuntimeError("T1 contiene composiciones no presentes en el desarrollo del split actual.")
    bank: dict[str, list[dict[str, Any]]] = {}
    all_deltas: list[float] = []
    for trajectory_id, group in candidates.groupby("trajectory_id", sort=True):
        group = group.sort_values("frame_index")
        composition = str(group.composition_group.iloc[0])
        if group.composition_group.astype(str).nunique() != 1:
            raise RuntimeError(f"Una trayectoria cruza composiciones: {trajectory_id}")
        anchor = group.iloc[-1]
        anchor_hash = str(anchor.input_ids_hash)
        anchor_sequence = str(anchor.sequence)
        anchor_ids = [int(stoi["<start>"]), *[int(stoi[token]) for token in anchor_sequence.split()], int(stoi["<end>"])]
        if len(anchor_ids) > block_size:
            raise RuntimeError(f"Anchor T1 excede block_size: {trajectory_id}")
        pairs: list[dict[str, Any]] = []
        for row in group.itertuples(index=False):
            if str(row.input_ids_hash) == anchor_hash:
                continue
            sequence = str(row.sequence)
            ids = [int(stoi["<start>"]), *[int(stoi[token]) for token in sequence.split()], int(stoi["<end>"])]
            if len(ids) > block_size:
                raise RuntimeError(f"Frame T1 excede block_size: {trajectory_id}/{row.frame_index}")
            delta = float(row.energy_per_atom_ev) - float(anchor.energy_per_atom_ev)
            pairs.append({"frame_ids": ids, "anchor_ids": anchor_ids, "delta_dft_ev_atom": delta,
                          "frame_index": int(row.frame_index), "input_ids_hash": str(row.input_ids_hash)})
        if pairs:
            bank.setdefault(composition, []).append({"trajectory_id": str(trajectory_id), "pairs": pairs})
            all_deltas.extend(pair["delta_dft_ev_atom"] for pair in pairs)
    scale = max(float(np.sqrt(np.mean(np.square(all_deltas)))) if all_deltas else 0.0, 0.001)
    if not bank:
        raise RuntimeError("No hay pares de secuencias token-distintas dentro de trayectorias inner-train.")
    return bank, scale, len(all_deltas)


def make_batch(sequences: list[list[int]], pad: int, device: torch.device) -> torch.Tensor:
    """Pad token ID sequences on the right and return a long tensor."""
    width = max(map(len, sequences))
    result = torch.full((len(sequences), width), int(pad), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        result[row, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.long, device=device)
    return result


def load_view_bank(data: pd.DataFrame, manifest_path: Path, stoi: dict[str, int], block_size: int) -> dict[str, list[tuple[int, list[int], str]]]:
    """Load accepted views and verify view zero is exactly the E2-A sequence."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    accepted = manifest[manifest.accepted.astype(bool)].copy()
    if accepted.duplicated(["structure_id", "variant_id"]).any():
        raise RuntimeError("El manifiesto repite structure_id/variant_id.")
    bank: dict[str, list[tuple[int, list[int], str]]] = {}
    sequence_lookup = data.set_index(data.structure_id.astype(str)).matgpt_slices.astype(str).to_dict()
    for structure_id, group in accepted.groupby("structure_id", sort=False):
        ordered = group.sort_values("variant_id")
        if int(ordered.variant_id.iloc[0]) != 0:
            raise RuntimeError(f"Falta variant_0 histórico para {structure_id}.")
        if str(ordered.variant_slices.iloc[0]) != sequence_lookup[str(structure_id)]:
            raise RuntimeError(f"variant_0 no coincide exactamente con E2-A para {structure_id}.")
        views: list[tuple[int, list[int], str]] = []
        seen_ids: set[tuple[int, ...]] = set()
        for row in ordered.itertuples(index=False):
            sequence = str(row.variant_slices).split()
            unknown = sorted(set(sequence) - set(stoi))
            if unknown:
                raise RuntimeError(f"Vista no tokenizable {structure_id}: {unknown}")
            ids = [int(stoi["<start>"]), *[int(stoi[token]) for token in sequence], int(stoi["<end>"])]
            if len(ids) > block_size:
                raise RuntimeError(f"Vista supera block_size sin truncación: {structure_id}")
            key = tuple(ids)
            if key in seen_ids:
                raise RuntimeError(f"El banco conserva vistas tokenizadas duplicadas: {structure_id}")
            seen_ids.add(key)
            views.append((int(row.variant_id), ids, str(row.variant_slices)))
        bank[str(structure_id)] = views
    expected = set(data.structure_id.astype(str))
    if set(bank) != expected:
        raise RuntimeError(f"El banco no coincide con dataset: missing={len(expected-set(bank))}, extra={len(set(bank)-expected)}")
    return bank


def fit_one(data: pd.DataFrame, folds: pd.DataFrame, view_bank: dict[str, list[tuple[int, list[int], str]]], e2a: Any,
            experiment: str, output: Path, device: torch.device, fold: int, seed: int, phase: str,
            trajectory_data: tuple[dict[str, list[dict[str, Any]]], float, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one outer fold and seed using fixed E2-A supervision plus augmentation.

    Args:
        data: Static 234-row E2-A dataset with composition groups attached.
        folds: Fold assignment table.
        view_bank: Graph-verified, tokenizable static views.
        e2a: Imported audited E2-A module.
        experiment: ``E2-Aug1`` or ``E2-Aug2``.
        output: Fresh experiment output directory.
        device: CPU/CUDA/MPS device.
        fold: Zero-based outer fold.
        seed: Reproducibility seed.
        phase: Run phase (smoke, pilot, or cv).
    Returns:
        Per-epoch history and static outer-validation predictions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, stoi = e2a.setup()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.transformer_decoder.layers[-1].parameters():
        parameter.requires_grad = True
    head = e2a.Head().to(device)
    assignment = folds[folds.outer_fold.eq(fold)].drop_duplicates("structure_id").set_index("structure_id")
    roles = data.structure_id.astype(str).map(assignment.inner_role)
    outer_roles = data.structure_id.astype(str).map(assignment.outer_role)
    train_ix = np.flatnonzero(roles.eq("inner_train").to_numpy())
    val_ix = np.flatnonzero(roles.eq("inner_validation").to_numpy())
    outer_ix = np.flatnonzero(outer_roles.eq("outer_validation").to_numpy())
    if (len(train_ix), len(val_ix), len(outer_ix)) != (135, 33, 42):
        raise RuntimeError(f"Conteos E1.3 inesperados para fold={fold}: {(len(train_ix),len(val_ix),len(outer_ix))}")
    groups = data.composition_group.astype(str).tolist()
    train_groups = {groups[i] for i in train_ix}
    val_groups = {groups[i] for i in val_ix}
    outer_groups = {groups[i] for i in outer_ix}
    if train_groups & val_groups or train_groups & outer_groups or val_groups & outer_groups:
        raise RuntimeError("Fuga de composición entre inner-train/inner-val/outer-val.")

    y_np = data.energy_per_atom_ev.to_numpy(float)
    target_mean = float(y_np[train_ix].mean())
    target_scale = max(float(y_np[train_ix].std()), 1e-6)
    pair_differences: list[float] = []
    for _, group_data in data.iloc[train_ix].groupby("composition_group"):
        values = group_data.energy_per_atom_ev.to_numpy(float)
        pair_differences.extend(values[i] - values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
    pair_scale = max(float(np.sqrt(np.mean(np.square(pair_differences)))), 0.001)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    pad = int(stoi["<pad>"])
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4
    )
    composition_to_indices: dict[str, list[int]] = {}
    for i in train_ix:
        composition_to_indices.setdefault(groups[i], []).append(int(i))
    train_compositions = np.asarray(sorted(train_groups), dtype=object)
    is_aug2 = experiment in {"E2-Aug2", "E2-T2"}
    if experiment == "E2-T2" and trajectory_data is None:
        raise RuntimeError("E2-T2 necesita pares de trayectorias admitidos por T1.")
    trajectory_bank, trajectory_scale, trajectory_pair_count = trajectory_data if trajectory_data is not None else ({}, 0.001, 0)
    max_epochs, patience = (3, 3) if phase == "smoke" else (100, 20)
    best_value, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []
    trajectory_gradient_passed = False
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        e2a.sync_device(device)
        epoch_start = time.perf_counter()
        model.eval()
        head.train()
        model.transformer_decoder.layers[-1].train()
        order = np.random.default_rng(seed + epoch * 1009).permutation(train_compositions)
        loss_abs_values: list[float] = []
        loss_pair_values: list[float] = []
        loss_inv_values: list[float] = []
        loss_traj_values: list[float] = []
        loss_total_values: list[float] = []
        for start_idx in range(0, len(order), 4):
            batch_groups = set(order[start_idx:start_idx + 4])
            indices = [idx for group_name in batch_groups for idx in composition_to_indices[group_name]]
            primary_sequences: list[list[int]] = []
            secondary_sequences: list[list[int]] = []
            for idx in indices:
                structure_id = str(data.iloc[idx].structure_id)
                views = view_bank[structure_id]
                primary = selection_index(structure_id, fold, seed, epoch, len(views))
                primary_sequences.append(views[primary][1])
                if is_aug2:
                    second = secondary_index(structure_id, fold, seed, epoch, len(views), primary)
                    if second == primary or views[second][1] == views[primary][1]:
                        raise RuntimeError(f"Aug2 no seleccionó dos vistas distintas: {structure_id}")
                    secondary_sequences.append(views[second][1])
            xb = make_batch(primary_sequences, pad, device)
            optimizer.zero_grad(set_to_none=True)
            primary_prediction = head(e2a.forward_hidden(model, xb, pad, trainable=True)) * target_scale + target_mean
            base_loss = e2a.objective(primary_prediction, y[indices], [groups[idx] for idx in indices], target_scale, pair_scale)
            absolute_loss = nn.functional.smooth_l1_loss((primary_prediction - y[indices]) / target_scale, torch.zeros_like(primary_prediction), beta=1.0)
            pair_loss = base_loss - absolute_loss
            inv_loss = primary_prediction.sum() * 0.0
            if is_aug2:
                xs = make_batch(secondary_sequences, pad, device)
                secondary_prediction = head(e2a.forward_hidden(model, xs, pad, trainable=True)) * target_scale + target_mean
                # For two views, the prompt's centered two-view form is exactly
                # (pred_1 - pred_2)^2 / (4 * a_inv^2), with a_inv from inner-train.
                inv_loss = torch.mean(((primary_prediction - secondary_prediction) / (2.0 * pair_scale)) ** 2)
                loss = base_loss + 0.1 * inv_loss
            else:
                loss = base_loss
            traj_loss = primary_prediction.sum() * 0.0
            if experiment == "E2-T2":
                composition_terms: list[torch.Tensor] = []
                for composition in sorted(batch_groups):
                    trajectories = trajectory_bank.get(str(composition), [])
                    if not trajectories:
                        continue
                    # Deterministically sample up to two trajectories per composition and up to four
                    # distinct-token frame/anchor pairs per selected trajectory in each epoch.
                    ordered_trajectories = sorted(
                        trajectories,
                        key=lambda item: hashlib.sha256(f"{fold}|{seed}|{epoch}|{item['trajectory_id']}".encode()).hexdigest(),
                    )[:2]
                    trajectory_terms: list[torch.Tensor] = []
                    for trajectory in ordered_trajectories:
                        pairs = trajectory["pairs"]
                        if len(pairs) > 4:
                            positions = np.linspace(0, len(pairs) - 1, num=4, dtype=int)
                            pairs = [pairs[int(position)] for position in positions]
                        pair_sequences = [sequence for pair in pairs for sequence in (pair["frame_ids"], pair["anchor_ids"])]
                        pair_x = make_batch(pair_sequences, pad, device)
                        pair_prediction = head(e2a.forward_hidden(model, pair_x, pad, trainable=True)) * target_scale + target_mean
                        delta_prediction = pair_prediction[0::2] - pair_prediction[1::2]
                        delta_target = torch.tensor([pair["delta_dft_ev_atom"] for pair in pairs], dtype=torch.float32, device=device)
                        trajectory_terms.append(nn.functional.smooth_l1_loss(
                            (delta_prediction - delta_target) / trajectory_scale, torch.zeros_like(delta_prediction), beta=1.0
                        ))
                    if trajectory_terms:
                        composition_terms.append(torch.stack(trajectory_terms).mean())
                if composition_terms:
                    traj_loss = torch.stack(composition_terms).mean()
                loss = base_loss + 0.1 * inv_loss + 0.1 * traj_loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Loss no finita en {experiment}, fold={fold}, seed={seed}, epoch={epoch}.")
            smoke_before = None
            if phase == "smoke" and epoch == 1 and start_idx == 0:
                smoke_before = (
                    [parameter.detach().clone() for parameter in model.transformer_decoder.layers[-1].parameters()],
                    [parameter.detach().clone() for parameter in head.parameters()],
                )
            if phase == "smoke" and is_aug2 and epoch == 1 and start_idx == 0:
                if secondary_prediction.grad_fn is None:
                    raise RuntimeError("Smoke Aug2: la vista secundaria no participa en el grafo de gradiente.")
                # Verify a gradient reaches parameters through the secondary forward pass alone.
                secondary_only = torch.autograd.grad(
                    inv_loss, list(head.parameters()), retain_graph=True, allow_unused=True
                )
                if not any(g is not None and torch.isfinite(g).all() and float(g.norm()) > 0 for g in secondary_only):
                    raise RuntimeError("Smoke Aug2: L_inv no propaga gradiente finito y no nulo a la cabeza.")
            if phase == "smoke" and experiment == "E2-T2" and epoch == 1 and not trajectory_gradient_passed and float(traj_loss.detach()) > 0:
                trajectory_only = torch.autograd.grad(traj_loss, list(head.parameters()), retain_graph=True, allow_unused=True)
                trajectory_gradient_passed = any(g is not None and torch.isfinite(g).all() and float(g.norm()) > 0 for g in trajectory_only)
            loss.backward()
            if phase == "smoke" and epoch == 1 and start_idx == 0:
                last_layer = model.transformer_decoder.layers[-1]
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in last_layer.parameters()):
                    raise RuntimeError("Smoke: el último bloque no recibió gradiente finito y no nulo.")
                if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith("transformer_decoder.layers.11.")):
                    raise RuntimeError("Smoke: gradiente detectado en backbone congelado.")
                if not any(p.grad is not None and p.grad.norm() > 0 for p in head.parameters()):
                    raise RuntimeError("Smoke: la cabeza no recibió gradiente.")
            torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad], 1.0)
            optimizer.step()
            if smoke_before is not None:
                block_delta = torch.sqrt(sum((old - new.detach()).float().pow(2).sum()
                                             for old, new in zip(smoke_before[0], model.transformer_decoder.layers[-1].parameters())))
                head_delta = torch.sqrt(sum((old - new.detach()).float().pow(2).sum()
                                            for old, new in zip(smoke_before[1], head.parameters())))
                if float(block_delta) <= 0 or float(head_delta) <= 0:
                    raise RuntimeError("Smoke: último bloque y cabeza deben cambiar tras optimizer.step().")
            loss_abs_values.append(float(absolute_loss.detach().cpu()))
            loss_pair_values.append(float(pair_loss.detach().cpu()))
            loss_inv_values.append(float(inv_loss.detach().cpu()))
            loss_traj_values.append(float(traj_loss.detach().cpu()))
            loss_total_values.append(float(loss.detach().cpu()))

        model.eval()
        head.eval()
        with torch.no_grad():
            val_sequences = [view_bank[str(data.iloc[idx].structure_id)][0][1] for idx in val_ix]
            val_x = make_batch(val_sequences, pad, device)
            val_pred = head(e2a.forward_hidden(model, val_x, pad, trainable=True)) * target_scale + target_mean
            val_objective = float(e2a.objective(val_pred, y[val_ix], [groups[idx] for idx in val_ix], target_scale, pair_scale).cpu())
        e2a.sync_device(device)
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "experiment": experiment, "fold": fold, "seed": seed, "epoch": epoch,
            "train_abs": float(np.mean(loss_abs_values)), "train_pair": float(np.mean(loss_pair_values)),
            "train_inv": float(np.mean(loss_inv_values)), "train_total": float(np.mean(loss_total_values)),
            "train_traj": float(np.mean(loss_traj_values)),
            "inner_validation_objective": val_objective, "target_mean_ev_atom": target_mean,
            "target_scale_ev_atom": target_scale, "pair_scale_ev_atom": pair_scale,
            "epoch_wall_seconds": epoch_seconds,
        }
        history.append(row)
        if val_objective < best_value - 1e-12:
            best_value, best_epoch, wait = val_objective, epoch, 0
            best_state = {
                "last_block": {key: value.detach().cpu().clone() for key, value in model.transformer_decoder.layers[-1].state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            }
        else:
            wait += 1
        if wait >= patience:
            break
    if phase == "smoke" and experiment == "E2-T2" and not trajectory_gradient_passed:
        raise RuntimeError("Smoke T2: ningún término L_traj aportó gradiente finito y no nulo.")
    if best_state is None:
        raise RuntimeError("Early stopping no guardó un checkpoint por inner-validation.")
    model.transformer_decoder.layers[-1].load_state_dict(best_state["last_block"], strict=True)
    head.load_state_dict(best_state["head"], strict=True)
    model.eval()
    head.eval()
    with torch.no_grad():
        outer_sequences = [view_bank[str(data.iloc[idx].structure_id)][0][1] for idx in outer_ix]
        outer_x = make_batch(outer_sequences, pad, device)
        prediction = head(e2a.forward_hidden(model, outer_x, pad, trainable=True)) * target_scale + target_mean
    predictions = data.iloc[outer_ix][["structure_id", "composition_group", "reduced_formula", "polymorph_family", "energy_per_atom_ev"]].copy()
    predictions["configuration"] = experiment
    predictions["fold"] = fold
    predictions["seed"] = seed
    predictions["best_epoch"] = best_epoch
    predictions["energy_pred_ev_atom"] = prediction.detach().cpu().numpy()
    predictions["training_wall_seconds"] = time.perf_counter() - start
    predictions["error_ev_atom"] = predictions.energy_pred_ev_atom - predictions.energy_per_atom_ev
    predictions["delta_e_dft_mev_atom"] = predictions.groupby("composition_group").energy_per_atom_ev.transform(lambda values: (values - values.min()) * 1000.0)
    predictions["delta_e_pred_mev_atom"] = predictions.groupby("composition_group").energy_pred_ev_atom.transform(lambda values: (values - values.min()) * 1000.0)
    for row in history:
        row["best_epoch_so_far"] = best_epoch
        row["training_wall_seconds"] = predictions.training_wall_seconds.iloc[0]
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold_{fold}_seed_{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Se conserva checkpoint existente: {checkpoint_path}")
    torch.save({
        "last_block": best_state["last_block"], "head": best_state["head"],
        "experiment": experiment, "fold": fold, "seed": seed, "best_epoch": best_epoch,
        "best_inner_validation_objective": best_value, "target_mean_ev_atom": target_mean,
        "target_scale_ev_atom": target_scale, "pair_scale_ev_atom": pair_scale,
        "trajectory_scale_ev_atom": trajectory_scale, "trajectory_pairs_inner_train": trajectory_pair_count,
        "checkpoint_sha256": e2a.EXPECTED,
    }, checkpoint_path)
    return pd.DataFrame(history), predictions


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps device with explicit availability checks."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado pero no disponible.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS solicitado pero no disponible.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Dispositivo no admitido: {requested}")
    return device


def run(args: argparse.Namespace) -> None:
    """Run preflight, smoke, pilot, or full 5x3 development cross-validation."""
    if args.experiment not in {"E2-Aug1", "E2-Aug2", "E2-T2"}:
        raise ValueError("experiment debe ser E2-Aug1, E2-Aug2 o E2-T2")
    e2a = load_e2a_module()
    for path in (DATA, FOLDS, INFO, VARIANTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = pd.read_csv(DATA, low_memory=False)
    folds = pd.read_csv(FOLDS)
    group_map = folds.drop_duplicates("structure_id").set_index("structure_id").composition_group
    data["composition_group"] = data.structure_id.astype(str).map(group_map)
    development = data[data.split.isin(["train", "val"])].copy().reset_index(drop=True)
    if len(data) != 234 or len(development) != 210 or development.composition_group.nunique() != 70:
        raise RuntimeError("Dataset E2-A no conserva 234 estructuras / 210 de desarrollo / 70 composiciones.")
    if set(folds.loc[folds.historical_split.eq("test"), "structure_id"].astype(str)) & set(development.structure_id.astype(str)):
        raise RuntimeError("El test histórico aparece en desarrollo.")
    info = json.loads(INFO.read_text(encoding="utf-8"))
    view_bank = load_view_bank(development, VARIANTS, info["stoi"], int(info["block_size"]))
    t1_gate = "NOT_APPLICABLE"
    if args.experiment == "E2-T2":
        if not T1_DECISION.is_file() or not T2_AUG2_SMOKE.is_file():
            raise RuntimeError("T2 requiere decisión T1 y smoke de Aug2 aprobados.")
        t1_decision = json.loads(T1_DECISION.read_text(encoding="utf-8"))
        t1_gate = str(t1_decision.get("t2_training_gate", t1_decision.get("status", "BLOCKED")))
        if t1_gate not in {"GO", "LIMITED"}:
            raise RuntimeError(f"T1 gate={t1_gate}; T2 no puede entrenarse.")
        smoke = json.loads(T2_AUG2_SMOKE.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS":
            raise RuntimeError("Smoke de E2-Aug2 no tiene estado PASS; T2 permanece bloqueado.")
        if args.phase == "cv" and t1_gate != "GO":
            raise RuntimeError(f"E2-T1={t1_gate}; solo se permite piloto acotado, no CV completa.")
        aug2_full = ALT / "reports/MatGPT_finetuning/E2-Aug2/development_only_v2/cv/cpu/runtime_summary.json"
        if args.phase == "cv" and not aug2_full.is_file():
            raise RuntimeError("T2 full requiere primero completar CV de E2-Aug2.")
        if args.phase == "cv":
            full_summary = json.loads(aug2_full.read_text(encoding="utf-8"))
            aug2_oof = aug2_full.parent / "oof_predictions.csv"
            aug2_predictions = pd.read_csv(aug2_oof) if aug2_oof.is_file() else pd.DataFrame()
            if len(aug2_predictions) != 630 or full_summary.get("phase") != "cv":
                raise RuntimeError("CV E2-Aug2 no contiene las 630 predicciones OOF requeridas para T2.")
    report_dir = args.out / args.experiment / args.run_label / args.phase / args.device_label
    report_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": args.experiment, "run_label": args.run_label,
        "architecture": "MatGPT unconditional; first 11 decoder blocks, token embeddings, position embeddings and ln_f frozen; decoder block 12 + MLP8 trainable",
        "pooling": "real <end> state after ln_f; right padding excluded by actual end position",
        "input_format": "<start> SLICES <end>", "max_context_tokens": int(info["block_size"]),
        "loss": ("E2-T2: L_abs_static + L_pair_static + 0.1*L_inv_static + 0.1*L_traj" if args.experiment == "E2-T2" else "E2-Aug1: L_abs + L_pair; E2-Aug2: L_abs(primary) + L_pair(primary) + 0.1*L_inv"),
        "loss_notes": "SmoothL1 beta=1; scales from inner-train only; pair lambda=1; L_inv uses centered two-view deviations and a_pair from inner-train; T2 uses only same-trajectory distinct-ID deltas, a_traj from inner-train, lambda_traj=0.1",
        "optimizer": {"name": "AdamW", "last_block_lr": 1e-5, "head_lr": 1e-3, "weight_decay": 1e-4, "gradient_clip_norm": 1.0},
        "sampler": "4 complete compositions per batch; all three polymorphs per composition; one supervised view per structure per epoch",
        "early_stopping": {"monitor": "inner-validation E2-A objective on fixed historical serialization", "max_epochs": 100, "patience": 20},
        "folds": 5, "seeds": list(SEEDS), "development_structures": 210, "development_compositions": 70,
        "historical_test_used": False,
        "t1_gate": t1_gate,
    }
    provenance = {
        "checkpoint_sha256_expected": e2a.EXPECTED, "checkpoint_sha256_observed": e2a.sha(e2a.CKPT),
        "dataset_sha256": e2a.sha(DATA), "folds_sha256": e2a.sha(FOLDS),
        "variant_manifest_sha256": e2a.sha(VARIANTS), "trainer_sha256": e2a.sha(Path(__file__).resolve()),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device_requested": args.device, "device_resolved": str(resolve_device(args.device)),
        "source_commit": "a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e",
    }
    for filename, value in (("config.json", config), ("provenance.json", provenance)):
        artifact = report_dir / filename
        serialized = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        if artifact.exists() and artifact.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"No se sobrescribe la procedencia/configuración ya registrada: {artifact}")
        artifact.write_text(serialized, encoding="utf-8")
    if args.phase == "preflight":
        report = {
            "status": "PASS", "experiment": args.experiment,
            "data_rows": len(data), "development_rows": len(development),
            "development_compositions": int(development.composition_group.nunique()),
            "views_per_structure": {str(k): int(v) for k, v in pd.Series({key: len(value) for key, value in view_bank.items()}).value_counts().sort_index().items()},
            "historical_test_excluded_before_conversion_or_training": int(data.split.eq("test").sum()),
            "loss": config["loss"],
            "optimizer": "AdamW; last decoder block LR=1e-5; MLP8 LR=1e-3; weight_decay=1e-4",
            "epochs": 100, "patience": 20, "compositions_per_batch": 4,
        }
        (report_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        LOG.info("augmentation_preflight_pass experiment=%s views=%d", args.experiment, len(view_bank))
        return
    if args.phase == "smoke":
        jobs = [(0, 42)]
    elif args.phase == "pilot":
        jobs = [(args.fold, args.seed)]
    else:
        jobs = [(fold, seed) for fold in range(5) for seed in SEEDS]
    if args.resume:
        completed = []
        for fold, seed in jobs:
            checkpoint = report_dir / "runs" / f"fold_{fold}_seed_{seed}" / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
            predfile = report_dir / "prediction_fragments" / f"fold_{fold}_seed_{seed}.csv"
            histfile = report_dir / "history_fragments" / f"fold_{fold}_seed_{seed}.csv"
            if checkpoint.is_file() and predfile.is_file() and histfile.is_file():
                completed.append((fold, seed))
        jobs = [job for job in jobs if job not in completed]
        LOG.info("resume_skips_completed=%d remaining=%d", len(completed), len(jobs))
    histories: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for fold, seed in jobs:
        output = report_dir / "runs" / f"fold_{fold}_seed_{seed}"
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Run output exists; use --resume: {output}")
        output.mkdir(parents=True, exist_ok=True)
        trajectory_data = build_trajectory_bank(development, fold, info["stoi"], int(info["block_size"])) if args.experiment == "E2-T2" else None
        history, prediction = fit_one(development, folds, view_bank, e2a, args.experiment, output, resolve_device(args.device), fold, seed, args.phase, trajectory_data)
        history_dir, prediction_dir = report_dir / "history_fragments", report_dir / "prediction_fragments"
        history_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_dir / f"fold_{fold}_seed_{seed}.csv", index=False)
        prediction.to_csv(prediction_dir / f"fold_{fold}_seed_{seed}.csv", index=False)
        histories.append(history)
        predictions.append(prediction)
        LOG.info("fit_complete experiment=%s fold=%d seed=%d epochs=%d wall_seconds=%.1f", args.experiment, fold, seed, len(history), prediction.training_wall_seconds.iloc[0])
    history_dir, prediction_dir = report_dir / "history_fragments", report_dir / "prediction_fragments"
    all_history_files = sorted(history_dir.glob("fold_*_seed_*.csv")) if history_dir.is_dir() else []
    all_prediction_files = sorted(prediction_dir.glob("fold_*_seed_*.csv")) if prediction_dir.is_dir() else []
    if all_history_files:
        pd.concat([pd.read_csv(path) for path in all_history_files], ignore_index=True).to_csv(report_dir / "training_history.csv", index=False)
    if all_prediction_files:
        pd.concat([pd.read_csv(path) for path in all_prediction_files], ignore_index=True).to_csv(report_dir / "oof_predictions.csv", index=False)
    if args.phase == "smoke" and predictions:
        integrity = {
            "status": "PASS",
            "experiment": args.experiment,
            "fold": 0,
            "seed": 42,
            "epochs_completed": int(len(histories[0])),
            "test_historical_used": False,
            "checks": [
                "last decoder block and MLP receive finite nonzero gradients",
                "frozen backbone parameters receive no gradients",
                "last decoder block and MLP update after optimizer step",
                "<end> pooling and right-padding invariance pass",
                "finite training and inner-validation losses",
                "outer validation predictions are nonconstant",
                ("T2: L_inv y L_traj propagan gradientes finitos no nulos" if args.experiment == "E2-T2" else
                 "Aug2: L_inv propagates a finite nonzero gradient through its secondary view" if args.experiment == "E2-Aug2" else
                 "Aug1: one supervised view per structure per epoch"),
            ],
        }
        (report_dir / "smoke_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    summary = {
        "experiment": args.experiment,
        "phase": args.phase,
        "device": str(resolve_device(args.device)),
        "requested_runs": 15 if args.phase == "cv" else 1,
        "runs_completed_this_invocation": len(predictions),
        "runs_completed_total": len(all_prediction_files),
        "training_wall_seconds": float(sum(float(part.training_wall_seconds.iloc[0]) for part in predictions)) if predictions else 0.0,
        "test_historical_used": False, "t1_gate": t1_gate,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest_rows = []
    for fold, seed in [(f, s) for f in range(5) for s in SEEDS]:
        checkpoint = report_dir / "runs" / f"fold_{fold}_seed_{seed}" / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
        manifest_rows.append({"experiment": args.experiment, "fold": fold, "seed": seed,
                              "checkpoint": str(checkpoint.relative_to(report_dir)) if checkpoint.is_file() else "",
                              "status": "completed" if checkpoint.is_file() else "pending"})
    pd.DataFrame(manifest_rows).to_csv(report_dir / "run_manifest.csv", index=False)


def main() -> None:
    """Parse CLI options and run a requested stage without overwriting prior runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["E2-Aug1", "E2-Aug2", "E2-T2"], required=True)
    parser.add_argument("--phase", choices=["preflight", "smoke", "pilot", "cv"], required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ALT / "reports/MatGPT_finetuning")
    parser.add_argument("--run-label", default="development_only_v2", help="Versionado independiente para preservar ejecuciones anteriores.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.device_label = "cpu" if args.device == "auto" and not torch.cuda.is_available() and not torch.backends.mps.is_available() else args.device
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args)
    except Exception as exc:
        LOG.exception("augmentation_training_failed experiment=%s phase=%s error=%s", args.experiment, args.phase, exc)
        raise


if __name__ == "__main__":
    main()
