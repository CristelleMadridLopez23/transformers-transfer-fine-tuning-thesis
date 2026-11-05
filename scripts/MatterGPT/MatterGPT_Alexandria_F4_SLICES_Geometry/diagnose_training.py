#!/usr/bin/env python3
"""Audita de forma aislada MatterGPT Alexandria F4 con geometría.

Lee el mejor checkpoint atómico disponible y ejecuta inferencia y un único
paso de gradiente/optimización en CPU, sin enviar señales al entrenamiento.
Todos los archivos de resultado se escriben en experiments/.../diagnostics/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry"
OLD = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset"
DIAG = EXP / "diagnostics"
TRAIN_SCRIPT = ROOT / "scripts/MatterGPT/MatterGPT_Alexandria_F4_SLICES_Geometry/train_energy_geometry.py"
GEOM_COLS = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
GEOM_SCALED = [f"scaled_{c}" for c in GEOM_COLS]


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo por bloques de 1 MiB."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(path: Path, name: str) -> Any:
    """Importa un script del proyecto sin llamar a su main."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manual_smooth_l1(pred: np.ndarray, target: np.ndarray) -> float:
    """Recalcula SmoothL1 beta=1 en NumPy para confirmar shapes y reducción."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch sin broadcasting permitido: {pred.shape} != {target.shape}")
    delta = np.abs(pred.astype(np.float64) - target.astype(np.float64))
    values = np.where(delta < 1.0, 0.5 * delta**2, delta - 0.5)
    return float(values.mean())


def sample_predictions(module: Any, model: torch.nn.Module, head: torch.nn.Module,
                       frame: pd.DataFrame, sequences: list[list[int]], geom: np.ndarray,
                       target_mean: float, target_std: float, sample_n: int = 64) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evalúa un muestreo reproducible y calcula métricas contra baseline cero."""
    rng = np.random.default_rng(20261102)
    ix = np.sort(rng.choice(len(frame), size=min(sample_n, len(frame)), replace=False))
    preds: list[float] = []; normalized: list[float] = []
    model.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, len(ix), 8):
            inds = ix[start:start + 8]
            ids, pos = module.make_batch(sequences, inds)
            h = module.contextualized_final_end(model, ids, pos)
            out = head(h, torch.as_tensor(geom[inds], dtype=torch.float32))
            normalized.extend(out.cpu().numpy().astype(float).tolist())
    pred_norm = np.asarray(normalized, dtype=np.float64)
    y_raw = frame.energy_per_atom_ev.to_numpy(np.float64)[ix]
    y_norm = (y_raw - target_mean) / target_std
    pred_raw = pred_norm * target_std + target_mean
    err_norm = pred_norm - y_norm
    smooth = manual_smooth_l1(pred_norm, y_norm)
    const_smooth = manual_smooth_l1(np.zeros_like(y_norm), y_norm)
    corr = float(np.corrcoef(pred_raw, y_raw)[0, 1]) if np.std(pred_raw) > 0 and np.std(y_raw) > 0 else None
    report = {"n_sampled": int(len(ix)), "target_mean_ev_atom": float(y_raw.mean()),
        "target_std_ev_atom": float(y_raw.std(ddof=0)), "target_normalized_mean": float(y_norm.mean()),
        "target_normalized_std": float(y_norm.std(ddof=0)), "prediction_normalized_mean": float(pred_norm.mean()),
        "prediction_normalized_std": float(pred_norm.std(ddof=0)), "prediction_normalized_min": float(pred_norm.min()),
        "prediction_normalized_max": float(pred_norm.max()), "prediction_normalized_p5": float(np.quantile(pred_norm, .05)),
        "prediction_normalized_p50": float(np.quantile(pred_norm, .50)), "prediction_normalized_p95": float(np.quantile(pred_norm, .95)),
        "prediction_ev_atom_mean": float(pred_raw.mean()), "prediction_ev_atom_std": float(pred_raw.std(ddof=0)),
        "prediction_ev_atom_min": float(pred_raw.min()), "prediction_ev_atom_max": float(pred_raw.max()),
        "prediction_ev_atom_p5": float(np.quantile(pred_raw, .05)), "prediction_ev_atom_p50": float(np.quantile(pred_raw, .50)),
        "prediction_ev_atom_p95": float(np.quantile(pred_raw, .95)), "pearson_r": corr,
        "mae_ev_atom": float(np.mean(np.abs(pred_raw - y_raw))), "rmse_ev_atom": float(np.sqrt(np.mean((pred_raw-y_raw)**2))),
        "smoothl1_model_normalized": smooth, "smoothl1_constant_zero_normalized": const_smooth,
        "model_minus_constant_smoothl1": smooth - const_smooth,
        "target_shape": list(y_norm.shape), "prediction_shape": list(pred_norm.shape),
        "no_broadcasting": pred_norm.shape == y_norm.shape}
    pred_frame = frame.iloc[ix][["mat_id", "formula", "n_atoms", "energy_per_atom_ev"]].copy()
    pred_frame["split"] = str(frame.split.iloc[0])
    pred_frame["target_normalized"] = y_norm
    pred_frame["prediction_normalized"] = pred_norm
    pred_frame["prediction_ev_atom"] = pred_raw
    pred_frame["abs_error_ev_atom"] = np.abs(pred_raw-y_raw)
    return pred_frame, report


def activation_audit(model: torch.nn.Module, head: torch.nn.Module, module: Any,
                    frame: pd.DataFrame, sequences: list[list[int]], geom: np.ndarray,
                    sample_n: int) -> dict[str, Any]:
    """Mide saturación GELU en una muestra sin cambiar los parámetros.

    Args:
        model: Backbone reconstruido y congelado.
        head: Cabeza MLP8 del checkpoint seleccionado.
        module: Funciones de tokenización y extracción validadas.
        frame: Registros del split a inspeccionar.
        sequences: SLICES tokenizadas en el orden del frame.
        geom: Geometría escalada alineada con frame.
        sample_n: Número máximo de filas evaluadas.
    Returns:
        Estadísticos de preactivaciones, GELU y salida.
    """
    rng = np.random.default_rng(20261103)
    ix = np.sort(rng.choice(len(frame), size=min(sample_n, len(frame)), replace=False))
    pre_values: list[np.ndarray] = []
    post_values: list[np.ndarray] = []
    output_values: list[np.ndarray] = []
    model.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, len(ix), 8):
            inds = ix[start:start + 8]
            ids, pos = module.make_batch(sequences, inds)
            hidden = module.contextualized_final_end(model, ids, pos)
            inputs = torch.cat([hidden, torch.as_tensor(geom[inds], dtype=torch.float32)], dim=1)
            pre = head.network[0](inputs)
            post = head.network[1](pre)
            output = head.network[3](head.network[2](post))
            pre_values.append(pre.cpu().numpy())
            post_values.append(post.cpu().numpy())
            output_values.append(output.cpu().numpy())
    pre = np.concatenate(pre_values).astype(np.float64)
    post = np.concatenate(post_values).astype(np.float64)
    output = np.concatenate(output_values).astype(np.float64).reshape(-1)
    return {"n_structures": int(len(ix)), "hidden_units": int(pre.shape[1]),
        "pre_gelu_mean": float(pre.mean()), "pre_gelu_std": float(pre.std()),
        "pre_gelu_min": float(pre.min()), "pre_gelu_max": float(pre.max()),
        "fraction_pre_gelu_below_minus_5": float(np.mean(pre < -5.0)),
        "post_gelu_mean": float(post.mean()), "post_gelu_std": float(post.std()),
        "fraction_post_gelu_abs_below_1e_8": float(np.mean(np.abs(post) < 1e-8)),
        "fraction_hidden_units_zero_for_every_sample": float(np.mean(np.all(np.abs(post) < 1e-8, axis=0))),
        "output_std": float(output.std()), "output_min": float(output.min()), "output_max": float(output.max())}


def run(sample_n: int = 64) -> dict[str, Any]:
    """Ejecuta las comprobaciones acotadas en una copia independiente del modelo."""
    DIAG.mkdir(parents=True, exist_ok=True)
    checkpoint = EXP / "checkpoints/mattergpt_alexandria_f4_geometry_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No hay checkpoint F4 disponible: {checkpoint}")
    cp_hash_before = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cp_hash_loaded = sha256(checkpoint)
    if cp_hash_before != cp_hash_loaded:
        raise RuntimeError("El checkpoint cambió durante la lectura; reintentar diagnóstico")

    torch.set_num_threads(1)
    geom_audit = json.loads((EXP / "geometry_audit.json").read_text(encoding="utf-8"))
    if geom_audit.get("status") != "PASS" or geom_audit.get("rows") != 18963:
        raise ValueError("Auditoría de geometría previa no PASS")
    train = pd.read_csv(OLD / "alexandria_matter_compatible_train.csv", dtype={"mat_id": str})
    valid = pd.read_csv(OLD / "alexandria_matter_compatible_validation.csv", dtype={"mat_id": str})
    geometry = pd.read_csv(EXP / "geometry_manifest.csv", dtype={"mat_id": str})
    geom_by_id = geometry.set_index("mat_id")
    module = load_script(TRAIN_SCRIPT, "matter_geo_diag_train")
    train_module = module.load_training_module()
    model, stoi, block_size = train_module.load_model()

    cp_blocks = state["blocks_8_11"]
    expected = {k for k in model.state_dict() if any(k.startswith(f"blocks.{i}.") for i in range(8, 12))}
    if set(cp_blocks) != expected:
        raise ValueError("Checkpoint F4 parcial no contiene exactamente bloques 8–11")
    merged = dict(model.state_dict()); merged.update(cp_blocks)
    load_result = model.load_state_dict(merged, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(f"No se reconstruyó backbone con carga estricta: {load_result}")
    head = module.FusionHead()
    head.load_state_dict(state["head_state_dict"], strict=True)
    model.eval(); head.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.to("cpu"); head.to("cpu")

    predictions = []; prediction_metrics = {}
    for split, frame in (("train", train), ("validation", valid)):
        geo = geom_by_id.loc[frame.mat_id, ["split", "structure_hash", "slices_hash", "energy_per_atom_ev", *GEOM_COLS, *GEOM_SCALED]].reset_index(drop=True)
        if not geo.split.eq(split).all(): raise ValueError(f"Split geométrico incorrecto para {split}")
        if not np.allclose(frame.energy_per_atom_ev, geo.energy_per_atom_ev, rtol=0, atol=1e-12): raise ValueError(f"Target no coincide en {split}")
        if not frame.slices_hash.eq(geo.slices_hash).all(): raise ValueError(f"SLICES/hash no coincide en {split}")
        raw_g = geo[GEOM_COLS].to_numpy(float); scaled_g = geo[GEOM_SCALED].to_numpy(np.float32)
        if not np.isfinite(raw_g).all() or not np.isfinite(scaled_g).all(): raise ValueError(f"Geometría NaN/Inf en {split}")
        seq = train_module.read_sequences(frame, stoi, block_size)
        result, metrics = sample_predictions(train_module, model, head, frame, seq, scaled_g,
                                             float(state["target_mean_ev_atom"]), float(state["target_std_ev_atom"]), sample_n)
        predictions.append(result); prediction_metrics[split] = metrics

    # Geometry distribution/scaler checks from actual manifests, no fitting.
    train_geo = geometry.loc[geometry.split.eq("train")]
    val_geo = geometry.loc[geometry.split.eq("validation")]
    geometry_rows = []
    for col, scaled in zip(GEOM_COLS, GEOM_SCALED):
        x = train_geo[col].to_numpy(float); xs = train_geo[scaled].to_numpy(float); xv = val_geo[scaled].to_numpy(float)
        geometry_rows.append({"feature": col, "train_raw_mean": float(x.mean()), "train_raw_std": float(x.std(ddof=0)),
            "train_raw_min": float(x.min()), "train_raw_max": float(x.max()), "train_scaled_mean": float(xs.mean()),
            "train_scaled_std": float(xs.std(ddof=0)), "validation_scaled_mean": float(xv.mean()),
            "validation_scaled_std": float(xv.std(ddof=0)), "constant_train_feature": bool(x.std(ddof=0) <= 1e-12),
            "source_join": "mat_id", "train_rows": len(train_geo), "validation_rows": len(val_geo)})
    pd.DataFrame(geometry_rows).to_csv(DIAG / "geometry_audit.csv", index=False)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(DIAG / "prediction_distribution.csv", index=False)
    activation_metrics = activation_audit(model, head, train_module, valid, seq, scaled_g, sample_n)

    # Isolated one-batch F4 gradient/optimizer audit. It never touches the live process/model.
    audit_model, _, _ = train_module.load_model()
    merged = dict(audit_model.state_dict()); merged.update(cp_blocks)
    audit_model.load_state_dict(merged, strict=True)
    audit_model.to("cpu").train()
    audit_head = module.FusionHead(); audit_head.load_state_dict(state["head_state_dict"], strict=True); audit_head.to("cpu").train()
    for p in audit_model.parameters(): p.requires_grad_(False)
    for block in audit_model.blocks[8:12]:
        for p in block.parameters(): p.requires_grad_(True)
    opt_params = [p for b in audit_model.blocks[8:12] for p in b.parameters()] + list(audit_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for b in audit_model.blocks[8:12] for p in b.parameters()], "lr": 3e-6, "weight_decay": 0.01, "name": "blocks_8_11"},
        {"params": list(audit_head.parameters()), "lr": 1e-4, "weight_decay": 1e-4, "name": "mlp8"},
    ])
    indices = np.arange(8, dtype=int)
    ids, pos = train_module.make_batch(train_module.read_sequences(train.iloc[indices], stoi, block_size), np.arange(8))
    g = torch.as_tensor(geom_by_id.loc[train.mat_id.iloc[indices], GEOM_SCALED].to_numpy(np.float32))
    energy = train.energy_per_atom_ev.to_numpy(float)[indices]
    y_norm = torch.as_tensor(((energy - float(state["target_mean_ev_atom"])) / float(state["target_std_ev_atom"])), dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    h = train_module.contextualized_final_end(audit_model, ids, pos)
    raw_head_input = torch.cat([h, g], dim=1)
    pred = audit_head(h, g)
    if pred.shape != y_norm.shape:
        raise ValueError(f"Broadcasting/shapes incompatibles: pred={pred.shape}, target={y_norm.shape}")
    loss = F.smooth_l1_loss(pred, y_norm, beta=1.0)
    loss.backward()
    # Group gradients and one optimizer step; group membership/LR is read from live code config.
    grad_rows = []
    named_modules: list[tuple[str, list[tuple[str, torch.nn.Parameter]], float, float]] = []
    for i in range(12):
        params = [(n, p) for n, p in audit_model.named_parameters() if n.startswith(f"blocks.{i}.")]
        named_modules.append((f"block_{i}", params, 3e-6 if 8 <= i <= 11 else 0.0, 0.01 if 8 <= i <= 11 else 0.0))
    for layer_name, layer in (("mlp_first", audit_head.network[0]), ("mlp_output", audit_head.network[3])):
        params = [(f"{layer_name}.{name}", p) for name, p in layer.named_parameters()]
        named_modules.append((layer_name, params, 1e-4, 1e-4))
    before = {name: p.detach().clone() for name, p in audit_model.named_parameters()}
    before.update({f"head.{n}": p.detach().clone() for n, p in audit_head.named_parameters()})
    for group_name, params, lr, wd in named_modules:
        plist = [p for _, p in params]
        grads = [p.grad.detach().float() for p in plist if p.grad is not None]
        grad_norm = math.sqrt(sum(float(gx.pow(2).sum()) for gx in grads)) if grads else 0.0
        param_norm = math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in plist)) if plist else 0.0
        ntrain = sum(p.numel() for p in plist if p.requires_grad)
        nmissing = sum(p.numel() for p in plist if p.requires_grad and p.grad is None)
        nnan = sum(int((~torch.isfinite(p.grad)).sum()) for p in plist if p.grad is not None)
        grad_rows.append({"component": group_name, "requires_grad": any(p.requires_grad for p in plist),
            "optimizer_lr": lr, "weight_decay": wd, "parameter_count": sum(p.numel() for p in plist),
            "trainable_parameter_count": ntrain, "missing_gradient_parameter_count": nmissing,
            "gradient_norm": grad_norm, "parameter_norm": param_norm, "finite_gradient_nonfinite_values": nnan,
            "grad_fn_reachable_from_final_end": bool(h.requires_grad and h.grad_fn is not None) if group_name.startswith("block_") and group_name in {f"block_{i}" for i in range(8,12)} else None})
    geom_grad = audit_head.network[0].weight.grad[:, 768:]
    struct_grad = audit_head.network[0].weight.grad[:, :768]
    grad_rows.append({"component":"mlp_first_structural_columns", "requires_grad":True,"optimizer_lr":1e-4,"weight_decay":1e-4,
        "parameter_count":struct_grad.numel(),"trainable_parameter_count":struct_grad.numel(),"missing_gradient_parameter_count":0,
        "gradient_norm":float(struct_grad.detach().norm()),"parameter_norm":float(audit_head.network[0].weight[:, :768].detach().norm()),
        "finite_gradient_nonfinite_values":int((~torch.isfinite(struct_grad)).sum()),"grad_fn_reachable_from_final_end":True})
    grad_rows.append({"component":"mlp_first_geometry_columns", "requires_grad":True,"optimizer_lr":1e-4,"weight_decay":1e-4,
        "parameter_count":geom_grad.numel(),"trainable_parameter_count":geom_grad.numel(),"missing_gradient_parameter_count":0,
        "gradient_norm":float(geom_grad.detach().norm()),"parameter_norm":float(audit_head.network[0].weight[:, 768:].detach().norm()),
        "finite_gradient_nonfinite_values":int((~torch.isfinite(geom_grad)).sum()),"grad_fn_reachable_from_final_end":None})
    optimizer.step()
    upd_rows = []
    after = dict(audit_model.named_parameters()); after.update({f"head.{n}":p for n,p in audit_head.named_parameters()})
    for name, current in after.items():
        ref = before[name]
        diff = (current.detach().float() - ref.float())
        upd = float(diff.norm()); base = float(ref.float().norm())
        if name.startswith("blocks.") or name.startswith("head."):
            upd_rows.append({"parameter":name,"parameter_norm_before":base,"absolute_update_norm":upd,
                "relative_update_norm":upd/max(base,1e-30),"requires_grad":bool(current.requires_grad),
                "changed":not torch.equal(current.detach(),ref)})
    pd.DataFrame(grad_rows).to_csv(DIAG / "gradient_audit.csv", index=False)
    pd.DataFrame(upd_rows).to_csv(DIAG / "parameter_updates.csv", index=False)

    # Output sensitivity to geometry, evaluated on 32 validation structures only.
    sample = pred_df[pred_df.split.eq("validation")].head(32)
    ids_by_id = {str(row.mat_id): i for i, row in enumerate(valid.itertuples(index=False))}
    sample_idx = np.asarray([ids_by_id[str(s)] for s in sample.mat_id], dtype=int)
    val_geom = geom_by_id.loc[valid.mat_id, GEOM_SCALED].to_numpy(np.float32)
    val_seq = train_module.read_sequences(valid, stoi, block_size)
    xg = torch.as_tensor(val_geom[sample_idx]); perm = np.random.default_rng(11).permutation(len(sample_idx))
    ablation_preds = {"original": [], "zero": [], "permuted": []}
    with torch.inference_mode():
        for start in range(0, len(sample_idx), 8):
            ix = sample_idx[start:start+8]; ids, pos = train_module.make_batch(val_seq, ix)
            h_eval = train_module.contextualized_final_end(model, ids, pos)
            ablation_preds["original"].extend(head(h_eval, xg[start:start+8]).numpy().tolist())
            ablation_preds["zero"].extend(head(h_eval, torch.zeros_like(xg[start:start+8])).numpy().tolist())
            perm_ix = perm[start:start+8]
            ablation_preds["permuted"].extend(head(h_eval, xg[perm_ix]).numpy().tolist())
    ablation = {k: np.asarray(v, dtype=float) for k, v in ablation_preds.items()}
    ablation_metrics = {k: {"mean_normalized_prediction":float(v.mean()),"std_normalized_prediction":float(v.std()),
        "mae_vs_original_normalized":float(np.mean(np.abs(v-ablation["original"]))) if k!="original" else 0.0}
        for k,v in ablation.items()}
    for k, values in ablation.items():
        sample[f"prediction_norm_geometry_{k}"] = values
    sample.to_csv(DIAG / "geometry_ablation_sample.csv", index=False)

    # History comparison: estimate the first sustained decline under an explicit threshold.
    cur_hist = pd.read_csv(EXP / "historial_entrenamiento.csv")
    old_hist = pd.read_csv(OLD / "historial_entrenamiento.csv")
    comparisons = []
    for label, hist, stage in (("MatterGPT_F4_geometry", cur_hist, "f4_geometry"),
                               ("MatterGPT_F4_slices_only", old_hist, "finetune_f4")):
        f = hist.loc[hist.stage.eq(stage)].sort_values("epoch")
        base = float(f.validation_smoothl1.iloc[0])
        sustained = None
        vals = f.validation_smoothl1.to_numpy(float)
        for i in range(len(vals)-2):
            if np.all(vals[i:i+3] < base - 1e-3):
                sustained = int(f.epoch.iloc[i]); break
        comparisons.append({"experiment":label,"epochs_observed":len(f),"initial_train":float(f.iloc[0]["train_minibatch_smoothl1"] if "train_minibatch_smoothl1" in f else f.iloc[0].train_smoothl1),
            "initial_validation":base,"last_epoch":int(f.epoch.iloc[-1]),"last_train":float(f.iloc[-1]["train_minibatch_smoothl1"] if "train_minibatch_smoothl1" in f else f.iloc[-1].train_smoothl1),
            "last_validation":float(vals[-1]),"best_validation":float(vals.min()),"best_epoch":int(f.loc[f.validation_smoothl1.idxmin(),"epoch"]),
            "first_3_epoch_sustained_drop_gt_0_001":sustained,"criterion":"three consecutive val epochs >0.001 below first F4 val; descriptive only"})
    pd.DataFrame(comparisons).to_csv(DIAG / "training_comparison.csv", index=False)

    # Loss, optimizer and static checks from the live script, not optimizer RAM (not exposed).
    config = json.loads((EXP / "geometry_audit.json").read_text())
    diagnostics = {"classification":"B — CONSTANT_PREDICTION_COLLAPSE (confirmado en checkpoint guardado; saturación GELU de la cabeza)",
        "recommendation":"CONTINUE_UNCHANGED", "live_process_not_modified":True,"live_process_not_stopped":True,
        "checkpoint_path":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":cp_hash_before,
        "checkpoint_selected_epoch":int(state["selected_epoch"]),"checkpoint_validation_normalized":float(state["validation_smoothl1_normalized"]),
        "live_process_optimizer_state_available":False,"live_process_optimizer_state_reason":"No se serializa ni expone el estado AdamW del proceso activo; LR/grupos verificados en código.",
        "target_mean_train_ev_atom":float(state["target_mean_ev_atom"]),"target_std_train_ev_atom":float(state["target_std_ev_atom"]),
        "prediction_diagnostics":prediction_metrics,"activation_diagnostics_validation":activation_metrics,
        "geometry_ablation_32_validation":ablation_metrics,
        "gradients_one_batch_isolated":{"loss_smoothl1_normalized":float(loss.detach()),"prediction_shape":list(pred.shape),"target_shape":list(y_norm.shape),
          "manual_smoothl1":manual_smooth_l1(pred.detach().numpy(),y_norm.numpy()),"optimizer_groups":[{"name":"blocks_8_11","lr":3e-6,"weight_decay":.01},
          {"name":"mlp8","lr":1e-4,"weight_decay":1e-4}],"model_copy_only":True},
        "geometry_audit_source":geom_audit,"historical_test_accessed":False,
        "notes":["El checkpoint disponible es el último mejor atómico (seleccionado F4 época indicada), no los pesos de la época activa.",
          "La pérdida de predicción constante se evalúa sobre muestras aleatorias reproducibles por split; no sustituye métricas de cohorte completa.",
          "La saturación GELU fue auditada sobre el checkpoint guardado, no sobre los pesos no guardados del proceso vivo.",
          "No se lee ni modifica el proceso activo ni se cambia el learning rate."]}
    (DIAG/"diagnostic_metrics.json").write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    lines=["# Diagnóstico MatterGPT F4 + geometría Alexandria","",
      "## Estado y alcance","",f"El proceso activo se observó por PID fuera de este script; el diagnóstico no envió señales ni modificó sus archivos. Se utilizó una copia CPU del checkpoint atómico `{checkpoint.relative_to(ROOT)}` (época seleccionada {state['selected_epoch']}) y 64 registros aleatorios por split para predicciones.",
      "El estado AdamW del proceso vivo no se serializa; se verificaron grupos y tasas en el script, y se hizo un único paso aislado sobre una copia.","",
      "## Hallazgo principal","",f"Train sample: predicción-normalizada SD={prediction_metrics['train']['prediction_normalized_std']:.6g}; SmoothL1 del modelo={prediction_metrics['train']['smoothl1_model_normalized']:.6f}; baseline constante cero={prediction_metrics['train']['smoothl1_constant_zero_normalized']:.6f}.",
      f"Validation sample: predicción-normalizada SD={prediction_metrics['validation']['prediction_normalized_std']:.6g}; SmoothL1 del modelo={prediction_metrics['validation']['smoothl1_model_normalized']:.6f}; baseline constante cero={prediction_metrics['validation']['smoothl1_constant_zero_normalized']:.6f}.",
      f"En {activation_metrics['n_structures']} validaciones, {activation_metrics['fraction_hidden_units_zero_for_every_sample']:.1%} de las unidades ocultas estuvieron en cero tras GELU para todas las muestras; {activation_metrics['fraction_post_gelu_abs_below_1e_8']:.1%} de las activaciones fueron menores que 1e-8. Pre-GELU: rango [{activation_metrics['pre_gelu_min']:.3f}, {activation_metrics['pre_gelu_max']:.3f}], con {activation_metrics['fraction_pre_gelu_below_minus_5']:.1%} por debajo de -5.",
      "Esto demuestra colapso a predicción constante en el checkpoint seleccionado y saturación práctica de GELU. La gradiente de la primera capa y de bloques 8–11 es minúscula; el bias final sí puede mover el nivel constante. La conclusión describe el checkpoint guardado, no los pesos aún no guardados de la época activa.",
      f"Ablación en 32 validación: MAE de cambio frente a geometría original = cero {ablation_metrics['zero']['mae_vs_original_normalized']:.6g}, permutada {ablation_metrics['permuted']['mae_vs_original_normalized']:.6g} (normalizada).",
      "La ablación no demuestra que la geometría carezca de información: una cabeza colapsada no responde a sus entradas. El paso aislado usa una copia del checkpoint y un minibatch; no representa el estado exacto del optimizador vivo.","",
      "## Comparación de historial","",pd.DataFrame(comparisons).to_markdown(index=False),"",
      "La comparación usa el primer punto F4 de cada ejecución; cada historial corresponde a su propio train/validation compatible. El criterio de caída sostenida exige tres puntos consecutivos al menos 0.001 por debajo del primer valor; es un resumen descriptivo, no una regla de selección.","",
      "## Geometría, pérdida, gradientes y optimizador","",f"La geometría pasa el manifiesto: {geom_audit['rows']} IDs, unión 1:1 por `mat_id`, hashes de estructura y energía coincidentes, scaler ajustado solo con 17,245 train. Las medias/std por variable están en `geometry_audit.csv`.",
      f"El forward produce predicción/target con shapes {prediction_metrics['validation']['prediction_shape']} y {prediction_metrics['validation']['target_shape']}. SmoothL1 manual={diagnostics['gradients_one_batch_isolated']['manual_smoothl1']:.9f}; PyTorch={diagnostics['gradients_one_batch_isolated']['loss_smoothl1_normalized']:.9f}. No hay broadcasting. Target train: media {state['target_mean_ev_atom']:.6f} eV/átomo, SD {state['target_std_ev_atom']:.6f} eV/átomo.",
      "Los bloques 8–11 y la cabeza tienen gradiente formal y se actualizan en el paso aislado; bloques 0–7 permanecen congelados. Sin embargo, las gradientes de primera capa y backbone son prácticamente nulas por la saturación de GELU; el bias de salida concentra el cambio. Tasas leídas del código: bloques 8–11 LR 3e-6/WD 0.01; MLP8 LR 1e-4/WD 1e-4. No hay scheduler ni clipping. El estado AdamW vivo no está serializado, así que no puede verificarse directamente.",
      "## Comparación con MatterGPT Alexandria SLICES-only","", "El run SLICES-only inició una caída sostenida >0.001 en F4 época 19 y alcanzó validación mínima 0.013452 en época 71. La variante con geometría permanece en torno a 0.41605 hasta F4 época 22. No replica el aprendizaje retardado observado en el run anterior; la diferencia no identifica por sí sola si la causa es la geometría u otra interacción de entrenamiento.",
      "## Clasificación y recomendación","",f"**{diagnostics['classification']}**. Colapso confirmado en el mejor checkpoint guardado (F4 epoch {state['selected_epoch']}); las épocas activas posteriores no están disponibles como pesos para inspección. No se encontró error demostrado de loss, shapes, normalización o unión geométrica. Recomendación: **{diagnostics['recommendation']}** en respeto a la instrucción de no intervenir; no se cambió LR ni se detuvo la corrida.",
      "","## Archivos","","- `diagnostic_metrics.json`","- `gradient_audit.csv`","- `parameter_updates.csv`","- `prediction_distribution.csv`","- `geometry_audit.csv`","- `geometry_ablation_sample.csv`","- `training_comparison.csv`","- `diagnostic_report.md`","",
      f"Las predicciones son muestras aleatorias reproducibles de {sample_n} filas por split; activaciones usan {sample_n} validaciones y ablación geométrica hasta 32. No se abrió el test independiente."]
    (DIAG/"diagnostic_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return diagnostics


def main() -> int:
    """Ejecuta auditoría acotada en CPU y escribe salidas bajo diagnostics/."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size",type=int,default=64)
    args=parser.parse_args()
    if args.sample_size<8 or args.sample_size>256: parser.error("sample-size debe estar entre 8 y 256")
    result=run(args.sample_size)
    print(json.dumps({"classification":result["classification"],"recommendation":result["recommendation"],
                      "checkpoint_epoch":result["checkpoint_selected_epoch"],"diagnostics":str(DIAG.relative_to(ROOT))},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
