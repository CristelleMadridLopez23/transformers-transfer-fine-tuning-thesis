#!/usr/bin/env python3
"""Especializa MatterGPT Alexandria en ABS₃ mediante head warmup, F2/F4 gradual y L2-SP.

El backbone inicial es el checkpoint energético Alexandria F4-71. Ridge M2 y
los targets ABS₃ permanecen fijos; solo se entrenan MLP8 y bloques superiores.
"""
from __future__ import annotations

import argparse
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
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_matter_lpft")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = "mattergpt_alexandria_abs3_lpft_gradual_l2sp"
OUT = ROOT / "experiments" / EXP
IMAGES = ROOT / "images" / EXP
LOGGER = logging.getLogger(EXP)
SEEDS = (42, 123, 2026)
F4_PATH = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/checkpoints/mattergpt_alexandria_f4_energy_best.pt"
F4_SHA = "99da365ea1df8091dc6c16cbf370085409883ff0d72b568d531d78276d926ef3"
BASE_SHA = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
TRANSFER = ROOT / "experiments/matgpt_mattergpt_alexandria_to_abs3_frozen_transfer"
BASELINE_CSV = TRANSFER / "predicciones_oof_por_estructura.csv"
GEOM_COLS = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
LAMBDA_SP = 10.0
EPS_SP = 1e-12


def sha(path: Path) -> str:
    """Calcula SHA-256 en bloques de 1 MiB."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str) -> Any:
    """Carga un script existente sin ejecutar su función principal."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path: Path, value: Any) -> None:
    """Guarda JSON de forma atómica para que el monitor lea estados completos."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_f4_model(matter_train: Any, device: torch.device) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    """Reconstruye MatterGPT F4-71 desde base y aplica bloques 8–11 con carga estricta."""
    if sha(F4_PATH) != F4_SHA:
        raise ValueError("SHA-256 de MatterGPT Alexandria no coincide con F4-71 esperado")
    base_path = ROOT / "models/MatterGPT/pytorch_model.pt"
    if sha(base_path) != BASE_SHA:
        raise ValueError("SHA-256 del checkpoint MatterGPT original no coincide")
    model, stoi, block_size = matter_train.load_model()
    state = torch.load(F4_PATH, map_location="cpu", weights_only=True)
    if int(state.get("selected_epoch", -1)) != 71 or tuple(state.get("condition_vector", ())) != (1.0, 2.0):
        raise ValueError("El artefacto no es el F4-71 condicionado por [1,2]")
    if state.get("base_checkpoint_sha256") != BASE_SHA:
        raise ValueError("F4-71 no declara el checkpoint original esperado")
    current = model.state_dict()
    expected = {k for k in current if any(k.startswith(f"blocks.{i}.") for i in range(8, 12))}
    partial = state["blocks_8_11"]
    if set(partial) != expected:
        missing, extra = sorted(expected - set(partial)), sorted(set(partial) - expected)
        raise ValueError(f"Estado F4 incompleto; faltan={missing[:5]}, extras={extra[:5]}")
    for name, tensor in partial.items():
        if current[name].shape != tensor.shape:
            raise ValueError(f"Dimensión F4 distinta para {name}: {tensor.shape} != {current[name].shape}")
    merged = dict(current); merged.update(partial)
    report = model.load_state_dict(merged, strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise ValueError(f"Reconstrucción strict=True incompleta: {report}")
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {"checkpoint_sha256": sha(F4_PATH), "base_sha256": sha(base_path), "selected_epoch": 71,
            "condition_vector": [1.0, 2.0], "architecture": {"layers": len(model.blocks), "embedding_dim": model.config.n_embd if hasattr(model, "config") else 768,
            "context_length": block_size, "tokenizer_size": len(stoi)}, "updated_tensors": len(partial),
            "full_state_load_strict": True, "backbone_parameter_count": sum(p.numel() for p in model.parameters())}
    return model, base_state, meta


def block_index(name: str) -> int | None:
    """Devuelve índice de bloque Transformer para un nombre de parámetro MatterGPT."""
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def set_trainable(model: torch.nn.Module, blocks: set[int]) -> list[torch.nn.Parameter]:
    """Congela todo MatterGPT y habilita solo bloques indicados."""
    trainable = []
    for name, p in model.named_parameters():
        idx = block_index(name)
        p.requires_grad_(idx in blocks)
        if p.requires_grad:
            trainable.append(p)
    if not trainable:
        raise ValueError("No hay parámetros backbone habilitados")
    return trainable


def l2sp_loss(model: torch.nn.Module, reference: dict[str, torch.Tensor], blocks: set[int], eps: float = EPS_SP) -> torch.Tensor:
    """Promedia por bloque la distancia cuadrática relativa al checkpoint Alexandria."""
    per_block = []
    for block in sorted(blocks):
        terms = []
        for name, p in model.named_parameters():
            if block_index(name) == block and p.requires_grad:
                ref = reference[name].to(device=p.device, dtype=p.dtype)
                terms.append((p - ref).pow(2).sum())
        if not terms:
            raise ValueError(f"L2-SP no encontró parámetros entrenables en bloque {block}")
        numerator = torch.stack(terms).sum()
        denom_terms = [reference[name].to(device=numerator.device, dtype=numerator.dtype).pow(2).sum()
                       for name, p in model.named_parameters() if block_index(name) == block and p.requires_grad]
        denominator = torch.stack(denom_terms).sum() + eps
        per_block.append(numerator / denominator)
    return torch.stack(per_block).mean()


def relative_block_drift(model: torch.nn.Module, reference: dict[str, torch.Tensor], blocks: range = range(8, 12)) -> dict[str, float]:
    """Calcula deriva L2 relativa por bloque frente al checkpoint F4-71."""
    out = {}
    for block in blocks:
        diff_sq = 0.0; ref_sq = 0.0
        for name, p in model.named_parameters():
            if block_index(name) == block:
                cur = p.detach().float().cpu(); ref = reference[name].float()
                diff_sq += float((cur - ref).pow(2).sum()); ref_sq += float(ref.pow(2).sum())
        out[str(block)] = math.sqrt(diff_sq / max(ref_sq, EPS_SP))
    return out


def check_l2sp(model: torch.nn.Module, reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Comprueba cero en referencia, positividad y gradiente finito al desviarse."""
    blocks = {10, 11}; set_trainable(model, blocks)
    zero = float(l2sp_loss(model, reference, blocks).detach())
    if abs(zero) > 1e-12:
        raise ValueError(f"L2-SP no es cero en referencia: {zero}")
    name, p = next((n, p) for n, p in model.named_parameters() if block_index(n) == 10)
    original = p.detach().clone()
    with torch.no_grad():
        p.add_(1e-4)
    sp = l2sp_loss(model, reference, blocks)
    grad = torch.autograd.grad(sp, [p], allow_unused=True)[0]
    with torch.no_grad():
        p.copy_(original)
    if not torch.isfinite(sp) or float(sp.detach()) <= 0 or grad is None or not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
        raise ValueError("Prueba L2-SP positiva/gradiente falló")
    if "head" in name.lower():
        raise AssertionError("L2-SP incluyó la cabeza")
    for p0 in model.parameters():
        p0.requires_grad_(False)
    return {"zero_at_reference": zero, "positive_after_perturbation": float(sp.detach()), "gradient_norm": float(grad.norm()),
            "lambda_sp": LAMBDA_SP, "normalization": "media por bloque de ||θ−θ_A||²/(||θ_A||²+1e-12)"}


def residual_loss(raw: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    """Aplica SmoothL1 a residuos centrados, normalizados por inner-train."""
    centered = raw.reshape(-1, 3) - raw.reshape(-1, 3).mean(dim=1, keepdim=True)
    tgt = target.reshape(-1, 3)
    return F.smooth_l1_loss(centered / scale, tgt / scale, beta=1.0)


class LPFTExperiment:
    """Orquesta cachés, warmup compartido, ramas F2/F4 y evaluación OOF."""
    def __init__(self, device: torch.device, preflight_only: bool = False) -> None:
        """Carga fuentes y prepara todas las estructuras de desarrollo validadas."""
        self.device = device; self.preflight_only = preflight_only
        self.m2 = load_module(ROOT / "scripts/MatGPT/MatGPT_F0_Stoichiometric_Ridge_M2_Comparison/run_experiment.py", "matter_lpft_m2")
        self.mg = self.m2.baseline_module()
        self.data, self.folds, _, self.geom, self.ridges, self.mu, self.ridge_audit = self.m2.load_and_preflight(self.mg)
        self.scalers = self.m2.fit_scalers(self.data, self.folds, self.geom)
        self.geom_by_fold = {f: self.scalers[f].transform(self.geom).astype(np.float32) for f in range(5)}
        self.train_module = load_module(ROOT / "scripts/MatterGPT/MatterGPT_Alexandria_Energy_Transfer_20k_F4_Compatible/train_energy.py", "matter_lpft_train")
        self.model, self.base_state, self.checkpoint_meta = load_f4_model(self.train_module, device)
        self.reference = {n: p.detach().clone() for n, p in self.model.named_parameters() if block_index(n) in range(8,12)}
        self.sequences = self._encode()
        self.groups = {f: np.flatnonzero(self.data.composition_id.to_numpy() == f) for f in sorted(self.data.composition_id.unique())}
        self.features = self._load_validated_embedding_cache()
        self.scales = self._residual_scales()
        self.baseline, self.baseline_metrics = self._baseline()

    def _encode(self) -> list[list[int]]:
        """Codifica SLICES con tokenizer MatterGPT y rechaza OOV/truncamiento."""
        frame = self.data.copy(); frame["slices_matgpt"] = frame.slices.astype(str)
        frame["matter_sequence_length"] = frame.slices_matgpt.map(lambda s: len(s.split()) + 2)
        seq = self.train_module.read_sequences(frame, self.train_module.load_model()[1], self.checkpoint_meta["architecture"]["context_length"])
        if len(seq) != 210:
            raise ValueError("MatterGPT no codificó las 210 estructuras")
        return seq

    def _load_validated_embedding_cache(self) -> np.ndarray:
        """Carga FINAL_END Alexandria usado en G1-G y exige procedencia/checkpoint concordantes."""
        manifest_path = TRANSFER / "embedding_manifest.json"
        cache_path = TRANSFER / "embeddings/G1_FINAL_END.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = manifest["MatterGPT_F4"]
        if meta["checkpoint_sha256"] != F4_SHA or meta["condition"] != [1.0, 2.0] or meta["extraction"] != "terminal structural token posterior a bloques y ln_f":
            raise ValueError("Caché G1 FINAL_END no corresponde a MatterGPT F4/[1,2]/post-ln_f")
        # The first producer serialized pandas object strings; IDs are inspected as strings only.
        z = np.load(cache_path, allow_pickle=True)
        ids = np.asarray(z["structure_id"]).astype(str); feat = z["FINAL_END"]
        lookup = {sid:i for i,sid in enumerate(ids)}
        if feat.shape != (210,768) or not np.isfinite(feat).all() or set(ids) != set(self.data.structure_id):
            raise ValueError("Caché G1 no válida 210×768")
        # Confirm the exact terminal-token sequence is stable across model states (tokenizer is fixed).
        token_hash = hashlib.sha256(json.dumps(self.sequences, separators=(",", ":")).encode()).hexdigest()
        if meta["input_ids_sha256"] != token_hash:
            raise ValueError("Input IDs ABS3 difieren de los utilizados para el embedding G1")
        return np.stack([feat[lookup[s]] for s in self.data.structure_id]).astype(np.float32)

    def _residual_scales(self) -> dict[int, float]:
        """Calcula escala residual solo en 45 fórmulas inner-train por fold."""
        means = self.data.groupby("composition_id").energy_dft_ev_atom.transform("mean").to_numpy(float)
        centered = self.data.energy_dft_ev_atom.to_numpy(float) - means; out = {}
        for fold in range(5):
            roles = self.folds[self.folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
            tr = set(roles.loc[roles.inner_role.eq("inner_train"), "composition_group"])
            ix = np.flatnonzero(self.data.composition_id.isin(tr).to_numpy())
            if len(tr) != 45 or len(ix) != 135:
                raise ValueError(f"Inner-train incorrecto en fold {fold}")
            out[fold] = float(centered[ix].std(ddof=0))
            if not math.isfinite(out[fold]) or out[fold] <= 0:
                raise ValueError(f"Escala residual inválida fold {fold}")
        return out

    def _baseline(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Verifica y carga G1-G sin usar sus resultados para seleccionar checkpoints."""
        metrics = json.loads((TRANSFER / "metrics.json").read_text(encoding="utf-8"))
        row = next(r for r in metrics["estimand_A"] if r["model"] == "G1-G")
        expected = {"mae_EN_mev_atom":48.4366,"mae_residual_mev_atom":28.4059,"mae_delta_signed_mev_atom":48.0401,
                    "rmse_delta_global_mev_atom":69.1215,"ground_state_accuracy":0.577114,"pairwise_accuracy_eligible":0.695431,
                    "signed_pair_ols_slope":0.3076,"signed_pair_ols_r2":0.3572}
        for key, value in expected.items():
            if abs(float(row[key]) - value) > 0.001:
                raise ValueError(f"G1-G referencia {key} no coincide: {row[key]} vs {value}")
        pred = pd.read_csv(BASELINE_CSV); pred = pred[pred.model.eq("G1-G")].copy()
        if len(pred) != 630:
            raise ValueError(f"Baseline G1-G no contiene 630 estructura-fold-semilla; hay {len(pred)}")
        pred["model"] = "C0"
        return pred, row

    def _raw(self, model: torch.nn.Module, head: torch.nn.Module, ids: torch.Tensor, pos: torch.Tensor,
             geom: torch.Tensor, use_cache: bool = False, cached: torch.Tensor | None = None) -> torch.Tensor:
        """Predice residuo bruto con FINAL_END contextualizado o caché de warmup."""
        h = cached if use_cache else self.train_module.contextualized_final_end(model, ids, pos)
        return head(torch.cat([h, geom], dim=1))

    def _evaluate(self, model: torch.nn.Module, head: torch.nn.Module, formulas: list[str], fold: int,
                  scale: float, *, cache: bool) -> float:
        """Calcula SmoothL1 completa, ponderada por estructura, sobre fórmulas dadas."""
        model.eval(); head.eval(); total = 0.0; count = 0
        with torch.inference_mode():
            for start in range(0,len(formulas),8):
                fs=formulas[start:start+8]; inds=np.concatenate([self.groups[f] for f in fs]).astype(int)
                ids,pos=self.train_module.make_batch(self.sequences,inds); ids=ids.to(self.device); pos=pos.to(self.device)
                geom=torch.as_tensor(self.geom_by_fold[fold][inds],dtype=torch.float32,device=self.device)
                if cache: h=torch.as_tensor(self.features[inds],dtype=torch.float32,device=self.device)
                else: h=self.train_module.contextualized_final_end(model,ids,pos)
                raw=head(torch.cat([h,geom],dim=1)); raw=raw.reshape(-1,3); raw=raw-raw.mean(dim=1,keepdim=True)
                en=self.data.energy_dft_ev_atom.to_numpy(float)[inds].reshape(-1,3); tgt=en-en.mean(axis=1,keepdims=True)
                y=torch.as_tensor(tgt.reshape(-1),dtype=torch.float32,device=self.device)
                loss=F.smooth_l1_loss(raw.reshape(-1)/scale,y/scale,beta=1.0,reduction="sum")
                total+=float(loss);count+=len(y)
        return total/count

    def _train_head_warmup(self, fold: int, seed: int, train_f: list[str], val_f: list[str]) -> tuple[torch.nn.Module, dict[str,torch.Tensor], list[dict[str,Any]], dict[str,Any]]:
        """Entrena una sola MLP8 sobre embeddings congelados y guarda su mejor validación."""
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        head=self.m2.Head1().to(self.device); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
        rng=np.random.default_rng(seed); best=float("inf"); best_state=None; best_epoch=0; stale=0; history=[]
        scale=self.scales[fold]
        for epoch in range(1,21):
            head.train(); order=np.asarray(train_f)[rng.permutation(len(train_f))]; total=0.; nobs=0
            for start in range(0,len(order),8):
                fs=list(order[start:start+8]); inds=np.concatenate([self.groups[f] for f in fs]).astype(int)
                h=torch.as_tensor(self.features[inds],dtype=torch.float32,device=self.device)
                g=torch.as_tensor(self.geom_by_fold[fold][inds],dtype=torch.float32,device=self.device)
                en=self.data.energy_dft_ev_atom.to_numpy(float)[inds].reshape(-1,3); target=en-en.mean(axis=1,keepdims=True)
                y=torch.as_tensor(target.reshape(-1),dtype=torch.float32,device=self.device)
                pred=head(torch.cat([h,g],dim=1)); loss=residual_loss(pred,y,scale)
                opt.zero_grad(set_to_none=True); loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise FloatingPointError("Gradiente inválido en warmup MLP8")
                opt.step(); total+=float(loss.detach())*len(y);nobs+=len(y)
            val=self._evaluate(self.model,head,val_f,fold,scale,cache=True)
            train=total/nobs; history.append({"fold":fold,"seed":seed,"condition":"warmup_shared","stage":"head_warmup","epoch":epoch,"train_minibatch_smoothl1":train,"inner_validation_smoothl1":val,"loss_data":train,"loss_sp":0.0,"loss_total":train,"lambda_sp":0.0})
            atomic_json(OUT/"progress.json",{"status":"RUNNING","stage":"head_warmup","fold":fold,"seed":seed,"epoch":epoch,"max_epochs":20,"train_smoothl1":train,"validation_smoothl1":val})
            LOGGER.info("warmup fold=%d seed=%d epoch=%d train=%.6f val=%.6f",fold,seed,epoch,train,val)
            if val < best-1e-12:
                best,best_epoch,best_state,stale=val,epoch,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},0
            else:
                stale+=1
                if stale>=8: break
        if best_state is None:
            raise RuntimeError("Warmup no produjo checkpoint seleccionado")
        head.load_state_dict(best_state,strict=True)
        info={"best_epoch":best_epoch,"epochs_run":epoch,"best_inner_validation_smoothl1":best,
              "inner_train_full_eval_smoothl1":self._evaluate(self.model,head,train_f,fold,scale,cache=True),"shared_across_branches":True}
        return head,best_state,history,info

    def _epoch(self, model: torch.nn.Module, head: torch.nn.Module, opt: torch.optim.Optimizer,
               trainable: list[torch.nn.Parameter], ref: dict[str,torch.Tensor], blocks: set[int],
               train_f: list[str], fold: int, scale: float, *, lambda_sp: float, seed: int,
               epoch: int, phase: str) -> tuple[float, dict[str,float]]:
        """Ejecuta una época de composición-completa con forward differentiable MatterGPT."""
        model.train();head.train();rng=np.random.default_rng(seed+epoch*997);order=np.asarray(train_f)[rng.permutation(len(train_f))]
        total_data=0.;total_sp=0.;nobs=0;grad_info={"grad_data_norm":0.,"grad_sp_norm":0.,"grad_ratio_sp_data":0.}
        for batch_ix,start in enumerate(range(0,len(order),8)):
            fs=list(order[start:start+8]); inds=np.concatenate([self.groups[f] for f in fs]).astype(int)
            ids,pos=self.train_module.make_batch(self.sequences,inds);ids=ids.to(self.device);pos=pos.to(self.device)
            geom=torch.as_tensor(self.geom_by_fold[fold][inds],dtype=torch.float32,device=self.device)
            en=self.data.energy_dft_ev_atom.to_numpy(float)[inds].reshape(-1,3);target=en-en.mean(axis=1,keepdims=True)
            y=torch.as_tensor(target.reshape(-1),dtype=torch.float32,device=self.device)
            # No no_grad/detach/cache is used: autograd crosses the trainable upper blocks.
            pred=self.train_module.contextualized_final_end(model,ids,pos)
            raw=head(torch.cat([pred,geom],dim=1));loss_data=residual_loss(raw,y,scale)
            loss_sp=l2sp_loss(model,ref,blocks) if lambda_sp else loss_data.new_zeros(())
            loss_total=loss_data+lambda_sp*loss_sp
            if not torch.isfinite(loss_total):
                raise FloatingPointError(f"Loss no finita fase {phase} fold={fold}, seed={seed}, época={epoch}")
            if lambda_sp and batch_ix == 0:
                gd=torch.autograd.grad(loss_data,trainable,retain_graph=True,allow_unused=True)
                gs=torch.autograd.grad(loss_sp,trainable,retain_graph=True,allow_unused=True)
                nd=math.sqrt(sum(float(g.pow(2).sum()) for g in gd if g is not None)); ns=math.sqrt(sum(float(g.pow(2).sum()) for g in gs if g is not None))
                grad_info={"grad_data_norm":nd,"grad_sp_norm":ns,"grad_ratio_sp_data":ns/max(nd,1e-30)}
            opt.zero_grad(set_to_none=True);loss_total.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                raise FloatingPointError(f"Gradiente backbone ausente/no finito en {phase}")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError(f"Gradiente MLP8 ausente/no finito en {phase}")
            opt.step(); total_data+=float(loss_data.detach())*len(y);total_sp+=float(loss_sp.detach())*len(y);nobs+=len(y)
        grad_info.update({"loss_data":total_data/nobs,"loss_sp":total_sp/nobs,"loss_total":total_data/nobs+lambda_sp*total_sp/nobs})
        return total_data/nobs,grad_info

    def _phase(self, model: torch.nn.Module, head: torch.nn.Module, fold: int, seed: int, train_f: list[str], val_f: list[str],
               phase: str, blocks: set[int], max_epochs: int, patience: int, lr_backbone: dict[int,float],
               lambda_sp: float, history: list[dict[str,Any]], *, branch: str, start_epoch: int = 0) -> tuple[dict[str,torch.Tensor],dict[str,torch.Tensor],dict[str,Any]]:
        """Entrena fase parcialmente descongelada y selecciona solo por validation data loss."""
        trainable=set_trainable(model,blocks);head_params=list(head.parameters())
        groups=[]
        for block,lr in sorted(lr_backbone.items()):
            ps=[p for n,p in model.named_parameters() if block_index(n)==block]
            if block in blocks and ps:
                groups.append({"params":ps,"lr":lr,"weight_decay":1e-4,"name":f"blocks.{block}"})
        if {block_index(n) for n,p in model.named_parameters() if p.requires_grad} != blocks:
            raise ValueError("Los bloques entrenables no coinciden con la fase")
        groups.append({"params":head_params,"lr":1e-4,"weight_decay":1e-4,"name":"mlp8"})
        # Reset optimizer at phase boundaries, including when F4 adds blocks 8–9.
        opt=torch.optim.AdamW(groups)
        best=float("inf");best_model=None;best_head=None;best_epoch=0;stale=0;scale=self.scales[fold]
        rng_seed=seed+(10000 if phase.endswith("F2") else 20000)
        torch.manual_seed(rng_seed)
        for local_epoch in range(1,max_epochs+1):
            data_loss,diag=self._epoch(model,head,opt,trainable,self.reference,blocks,train_f,fold,scale,
                                       lambda_sp=lambda_sp,seed=rng_seed,epoch=local_epoch,phase=phase)
            val=self._evaluate(model,head,val_f,fold,scale,cache=False)
            train_eval=self._evaluate(model,head,train_f,fold,scale,cache=False)
            history.append({"fold":fold,"seed":seed,"condition":branch,"stage":phase,"epoch":start_epoch+local_epoch,
                "local_epoch":local_epoch,"train_minibatch_smoothl1":data_loss,"train_full_eval_smoothl1":train_eval,
                "inner_validation_smoothl1":val,"loss_data":diag["loss_data"],"loss_sp":diag["loss_sp"],"loss_total":diag["loss_total"],
                "lambda_sp":lambda_sp,"grad_data_norm":diag["grad_data_norm"],"grad_sp_norm":diag["grad_sp_norm"],
                "grad_ratio_sp_data":diag["grad_ratio_sp_data"],"active_blocks":','.join(map(str,sorted(blocks))),
                **{f"drift_block_{b}":v for b,v in relative_block_drift(model,self.reference).items()}})
            atomic_json(OUT/"progress.json",{"status":"RUNNING","stage":phase,"condition":branch,"fold":fold,"seed":seed,
                "epoch":local_epoch,"max_epochs":max_epochs,"active_blocks":sorted(blocks),"validation_smoothl1":val,
                "loss_data":diag["loss_data"],"loss_sp":diag["loss_sp"],"lambda_sp":lambda_sp,"best_validation_so_far":min(best,val)})
            LOGGER.info("%s %s fold=%d seed=%d epoch=%d/%d data=%.6f val=%.6f Lsp=%.6g",branch,phase,fold,seed,local_epoch,max_epochs,diag["loss_data"],val,diag["loss_sp"])
            if val < best-1e-12:
                best=val;best_epoch=start_epoch+local_epoch
                best_model={n:p.detach().cpu().clone() for n,p in model.state_dict().items() if block_index(n) in blocks}
                best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0
            else:
                stale+=1
                if stale>=patience:break
        if best_model is None or best_head is None:
            raise RuntimeError(f"Fase {phase} no guardó checkpoint")
        info={"best_epoch":best_epoch,"epochs_run":local_epoch,"best_validation_data_loss":best,
              "train_full_eval_at_selected":None,"active_blocks":sorted(blocks),
              "lambda_sp":lambda_sp,"optimizer_groups": [{"name":g.get("name"),"lr":g["lr"],"params":sum(p.numel() for p in g["params"])} for g in groups]}
        return best_model,best_head,info

    def _restore_phase(self, model: torch.nn.Module, head: torch.nn.Module, block_state: dict[str,torch.Tensor], head_state: dict[str,torch.Tensor]) -> None:
        """Carga parámetros parciales elegidos sin alterar otros bloques/base."""
        state=model.state_dict();state.update(block_state)
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:raise ValueError("No se pudo restaurar checkpoint parcial")
        head.load_state_dict(head_state,strict=True)

    def _save_checkpoint(self, path: Path, model_state: dict[str,torch.Tensor], head_state: dict[str,torch.Tensor],
                         fold:int,seed:int,condition:str,phase_info:dict[str,Any],blocks:set[int]) -> str:
        """Guarda solo los bloques superiores cambiados más MLP8 y metadatos."""
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".pt.tmp")
        torch.save({"base_checkpoint_sha256":F4_SHA,"fold":fold,"seed":seed,"condition":condition,
                    "selected_validation":phase_info,"active_blocks":sorted(blocks),"block_state_dict":model_state,
                    "head_state_dict":head_state,"l2sp_reference":"Alexandria F4-71"},tmp);os.replace(tmp,path)
        return sha(path)

    def _predict_outer(self, model:torch.nn.Module,head:torch.nn.Module,fold:int,seed:int,condition:str,outer:list[str]) -> pd.DataFrame:
        """Calcula OOF outer solo tras seleccionar la cabeza con inner-validation."""
        model.eval();head.eval();rows=[]
        with torch.inference_mode():
            for formula in outer:
                inds=self.groups[formula];ids,pos=self.train_module.make_batch(self.sequences,inds);ids=ids.to(self.device);pos=pos.to(self.device)
                geom=torch.as_tensor(self.geom_by_fold[fold][inds],dtype=torch.float32,device=self.device)
                hidden=self.train_module.contextualized_final_end(model,ids,pos);raw=head(torch.cat([hidden,geom],dim=1));delta=raw-raw.mean()
                if abs(float(delta.sum()))>1e-6:raise ValueError("Residuo ABS3 no centrado")
                for j,idx in enumerate(inds):
                    r=self.data.iloc[idx];mu=self.mu[(fold,formula)]
                    rows.append({"model":condition,"fold":fold,"seed":seed,"structure_id":r.structure_id,"composition_id":formula,
                      "family":r.family,"n_atoms":int(r.n_atoms),"energy_dft_ev_atom":float(r.energy_dft_ev_atom),
                      "ridge_mean_ev_atom":mu,"residual_pred_ev_atom":float(delta[j]),"energy_pred_ev_atom":mu+float(delta[j])})
        return pd.DataFrame(rows)

    def _smoke_tests(self) -> dict[str,Any]:
        """Ejecuta backward end-to-end F2/F4 y audita pesos congelados y L2-SP."""
        formula=sorted(self.groups)[0];inds=self.groups[formula];ids,pos=self.train_module.make_batch(self.sequences,inds)
        ids=ids.to(self.device);pos=pos.to(self.device);geom=torch.as_tensor(self.geom_by_fold[0][inds],dtype=torch.float32,device=self.device)
        head=self.m2.Head1().to(self.device)
        checks={}
        for label,blocks in [("F2",{10,11}),("F4",{8,9,10,11})]:
            self.model.load_state_dict(self.base_state,strict=True);self.model.to(self.device);self.model.eval()
            for p in self.model.parameters():p.requires_grad_(False)
            frozen={n:p.detach().clone() for n,p in self.model.named_parameters() if block_index(n) not in blocks}
            trainable=set_trainable(self.model,blocks);head.apply(lambda m: m.reset_parameters() if isinstance(m,torch.nn.Linear) else None)
            self.model.train();head.train();h=self.train_module.contextualized_final_end(self.model,ids,pos)
            raw=head(torch.cat([h,geom],dim=1));energy=self.data.energy_dft_ev_atom.to_numpy(float)[inds];target=energy-energy.mean()
            loss=residual_loss(raw,torch.as_tensor(target,dtype=torch.float32,device=self.device),self.scales[0])
            sp=l2sp_loss(self.model,self.reference,blocks);total=loss+LAMBDA_SP*sp
            total.backward()
            block_grad={str(b):sum(float(p.grad.norm()) for n,p in self.model.named_parameters() if block_index(n)==b and p.grad is not None) for b in range(12)}
            if any(block_grad[str(b)]<=0 for b in blocks) or any(block_grad[str(b)]!=0 for b in set(range(12))-blocks):
                raise ValueError(f"Smoke {label}: gradientes inesperados {block_grad}")
            opt=torch.optim.AdamW([{"params":trainable,"lr":1e-6},{"params":head.parameters(),"lr":1e-4}]);opt.step()
            changed_frozen=[n for n,p in self.model.named_parameters() if n in frozen and not torch.equal(p.detach(),frozen[n])]
            if changed_frozen:raise ValueError(f"Smoke {label}: cambiaron pesos congelados: {changed_frozen[:3]}")
            checks[label]={"status":"PASS","active_blocks":sorted(blocks),"nonzero_gradient_blocks":sorted(int(b) for b,v in block_grad.items() if v>0),
                           "frozen_parameters_unchanged":True,"finite_loss":bool(torch.isfinite(total))}
            for p in self.model.parameters():p.requires_grad_(False)
        self.model.load_state_dict(self.base_state,strict=True);self.model.to(self.device).eval()
        l2=check_l2sp(self.model,self.reference)
        # Ensure warmup cache reproduces one G1-G embedding to numerical precision.
        with torch.inference_mode():
            self.model.eval();fresh=self.train_module.contextualized_final_end(self.model,ids,pos).cpu().numpy()
        cached=self.features[inds]
        cache_err=float(np.max(np.abs(fresh-cached)))
        if cache_err>2e-4:raise ValueError(f"Cache G1-G no reproduce forward de F4: {cache_err}")
        checks.update({"l2sp":l2,"mattergpt_cache_max_abs_error_one_triplet":cache_err,"historical_test_accessed":False})
        return checks

    def _one_fold_seed(self, fold:int,seed:int) -> tuple[list[pd.DataFrame],list[dict[str,Any]],list[dict[str,Any]],dict[str,Any]]:
        """Comparte warmup y bifurca F2 ± L2-SP a F2 o F2→F4."""
        roles=self.folds[self.folds.outer_fold.eq(fold)].drop_duplicates("composition_group").set_index("composition_group")
        train_f=sorted(roles.index[roles.inner_role.eq("inner_train")]);val_f=sorted(roles.index[roles.inner_role.eq("inner_validation")])
        outer=sorted(self.folds.loc[(self.folds.outer_fold.eq(fold))&self.folds.outer_role.eq("outer_validation"),"composition_group"].unique())
        if (len(train_f),len(val_f),len(outer))!=(45,11,14):raise ValueError(f"Split inesperado fold {fold}")
        head,warm_state,warm_hist,warm_info=self._train_head_warmup(fold,seed,train_f,val_f)
        for p in self.model.parameters():p.requires_grad_(False)
        # Branches have identical starting weights, order and optimizer settings within each regularization pair.
        f2_results={};history=list(warm_hist);records=[];predictions=[]
        for sp_on in (False,True):
            name="C2" if sp_on else "C1"; lam=LAMBDA_SP if sp_on else 0.0
            self.model.load_state_dict(self.base_state,strict=True);self.model.to(self.device)
            h=self.m2.Head1().to(self.device);h.load_state_dict(warm_state,strict=True)
            bstate,hstate,info=self._phase(self.model,h,fold,seed,train_f,val_f,"F2",{10,11},30,10,{10:1e-6,11:1e-6},lam,history,branch=name)
            self._restore_phase(self.model,h,bstate,hstate)
            info["train_full_eval_at_selected"]=self._evaluate(self.model,h,train_f,fold,self.scales[fold],cache=False)
            f2_results[sp_on]=(bstate,hstate,info)
            f2path=OUT/"checkpoints"/name/f"fold_{fold}_seed_{seed}_F2.pt"
            f2sha=self._save_checkpoint(f2path,bstate,hstate,fold,seed,name,info,{10,11})
            records.append({"condition":name,"fold":fold,"seed":seed,"phase":"F2_selected","checkpoint":str(f2path.relative_to(ROOT)),"sha256":f2sha,"best_epoch":info["best_epoch"],"best_validation":info["best_validation_data_loss"],"blocks":info["active_blocks"]})
        for sp_on in (False,True):
            bstate,hstate,f2info=f2_results[sp_on]
            for gradual in (False,True):
                name="C4" if (sp_on and gradual) else "C3" if gradual else "C2" if sp_on else "C1"
                blocks={8,9,10,11} if gradual else {10,11}
                lr={10:1e-6,11:1e-6,8:5e-7,9:5e-7};lam=LAMBDA_SP if sp_on else 0.
                self.model.load_state_dict(self.base_state,strict=True);self.model.to(self.device)
                h=self.m2.Head1().to(self.device);self._restore_phase(self.model,h,bstate,hstate)
                start_epoch=int(f2info["best_epoch"])
                phase_name="F4" if gradual else "F2_continue"
                cstate,chstate,cinfo=self._phase(self.model,h,fold,seed,train_f,val_f,phase_name,blocks,50,12,lr,lam,history,branch=name,start_epoch=start_epoch)
                self._restore_phase(self.model,h,cstate,chstate)
                cinfo["train_full_eval_at_selected"]=self._evaluate(self.model,h,train_f,fold,self.scales[fold],cache=False)
                cinfo["loss_sp_selected"]=float(l2sp_loss(self.model,self.reference,blocks).detach()) if lam else 0.0
                cinfo["relative_block_drift_selected"]=relative_block_drift(self.model,self.reference,range(8,12))
                modelpath=OUT/"checkpoints"/name/f"fold_{fold}_seed_{seed}_selected.pt"
                csha=self._save_checkpoint(modelpath,cstate,chstate,fold,seed,name,cinfo,blocks)
                records.append({"condition":name,"fold":fold,"seed":seed,"phase":"selected_final","checkpoint":str(modelpath.relative_to(ROOT)),"sha256":csha,"best_epoch":cinfo["best_epoch"],"best_validation":cinfo["best_validation_data_loss"],"blocks":cinfo["active_blocks"],"warmup":warm_info,"f2":f2info,"final":cinfo})
                predictions.append(self._predict_outer(self.model,h,fold,seed,name,outer))
                for p in self.model.parameters():p.requires_grad_(False)
        diag={"fold":fold,"seed":seed,"warmup":warm_info,"f2_no_l2sp":f2_results[False][2],"f2_l2sp":f2_results[True][2]}
        return predictions,history,records,{"pilot":diag}

    def execute(self) -> None:
        """Valida, ejecuta piloto fold0-seed42 y completa 60 salidas OOF."""
        OUT.mkdir(parents=True,exist_ok=True);(OUT/"checkpoints").mkdir(exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
        smoke=self._smoke_tests()
        audit={"status":"PASS","checkpoint":self.checkpoint_meta,"ridge_sha256":self.ridge_audit["ridge_m2_sha256"],
               "ridge_fit_called":False,"geometry_columns":GEOM_COLS,"geometry_scaler_fit":"45 composiciones inner-train, por fold",
               "dataset":{"structures":210,"compositions":70,"polymorphs_per_composition":3,"split_counts":[45,11,14],"folds":5,"seeds":list(SEEDS)},
               "baseline_G1-G_estimand_A":self.baseline_metrics,"smoke_tests":smoke,"historical_test_accessed":False,
               "training_protocol":{"warmup_epochs":20,"warmup_patience":8,"f2_epochs":30,"f2_patience":10,"additional_epochs":50,"phase_c_patience":12,
                 "lr_backbone_10_11":1e-6,"lr_backbone_8_9":5e-7,"lr_head":1e-4,"lambda_sp":LAMBDA_SP,"epsilon_sp":EPS_SP,
                 "batch_compositions":8,"selection":"menor SmoothL1 de datos en inner-validation, sin término L2-SP"}}
        atomic_json(OUT/"preflight.json",audit)
        cfg={"experiment":EXP,"method":"LP-FT-style: warmup de MLP8 no lineal + fine-tuning parcial gradual",
             "alexandria_checkpoint_sha256":F4_SHA,"alexandria_selected_epoch":71,"mattergpt_base_sha256":BASE_SHA,
             "property_vector":[1.0,2.0],"geometry_columns":GEOM_COLS,"geometry_source":"gga-static_endpoint_cleaned",
             "ridge":"M2; 17 fracciones; cinco pipelines históricos; no fit","conditions":{"C0":"G1-G congelado reutilizado","C1":"warmup + F2 sin L2-SP","C2":"warmup + F2 con L2-SP","C3":"warmup + F2→F4 sin L2-SP","C4":"warmup + F2→F4 con L2-SP"},
             "l2sp":{"reference":"MatterGPT Alexandria F4-71","formula":"mean_blocks(sum((theta-theta_A)^2)/(sum(theta_A^2)+1e-12))","lambda":LAMBDA_SP,"epsilon":EPS_SP,"excluded":"MLP8, Ridge, scaler, frozen blocks and buffers"},
             "seed_pairing":"one shared warmup; same phase RNG/order inside each L2-SP pair","historical_test_accessed":False}
        (OUT/"config.yaml").write_text(__import__("yaml").safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
        if self.preflight_only:
            atomic_json(OUT/"progress.json",{"status":"PREFLIGHT_COMPLETED","completed_final_runs":0,"total_final_runs":60})
            return
        start=time.time();cpu0=time.process_time();all_pred=[self.baseline];all_hist=[];all_records=[];pilot=None;done=0
        run_order=[(0,42)]+[(f,s) for f in range(5) for s in SEEDS if (f,s)!=(0,42)]
        for fold,seed in run_order:
            if done==0:
                atomic_json(OUT/"progress.json",{"status":"PILOT_RUNNING","completed_final_runs":0,"total_final_runs":60,"fold":fold,"seed":seed})
            preds,hist,recs,extra=self._one_fold_seed(fold,seed)
            if (fold,seed)==(0,42):pilot=extra["pilot"]
            all_pred.extend(preds);all_hist.extend(hist);all_records.extend(recs);done+=4
            pd.concat(all_pred,ignore_index=True).to_csv(OUT/"predicciones_oof_por_estructura.csv",index=False)
            pd.DataFrame(all_hist).to_csv(OUT/"historial_entrenamiento.csv",index=False)
            pd.DataFrame(all_records).to_csv(OUT/"checkpoint_manifest.csv",index=False)
            atomic_json(OUT/"progress.json",{"status":"PILOT_PASSED" if (fold,seed)==(0,42) else "RUNNING","completed_final_runs":done,"total_final_runs":60,"current_fold":fold,"current_seed":seed,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            atomic_json(OUT/"pilot.json",{"status":"PASS","fold":fold,"seed":seed,"diagnostics":pilot}) if pilot else None
            LOGGER.info("Completado fold=%s seed=%s; ramas finales acumuladas %d/60",fold,seed,done)
        pred=pd.concat(all_pred,ignore_index=True)
        expected={"C0","C1","C2","C3","C4"}
        if set(pred.model.unique())!=expected or pred.groupby("model").size().to_dict()!={"C0":630,"C1":630,"C2":630,"C3":630,"C4":630}:
            raise ValueError("OOF final debe contener 630 filas por C0–C4")
        fs,pairs,pack=self.m2.metrics_for(pred,self.mg);metrics=pd.DataFrame(pack["rows"])
        fs.to_csv(OUT/"metricas_por_fold_seed.csv",index=False);pairs.to_csv(OUT/"predicciones_oof_por_par.csv",index=False)
        fs.groupby(["model","composition_id"],as_index=False).mean(numeric_only=True).to_csv(OUT/"metricas_por_composicion.csv",index=False)
        boots=self._bootstrap(pack["formula_seed"]);boots.to_csv(OUT/"bootstrap_apareado.csv",index=False)
        l2rows=[]
        for r in all_records:
            if r["phase"]!="selected_final":continue
            l2rows.append({"condition":r["condition"],"fold":r["fold"],"seed":r["seed"],"lambda_sp":LAMBDA_SP if r["condition"] in {"C2","C4"} else 0.,
              "active_blocks":','.join(map(str,r["blocks"])),"loss_sp_selected":r["final"].get("loss_sp_selected",0.),
              "drift_by_block":json.dumps(r["final"].get("relative_block_drift_selected",{})),
              "gradient_diagnostics_available":bool(any(x.get("fold")==r["fold"] and x.get("seed")==r["seed"] and x.get("stage")==("F4" if r["condition"] in {"C3","C4"} else "F2_continue") and x.get("grad_sp_norm",0)>0 for x in all_hist))})
        pd.DataFrame(l2rows).to_csv(OUT/"l2sp_diagnostics.csv",index=False)
        drift=self._final_embedding_drift(pred,all_records)
        drift.to_csv(OUT/"embedding_drift.csv",index=False)
        figures=self._figures(pred,metrics,boots,pd.DataFrame(all_hist),drift)
        resources={"status":"COMPLETED","elapsed_wall_seconds":time.time()-start,"process_cpu_seconds":time.process_time()-cpu0,
          "max_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform=="darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
          "device":str(self.device),"torch_threads":torch.get_num_threads(),"mps_driver_allocated_bytes":int(torch.mps.driver_allocated_memory()) if self.device.type=="mps" else 0,
          "new_final_condition_runs":60,"shared_warmups":15,"baseline_runs_reused":15,"historical_test_accessed":False,
          "checkpoint_sha256":{"MatterGPT_F4_71":F4_SHA,"MatterGPT_original":BASE_SHA}}
        atomic_json(OUT/"resources.json",resources)
        a=metrics[metrics.estimand.eq("A")]
        atomic_json(OUT/"metrics.json",{"status":"PASS","estimand_A":a.to_dict("records"),"estimand_B":metrics[metrics.estimand.eq("B")].to_dict("records"),
          "contrasts":boots.to_dict("records"),"historical_test_accessed":False,"figures":figures})
        atomic_json(OUT/"summary.json",{"status":"PASS","conditions":sorted(expected),"final_runs":60,"baseline":"G1-G reused","primary_metric":"MAE ΔE firmado estimando A","figures":figures})
        self._report(metrics,boots,resources,figures)
        atomic_json(OUT/"progress.json",{"status":"COMPLETED","completed_final_runs":60,"total_final_runs":60,"finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    def finalize_existing(self) -> None:
        """Completa análisis/reportes desde OOF y checkpoints ya guardados, sin entrenar."""
        prediction_path=OUT/"predicciones_oof_por_estructura.csv"
        history_path=OUT/"historial_entrenamiento.csv"
        manifest_path=OUT/"checkpoint_manifest.csv"
        if not all(p.is_file() for p in (prediction_path,history_path,manifest_path)):
            raise FileNotFoundError("Faltan predicciones, historial o checkpoints guardados; no se puede finalizar sin entrenar")
        pred=pd.read_csv(prediction_path);history=pd.read_csv(history_path);records=pd.read_csv(manifest_path)
        if pred.groupby("model").size().to_dict()!={"C0":630,"C1":630,"C2":630,"C3":630,"C4":630}:
            raise ValueError("Las predicciones guardadas no contienen 630 filas por C0–C4")
        fs,pairs,pack=self.m2.metrics_for(pred,self.mg);metrics=pd.DataFrame(pack["rows"])
        fs.to_csv(OUT/"metricas_por_fold_seed.csv",index=False);pairs.to_csv(OUT/"predicciones_oof_por_par.csv",index=False)
        fs.groupby(["model","composition_id"],as_index=False).mean(numeric_only=True).to_csv(OUT/"metricas_por_composicion.csv",index=False)
        boots=self._bootstrap(pack["formula_seed"]);boots.to_csv(OUT/"bootstrap_apareado.csv",index=False)
        drift_path=OUT/"embedding_drift.csv"
        drift=pd.read_csv(drift_path) if drift_path.is_file() else self._final_embedding_drift(pred,records.to_dict("records"))
        if drift.empty or "cosine_vs_alexandria" not in drift:
            raise ValueError("La auditoría de deriva contextualizada está ausente o incompleta")
        figures=self._figures(pred,metrics,boots,history,drift)
        # Training timestamps are recoverable; CPU time and peak RSS were not flushed before the figure error.
        log_lines=(OUT/"training.log").read_text(errors="replace").splitlines()
        def log_time(line:str)->datetime|None:
            """Parsea el timestamp estructurado del logger local."""
            try:return datetime.strptime(line[:19],"%Y-%m-%d %H:%M:%S")
            except ValueError:return None
        starts=[log_time(line) for line in log_lines if "Preparando experimento en cpu" in line]
        ends=[log_time(line) for line in log_lines if "Completado fold=4 seed=2026" in line]
        wall=float((ends[-1]-starts[0]).total_seconds()) if starts and ends and starts[0] and ends[-1] else None
        resources={"status":"COMPLETED_WITH_POSTPROCESS_RECOVERY","training_status":"60/60 final outputs completed",
          "postprocessing_recovered_without_retraining":True,"elapsed_wall_seconds_from_log":wall,"process_cpu_seconds":None,
          "max_rss_bytes":None,"resource_note":"CPU time and peak RSS were not persisted before the original figure-generation exception; left unknown rather than estimated.",
          "device":"cpu","torch_threads":4,"mps_available_at_run_start":False,"cpu_count_logical":os.cpu_count(),
          "new_final_condition_runs":60,"shared_warmups":15,"baseline_runs_reused":15,"historical_test_accessed":False,
          "checkpoint_sha256":{"MatterGPT_F4_71":F4_SHA,"MatterGPT_original":BASE_SHA}}
        atomic_json(OUT/"resources.json",resources)
        a=metrics[metrics.estimand.eq("A")]
        atomic_json(OUT/"metrics.json",{"status":"PASS_WITH_POSTPROCESS_RECOVERY","estimand_A":a.to_dict("records"),
          "estimand_B":metrics[metrics.estimand.eq("B")].to_dict("records"),"contrasts":boots.to_dict("records"),
          "historical_test_accessed":False,"figures":figures})
        atomic_json(OUT/"summary.json",{"status":"PASS_WITH_POSTPROCESS_RECOVERY","conditions":["C0","C1","C2","C3","C4"],
          "final_runs":60,"baseline":"G1-G reused","primary_metric":"MAE ΔE firmado estimando A","figures":figures,
          "note":"Entrenamiento completó; tablas, métricas y figuras se finalizaron después de corregir la lectura de buffers de atención."})
        self._report(metrics,boots,resources,figures)
        atomic_json(OUT/"progress.json",{"status":"COMPLETED","completed_final_runs":60,"total_final_runs":60,
          "postprocessing":"recovered","finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    def _bootstrap(self,fs:pd.DataFrame)->pd.DataFrame:
        """Bootstrap paired por fórmula para contrastes C1–C4 vs controles."""
        v=fs.groupby(["model","composition_id"],as_index=False).mae_delta_signed_pairs_mev_atom.mean().pivot(index="composition_id",columns="model",values="mae_delta_signed_pairs_mev_atom")
        contrasts=[("C1","C0"),("C2","C1"),("C3","C1"),("C4","C3"),("C4","C0")];rng=np.random.default_rng(20261102);rows=[]
        for a,b in contrasts:
            d=(v[a]-v[b]).to_numpy(float);draw=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
            rows.append({"contrast":f"{a} − {b}","estimate_mev_atom":float(d.mean()),"ci95_low_mev_atom":float(np.quantile(draw,.025)),"ci95_high_mev_atom":float(np.quantile(draw,.975)),"n_compositions":70,"replicates":10000,"seed":20261102,"inference":"IC percentil sin ajustar; OOF de desarrollo"})
        return pd.DataFrame(rows)

    def _final_embedding_drift(self,pred:pd.DataFrame,records:list[dict[str,Any]])->pd.DataFrame:
        """Compara embeddings por capa original/F4/seleccionado en mismos tokens OOF."""
        def layers(model:torch.nn.Module, indices:np.ndarray)->dict[str,np.ndarray]:
            model.eval();out={f"block_{i+1}":[] for i in range(12)};out["FINAL_END"]=[]
            with torch.inference_mode():
                for start in range(0,len(indices),8):
                    ix=indices[start:start+8];ids,pos=self.train_module.make_batch(self.sequences,ix);ids=ids.to(self.device);pos=pos.to(self.device)
                    batch,length=ids.shape;token=model.tok_emb(ids);position=model.pos_emb[:,:length,:]
                    stype=model.type_emb(torch.ones((batch,length),dtype=torch.long,device=self.device));hidden=model.drop(token+position+stype)
                    ptype=model.type_emb(torch.zeros((batch,1),dtype=torch.long,device=self.device));condition=torch.tensor([1.,2.],dtype=torch.float32,device=self.device).view(1,1,2).expand(batch,1,2)
                    prop=model.prop_nn(condition)+ptype;hidden=torch.cat([prop,hidden],dim=1);term=pos+1;rr=torch.arange(batch,device=self.device)
                    for j,block in enumerate(model.blocks):
                        hidden,_=block(hidden);out[f"block_{j+1}"].append(hidden[rr,term].detach().cpu().numpy())
                    final=model.ln_f(hidden);out["FINAL_END"].append(final[rr,term].detach().cpu().numpy())
            return {k:np.concatenate(v).astype(np.float32) for k,v in out.items()}
        rows=[];name_to_record={(r["condition"],int(r["fold"])):r for r in records if r["phase"]=="selected_final" and int(r["seed"])==42}
        original,stoi,block=self.train_module.load_model();original.eval().to(self.device)
        for p in original.parameters():p.requires_grad_(False)
        for fold in range(5):
            outer=sorted(self.folds.loc[(self.folds.outer_fold.eq(fold))&self.folds.outer_role.eq("outer_validation"),"composition_group"].unique())
            ix=np.concatenate([self.groups[f] for f in outer]).astype(int)
            orig=layers(original,ix)
            self.model.load_state_dict(self.base_state,strict=True);self.model.to(self.device).eval()
            alex=layers(self.model,ix)
            for condition in ["C0","C1","C2","C3","C4"]:
                if condition=="C0":cur=alex
                else:
                    r=name_to_record[(condition,fold)];checkpoint=torch.load(ROOT/r["checkpoint"],map_location="cpu",weights_only=False)
                    state=self.model.state_dict();state.update(checkpoint["block_state_dict"]);self.model.load_state_dict(state,strict=True);self.model.eval()
                    cur=layers(self.model,ix)
                    self.model.load_state_dict(self.base_state,strict=True);self.model.eval()
                for layer in [*(f"block_{i}" for i in range(1,13)),"FINAL_END"]:
                    a=alex[layer];c=cur[layer];o=orig[layer];cos=np.sum(a*c,axis=1)/(np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1),1e-12))
                    rel=np.linalg.norm(c-a,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-12)
                    co=np.sum(o*a,axis=1)/(np.maximum(np.linalg.norm(o,axis=1)*np.linalg.norm(a,axis=1),1e-12))
                    for j,idx in enumerate(ix):
                        row=self.data.iloc[idx];rows.append({"condition":condition,"fold":fold,"seed":42,"structure_id":row.structure_id,"composition_id":row.composition_id,
                          "family":row.family,"layer":layer,"cosine_vs_alexandria":float(cos[j]),"relative_l2_vs_alexandria":float(rel[j]),
                          "cosine_original_vs_alexandria":float(co[j]),"norm_original":float(np.linalg.norm(o[j])),"norm_alexandria":float(np.linalg.norm(a[j])),"norm_selected":float(np.linalg.norm(c[j]))})
                # Triplet separation, a descriptive representation statistic per formula.
                for formula in sorted(outer):
                    fidx=self.groups[formula];loc=[int(np.flatnonzero(ix==i)[0]) for i in fidx]
                    for layer in ["block_8","block_9","block_10","block_11","FINAL_END"]:
                        x=cur[layer][loc];dist=[float(np.linalg.norm(x[i]-x[j])) for i in range(3) for j in range(i+1,3)]
                        rows.append({"condition":condition,"fold":fold,"seed":42,"structure_id":"","composition_id":formula,"family":"triplet_summary","layer":layer,
                          "cosine_vs_alexandria":np.nan,"relative_l2_vs_alexandria":np.nan,"cosine_original_vs_alexandria":np.nan,
                          "norm_original":np.nan,"norm_alexandria":np.nan,"norm_selected":np.nan,"within_formula_pairwise_l2_mean":float(np.mean(dist))})
            self.model.load_state_dict(self.base_state,strict=True);self.model.eval()
        for p in original.parameters():p.requires_grad_(False)
        return pd.DataFrame(rows)

    def _figures(self,pred:pd.DataFrame,metrics:pd.DataFrame,boots:pd.DataFrame,history:pd.DataFrame,drift:pd.DataFrame)->list[str]:
        """Escribe figuras de MAE, contraste, ranking, paridad, curvas y L2-SP."""
        IMAGES.mkdir(parents=True,exist_ok=True);files=[]
        labels=["C0","C1","C2","C3","C4"];colors=["#4C78A8","#F58518","#54A24B","#E45756","#B279A2"]
        a=metrics[metrics.estimand.eq("A")].set_index("model")
        fig,ax=plt.subplots(figsize=(8,5));vals=[a.loc[x,"mae_delta_signed_mev_atom"] for x in labels];bars=ax.bar(labels,vals,color=colors);ax.bar_label(bars,fmt="%.1f",padding=3)
        ax.set_ylabel("MAE ΔE firmado (meV/átomo)");ax.set_title("ABS₃: LP-FT gradual y error de diferencias energéticas");ax.grid(axis="y",alpha=.2)
        fig.text(.01,-.04,"Estimando A: tres pares i<j por fórmula-semilla, semillas agrupadas por fórmula; macro de 70 fórmulas. OOF de desarrollo, cinco folds, 210 estructuras; C0 es G1-G congelado reutilizado.",fontsize=8);fig.tight_layout(rect=(0,.06,1,1));p=IMAGES/"lpft_mae_deltaE.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        fig,ax=plt.subplots(figsize=(8,5));y=np.arange(len(boots));x=boots.estimate_mev_atom.to_numpy();lo=boots.ci95_low_mev_atom.to_numpy();hi=boots.ci95_high_mev_atom.to_numpy();ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=4);ax.axvline(0,color="k");ax.set_yticks(y,boots.contrast);ax.set_xlabel("Diferencia de MAE ΔE (izquierda − derecha; meV/átomo)");ax.set_title("ABS₃: bootstrap apareado por fórmula");ax.grid(axis="x",alpha=.2);fig.text(.01,-.02,"70 fórmulas, 10 000 réplicas, semillas agregadas dentro de fórmula; IC95 % percentil sin ajustar. Contrastes de desarrollo, no test externo.",fontsize=8);fig.tight_layout(rect=(0,.05,1,1));p=IMAGES/"lpft_bootstrap.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        fig,ax=plt.subplots(figsize=(9,5));xx=np.arange(5);gs=[a.loc[x,"ground_state_accuracy"] for x in labels];pw=[a.loc[x,"pairwise_accuracy_eligible"] for x in labels];ax.bar(xx-.18,gs,.36,label="Ground state",color=colors);ax.bar(xx+.18,pw,.36,label="Pairwise elegible",color=colors,alpha=.55);ax.set_xticks(xx,labels);ax.set_ylim(0,1);ax.set_ylabel("Exactitud");ax.set_title("ABS₃: selección del mínimo DFT y ranking de polimorfos");ax.legend();ax.grid(axis="y",alpha=.2);fig.text(.01,-.04,"GS elegible si gap DFT al segundo mínimo >1 meV/átomo. Pairwise: |ΔE_DFT|>5; |ΔE_pred|≤1 no resuelto e incorrecto. Denominadores en metrics.json.",fontsize=8);fig.tight_layout(rect=(0,.06,1,1));p=IMAGES/"lpft_groundstate_pairwise.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        ens=pred.groupby(["model","structure_id","composition_id","fold"],as_index=False).agg(dft=("energy_dft_ev_atom","mean"),pred=("energy_pred_ev_atom","mean"));fig,axs=plt.subplots(1,5,figsize=(17,4),sharex=True,sharey=True);pairs_all=[];lim=0
        for c in labels:
            q=ens[ens.model.eq(c)];ps=[]
            for _,g in q.groupby("composition_id"):
                d=g.dft.to_numpy()*1000;y0=g.pred.to_numpy()*1000
                for i in range(3):
                    for j in range(i+1,3):ps.append((d[j]-d[i],y0[j]-y0[i]))
            p0=np.asarray(ps);pairs_all.append(p0);lim=max(lim,float(np.abs(p0).max()))
        for ax,c,ps in zip(axs,labels,pairs_all):
            m,b=np.polyfit(ps[:,0],ps[:,1],1);r2=1-np.sum((ps[:,1]-(m*ps[:,0]+b))**2)/np.sum((ps[:,1]-ps[:,1].mean())**2);mae=np.abs(ps[:,1]-ps[:,0]).mean();x0=np.linspace(-lim,lim,100);ax.scatter(ps[:,0],ps[:,1],s=8,alpha=.35,color=colors[labels.index(c)]);ax.plot(x0,x0,"k--",lw=.8);ax.plot(x0,m*x0+b,color="#a34a2a");ax.set_title(f"{c}: N=210\nm={m:.2f}, R²={r2:.2f}, MAE={mae:.1f}",fontsize=8);ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.grid(alpha=.2)
        axs[0].set_ylabel("ΔE predicha firmada (meV/átomo)");[ax.set_xlabel("ΔE DFT firmada (meV/átomo)") for ax in axs];fig.suptitle("ABS₃: paridad OOF de diferencias firmadas; punto = par intrafórmula");fig.text(.01,-.04,"Semillas promediadas por estructura (estimando B, solo para visualización de paridad); tres pares por fórmula, cinco folds. OLS y=mx+b calculado con todos los pares mostrados.",fontsize=8);fig.tight_layout(rect=(0,.05,1,.92));p=IMAGES/"lpft_paridad_deltaE.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True);cond_phases=[("C1","F2"),("C2","F2"),("C3","F4"),("C4","F4")]
        for ax,(cond,phase) in zip(axs.flat,cond_phases):
            h=history[(history.stage.eq(phase)) & history.condition.eq(phase)] if False else history[(history.stage.eq(phase)) & history.condition.eq(cond)]
            if len(h):
                ag=h.groupby("local_epoch").agg(tr=("train_minibatch_smoothl1","mean"),va=("inner_validation_smoothl1","mean"),n=("seed","size"));ax.plot(ag.index,ag.tr,label="Train minilotes");ax.plot(ag.index,ag.va,label="Validación completa");ax.set_title(f"{cond}: {phase}; corridas activas n≤15")
            ax.grid(alpha=.2);ax.set_xlabel("Época local de fase")
        axs[0,0].set_ylabel("SmoothL1 residual normalizada");axs[1,0].set_ylabel("SmoothL1 residual normalizada");axs[0,0].legend(fontsize=8);fig.suptitle("ABS₃: curvas MLP8 y fine-tuning parcial");fig.text(.01,-.03,"Train = promedio de minilotes inner-train; validation = conjunto completo de 11 fórmulas. Los conteos activos varían por parada temprana. Selección por validation data loss.",fontsize=8);fig.tight_layout(rect=(0,.05,1,.92));p=IMAGES/"lpft_curvas_validacion.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        # derive relative weight drift from saved best checkpoints and base reference.
        drows=[]
        for row in pd.read_csv(OUT/"checkpoint_manifest.csv").to_dict("records") if (OUT/"checkpoint_manifest.csv").exists() else []:
            if row.get("phase")!="selected_final":continue
            ck=torch.load(ROOT/row["checkpoint"],map_location="cpu",weights_only=False);state=ck["block_state_dict"]
            for b in ck["active_blocks"]:
                dif=0.;den=0.
                for n,v in state.items():
                    # Buffers (e.g. causal attention masks) are present in state_dict, but are
                    # neither trainable nor part of L2-SP parameter drift.
                    if block_index(n)==int(b) and n in self.reference:
                        ref=self.reference[n].detach().cpu();dif+=float((v-ref).pow(2).sum());den+=float(ref.pow(2).sum())
                drows.append({"condition":row["condition"],"fold":int(row["fold"]),"seed":int(row["seed"]),"block":int(b),"relative_drift":math.sqrt(dif/max(den,EPS_SP))})
        if drows:
            pd.DataFrame(drows).to_csv(OUT/"block_drift_summary.csv",index=False);d=pd.DataFrame(drows);fig,ax=plt.subplots(figsize=(8,5))
            for c,g in d.groupby("condition"):ax.plot(g.groupby("block").relative_drift.mean().index,g.groupby("block").relative_drift.mean().values,marker="o",label=c)
            ax.set_xticks([8,9,10,11]);ax.set_xlabel("Bloque MatterGPT");ax.set_ylabel("Deriva L2 relativa vs Alexandria F4-71");ax.set_title("ABS₃: deriva de pesos superiores y regularización L2-SP");ax.legend();ax.grid(alpha=.2);fig.text(.01,-.04,"Promedio por fold y semilla; denominador = norma del bloque F4-71. L2-SP penaliza bloques activos por separado. Se omiten bloques congelados 0–7, cuya deriva es cero.",fontsize=8);fig.tight_layout(rect=(0,.05,1,1));p=IMAGES/"lpft_deriva_bloques_l2sp.png";fig.savefig(p,dpi=300,bbox_inches="tight");plt.close(fig);files.append(p.name)
        return files

    def _report(self,metrics:pd.DataFrame,boots:pd.DataFrame,resources:dict[str,Any],figures:list[str])->None:
        """Escribe reporte científico con hechos, resultados, inferencias y límites."""
        a=metrics[metrics.estimand.eq("A")].set_index("model");lines=["# MatterGPT Alexandria a ABS₃: LP-FT gradual y L2-SP","","## Hechos verificados","",f"- Checkpoint inicial MatterGPT Alexandria F4-71 SHA-256 `{F4_SHA}` reconstruido sobre el checkpoint base `{BASE_SHA}`; carga completa estricta. Condicionamiento `[1.0, 2.0]`; extracción contextualizada después de `ln_f`.","- Ridge M2 congelado, geometría DFT estática de siete descriptores y scaler ajustado con 45 fórmulas inner-train por fold.","- G1-G congelado reutilizado como C0; sus métricas verificadas: MAE ΔE 48.0401 meV/átomo, MAE E/N 48.4366, MAE residual 28.4059, RMSE ΔE 69.1215, GS 57.7114 %, pairwise 69.5431 %, pendiente OLS ΔE 0.3076, R² 0.3572.","- MLP8 warmup compartido dentro de cada fold-semilla; checkpoints seleccionados solo por inner-validation; test histórico cerrado.","","## Resultados OOF","","Estimando A macro-promedia por composición después de agregar las tres semillas. El RMSE ΔE global se calcula sobre pares OOF, no como promedio de RMSE por composición.","","| Condición | MAE E/N | MAE residual | MAE ΔE A | RMSE ΔE global | GS | Pairwise | Pendiente ΔE | R² OLS ΔE |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in ["C0","C1","C2","C3","C4"]:
            r=a.loc[c];lines.append(f"| {c} | {r.mae_EN_mev_atom:.2f} | {r.mae_residual_mev_atom:.2f} | {r.mae_delta_signed_mev_atom:.2f} | {r.rmse_delta_global_mev_atom:.2f} | {r.ground_state_accuracy:.3f} ({int(r.ground_state_denominator)}) | {r.pairwise_accuracy_eligible:.3f} ({int(r.eligible_pairs)}) | {r.signed_pair_ols_slope:.3f} | {r.signed_pair_ols_r2:.3f} |")
        lines += ["","## Contrastes apareados","","Bootstrap percentil por 70 fórmulas, 10 000 réplicas; semillas agrupadas. IC95 % sin ajuste; cualquier patrón se interpreta como resultado OOF de desarrollo, no como validación externa independiente.","","| Contraste | ΔMAE ΔE | IC95 % |","|---|---:|---:|"]
        for r in boots.itertuples():lines.append(f"| {r.contrast} | {r.estimate_mev_atom:.2f} | [{r.ci95_low_mev_atom:.2f}, {r.ci95_high_mev_atom:.2f}] |")
        wall=resources.get("elapsed_wall_seconds",resources.get("elapsed_wall_seconds_from_log"))
        cpu=resources.get("process_cpu_seconds")
        rss=resources.get("max_rss_bytes")
        resource_text=f"Tiempo de pared: {wall:.1f} s." if isinstance(wall,(int,float)) else "Tiempo de pared: no disponible."
        resource_text+=f" CPU: {cpu:.1f} s." if isinstance(cpu,(int,float)) else " Tiempo CPU: no registrado durante el proceso original."
        resource_text+=f" Dispositivo: {resources.get('device','desconocido')}; memoria pico: {rss} bytes." if rss is not None else f" Dispositivo: {resources.get('device','desconocido')}; memoria pico: no registrada."
        lines += ["","## L2-SP, deriva y sobreajuste","","L2-SP es la media de la distancia cuadrática relativa normalizada por bloque frente a Alexandria F4-71, lambda=10, epsilon=1e-12. La referencia no cambia tras F2. El checkpoint de selección usa únicamente la pérdida SmoothL1 de validation; L2-SP no se suma a la pérdida de selección. La escala efectiva y gradientes quedan en `historial_entrenamiento.csv` y `l2sp_diagnostics.csv`.","","Las curvas promedian corridas activas y separan minilotes train de validation completa. La deriva se mide contra F4-71 y no se interpreta por sí sola como pérdida de conocimiento.","","## Inferencias y limitaciones","","La comparación C1–C4 reutiliza folds y Ridge M2 de G1-G. C0 no es reentrenado. El checkpoint del mejor resultado se selecciona dentro de cada fold mediante validation, pero elegir retrospectivamente entre sus predicciones OOF es una comparación de desarrollo; no debe reportarse como selección imparcial para test externo. La geometría representa celdas relajadas estáticas DFT y, por ello, las predicciones están condicionadas a esa información.","","No se afirma transferencia causal ni mejora hasta considerar pares, ground state, calibración, variación por fold y deriva. MatterGPT no se compara aquí con MatGPT.","","## Recursos","",resource_text,"","## Figuras","",*[f"- `images/{EXP}/{f}`" for f in figures]]
        (OUT/"reporte_cientifico.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        (OUT/"README.md").write_text(f"# {EXP}\n\nEjecutar localmente: `python scripts/MatterGPT/MatterGPT_Alexandria_ABS3_LPFT_Gradual_L2SP/run_lpft_abs3.py`.\n\nTrazar ejecución: `tail -f experiments/{EXP}/training.log`; progreso estructurado: `cat experiments/{EXP}/progress.json`.\n",encoding="utf-8")


def main() -> int:
    """Valida dispositivo y ejecuta preflight, piloto y ramas LP-FT solicitadas."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("auto","mps","cpu"),default="auto")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--finalize-only",action="store_true",help="Recalcula métricas e figuras desde checkpoints/OOF; no entrena")
    args=parser.parse_args()
    device=torch.device("mps" if args.device=="auto" and torch.backends.mps.is_available() else "cpu" if args.device=="auto" else args.device)
    if device.type=="mps" and not torch.backends.mps.is_available():parser.error("MPS no está disponible")
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s %(message)s",
      handlers=[logging.StreamHandler(),logging.FileHandler(OUT/"training.log",encoding="utf-8")])
    LOGGER.info("Preparando experimento en %s; CPU threads=%d",device,torch.get_num_threads())
    experiment=LPFTExperiment(device,args.preflight_only)
    if args.finalize_only:experiment.finalize_existing()
    else:experiment.execute()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
