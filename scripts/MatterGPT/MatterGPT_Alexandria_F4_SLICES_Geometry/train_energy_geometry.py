#!/usr/bin/env python3
"""Entrena MatterGPT F4 con fusión tardía de SLICES y geometría Alexandria.

Reutiliza los IDs, SLICES, targets, tokenizador, condicionamiento y política F4
del protocolo histórico. La geometría procede de ComputedStructureEntry y su
scaler ya quedó ajustado exclusivamente sobre el train Alexandria.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = ROOT / "experiments/mattergpt_alexandria_f4_slices_geometry"
BASE = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset"
GEOM = EXP / "geometry_manifest.csv"
MODEL_DIR = ROOT / "models/MatterGPT"
EXPECTED_BASE_SHA = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
GEOM_COLS = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
LOG = logging.getLogger("mattergpt_alexandria_f4_geometry")


class FusionHead(nn.Module):
    """MLP8 de fusión tardía para FINAL_END(768) y geometría escalada(7)."""
    def __init__(self) -> None:
        """Crea la cabeza 775→8→1 con GELU y dropout 0.1."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(775, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, hidden: Tensor, geometry: Tensor) -> Tensor:
        """Predice energía normalizada desde embedding contextual y celda."""
        return self.network(torch.cat([hidden, geometry], dim=-1)).squeeze(-1)


def sha256(path: Path) -> str:
    """Calcula SHA-256 por bloques para archivos grandes."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON mediante reemplazo atómico."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def peak_rss() -> int:
    """Devuelve RSS máximo en bytes, según la convención del sistema."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_training_module() -> Any:
    """Carga funciones auditadas de MatterGPT F4 sin ejecutar su main."""
    path = ROOT / "scripts/MatterGPT/MatterGPT_Alexandria_Energy_Transfer_20k_F4_Compatible/train_energy.py"
    spec = importlib.util.spec_from_file_location("historical_matter_energy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el pipeline histórico: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Guarda checkpoint mediante archivo temporal y rename atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def evaluate(module: Any, model: nn.Module, head: FusionHead, sequences: list[list[int]],
             geom: np.ndarray, targets: np.ndarray, batch: int, device: torch.device,
             zero_geometry: bool = False, permute_geometry: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Evalúa SmoothL1 completa y devuelve energías normalizadas predichas."""
    model.eval(); head.eval(); losses: list[float] = []; predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch):
            ix = np.arange(start, min(start + batch, len(sequences)))
            ids, pos = module.make_batch(sequences, ix)
            g = np.zeros_like(geom[ix]) if zero_geometry else (permute_geometry[ix] if permute_geometry is not None else geom[ix])
            pred = head(module.contextualized_final_end(model, ids.to(device), pos.to(device)),
                        torch.as_tensor(g, dtype=torch.float32, device=device))
            target = torch.as_tensor(targets[ix], dtype=torch.float32, device=device)
            losses.extend(F.smooth_l1_loss(pred, target, beta=1., reduction="none").cpu().numpy().tolist())
            predictions.extend(pred.cpu().numpy())
    return float(np.mean(losses)), np.asarray(predictions, dtype=np.float64)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Ejecuta warmup congelado y F4 con geometría, guardando artefactos y uso."""
    started = time.monotonic(); started_cpu = time.process_time()
    if EXP.exists() and any((EXP / name).exists() for name in ("historial_entrenamiento.csv", "checkpoints", "metrics.json")):
        raise FileExistsError(f"La salida ya contiene resultados; no se sobrescribe: {EXP}")
    EXP.mkdir(parents=True, exist_ok=True)
    train_csv = BASE / "alexandria_matter_compatible_train.csv"
    val_csv = BASE / "alexandria_matter_compatible_validation.csv"
    for path in (train_csv, val_csv, GEOM, EXP / "geometry_audit.json"):
        if not path.is_file(): raise FileNotFoundError(f"Falta artefacto requerido: {path}")
    audit = json.loads((EXP / "geometry_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("rows") != 18963: raise ValueError("Auditoría geométrica no PASS")
    tr, va = pd.read_csv(train_csv), pd.read_csv(val_csv)
    geom = pd.read_csv(GEOM)
    if (len(tr), len(va)) != (17245, 1718): raise ValueError("Conteos históricos de Alexandria cambiaron")
    if geom.mat_id.duplicated().any() or geom.mat_id.nunique() != 18963: raise ValueError("Manifiesto geométrico no es 1:1")
    geom_by_id = geom.set_index("mat_id")
    for frame, split in ((tr, "train"), (va, "validation")):
        joined = frame.merge(geom_by_id[GEOM_COLS + ["split", "slices_hash", "energy_per_atom_ev"]].reset_index(),
                             on="mat_id", how="left", validate="one_to_one", suffixes=("", "_geometry"))
        if joined[GEOM_COLS].isna().any().any() or not joined.split.eq(split).all(): raise ValueError(f"Join geométrico incompleto para {split}")
        if not joined.slices_hash.eq(joined.slices_hash_geometry).all(): raise ValueError(f"SLICES distinta para {split}")
        if not np.allclose(joined.energy_per_atom_ev, joined.energy_per_atom_ev_geometry, rtol=0, atol=1e-12): raise ValueError(f"Target distinto para {split}")
        frame[GEOM_COLS] = joined[GEOM_COLS].to_numpy(float)
    scaled_cols = [f"scaled_{c}" for c in GEOM_COLS]
    tr_g = geom_by_id.loc[tr.mat_id, scaled_cols].to_numpy(np.float32)
    va_g = geom_by_id.loc[va.mat_id, scaled_cols].to_numpy(np.float32)
    if not np.isfinite(tr_g).all() or not np.isfinite(va_g).all(): raise ValueError("Geometría escalada contiene NaN/Inf")
    module = load_training_module()
    if module.sha256(MODEL_DIR / "pytorch_model.pt") != EXPECTED_BASE_SHA: raise ValueError("Checkpoint generativo base SHA incorrecto")
    model, stoi, block = module.load_model()
    tr_seq, va_seq = module.read_sequences(tr, stoi, block), module.read_sequences(va, stoi, block)
    if len(tr_seq) != 17245 or len(va_seq) != 1718:
        raise ValueError("Conteo de secuencias tokenizadas distinto al dataset validado")
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS no está disponible")
    torch.set_num_threads(args.threads); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "mps": torch.mps.manual_seed(args.seed)
    model.to(device)
    for p in model.parameters(): p.requires_grad_(False)
    head = FusionHead().to(device)
    train_targets = tr.energy_per_atom_ev.to_numpy(np.float64); val_targets = va.energy_per_atom_ev.to_numpy(np.float64)
    ymean, ystd = float(train_targets.mean()), float(train_targets.std(ddof=0))
    if not np.isfinite(ystd) or ystd <= 0: raise ValueError("Escala de target inválida")
    ytr = ((train_targets-ymean)/ystd).astype(np.float32); yva=((val_targets-ymean)/ystd).astype(np.float32)
    history: list[dict[str, Any]] = []; history_path=EXP/"historial_entrenamiento.csv"
    resources={"status":"RUNNING","device":str(device),"threads":args.threads,"train_rows":len(tr),"validation_rows":len(va),
               "base_checkpoint_sha256":EXPECTED_BASE_SHA,"started_unix":time.time()}
    atomic_json(EXP/"resources.json",resources)
    # Warmup uses frozen contextual vectors; only MLP8 is optimized.
    htr=module.extract_embeddings(model,tr_seq,args.batch_size,device); hva=module.extract_embeddings(model,va_seq,args.batch_size,device)
    rng=np.random.default_rng(args.seed); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    initial_head={k:v.detach().clone() for k,v in head.state_dict().items()}
    warm_grad_max=0.0; warm_loss_first=float("nan")
    for epoch in range(1,21):
        t=time.monotonic();head.train();losses=[]
        for start in range(0,len(tr),args.batch_size):
            ix=rng.permutation(len(tr))[start:start+args.batch_size] if False else None
        order=rng.permutation(len(tr))
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size]; x=torch.from_numpy(htr[ix]).to(device); g=torch.from_numpy(tr_g[ix]).to(device); y=torch.from_numpy(ytr[ix]).to(device)
            opt.zero_grad(set_to_none=True); loss=F.smooth_l1_loss(head(x,g),y,beta=1.);loss.backward()
            if head.network[0].weight.grad is None or not torch.isfinite(head.network[0].weight.grad).all(): raise FloatingPointError("Gradiente MLP warmup inválido")
            warm_grad_max=max(warm_grad_max,float(head.network[0].weight.grad.norm().item()));opt.step();losses.append(float(loss.detach()))
        vl,vpred=evaluate(module,model,head,va_seq,va_g,yva,args.validation_batch_size,device)
        if epoch==1:warm_loss_first=float(np.mean(losses))
        history.append({"stage":"head_warmup","epoch":epoch,"train_minibatch_smoothl1":float(np.mean(losses)),"validation_smoothl1":vl,"epoch_wall_seconds":time.monotonic()-t})
        pd.DataFrame(history).to_csv(history_path,index=False)
    head_changed=any(not torch.equal(initial_head[k],v.detach()) for k,v in head.state_dict().items())
    if not head_changed or warm_grad_max<=0: raise RuntimeError("La cabeza no aprendió durante warmup")
    del htr,hva
    for p in model.parameters():p.requires_grad_(False)
    for block_module in model.blocks[8:12]:
        for p in block_module.parameters():p.requires_grad_(True)
    trainable=[p for b in model.blocks[8:12] for p in b.parameters()]+list(head.parameters())
    opt=torch.optim.AdamW([{"params":[p for b in model.blocks[8:12] for p in b.parameters()],"lr":3e-6,"weight_decay":0.01},
                           {"params":list(head.parameters()),"lr":1e-4,"weight_decay":1e-4}])
    best=float("inf");best_epoch=0;stale=0;ckpt=EXP/"checkpoints/mattergpt_alexandria_f4_geometry_best.pt"
    for epoch in range(1,args.max_epochs+1):
        t=time.monotonic();model.train();head.train();order=rng.permutation(len(tr));total=0.;seen=0;grad_norms={i:0. for i in range(8,12)}
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size];ids,pos=module.make_batch(tr_seq,ix);ids=ids.to(device);pos=pos.to(device)
            xg=torch.from_numpy(tr_g[ix]).to(device);y=torch.from_numpy(ytr[ix]).to(device);opt.zero_grad(set_to_none=True)
            hidden=module.contextualized_final_end(model,ids,pos);pred=head(hidden,xg);loss=F.smooth_l1_loss(pred,y,beta=1.)
            if not torch.isfinite(loss):raise FloatingPointError(f"Loss no finita en F4 {epoch}")
            loss.backward()
            for i in range(8,12):
                params=[p for p in model.blocks[i].parameters() if p.grad is not None]
                if not params:raise RuntimeError(f"Bloque F4 {i} sin gradiente")
                grad_norms[i]=max(grad_norms[i],float(torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in params)).item()))
            if any(not torch.isfinite(p.grad).all() for p in trainable if p.grad is not None):raise FloatingPointError("Gradiente F4 NaN/Inf")
            opt.step();total+=float(loss.detach())*len(ix);seen+=len(ix)
        val_loss,_=evaluate(module,model,head,va_seq,va_g,yva,args.validation_batch_size,device)
        record={"stage":"f4_geometry","epoch":epoch,"train_minibatch_smoothl1":total/seen,"validation_smoothl1":val_loss,
                "epoch_wall_seconds":time.monotonic()-t,**{f"block_{i}_max_grad_norm":v for i,v in grad_norms.items()}}
        history.append(record);pd.DataFrame(history).to_csv(history_path,index=False)
        if val_loss<best:
            best,best_epoch,stale=val_loss,epoch,0
            state={"base_checkpoint_sha256":EXPECTED_BASE_SHA,"geometry_manifest_sha256":sha256(GEOM),"condition_vector":[1.,2.],
                   "selected_epoch":epoch,"validation_smoothl1_normalized":val_loss,"target_mean_ev_atom":ymean,"target_std_ev_atom":ystd,
                   "head_state_dict":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                   "blocks_8_11":{k:v.detach().cpu().clone() for k,v in model.state_dict().items() if any(k.startswith(f"blocks.{i}.") for i in range(8,12))}}
            save_state(ckpt,state)
        else:stale+=1
        resources={"status":"RUNNING","device":str(device),"threads":args.threads,"completed_f4_epochs":epoch,"best_epoch":best_epoch,
                   "best_validation_smoothl1":best,"elapsed_wall_seconds":time.monotonic()-started,"elapsed_cpu_seconds":time.process_time()-started_cpu,
                   "peak_rss_bytes":peak_rss(),"mps_driver_allocated_bytes":int(torch.mps.driver_allocated_memory()) if device.type=="mps" else 0}
        atomic_json(EXP/"resources.json",resources)
        LOG.info("epoch=%d/%d train=%.6f validation=%.6f best_epoch=%d stale=%d",epoch,args.max_epochs,total/seen,val_loss,best_epoch,stale)
        if stale>=args.patience:break
    if best_epoch<1:raise RuntimeError("No se seleccionó checkpoint por validación Alexandria")
    state=torch.load(ckpt,map_location="cpu",weights_only=True);missing=model.load_state_dict(state["blocks_8_11"],strict=False)
    unexpected=set(missing.unexpected_keys);expected_missing={k for k in model.state_dict() if not any(k.startswith(f"blocks.{i}.") for i in range(8,12))}
    if unexpected or set(missing.missing_keys)!=expected_missing:raise RuntimeError("Reconstrucción parcial de bloques F4 no coincide con máscara esperada")
    head.load_state_dict(state["head_state_dict"],strict=True)
    base_val_loss,pred_norm=evaluate(module,model,head,va_seq,va_g,yva,args.validation_batch_size,device)
    zero_loss,zero_pred=evaluate(module,model,head,va_seq,va_g,yva,args.validation_batch_size,device,zero_geometry=True)
    perm=rng.permutation(len(va_g));perm_loss,perm_pred=evaluate(module,model,head,va_seq,va_g,yva,args.validation_batch_size,device,permute_geometry=va_g[perm])
    prediction=pred_norm*ystd+ymean;error=prediction-val_targets
    def regression_metrics(y: np.ndarray,p: np.ndarray)->dict[str,float]:
        e=p-y; slope,intercept=np.polyfit(y,p,1); ss=float(np.sum((y-y.mean())**2));r2=1-float(np.sum(e**2))/ss if ss else float("nan")
        return {"mae_ev_atom":float(np.mean(np.abs(e))),"rmse_ev_atom_global":float(np.sqrt(np.mean(e**2))),"bias_ev_atom":float(np.mean(e)),"slope_ols":float(slope),"intercept_ev_atom":float(intercept),"r2_predictive":r2}
    metrics={"validation_rows":len(va),"unit":"eV/atom","selected_epoch":best_epoch,"best_validation_smoothl1_normalized":best,
             "target_mean_ev_atom_train":ymean,"target_std_ev_atom_train":ystd,"geometry_original":regression_metrics(val_targets,prediction),
             "geometry_zero_diagnostic":regression_metrics(val_targets,zero_pred*ystd+ymean),"geometry_permuted_diagnostic":regression_metrics(val_targets,perm_pred*ystd+ymean),
             "smoothl1_original":base_val_loss,"smoothl1_geometry_zero":zero_loss,"smoothl1_geometry_permuted":perm_loss,
             "diagnostic_note":"Las ablaciones se evalúan con el predictor entrenado; no son intervenciones causales ni nueva selección de checkpoint."}
    atomic_json(EXP/"metrics.json",metrics)
    atomic_json(EXP/"summary.json",{"status":"COMPLETED","experiment":EXP.name,"training_executed":True,
        "backbone":"MatterGPT generativo original congelado salvo bloques 8-11 durante F4","condition_vector":[1.,2.],"train_rows":len(tr),"validation_rows":len(va),
        "selected_epoch":best_epoch,"checkpoint":str(ckpt.relative_to(ROOT)),"checkpoint_sha256":sha256(ckpt),"metrics":metrics,
        "warmup":{"epochs":20,"first_epoch_train_loss":warm_loss_first,"max_head_gradient_norm":warm_grad_max,"head_parameters_changed":head_changed},
        "geometry_fusion":"concat(FINAL_END(768), StandardScaler(train-only)(7)) after Transformer; no geometric tokens"})
    atomic_json(EXP/"resources.json",{"status":"COMPLETED","device":str(device),"threads":args.threads,"best_epoch":best_epoch,
        "elapsed_wall_seconds":time.monotonic()-started,"elapsed_cpu_seconds":time.process_time()-started_cpu,"peak_rss_bytes":peak_rss(),
        "mps_driver_allocated_bytes":int(torch.mps.driver_allocated_memory()) if device.type=="mps" else 0})
    return metrics


def main() -> int:
    """Valida opciones de ejecución y comienza el entrenamiento Alexandria F4."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    parser.add_argument("--threads",type=int,default=4);parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--validation-batch-size",type=int,default=8);parser.add_argument("--max-epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=15);parser.add_argument("--seed",type=int,default=42);args=parser.parse_args()
    if min(args.threads,args.batch_size,args.validation_batch_size,args.max_epochs,args.patience)<=0:parser.error("Los tamaños, hilos y épocas deben ser positivos")
    EXP.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(EXP/"training.log",encoding="utf-8")])
    train(args);return 0


if __name__=="__main__":raise SystemExit(main())
