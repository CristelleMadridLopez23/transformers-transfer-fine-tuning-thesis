#!/usr/bin/env python3
"""Entrena D2 con energía absoluta más energía centrada por composición.

La arquitectura y el optimizador siguen E2-A: checkpoint original, último bloque
del decoder ajustable, end pooling, MLP8, AdamW y minibatches de cuatro grupos.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import time
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata, spearmanr

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development, sha256, write_json

LOG = logging.getLogger("matgpt_diagnostics.d2")
E2_SCRIPT = Path(__file__).resolve().parents[1] / "MatGPT_finetuning/matgpt_e2/finetune.py"
SEEDS = (42, 123, 2026)
VARIANTS = ("D2-CENTER", "D2-CENTER-ONLY")


def helpers():
    """Importa utilidades E2-A auditadas para reproducir el forward congelado/parcial."""
    sys.path.insert(0, str(E2_SCRIPT.parent))
    import finetune  # type: ignore[import-not-found]
    return finetune


def centered_loss(prediction: torch.Tensor, target: torch.Tensor, groups: list[str], scale: float) -> torch.Tensor:
    """Huber centrada por grupo completo, promediada primero por composición."""
    losses = []
    for composition in sorted(set(groups)):
        indexes = [i for i, value in enumerate(groups) if value == composition]
        if len(indexes) != 3:
            raise ValueError(f"D2 requiere tres polimorfos del grupo {composition}; encontró {len(indexes)}.")
        p = prediction[indexes]; y = target[indexes]
        p_centered = p - p.mean()
        y_centered = y - y.mean()
        losses.append(nn.functional.smooth_l1_loss((p_centered-y_centered)/scale,
                                                   torch.zeros_like(y_centered), beta=1.0))
    return torch.stack(losses).mean()


def group_metrics(group: pd.DataFrame) -> dict[str, float]:
    """Calcula MAE, ΔE y ranking con tolerancia de 1 meV/átomo."""
    y = group.energy_per_atom_ev.to_numpy(float)*1000
    p = group.energy_pred_ev_atom.to_numpy(float)*1000
    iy, ip = np.argsort(y), np.argsort(p)
    eligible = float(y[iy[1]]-y[iy[0]] > 1.0)
    pred_tie = bool(p[ip[1]]-p[ip[0]] <= 1.0)
    pairs = [(i,j) for i in range(len(y)) for j in range(i+1,len(y)) if abs(y[i]-y[j]) > 1.0]
    pair_errors = [((p[i]-p[j])-(y[i]-y[j])) for i,j in pairs]
    rank_error = rankdata(p, method="average")-rankdata(y, method="average")
    rho = spearmanr(y,p).statistic if np.ptp(p)>1e-12 and np.ptp(y)>1e-12 else np.nan
    return {"absolute_mae_mev_atom": float(np.abs(y-p).mean()),
            "delta_e_mae_mev_atom": float(np.abs((y-y.min())-(p-p.min())).mean()),
            "ground_state_accuracy": float(iy[0] == ip[0] and not pred_tie) if eligible else float("nan"),
            "ground_state_eligible": eligible, "predicted_tie": float(pred_tie),
            "pairwise_accuracy": float(np.mean([np.sign(y[i]-y[j]) == np.sign(p[i]-p[j]) for i,j in pairs])) if pairs else float("nan"),
            "dft_range_mev_atom": float(np.ptp(y)), "pred_range_mev_atom": float(np.ptp(p)),
            "collapsed": float(np.ptp(p) < 1.0), "n_pairs_eligible": len(pairs),
            "pairwise_energy_mae_mev_atom": float(np.mean(np.abs(pair_errors))) if pair_errors else float("nan"),
            "pairwise_signed_bias_mev_atom": float(np.mean(pair_errors)) if pair_errors else float("nan"),
            "spearman_by_composition": float(rho), "absolute_rank_error_mean": float(np.mean(np.abs(rank_error))),
            "signed_rank_error_mean": float(np.mean(rank_error))}


def fit_one(data: pd.DataFrame, assignments: pd.DataFrame, stoi: dict[str, int], fold: int,
            seed: int, variant: str, phase: str, device: torch.device,
            output: Path, epochs: int, patience: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Entrena en inner-train, para con inner-validation y predice outer una vez."""
    helper = helpers()
    expected_checkpoint = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
    if sha256(helper.CKPT) != expected_checkpoint:
        raise RuntimeError("Checkpoint MatGPT difiere del SHA-256 fijado por E0/E2-A.")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, stoi = helper.setup(); model.to(device).eval()
    for param in model.parameters(): param.requires_grad_(False)
    trainable_layer = model.transformer_decoder.layers[-1]
    for param in trainable_layer.parameters(): param.requires_grad_(True)
    head = helper.Head().to(device)
    role = assignments.drop_duplicates("structure_id").set_index("structure_id")
    ids = data.structure_id.astype(str).to_numpy()
    groups = data.composition_group.astype(str).to_numpy()
    tr = np.flatnonzero(np.array([role.loc[x, "inner_role"] == "inner_train" for x in ids]))
    va = np.flatnonzero(np.array([role.loc[x, "inner_role"] == "inner_validation" for x in ids]))
    outer = np.flatnonzero(np.array([role.loc[x, "outer_role"] == "outer_validation" for x in ids]))
    if (len(tr),len(va),len(outer)) != (135,33,42):
        raise ValueError(f"Conteos inesperados fold={fold}: {len(tr)}/{len(va)}/{len(outer)}")
    train_groups, val_groups, outer_groups = set(groups[tr]), set(groups[va]), set(groups[outer])
    if train_groups&val_groups or train_groups&outer_groups or val_groups&outer_groups:
        raise ValueError("D2 detectó composiciones compartidas entre particiones.")
    y_np = data.energy_per_atom_ev.to_numpy(float)
    abs_center = float(y_np[tr].mean())
    abs_scale = max(float(y_np[tr].std()), 1e-6)
    centered_train = np.concatenate([g.energy_per_atom_ev.to_numpy(float)-g.energy_per_atom_ev.mean()
                                     for _,g in data.iloc[tr].groupby("composition_group")])
    center_scale = max(float(np.sqrt(np.mean(centered_train**2))), 1e-3)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    sequences = [helper.encode(value, stoi, device) for value in data.matgpt_slices.astype(str)]
    pad = stoi["<pad>"]
    optimizer = torch.optim.AdamW(
        [{"params": list(trainable_layer.parameters()), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}],
        weight_decay=1e-4)
    best_value = float("inf"); best_state = None; best_epoch = 0; wait = 0; history=[]
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.eval(); trainable_layer.train(); head.train()
        order = rng.permutation(sorted(train_groups)); train_losses=[]
        for begin in range(0,len(order),4):
            chosen = set(order[begin:begin+4])
            batch_idx = np.array([i for i in tr if groups[i] in chosen], dtype=int)
            xb = helper.batch(sequences, batch_idx, pad, device)
            optimizer.zero_grad(set_to_none=True)
            hidden = helper.forward_hidden(model, xb, pad, True)
            prediction = head(hidden)*abs_scale+abs_center
            loss_abs = nn.functional.smooth_l1_loss((prediction-y[batch_idx])/abs_scale,
                                                     torch.zeros_like(y[batch_idx]), beta=1.0)
            loss_center = centered_loss(prediction, y[batch_idx], [groups[i] for i in batch_idx], center_scale)
            loss = loss_center if variant == "D2-CENTER-ONLY" else loss_abs+loss_center
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Pérdida D2 NaN/Inf fold={fold} seed={seed} epoch={epoch}.")
            loss.backward()
            if phase == "smoke" and epoch == 1 and begin == 0:
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0
                           for p in trainable_layer.parameters()):
                    raise RuntimeError("Smoke D2: último bloque sin gradiente finito y no nulo.")
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0
                           for p in head.parameters()):
                    raise RuntimeError("Smoke D2: MLP sin gradiente finito y no nulo.")
                if any(p.grad is not None for name,p in model.named_parameters()
                       if not name.startswith("transformer_decoder.layers.11.")):
                    raise RuntimeError("Smoke D2: se detectó gradiente en pesos congelados.")
            grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable_layer.parameters())+list(head.parameters()), 1.0)
            if not torch.isfinite(grad_norm): raise FloatingPointError("Norma de gradiente no finita.")
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); head.eval()
        with torch.no_grad():
            vbatch = helper.batch(sequences, va, pad, device)
            vpred = head(helper.forward_hidden(model, vbatch, pad, True))*abs_scale+abs_center
            vabs = nn.functional.smooth_l1_loss((vpred-y[va])/abs_scale, torch.zeros_like(y[va]), beta=1.0)
            vcenter = centered_loss(vpred, y[va], [groups[i] for i in va], center_scale)
            vobj = vcenter if variant == "D2-CENTER-ONLY" else vabs+vcenter
        history.append({"configuration":variant,"phase":phase,"fold":fold,"seed":seed,"epoch":epoch,
                        "train_objective":float(np.mean(train_losses)),"inner_validation_objective":float(vobj),
                        "inner_val_abs_component":float(vabs),"inner_val_centered_component":float(vcenter),
                        "center_scale_ev_atom":center_scale,"absolute_scale_ev_atom":abs_scale,
                        "elapsed_seconds":time.perf_counter()-start,"gradient_norm_last_batch":float(grad_norm.detach().cpu())})
        if epoch == 1 or epoch % 10 == 0:
            LOG.info("d2_epoch phase=%s config=%s fold=%d seed=%d epoch=%d/%d train=%.5f inner_val=%.5f elapsed_s=%.1f",
                     phase, variant, fold, seed, epoch, epochs, float(np.mean(train_losses)), float(vobj), time.perf_counter()-start)
        if float(vobj) < best_value-1e-12:
            best_value=float(vobj); best_epoch=epoch; wait=0
            best_state={"last_block":{k:v.detach().cpu().clone() for k,v in trainable_layer.state_dict().items()},
                        "head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        else:
            wait += 1
        if wait >= patience: break
    if best_state is None: raise RuntimeError("Inner-validation no seleccionó checkpoint D2.")
    trainable_layer.load_state_dict(best_state["last_block"], strict=True); head.load_state_dict(best_state["head"], strict=True)
    model.eval(); head.eval()
    with torch.no_grad():
        obatch = helper.batch(sequences, outer, pad, device)
        outer_prediction = (head(helper.forward_hidden(model, obatch, pad, True))*abs_scale+abs_center).cpu().numpy()
    predictions = data.iloc[outer][["structure_id","composition_group","reduced_formula","polymorph_family","energy_per_atom_ev"]].copy()
    predictions["configuration"] = variant; predictions["fold"] = fold; predictions["seed"] = seed
    predictions["best_epoch"] = best_epoch; predictions["energy_pred_ev_atom"] = outer_prediction
    training_seconds = float(time.perf_counter()-start)
    predictions["training_wall_seconds"] = training_seconds
    predictions["checkpoint_sha256"] = expected_checkpoint
    checkpoint_path = output / "checkpoints" / variant / f"fold_{fold}_seed_{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Se conserva el checkpoint D2 existente: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({**best_state,"configuration":variant,"fold":fold,"seed":seed,"best_epoch":best_epoch,
                "absolute_scale_ev_atom":abs_scale,"center_scale_ev_atom":center_scale,
                "checkpoint_sha256":expected_checkpoint}, checkpoint_path)
    return history, predictions


def run() -> None:
    """Ejecuta smoke, piloto o CV por composición en CPU/GPU local."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["smoke","pilot","cv"],required=True)
    parser.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
    parser.add_argument("--fold",type=int,default=0); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,default=REPORT/"D2"/"v1")
    args=parser.parse_args()
    data,folds,meta=load_development()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "mps" if args.device=="auto" and torch.backends.mps.is_available() else "cpu" if args.device=="auto" else args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Se preservan las salidas D2 existentes; elige un directorio --output nuevo.")
    if args.phase=="smoke": folds_run=[0]; seeds=[42]; epochs=3; patience=3
    elif args.phase=="pilot": folds_run=[args.fold]; seeds=[args.seed]; epochs=100; patience=20
    else: folds_run=sorted(int(x) for x in folds.outer_fold.unique()); seeds=list(SEEDS); epochs=100; patience=20
    histories=[]; predictions=[]
    for fold in folds_run:
        assignment=folds[folds.outer_fold.eq(fold)]
        for seed in seeds:
            for variant in VARIANTS:
                history,pred=fit_one(data,assignment,{},fold,seed,variant,args.phase,device,args.output,epochs,patience)
                histories.extend(history); predictions.append(pred)
    history_frame=pd.DataFrame(histories); pred_frame=pd.concat(predictions,ignore_index=True)
    args.output.mkdir(parents=True,exist_ok=True)
    history_frame.to_csv(args.output/f"training_history_{args.phase}.csv",index=False)
    pred_frame.to_csv(args.output/f"oof_predictions_{args.phase}.csv",index=False)
    comp=[]
    for keys,g in pred_frame.groupby(["configuration","fold","seed","composition_group"]):
        comp.append({"configuration":keys[0],"fold":keys[1],"seed":keys[2],"composition_group":keys[3],**group_metrics(g)})
    cm=pd.DataFrame(comp); cm.to_csv(args.output/f"metrics_composition_{args.phase}.csv",index=False)
    summary=cm.groupby(["configuration","fold","seed"]).agg(n_compositions=("composition_group","nunique"),n_gs_eligible=("ground_state_eligible","sum"),**{m:(m,"mean") for m in ["absolute_mae_mev_atom","delta_e_mae_mev_atom","ground_state_accuracy","pairwise_accuracy","pairwise_energy_mae_mev_atom","pairwise_signed_bias_mev_atom","absolute_rank_error_mean","signed_rank_error_mean","spearman_by_composition","pred_range_mev_atom","collapsed"]}).reset_index()
    summary.to_csv(args.output/f"metrics_per_seed_{args.phase}.csv",index=False)
    write_json(args.output/f"provenance_{args.phase}.json",{**meta,"phase":args.phase,"folds":folds_run,"seeds":seeds,"epochs_max":epochs,"patience":patience,"optimizer":{"name":"AdamW","backbone_lr":1e-5,"head_lr":1e-3,"weight_decay":1e-4},"architecture":"last decoder block + end pooling + MLP 768-8-1","variants":list(VARIANTS),"device":str(device),"source_e2_script_sha256":sha256(E2_SCRIPT),"historical_test_used":False})
    LOG.info("d2_complete phase=%s fits=%d device=%s",args.phase,len(folds_run)*len(seeds)*len(VARIANTS),device)


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run()


if __name__=="__main__":
    main()
