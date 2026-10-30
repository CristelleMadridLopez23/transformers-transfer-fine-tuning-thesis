"""Evalúa H3-C fijo con extractor aleatorio o solo composición, sin test.

El control aleatorio conserva exactamente Ridge sobre medias L03 y MLP8 sobre
FINAL_END. El control composicional conserva Ridge y produce residuo cero,
identidad inevitable al recibir el mismo vector para tres polimorfos.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from reproduce import FAMILIES,project_root,read_development,sha256

sys.path.insert(0,str(project_root()))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import (
    StructuralMLP8,centered_smooth_l1,fit_ridge_by_validation,make_ridge)

LOG=logging.getLogger("h3c_final_controls")
ALPHAS=(.01,.1,1.,10.,100.,1000.)


def formula_fractions(formulas: list[str]) -> tuple[np.ndarray,list[str]]:
    """Codifica solo fracciones estequiométricas; igual para cada polimorfo."""
    parsed=[]
    for formula in formulas:
        tokens=re.findall(r"([A-Z][a-z]?)(\d*)",formula)
        if not tokens or "".join(element+count for element,count in tokens)!=formula:
            raise ValueError(f"Fórmula no interpretable: {formula}")
        counts={}
        for element,count in tokens:
            counts[element]=counts.get(element,0)+int(count or "1")
        parsed.append(counts)
    elements=sorted(set().union(*(set(item) for item in parsed)))
    matrix=np.array([[item.get(element,0)/sum(item.values()) for element in elements]
                     for item in parsed],dtype=np.float32)
    if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(axis=1),1):
        raise ValueError("Fracciones composicionales inválidas")
    return matrix,elements


def group_arrays(data: pd.DataFrame, ids: list[str], x: np.ndarray) -> tuple[list[str],np.ndarray,np.ndarray,list[list[str]]]:
    """Ordena tripletes por fórmula y familia sin mezclar particiones."""
    byid=data.set_index("structure_id")
    lookup={sid:i for i,sid in enumerate(data.structure_id)}
    subset=data[data.structure_id.isin(ids)]
    formulas=sorted(subset.composition_id.unique())
    groups=[]
    for formula in formulas:
        part=subset[subset.composition_id==formula]
        if set(part.polymorph_family)!=set(FAMILIES):raise ValueError("Triplete incompleto")
        groups.append([str(part[part.polymorph_family==family].structure_id.iloc[0]) for family in FAMILIES])
    if sum(map(len,groups))!=len(ids):raise ValueError("IDs mal agrupados")
    values=np.stack([x[[lookup[sid] for sid in group]] for group in groups])
    target=np.array([[byid.loc[sid,"energy_per_atom_dft_ev"] for sid in group] for group in groups],dtype=np.float32)
    return formulas,values,target,groups


def roles(data: pd.DataFrame, folds: pd.DataFrame, fold: int) -> tuple[list[str],list[str],list[str],list[str]]:
    """Extrae 45/11/14 composiciones por rol histórico y valida disjunción."""
    part=folds[folds.outer_fold==fold]
    train=part[part.inner_role=="inner_train"].structure_id.astype(str).tolist()
    valid=part[part.inner_role=="inner_validation"].structure_id.astype(str).tolist()
    outer=part[part.outer_role=="outer_validation"].structure_id.astype(str).tolist()
    outer_train=part[part.outer_role!="outer_validation"].structure_id.astype(str).tolist()
    byid=data.set_index("structure_id")
    group_sets=[set(byid.loc[group,"composition_id"]) for group in (train,valid,outer)]
    if tuple(map(len,group_sets))!=(45,11,14) or any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Fuga o tamaño incorrecto de pliegue")
    return train,valid,outer,outer_train


def train_structural(xtrain:np.ndarray,ytrain:np.ndarray,xvalid:np.ndarray,yvalid:np.ndarray,
                     seed:int) -> tuple[StructuralMLP8,dict[str,Any]]:
    """Repite presupuesto, inicialización y selección inner de MLP8 histórica."""
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train_target=ytrain-ytrain.mean(axis=1,keepdims=True)
    valid_target=yvalid-yvalid.mean(axis=1,keepdims=True)
    scale=max(float(train_target.std()),1e-6)
    Xtr=torch.tensor(np.array(xtrain,copy=True),dtype=torch.float32)
    Ytr=torch.tensor(np.array(train_target,copy=True),dtype=torch.float32)
    Xva=torch.tensor(np.array(xvalid,copy=True),dtype=torch.float32)
    Yva=torch.tensor(np.array(valid_target,copy=True),dtype=torch.float32)
    model=StructuralMLP8(input_dim=768,dropout=.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    best=float("inf");best_state=None;best_epoch=0;waited=0;history=[]
    for epoch in range(1,201):
        model.train();losses=[]
        order=rng.permutation(len(Xtr))
        for start in range(0,len(order),8):
            idx=order[start:start+8]
            pred=model(Xtr[idx]);loss=centered_smooth_l1(pred,Ytr[idx],scale,beta=1.)
            if not torch.isfinite(loss):raise ValueError("Loss no finita")
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            score=float(centered_smooth_l1(model(Xva),Yva,scale,beta=1.))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"inner_loss":score})
        if score<best-1e-12:
            best=score;best_epoch=epoch;waited=0
            best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        else:waited+=1
        if waited>=20:break
    if best_state is None:raise ValueError("MLP8 sin checkpoint")
    model.load_state_dict(best_state,strict=True);model.eval()
    return model,{"best_epoch":best_epoch,"epochs":len(history),"inner_loss":best,
                  "target_scale_ev_atom":scale,"history":history,
                  "trainable_parameters":sum(p.numel() for p in model.parameters())}


def control(root:Path,kind:str,seed:int) -> dict[str,Any]:
    """Ajusta control fijado por los cinco pliegues; guarda predicciones externas."""
    if kind not in {"random","composition"} or seed not in (42,123,2026):
        raise ValueError("Control/semilla inválidos")
    report=root/"reports/MatGPT/MatGPT_H3C_Final"
    output=report/f"controls/{kind}_fit_seed_{seed}" if kind=="random" else report/f"controls/composition_seed_{seed}"
    if output.exists() and any(output.iterdir()):raise ValueError("Control existente; no sobrescribir")
    data_path=root/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
    data=read_development(data_path)
    folds=pd.read_csv(root/"reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv")
    folds=folds[folds.historical_split.isin(["train","val"])].copy()
    if kind=="random":
        archive=report/f"controls/random_seed_{seed}/features.npz"
        manifest=json.loads(archive.with_name("manifest.json").read_text())
        if sha256(archive)!=manifest["feature_sha256"]:raise ValueError("Features random cambiaron")
        with np.load(archive,allow_pickle=False) as bundle:
            if bundle["structure_id"].astype(str).tolist()!=data.structure_id.tolist():
                raise ValueError("IDs random desalineados")
            x_ridge=bundle["L03_MEAN"].astype(np.float32)
            x_structural=bundle["FINAL_END"].astype(np.float32)
    else:
        x_ridge,elements=formula_fractions(data.composition_id.tolist())
        x_structural=None
    if output.exists() and any(output.iterdir()):raise ValueError("Control fit existente")
    rows=[];fold_records=[]
    for fold in range(5):
        tr,va,outer,fit=roles(data,folds,fold)
        _,xtr,ytr,_=group_arrays(data,tr,x_ridge)
        _,xva,yva,_=group_arrays(data,va,x_ridge)
        _,xfit,yfit,_=group_arrays(data,fit,x_ridge)
        formulas,xout,yout,groups=group_arrays(data,outer,x_ridge)
        alpha,inner_mae,_,_=fit_ridge_by_validation(xtr.mean(axis=1),ytr.mean(axis=1),
                                                     xva.mean(axis=1),yva.mean(axis=1),ALPHAS)
        ridge=make_ridge(alpha);ridge.fit(xfit.mean(axis=1),yfit.mean(axis=1))
        means=ridge.predict(xout.mean(axis=1))
        if kind=="random":
            _,xtr_s,ytr_s,_=group_arrays(data,tr,x_structural)
            _,xva_s,yva_s,_=group_arrays(data,va,x_structural)
            _,xout_s,_,_=group_arrays(data,outer,x_structural)
            mlp,training=train_structural(xtr_s,ytr_s,xva_s,yva_s,seed)
            with torch.inference_mode():residual=mlp(torch.tensor(xout_s,dtype=torch.float32)).numpy()
        else:
            residual=np.zeros((len(formulas),3),dtype=float)
            training={"best_epoch":None,"epochs":0,"inner_loss":None,"target_scale_ev_atom":None,
                      "trainable_parameters":0,"structural_branch":"identical input per formula implies exactly zero centered residual"}
        output.mkdir(parents=True,exist_ok=True)
        if kind=="random":
            torch.save({"state_dict":mlp.state_dict(),"architecture":[768,8,1],"fold":fold,"seed":seed,
                        "best_epoch":training["best_epoch"]},output/f"fold_{fold}_mlp8.pt")
            pd.DataFrame(training.pop("history")).to_csv(output/f"fold_{fold}_history.csv",index=False)
        fold_records.append({"fold":fold,"seed":seed,"selected_alpha":alpha,
                             "inner_ridge_mae_ev_atom":inner_mae,"ridge_input_dim":x_ridge.shape[1],
                             **training})
        lookup=data.set_index("structure_id")
        for i,(formula,group) in enumerate(zip(formulas,groups)):
            for j,sid in enumerate(group):
                source=lookup.loc[sid]
                pred=float(means[i]+residual[i,j])
                rows.append({"structure_id":sid,"composition_id":formula,"polymorph_family":FAMILIES[j],
                             "fold":fold,"seed":seed,"model":kind,
                             "n_atoms":int(source.n_atoms),"energy_total_dft_ev":float(source.energy_total_dft_ev),
                             "energy_per_atom_dft_ev":float(source.energy_per_atom_dft_ev),
                             "ridge_mean_ev_atom":float(means[i]),"residual_ev_atom":float(residual[i,j]),
                             "energy_per_atom_pred_ev":pred,"energy_total_pred_ev":pred*int(source.n_atoms)})
    frame=pd.DataFrame(rows)
    if len(frame)!=210 or frame.structure_id.nunique()!=210 or frame.composition_id.nunique()!=70:
        raise ValueError("Control no cubre 210 estructuras físicas")
    frame.to_csv(output/"predictions.csv",index=False)
    pd.DataFrame(fold_records).to_csv(output/"fold_records.csv",index=False)
    result={"status":"PASS","kind":kind,"seed":seed,"rows":210,"compositions":70,
            "features":"L03_MEAN and FINAL_END random" if kind=="random" else "elemental stoichiometric fractions",
            "input_dimension_ridge":int(x_ridge.shape[1]),
            "elements":elements if kind=="composition" else None,
            "structural_trainable_parameters":6161 if kind=="random" else 0,
            "identical_polymorph_input":kind=="composition",
            "predictions_sha256":sha256(output/"predictions.csv"),"historical_test_accessed":False}
    (output/"manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    LOG.info(json.dumps({"event":"control_complete","kind":kind,"seed":seed}))
    return result


def main() -> None:
    """Selecciona control y semilla sin modificar arquitectura H3-C."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    parser.add_argument("--kind",choices=["random","composition"],required=True)
    parser.add_argument("--seed",type=int,choices=[42,123,2026],required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(message)s")
    control(args.repo.resolve(),args.kind,args.seed)


if __name__=="__main__":main()
