"""Utilidades para el experimento controlado SLICES + geometría de celda.
No importa el backbone para entrenar: consume exclusivamente embeddings H3-C congelados.
"""
from __future__ import annotations
import hashlib, json, csv, itertools, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

FAMILIES = ('dist_perovskite', 'hexagonal', 'needle')
SEEDS = (42, 123, 2026)
GEOM_COLS = ('a_reduced_shape','b_reduced_shape','c_reduced_shape',
             'alpha_reduced_deg','beta_reduced_deg','gamma_reduced_deg',
             'volume_per_atom_ang3')
EXPECTED_CACHE_HASH = '2f790a30ba0fa3cd961c12de82b65febf0ae651b1f217efd5df49c8c5a9cfb62'
EXPECTED_CHECKPOINT_HASH='4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414'

class ResidualMLP8(nn.Module):
    """Misma arquitectura bloqueada 768→8→1; devuelve residuos centrados por triplete."""
    def __init__(self, dim:int, dropout:float=0.1):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(dim,8),nn.GELU(),nn.Dropout(dropout),nn.Linear(8,1))
    def forward(self, x):
        r=self.network(x).squeeze(-1)
        return r-r.mean(dim=1,keepdim=True)

def matched_pair(dim:int, geom_dim:int=7)->tuple[ResidualMLP8,ResidualMLP8]:
    """Mismas proyecciones SLICES, resto de pesos y salidas iniciales.
    Geometría se incorpora con coeficientes inicialmente cero y gradiente no nulo.
    """
    base=ResidualMLP8(dim)
    fusion=ResidualMLP8(dim+geom_dim)
    with torch.no_grad():
        fusion.network[0].weight[:,:dim].copy_(base.network[0].weight)
        fusion.network[0].weight[:,dim:].zero_()
        fusion.network[0].bias.copy_(base.network[0].bias)
        fusion.network[3].load_state_dict(base.network[3].state_dict())
    return base,fusion

def centered_loss(pred,target,scale):
    return F.smooth_l1_loss(pred/scale,target/scale,beta=1.0)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_development(csvpath:Path)->pd.DataFrame:
    """Filtra TEST antes de leer objetivos energéticos."""
    rec=[]
    with open(csvpath,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] not in ('train','val'):continue
            n=int(row['num_atoms']);tot=float(row['energy_total_ev']);ep=float(row['energy_per_atom_ev'])
            if n<=0 or abs(tot/n-ep)>1e-8:raise ValueError('E/N inconsistente: '+row['structure_id'])
            rec.append(dict(structure_id=row['structure_id'],composition_id=row['reduced_formula'],
                        polymorph_family=row['polymorph_family'],n_atoms=n,
                        energy_total_dft_ev=tot,energy_per_atom_dft_ev=ep,slices=row['matgpt_slices']))
    data=pd.DataFrame(rec)
    if len(data)!=210 or data.structure_id.nunique()!=210 or data.composition_id.nunique()!=70:
        raise ValueError('Se esperan 210 estructuras de desarrollo y 70 fórmulas')
    for formula,part in data.groupby('composition_id'):
        if len(part)!=3 or set(part.polymorph_family)!=set(FAMILIES):
            raise ValueError('Triplete/familia inválida: '+str(formula))
    return data

def load_features(path:Path, ids:list[str], strict_hash:bool=True)->dict:
    actual_hash=sha256(path)
    if strict_hash and actual_hash!=EXPECTED_CACHE_HASH:
        # El cache histórico fue retirado del árbol activo. Solo se acepta una
        # reconstrucción numérica si un cotejo independiente de los 210 pares
        # y las cinco medias Ridge históricas dejó manifiesto verificable.
        audit_path=path.with_suffix('.numeric_audit.json')
        if not audit_path.is_file():
            raise ValueError('Cache distinto de H3-C sin auditoría numérica')
        audit=json.loads(audit_path.read_text())
        if (audit.get('status')!='PASS_NUMERIC' or audit.get('cache_sha256')!=actual_hash
                or audit.get('historical_cache_sha256')!=EXPECTED_CACHE_HASH
                or float(audit.get('max_ridge_mean_difference_ev_atom',np.inf))>=1e-5):
            raise ValueError('Cache reconstruido no pasó la auditoría H3-C')
    with np.load(path,allow_pickle=False) as a:
        if a['structure_id'].astype(str).tolist()!=ids:raise ValueError('IDs de features desalineados')
        out={k:a[k].astype(np.float32) for k in ('L03_MEAN','FINAL_END')}
    for v in out.values():
        if v.shape!=(210,768) or not np.isfinite(v).all():raise ValueError('Feature corrupta')
    return out

def load_folds(path:Path,data:pd.DataFrame)->pd.DataFrame:
    folds=pd.read_csv(path)
    folds=folds[folds.historical_split.isin(['train','val'])].copy()
    if len(folds)!=1050 or set(folds.structure_id)!=set(data.structure_id):
        raise ValueError('Folds históricos incompatibles')
    for f in range(5):
        p=folds[folds.outer_fold==f]
        if len(p)!=210:raise ValueError('Fold incompleto')
        group=data.set_index('structure_id').composition_id
        sets=[]
        for role in ('inner_train','inner_validation'):
            sets.append(set(group.loc[p[p.inner_role==role].structure_id]))
        sets.append(set(group.loc[p[p.outer_role=='outer_validation'].structure_id]))
        if list(map(len,sets))!=[45,11,14] or any(sets[i]&sets[j] for i,j in itertools.combinations(range(3),2)):
            raise ValueError('Fuga composicional fold '+str(f))
    return folds

def roles(folds:pd.DataFrame,fold:int):
    p=folds[folds.outer_fold==fold]
    return (p[p.inner_role=='inner_train'].structure_id.astype(str).tolist(),
            p[p.inner_role=='inner_validation'].structure_id.astype(str).tolist(),
            p[p.outer_role=='outer_validation'].structure_id.astype(str).tolist())

def triplets(data:pd.DataFrame, ids:list[str], x:np.ndarray)->tuple[list[str],np.ndarray,np.ndarray,list[list[str]]]:
    lookup={sid:i for i,sid in enumerate(data.structure_id)}
    sub=data[data.structure_id.isin(ids)]
    formulas=sorted(sub.composition_id.unique())
    groups=[]
    for f in formulas:
        p=sub[sub.composition_id==f].set_index('polymorph_family')
        groups.append([p.loc[family,'structure_id'] for family in FAMILIES])
    if sum(map(len,groups))!=len(ids) or set(sum(groups,[]))!=set(ids):
        raise ValueError('Tripletes desalineados/incompletos')
    vals=np.array([[x[lookup[sid]] for sid in group] for group in groups],dtype=np.float32)
    target=data.set_index('structure_id')
    y=np.array([[target.loc[sid,'energy_per_atom_dft_ev'] for sid in g] for g in groups],dtype=np.float32)
    return formulas,vals,y,groups

def load_geometry(path:Path,data:pd.DataFrame)->np.ndarray:
    df=pd.read_csv(path)
    if len(df)!=210 or df.structure_id.nunique()!=210 or set(df.structure_id)!=set(data.structure_id):
        raise ValueError('Geometría debe contener exactamente los 210 IDs de desarrollo')
    if any(c not in df.columns for c in GEOM_COLS):
        raise ValueError('Faltan descriptores geométricos: '+str(set(GEOM_COLS)-set(df.columns)))
    if 'geometry_source' not in df or df.geometry_source.isna().any():
        raise ValueError('Se exige procedencia geometry_source por estructura')
    if 'source_path' not in df or df.source_path.isna().any():
        raise ValueError('Se exige source_path por estructura')
    if 'n_atoms' in df:
        chk=df.set_index('structure_id').loc[data.structure_id]
        if not np.array_equal(chk.n_atoms.astype(int),data.n_atoms.astype(int)):
            raise ValueError('N_atoms de geometría inconsistente con H3-C')
    x=df.set_index('structure_id').loc[data.structure_id,list(GEOM_COLS)].to_numpy(float)
    if not np.isfinite(x).all() or (x[:,:3]<=0).any() or (x[:,6]<=0).any():
        raise ValueError('Valores geométricos no válidos')
    if (x[:,3:6]<=0).any() or (x[:,3:6]>=180).any():raise ValueError('Ángulos inválidos')
    return x.astype(np.float32)

def seeded(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def train(model:ResidualMLP8,xtrain,ytrain,xvalid,yvalid,seed:int,scale:float):
    """200 épocas, AdamW, 8 fórmulas/batch, early stopping 20.
    El criterio es SmoothL1 de residuos centrados / escala train.
    """
    seeded(seed)
    # Seeded is called after model creation; dropout/order deterministic per condition.
    xtr=torch.tensor(xtrain,dtype=torch.float32)
    xva=torch.tensor(xvalid,dtype=torch.float32)
    ttr=torch.tensor(ytrain-ytrain.mean(axis=1,keepdims=True),dtype=torch.float32)
    tva=torch.tensor(yvalid-yvalid.mean(axis=1,keepdims=True),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    rng=np.random.default_rng(seed)
    best=np.inf; best_state=None; patience=0; history=[]; epoch_best=-1
    for epoch in range(1,201):
        model.train(); losses=[]
        for ix in np.array_split(rng.permutation(len(xtr)),range(8,len(xtr),8)):
            output=model(xtr[ix])
            loss=centered_loss(output,ttr[ix],scale)
            if not torch.isfinite(loss):raise ValueError('Loss no finita')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode(): val=float(centered_loss(model(xva),tva,scale))
        history.append(dict(epoch=epoch,loss_train=float(np.mean(losses)),loss_inner_validation=val))
        if val<best-1e-12:
            best=val;epoch_best=epoch;patience=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:patience+=1
        if patience>=20:break
    if best_state is None:raise ValueError('No converge/selecciona')
    model.load_state_dict(best_state,strict=True);model.eval()
    return model,dict(best_epoch=epoch_best,best_inner_loss=best,epochs=len(history),
                      residual_scale_ev_atom=scale,parameters=sum(p.numel() for p in model.parameters())),history

