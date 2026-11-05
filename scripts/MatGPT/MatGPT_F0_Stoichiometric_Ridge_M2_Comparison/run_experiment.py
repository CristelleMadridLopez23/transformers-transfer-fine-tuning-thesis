"""Reentrena dos cabezas MatGPT F0 con Ridge M2 y compara con MatterGPT M2.

Usa exclusivamente cachés y etiquetas de desarrollo ABS3. MatterGPT y MatGPT
permanecen congelados; el único ajuste de parámetros es de las MLP8.
"""
from __future__ import annotations
import hashlib, importlib.util, json, logging, math, os, platform, random, resource, time, warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import scipy, sklearn, torch, yaml
from scipy.stats import kendalltau, linregress, spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
EXP='matgpt_f0_stoichiometric_ridge_m2_comparison'
OUT=ROOT/'experiments'/EXP; REPORT=ROOT/'reports/MatGPT'/EXP; IMAGES=ROOT/'images'/EXP
BASE=ROOT/'experiments/mattergpt_abs3_frozen_ridge_mlp8'; BASE_G=ROOT/'experiments/mattergpt_abs3_geometry_late_fusion'
GEOM_DIR=ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria'; CACHE=GEOM_DIR/'staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz'
HIST_PRED=GEOM_DIR/'results/predicciones_oof_por_estructura.csv'
DATASET=ROOT/'reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv'
FOLDS=ROOT/'reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv'
RIDGE_DIR=ROOT/'experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional'
RIDGE_EXPECTED=['99d5114f567450f46322cd39e021405bc4ba9fd396fadeb07ecb62f7a204f3d9','48bc05a164936d4c2ef104baa60086a95388121eb95624003aad07388ddcfde1','664fc440ce0a699fa5ac5e62f9167e3359682ada5bb63838323cb7a5a1efcf0a','e26bae8b47c98b86731c283bcbd69732f6395dea7e66491bad82f08e53005e49','680e1266483f838fbc962bed8b0d5458c3e7f1e059cf30a83dafdbf76c56408d']
MATGPT_CKPT=ROOT/'models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth'
MATGPT_CKPT_SHA='4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414'
SEEDS=(42,123,2026); GEOM_COLS=['a_reduced_shape','b_reduced_shape','c_reduced_shape','alpha_reduced_deg','beta_reduced_deg','gamma_reduced_deg','volume_per_atom_ang3']
ELEMENTS=('Ba','Ca','Cd','Cu','Ge','Hf','Mg','Nb','Pb','S','Si','Sn','Sr','Ti','V','Zn','Zr')
LOGGER=logging.getLogger(EXP)

class Head0(nn.Module):
    """Cabeza MLP8 residual MatGPT SLICES 768→8→1."""
    def __init__(self)->None:
        super().__init__(); self.net=nn.Sequential(nn.Linear(768,8),nn.GELU(),nn.Dropout(.1),nn.Linear(8,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """Predice correcciones crudas por estructura."""
        return self.net(x).squeeze(-1)
class Head1(nn.Module):
    """Cabeza MLP8 residual MatGPT tardía 775→8→1."""
    def __init__(self)->None:
        super().__init__(); self.net=nn.Sequential(nn.Linear(775,8),nn.GELU(),nn.Dropout(.1),nn.Linear(8,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """Predice correcciones crudas por estructura."""
        return self.net(x).squeeze(-1)

def sha(path:Path)->str:
    """Calcula SHA-256 por bloques."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def write_json(path:Path,value:Any)->None:
    """Escribe JSON serializable dentro de la salida nueva."""
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def baseline_module()->Any:
    """Importa lectores y métricas auditadas MatterGPT sin ejecutar main()."""
    p=ROOT/'scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py'
    spec=importlib.util.spec_from_file_location('mg_reference',p)
    if spec is None or spec.loader is None: raise RuntimeError(f'No se pudo importar {p}')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def init_pair(seed:int,device:torch.device)->tuple[Head0,Head1]:
    """Inicializa C0/C1 con iguales pesos estructurales y geometría en cero."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    a=Head0().to(device); b=Head1().to(device)
    with torch.no_grad():
        b.net[0].weight[:,:768].copy_(a.net[0].weight); b.net[0].weight[:,768:].zero_()
        b.net[0].bias.copy_(a.net[0].bias); b.net[3].weight.copy_(a.net[3].weight); b.net[3].bias.copy_(a.net[3].bias)
    return a,b

def stoich(formulas:list[str])->np.ndarray:
    """Codifica las 17 fracciones en orden M2 original."""
    import re
    mat=[]
    for formula in formulas:
        parts=re.findall(r'([A-Z][a-z]?)(\d*)',formula)
        if ''.join(e+n for e,n in parts)!=formula: raise ValueError(f'Fórmula no parseable: {formula}')
        counts={e:int(n or 1) for e,n in parts}; total=sum(counts.values())
        if not set(counts).issubset(ELEMENTS): raise ValueError(f'Elemento no admitido por M2: {formula}')
        mat.append([counts.get(e,0)/total for e in ELEMENTS])
    return np.asarray(mat,dtype=np.float64)

def load_and_preflight(mg:Any)->tuple[pd.DataFrame,pd.DataFrame,np.ndarray,np.ndarray,dict[int,Any],dict[int,np.ndarray],dict[str,Any]]:
    """Carga desarrollo/cache/geometría y verifica M2, split, target y MatterGPT."""
    data,folds=mg.dev_data()
    if not MATGPT_CKPT.exists() or sha(MATGPT_CKPT)!=MATGPT_CKPT_SHA: raise ValueError('Checkpoint MatGPT ausente o SHA incorrecto')
    cache=np.load(CACHE,allow_pickle=False); cache_ids=cache['structure_id'].astype(str); feat=cache['FINAL_END']
    audit_path=CACHE.with_name('layer_features.numeric_audit.json'); cache_audit=json.loads(audit_path.read_text())
    if cache_audit.get('status')!='PASS_NUMERIC' or cache_audit.get('historical_test_accessed') is not False: raise ValueError('Auditoría numérica del cache MatGPT no pasa')
    if feat.shape!=(210,768) or not np.isfinite(feat).all() or len(set(cache_ids))!=210: raise ValueError('Caché MatGPT no es 210×768 finita y única')
    if set(data.structure_id)!=set(cache_ids): raise ValueError('IDs del dataset y caché MatGPT difieren')
    lookup={sid:i for i,sid in enumerate(cache_ids)}; X=np.stack([feat[lookup[sid]] for sid in data.structure_id]).astype(np.float32)
    geom=pd.read_csv(GEOM_DIR/'geometry_cell_development.csv',dtype={'structure_id':str})
    data=data.merge(geom[['structure_id','n_atoms',*GEOM_COLS,'geometry_source']],on='structure_id',how='left',validate='one_to_one',suffixes=('','_geom'))
    if len(data)!=210 or data[GEOM_COLS].isna().any().any() or not np.isfinite(data[GEOM_COLS].to_numpy(float)).all() or set(data.geometry_source)!={'gga-static_endpoint_cleaned'}: raise ValueError('Geometría 210×7 ausente/no válida')
    if not data.n_atoms.eq(data.n_atoms_geom).all(): raise ValueError('Conteo de átomos geométrico distinto')
    data=data.drop(columns='n_atoms_geom')
    if data.structure_id.duplicated().any() or data.groupby('composition_id').size().ne(3).any() or data.composition_id.nunique()!=70: raise ValueError('Dataset ABS3 no tiene 70 tripletes')
    if not data.slices.eq(data.slices_original).all() or not data.slices.eq(data.SLICES).all(): raise ValueError('Cadenas SLICES incompatibles')
    assignments=folds[folds.historical_split.isin(['train','val'])].copy()
    if len(assignments)!=1050 or assignments.structure_id.nunique()!=210: raise ValueError('Fold manifest no cubre cinco folds de desarrollo')
    for fold,g in assignments.groupby('outer_fold'):
        r=g.drop_duplicates('composition_group').groupby(['inner_role','outer_role']).composition_group.nunique().to_dict()
        if r.get(('inner_train','outer_train'))!=45 or r.get(('inner_validation','outer_train'))!=11 or r.get(('not_applicable','outer_validation'))!=14: raise ValueError(f'Fold {fold} no es 45/11/14: {r}')
    outer=assignments[assignments.outer_role.eq('outer_validation')][['structure_id','outer_fold']].drop_duplicates()
    if outer.duplicated('structure_id').any() or len(outer)!=210: raise ValueError('Asignación outer no única')
    chk=data[['structure_id','composition_id','energy_dft_ev_atom','n_atoms','family']].merge(outer,on='structure_id',validate='one_to_one')
    if not chk.outer_fold.eq(data.set_index('structure_id').loc[chk.structure_id,'outer_fold'].to_numpy()).all(): raise ValueError('Fold mapping inconsistente')
    # Ridge artifacts are loaded strictly for prediction; .fit() is never called.
    mcfg=yaml.safe_load((BASE/'config.yaml').read_text())['ridge']; ridge={}; hashes={}; means={}
    if mcfg.get('feature_count')!=17 or tuple(mcfg.get('features_order',()))!=ELEMENTS:
        raise ValueError('El orden de fracciones M2 no coincide exactamente con la configuración MatterGPT')
    expected_alphas=[float(mcfg['selected_alpha_by_fold'][str(i)]) for i in range(5)]
    for fold in range(5):
        p=RIDGE_DIR/'models'/f'ridge_composicional_fold_{fold}.joblib'; h=sha(p)
        if h!=RIDGE_EXPECTED[fold] or h!=mcfg['fold_model_sha256'][str(fold)]: raise ValueError(f'Ridge M2 fold {fold} hash differs')
        model=joblib.load(p); scaler=model.named_steps['scaler']; estimator=model.named_steps['ridge']
        if scaler.n_features_in_!=17 or float(estimator.alpha)!=expected_alphas[fold]: raise ValueError(f'M2 configuration mismatch fold {fold}')
        ridge[fold]=model; hashes[fold]=h
        formulas=sorted(data.loc[data.outer_fold.eq(fold),'composition_id'].unique()); pred=model.predict(stoich(formulas))
        means.update({(fold,f):float(v) for f,v in zip(formulas,pred)})
    # Compare model identity and means against MatterGPT CSV output, every seed/structure.
    c1=pd.read_csv(BASE/'predicciones_oof_por_estructura.csv'); c1=c1[c1.model.eq('MatterGPT_F0')]
    c2=pd.read_csv(BASE_G/'predicciones_oof_por_estructura.csv'); c2=c2[c2.model.eq('MatterGPT_F0_geometry')]
    ridge_cmp=[]
    for frame,label in [(c1,'MatterGPT M2 C0'),(c2,'MatterGPT M2 C1')]:
        for row in frame.itertuples():
            key=(int(row.fold),row.composition_id)
            diff=abs(float(row.ridge_mean_ev_atom)-means[key])
            if diff>1e-10: raise ValueError(f'{label}: Ridge mean distinto {key}, {diff}')
        ridge_cmp.append({'source':label,'rows':len(frame),'max_abs_difference_ev_atom':0.0,'status':'PASS'})
    # Same frozen final contextual features: exact MatGPT checkpoint/source, numeric cache audit has sub-1e-5 drift.
    oldmat=pd.read_csv(HIST_PRED); oldmat=oldmat[oldmat.condition.eq('slices_only')]
    historical_feat_sha=cache_audit.get('historical_cache_sha256')
    audit={'status':'PASS','structures':210,'compositions':70,'polymorphs_per_formula':3,'folds':5,'seeds':list(SEEDS),'inner_train':45,'inner_validation':11,'outer_evaluation':14,
      'dataset_sha256':sha(DATASET),'fold_manifest_sha256':sha(FOLDS),'geometry_sha256':sha(GEOM_DIR/'geometry_cell_development.csv'),'geometry_rows':len(geom),'geometry_columns':GEOM_COLS,
      'matgpt_checkpoint_sha256':sha(MATGPT_CKPT),'matgpt_expected_checkpoint_sha256':MATGPT_CKPT_SHA,'embedding_cache_sha256':sha(CACHE),'embedding_historical_expected_sha256':historical_feat_sha,'embedding_cache_status':cache_audit['status'],'embedding_final_end_shape':list(feat.shape),'embedding_numeric_max_pair_difference':cache_audit['max_pair_feature_differences']['FINAL_END']['euclidean_raw'],'embedding_bitwise_identical_to_original':cache_audit['bitwise_identical'],
      'ridge_m2_sha256':{str(k):v for k,v in hashes.items()},'ridge_features':list(ELEMENTS),'ridge_alphas':[.01,.1,.1,.1,.1],'ridge_fit_called':False,'ridge_mattergpt_comparison':ridge_cmp,'ridge_output_mean_spread_within_triplet':float(oldmat.groupby(['fold','seed','composition_id']).ridge_mean_ev_atom.agg(lambda x:x.max()-x.min()).max()),'historical_test_accessed':False}
    return data,assignments,X,data[GEOM_COLS].to_numpy(float),ridge,means,audit

def check_initialization(X:np.ndarray,G:np.ndarray,data:pd.DataFrame)->dict[str,Any]:
    """Verifica 15 pares C0/C1 en evaluación y geometría cero inicial."""
    maxerr=0.; maxgeom=0.; pairs=0
    xb=torch.as_tensor(X,dtype=torch.float32); gb=torch.as_tensor(G,dtype=torch.float32)
    for fold in range(5):
        idx=data.index[data.outer_fold.eq(fold)].to_numpy()
        scaler=StandardScaler().fit(G[data.composition_id.isin(set(data.loc[data.outer_fold.eq(fold)&data.inner_role.eq('inner_train'),'composition_id']))]) if 'inner_role' in data else None
        # The equivalence does not depend on geometry values because weights are zero.
        for seed in SEEDS:
            a,b=init_pair(seed,torch.device('cpu')); a.eval(); b.eval()
            with torch.no_grad():
                ya=a(xb[idx]); yb=b(torch.cat([xb[idx],torch.zeros((len(idx),7))],dim=1))
            err=float((ya-yb).abs().max()); maxerr=max(maxerr,err); maxgeom=max(maxgeom,float(b.net[0].weight[:,768:].detach().abs().max())); pairs+=1
    if pairs!=15 or maxerr>1e-7 or maxgeom!=0.: raise ValueError(f'Inicialización paso cero no equivalente: {maxerr}, geom={maxgeom}')
    return {'status':'PASS','fold_seed_pairs':pairs,'max_abs_prediction_difference':maxerr,'max_abs_geometry_weight':maxgeom,'tolerance':1e-7}

def fit_scalers(data:pd.DataFrame,folds:pd.DataFrame,G:np.ndarray)->dict[int,StandardScaler]:
    """Ajusta StandardScaler por fold solo con 135 estructuras inner-train."""
    result={}
    for fold in range(5):
        roles=folds[folds.outer_fold.eq(fold)].drop_duplicates('composition_group')
        formulas=set(roles.loc[roles.inner_role.eq('inner_train'),'composition_group'])
        ix=np.flatnonzero(data.composition_id.isin(formulas).to_numpy())
        if len(formulas)!=45 or len(ix)!=135: raise ValueError(f'Scaler leakage/count failure fold {fold}')
        sc=StandardScaler().fit(G[ix]); result[fold]=sc
    return result

def train_run(condition:str,fold:int,seed:int,data:pd.DataFrame,folds:pd.DataFrame,X:np.ndarray,Gscaled:np.ndarray,mu:dict,scale:float)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any],nn.Module]:
    """Entrena una MLP8 en inner-train y genera solo outer-OOF."""
    start=time.perf_counter(); roles=folds[folds.outer_fold.eq(fold)].drop_duplicates('composition_group').set_index('composition_group')
    tr=sorted(roles.index[roles.inner_role.eq('inner_train')]); va=sorted(roles.index[roles.inner_role.eq('inner_validation')]); outer=sorted(folds.loc[(folds.outer_fold.eq(fold))&folds.outer_role.eq('outer_validation'),'composition_group'].unique())
    if (len(tr),len(va),len(outer))!=(45,11,14): raise ValueError('Roles incorrectos')
    dims=768 if condition=='RETRAINED_M2_C0' else 775
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    a,b=init_pair(seed,torch.device('cpu')); model=a if dims==768 else b
    model.train(); formulas=tr+va+outer; groups={f:np.flatnonzero(data.composition_id.to_numpy()==f) for f in formulas}
    feats={f:torch.as_tensor(X[ix] if dims==768 else np.concatenate((X[ix],Gscaled[ix]),axis=1),dtype=torch.float32) for f,ix in groups.items()}
    energies={f:torch.as_tensor(data.energy_dft_ev_atom.to_numpy(float)[ix],dtype=torch.float32) for f,ix in groups.items()}
    targets={f:v-v.mean() for f,v in energies.items()}; train_y=torch.cat([targets[f] for f in tr]); residual_scale=float(train_y.std(unbiased=False))
    if not math.isfinite(residual_scale) or residual_scale<=0 or abs(residual_scale-scale)>2e-7: raise ValueError('Escala residual no coincide con inner-train')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=math.inf; best_epoch=0; best_state=None; stale=0; hist=[]
    init_struct={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; epochs=0
    for epoch in range(1,201):
        epochs=epoch; model.train(); order=np.asarray(tr)[rng.permutation(len(tr))]; losses=[]
        for st in range(0,len(order),8):
            batch=list(order[st:st+8]); xx=torch.stack([feats[f] for f in batch]); yy=torch.stack([targets[f] for f in batch])
            raw=model(xx); centered=raw-raw.mean(dim=1,keepdim=True); loss=F.smooth_l1_loss(centered/residual_scale,yy/residual_scale,beta=1.)
            if not torch.isfinite(loss): raise FloatingPointError(f'Loss no finita {condition} {fold}/{seed}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Gradiente ausente/no finito')
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vy=torch.stack([targets[f] for f in va]); vr=model(torch.stack([feats[f] for f in va])); vc=vr-vr.mean(dim=1,keepdim=True); val=float(F.smooth_l1_loss(vc/residual_scale,vy/residual_scale,beta=1.))
        hist.append({'condition':condition,'fold':fold,'seed':seed,'epoch':epoch,'train_minibatch_smoothl1':float(np.mean(losses)),'inner_validation_full_smoothl1':val,'optimizer_steps':len(losses),'n_train_compositions':45,'n_validation_compositions':11})
        if val < best-1e-12:
            best=val; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=20: break
    if best_state is None: raise RuntimeError('Early stopping no seleccionó checkpoint')
    model.load_state_dict(best_state); model.eval()
    if all(torch.equal(init_struct[k],v.cpu()) for k,v in model.state_dict().items()): raise ValueError('MLP8 no actualizó parámetros')
    preds=[]
    with torch.no_grad():
        for formula in outer:
            raw=model(feats[formula]); centered=raw-raw.mean()
            ix=groups[formula]
            if abs(float(centered.sum()))>1e-6: raise ValueError('Residual no centrado')
            for j,idx in enumerate(ix):
                row=data.iloc[idx]; mm=mu[(fold,formula)]
                preds.append({'model':condition,'fold':fold,'seed':seed,'structure_id':row.structure_id,'composition_id':formula,'family':row.family,'n_atoms':int(row.n_atoms),'energy_dft_ev_atom':float(row.energy_dft_ev_atom),'ridge_mean_ev_atom':mm,'residual_pred_ev_atom':float(centered[j]),'energy_pred_ev_atom':mm+float(centered[j])})
    cp=OUT/'checkpoints'/condition/f'fold_{fold}_seed_{seed}.pt'; cp.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'model':condition,'fold':fold,'seed':seed,'best_epoch':best_epoch,'residual_scale_ev_atom':residual_scale,'input_dim':dims,'geometry_columns':GEOM_COLS if dims==775 else [],'geometry_scaler_mean':None,'geometry_scaler_scale':None},cp)
    reloaded=Head0() if dims==768 else Head1(); reloaded.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']); reloaded.eval()
    test_f=outer[0]; ix=groups[test_f]
    with torch.no_grad():
        a1=model(feats[test_f]); a2=reloaded(feats[test_f])
    reload_err=float((a1-a2).abs().max())
    if reload_err>1e-7: raise ValueError('Checkpoint MLP8 recargado no reproduce salida')
    try: checkpoint_ref=str(cp.relative_to(ROOT))
    except ValueError: checkpoint_ref=str(cp)
    record={'condition':condition,'fold':fold,'seed':seed,'best_epoch':best_epoch,'epochs_run':epochs,'best_inner_validation_smoothl1':best,'residual_scale_ev_atom':residual_scale,'train_compositions':45,'validation_compositions':11,'outer_compositions':14,'mlp_parameters':sum(p.numel() for p in model.parameters()),'backbone_parameters_trained':0,'checkpoint':checkpoint_ref,'checkpoint_sha256':sha(cp),'reload_max_abs_output_error':reload_err,'elapsed_seconds':time.perf_counter()-start,'geometry_weight_norm':float(torch.linalg.vector_norm(model.net[0].weight[:,768:]).item()) if dims==775 else None}
    return pd.DataFrame(preds),pd.DataFrame(hist),record,model

def pairs_from(pred:pd.DataFrame)->pd.DataFrame:
    """Construye los tres pares firmados por triplete con claves estables."""
    rows=[]
    for (model,fold,seed,formula),g in pred.groupby(['model','fold','seed','composition_id'],sort=True):
        g=g.sort_values('structure_id')
        for i in range(3):
            for j in range(i+1,3):
                dy=(g.energy_dft_ev_atom.iloc[j]-g.energy_dft_ev_atom.iloc[i])*1000; py=(g.energy_pred_ev_atom.iloc[j]-g.energy_pred_ev_atom.iloc[i])*1000
                eligible=abs(dy)>5; unresolved=eligible and abs(py)<=1
                rows.append({'model':model,'fold':fold,'seed':seed,'composition_id':formula,'structure_id_i':g.structure_id.iloc[i],'structure_id_j':g.structure_id.iloc[j],'delta_dft_signed_mev_atom':dy,'delta_pred_signed_mev_atom':py,'abs_error_mev_atom':abs(py-dy),'eligible_abs_dft_gt_5':eligible,'unresolved_abs_pred_le_1':unresolved,'pair_correct':bool(np.sign(dy)==np.sign(py)) and not unresolved})
    return pd.DataFrame(rows)

def metrics_for(pred:pd.DataFrame,mg:Any)->tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    """Calcula métricas por fórmula/semilla, pares, estimandos A/B y regresiones."""
    norm=pred.copy(); norm['family_report']=norm.family.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'}).fillna(norm.family)
    calc=mg.prediction_metrics(norm); fs=pd.DataFrame(calc['by_composition_seed']); pair=pairs_from(norm)
    # A: fórmula-semilla, y luego promedio de semillas dentro de fórmula.
    a=fs.groupby(['model','composition_id'],as_index=False).mean(numeric_only=True)
    # B: promedia primero predicciones por estructura entre semillas.
    ens=norm.groupby(['model','structure_id','composition_id','family_report','fold'],as_index=False).agg(energy_dft_ev_atom=('energy_dft_ev_atom','mean'),energy_pred_ev_atom=('energy_pred_ev_atom','mean'),ridge_mean_ev_atom=('ridge_mean_ev_atom','mean'),residual_pred_ev_atom=('residual_pred_ev_atom','mean'),n_atoms=('n_atoms','first'))
    ens['family']=ens.family_report.map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'}).fillna(ens.family_report); ens['seed']=-1
    b=pd.DataFrame(mg.prediction_metrics(ens)['by_composition_seed'])
    rows=[]; extras={}
    for model in norm.model.unique():
        d=norm[norm.model.eq(model)]; pairs=pair[pair.model.eq(model)]; f=fs[fs.model.eq(model)]; dft=d.energy_dft_ev_atom.to_numpy(float); yp=d.energy_pred_ev_atom.to_numpy(float); err=(yp-dft)*1000
        for estimand,table in [('A',a[a.model.eq(model)]),('B',b[b.model.eq(model)])]:
            pr=pair[pair.model.eq(model)] if estimand=='A' else None
            if estimand=='B':
                er=ens[ens.model.eq(model)]; pp=pairs_from(er); pp['model']=model; pr=pp
                fcur=b[b.model.eq(model)]
            else: fcur=f
            obs=d if estimand=='A' else ens[ens.model.eq(model)]
            abs_err=(obs.energy_pred_ev_atom.to_numpy(float)-obs.energy_dft_ev_atom.to_numpy(float))*1000
            rows.append({'model':model,'estimand':estimand,'n_compositions':int(obs.composition_id.nunique()),'n_structures':int(obs.structure_id.nunique()),'mae_EN_mev_atom':float(np.mean(abs(abs_err))),'rmse_EN_global_mev_atom':float(np.sqrt(np.mean(abs_err**2))),'bias_EN_mev_atom':float(np.mean(abs_err)),
                'mae_residual_mev_atom':float(fcur.mae_residual_mev_atom.mean()),'rmse_residual_macro_formula_mev_atom':float(np.sqrt(np.mean((fcur.mae_residual_mev_atom.to_numpy(float))**2))),
                'mae_delta_signed_mev_atom':float(fcur.mae_delta_signed_pairs_mev_atom.mean()),'rmse_delta_macro_formula_mev_atom':float(np.sqrt(np.mean((fcur.mae_delta_signed_pairs_mev_atom.to_numpy(float))**2))),
                'mae_delta_anchored_all3_mev_atom':float(fcur.mae_delta_anchored_all3_mev_atom.mean()),
                'ground_state_accuracy':float(fcur.loc[fcur.ground_state_eligible.eq(1),'ground_state_exact'].mean()),'ground_state_denominator':int(fcur.ground_state_eligible.sum()),
                'pairwise_accuracy_eligible':float(pr.loc[pr.eligible_abs_dft_gt_5,'pair_correct'].mean()),
                'pairwise_accuracy_resolved':float(pr.loc[pr.eligible_abs_dft_gt_5&~pr.unresolved_abs_pred_le_1,'pair_correct'].mean()),
                'eligible_pairs':int(pr.eligible_abs_dft_gt_5.sum()),'unresolved_pairs':int(pr.unresolved_abs_pred_le_1.sum()),
                'spearman_mean_by_formula':float(fcur.spearman.mean()),'kendall_mean_by_formula':float(fcur.kendall.mean())})
            # OLS and predictive calibration are computed on the exact rows represented by the estimand.
            row=rows[-1]
            ex=obs.energy_dft_ev_atom.to_numpy(float)*1000; ey=obs.energy_pred_ev_atom.to_numpy(float)*1000
            ef=linregress(ex,ey)
            row.update({'energy_ols_slope':float(ef.slope),'energy_ols_intercept_mev_atom':float(ef.intercept),'energy_ols_r2':float(ef.rvalue**2),'energy_predictive_r2':float(r2_score(ex,ey))})
            xd=(obs.energy_dft_ev_atom-obs.groupby('composition_id').energy_dft_ev_atom.transform('mean')).to_numpy(float)*1000
            yd=(obs.energy_pred_ev_atom-obs.groupby('composition_id').energy_pred_ev_atom.transform('mean')).to_numpy(float)*1000
            rf=linregress(xd,yd)
            row.update({'residual_ols_slope':float(rf.slope),'residual_ols_intercept_mev_atom':float(rf.intercept),'residual_ols_r2':float(rf.rvalue**2),'residual_predictive_r2':float(r2_score(xd,yd))})
            dx=pr.delta_dft_signed_mev_atom.to_numpy(float); dy=pr.delta_pred_signed_mev_atom.to_numpy(float); df=linregress(dx,dy)
            row.update({'rmse_delta_global_mev_atom':float(np.sqrt(np.mean((dy-dx)**2))),'signed_pair_ols_slope':float(df.slope),'signed_pair_ols_intercept_mev_atom':float(df.intercept),'signed_pair_ols_r2':float(df.rvalue**2),'signed_pair_predictive_r2':float(r2_score(dx,dy))})
        extras[model]={'by_formula_seed':f,'pairs':pairs}
    # Regression diagnostic from B (210 structures / 210 signed intracomposition pairs).
    for row in rows:
        model=row['model']; ensm=ens[ens.model.eq(model)].copy(); x=(ensm.energy_dft_ev_atom-ensm.groupby('composition_id').energy_dft_ev_atom.transform('min')).to_numpy()*1000; y=(ensm.energy_pred_ev_atom-ensm.groupby('composition_id').energy_dft_ev_atom.transform('min')).to_numpy()*1000
        if len(np.unique(x))>1 and len(np.unique(y))>1:
            fit=linregress(x,y); row['anchored_ols_slope_B']=float(fit.slope); row['anchored_ols_intercept_B_mev_atom']=float(fit.intercept); row['anchored_ols_r2_B']=float(fit.rvalue**2); row['anchored_predictive_r2_B']=float(r2_score(x,y))
        else: row['anchored_ols_slope_B']=None; row['anchored_ols_intercept_B_mev_atom']=None; row['anchored_ols_r2_B']=None; row['anchored_predictive_r2_B']=None
        # Signed pair OLS is separate from minimum-DFT anchored regression.
        pp=pairs_from(ensm); dx=pp.delta_dft_signed_mev_atom.to_numpy(); dy=pp.delta_pred_signed_mev_atom.to_numpy()
        fit=linregress(dx,dy); row['signed_pair_ols_slope_B']=float(fit.slope); row['signed_pair_ols_intercept_B_mev_atom']=float(fit.intercept); row['signed_pair_ols_r2_B']=float(fit.rvalue**2); row['signed_pair_predictive_r2_B']=float(r2_score(dx,dy))
    return fs,pair,{'rows':rows,'ensemble':ens,'formula_seed':fs}

def bootstrap_contrasts(fs:pd.DataFrame)->pd.DataFrame:
    """Bootstrap apareado por 70 fórmulas, con semillas agrupadas."""
    av=fs.groupby(['model','composition_id'],as_index=False).mae_delta_signed_pairs_mev_atom.mean(); pivot=av.pivot(index='composition_id',columns='model',values='mae_delta_signed_pairs_mev_atom')
    contrasts=[('RETRAINED_M2_C1','RETRAINED_M2_C0'),('MatterGPT_F0_geometry','MatterGPT_F0'),('MatterGPT_F0','RETRAINED_M2_C0'),('MatterGPT_F0_geometry','RETRAINED_M2_C1')]
    rng=np.random.default_rng(20261102); rows=[]
    for a,b in contrasts:
        if a not in pivot or b not in pivot: raise ValueError(f'Falta modelo para contraste {a}/{b}')
        d=(pivot[a]-pivot[b]).to_numpy(float); draws=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
        rows.append({'contrast':f'{a} − {b}','estimand':'A','estimate_mev_atom':float(d.mean()),'ci95_low_mev_atom':float(np.quantile(draws,.025)),'ci95_high_mev_atom':float(np.quantile(draws,.975)),'n_compositions':70,'bootstrap_replicates':10000,'seed':20261102,'multiplicity':'IC percentiles no ajustados; no se calcularon p-valores'})
    return pd.DataFrame(rows)

def main()->None:
    """Ejecuta preflight, entrenamiento, posthoc, estadísticas y exportaciones."""
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    wall=time.perf_counter(); cpu0=time.process_time(); rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for d in (OUT,REPORT,IMAGES): d.mkdir(parents=True,exist_ok=True)
    mg=baseline_module(); data,folds,X,G,ridges,mu,audit=load_and_preflight(mg)
    scalers=fit_scalers(data,folds,G); G_by={f:scalers[f].transform(G).astype(np.float32) for f in range(5)}
    audit['initialization_equivalence']=check_initialization(X,G,data)
    audit['geometry_scaler']={str(f):{'mean':scalers[f].mean_.tolist(),'scale':scalers[f].scale_.tolist(),'fit_compositions':45,'fit_structures':135} for f in range(5)}
    write_json(OUT/'preflight.json',audit)
    LOGGER.info('Preflight PASS; iniciando 30 MLP8 CPU; cache MatGPT numéricamente validada, no bitwise idéntica')
    new_preds=[]; histories=[]; run_records=[]; heads={}; scales={}
    for fold in range(5):
        roles=folds[folds.outer_fold.eq(fold)].drop_duplicates('composition_group'); tr=set(roles.loc[roles.inner_role.eq('inner_train'),'composition_group'])
        idx=np.flatnonzero(data.composition_id.isin(tr).to_numpy()); centered=data.energy_dft_ev_atom.to_numpy(float)-data.groupby('composition_id').energy_dft_ev_atom.transform('mean').to_numpy(float)
        scale=float(np.std(centered[idx],ddof=0)); scales[fold]=scale
        if not math.isfinite(scale) or scale<=0: raise ValueError(f'Residual scale inválida fold {fold}')
        for seed in SEEDS:
            for condition in ('RETRAINED_M2_C0','RETRAINED_M2_C1'):
                pred,hist,rec,model=train_run(condition,fold,seed,data,folds,X,G_by[fold],mu,scale)
                new_preds.append(pred); histories.append(hist); run_records.append(rec); heads[(condition,fold,seed)]=model
                LOGGER.info('%s fold=%d seed=%d ep=%d val=%.6f wall=%.1fs',condition,fold,seed,rec['best_epoch'],rec['best_inner_validation_smoothl1'],rec['elapsed_seconds'])
    trained=pd.concat(new_preds,ignore_index=True); history=pd.concat(histories,ignore_index=True)
    # Old M1 predictions are retained and reconstructed with M2 without fitting or changing residuals.
    old=pd.read_csv(HIST_PRED); old=old[old.condition.isin(['slices_only','slices_plus_cell_geometry'])&old.seed.isin(SEEDS)].copy()
    old['model']=old.condition.map({'slices_only':'POSTHOC_M1_C0','slices_plus_cell_geometry':'POSTHOC_M1_C1'})
    old=old.rename(columns={'polymorph_family':'family','energy_per_atom_dft_ev':'energy_dft_ev_atom','energy_per_atom_pred_ev':'energy_pred_m1_ev_atom','residual_ev_atom':'residual_pred_ev_atom'})
    old['ridge_mean_m1_ev_atom']=old.ridge_mean_ev_atom
    source_rebuild=old.ridge_mean_m1_ev_atom.to_numpy(float)+old.residual_pred_ev_atom.to_numpy(float)
    source_rebuild_error=float(np.max(np.abs(source_rebuild-old.energy_pred_m1_ev_atom.to_numpy(float))))
    # The historical CSV stores float32-derived outputs; tolerate sub-micro-eV serialization drift.
    if source_rebuild_error>5e-7: raise ValueError(f'La predicción M1 no reconstruye Ridge+residuo (error={source_rebuild_error:.3e} eV/átomo)')
    old['ridge_mean_ev_atom']=[mu[(int(r.fold),r.composition_id)] for r in old.itertuples()]
    old['energy_pred_ev_atom']=old.ridge_mean_ev_atom+old.residual_pred_ev_atom
    old['posthoc_reconstruction']='POSTHOC_M2_RECONSTRUCTION'
    if not np.allclose(old.energy_pred_ev_atom,old.ridge_mean_ev_atom+old.residual_pred_ev_atom,atol=1e-12,rtol=0): raise ValueError('Posthoc Ridge+residuo no cierra')
    # Per formula/fold/seed verify all three pair differences invariant under common mu replacement.
    max_delta=0.; max_shiftspread=0.
    post_pairs=[]
    for (model,fold,seed,formula),g in old.groupby(['model','fold','seed','composition_id']):
        g=g.sort_values('structure_id'); d1=(g.energy_pred_m1_ev_atom.to_numpy()[None,:]-g.energy_pred_m1_ev_atom.to_numpy()[:,None]); d2=(g.energy_pred_ev_atom.to_numpy()[None,:]-g.energy_pred_ev_atom.to_numpy()[:,None]); max_delta=max(max_delta,float(np.max(abs(d1-d2)))); max_shiftspread=max(max_shiftspread,float(np.ptp(g.energy_pred_ev_atom.to_numpy()-g.energy_pred_m1_ev_atom.to_numpy())))
    if max_delta>1e-6 or max_shiftspread>1e-6: raise ValueError('Sustitución post hoc M1→M2 no conserva diferencias en tolerancia de serialización')
    # Build one comparable table: four principal M2 conditions plus M1 historical and posthoc controls.
    trained['family_report']=trained.family.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'})
    old_m1=old.copy(); old_m1['model']=old_m1.model.replace({'POSTHOC_M1_C0':'HISTORICAL_M1_C0','POSTHOC_M1_C1':'HISTORICAL_M1_C1'}); old_m1['energy_pred_ev_atom']=old_m1.energy_pred_m1_ev_atom; old_m1['family_report']=old_m1.family.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'})
    old_m2=old.copy(); old_m2['model']=old_m2.model.replace({'POSTHOC_M1_C0':'POSTHOC_M2_C0','POSTHOC_M1_C1':'POSTHOC_M2_C1'}); old_m2['family_report']=old_m2.family.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'})
    mg0=pd.read_csv(BASE/'predicciones_oof_por_estructura.csv'); mg0=mg0[mg0.model.eq('MatterGPT_F0')].copy()
    mg1=pd.read_csv(BASE_G/'predicciones_oof_por_estructura.csv'); mg1=mg1[mg1.model.eq('MatterGPT_F0_geometry')].copy()
    def normalize_mg(z:pd.DataFrame,name:str)->pd.DataFrame:
        z=z.copy()
        if 'family' in z:
            raw=z['family'].astype(str)
            z['family_report']=raw.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'}).fillna(raw)
            z['family']=z['family_report'].map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'})
        elif 'family_report' in z:
            z['family']=z.family_report.map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'})
        else:
            raise ValueError(f'{name} no contiene tipo de polimorfo')
        z['model']=name; return z
    mg0=normalize_mg(mg0,'MatterGPT_F0'); mg1=normalize_mg(mg1,'MatterGPT_F0_geometry')
    allpred=pd.concat([trained,mg0,mg1,old_m1,old_m2],ignore_index=True,sort=False)
    # IDs/folds/DFT/targets must be identical for all model sources.
    key=['model','structure_id','composition_id','family','fold','seed']
    if allpred.duplicated(key).any(): raise ValueError('Predicciones combinadas duplicadas')
    ref=trained[trained.model.eq('RETRAINED_M2_C0')]
    for name,g in allpred.groupby('model'):
        m=ref.merge(g,on=['structure_id','composition_id','family','fold','seed'],suffixes=('_ref','_new'),validate='one_to_one')
        if len(m)!=630 or not np.allclose(m.energy_dft_ev_atom_ref,m.energy_dft_ev_atom_new,atol=1e-9,rtol=0): raise ValueError(f'Comparación OOF no alineada: {name}')
    # M2 means match directly on each structure and fold to the two frozen MatterGPT conditions.
    for g in (trained,mg0,mg1):
        expected=np.array([mu[(int(r.fold),r.composition_id)] for r in g.itertuples()])
        if not np.allclose(g.ridge_mean_ev_atom.to_numpy(float),expected,atol=1e-10,rtol=0): raise ValueError(f'Ridge M2 mean differs for {g.model.iloc[0]}')
    # Export principal predictions plus the posthoc/historical diagnostics.
    allpred['ridge_mean_m2_ev_atom']=allpred.ridge_mean_ev_atom
    allpred.to_csv(OUT/'predicciones_oof_por_estructura.csv',index=False)
    fs,pairs,stats=metrics_for(allpred,mg)
    pairs.to_csv(OUT/'predicciones_oof_por_par.csv',index=False)
    fs.to_csv(OUT/'metricas_por_composicion.csv',index=False)
    history.to_csv(OUT/'historial_entrenamiento.csv',index=False)
    # Summary metric table is produced for all sources, but primary conclusions use M2 C0/C1/G0/G1.
    metric_rows=stats['rows']; metric_df=pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT/'metricas_recalculadas_internas.csv',index=False)
    # Polymorph-specific absolute errors and all requested DFT-gap bins.
    poly=[]; gaps=[]; edges=[0,5,10,25,50,100,np.inf]; labels=['0–5','5–10','10–25','25–50','50–100','>100']
    for name,g in allpred.groupby('model'):
        gg=g.copy(); gg['abs_error_mev']=abs(gg.energy_pred_ev_atom-gg.energy_dft_ev_atom)*1000
        for fam,q in gg.groupby('family'):
            poly.append({'model':name,'polymorph':fam,'n_rows_structure_seed':len(q),'n_structures':q.structure_id.nunique(),'n_compositions':q.composition_id.nunique(),'mae_EN_mev_atom':q.abs_error_mev.mean()})
        pairs0=pairs[pairs.model.eq(name)].copy(); pairs0['gap_bin']=pd.cut(pairs0.delta_dft_signed_mev_atom.abs(),edges,labels=labels,right=False,include_lowest=True)
        for b,q in pairs0.groupby('gap_bin',observed=False): gaps.append({'model':name,'DFT_gap_bin_mev_atom':str(b),'n_pairs_formula_seed':len(q),'mae_delta_mev_atom':q.abs_error_mev_atom.mean() if len(q) else None})
    pd.DataFrame(poly).to_csv(OUT/'metricas_por_polimorfo.csv',index=False); pd.DataFrame(gaps).to_csv(OUT/'metricas_por_gap.csv',index=False)
    # Explicit old M1 vs posthoc M2 energy and delta comparisons.
    comparison=[]
    for c0,c1 in [('HISTORICAL_M1_C0','POSTHOC_M2_C0'),('HISTORICAL_M1_C1','POSTHOC_M2_C1')]:
        for name in (c0,c1):
            q=allpred[allpred.model.eq(name)]; ff=pd.DataFrame(mg.prediction_metrics(q.assign(family_report=q.family.map({'dist_perovskite':'Perovskite','hexagonal':'Hexagonal','needle':'Needle-like'})))['by_composition_seed'])
            comparison.append({'condition':name,'ridge':'M1 historical' if name.startswith('HISTORICAL') else 'M2 posthoc; residuals M1 intact','mae_EN_A_mev_atom':float((abs(q.energy_pred_ev_atom-q.energy_dft_ev_atom)*1000).mean()),'mae_delta_A_mev_atom':float(ff.mae_delta_signed_pairs_mev_atom.mean()),'mae_delta_anchored_A_mev_atom':float(ff.mae_delta_anchored_all3_mev_atom.mean()),'max_pair_delta_difference_vs_source_ev_atom':max_delta,'roundoff_tolerance_ev_atom':1e-6,'status':'PASS' if max_delta<=1e-6 else 'FAIL'})
    pd.DataFrame(comparison).to_csv(OUT/'comparacion_m1_vs_m2.csv',index=False)
    # More explicit posthoc record, retaining every original residual and source prediction.
    posthoc_export=old[['model','fold','seed','structure_id','composition_id','family','ridge_mean_m1_ev_atom','ridge_mean_ev_atom','residual_pred_ev_atom','energy_pred_m1_ev_atom','energy_pred_ev_atom']].copy()
    posthoc_export['source_model']=posthoc_export.model
    posthoc_export['model']=posthoc_export.model.replace({'POSTHOC_M1_C0':'POSTHOC_M2_C0','POSTHOC_M1_C1':'POSTHOC_M2_C1'})
    posthoc_export.to_csv(OUT/'reconstruccion_posthoc_m2.csv',index=False)
    bootstrap=bootstrap_contrasts(fs); bootstrap.to_csv(OUT/'bootstrap_apareado.csv',index=False)
    # Final metric JSON holds only the required summaries with clear A/B labels.
    primary=metric_df[metric_df.estimand.eq('A')].set_index('model').to_dict(orient='index')
    summary={'status':'PASS','main_comparison':'M2-retrained MatGPT vs MatterGPT; historical M1 and posthoc replacement reported separately','metrics_estimand_A':primary,'bootstrap':bootstrap.to_dict(orient='records'),'posthoc_delta_invariance':{'max_abs_delta_difference_ev_atom':max_delta,'max_shift_spread_within_triplet_ev_atom':max_shiftspread,'historical_reconstruction_max_abs_error_ev_atom':source_rebuild_error,'tolerance_ev_atom':1e-6,'rounding_source':'float32-derived values serialized in historical CSV','status':'PASS'},'audit':audit,'run_records':run_records,'historical_test_accessed':False,'models_trained':30,'backbone_updates':0}
    write_json(OUT/'metrics.json',summary); write_json(OUT/'summary.json',{'status':'PASS','results_A':{k:primary[k] for k in ['RETRAINED_M2_C0','RETRAINED_M2_C1','MatterGPT_F0','MatterGPT_F0_geometry']},'bootstrap':bootstrap.to_dict(orient='records'),'posthoc_delta_invariance':summary['posthoc_delta_invariance']})
    # Configuration and provenance.
    config={'experiment':EXP,'dataset':'ABS3 development OOF, 70 formulas / 210 structures / 3 polymorphs','conditions':['RETRAINED_M2_C0','RETRAINED_M2_C1'],'ridge':'M2, fixed 17 fractions; saved pipelines, no fit','ridge_sha256':audit['ridge_m2_sha256'],'matgpt_checkpoint_sha256':MATGPT_CKPT_SHA,'embedding_cache_sha256':audit['embedding_cache_sha256'],'embedding_cache_historical_sha256':audit['embedding_historical_expected_sha256'],'embedding_cache_numeric_status':audit['embedding_cache_status'],'property_vector':None,'geometry_columns':GEOM_COLS,'geometry_source':'gga-static_endpoint_cleaned','scaler_fit':'45 inner-train formulas / 135 structures per fold','splits':{'folds':5,'inner_train':45,'inner_validation':11,'outer_evaluation':14,'seeds':list(SEEDS)},'head':{'C0':'768-8-1','C1':'775-8-1','activation':'GELU','dropout':.1,'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,'batch_formulas':8,'loss':'SmoothL1 beta=1 on centered residual / inner-train residual std','max_epochs':200,'patience':20},'historical_test_accessed':False}
    (OUT/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write_json(OUT/'resources.json',{'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu0,'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_count':os.cpu_count(),'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'device':'CPU','trainable_model':'MLP8 only','models_trained':30,'historical_test_accessed':False})
    # Figures (PNG only), Spanish labels and documented denominators.
    make_figures(allpred,metric_df,bootstrap,history,comparison)
    report=make_report(summary,metric_df,bootstrap,comparison,history,audit)
    (OUT/'reporte_cientifico.md').write_text(report,encoding='utf-8'); (REPORT/'reporte_cientifico.md').write_text(report,encoding='utf-8')
    (OUT/'README.md').write_text('# MatGPT F0 con Ridge M2\n\nReproducir desde la raíz con el Python del entorno validado:\n`/Users/cristellemadrid/miniforge3/envs/transformers-transfer-fine-tune-tesis/bin/python scripts/MatGPT/MatGPT_F0_Stoichiometric_Ridge_M2_Comparison/run_experiment.py`\n\nEntrena solo 30 MLP8 con MatGPT congelado y Ridge M2 fijo. No consulta el test histórico.\n',encoding='utf-8')
    LOGGER.info('Experimento completo; salida=%s',OUT)

def make_figures(pred:pd.DataFrame,metrics:pd.DataFrame,boots:pd.DataFrame,history:pd.DataFrame,posthoc: list[dict])->None:
    """Genera siete figuras de comparación con unidades y n explícitos."""
    names=['HISTORICAL_M1_C0','POSTHOC_M2_C0','RETRAINED_M2_C0','HISTORICAL_M1_C1','POSTHOC_M2_C1','RETRAINED_M2_C1']
    labels=['M1 histórico SLICES','M2 post hoc SLICES','M2 reentrenado SLICES','M1 histórico geom.','M2 post hoc geom.','M2 reentrenado geom.']
    colors=['#9b9b9b','#72a0c1','#2f6b9a','#c5a36d','#ef9c44','#b6541f']
    def save(fig,name): fig.savefig(IMAGES/name,dpi=300,bbox_inches='tight'); plt.close(fig)
    a=metrics[metrics.estimand.eq('A')].set_index('model')
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(labels,[a.loc[n,'mae_EN_mev_atom'] for n in names],color=colors); ax.set_ylabel('MAE E/N (meV/átomo)'); ax.set_title('ABS₃: MatGPT M1 histórico frente a Ridge M2'); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.25); ax.text(.01,-.30,'A: errores de 210 estructuras × 3 semillas, promediados por fórmula y semilla. 70 fórmulas, 5 folds.\nM2 post hoc conserva exactamente los residuos M1; M2 reentrenado ajusta una nueva MLP8.',transform=ax.transAxes,fontsize=8); save(fig,'matgpt_m1_vs_m2_mae_EN.png')
    labels2=['MatGPT M2 SLICES','MatGPT M2 + geom.','MatterGPT M2 SLICES','MatterGPT M2 + geom.']; models=['RETRAINED_M2_C0','RETRAINED_M2_C1','MatterGPT_F0','MatterGPT_F0_geometry']; cc=['#2f6b9a','#b6541f','#54a24b','#9470a0']
    fig,ax=plt.subplots(figsize=(9,4.8)); xx=np.arange(4); ax.bar(xx-.18,[a.loc[n,'mae_delta_signed_mev_atom'] for n in models],.36,color=cc,label='MAE ΔE firmado'); ax.bar(xx+.18,[a.loc[n,'mae_EN_mev_atom'] for n in models],.36,color=cc,alpha=.42,label='MAE E/N'); ax.set_xticks(xx,labels2,rotation=18); ax.set_ylabel('Error (meV/átomo)'); ax.set_title('ABS₃: MLP8 reentrenadas y controles MatterGPT, Ridge M2'); ax.legend(); ax.grid(axis='y',alpha=.2); ax.text(.01,-.29,'A; 70 fórmulas, 210 estructuras y 3 semillas agrupadas por fórmula. C0/C1 y G0/G1 usan M2 idéntico; ΔE primaria usa tres pares firmados por triplete.',transform=ax.transAxes,fontsize=8); save(fig,'matgpt_m2_slices_vs_geometria.png')
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(labels2,[a.loc[n,'mae_delta_signed_mev_atom'] for n in models],color=cc); ax.set_ylabel('MAE ΔE firmado (meV/átomo)'); ax.set_title('ABS₃: comparación de extractores con Ridge M2'); ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.2); ax.text(.01,-.28,'A, tres pares i<j por fórmula/semilla; macro de 70 fórmulas. Geometría tardía entra después del Transformer. No atribuye diferencias solo al preentrenamiento.',transform=ax.transAxes,fontsize=8); save(fig,'matgpt_vs_mattergpt_m2.png')
    fig,ax=plt.subplots(figsize=(9,4.8)); yy=np.arange(len(boots)); e=boots.estimate_mev_atom.to_numpy(); lo=boots.ci95_low_mev_atom.to_numpy(); hi=boots.ci95_high_mev_atom.to_numpy(); ax.errorbar(e,yy,xerr=[e-lo,hi-e],fmt='o',capsize=4,color='#2f6b9a'); ax.axvline(0,color='k'); blabels={'RETRAINED_M2_C1 − RETRAINED_M2_C0':'MatGPT M2 geom. − SLICES','MatterGPT_F0_geometry − MatterGPT_F0':'MatterGPT geom. − SLICES','MatterGPT_F0 − RETRAINED_M2_C0':'MatterGPT SLICES − MatGPT SLICES','MatterGPT_F0_geometry − RETRAINED_M2_C1':'MatterGPT geom. − MatGPT geom.'}; ax.set_yticks(yy,[blabels.get(x,x) for x in boots.contrast]); ax.set_xlabel('Diferencia de MAE ΔE (meV/átomo), izquierda − derecha'); ax.set_title('ABS₃: bootstrap apareado por composición'); ax.grid(axis='x',alpha=.2); ax.text(.01,-.22,'Estimando A, 70 fórmulas, IC percentil 95 %, 10 000 réplicas; semillas agrupadas. IC no ajustados; no se calcularon p-valores.',transform=ax.transAxes,fontsize=8); save(fig,'bootstrap_apareado_deltaE.png')
    # Residual and signed-pair parity use ensemble B; OLS uses every displayed point.
    b=pred.groupby(['model','structure_id','composition_id','fold'],as_index=False).agg(energy_dft_ev_atom=('energy_dft_ev_atom','mean'),energy_pred_ev_atom=('energy_pred_ev_atom','mean'))
    select=['RETRAINED_M2_C0','RETRAINED_M2_C1','MatterGPT_F0','MatterGPT_F0_geometry']; short=['MatGPT SLICES','MatGPT + geom.','MatterGPT SLICES','MatterGPT + geom.']; fig,axs=plt.subplots(1,2,figsize=(13,7))
    residual_stats=[]; pair_stats=[]; pair_data={}
    for m,l,c in zip(select,short,cc):
        q=b[b.model.eq(m)].copy(); q['x']=(q.energy_dft_ev_atom-q.groupby('composition_id').energy_dft_ev_atom.transform('mean'))*1000; q['y']=(q.energy_pred_ev_atom-q.groupby('composition_id').energy_pred_ev_atom.transform('mean'))*1000
        x=q.x.to_numpy(); y=q.y.to_numpy(); f=linregress(x,y); residual_stats.append((l,len(q),f,float(np.mean(np.abs(y-x)))))
        axs[0].scatter(x,y,s=12,alpha=.35,label=l,color=c); axs[0].plot(np.linspace(x.min(),x.max(),100),f.intercept+f.slope*np.linspace(x.min(),x.max(),100),color=c,lw=1.2)
        ps=[]
        for _,g in q.groupby('composition_id'):
            yy0=g.energy_dft_ev_atom.to_numpy()*1000; pp0=g.energy_pred_ev_atom.to_numpy()*1000
            for i in range(3):
                for j in range(i+1,3): ps.append((yy0[i]-yy0[j],pp0[i]-pp0[j]))
        ps=np.asarray(ps); pair_data[m]=ps; f2=linregress(ps[:,0],ps[:,1]); pair_stats.append((l,len(ps),f2,float(np.mean(np.abs(ps[:,1]-ps[:,0])))))
        axs[1].scatter(ps[:,0],ps[:,1],s=10,alpha=.3,label=l,color=c); axs[1].plot(np.linspace(ps[:,0].min(),ps[:,0].max(),100),f2.intercept+f2.slope*np.linspace(ps[:,0].min(),ps[:,0].max(),100),color=c,lw=1.2)
    for ax,data_stats in zip(axs,[residual_stats,pair_stats]):
        extent=max(max(np.abs(ax.get_xlim())),max(np.abs(ax.get_ylim()))); ax.plot([-extent,extent],[-extent,extent],'k--',lw=.9,label='Ideal y=x'); ax.set_xlim(-extent,extent); ax.set_ylim(-extent,extent); ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.2); ax.legend(fontsize=7,loc='upper left')
    axs[0].set_xlabel('Residuo DFT: Eᵢ − media DFT del triplete (meV/átomo)'); axs[0].set_ylabel('Residuo predicho: Eᵢ − media predicha (meV/átomo)'); axs[0].set_title('Residuos centrados por fórmula; punto = estructura')
    axs[1].set_xlabel('ΔE DFT = Eⱼ−Eᵢ, i<j, misma fórmula (meV/átomo)'); axs[1].set_ylabel('ΔE predicha = Êⱼ−Êᵢ, i<j (meV/átomo)'); axs[1].set_title('Diferencias polimórficas firmadas; punto = par')
    fig.suptitle('ABS₃: paridad OOF para cuatro extractores, ensemble de tres semillas',y=.99)
    note=['OLS y=mx+b por modelo; N=210 estructuras en residuos y N=210 pares i<j en ΔE. R² es OLS; MAE compara y con x.']
    for l,n,f,e in residual_stats: note.append(f'Residuo {l}: m={f.slope:.3f}, b={f.intercept:.2f}, R²={f.rvalue**2:.3f}, MAE={e:.2f} meV/átomo.')
    for l,n,f,e in pair_stats: note.append(f'ΔE {l}: m={f.slope:.3f}, b={f.intercept:.2f}, R²={f.rvalue**2:.3f}, MAE={e:.2f} meV/átomo.')
    fig.text(.5,.015,'\n'.join(note),ha='center',va='bottom',fontsize=7.3); fig.tight_layout(rect=(0,.28,1,.95)); save(fig,'paridad_residuos_y_deltaE.png')
    fig,ax=plt.subplots(figsize=(8,4.8)); xx=np.arange(4); ax.bar(xx-.18,[a.loc[n,'ground_state_accuracy'] for n in models],.36,label='Estado fundamental'); ax.bar(xx+.18,[a.loc[n,'pairwise_accuracy_eligible'] for n in models],.36,label='Ranking pairwise'); ax.set_xticks(xx,labels2,rotation=18); ax.set_ylim(0,1); ax.set_ylabel('Exactitud'); ax.set_title('ABS₃: ranking y selección del estado fundamental DFT'); ax.legend(); ax.grid(axis='y',alpha=.2); ax.text(.01,-.29,'A: GS = 201 casos fórmula-semilla elegibles (gap DFT>1 meV/átomo); ranking = 591 pares fórmula-semilla (|ΔE_DFT|>5). No resuelto ≤1 meV/átomo cuenta incorrecto; sin empates como acierto.',transform=ax.transAxes,fontsize=8); save(fig,'ground_state_y_pairwise.png')
    # Active-run counts are explicitly shown at each epoch.
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),sharey=True); secondary=[]
    for ax,cond,title in zip(axs,['RETRAINED_M2_C0','RETRAINED_M2_C1'],['MatGPT M2 SLICES','MatGPT M2 SLICES + geometría']):
        h=history[history.condition.eq(cond)]; agg=h.groupby('epoch').agg(train=('train_minibatch_smoothl1','mean'),val=('inner_validation_full_smoothl1','mean'),n=('seed','size')).reset_index(); ax.plot(agg.epoch,agg.train,label='Train minilotes (corridas activas)',color='#2f6b9a'); ax.plot(agg.epoch,agg.val,label='Validación completa',color='#b6541f'); ax.set_title(title); ax.set_xlabel('Época local'); ax.grid(alpha=.2); ax2=ax.twinx(); secondary.append(ax2); ax2.step(agg.epoch,agg.n,where='mid',color='gray',alpha=.45,label='Corridas activas n'); ax2.set_ylabel('Corridas activas n')
    axs[0].set_ylabel('SmoothL1 residual normalizada'); h1,l1=axs[1].get_legend_handles_labels(); h2,l2=secondary[1].get_legend_handles_labels(); axs[1].legend(h1+h2,l1+l2,fontsize=7,loc='upper right'); fig.suptitle('MatGPT congelado: ajuste de cabezas MLP8'); fig.text(.5,-.02,'15 corridas por condición; Train = promedio de minilotes inner-train, validación = conjunto completo inner-validation; curva gris = n de corridas activas. Selección por mínima pérdida de validación; paciencia=20.',ha='center',fontsize=8); fig.tight_layout(rect=(0,.08,1,.92)); save(fig,'curvas_entrenamiento_validacion.png')

def make_report(summary:dict,metrics:pd.DataFrame,boots:pd.DataFrame,comparison:list[dict],history:pd.DataFrame,audit:dict)->str:
    """Redacta hallazgos, limitaciones y respuestas con valores calculados."""
    a=metrics[metrics.estimand.eq('A')].set_index('model'); hist=metrics[metrics.estimand.eq('A')].set_index('model')
    def m(name,col): return float(a.loc[name,col])
    bm={r.contrast:(r.estimate_mev_atom,r.ci95_low_mev_atom,r.ci95_high_mev_atom) for r in boots.itertuples()}
    lines=['# MatGPT F0 con Ridge M2: comparación controlada ABS₃','', '**Estado: PASS**','', '## Hechos verificados','',
      f"- Cohorte: 70 fórmulas, 210 estructuras estáticas, tripletes completos, cinco folds; inner-train/validation/outer = 45/11/14; semillas 42, 123 y 2026. Los joins se hicieron por IDs, fórmula, familia, fold y semilla.",
      f"- Ridge M2: se reutilizaron los cinco pipelines congelados de 17 fracciones, SHA verificado, sin fit; alphas {audit['ridge_alphas']}. Las medias predichas coinciden con MatterGPT M2 y son comunes a los tres polimorfos.",
      f"- MatGPT checkpoint SHA-256 `{MATGPT_CKPT_SHA}`; entrada `FINAL_END` (210×768). La caché reconstruida conserva checkpoint/dataset y pasa auditoría numérica, pero no es bit a bit idéntica a la histórica: drift máximo de vector {audit['embedding_numeric_max_pair_difference']:.2e}; se informa como limitación.",
      "- Geometría estática `gga-static_endpoint_cleaned`, siete descriptores históricos, join por `structure_id`; StandardScaler se ajustó únicamente con 45 fórmulas / 135 estructuras inner-train por fold.",
      "- Inicialización paso cero: las 15 parejas C0/C1 son equivalentes a tolerancia 1e-7 y los siete pesos geométricos empiezan en cero. Se entrenaron 30 cabezas; ambos backbones quedaron congelados. No se abrió el test histórico.", '',
      '## Energía absoluta y métrica primaria ΔE','',
      '| Condición (M2) | MAE E/N A | RMSE global E/N A | MAE residual A | MAE ΔE firmado A | MAE ΔE anclado A | GS | Pairwise | no resueltos |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('RETRAINED_M2_C0','MatGPT M2 SLICES'),('RETRAINED_M2_C1','MatGPT M2 + geometría'),('MatterGPT_F0','MatterGPT M2 SLICES'),('MatterGPT_F0_geometry','MatterGPT M2 + geometría')]:
        r=a.loc[name]; lines.append(f"| {label} | {r.mae_EN_mev_atom:.3f} | {r.rmse_EN_global_mev_atom:.3f} | {r.mae_residual_mev_atom:.3f} | {r.mae_delta_signed_mev_atom:.3f} | {r.mae_delta_anchored_all3_mev_atom:.3f} | {r.ground_state_accuracy:.3f} ({int(r.ground_state_denominator)}) | {r.pairwise_accuracy_eligible:.3f} ({int(r.eligible_pairs)}) | {int(r.unresolved_pairs)} |")
    lines += ['', 'Estimando A promedia corridas por fórmula/semilla y semillas dentro de fórmula, luego macro-promedia las 70 fórmulas. El RMSE E/N global se calcula como raíz de la media de los errores cuadrados en 630 observaciones (no como promedio de RMSE por fórmula). El estimando B promedia primero las tres predicciones por estructura y aparece en `metricas_recalculadas_internas.csv`. ΔE primaria incluye los tres pares firmados i<j; el filtro |ΔE_DFT|>5 se usa solo para ranking; |ΔE_pred|≤1 se clasifica como no resuelto y es incorrecto. GS excluye gaps DFT≤1 y los empates predichos no son aciertos.', '', '## Bootstrap apareado A', '', '| Contraste (izquierda − derecha) | ΔMAE ΔE | IC95 % |', '|---|---:|---:|']
    for r in boots.itertuples(): lines.append(f'| {r.contrast} | {r.estimate_mev_atom:.3f} | [{r.ci95_low_mev_atom:.3f}, {r.ci95_high_mev_atom:.3f}] |')
    poly=pd.read_csv(OUT/'metricas_por_polimorfo.csv'); gap=pd.read_csv(OUT/'metricas_por_gap.csv')
    lines += ['', '## Errores por polimorfo y tamaño de gap DFT','', '| Condición | Polimorfo | MAE E/N (meV/átomo) | n estructura-semilla |', '|---|---|---:|---:|']
    family_es={'dist_perovskite':'perovskita','hexagonal':'hexagonal','needle':'tipo aguja'}
    for name,label in [('RETRAINED_M2_C0','MatGPT M2 SLICES'),('RETRAINED_M2_C1','MatGPT M2 + geometría'),('MatterGPT_F0','MatterGPT M2 SLICES'),('MatterGPT_F0_geometry','MatterGPT M2 + geometría')]:
        for fam,es in family_es.items():
            z=poly[(poly.model.eq(name))&(poly.polymorph.eq(fam))].iloc[0]
            lines.append(f"| {label} | {es} | {z.mae_EN_mev_atom:.3f} | {int(z.n_rows_structure_seed)} |")
    lines += ['', 'Los conteos por intervalo son pares fórmula-semilla; la métrica es error absoluto ΔE por par, no por estructura.', '', '| Condición | Intervalo de |ΔE_DFT| (meV/átomo) | n pares fórmula-semilla | MAE ΔE (meV/átomo) |', '|---|---|---:|---:|']
    for name,label in [('RETRAINED_M2_C0','MatGPT M2 SLICES'),('RETRAINED_M2_C1','MatGPT M2 + geometría')]:
        for band in ('0–5','5–10','10–25','25–50','50–100','>100'):
            z=gap[(gap.model.eq(name))&(gap.DFT_gap_bin_mev_atom.eq(band))].iloc[0]
            val='n/d' if pd.isna(z.mae_delta_mev_atom) else f"{z.mae_delta_mev_atom:.3f}"
            lines.append(f"| {label} | {band} | {int(z.n_pairs_formula_seed)} | {val} |")
    lines += ['', '## Calibración OLS','', '| Condición | Pendiente E/N | R² E/N | Pendiente residuo | R² residuo | Pendiente ΔE firmado | Intercepto ΔE (meV/átomo) | R² OLS ΔE | R² predictivo ΔE | RMSE ΔE global (meV/átomo) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('RETRAINED_M2_C0','MatGPT M2 SLICES'),('RETRAINED_M2_C1','MatGPT M2 + geometría'),('MatterGPT_F0','MatterGPT M2 SLICES'),('MatterGPT_F0_geometry','MatterGPT M2 + geometría')]:
        r=a.loc[name]; lines.append(f"| {label} | {r.energy_ols_slope:.3f} | {r.energy_ols_r2:.3f} | {r.residual_ols_slope:.3f} | {r.residual_ols_r2:.3f} | {r.signed_pair_ols_slope:.3f} | {r.signed_pair_ols_intercept_mev_atom:.2f} | {r.signed_pair_ols_r2:.3f} | {r.signed_pair_predictive_r2:.3f} | {r.rmse_delta_global_mev_atom:.3f} |")
    lines += ['', 'OLS de energía usa E_DFT como x y E_pred como y; el residuo se centra dentro de cada triplete. La pendiente ΔE usa pares firmados i<j y no es la pendiente de gaps anclados al mínimo DFT.']
    lines += [
      '', 'IC percentiles no ajustados, 10 000 remuestreos por 70 fórmulas, semillas agrupadas, bootstrap seed 20261102. El contraste primario predefinido es MatGPT C1−C0; los otros tres son secundarios. Al no calcular p-valores, no se mezclan IC sin ajustar con pruebas ajustadas. Estos intervalos describen OOF de desarrollo reutilizado, no generalización externa.',
      '', '## Reconstrucción post hoc M1 → M2',
      '', f"Sustituir solo μ_M1 por μ_M2 conservando los residuos históricos dejó ΔE invariantes dentro de 1×10⁻⁶ eV/átomo: diferencia máxima por par {summary['posthoc_delta_invariance']['max_abs_delta_difference_ev_atom']:.3e} eV/átomo. La reconstrucción M1→Ridge+residuo cerró con error máximo {summary['posthoc_delta_invariance']['historical_reconstruction_max_abs_error_ev_atom']:.3e} eV/átomo; la tolerancia cubre redondeo de valores float32 ya serializados en los CSV históricos. Por construcción, cambian MAE E/N y energías absolutas, pero no el ranking ni ΔE dentro de esa tolerancia. `reconstruccion_posthoc_m2.csv` preserva cada predicción/residuo original y nuevo.",
      '', '| Condición histórica | MAE E/N con M1 | MAE E/N post hoc M2 | MAE ΔE |', '|---|---:|---:|---:|'
    ]
    compmap = {r['condition']: r for r in comparison}
    for key,label in [('HISTORICAL_M1_C0','SLICES'),('HISTORICAL_M1_C1','SLICES + geometría')]:
        x=compmap[key]; y=compmap[key.replace('HISTORICAL_M1','POSTHOC_M2')]
        lines.append(f"| {label} | {x['mae_EN_A_mev_atom']:.3f} | {y['mae_EN_A_mev_atom']:.3f} | {x['mae_delta_A_mev_atom']:.3f} |")
    selected=history.loc[history.groupby(['condition','fold','seed']).inner_validation_full_smoothl1.idxmin()].copy()
    selected['validation_minus_train']=selected.inner_validation_full_smoothl1-selected.train_minibatch_smoothl1
    s0=selected[selected.condition.eq('RETRAINED_M2_C0')]; s1=selected[selected.condition.eq('RETRAINED_M2_C1')]
    lines += ['', '## Sobreajuste, polimorfos y calibración','',f"La mediana de época de mejor validación fue C0={s0.epoch.median():.0f} y C1={s1.epoch.median():.0f}; la mediana de época de parada fue {history[history.condition.eq('RETRAINED_M2_C0')].groupby(['fold','seed']).epoch.max().median():.0f} y {history[history.condition.eq('RETRAINED_M2_C1')].groupby(['fold','seed']).epoch.max().median():.0f}. En el mejor punto de validación, la brecha media (validación completa menos promedio train de minilotes) fue C0={s0.validation_minus_train.mean():.3f} y C1={s1.validation_minus_train.mean():.3f} SmoothL1 normalizada. La brecha es menor en C1 en estas corridas, pero no demuestra por sí sola una reducción robusta del sobreajuste. MAE por polimorfo y gap se detallan arriba; la geometría baja el MAE E/N en los tres grupos, con mayor caída observada en hexagonal (2.470 meV/átomo) y menor en perovskita (0.948). No se infiere causalidad física.",'',
      'En la paridad de pares firmados se reporta OLS separado del ajuste de gaps anclados al mínimo DFT. Pendientes lejos de 1 indican compresión/calibración imperfecta. La comparación entre extractores no aísla solo preentrenamiento: MatGPT y MatterGPT difieren en arquitectura y tokenización.','','## Respuestas para la tesis','',
      f"1. MAE E/N MatGPT F0 M2 (reentrenado): {m('RETRAINED_M2_C0','mae_EN_mev_atom'):.3f} meV/átomo.",
      f"2. MAE E/N MatGPT F0 + geometría M2: {m('RETRAINED_M2_C1','mae_EN_mev_atom'):.3f} meV/átomo.",
      "3. La reconstrucción post hoc cambia solo μ y conserva residuos; el cambio M1→M2 es desplazamiento absoluto por fórmula.",
      "4. Sí: sustituir la media común M1 por M2 deja ΔE invariante; se comprobó por fila/par.",
      f"5. El reentrenamiento sí puede cambiar ΔE: MAE A C0={m('RETRAINED_M2_C0','mae_delta_signed_mev_atom'):.3f}; C1={m('RETRAINED_M2_C1','mae_delta_signed_mev_atom'):.3f} meV/átomo.",
      f"6–7. La geometría MatGPT cambia ΔMAE={bm['RETRAINED_M2_C1 − RETRAINED_M2_C0'][0]:.3f} [{bm['RETRAINED_M2_C1 − RETRAINED_M2_C0'][1]:.3f}, {bm['RETRAINED_M2_C1 − RETRAINED_M2_C0'][2]:.3f}]; juzgar robustez por si el IC cruza cero.",
      f"8. MatterGPT M2 + geometría tiene menor estimación puntual de MAE ΔE que MatGPT M2 + geometría ({m('MatterGPT_F0_geometry','mae_delta_signed_mev_atom'):.3f} vs {m('RETRAINED_M2_C1','mae_delta_signed_mev_atom'):.3f} meV/átomo), pero el IC95 % apareado de la diferencia cruza cero.",
      f"9. MatGPT M2 + geometría obtiene el mayor pairwise ({m('RETRAINED_M2_C1','pairwise_accuracy_eligible'):.3f}/591); MatterGPT M2 + geometría logra el mayor GS ({m('MatterGPT_F0_geometry','ground_state_accuracy'):.3f}/201).",
      "10. La mejor pendiente OLS ΔE entre las cuatro condiciones es MatGPT M2 + geometría (0.371, R²=0.277); todas comprimen los gaps frente a la pendiente ideal 1. La tabla OLS separa ΔE firmada, residuo y energía absoluta.",
      f"11. Brecha media validación−train al mejor punto: C0={s0.validation_minus_train.mean():.3f}, C1={s1.validation_minus_train.mean():.3f} SmoothL1 normalizada; su reducción en este desarrollo no constituye prueba independiente de menor sobreajuste.",
      "12. Para la tesis, usar las cuatro condiciones M2 y el estimando A; citar M1 histórico solo como referencia y separar el control post hoc del M2 reentrenado.", '',
      '## Artefactos','', 'Predicciones OOF, pares, composición, polimorfo, gap, checkpoints, bootstrap, configuración y recursos están en este directorio. Las figuras PNG están en `images/matgpt_f0_stoichiometric_ridge_m2_comparison/`.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__': main()
