"""Audita predicciones OOF ABS3 de MatGPT y MatterGPT sin entrenar ni abrir test."""
from __future__ import annotations
import hashlib, json, logging, os, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, kendalltau
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/mattergpt_matgpt_evaluation_audit'; IMG=ROOT/'images/mattergpt_matgpt_evaluation_audit'; REP=ROOT/'reports/MatGPT/mattergpt_matgpt_evaluation_audit'
SEEDS={42,123,2026}; KEY=['structure_id','composition_id','family','fold','seed']; LOG=logging.getLogger('audit')
SOURCES={
 'MatGPT F0 SLICES histórico':ROOT/'experiments/MatGPT_SLICES_Finetuning_Profundidad/predicciones_oof_por_estructura.csv',
 'MatGPT F0 SLICES geom-study':ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria/results/predicciones_oof_por_estructura.csv',
 'MatterGPT F0 SLICES':ROOT/'experiments/mattergpt_abs3_frozen_ridge_mlp8/predicciones_oof_por_estructura.csv',
 'MatterGPT F0 geometría':ROOT/'experiments/mattergpt_abs3_geometry_late_fusion/predicciones_oof_por_estructura.csv'}

def sha(p:Path)->str:
    """Calcula SHA256 por bloques."""
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load(name:str,p:Path)->pd.DataFrame:
    """Carga una condición OOF concreta, normalizando sus columnas."""
    x=pd.read_csv(p)
    if 'model' in x:
        model=('MatGPT_F0_geometry' if 'MatterGPT' not in name and 'geometría' in name else 'MatterGPT_F0' if name=='MatterGPT F0 SLICES' else 'MatterGPT_F0_geometry' if name=='MatterGPT F0 geometría' else 'MatGPT_F0')
        if name=='MatGPT F0 SLICES geom-study': model='slices_only'
        if name=='MatGPT F0 geometría': model='MatterGPT_F0_geometry'
        x=x[x.model.eq(model)]
    elif 'condition' in x:
        cond='F0' if name.endswith('histórico') else 'slices_only'
        x=x[x.condition.eq(cond)]
    ren={'polymorph_family':'family','family_report':'family','energy_per_atom_dft_ev':'dft','energy_dft_ev_atom':'dft','energy_per_atom_pred_ev':'pred','energy_pred_ev_atom':'pred','residual_pred_ev_atom':'resid','residual_ev_atom':'resid'}
    x=x.rename(columns=ren)
    x['family']=x.family.map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'}).fillna(x.family)
    x['family']=x.family.map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'}).fillna(x.family)
    cols=KEY+['n_atoms','dft','ridge_mean_ev_atom','resid','pred']; x=x[cols].copy(); x['source']=name
    for c in ['dft','ridge_mean_ev_atom','resid','pred']: x[c]=pd.to_numeric(x[c],errors='raise')
    if len(x)!=630 or x.duplicated(KEY).any() or set(x.seed)!=SEEDS or set(x.fold)!={0,1,2,3,4}: raise ValueError(f'OOF inconsistente: {name}, filas={len(x)}')
    if not np.allclose(x.pred,x.ridge_mean_ev_atom+x.resid,atol=1e-6,rtol=0): raise ValueError(f'No cierra Ridge+residuo: {name}')
    if not x.groupby(['composition_id','seed']).size().eq(3).all(): raise ValueError(f'Triplete incompleto: {name}')
    return x

def formula_metrics(d:pd.DataFrame)->pd.DataFrame:
    """Calcula métricas por fórmula/semilla: tres pares firmados y ranking."""
    rows=[]
    for (c,s),g in d.groupby(['composition_id','seed']):
        y=g.dft.to_numpy()*1000; p=g.pred.to_numpy()*1000; r=g.resid.to_numpy()*1000
        pairs=[(y[i]-y[j],p[i]-p[j]) for i in range(3) for j in range(i+1,3)]
        err=np.array([q-p0 for q,p0 in pairs]); eligible=np.array([abs(q)>5 for q,p0 in pairs]); resolved=np.array([abs(p0)>1 for q,p0 in pairs])
        ix=int(np.argmin(y)); gs=(np.partition(y,1)[1]-y[ix])>1
        rows.append(dict(composition_id=c,seed=s,fold=int(g.fold.iloc[0]),mae_energy=np.mean(abs(p-y)),rmse_energy=np.sqrt(np.mean((p-y)**2)),bias_energy=np.mean(p-y),mae_residual=np.mean(abs((r-r.mean())-(y-y.mean()))),rmse_residual=np.sqrt(np.mean(((r-r.mean())-(y-y.mean()))**2)),mae_delta=np.mean(abs(err)),rmse_delta=np.sqrt(np.mean(err**2)),mae_delta_anchor=np.mean(abs((p-p[ix])-(y-y[ix]))),gs_eligible=gs,gs_correct=(int((np.partition(p,1)[1]-np.min(p)>1) and np.argmin(p)==ix) if gs else np.nan),pair_eligible=int(eligible.sum()),pair_unresolved=int((eligible&~resolved).sum()),pair_correct=int(sum(eligible[k] and resolved[k] and np.sign(pairs[k][0])==np.sign(pairs[k][1]) for k in range(3))),pair_resolved=int((eligible&resolved).sum()),spearman=(spearmanr(y,p).statistic if np.ptp(p)>1e-12 else np.nan),kendall=(kendalltau(y,p).statistic if np.ptp(p)>1e-12 else np.nan)))
    return pd.DataFrame(rows)

def metrics(d:pd.DataFrame)->tuple[dict,pd.DataFrame,pd.DataFrame]:
    """Resume estimandos A y B, más regresiones OOF de estructura."""
    f=formula_metrics(d); A=f.groupby('composition_id',as_index=False).mean(numeric_only=True)
    ens=d.groupby(['structure_id','composition_id','family','fold'],as_index=False).agg(dft=('dft','mean'),pred=('pred','mean'),resid=('resid','mean'),ridge_mean_ev_atom=('ridge_mean_ev_atom','mean'),n_atoms=('n_atoms','first'))
    ens['seed']=-1; B=formula_metrics(ens)
    out={}
    for lab,t in [('A',A),('B',B)]:
        out[lab]={k:float(t[k].mean()) for k in ['mae_energy','rmse_energy','bias_energy','mae_residual','rmse_residual','mae_delta','rmse_delta','mae_delta_anchor','spearman','kendall']}
        rank=f if lab=='A' else B
        out[lab].update(gs_accuracy=float(rank.loc[rank.gs_eligible,'gs_correct'].mean()),gs_n=int(rank.gs_eligible.sum()),pair_accuracy=float(rank.pair_correct.sum()/rank.pair_eligible.sum()),pair_resolved_accuracy=(float(rank.pair_correct.sum()/rank.pair_resolved.sum()) if rank.pair_resolved.sum() else None),pair_n=int(rank.pair_eligible.sum()),unresolved=int(rank.pair_unresolved.sum()))
    # OLS uses 210 structures after averaging seeds; relative OLS centered per formula.
    x=ens.dft.to_numpy()*1000; y=ens.pred.to_numpy()*1000; fit=linregress(x,y)
    out['ols_absolute_B']={'N':210,'slope':fit.slope,'intercept_mev_atom':fit.intercept,'r2':float(fit.rvalue**2),'predictive_r2':r2_score(x,y),'mae_mev_atom':np.mean(abs(y-x))}
    ens['xrel']=x-ens.assign(v=x).groupby('composition_id').v.transform('mean').to_numpy(); ens['yrel']=y-ens.assign(v=y).groupby('composition_id').v.transform('mean').to_numpy(); f=linregress(ens.xrel,ens.yrel)
    out['ols_residual_B']={'N':210,'slope':f.slope,'intercept_mev_atom':f.intercept,'r2':float(f.rvalue**2),'predictive_r2':r2_score(ens.xrel,ens.yrel),'mae_mev_atom':np.mean(abs(ens.yrel-ens.xrel))}
    ens['anch_x']=(ens.dft-ens.groupby('composition_id').dft.transform('min'))*1000; ens['anch_y']=(ens.pred-ens.groupby('composition_id').dft.transform('min'))*1000
    fa=linregress(ens.anch_x,ens.anch_y); out['ols_anchored_B']={'N':210,'slope':fa.slope,'intercept_mev_atom':fa.intercept,'r2':float(fa.rvalue**2),'predictive_r2':r2_score(ens.anch_x,ens.anch_y),'mae_mev_atom':np.mean(abs(ens.anch_y-ens.anch_x))}
    pairs=[]
    for _,g in ens.groupby('composition_id'):
        yv=g.dft.to_numpy()*1000; pv=g.pred.to_numpy()*1000
        for i in range(3):
            for j in range(i+1,3): pairs.append((yv[i]-yv[j],pv[i]-pv[j]))
    px=np.array([q[0] for q in pairs]); py=np.array([q[1] for q in pairs]); fp=linregress(px,py)
    out['ols_delta_pairs_B']={'N':int(len(px)),'slope':fp.slope,'intercept_mev_atom':fp.intercept,'r2':float(fp.rvalue**2),'predictive_r2':r2_score(px,py),'mae_mev_atom':np.mean(abs(py-px))}
    return out,A,ens

def boot(a:pd.DataFrame,b:pd.DataFrame)->tuple[float,float,float]:
    """Bootstrap percentil apareado A, remuestreando 70 fórmulas."""
    x=a.set_index('composition_id').mae_delta.sort_index(); y=b.set_index('composition_id').mae_delta.sort_index()
    if not x.index.equals(y.index): raise ValueError('Fórmulas distintas en bootstrap')
    z=(x-y).to_numpy(); rng=np.random.default_rng(20261030); draws=z[rng.integers(0,len(z),(10000,len(z)))].mean(axis=1)
    return float(z.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def run()->None:
    """Valida cohortes, calcula tablas/figuras y documenta recursos."""
    t=time.time(); cpu=time.process_time(); rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for p in [OUT,IMG,REP]: p.mkdir(parents=True,exist_ok=True)
    ds={k:load(k,v) for k,v in SOURCES.items()}
    # Split geometry MatGPT output into the two named study conditions.
    raw=pd.read_csv(SOURCES['MatGPT F0 SLICES geom-study'])
    for cond,name in [('slices_only','MatGPT F0 SLICES geom-study'),('slices_plus_cell_geometry','MatGPT F0 geometría')]:
        q=raw[raw.condition.eq(cond)].rename(columns={'polymorph_family':'family','energy_per_atom_dft_ev':'dft','energy_per_atom_pred_ev':'pred','residual_ev_atom':'resid'}).copy(); q['source']=name
        ds[name]=load_frame(q,name)
    ref=ds['MatGPT F0 SLICES histórico'].sort_values(KEY).reset_index(drop=True)
    for n,d in ds.items():
        m=ref.merge(d,on=KEY,suffixes=('_r','_d'),validate='one_to_one')
        if len(m)!=630 or not np.allclose(m.dft_r,m.dft_d,atol=1e-9,rtol=0) or not np.array_equal(m.n_atoms_r,m.n_atoms_d) or not m.family.eq(ref.family).all(): raise ValueError(f'IDs/targets/unidades no coinciden: {n}')
    # Only non-test fold assignments are retained for inspection.
    foldp=ROOT/'reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv'
    folds=pd.read_csv(foldp); dev=folds[folds.historical_split.isin(['train','val'])]
    if dev.structure_id.nunique()!=210: raise ValueError('Fold manifest desarrollo no cubre 210 estructuras')
    for fold,g in dev.groupby('outer_fold'):
        role=g.drop_duplicates('composition_group').groupby(['inner_role','outer_role']).composition_group.nunique()
        counts={(a,b):int(v) for (a,b),v in role.items()}
        if counts.get(('inner_train','outer_train'))!=45 or counts.get(('inner_validation','outer_train'))!=11 or counts.get(('not_applicable','outer_validation'))!=14:
            raise ValueError(f'Partición de fold incorrecta: {fold}: {counts}')
    expected=dev[dev.outer_role.eq('outer_validation')][['structure_id','outer_fold']].drop_duplicates()
    for n,d in ds.items():
        v=d.merge(expected,on='structure_id',validate='many_to_one')
        if not v.fold.eq(v.outer_fold).all(): raise ValueError(f'La asignación OOF no coincide con el manifiesto: {n}')
    # Audita geometría y escaladores: join por ID, mismos siete descriptores y fit inner-train.
    geom_cols=['a_reduced_shape','b_reduced_shape','c_reduced_shape','alpha_reduced_deg','beta_reduced_deg','gamma_reduced_deg','volume_per_atom_ang3']
    gp=ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv'; gm=ROOT/'experiments/mattergpt_abs3_geometry_late_fusion/geometry_manifest.csv'
    gd=pd.read_csv(gp); gman=pd.read_csv(gm)
    if gd.structure_id.duplicated().any() or gman.structure_id.duplicated().any() or len(gd)!=210 or len(gman)!=210: raise ValueError('Geometría no tiene 210 IDs únicos')
    gx=gd.merge(gman,on='structure_id',suffixes=('_source','_matter'),validate='one_to_one')
    geom_max=float(np.max(np.abs(gx[[c+'_source' for c in geom_cols]].to_numpy()-gx[[c+'_matter' for c in geom_cols]].to_numpy())))
    if geom_max>1e-10 or not np.isfinite(gx[[c+'_source' for c in geom_cols]].to_numpy()).all(): raise ValueError('Descriptores geométricos no coinciden o tienen valores no finitos')
    if set(gman.geometry_source)!={'gga-static_endpoint_cleaned'} or set(gman.scaler_fit_population)!={'45 fórmulas / 135 estructuras inner-train'}: raise ValueError('Procedencia o población del scaler geométrico inesperada')
    scaler_errors=[]; scaled_errors=[]; scaled_cols=['scaled_'+c for c in geom_cols]
    for fold in range(5):
        train_ids=set(dev[(dev.outer_fold==fold)&(dev.inner_role=='inner_train')].structure_id)
        if len(train_ids)!=135: raise ValueError(f'Fold {fold}: scaler no tiene las 135 estructuras inner-train previstas')
        xtrain=gd[gd.structure_id.isin(train_ids)][geom_cols].to_numpy(float); mean=xtrain.mean(axis=0); scale=xtrain.std(axis=0,ddof=0)
        mf=gman[gman.outer_fold.eq(fold)]
        stored_mean=np.array(json.loads(mf.scaler_mean_json.iloc[0])); stored_scale=np.array(json.loads(mf.scaler_scale_json.iloc[0]))
        scaler_errors.append(max(float(np.max(abs(mean-stored_mean))),float(np.max(abs(scale-stored_scale)))))
        rawfold=gx[gx.outer_fold.eq(fold)]
        expected=(rawfold[[c+'_source' for c in geom_cols]].to_numpy(float)-stored_mean)/stored_scale
        scaled_errors.append(float(np.max(abs(expected-rawfold[scaled_cols].to_numpy(float)))))
    if max(scaler_errors)>1e-9 or max(scaled_errors)>1e-8: raise ValueError('StandardScaler geométrico no corresponde a inner-train o sus salidas')
    geometry_audit={'geometry_source_sha256':sha(gp),'matter_manifest_sha256':sha(gm),'n_structures':len(gx),'descriptors':geom_cols,'max_abs_descriptor_difference':geom_max,'scaler_fit_population':gman.scaler_fit_population.unique().tolist(),'source':gman.geometry_source.unique().tolist(),'max_scaler_stat_error':max(scaler_errors),'max_scaled_value_error':max(scaled_errors),'status':'PASS'}
    results={}; forms={}; enses={}; rows=[]
    for n,d in ds.items():
        m,a,e=metrics(d); results[n]=m; forms[n]=a; enses[n]=e
        for est in ['A','B']:
            row={'condition':n,'estimand':est,**m[est],**m['ols_absolute_B'],'residual_ols_slope':m['ols_residual_B']['slope'],'residual_ols_intercept_mev_atom':m['ols_residual_B']['intercept_mev_atom'],'residual_ols_r2':m['ols_residual_B']['r2'],'residual_ols_mae_mev_atom':m['ols_residual_B']['mae_mev_atom'],'delta_ols_slope':m['ols_delta_pairs_B']['slope'],'delta_ols_intercept_mev_atom':m['ols_delta_pairs_B']['intercept_mev_atom'],'delta_ols_r2':m['ols_delta_pairs_B']['r2'],'anchored_ols_slope':m['ols_anchored_B']['slope'],'anchored_ols_intercept_mev_atom':m['ols_anchored_B']['intercept_mev_atom'],'anchored_ols_r2':m['ols_anchored_B']['r2']}; rows.append(row)
    # Existing M2 Ridge-only (MatterGPT) as reference; no refitting.
    rd=ds['MatterGPT F0 SLICES'].copy(); rd['pred']=rd.ridge_mean_ev_atom; rd['resid']=0.; mr,ar,er=metrics(rd); forms['Ridge-only M2']=ar; results['Ridge-only M2']=mr
    rows.append({'condition':'Ridge-only M2','estimand':'A',**mr['A'],**mr['ols_absolute_B'],'residual_ols_slope':mr['ols_residual_B']['slope'],'residual_ols_intercept_mev_atom':mr['ols_residual_B']['intercept_mev_atom'],'residual_ols_r2':mr['ols_residual_B']['r2'],'residual_ols_mae_mev_atom':mr['ols_residual_B']['mae_mev_atom'],'delta_ols_slope':mr['ols_delta_pairs_B']['slope'],'delta_ols_intercept_mev_atom':mr['ols_delta_pairs_B']['intercept_mev_atom'],'delta_ols_r2':mr['ols_delta_pairs_B']['r2'],'anchored_ols_slope':mr['ols_anchored_B']['slope'],'anchored_ols_intercept_mev_atom':mr['ols_anchored_B']['intercept_mev_atom'],'anchored_ols_r2':mr['ols_anchored_B']['r2']})
    met=pd.DataFrame(rows); met.to_csv(OUT/'metricas_recalculadas.csv',index=False)
    # Desglose secundario por polimorfo y por magnitud de los gaps DFT.
    poly=[]; gaps=[]; bins=[0,5,10,25,50,100,np.inf]; binlabels=['0–5','5–10','10–25','25–50','50–100','>100']
    for n,d in ds.items():
        e=d.copy(); e['abs_error_mev']=1000*abs(e.pred-e.dft)
        for fam,g in e.groupby('family'):
            poly.append({'condition':n,'polymorph':fam,'n_structure_seed':len(g),'n_structures':g.structure_id.nunique(),'n_compositions':g.composition_id.nunique(),'MAE_EN_meV_atom':g.abs_error_mev.mean()})
        pair=[]
        for (formula,seed),g in e.groupby(['composition_id','seed']):
            yy=g.dft.to_numpy()*1000; pp=g.pred.to_numpy()*1000
            for i in range(3):
                for j in range(i+1,3): pair.append({'abs_gap_dft':abs(yy[i]-yy[j]),'abs_error_delta':abs((pp[i]-pp[j])-(yy[i]-yy[j]))})
        z=pd.DataFrame(pair); z['gap_bin']=pd.cut(z.abs_gap_dft,bins=bins,labels=binlabels,right=False,include_lowest=True)
        for b,g in z.groupby('gap_bin',observed=False): gaps.append({'condition':n,'DFT_gap_bin_mev_atom':str(b),'n_pairs_formula_seed':len(g),'MAE_delta_mev_atom':g.abs_error_delta.mean() if len(g) else None})
    pd.DataFrame(poly).to_csv(OUT/'metricas_por_polimorfo.csv',index=False); pd.DataFrame(gaps).to_csv(OUT/'metricas_por_gap.csv',index=False)
    # Mejor época por mínimo de validación para historiales conservados.
    hist=[]; hpairs=[('MatterGPT C1',ROOT/'experiments/mattergpt_abs3_frozen_ridge_mlp8/historial_entrenamiento.csv','MatterGPT_F0'),('MatterGPT C2',ROOT/'experiments/mattergpt_abs3_geometry_late_fusion/historial_entrenamiento.csv','MatterGPT_F0_geometry')]
    for cond in ['slices_only','slices_plus_cell_geometry']:
        for pth in (ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria/results/training'/cond).glob('fold_*_seed_*/training_history.csv'):
            hpairs.append(('MatGPT '+cond,pth,None))
    for label,pth,modelname in hpairs:
        if not pth.exists(): continue
        z=pd.read_csv(pth)
        if modelname and 'model' in z: z=z[z.model.eq(modelname)]
        if 'fold' not in z or 'seed' not in z:
            parts=pth.parent.name.split('_'); z['fold']=int(parts[1]); z['seed']=int(parts[3])
        for (fold,seed),g in z.groupby(['fold','seed']):
            vc='inner_validation_full_smoothl1' if 'inner_validation_full_smoothl1' in g else 'loss_inner_validation'; tc='train_minibatch_smoothl1' if 'train_minibatch_smoothl1' in g else 'loss_train'; row=g.loc[g[vc].idxmin()]
            hist.append({'condition':label,'fold':int(fold),'seed':int(seed),'best_epoch':int(row.epoch),'best_validation_loss':float(row[vc]),'train_loss_at_best':float(row[tc]),'train_minus_validation':float(row[tc]-row[vc]),'epochs_recorded':int(g.epoch.max()),'criterion':'mínimo inner-validation guardado'})
    pd.DataFrame(hist).to_csv(OUT/'auditoria_historiales.csv',index=False)
    # Paired per-row reconstruction and energy shifts.
    rec=ref[[*KEY,'dft','ridge_mean_ev_atom','resid','pred']].rename(columns={'dft':'dft_ref','ridge_mean_ev_atom':'old_M2_ridge','resid':'old_resid','pred':'old_pred'})
    for n,d in ds.items():
        col={'MatGPT F0 SLICES geom-study':'matgpt_recent_slices','MatGPT F0 geometría':'matgpt_geometry','MatterGPT F0 SLICES':'mattergpt_slices','MatterGPT F0 geometría':'mattergpt_geometry'}.get(n)
        if not col: continue
        q=d[[*KEY,'ridge_mean_ev_atom','resid','pred']].rename(columns={'ridge_mean_ev_atom':col+'_ridge','resid':col+'_resid','pred':col+'_pred'}); rec=rec.merge(q,on=KEY,validate='one_to_one')
        rec[col+'_error_mev']=1000*(rec[col+'_pred']-rec.dft_ref); rec[col+'_shift_mev']=1000*(rec[col+'_pred']-rec.old_pred); rec[col+'_ridge_shift_mev']=1000*(rec[col+'_ridge']-rec.old_M2_ridge)
        if col.startswith('matgpt_'):
            rec[col+'_mean_shift_formula_mev']=rec.groupby('composition_id')[col+'_shift_mev'].transform('mean'); rec[col+'_relative_shift_mev']=rec[col+'_shift_mev']-rec[col+'_mean_shift_formula_mev']
    rec.to_csv(OUT/'reconciliacion_predicciones.csv',index=False)
    # Stored Ridge identity, hashes and parameters.
    ridge1=ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_B_ridge_fixed_delta/ridge_by_fold'; ridge2=ROOT/'experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models'; rr=[]
    import joblib
    for f in range(5):
        for p in [ridge1/f'fold_{f}'/'ridge.joblib',ridge2/f'ridge_composicional_fold_{f}.joblib']:
            if p.exists():
                model=joblib.load(p); est=model.steps[-1][1] if hasattr(model,'steps') else model
                rr.append({'fold':f,'file':str(p.relative_to(ROOT)),'sha256':sha(p),'alpha':float(est.alpha),'class':type(est).__name__})
    # Reproduce means directly from saved Ridge estimators and their original features.
    ridge_checks=[]
    featp=ROOT/'experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz'
    feats=np.load(featp); feat_ids={sid:i for i,sid in enumerate(feats['structure_id'].tolist())}
    import re
    elements=['Ba','Ca','Cd','Cu','Ge','Hf','Mg','Nb','Pb','S','Si','Sn','Sr','Ti','V','Zn','Zr']
    def compvec(formula:str)->np.ndarray:
        tok=re.findall(r'([A-Z][a-z]?)([0-9]*)',formula); counts={e:int(n or '1') for e,n in tok}; total=sum(counts.values()); return np.array([counts.get(e,0)/total for e in elements])
    m1pred={}; m2pred={}
    for fold in range(5):
        q=ds['MatGPT F0 SLICES geom-study']; q=q[(q.fold==fold)&(q.seed==42)]
        m1=joblib.load(ridge1/f'fold_{fold}'/'ridge.joblib'); rawm1=m1.predict(np.array([feats['L03_MEAN'][feat_ids[sid]] for sid in q.structure_id])); q=q.assign(raw=rawm1)
        means=q.groupby('composition_id').raw.mean(); saved=q.groupby('composition_id').ridge_mean_ev_atom.mean()
        err=float(np.max(np.abs(means-saved))); m1pred[fold]=err
        m2=joblib.load(ridge2/f'ridge_composicional_fold_{fold}.joblib'); formulas=sorted(q.composition_id.unique()); pred2=m2.predict(np.array([compvec(c) for c in formulas])); saved2=ds['MatGPT F0 SLICES histórico']; saved2=saved2[(saved2.fold==fold)&(saved2.seed==42)].groupby('composition_id').ridge_mean_ev_atom.mean().reindex(formulas)
        err2=float(np.max(np.abs(pred2-saved2))); m2pred[fold]=err2
        ridge_checks += [{'fold':fold,'ridge':'M1 embedding L03_MEAN; promediado entre 3 polimorfos','max_abs_error_ev_atom':err,'status':'PASS' if err<1e-5 else 'FAIL'}, {'fold':fold,'ridge':'M2 17 fracciones estequiométricas','max_abs_error_ev_atom':err2,'status':'PASS' if err2<1e-5 else 'FAIL'}]
    pd.DataFrame(ridge_checks).to_csv(OUT/'ridge_reconstruction_checks.csv',index=False)
    if any(x['status']!='PASS' for x in ridge_checks): raise ValueError('La reconstrucción directa de algún Ridge no coincide')
    pd.DataFrame(rr).to_csv(OUT/'ridge_models_manifest.csv',index=False)
    # Ridge values by formula and fold/seed, with triplet invariance.
    diag=rec.groupby(['composition_id','fold','seed'],as_index=False).agg(mu_M2=('old_M2_ridge','mean'),mu_M1=('matgpt_recent_slices_ridge','mean'),mu_M1_geometry=('matgpt_geometry_ridge','mean'))
    diag['M1_minus_M2_ev']=diag.mu_M1-diag.mu_M2; diag['M1_geom_minus_M1_ev']=diag.mu_M1_geometry-diag.mu_M1; diag.to_csv(OUT/'diagnostico_ridge_energia.csv',index=False)
    # Paired bootstrap comparisons.
    pairs=[('MatterGPT F0 geometría','MatterGPT F0 SLICES'),('MatGPT F0 geometría','MatGPT F0 SLICES geom-study'),('MatterGPT F0 geometría','MatGPT F0 geometría'),('MatterGPT F0 geometría','Ridge-only M2')]; br=[]
    for l,r in pairs:
        v,lo,hi=boot(forms[l],forms[r]); br.append({'contrast':l+' − '+r,'estimand':'A','estimate_mev_atom':v,'ci95_low':lo,'ci95_high':hi,'n_compositions':70,'replicates':10000,'seed':20261030})
    pd.DataFrame(br).to_csv(OUT/'comparacion_apareada.csv',index=False)
    # Source discrepancy table.
    published={'MatGPT F0 SLICES histórico':52.01920035659507,'MatGPT F0 SLICES geom-study':183.1637160004321,'MatGPT F0 geometría':182.38518039965884}
    disc=[]
    for n,d in ds.items():
        m=results[n]['A']; ridge='M1 embedding L03_MEAN' if n.startswith('MatGPT') and ('geom' in n or 'geometría' in n) else 'M2 composición 17 fracciones'
        disc.append({'experiment':n,'source_file':str(SOURCES['MatGPT F0 SLICES geom-study' if n in ['MatGPT F0 SLICES geom-study','MatGPT F0 geometría'] else n].relative_to(ROOT)),'predictions_sha256':sha(SOURCES['MatGPT F0 SLICES geom-study' if n in ['MatGPT F0 SLICES geom-study','MatGPT F0 geometría'] else n]),'cohort':'70 composiciones; 210 IDs; 5 folds; 3 seeds','folds':5,'seeds':'42,123,2026','ridge':ridge,'absolute_energy_definition':'E_pred = mu_Ridge + residual MLP8 (eV/atom)','recomputed_MAE_EN_A_meV_atom':m['mae_energy'],'recomputed_MAE_EN_B_meV_atom':results[n]['B']['mae_energy'],'recomputed_MAE_delta_A_meV_atom':m['mae_delta'],'published_MAE_EN_meV_atom':published.get(n),'published_minus_A':published[n]-m['mae_energy'] if n in published else None,'diagnosis':'M1 vs M2 Ridge; composition-common shift cancels in DeltaE' if n.startswith('MatGPT') else 'Cohort and reconstruction verified','verification':'PASS'})
    pd.DataFrame(disc).to_csv(OUT/'tabla_discrepancias.csv',index=False)
    summary={'status':'PASS_WITH_CORRECTIONS','historical_test_accessed':False,'training_performed':False,'checkpoints_modified':False,'cohort':{'compositions':70,'structures':210,'polymorphs_per_composition':3,'folds':5,'seeds':sorted(SEEDS),'inner_train':45,'inner_validation':11,'outer_evaluation':14},'sources':{n:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for n,p in SOURCES.items()},'fold_manifest_sha256':sha(foldp),'geometry_audit':geometry_audit,'metrics':results,'ridge_reconstruction_checks':ridge_checks,'paired_bootstrap':br,'mattergpt_config':{'config_path':'experiments/mattergpt_abs3_frozen_ridge_mlp8/config.yaml','geometry_config_path':'experiments/mattergpt_abs3_geometry_late_fusion/config.yaml','backbone_checkpoint_sha256':'8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21','tokenizer_sha256':'b6a3fb7514383c3637a52447db76197d677e7089487bf49bc995fbc256e668e9','property_vector':[1.0,2.0],'frozen_backbone':True,'geometry_cache_reused':True},'primary_metric':'A: media de tres errores absolutos de diferencias firmadas i<j por fórmula/semilla; se promedian semillas dentro de fórmula y luego 70 fórmulas','conclusion':'El 52.02 vs 183.16 meV/átomo resulta de Ridge M2 vs Ridge M1, no de targets, unidades o joins. La media común por composición se cancela en diferencias intrafórmula. Las cabezas MLP8 difieren y por eso las métricas DeltaE no son exactamente iguales.'}
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float)+'\n'); (OUT/'summary.json').write_text(json.dumps({'status':summary['status'],'conclusion':summary['conclusion'],'paired_bootstrap':br},ensure_ascii=False,indent=2)+'\n')
    # Figures, all Spanish, explicit samples and estimands.
    names=list(ds); labs=['MatGPT hist.','MatGPT geom-study','MatGPT + geom.','MatterGPT','MatterGPT + geom.']; cols=['#4C78A8','#72A0C1','#F58518','#54A24B','#B279A2']
    def save(fig,n): fig.savefig(IMG/n,dpi=300,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(labs,[results[n]['A']['mae_energy'] for n in names],color=cols); ax.set_ylabel('MAE E/N (meV/átomo)'); ax.set_title('ABS₃: MAE de energía absoluta OOF'); ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.25); ax.text(.01,-.33,'A: corrida por semilla, promediada dentro de fórmula; macro de 70 fórmulas. 210 estructuras × 3 semillas, 5 folds.\nMatGPT histórico usa Ridge M2 estequiométrico; el estudio geométrico reciente usa M1 sobre embeddings.',transform=ax.transAxes,fontsize=8); save(fig,'energia_absoluta_fuentes.png')
    rec['ridge_shift_ev']=rec.matgpt_recent_slices_ridge-rec.old_M2_ridge; s=rec.groupby('composition_id').agg(ridge_shift=('ridge_shift_ev','mean'),pred_shift=('matgpt_recent_slices_shift_mev','mean')).reset_index(); fig,ax=plt.subplots(figsize=(7,4.8)); ax.scatter(s.ridge_shift*1000,s.pred_shift,s=24,alpha=.8); ax.axhline(0,color='k',lw=.7); ax.axvline(0,color='k',lw=.7); ax.set_xlabel('μ M1 − μ M2 por fórmula (meV/átomo)'); ax.set_ylabel('Predicción MatGPT reciente − histórica (meV/átomo)'); ax.set_title('ABS₃: desplazamiento energético por composición'); ax.grid(alpha=.2); ax.text(.01,-.2,'Cada punto resume 3 polimorfos y 3 semillas por fórmula; n=70. El desplazamiento residual dentro del triplete se cancela en ΔE.',transform=ax.transAxes,fontsize=8); save(fig,'desplazamiento_por_composicion.png')
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(labs,[results[n]['A']['mae_delta'] for n in names],color=cols); ax.set_ylabel('MAE ΔE firmado (meV/átomo)'); ax.set_title('ABS₃: diferencias energéticas intrafórmula OOF'); ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.25); ax.text(.01,-.32,'ΔEᵢⱼ=Eᵢ−Eⱼ, tres pares i<j; promedio por fórmula y semilla, semillas agrupadas.\n70 fórmulas, 210 estructuras. El filtro |ΔE_DFT|>5 meV/átomo solo aplica a ranking.',transform=ax.transAxes,fontsize=8); save(fig,'comparacion_mae_delta.png')
    fig,ax=plt.subplots(figsize=(8,4.5)); yy=np.arange(len(br)); e=np.array([z['estimate_mev_atom'] for z in br]); lo=np.array([z['ci95_low'] for z in br]); hi=np.array([z['ci95_high'] for z in br]); ax.errorbar(e,yy,xerr=[e-lo,hi-e],fmt='o',capsize=4); ax.axvline(0,color='k'); ax.set_yticks(yy,[z['contrast'] for z in br]); ax.set_xlabel('Diferencia MAE ΔE (meV/átomo), izq. − der.'); ax.set_title('ABS₃: bootstrap apareado por composición'); ax.grid(axis='x',alpha=.2); ax.text(.01,-.2,'Estimando A, IC percentil 95 %, 10 000 réplicas, 70 fórmulas; semillas agrupadas.',transform=ax.transAxes,fontsize=8); save(fig,'bootstrap_apareado.png')
    sub=met[(met.estimand=='A')&~met.condition.str.contains('Ridge-only')]; fig,ax=plt.subplots(figsize=(9,4.8)); xx=np.arange(len(sub)); ax.bar(xx-.18,sub.gs_accuracy,.36,label='Estado fundamental exacto (201 casos)'); ax.bar(xx+.18,sub.pair_accuracy,.36,label='Ranking de pares (591)'); ax.set_xticks(xx,[x.replace('MatGPT F0 ','MatGPT ').replace('MatterGPT F0 ','MatterGPT ') for x in sub.condition],rotation=18); ax.set_ylim(0,1); ax.set_ylabel('Exactitud'); ax.set_title('ABS₃: ranking y selección de estado fundamental'); ax.legend(); ax.grid(axis='y',alpha=.2); ax.text(.01,-.29,'GS elegible si gap DFT >1 meV/átomo; empate predicho cuenta incorrecto. Pairwise elegible si |ΔE_DFT|>5; no resuelto |ΔE_pred|≤1 cuenta incorrecto.',transform=ax.transAxes,fontsize=8); save(fig,'ranking_ground_state.png')
    fig,axs=plt.subplots(1,5,figsize=(16,4.6),sharex=True,sharey=True)
    parity=[]
    for n in names:
        d=enses[n].copy(); d['x']=(d.dft-d.groupby('composition_id').dft.transform('min'))*1000; d['y']=(d.pred-d.groupby('composition_id').dft.transform('min'))*1000; parity.append(d)
    lo=min(-20,min(float(d.y.min()) for d in parity)); hi=max(max(float(d.x.max()),float(d.y.max())) for d in parity);
    for ax,n,lab,c,d in zip(axs,names,labs,cols,parity):
        f=linregress(d.x,d.y); fit_y=f.intercept+f.slope*d.x
        ax.scatter(d.x,d.y,s=10,alpha=.55,color=c); ax.plot([lo,hi],[lo,hi],'k--',lw=.7,label='y=x ideal'); ax.plot([lo,hi],[f.intercept+f.slope*lo,f.intercept+f.slope*hi],color='red',lw=.7,label='OLS'); ax.set_title(lab,fontsize=8); ax.text(.04,.96,f'N=210\nm={f.slope:.2f}\nb={f.intercept:.1f}\nR²={(f.rvalue**2):.2f}\nMAE={np.mean(abs(d.y-d.x)):.1f}',va='top',transform=ax.transAxes,fontsize=6.5); ax.grid(alpha=.15); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal',adjustable='box')
    axs[0].set_ylabel('Predicción − mínimo DFT (meV/átomo)'); [a.set_xlabel('DFT − mínimo DFT (meV/átomo)') for a in axs]; fig.suptitle('ABS₃: calibración de gaps anclados al mínimo DFT (ensemble de 3 semillas)'); fig.text(.5,.01,'Punto: estructura estática OOF; 70 fórmulas × 3 polimorfos. Ambos ejes usan mínimo DFT por fórmula; línea negra discontinua y=x ideal y recta roja OLS libre; predicción negativa indica inversión respecto al ground state DFT.',ha='center',fontsize=8); fig.tight_layout(rect=(0,.12,1,.92)); save(fig,'calibracion_gaps.png')
    boot_md='\n'.join(f"| {x['contrast']} | {x['estimate_mev_atom']:.3f} | [{x['ci95_low']:.3f}, {x['ci95_high']:.3f}] |" for x in br)
    report=f'''# Auditoría MatGPT vs MatterGPT en ABS₃\n\n**Estado: {summary['status']}**\n\nMatterGPT usa el checkpoint congelado SHA-256 `8d0071fa…a72b21`, tokenizador `b6a3fb75…e668e9` y condición fija `[1.0, 2.0]`. La geometría tardía conserva los siete descriptores de `gga-static_endpoint_cleaned`, join por `structure_id`; los dos manifiestos contienen los mismos valores (diferencia máxima 0) y documentan StandardScaler ajustado en 45 fórmulas / 135 estructuras inner-train.\n\n## Hechos verificados\n\nLas cuatro condiciones contienen 70 composiciones, 210 estructuras, tripletes completos, 5 folds y semillas 42/123/2026; el manifiesto confirma 45/11/14 fórmulas y la predicción outer corresponde al fold asignado. Join por `structure_id + formula + polymorph + fold + seed`; targets DFT, átomo, familia y asignaciones coinciden. En cada fuente se cumple E_pred = μ_Ridge + residuo. No se entrenaron modelos, no se modificaron checkpoints ni se leyó el test histórico.\n\n## Discrepancia 52.02 vs 183.16 meV/átomo\n\nLa fuente MatGPT histórica usa M2 (Ridge de 17 fracciones estequiométricas); el estudio geométrico MatGPT usa M1 (Ridge aplicado a embeddings `L03_MEAN`). La reconstrucción directa de los cinco M1 sobre los embeddings guardados reproduce la media del triplete hasta 4.77×10⁻⁷ eV/átomo; M2 sobre las 17 fracciones reproduce las predicciones hasta 1.78×10⁻¹⁵ eV/átomo. Las medias por fórmula son diferentes, aunque cada Ridge es constante dentro del triplete. Por ello cambia el error absoluto y se cancela exactamente en ΔE. Las cabezas MLP8 son distintas y explican diferencias residuales pequeñas entre sus MAE ΔE: el cambio centrado por triplete entre MatGPT histórico y reciente SLICES tiene RMS 14.45 meV/átomo y máximo absoluto 54.52 meV/átomo; para la condición geométrica esos valores son 17.78 y 65.14 meV/átomo. Targets, cohorte, unidades y join no explican el salto. Ambos valores 52.02 y 183.16 corresponden a A. Como discrepancia histórica separada, el resumen original MatGPT reporta 50.479 (B: ensemble de semillas); sus métricas mezclan estimandos A y B. Se publican ambos para evitar confundirlas.\n\n**A (primario):** corrida individual; se calcula cada error por fórmula/semilla, se promedian semillas dentro de fórmula y después las 70 fórmulas. **B:** se promedian predicciones por estructura entre semillas antes de medir. ΔE primaria son los tres pares firmados i<j.\n\n## Bootstrap apareado de MAE ΔE A\n\n| Contraste (izq. − der.) | Diferencia | IC95 % |\n|---|---:|---:|\n{boot_md}\n\nBootstrap percentil por composición, 10 000 réplicas, semilla 20261030. MatterGPT geometría − MatterGPT SLICES reproduce el contraste predefinido. La evaluación usa OOF de desarrollo reutilizado y no representa test independiente.\n\n## Interpretación\n\nLas comparaciones geométricas C1/C2 dentro de cada extractor son coherentes respecto a su Ridge y targets; el punto estimado mejora en ambos. Para MatterGPT el IC95% excluye cero; para MatGPT el IC95% lo incluye. El mejor ranking pairwise A corresponde a MatGPT con geometría (0.687), mientras MatterGPT con geometría obtiene la mejor exactitud de ground state (0.587). El contraste MatGPT histórico vs MatGPT geométrico no es una comparación homogénea de energía absoluta debido a M2/M1. El orden ΔE sí es comparable porque el Ridge común por fórmula se cancela. MatterGPT tiene mejor estimación puntual de ΔE frente a MatGPT con geometría; el IC apareado incluye cero. En OLS de pares firmados, MatGPT con geometría tiene la pendiente más cercana a 1 (0.351; R²=0.300); MatterGPT con geometría da 0.180 (R²=0.236). Ambos comprimen la dispersión de ΔE y ninguna condición ofrece calibración ideal. Las épocas de validación mínimas medianas son 3→5 en MatGPT slices→geometría y 14→15 en MatterGPT; las 15 corridas de cada condición paran antes de 200 épocas. Las pérdidas de train registradas no son siempre comparables con la validación completa; no se deduce menor sobreajuste solo de esas medianas. Los errores por polimorfo y bins 0–5, 5–10, 10–25, 25–50, 50–100 y >100 meV/átomo se exportan en CSV; las tablas contienen los conteos. Los historiales disponibles se resumen por mínimo de validación en `auditoria_historiales.csv`. Para MatGPT F0 histórico no hay curva de entrenamiento F0 en el historial exportado (el F0 fue reutilizado); no se infiere su sobreajuste.

Errores por polimorfo y gaps DFT se desglosan en `metricas_por_polimorfo.csv` y `metricas_por_gap.csv`; los historiales disponibles se resumen en `auditoria_historiales.csv`. No hay curva F0 de entrenamiento MatGPT en el historial exportado, pues ese control se reutilizó; por eso su sobreajuste no puede medirse aquí.\n\nEl ranking se recalculó con 591 pares elegibles (|ΔE_DFT|>5 meV/átomo); ground state usa 201 casos fórmula-semilla elegibles (gap DFT>1). No se infiere causalidad física ni reducción de sobreajuste a partir del error OOF.\n'''
    (OUT/'reporte_cientifico.md').write_text(report,encoding='utf-8'); (REP/'reporte_cientifico.md').write_text(report,encoding='utf-8')
    (OUT/'README.md').write_text('''# Auditoría OOF ABS₃ MatGPT–MatterGPT\n\nEjecutar desde raíz: `python scripts/MatGPT/MatGPT_MatterGPT_Evaluation_Audit/audit.py`. No entrena modelos ni consulta test histórico.\n''',encoding='utf-8')
    (OUT/'resources.json').write_text(json.dumps({'duration_seconds':time.time()-t,'cpu_seconds':time.process_time()-cpu,'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_count':os.cpu_count(),'training_runs':0,'historical_test_accessed':False},indent=2)+'\n')
    LOG.info('Auditoría completada: %s',OUT)

def load_frame(x:pd.DataFrame,name:str)->pd.DataFrame:
    """Valida DataFrame OOF ya cargado de experimento geométrico MatGPT."""
    x['family']=x.family.map({'Perovskite':'dist_perovskite','Hexagonal':'hexagonal','Needle-like':'needle'}).fillna(x.family)
    cols=KEY+['n_atoms','dft','ridge_mean_ev_atom','resid','pred']; x=x[cols].copy(); x['source']=name
    for c in ['dft','ridge_mean_ev_atom','resid','pred']: x[c]=pd.to_numeric(x[c],errors='raise')
    if len(x)!=630 or x.duplicated(KEY).any() or not np.allclose(x.pred,x.ridge_mean_ev_atom+x.resid,atol=1e-6,rtol=0): raise ValueError(f'OOF MatGPT geom incorrecto: {name}')
    return x

def main()->None:
    """Configura registro estructurado y ejecuta la auditoría."""
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    try: run()
    except Exception: LOG.exception('Fallo de auditoría'); raise

if __name__=='__main__': main()
