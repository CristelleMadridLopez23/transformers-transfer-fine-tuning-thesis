"""H3-C SLICES-only vs SLICES + celda: mismo Ridge, MLP8 ancho 8 y splits.
No usa test histórico. No reentrena el Transformer.
"""
from __future__ import annotations
import argparse,json,os,resource,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from core import (FAMILIES,SEEDS,GEOM_COLS,EXPECTED_CACHE_HASH,EXPECTED_CHECKPOINT_HASH,
                  ResidualMLP8,matched_pair,seeded,sha256,load_development,
                  load_features,load_folds,roles,triplets,load_geometry,train)

PATHS={
 'dataset':'reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv',
 'folds':'reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv',
 'cache':'reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz',
 'checkpoint':'models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth',
 'ridge':'reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8/H3_B_ridge_fixed_delta/ridge_by_fold',
 'historic':'reports/MatGPT/MatGPT_H3C_Final/reproduction/replayed_predictions.csv',
 'output':'reports/MatGPT/MatGPT_SLICES_Geometria',
}

def run(root:Path,geometry:Path,output:Path,allow_existing=False):
    root=root.resolve();output=output.resolve();geometry=geometry.resolve()
    if output.exists() and any(output.iterdir()) and not allow_existing:
        raise FileExistsError(f'Directorio de resultados no vacío; no sobrescribir: {output}')
    paths={k:root/v for k,v in PATHS.items() if k not in ('output',)}
    for k in ('dataset','folds','cache','checkpoint','historic'):
        if not paths[k].is_file():raise FileNotFoundError(f'Falta dependencia del H3-C real ({k}): {paths[k]}')
    if sha256(paths['checkpoint'])!=EXPECTED_CHECKPOINT_HASH:
        raise ValueError('El checkpoint MatGPT no coincide con lock H3-C')
    data=load_development(paths['dataset']); features=load_features(paths['cache'],data.structure_id.tolist())
    folds=load_folds(paths['folds'],data); geometry_features=load_geometry(geometry,data)
    historic=pd.read_csv(paths['historic'])
    if len(historic)!=630 or historic.structure_id.nunique()!=210:
        raise ValueError('La reproducción base H3-C debe contener 630 predicciones')
    records=[];training=[];checkpoints=[]
    for fold in range(5):
        ids_train,ids_val,ids_outer=roles(folds,fold)
        ftr,xtr,ytr,_=triplets(data,ids_train,features['FINAL_END'])
        fva,xva,yva,_=triplets(data,ids_val,features['FINAL_END'])
        fout,xout,yout,groups=triplets(data,ids_outer,features['FINAL_END'])
        _,gtr,_,_=triplets(data,ids_train,geometry_features)
        _,gva,_,_=triplets(data,ids_val,geometry_features)
        _,gout,_,_=triplets(data,ids_outer,geometry_features)
        # Importantísimo: ajusta scaler sin acceder a val/outer.
        scaler=StandardScaler().fit(gtr.reshape(-1,len(GEOM_COLS)))
        gtr=scaler.transform(gtr.reshape(-1,len(GEOM_COLS))).reshape(gtr.shape).astype('float32')
        gva=scaler.transform(gva.reshape(-1,len(GEOM_COLS))).reshape(gva.shape).astype('float32')
        gout=scaler.transform(gout.reshape(-1,len(GEOM_COLS))).reshape(gout.shape).astype('float32')
        for arr in (gtr,gva,gout):
            if not np.isfinite(arr).all():raise ValueError('Geometría estandarizada no finita')
        old_ridge=paths['ridge']/f'fold_{fold}/ridge.joblib'
        if not old_ridge.is_file():raise FileNotFoundError(old_ridge)
        ridge=joblib.load(old_ridge)
        x_ridge=features['L03_MEAN']
        _,lr,_,_=triplets(data,ids_outer,x_ridge)
        means=ridge.predict(lr.mean(axis=1))
        baseline=np.array([historic[(historic.fold==fold)&(historic.seed==42)&
                     (historic.structure_id==sid)].ridge_mean_ev_atom.iloc[0]
                     for group in groups for sid in group]).reshape(-1,3)
        if not np.allclose(baseline[:,0],means,atol=1e-5) or not np.allclose(baseline,means[:,None],atol=1e-5):
            raise ValueError('La rama Ridge no coincide con H3-C histórico')
        scale=max(float((ytr-ytr.mean(axis=1,keepdims=True)).std()),1e-6)
        for seed in SEEDS:
            seeded(seed)
            base,fusion=matched_pair(768,len(GEOM_COLS))
            base.eval();fusion.eval()
            with torch.inference_mode():
                assert torch.allclose(base(torch.tensor(xtr)),fusion(torch.tensor(np.concatenate([xtr,gtr],axis=-1))),atol=1e-6,rtol=1e-5)
            conditions=[('slices_only',base,xtr,xva,xout),
                        ('slices_plus_cell_geometry',fusion,
                          np.concatenate([xtr,gtr],axis=-1),
                          np.concatenate([xva,gva],axis=-1),
                          np.concatenate([xout,gout],axis=-1))]
            for name,model,xt,xv,xo in conditions:
                run_started=time.monotonic()
                model,run_meta,history=train(model,xt,ytr,xv,yva,seed,scale)
                run_meta['wall_seconds']=time.monotonic()-run_started
                run_meta['process_peak_rss_bytes_observed']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                with torch.inference_mode():
                    residual=model(torch.tensor(xo,dtype=torch.float32)).numpy()
                if not np.allclose(residual.mean(axis=1),0,atol=2e-6):
                    raise ValueError('Residuos no centrados por fórmula')
                run_dir=output/'training'/name/f'fold_{fold}_seed_{seed}'
                run_dir.mkdir(parents=True,exist_ok=True)
                state={'state_dict':model.state_dict(),'architecture':[xt.shape[-1],8,1],
                       'fold':fold,'seed':seed,'condition':name,
                       'geometry_columns':list(GEOM_COLS) if name!='slices_only' else [],
                       'best_epoch':run_meta['best_epoch']}
                torch.save(state,run_dir/'residual_mlp8.pt')
                pd.DataFrame(history).to_csv(run_dir/'training_history.csv',index=False)
                joblib.dump(scaler,run_dir/'geometry_scaler_train_only.joblib')
                training.append(dict(condition=name,fold=fold,seed=seed,**run_meta))
                checkpoint_rows=run_meta.copy();checkpoint_rows['checkpoint_sha256']=sha256(run_dir/'residual_mlp8.pt')
                checkpoints.append(dict(condition=name,fold=fold,seed=seed,**checkpoint_rows))
                lookup=data.set_index('structure_id')
                for i,(formula,g) in enumerate(zip(fout,groups)):
                    for j,sid in enumerate(g):
                        r=lookup.loc[sid]
                        e=float(means[i]+residual[i,j])
                        records.append(dict(condition=name,fold=fold,seed=seed,structure_id=sid,
                                            composition_id=formula,polymorph_family=FAMILIES[j],
                                            n_atoms=int(r.n_atoms),energy_total_dft_ev=float(r.energy_total_dft_ev),
                                            energy_per_atom_dft_ev=float(r.energy_per_atom_dft_ev),
                                            ridge_mean_ev_atom=float(means[i]),residual_ev_atom=float(residual[i,j]),
                                            energy_per_atom_pred_ev=e,energy_total_pred_ev=e*int(r.n_atoms)))
            print(f'fold={fold} seed={seed} terminado: baseline + fusión',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    pred=pd.DataFrame(records)
    if len(pred)!=1260 or any(len(part)!=210 for _,part in pred.groupby(['condition','seed'])):
        raise ValueError('OOF incompleto: se esperan 2×3×210 filas')
    pred.to_csv(output/'predicciones_oof_por_estructura.csv',index=False)
    pd.DataFrame(training).to_csv(output/'seleccion_y_entrenamiento_por_fold.csv',index=False)
    pd.DataFrame(checkpoints).to_csv(output/'checkpoints_seleccionados.csv',index=False)
    manifest=dict(status='PASS',conditions=['slices_only','slices_plus_cell_geometry'],
                  physical_structures=210,physical_compositions=70,folds=5,seeds=list(SEEDS),
                  geometry_columns=list(GEOM_COLS),geometry_source_file=str(geometry),
                  geometry_sha256=sha256(geometry),
                  code_strategy='Matched head initialization + zero initialized geometry weights',
                  baseline_same_ridge='Historical five frozen fitted Ridge; no Ridge retraining',
                  geometry_scaling='StandardScaler fit only on 45 inner train compositions, per fold',
                  measured_files={p.name:sha256(p) for p in [paths['dataset'],paths['folds'],paths['cache'],
                            paths['checkpoint'],paths['historic'],geometry]},
                  output_prediction_sha256=sha256(output/'predicciones_oof_por_estructura.csv'),
                  historical_test_accessed=False,
                  note='Solo una adición de descriptores de celda a la rama residual, sin fuerza ni tensión.')
    (output/'manifest_experimento.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('measured_files',)},indent=2,ensure_ascii=False))
    return pred

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',required=True,type=Path)
    p.add_argument('--geometry-csv',required=True,type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    out=a.output or a.repo_root/PATHS['output']
    run(a.repo_root,a.geometry_csv,out)
if __name__=='__main__':main()
