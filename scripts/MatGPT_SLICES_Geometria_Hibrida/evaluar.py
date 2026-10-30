"""Evalúa SLICES vs SLICES+geometría con diferencias firmadas, GS y bootstrap por fórmula.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from core import FAMILIES,SEEDS,sha256

BINS=((0,5),(5,10),(10,25),(25,50),(50,100),(100,float('inf')))

def diagnostic_ols(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if len(y)<3 or np.std(y)<1e-15 or np.std(p)<1e-15:
        return dict(slope=np.nan,intercept=np.nan,correlation=np.nan,std_ratio=np.nan)
    slope,intercept=np.polyfit(y,p,1)
    return dict(slope=float(slope),intercept=float(intercept),
                correlation=float(np.corrcoef(y,p)[0,1]),std_ratio=float(np.std(p)/np.std(y)))

def metrics(pred:pd.DataFrame):
    records=[];pair_records=[];by_formula=[];bins=[]
    for (cond,seed),frame in pred.groupby(['condition','seed'],sort=True):
        if len(frame)!=210 or frame.structure_id.nunique()!=210:raise ValueError('OOF incompleto')
        truth=frame.energy_per_atom_dft_ev.to_numpy(float);estimated=frame.energy_per_atom_pred_ev.to_numpy(float)
        ys=[];ps=[];yres=[];pres=[];anch=[];gs_num=0;gs_denom=0;g_raw=0
        for formula,trip in frame.groupby('composition_id'):
            trip=trip.set_index('polymorph_family').loc[list(FAMILIES)]
            y=trip.energy_per_atom_dft_ev.to_numpy(float)
            p=trip.energy_per_atom_pred_ev.to_numpy(float)
            ry=y-y.mean();rp=p-p.mean()
            yres.extend(ry);pres.extend(rp)
            anchor=int(y.argmin())
            anch.extend(1000*((p-p[anchor])-(y-y[anchor])))
            dft_margin=np.partition(y,1)[1]-y.min()
            pred_margin=np.partition(p,1)[1]-p.min()
            eligible_gs=(dft_margin>0.001)
            correct_gs=bool(eligible_gs and pred_margin>0.001 and y.argmin()==p.argmin())
            gs_denom+=int(eligible_gs);gs_num+=int(correct_gs)
            g_raw+=int(y.argmin()==p.argmin())
            errors=(p-y)*1000
            formula_pairs=[]
            formula_correct=[];formula_eligible=[]
            for i,j in itertools.combinations(range(3),2):
                dy=1000*(y[j]-y[i]);dp=1000*(p[j]-p[i]);elig=abs(dy)>5
                correct=bool(elig and abs(dp)>1 and np.sign(dy)==np.sign(dp))
                ys.append(dy);ps.append(dp);formula_pairs.append(abs(dp-dy))
                formula_correct.append(int(correct));formula_eligible.append(int(elig))
                pair_records.append(dict(condition=cond,seed=seed,composition_id=formula,
                    family_a=FAMILIES[i],family_b=FAMILIES[j],
                    delta_dft_mev_atom=dy,delta_pred_mev_atom=dp,
                    gap_dft_absolute_mev_atom=abs(dy),mae_pair_mev_atom=abs(dp-dy),
                    mae_zero_mev_atom=abs(dy),eligible_gt5=int(elig),correct_gt5=int(correct)))
            by_formula.append(dict(condition=cond,seed=seed,composition_id=formula,
                absolute_mae_mev_atom=float(np.abs(errors).mean()),
                composition_mean_mae_mev_atom=float(abs(errors.mean())),
                residual_mae_mev_atom=float(np.abs((rp-ry)*1000).mean()),
                pair_mae_mev_atom=float(np.mean(formula_pairs)),
                gs_eligible=int(eligible_gs),gs_correct=int(correct_gs),
                gs_argmin_raw_correct=int(y.argmin()==p.argmin()),
                eligible_pairs=int(sum(formula_eligible)),correct_pairs=int(sum(formula_correct))))
        ys=np.asarray(ys);ps=np.asarray(ps);yres=np.asarray(yres);pres=np.asarray(pres)
        elig=abs(ys)>5
        records.append(dict(condition=cond,seed=seed,n_structures=210,n_compositions=70,n_pairs=210,
            n_pairs_eligible=int(elig.sum()),
            mae_energy_total_ev=float(np.mean(abs(frame.energy_total_pred_ev-frame.energy_total_dft_ev))),
            mae_energy_per_atom_mev=float(np.mean(abs(estimated-truth))*1000),
            mae_residual_intraformula_mev=float(np.mean(abs(yres-pres))*1000),
            mae_delta_pairs_mev=float(np.mean(abs(ys-ps))),
            mae_delta_zero_mev=float(np.mean(abs(ys))),
            mae_delta_anchored_mev=float(np.mean(np.abs(anch))),
            ground_state_eligible_accuracy=float(gs_num/gs_denom) if gs_denom else np.nan,
            ground_state_eligible_n=gs_denom,
            ground_state_argmin_raw_accuracy=float(g_raw/70),
            pairwise_accuracy_gt5=float(np.mean((np.sign(ps[elig])==np.sign(ys[elig]))&(abs(ps[elig])>1))),
            **{'energy_'+k:v for k,v in diagnostic_ols(truth,estimated).items()},
            **{'residual_'+k:v for k,v in diagnostic_ols(yres,pres).items()},
            **{'delta_'+k:v for k,v in diagnostic_ols(ys,ps).items()}))
        for lo,hi in BINS:
            mask=(abs(ys)>=lo if lo==0 else abs(ys)>lo)&(abs(ys)<=hi)
            eligible_mask=mask&elig
            bins.append(dict(condition=cond,seed=seed,gap_min=lo,gap_max=hi,
                n_pairs=int(mask.sum()),mae_model_mev=float(np.mean(abs(ps[mask]-ys[mask]))) if mask.any() else np.nan,
                mae_zero_mev=float(np.mean(abs(ys[mask]))) if mask.any() else np.nan,
                pair_accuracy_gt5=float(np.mean((np.sign(ps[eligible_mask])==np.sign(ys[eligible_mask]))&
                         (abs(ps[eligible_mask])>1))) if eligible_mask.any() else np.nan))
    return pd.DataFrame(records),pd.DataFrame(by_formula),pd.DataFrame(pair_records),pd.DataFrame(bins)

def bootstrap(composition:pd.DataFrame,n_reps=2000):
    base='slices_only';fusion='slices_plus_cell_geometry'
    result=[];rng=np.random.default_rng(20261028)
    for metric in ('absolute_mae_mev_atom','composition_mean_mae_mev_atom',
                   'residual_mae_mev_atom','pair_mae_mev_atom',
                   'gs_correct','gs_argmin_raw_correct'):
        wide=composition.pivot_table(index='composition_id',columns='condition',values=metric,aggfunc='mean')
        if len(wide)!=70 or wide.isna().any().any():raise ValueError('Bootstrap sin emparejar')
        d=(wide[fusion]-wide[base]).to_numpy(float)
        draw=d[rng.integers(0,len(d),size=(n_reps,len(d)))].mean(axis=1)
        result.append(dict(metric=metric,comparison=f'{fusion} - {base}',
                           difference_mean=float(d.mean()),ci95_lower=float(np.percentile(draw,2.5)),
                           ci95_upper=float(np.percentile(draw,97.5)),unit='formula',formulas=70,seeds=3))
    # Ranking por pares: denominadores agrupados por fórmula.
    sums=composition.groupby(['composition_id','condition'])[['correct_pairs','eligible_pairs','gs_correct','gs_eligible']].mean().reset_index()
    a=sums[sums.condition==base].set_index('composition_id').sort_index()
    b=sums[sums.condition==fusion].set_index('composition_id').sort_index()
    if not np.array_equal(a.eligible_pairs,b.eligible_pairs) or not np.array_equal(a.gs_eligible,b.gs_eligible):
        raise ValueError('Universos elegibles difieren')
    for name,numer,denom in (('pairwise_accuracy_gt5',b.correct_pairs-a.correct_pairs,a.eligible_pairs),
                             ('ground_state_eligible_accuracy',b.gs_correct-a.gs_correct,a.gs_eligible)):
        nu=numer.to_numpy(float);de=denom.to_numpy(float)
        ix=rng.integers(0,len(nu),size=(n_reps,len(nu)))
        dr=nu[ix].sum(axis=1)/de[ix].sum(axis=1)
        result.append(dict(metric=name,comparison=f'{fusion} - {base}',
                           difference_mean=float(nu.sum()/de.sum()),ci95_lower=float(np.percentile(dr,2.5)),
                           ci95_upper=float(np.percentile(dr,97.5)),unit='formula',formulas=70,seeds=3))
    return pd.DataFrame(result)

def evaluate(output:Path):
    pred_path=output/'predicciones_oof_por_estructura.csv'
    manifest=json.loads((output/'manifest_experimento.json').read_text())
    if sha256(pred_path)!=manifest['output_prediction_sha256']:
        raise ValueError('Predicciones modificadas después del entrenamiento')
    pred=pd.read_csv(pred_path)
    summary,formula,pairs,bins=metrics(pred)
    if set(summary.condition)!=set(manifest['conditions']):raise ValueError('Condiciones no coinciden')
    boots=bootstrap(formula)
    dest=output/'metricas';dest.mkdir(exist_ok=True)
    summary.to_csv(dest/'metricas_por_semilla.csv',index=False)
    formula.to_csv(dest/'metricas_por_composicion.csv',index=False)
    pairs.to_csv(dest/'diferencias_energeticas_por_par.csv',index=False)
    bins.to_csv(dest/'errores_por_intervalo_de_gap_dft.csv',index=False)
    boots.to_csv(dest/'intervalos_confianza_bootstrap_por_composicion.csv',index=False)
    grouped=summary.groupby('condition').mean(numeric_only=True).reset_index()
    grouped.to_csv(dest/'resumen_comparativo.csv',index=False)
    md=['# Resultados: SLICES frente a SLICES + geometría de celda','',
        '**Diseño:** MatGPT frozen, Ridge histórico sin cambios, MLP8 con 8 neuronas; 70 composiciones, 5 folds, 3 semillas.',
        '', '**Advertencia:** geometría puede proceder de estructuras relajadas DFT; no equivale a predicción sin geometría conocida.',
        '', '## Promedios por condición','',grouped.to_markdown(index=False,floatfmt='.4f'),'',
        '## Diferencias apareadas (fusión − SLICES; IC95% por composición)','',
        boots.to_markdown(index=False,floatfmt='.4f'),'',
        '## Interpretación','',
        'Los errores de ΔE y ranking corresponden únicamente a la rama residual MLP8: la salida Ridge común se cancela.',
        'Una mejora en MAE relativo, GS y ranking debería verificarse especialmente para gaps de 5–25 meV/átomo.',
        'Los resultados son exploratorios: estas 70 composiciones se usaron previamente para diseñar el estudio. El test histórico sigue cerrado.',
        'No atribuir a geometría una mejora si el control SLICES-only replicado cambia sustancialmente frente al H3-C histórico; documentar la diferencia.',
        '']
    (output/'reporte_resultados_geometria.md').write_text('\n'.join(md))
    print(grouped[['condition','mae_energy_per_atom_mev','mae_delta_pairs_mev',
                   'ground_state_eligible_accuracy','pairwise_accuracy_gt5']].to_string(index=False))
    return summary,formula,pairs,bins,boots

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();evaluate(a.output.resolve())
if __name__=='__main__':main()
