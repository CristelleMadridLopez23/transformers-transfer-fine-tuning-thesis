#!/usr/bin/env python3
"""Resume métricas pareadas de las predicciones OOF de E2-A."""
from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]; OUT=ROOT/'Alternative2'/'reports'/'MatGPT_finetuning'/'matgpt_e2'/'runs_endtoken_v2'/'cpu'
def main():
    """Calcula métricas macro por composición, bootstrap y figuras PNG en español."""
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    p=OUT/'oof_predictions.csv'
    if not p.is_file(): raise FileNotFoundError('Ejecuta primero E2-A CV completa.')
    d=pd.read_csv(p); required={'configuration','seed','fold','structure_id','composition_group','polymorph_family','energy_per_atom_ev','energy_pred_ev_atom'}
    if required-set(d): raise ValueError(f'Faltan columnas: {sorted(required-set(d))}')
    if d.structure_id.astype(str).str.contains('test',case=False).any(): raise ValueError('Predicciones test no permitidas.')
    rows=[]
    for (model,seed,fold,group),g in d.groupby(['configuration','seed','fold','composition_group']):
        y=g.energy_per_atom_ev.to_numpy(float); q=g.energy_pred_ev_atom.to_numpy(float); dy=(y-y.min())*1000; dq=(q-q.min())*1000
        pairs=[(i,j) for i in range(len(y)) for j in range(i+1,len(y)) if abs(y[i]-y[j])>=.001]
        pair_acc=float(np.mean([np.sign(y[i]-y[j])==np.sign(q[i]-q[j]) for i,j in pairs])) if pairs else np.nan
        rows.append({'configuration':model,'seed':seed,'fold':fold,'composition_group':group,'absolute_mae_ev_atom':float(np.mean(abs(q-y))),'delta_e_mae_mev_atom':float(np.mean(abs(dq-dy))),'ground_state_accuracy':float(int(np.argmin(y)==int(np.argmin(q)))),'pairwise_accuracy':pair_acc,'dft_range_mev_atom':float(np.ptp(dy)),'pred_range_mev_atom':float(np.ptp(dq)),'collapsed':bool(np.ptp(dq)<1.)})
    comp=pd.DataFrame(rows); comp.to_csv(OUT/'metrics_composition.csv',index=False)
    summary=comp.groupby(['configuration','seed']).agg(**{k:(k,'mean') for k in ['absolute_mae_ev_atom','delta_e_mae_mev_atom','ground_state_accuracy','pairwise_accuracy','dft_range_mev_atom','pred_range_mev_atom','collapsed']}).reset_index(); summary.to_csv(OUT/'metrics_per_seed.csv',index=False)
    # Pareado por composición, preservando juntos los tres polimorfos durante bootstrap.
    wide=comp.groupby(['configuration','composition_group'])[['absolute_mae_ev_atom','delta_e_mae_mev_atom','ground_state_accuracy','pairwise_accuracy']].mean().reset_index()
    rng=np.random.default_rng(13072026); boot={}; metrics=['absolute_mae_ev_atom','delta_e_mae_mev_atom','ground_state_accuracy','pairwise_accuracy']
    for metric in metrics:
        a=wide[wide.configuration.eq('frozen')].set_index('composition_group')[metric]; b=wide[wide.configuration.eq('partial_ft')].set_index('composition_group')[metric]; common=sorted(set(a.index)&set(b.index))
        if not common: continue
        delta=(b[common]-a[common]).to_numpy(float); draws=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        boot[metric]={'partial_ft_minus_frozen_mean':float(delta.mean()),'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'n_compositions':len(common),'resamples':2000,'seed':13072026}
    (OUT/'paired_bootstrap_ci.json').write_text(json.dumps(boot,indent=2)+'\n')
    figdir=OUT/'figures'; figdir.mkdir(exist_ok=True)
    for metric,label in [('absolute_mae_ev_atom','MAE absoluto (eV/átomo)'),('delta_e_mae_mev_atom','MAE de ΔE (meV/átomo)'),('ground_state_accuracy','Exactitud del estado fundamental'),('pairwise_accuracy','Exactitud de orden por pares')]:
        fig,ax=plt.subplots(figsize=(8,5)); groups=[comp.loc[comp.configuration.eq(m),metric].dropna().to_numpy() for m in ['frozen','partial_ft']]
        ax.boxplot(groups,tick_labels=['MatGPT congelado','Ajuste parcial']); ax.set_ylabel(label); ax.set_title('E2-A: validación externa por composición'); ax.grid(axis='y',alpha=.25)
        ax.text(.01,-.21,'Cada punto de origen es una composición; tres semillas resumidas por composición. 70 composiciones de desarrollo; test histórico excluido. Intervalos descriptivos.',transform=ax.transAxes,fontsize=8)
        fig.tight_layout(); fig.savefig(figdir/f'{metric}.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    (OUT/'E2_report.md').write_text('# E2-A — resultados OOF\n\nSolo se evalúan folds externos de desarrollo; el test histórico no participa. La selección de configuraciones anteriores usó la validation histórica; estos resultados no eliminan ese sesgo. Bootstrap por composición: ver `paired_bootstrap_ci.json`; intervalos exploratorios.\n\n## Métricas por semilla\n\n```text\n'+summary.to_string(index=False)+'\n```\n')
    logging.info('evaluation_complete compositions=%d configs=%s',comp.composition_group.nunique(),sorted(comp.configuration.unique()))
if __name__=='__main__': main()
