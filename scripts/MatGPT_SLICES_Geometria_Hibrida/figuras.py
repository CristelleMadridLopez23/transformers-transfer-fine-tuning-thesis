"""Figuras con nombres descriptivos para tesis; no IDs crípticos ni resultados simulados."""
from __future__ import annotations
import argparse,json,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import GEOM_COLS,sha256

LABELS={'slices_only':'SLICES (MatGPT congelado)',
        'slices_plus_cell_geometry':'SLICES + geometría de celda'}
COLORS={'slices_only':'#2C6399','slices_plus_cell_geometry':'#C17639'}

def save(fig,path,description):
    fig.text(.055,.018,textwrap.fill(description,width=155),ha='left',va='bottom',fontsize=7.4,color='#53657B')
    fig.tight_layout(rect=(.015,.14,.985,.91))
    fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)

def scatter_comparison(xcol,ycol,frame,title,xlabel,ylabel,file,output):
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharex=True,sharey=True)
    low=float(min(frame[xcol].min(),frame[ycol].min()))
    high=float(max(frame[xcol].max(),frame[ycol].max()))
    pad=(high-low)*.06 or 1
    low-=pad;high+=pad
    rows=[]
    for ax,(cond,part) in zip(axes,frame.groupby('condition',sort=True)):
        x=part[xcol].to_numpy(float);y=part[ycol].to_numpy(float)
        ax.scatter(x,y,s=22,alpha=.65,c=COLORS[cond],edgecolors='none')
        slope,intercept=np.polyfit(x,y,1)
        fitted=slope*x+intercept
        ols_r2=1-float(np.sum((y-fitted)**2))/float(np.sum((y-y.mean())**2))
        mae=float(np.mean(np.abs(y-x)))
        ax.plot([low,high],[low,high],'--',color='#64748B',lw=1.3,label='Acuerdo ideal: y = x')
        ax.plot([low,high],[slope*low+intercept,slope*high+intercept],'-',color='#a124a0',lw=1.25,
                label='Ajuste OLS con intercepto libre')
        ax.set(xlabel=xlabel,ylabel=ylabel,title=LABELS[cond],xlim=(low,high),ylim=(low,high))
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        unit='eV/átomo' if 'eV/átomo' in xlabel and 'meV/átomo' not in xlabel else 'meV/átomo'
        ax.text(.04,.96,f'N = {len(x)}\nm = {slope:.3f}\nb = {intercept:.3f} {unit}\nR² OLS = {ols_r2:.3f}\nMAE = {mae:.2f} {unit}',
                transform=ax.transAxes,va='top',fontsize=8.5,
                bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
        rows.append(dict(figure=file,condition=cond,seed=42,n=len(x),unit=unit,
                         ols_slope=float(slope),ols_intercept=float(intercept),ols_r2=ols_r2,mae=mae))
    axes[0].legend(loc='lower right',frameon=False,fontsize=7.5)
    fig.suptitle(title,fontsize=15,fontweight='bold')
    observation='pares firmados' if xcol.startswith('delta_') else 'estructuras estáticas'
    save(fig,output/file,f'Cada punto representa una de 210 {observation} de 70 fórmulas en validación de cinco pliegues; semilla 42. Ambos paneles comparten observaciones y límites. OLS: mínimos cuadrados ordinarios; MAE: error absoluto medio. El test histórico sigue cerrado.')
    return rows

def build(output:Path,geometry_csv:Path|None=None):
    pred=pd.read_csv(output/'predicciones_oof_por_estructura.csv')
    metrics=output/'metricas'
    summary=pd.read_csv(metrics/'resumen_comparativo.csv')
    pairs=pd.read_csv(metrics/'diferencias_energeticas_por_par.csv')
    byformula=pd.read_csv(metrics/'metricas_por_composicion.csv')
    bins=pd.read_csv(metrics/'errores_por_intervalo_de_gap_dft.csv')
    boot=pd.read_csv(metrics/'intervalos_confianza_bootstrap_por_composicion.csv')
    figs=output/'figuras';figs.mkdir(parents=True,exist_ok=True)
    p=pred[pred.seed==42].copy()
    parity_stats=[]
    parity_stats+=scatter_comparison('energy_per_atom_dft_ev','energy_per_atom_pred_ev',p,
        'ABS₃: paridad de energía DFT por átomo',
        'DFT E/N (eV/átomo)','Predicción E/N (eV/átomo)',
        'paridad_energia_por_atomo_slices_frente_a_geometria.png',figs)
    rows=[]
    for (cond,seed,form),trip in pred[pred.seed==42].groupby(['condition','seed','composition_id']):
        for _,r in trip.iterrows():
            rows.append(dict(condition=cond,structure_id=r.structure_id,
                residual_dft_mev_atom=1000*(r.energy_per_atom_dft_ev-trip.energy_per_atom_dft_ev.mean()),
                residual_pred_mev_atom=1000*(r.energy_per_atom_pred_ev-trip.energy_per_atom_pred_ev.mean())))
    parity_stats+=scatter_comparison('residual_dft_mev_atom','residual_pred_mev_atom',pd.DataFrame(rows),
        'ABS₃: residuos energéticos dentro de cada composición',
        'Residuo DFT (meV/átomo)','Residuo predicho (meV/átomo)',
        'paridad_residuos_energeticos_intraformula.png',figs)
    parity_stats+=scatter_comparison('delta_dft_mev_atom','delta_pred_mev_atom',pairs[pairs.seed==42],
        'ABS₃: diferencias energéticas firmadas entre polimorfos',
        'ΔE DFT (meV/átomo)','ΔE predicho (meV/átomo)',
        'paridad_diferencias_energeticas_entre_polimorfos.png',figs)
    pd.DataFrame(parity_stats).to_csv(metrics/'estadisticas_paridad_semilla_42.csv',index=False)
    measures=[('mae_energy_per_atom_mev','MAE energía por átomo','meV/átomo'),
              ('mae_residual_intraformula_mev','MAE residuo intrafórmula','meV/átomo'),
              ('mae_delta_pairs_mev','MAE diferencias entre pares','meV/átomo'),
              ('ground_state_eligible_accuracy','Estado fundamental elegible','Fracción correcta'),
              ('pairwise_accuracy_gt5','Orden de pares |ΔE DFT| > 5','Fracción correcta')]
    fig,ax=plt.subplots(1,5,figsize=(17,4.6));fig.suptitle('Comparación controlada: SLICES y geometría de celda',fontsize=15,fontweight='bold')
    for a,(metric,title,unit) in zip(ax,measures):
        s=summary.set_index('condition').loc[list(LABELS),metric]
        a.bar([0,1],s.values,color=list(COLORS.values()),width=.55)
        a.set_xticks([0,1],['SLICES','SLICES + celda'],rotation=20)
        a.set(title=title,ylabel=unit)
        for i,val in enumerate(s.values):a.text(i,val+max(s.values)*.025,f'{val:.2f}',ha='center',fontsize=9)
        a.set_ylim(0,max(s.values)*1.2 if max(s.values)>0 else 1);a.grid(axis='y',alpha=.14)
    save(fig,figs/'comparacion_mae_residuos_ground_state_y_ranking.png',
         'Promedio de 5 pliegues y 3 semillas sobre los mismos 70 materiales físicos por condición.')
    bins=bins.groupby(['condition','gap_min','gap_max'],as_index=False).agg(mae_model_mev=('mae_model_mev','mean'),
               mae_zero_mev=('mae_zero_mev','mean'),pair_accuracy_gt5=('pair_accuracy_gt5','mean'),
               n_pairs=('n_pairs','mean'))
    seq=sorted(set(zip(bins.gap_min,bins.gap_max)))
    n_by_bin={(r.gap_min,r.gap_max):int(r.n_pairs) for r in bins[bins.condition=='slices_only'].itertuples()}
    tags=[(f'{int(lo)}–{int(hi)}' if np.isfinite(hi) else '>100')+f'\n(n={n_by_bin[(lo,hi)]})' for lo,hi in seq]
    fig,ax=plt.subplots(figsize=(10.3,5.5))
    for cond in LABELS:
        part=bins[bins.condition==cond].sort_values(['gap_min','gap_max'])
        ax.plot(tags,part.mae_model_mev,'-o',label=LABELS[cond],color=COLORS[cond])
    ref=bins[bins.condition=='slices_only'].sort_values(['gap_min','gap_max'])
    ax.plot(tags,ref.mae_zero_mev,'--o',label='Predictor ΔE = 0',color='#6B7280')
    ax.set(title='Error de ΔE según la magnitud del gap DFT',xlabel='|ΔE DFT| (meV/átomo)',ylabel='MAE ΔE (meV/átomo)')
    ax.legend();ax.grid(alpha=.15)
    save(fig,figs/'error_diferencias_energeticas_por_magnitud_gap_dft.png',
         'Los intervalos utilizan pares únicos por fórmula; las semillas son repeticiones de entrenamiento.')
    fig,ax=plt.subplots(figsize=(10.3,5.5))
    for cond in LABELS:
        part=bins[bins.condition==cond].sort_values(['gap_min','gap_max'])
        ax.plot(tags,part.pair_accuracy_gt5*100,'-o',label=LABELS[cond],color=COLORS[cond])
    ax.axhline(50,color='#6B7280',ls='--',label='Referencia aleatoria (50 %)')
    ax.set(title='Acierto del orden energético por intervalo de gap',xlabel='|ΔE DFT| (meV/átomo)',
           ylabel='Pares correctos (%)',ylim=(0,105))
    ax.legend();ax.grid(alpha=.15)
    save(fig,figs/'exactitud_ordenamiento_por_magnitud_gap_dft.png',
         'No se puntúa el intervalo de 0–5 meV/átomo; se requiere |ΔE DFT| > 5 y |ΔE pred| > 1.')
    chosen=boot[boot.metric.isin(['residual_mae_mev_atom','pair_mae_mev_atom',
                                'pairwise_accuracy_gt5','ground_state_eligible_accuracy'])]
    fig,ax=plt.subplots(figsize=(10.2,5.4))
    for i,(_,r) in enumerate(chosen.iterrows()):
        scalar=100 if r.metric.endswith('accuracy') else 1
        ax.errorbar(r.difference_mean*scalar,i,xerr=[[r.difference_mean*scalar-r.ci95_lower*scalar],
                 [r.ci95_upper*scalar-r.difference_mean*scalar]],fmt='o',capsize=4,color='#2C6399')
    ax.axvline(0,color='#6B7280',ls='--')
    names={'residual_mae_mev_atom':'MAE residuo (meV/átomo)',
           'pair_mae_mev_atom':'MAE pares (meV/átomo)',
           'pairwise_accuracy_gt5':'Acierto pares (puntos %)',
           'ground_state_eligible_accuracy':'Acierto estado fundamental (puntos %)'}
    ax.set_yticks(range(len(chosen)),[names[x] for x in chosen.metric]);ax.invert_yaxis()
    ax.set(title='Diferencia fusión − SLICES e incertidumbre bootstrap',xlabel='Diferencia (cada métrica con unidad indicada)')
    ax.grid(axis='x',alpha=.15)
    save(fig,figs/'diferencias_pareadas_con_intervalos_confianza.png',
         'IC del 95 % por remuestreo de 70 composiciones; los errores menores son mejores, las exactitudes mayores son mejores.')
    if geometry_csv is not None:
        geom=pd.read_csv(geometry_csv)
        fig,axes=plt.subplots(2,4,figsize=(13,6.8));axes=axes.flatten()
        titles={'a_reduced_shape':'Forma a/L (adimensional)',
                'b_reduced_shape':'Forma b/L (adimensional)',
                'c_reduced_shape':'Forma c/L (adimensional)',
                'alpha_reduced_deg':'Ángulo α reducido (°)',
                'beta_reduced_deg':'Ángulo β reducido (°)',
                'gamma_reduced_deg':'Ángulo γ reducido (°)',
                'volume_per_atom_ang3':'Volumen por átomo (Å³/átomo)'}
        for i,c in enumerate(GEOM_COLS):
            axes[i].hist(geom[c].astype(float),bins=20,color='#2C6399',alpha=.8)
            axes[i].set(title=titles[c],ylabel='Número de estructuras')
        axes[-1].axis('off')
        fig.suptitle('ABS₃: descriptores de celda de geometrías estáticas DFT',fontweight='bold')
        save(fig,figs/'distribucion_descriptores_geometricos_de_celda.png',
             '210 estructuras físicas de 70 fórmulas train/val; histogramas de conteos absolutos con 20 intervalos por panel. L es la raíz cúbica del volumen de la celda primitiva reducida por Niggli; no se usaron etiquetas energéticas.')
    manifest={p.name:sha256(p) for p in sorted(figs.glob('*.png'))}
    (figs/'manifiesto_figuras.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Figuras creadas:',len(manifest))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--geometry-csv',type=Path)
    a=p.parse_args();build(a.output.resolve(),a.geometry_csv)
if __name__=='__main__':main()
