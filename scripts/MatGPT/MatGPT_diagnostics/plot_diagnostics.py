#!/usr/bin/env python3
"""Produce figuras PNG de lectura sencilla para D1 y D3."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, r2_score

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development

LOG=logging.getLogger("matgpt_diagnostics.figures")
COLORS={"D1-A media train":"#777777","D1-B Ridge composición":"#2A6FBB","D1-C RandomForest pequeño":"#D17A00","E2-A partial_ft":"#9C3B79","D2-CENTER":"#1B8A78","D2-CENTER-ONLY":"#A04D95"}


def parity(ax: plt.Axes, y: np.ndarray, p: np.ndarray, label: str, color: str) -> dict[str,float]:
    """Dibuja paridad, ideal y ajuste OLS; devuelve métricas de esa población."""
    ax.scatter(y,p,s=18,alpha=.65,color=color,edgecolors="none",label="Una estructura estática")
    bounds=(float(min(y.min(),p.min())),float(max(y.max(),p.max())))
    ax.plot(bounds,bounds,color="#333333",linestyle="--",linewidth=1.4,label="Ideal: predicción = DFT")
    fit=linregress(y,p)
    ax.plot(bounds,fit.intercept+fit.slope*np.asarray(bounds),color="#D43F75",linewidth=1.5,label="OLS libre: predicción = m·DFT + b")
    mae=float(mean_absolute_error(y,p)); r2=float(r2_score(y,p))
    ax.set_xlim(bounds);ax.set_ylim(bounds);ax.set_xlabel("Energía DFT-PBE (eV/átomo)");ax.set_ylabel("Energía predicha (eV/átomo)")
    ax.set_title(label);ax.grid(alpha=.22)
    ax.text(.03,.97,f"N = {len(y)} estructuras\nOLS m = {fit.slope:.3f} (adimensional)\nb = {fit.intercept:.3f} eV/átomo\nR² = {r2:.3f}\nMAE = {mae*1000:.1f} meV/átomo",transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","alpha":.9,"edgecolor":"#bbbbbb"})
    return {"n":len(y),"slope":float(fit.slope),"intercept_ev_atom":float(fit.intercept),"r2":r2,"mae_mev_atom":mae*1000}


def run() -> None:
    """Crea y guarda figuras PNG en español; nunca genera PDF ni archivo comprimido."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1",type=Path,default=REPORT/"D1/v3")
    parser.add_argument("--d3",type=Path,default=REPORT/"D3/v5")
    parser.add_argument("--out",type=Path,default=REPORT)
    args=parser.parse_args()
    plt.rcParams.update({"font.size":9,"axes.titlesize":11,"axes.labelsize":9,"figure.dpi":130})
    # D1 parity: tres paneles, escala idéntica, cada punto es una estructura OOF.
    oof=pd.read_csv(args.d1/"oof_predictions.csv")
    configs=list(COLORS)[:3]
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
    reg={}
    for ax,config in zip(axes,configs):
        g=oof[oof.configuration.eq(config)]
        reg[config]=parity(ax,g.energy_per_atom_ev.to_numpy()*1, g.energy_pred_ev_atom.to_numpy(),config,COLORS[config])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,.91),frameon=False)
    fig.suptitle("ABS₃: cuánto predice la composición de la energía estática DFT-PBE",fontsize=14,fontweight="bold",y=.99)
    fig.text(.5,.02,"Cada punto = 1 estructura estática OOF; 210 estructuras / 70 composiciones; 5 folds agrupados. Solo features de fórmula; sin test histórico.\nLa recta rosa es OLS con intercepto libre y todas las estructuras del panel; no es una calibración externa.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.88));d1fig=args.d1/"figures";d1fig.mkdir(parents=True,exist_ok=True);fig.savefig(d1fig/"d1_paridad_composicion.png",dpi=200,bbox_inches="tight");plt.close(fig)
    pd.DataFrame([{"configuration":k,**v} for k,v in reg.items()]).to_csv(args.d1/"parity_regression.csv",index=False)
    # Composición-only vs E2-A: igual población, E2 predicción promediada por semilla.
    e2path=ALT/"reports/MatGPT_finetuning/matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv"
    e2=pd.read_csv(e2path);e2=e2[e2.configuration.eq("partial_ft")].groupby("structure_id",as_index=False).agg(energy_pred_ev_atom=("energy_pred_ev_atom","mean"))
    ridge=oof[oof.configuration.eq("D1-B Ridge composición")]
    joined=ridge.merge(e2,on="structure_id",validate="one_to_one")
    fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    reg2={"D1-B Ridge composición":parity(axes[0],joined.energy_per_atom_ev.to_numpy(),joined.energy_pred_ev_atom_x.to_numpy(),"Ridge: solo composición",COLORS["D1-B Ridge composición"]),
          "E2-A partial_ft":parity(axes[1],joined.energy_per_atom_ev.to_numpy(),joined.energy_pred_ev_atom_y.to_numpy(),"MatGPT E2-A: SLICES + último bloque",COLORS["E2-A partial_ft"])}
    fig.suptitle("ABS₃: paridad de energía absoluta en los mismos 210 casos OOF",fontsize=13,fontweight="bold")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,.91),frameon=False)
    fig.text(.5,.02,"Ridge recibe la fórmula; E2-A recibe SLICES y ajuste parcial del último bloque. E2-A promedia 3 semillas.\nMismos 5 folds/210 estructuras de desarrollo; los ocho grupos de test histórico están excluidos.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.88));fig.savefig(d1fig/"d1_ridge_vs_matgpt.png",dpi=200,bbox_inches="tight");plt.close(fig)
    pd.DataFrame([{"configuration":k,**v} for k,v in reg2.items()]).to_csv(args.d1/"ridge_vs_matgpt_regression.csv",index=False)
    # Errores por composición y desglose entre y dentro de grupos químicos.
    comp=pd.read_csv(args.d1/"metrics_composition.csv")
    e2_metrics=[]
    for composition,g in pd.read_csv(e2path).query("configuration == 'partial_ft'").groupby("composition_group"):
        p=g.groupby("structure_id").energy_pred_ev_atom.mean().to_numpy();y=g.groupby("structure_id").energy_per_atom_ev.first().to_numpy()
        e2_metrics.append({"configuration":"E2-A partial_ft","composition_group":composition,"absolute_mae_mev_atom":float(np.abs(p-y).mean()*1000)})
    comp=pd.concat([comp[["configuration","composition_group","absolute_mae_mev_atom"]],pd.DataFrame(e2_metrics)],ignore_index=True)
    order=["D1-A media train","D1-B Ridge composición","D1-C RandomForest pequeño","E2-A partial_ft"]
    fig,ax=plt.subplots(figsize=(9,5));vals=[comp.loc[comp.configuration.eq(k),"absolute_mae_mev_atom"].to_numpy() for k in order]
    ax.boxplot(vals,tick_labels=["Media train","Ridge fórmula","Random Forest","MatGPT E2-A"],showfliers=True)
    ax.set_ylabel("MAE absoluto por composición (meV/átomo)");ax.set_title("ABS₃: variación del error entre composiciones",fontweight="bold");ax.grid(axis="y",alpha=.22)
    ax.text(.01,-.21,"Cada caja resume 70 composiciones; en E2-A se promedian tres semillas antes del cálculo. La línea central es la mediana,\nla caja Q1–Q3 y los puntos fuera de bigotes son composiciones individuales; el promedio no pondera por número de filas.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(d1fig/"d1_error_por_composicion.png",dpi=200,bbox_inches="tight");plt.close(fig)
    data,_,_=load_development(); y=data.energy_per_atom_ev.to_numpy(float); means=data.assign(_e=y).groupby("reduced_formula")._e.mean().reindex(data.reduced_formula)
    total=float(np.var(y)); between=float(np.var(means.to_numpy())); within=float(np.mean((y-means.to_numpy())**2)); frac=between/(between+within)
    fig,ax=plt.subplots(figsize=(7,4.6));shares=[100*frac,100*(1-frac)]
    bars=ax.barh(["Entre composiciones","Dentro de composición"],shares,color=["#2A6FBB","#D17A00"])
    ax.set_xlim(0,100);ax.set_xlabel("Fracción de la varianza total (%)");ax.set_title("ABS₃: dónde cambia más la energía DFT-PBE",fontweight="bold");ax.grid(axis="x",alpha=.2)
    ax.text(97,0,f"{shares[0]:.1f}%",ha="right",va="center",color="white",fontweight="bold")
    ax.annotate(f"{shares[1]:.1f}%",xy=(shares[1],1),xytext=(5,1),ha="left",va="center",
                arrowprops={"arrowstyle":"->","color":"#555555"},fontsize=9)
    fig.text(.5,.93,f"Varianza entre medias: {between:.4f}; dentro del grupo: {within:.4f}; total: {total:.4f} eV²/átomo²",ha="center",fontsize=9)
    fig.text(.5,.015,"70 composiciones × 3 polimorfos. Porcentaje de varianza poblacional; describe asociación, no separa causalmente\nquímica y estructura. La barra pequeña dentro de composición es 0.4%.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.88));fig.savefig(d1fig/"d1_varianza_entre_dentro.png",dpi=200,bbox_inches="tight");plt.close(fig)
    # D3 identity/discriminability and pair representation associations.
    d3fig=args.d3/"figures";d3fig.mkdir(parents=True,exist_ok=True)
    collisions=pd.read_csv(args.d3/"collision_analysis.csv")
    counts=collisions.n_slices_strings.value_counts().reindex([1,2,3],fill_value=0)
    fig,ax=plt.subplots(figsize=(7,4));bars=ax.bar(["1 cadena distinta","2 cadenas distintas","3 cadenas distintas"],counts.values,color=["#9C3B79","#D17A00","#2A6FBB"])
    for bar,n in zip(bars,counts.values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,str(n),ha="center")
    ax.set_ylim(0,75);ax.set_ylabel("Composiciones ABS₃ (n = 70)");ax.set_title("D3: ¿cuántos SLICES distintos hay entre tres polimorfos?",fontweight="bold");ax.grid(axis="y",alpha=.2)
    ax.text(.01,-.20,"Cada composición aporta exactamente tres estructuras estáticas. La igualdad se evalúa sobre la cadena de entrada\noriginal de MatGPT; tres cadenas distintas no garantizan por sí solas grafos no equivalentes.",transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(d3fig/"d3_distincion_slices.png",dpi=200,bbox_inches="tight");plt.close(fig)
    pairs=pd.read_csv(args.d3/"embedding_pair_distances.csv")
    charts=[("end_embedding_euclidean","Distancia euclidiana de <end> (adimensional)"),("mean_embedding_euclidean","Distancia euclidiana masked-mean (adimensional)"),("geometry_descriptor_l1","Distancia L1 geométrica (mezcla de escalas)")]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,(column,label) in zip(axes,charts):
        x=pairs.dft_abs_difference_mev_atom.to_numpy();z=pairs[column].to_numpy();rho=float(pd.Series(x).corr(pd.Series(z),method="spearman"))
        ax.scatter(x,z,s=17,alpha=.6,color="#187D79",edgecolors="none")
        ax.set_xlabel("|ΔE DFT entre dos polimorfos| (meV/átomo)");ax.set_ylabel(label);ax.set_title(f"Spearman ρ = {rho:.2f}",fontsize=10);ax.grid(alpha=.2)
    fig.suptitle("D3: separación de pares de polimorfos frente a su diferencia DFT",fontweight="bold",fontsize=13)
    fig.text(.5,.01,"Cada punto = 1 par dentro de una composición; 210 pares de 70 composiciones, no 210 casos independientes.\nCorrelaciones exploratorias: una distancia grande no equivale a una predicción energética correcta.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.91));fig.savefig(d3fig/"d3_distancias_y_deltae.png",dpi=200,bbox_inches="tight");plt.close(fig)
    # D2 pilot plots are created after the paired summary exists.
    d2dir=REPORT/"D2/pilot_v1"
    history_path=d2dir/"training_history_pilot.csv"
    metric_path=d2dir/"metrics_composition_pilot_paired.csv"
    if history_path.is_file() and metric_path.is_file():
        history=pd.read_csv(history_path);metrics=pd.read_csv(metric_path)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=False)
        for ax,variant in zip(axes,["D2-CENTER","D2-CENTER-ONLY"]):
            g=history[history.configuration.eq(variant)]
            ax.plot(g.epoch,g.train_objective,marker="o",markersize=3,label="Inner-train: promedio de minibatches",color="#2A6FBB")
            ax.plot(g.epoch,g.inner_validation_objective,marker="o",markersize=3,label="Inner-validation: evaluación completa",color="#D17A00")
            ax.axvline(g.loc[g.inner_validation_objective.idxmin(),"epoch"],color="#333333",linestyle="--",label="Época elegida por inner-validation")
            ax.set_title(variant);ax.set_xlabel("Época local de D2");ax.set_ylabel("Objetivo normalizado (adimensional)");ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,.93),frameon=False)
        fig.suptitle("D2: seguimiento del entrenamiento y early stopping",fontsize=13,fontweight="bold",y=1.02)
        fig.text(.5,.01,"Fold 0, semilla 42. Las curvas no son MAE; las dos configuraciones tienen pérdidas diferentes.\nTrain es promedio de minibatches; validation usa los 11 grupos internos completos. El test no se consulta.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.09,1,.84));(d2dir/"figures").mkdir(parents=True,exist_ok=True);fig.savefig(d2dir/"figures/d2_pilot_curvas.png",dpi=200,bbox_inches="tight");plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True)
        base_name="D2-BASE E2-A partial_ft"
        b=metrics[metrics.configuration.eq(base_name)].set_index("composition_group")
        for ax,variant in zip(axes,["D2-CENTER","D2-CENTER-ONLY"]):
            v=metrics[metrics.configuration.eq(variant)].set_index("composition_group")
            common=b.index.intersection(v.index);x=b.loc[common,"delta_e_mae_mev_atom"];y=v.loc[common,"delta_e_mae_mev_atom"]
            ax.scatter(x,y,s=30,alpha=.75,color=COLORS.get(variant,"#2A6FBB"),edgecolors="white",linewidths=.4)
            bound=(0,float(max(x.max(),y.max())*1.05));ax.plot(bound,bound,color="#333333",linestyle="--",label="Igual ΔE MAE")
            ax.set_title(variant);ax.set_xlabel("E2-A ΔE MAE (meV/átomo)");ax.set_ylabel("D2 ΔE MAE (meV/átomo)");ax.grid(alpha=.2)
        fig.suptitle("D2: comparación pareada de error relativo por composición",fontweight="bold",fontsize=12)
        fig.text(.5,.01,"Un punto = una composición de outer-validation (14 en este piloto). Bajo la diagonal, D2 tiene menor error relativo.\nFold 0/semilla 42; resultado exploratorio, no representa estabilidad entre folds ni semillas.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.10,1,.90));fig.savefig(d2dir/"figures/d2_pilot_deltae_pareado.png",dpi=200,bbox_inches="tight");plt.close(fig)
    LOG.info("figures_created d1=4 d3=2")


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run()


if __name__=="__main__":
    main()
