"""Genera figuras PNG de energía absoluta, intrafórmula y controles H3-C."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduce import FAMILIES,project_root,sha256

COLORS={"pretrained":"#245b9b","random":"#ba6c2f","composition":"#6f8495"}
LABELS={"pretrained":"MatGPT preentrenado","random":"MatGPT aleatorio","composition":"Solo composición"}
MARKERS={"dist_perovskite":"o","hexagonal":"^","needle":"s"}
FAMILY_LABELS={"dist_perovskite":"Perovskita distorsionada","hexagonal":"Hexagonal","needle":"Aguja"}


def save(fig:Any,path:Path,note:str) -> None:
    """Coloca nota legible en espacio reservado y guarda PNG."""
    fig.text(.055,.012,textwrap.fill(note,width=160),ha="left",va="bottom",fontsize=8.1,color="#43546b")
    fig.tight_layout(rect=(.02,.105,.98,.92))
    fig.savefig(path,dpi=170,facecolor="white")
    plt.close(fig)


def parity(pred:pd.DataFrame,mode:str,path:Path) -> None:
    """Dibuja tres paneles con ejes compartidos, ideal y OLS libre."""
    if mode=="absolute":
        xcol="energy_per_atom_dft_ev";ycol="energy_per_atom_pred_ev"
        title="ABS₃: energía DFT por átomo frente a H3-C y controles"
        axis="Energía por átomo (eV/átomo)";unit="eV/átomo"
    elif mode=="residual":
        xcol="dft_centered";ycol="pred_centered"
        title="ABS₃: residuos energéticos dentro de cada fórmula"
        axis="Residuo respecto a la media de fórmula (meV/átomo)";unit="meV/átomo"
    else:
        xcol="delta_dft_mev_atom";ycol="delta_pred_mev_atom"
        title="ABS₃: diferencias firmadas entre polimorfos de una fórmula"
        axis="Diferencia B−A (meV/átomo)";unit="meV/átomo"
    all_x=pred[xcol].to_numpy(float);all_y=pred[ycol].to_numpy(float)
    low=min(all_x.min(),all_y.min());high=max(all_x.max(),all_y.max());margin=.03*(high-low)
    limits=(low-margin,high+margin)
    fig,axes=plt.subplots(1,3,figsize=(17.5,6.8),sharex=True,sharey=True)
    fig.suptitle(title,fontsize=16,weight="bold")
    for ax,kind in zip(axes,COLORS):
        part=pred[pred.model==kind]
        for family,marker in MARKERS.items():
            group=part[part.polymorph_family==family] if mode!="pairs" else part
            if mode=="pairs" and family!="dist_perovskite":continue
            ax.scatter(group[xcol],group[ycol],s=23,marker=marker,color=COLORS[kind],alpha=.65,
                       edgecolor="white",linewidth=.2,label=FAMILY_LABELS[family] if mode!="pairs" else "Par de polimorfos")
        x=part[xcol].to_numpy(float);y=part[ycol].to_numpy(float)
        slope,intercept=np.polyfit(x,y,1)
        fit=slope*x+intercept
        r2=1-np.sum((y-fit)**2)/np.sum((y-y.mean())**2) if np.var(y)>0 else np.nan
        ax.plot(limits,limits,"--",color="#69788c",label="Acuerdo ideal")
        ax.plot(limits,[slope*v+intercept for v in limits],color="#a73f98",label="Ajuste lineal")
        ax.set(xlim=limits,ylim=limits,title=LABELS[kind],xlabel="DFT: "+axis)
        if ax is axes[0]:ax.set_ylabel("Predicción: "+axis)
        ax.grid(alpha=.16)
        r2_text=f"{r2:.3f}" if np.isfinite(r2) else "no definido (salida constante)"
        ax.text(.03,.97,f"N = {len(x)}\nPendiente = {slope:.3f}\nIntercepto = {intercept:.3f} {unit}\nR² ajuste = {r2_text}\nError medio = {np.mean(np.abs(y-x)):.3f} {unit}",
                transform=ax.transAxes,va="top",fontsize=8.2,bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=len(labels),frameon=False,bbox_to_anchor=(.5,.08),fontsize=8.5)
    note="Validación de desarrollo; semilla 42, cinco pliegues, 70 fórmulas y 210 estructuras físicas. DFT: teoría del funcional de la densidad. Cada punto es "
    note+=("una estructura. Los residuos usan la media propia de cada método y fórmula." if mode=="residual" else "un par dentro de una fórmula; tres pares por fórmula." if mode=="pairs" else "una estructura estática.")
    save(fig,path,note)


def make(root:Path) -> dict[str,Any]:
    """Construye seis figuras desde tablas auditadas y guarda hashes."""
    report=root/"reports/MatGPT/MatGPT_H3C_Final"
    analysis=report/"analysis";output=report/"figures"
    pred=pd.read_csv(analysis/"predictions_by_structure.csv")
    pair=pd.read_csv(analysis/"pairs_gt5.csv")
    summary=pd.read_csv(analysis/"summary_by_seed.csv")
    bins=pd.read_csv(analysis/"delta_bins.csv")
    representations=pd.read_csv(analysis/"embedding_pairs.csv")
    selected=pred[pred.seed==42].copy()
    selected["dft_centered"]=1000*(selected.energy_per_atom_dft_ev-selected.groupby(["model","composition_id"]).energy_per_atom_dft_ev.transform("mean"))
    selected["pred_centered"]=1000*(selected.energy_per_atom_pred_ev-selected.groupby(["model","composition_id"]).energy_per_atom_pred_ev.transform("mean"))
    parity(selected,"absolute",output/"01_energia_absoluta.png")
    parity(selected,"residual",output/"02_residuos_intraformula.png")
    pairs42=pair[pair.seed==42].copy();pairs42["polymorph_family"]="dist_perovskite"
    parity(pairs42,"pairs",output/"03_diferencias_pares.png")
    fig,axes=plt.subplots(1,4,figsize=(15.5,5.6));fig.suptitle("ABS₃: energía media, residuos y ordenamiento por modelo",fontsize=16,weight="bold")
    metrics=[("atom_mae_mev","Error de energía por átomo","meV/átomo"),
             ("residual_mae_mev","Error de residuo intrafórmula","meV/átomo"),
             ("gs_accuracy","Estado fundamental exacto","fracción elegible"),
             ("pair_accuracy_gt5","Orden de pares con |ΔE DFT| > 5","fracción elegible")]
    for ax,(column,title,unit) in zip(axes,metrics):
        vals=[summary[summary.model==kind][column].mean() for kind in COLORS]
        ax.bar(range(3),vals,color=[COLORS[k] for k in COLORS]);ax.set_xticks(range(3),["Preentrenado","Aleatorio","Composición"],rotation=18,ha="right")
        ax.set(title=title,ylabel=unit);ax.grid(axis="y",alpha=.15)
        for i,value in enumerate(vals):ax.text(i,value,f"{value:.2f}",ha="center",va="bottom",fontsize=8)
    save(fig,output/"04_resumen_controles.png","Validación de desarrollo, cinco pliegues y 70 fórmulas. Cada barra promedia tres semillas sobre los mismos materiales. Estado fundamental: 67 fórmulas elegibles por semilla tras excluir empates DFT de 1 meV/átomo; orden: 197 pares elegibles por semilla con |ΔE DFT| > 5. Composición empata y no recibe aciertos arbitrarios.")
    fig,ax=plt.subplots(figsize=(10.8,6.3));fig.suptitle("ABS₃: error de diferencias por magnitud del gap DFT",fontsize=15,weight="bold")
    labels=["0–5","5–10","10–25","25–50","50–100",">100"]
    for kind in COLORS:
        sub=bins[bins.model==kind].groupby(["lower_mev","upper_mev"],as_index=False).agg(mae=("mae_model_mev","mean"),n=("n_pairs","mean"))
        ax.plot(range(6),sub.mae,"-o",color=COLORS[kind],label=LABELS[kind])
    zero=bins[bins.model=="pretrained"].groupby(["lower_mev","upper_mev"],as_index=False).mae_zero_mev.mean()
    ax.plot(range(6),zero.mae_zero_mev,"--o",color="#586370",label="Diferencia predicha igual a cero")
    ns=bins[bins.model=="pretrained"].groupby(["lower_mev","upper_mev"],as_index=False).n_pairs.mean().n_pairs.astype(int).tolist()
    ax.set_xticks(range(6),[f"{label}\n(n={n})" for label,n in zip(labels,ns)])
    ax.set(xlabel="|Diferencia DFT| (meV/átomo)",ylabel="Error absoluto medio de pares (meV/átomo)")
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    save(fig,output/"05_error_segun_gap.png","Cada intervalo contiene pares de las mismas fórmulas; n indica pares físicos por semilla. Tres semillas son repeticiones, no pares nuevos. El orden solo se puntúa cuando |ΔE DFT| > 5 meV/átomo.")
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.9));fig.suptitle("ABS₃: distancia entre embeddings y diferencia energética DFT",fontsize=15,weight="bold")
    for ax,(kind,seed) in zip(axes,(("pretrained",-1),("random",42))):
        part=representations[(representations.backbone==kind)&(representations.seed==seed)&(representations.feature=="FINAL_END")]
        ax.scatter(part.euclidean_standardized,part.dft_abs_gap_mev_atom,s=22,alpha=.6,color=COLORS[kind])
        ax.set(title=LABELS[kind],xlabel="Distancia euclidiana estandarizada",ylabel="|Diferencia DFT| (meV/átomo)")
        ax.set_xlim(0,56)
        ax.set_ylim(0,410)
        ax.grid(alpha=.15)
    save(fig,output/"06_distancia_embeddings_gap.png","210 pares de 70 fórmulas; semilla aleatoria 42. Cada dimensión del embedding se estandarizó para este diagnóstico, sin etiquetas. La asociación es descriptiva y no prueba causalidad.")
    files=sorted(output.glob("0[1-6]_*.png"))
    if len(files)!=6:raise ValueError(f"Se esperaban 6 figuras, hay {len(files)}")
    manifest={"status":"PASS","figures":{p.name:sha256(p) for p in files},"historical_test_accessed":False}
    (output/"data_figure_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    return manifest


def main() -> None:
    """Acepta raíz del proyecto y escribe solo figuras PNG de resultados."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    args=parser.parse_args()
    make(args.repo.resolve())


if __name__=="__main__":main()
