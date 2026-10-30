"""Dibuja la topología H3-C auditada en SVG/PNG y su fuente Mermaid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch

from reproduce import project_root,sha256


def box(ax:Any,x:float,y:float,w:float,h:float,title:str,detail:str,color:str) -> None:
    """Traza un bloque con título, dimensión y operación histórica."""
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.02",
                                facecolor=color,edgecolor="#23344d",linewidth=1.2))
    ax.text(x+w/2,y+h*.63,title,ha="center",va="center",fontsize=11,weight="bold",color="#16273b")
    ax.text(x+w/2,y+h*.28,detail,ha="center",va="center",fontsize=8.2,color="#23344d",linespacing=1.25)


def arrow(ax:Any,start:tuple[float,float],end:tuple[float,float],label:str="") -> None:
    """Conecta dos operaciones orientadas sin invadir las cajas."""
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,
                                 linewidth=1.35,color="#3c536f",connectionstyle="arc3,rad=0"))
    if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.015,label,ha="center",fontsize=8,color="#3c536f")


def route(ax:Any,points:list[tuple[float,float]]) -> None:
    """Traza un conector ortogonal por espacios blancos entre los bloques."""
    ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],
            color="#3c536f",linewidth=1.35,solid_capstyle="round")
    arrow(ax,points[-2],points[-1])


def architecture(out:Path) -> None:
    """Representa las dos ramas paralelas y su suma residual en el espacio físico."""
    fig,ax=plt.subplots(figsize=(17.5,10.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    fig.suptitle("H3-C: Ridge composicional + MLP8 residual sobre MatGPT congelado",fontsize=18,weight="bold",y=.98)
    box(ax,.025,.81,.15,.105,"SLICES original","una cadena por estructura", "#eaf2fb")
    box(ax,.215,.81,.17,.105,"Tokenizador MatGPT","172 tokens; <start>, <end>, <pad>\nIDs y máscara causal", "#eaf2fb")
    box(ax,.425,.81,.19,.105,"MatGPT preentrenado","12 bloques · 768 dimensiones\n114 026 668 parámetros congelados", "#dceaf8")
    box(ax,.66,.81,.27,.105,"Estados contextualizados","bloque 3 bruto y salida final\nningún peso actualizado", "#dceaf8")
    for a,b in [((.175,.862),(.215,.862)),((.385,.862),(.425,.862)),((.615,.862),(.66,.862))]:arrow(ax,a,b)
    box(ax,.035,.55,.22,.15,"Rama Ridge: capa 3","media de tokens SLICES válidos\npor estructura: [768] · antes de ln_f", "#dceaf8")
    box(ax,.29,.55,.21,.15,"Media de 3 polimorfos","una fila por fórmula: [768]\noperación sin parámetros", "#dceaf8")
    box(ax,.54,.55,.31,.15,"StandardScaler + Ridge","768 → 1; α elegido en validación interna\nrefit 56 fórmulas; 768 coef. + intercepto", "#d8f1e6")
    route(ax,[(.93,.862),(.965,.862),(.965,.74),(.145,.74),(.145,.705)])
    arrow(ax,(.255,.625),(.29,.625));arrow(ax,(.50,.625),(.54,.625))
    box(ax,.035,.29,.22,.15,"Rama MLP8: final","hidden de <end> tras ln_f [768]\npor estructura, antes de lm_head", "#dceaf8")
    box(ax,.29,.29,.28,.15,"MLP8 fija","Linear 768→8 · GELU · dropout 0.1\nLinear 8→1 · 6 161 parámetros", "#f9ead8")
    box(ax,.61,.29,.25,.15,"Centro por fórmula","δᵢ = rᵢ − media(r₁,r₂,r₃)\nSmoothL1 normalizada en entrenamiento", "#f9ead8")
    route(ax,[(.93,.84),(.98,.84),(.98,.48),(.145,.48),(.145,.445)])
    arrow(ax,(.255,.365),(.29,.365));arrow(ax,(.57,.365),(.61,.365))
    box(ax,.085,.08,.28,.125,"Media energética","μc de Ridge · eV/átomo", "#d8f1e6")
    box(ax,.395,.08,.28,.125,"Predicción histórica","eᵢ = μc + δᵢ · eV/átomo\nEtotal,i = Nᵢ × eᵢ (derivado)", "#eaf2fb")
    box(ax,.715,.08,.24,.125,"Evaluación posterior","ΔE = 1000(eB − eA)\nGS = argmin(eA,eB,eC)", "#eaf2fb")
    route(ax,[(.70,.55),(.70,.50),(.02,.50),(.02,.235),(.225,.235),(.225,.205)])
    route(ax,[(.735,.29),(.735,.255),(.535,.255),(.535,.205)])
    arrow(ax,(.365,.142),(.395,.142));arrow(ax,(.675,.142),(.715,.142))
    fig.text(.05,.017,"Implementación histórica auditada: Ridge y MLP8 son ramas paralelas; la MLP8 aprende residuos por átomo, no energía total ni una pérdida por pares.",fontsize=9,color="#42526b")
    fig.savefig(out/"h3c_architecture.svg",bbox_inches="tight",facecolor="white")
    fig.savefig(out/"h3c_architecture.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def polymorphs(out:Path) -> None:
    """Muestra mismo predictor aplicado a A/B/C y diferencias derivadas."""
    fig,ax=plt.subplots(figsize=(16,6.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    fig.suptitle("H3-C: tres polimorfos, un extractor y un regresor compartidos",fontsize=17,weight="bold",y=.98)
    for i,(name,color) in enumerate((("A · perovskita distorsionada","#dceaf8"),("B · hexagonal","#e6ecfa"),("C · aguja","#eee7f8"))):
        y=.72-i*.23
        box(ax,.03,y,.18,.15,name,"SLICESᵢ y Nᵢ",color)
        box(ax,.28,y,.27,.15,"MatGPT congelado","mismos pesos · L03 y final", "#dceaf8")
        box(ax,.63,y,.30,.15,"H3-C compartido",f"μc + δ{i+1} = e{i+1} (eV/átomo)","#d8f1e6")
        arrow(ax,(.21,y+.075),(.28,y+.075));arrow(ax,(.55,y+.075),(.63,y+.075))
    fig.text(.05,.065,"Después de predecir: Etotal,i = Nᵢ eᵢ; ΔEAB = 1000(eB−eA), ΔEAC = 1000(eC−eA), ΔEBC = 1000(eC−eB); estado fundamental = argmin(eA,eB,eC).",fontsize=9)
    fig.text(.05,.025,"Las diferencias y el orden se calculan después de inferencia: H3-C histórico entrenó media por fórmula y residuos centrados, sin pérdida pairwise.",fontsize=9,color="#42526b")
    fig.savefig(out/"h3c_three_polymorphs.svg",bbox_inches="tight",facecolor="white")
    fig.savefig(out/"h3c_three_polymorphs.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def source(out:Path) -> None:
    """Guarda descripción Mermaid editable equivalente a los diagramas renderizados."""
    mermaid="""flowchart LR
    S["SLICES original por estructura"] --> T["Tokenizador MatGPT: 172 tokens; start/end/pad"]
    T --> M["MatGPT preentrenado congelado: 12 bloques, d=768"]
    M --> L3["Bloque 3: media de tokens SLICES válidos; 768"]
    L3 --> G["Media de 3 embeddings por composición; 768"]
    G --> R["StandardScaler + Ridge: 768→1; alpha inner; refit 56 fórmulas"]
    M --> F["Salida post-ln_f en end; 768 por estructura"]
    F --> H["MLP8: Linear 768→8, GELU, Dropout .1, Linear 8→1"]
    H --> C["Centro de tres salidas: delta_i = r_i − media(r)"]
    R --> E["e_i = mu_c + delta_i; eV/átomo"]
    C --> E
    E --> U["E_total = N_i e_i; ΔE y argmin posteriores"]
"""
    (out/"h3c_architecture.mmd").write_text(mermaid)


def main() -> None:
    """Genera y verifica cuatro archivos de diagrama en carpeta final."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=project_root())
    args=parser.parse_args()
    out=args.repo.resolve()/"reports/MatGPT/MatGPT_H3C_Final/figures"
    out.mkdir(parents=True,exist_ok=True)
    source(out);architecture(out);polymorphs(out)
    files=[out/name for name in ("h3c_architecture.mmd","h3c_architecture.svg","h3c_architecture.png","h3c_three_polymorphs.svg","h3c_three_polymorphs.png")]
    if any(not p.is_file() or p.stat().st_size<(200 if p.suffix==".mmd" else 1000) for p in files):
        raise ValueError("Diagrama faltante o vacío")
    (out/"diagram_manifest.json").write_text(json.dumps({p.name:sha256(p) for p in files},indent=2)+"\n")


if __name__=="__main__":main()
