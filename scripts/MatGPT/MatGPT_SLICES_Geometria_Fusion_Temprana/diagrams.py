"""Genera los dos diagramas científicos de la fusión geométrica MatGPT."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").is_file())
EXP=ROOT/"experiments/MatGPT_SLICES_Geometria_Fusion_Temprana"
IMAGES=ROOT/"images/MatGPT_SLICES_Geometria_Fusion_Temprana"


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str,
        color: str, size: float=10) -> None:
    """Dibuja una caja redondeada y texto centrado."""
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=.03",
        facecolor=color,edgecolor="#43566b",linewidth=1.2)
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size)


def arrow(ax: plt.Axes, start: tuple[float,float], end: tuple[float,float]) -> None:
    """Conecta cajas con flecha."""
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,
        linewidth=1.3,color="#43566b",connectionstyle="arc3,rad=0"))


def architecture() -> None:
    """Genera arquitectura de inyección temprana y tardía equiparada."""
    fig,ax=plt.subplots(figsize=(16,7));ax.set_xlim(0,16);ax.set_ylim(0,7);ax.axis("off")
    ax.set_title("MatGPT: fusión geométrica antes de la atención causal",fontsize=16,weight="bold",pad=18)
    box(ax,.3,4.45,1.5,.8,"SLICES\nIDs originales","#d9e8f5")
    box(ax,2.1,4.45,1.8,.8,"Embedding token +\nposición [L,768]","#d9e8f5")
    box(ax,.3,2.7,1.5,.8,"7 descriptores\nde celda","#dff0e6")
    box(ax,2.1,2.7,1.8,.8,"StandardScaler\nsolo inner-train","#dff0e6")
    box(ax,4.35,2.7,1.9,.8,"Linear(7,768)\nsin sesgo; cero init","#fce7b2")
    box(ax,4.35,4.45,1.9,.8,"Suma por posición\nantes del bloque 0","#fce7b2")
    box(ax,6.7,4.45,2.0,.8,"MatGPT Transformer\n12 bloques causales","#e8def4")
    box(ax,9.1,4.45,1.35,.8,"ln_f\ncongelada","#dddddd")
    box(ax,10.85,4.45,1.5,.8,"FINAL_END\n768","#d9e8f5")
    box(ax,12.7,4.45,1.7,.8,"MLP8\n768→8→1","#fce7b2")
    box(ax,14.75,4.45,1.0,.8,"Residuo\ncentrado","#f4e5d0",9)
    for s,e in [((1.8,4.85),(2.1,4.85)),((3.9,4.85),(4.35,4.85)),((6.25,4.85),(6.7,4.85)),((8.7,4.85),(9.1,4.85)),((10.45,4.85),(10.85,4.85)),((12.35,4.85),(12.7,4.85)),((14.4,4.85),(14.75,4.85)),((1.8,3.1),(2.1,3.1)),((3.9,3.1),(4.35,3.1)),((5.3,3.5),(5.3,4.45))]:arrow(ax,s,e)
    ax.text(7.7,2.05,"F0: bloques congelados; el gradiente atraviesa MatGPT hacia la proyección\nF8: se actualizan únicamente los bloques 4–11",ha="center",fontsize=9,
            bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"#aaaaaa"})
    ax.text(8,.9,"Control tardío equiparado: MatGPT(SLICES) → FINAL_END + Linear(7,768) → la misma MLP8 (768→8→1).\nMisma proyección geométrica (5 376 pesos), mismo regresor y misma inicialización; solo cambia el punto de fusión.",ha="center",fontsize=11,
            bbox={"boxstyle":"round,pad=.5","facecolor":"#f7f7f7","edgecolor":"#888"})
    fig.text(.5,.02,"Esquema de la implementación auditada. No se añaden tokens, no se modifica la máscara causal ni el vocabulario; geometría de celda DFT estática.",ha="center",fontsize=9)
    IMAGES.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(IMAGES/"arquitectura_matgpt_geometria_inyectada_antes_de_atencion.png",dpi=300);plt.close(fig)


def hybrid() -> None:
    """Genera el pipeline de Ridge composicional y tres polimorfos."""
    fig,ax=plt.subplots(figsize=(15,8));ax.set_xlim(0,15);ax.set_ylim(0,8);ax.axis("off")
    ax.set_title("Predicción energética híbrida para tres polimorfos ABS₃",fontsize=16,weight="bold",pad=18)
    box(ax,.35,6.25,2.3,.75,"Composición química\n17 fracciones","#dff0e6")
    box(ax,3.2,6.25,2.1,.75,"Ridge composicional\nfijo por pliegue","#dff0e6")
    box(ax,5.85,6.25,1.9,.75,"Media μ_c\neV/átomo","#fce7b2")
    arrow(ax,(2.65,6.62),(3.2,6.62));arrow(ax,(5.3,6.62),(5.85,6.62))
    ys=[4.75,3.25,1.75]
    for lab,y in zip("ABC",ys):
        box(ax,.35,y,2.45,.8,f"Polimorfo {lab}\nSLICES_{lab} + geometría_{lab}","#d9e8f5",9)
    box(ax,3.45,3.25,2.35,1.0,"MatGPT compartido\nF0 o F8 + fusión\ntemprana/tardía","#e8def4")
    for y in ys:arrow(ax,(2.8,y+.4),(3.45,3.75))
    box(ax,6.35,3.25,1.9,1.0,"MLP8 fijo\n768→8→1","#fce7b2")
    arrow(ax,(5.8,3.75),(6.35,3.75))
    for lab,y in zip("ABC",ys):
        box(ax,8.8,y,1.7,.8,f"Residuo crudo r_{lab}","#f4e5d0",9)
    for y in ys:arrow(ax,(8.25,3.75),(8.8,y+.4))
    box(ax,10.95,3.25,1.7,1.0,"Centrado por\ntriplete: δ_i =\nr_i − media(r)","#f4e5d0",9)
    for y in ys:arrow(ax,(10.5,y+.4),(10.95,3.75))
    box(ax,13.05,3.25,1.65,1.0,"E/N_i = μ_c + δ_i\nΔE y ranking\nsolo intrafórmula","#dff0e6",9)
    ax.plot([7.75,14.15],[6.62,6.62],color="#43566b",linewidth=1.3)
    ax.plot([14.15,14.15],[6.62,4.25],color="#43566b",linewidth=1.3)
    arrow(ax,(14.15,4.25),(14.7,4.05));arrow(ax,(12.65,3.75),(13.05,3.75))
    ax.text(7.8,.55,"La energía total predicha se deriva: E_total = N × (E/N). Las etiquetas DFT solo se usan para supervisar/evaluar;\nno entran como características. Los tres polimorfos comparten pesos del extractor y MLP8.",ha="center",fontsize=10)
    fig.text(.5,.02,"Ridge predice el nivel medio de composición; la rama MatGPT predice residuos estructurales centrados. No es una pérdida pairwise directa.",ha="center",fontsize=9)
    fig.tight_layout();fig.savefig(IMAGES/"regresion_hibrida_ridge_matgpt_geometria_tres_polimorfos.png",dpi=300);plt.close(fig)


def main() -> None:
    """Escribe fuentes Mermaid y dos diagramas PNG."""
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    src=EXP/"diagram_sources";src.mkdir(parents=True,exist_ok=True)
    (src/"arquitectura_fusion_temprana_tardia.mmd").write_text("""flowchart LR
 S[SLICES IDs originales] --> E[Embedding token + posición 768]
 G[7 descriptores de celda] --> Z[StandardScaler fit inner-train]
 Z --> P[Linear 7 a 768, sin sesgo, cero init]
 E --> I[Suma por posición antes del bloque 0]
 P --> I
 I --> T[MatGPT causal 12 bloques]
 T --> L[ln_f congelada]
 L --> H[FINAL_END 768]
 H --> M[MLP8 768 a 8 a 1]
 M --> R[Residuo intrafórmula centrado]
 H --> Q[Fusión tardía equiparada: suma P a FINAL_END]
 Q --> M
 """,encoding="utf-8")
    (src/"regresion_hibrida_tres_polimorfos.mmd").write_text("""flowchart LR
 C[Composición, 17 fracciones] --> R[Ridge fijo] --> U[Media energética mu_c]
 A[SLICES A + geometría A] --> T[MatGPT compartido] --> H[MLP8] --> D[Centrar residuos A/B/C]
 B[SLICES B + geometría B] --> T
 X[SLICES C + geometría C] --> T
 U --> E[E/N por polimorfo]
 D --> E
 E --> DE[Delta E, ranking, estado fundamental]
 """,encoding="utf-8")
    architecture();hybrid()


if __name__=="__main__":main()
