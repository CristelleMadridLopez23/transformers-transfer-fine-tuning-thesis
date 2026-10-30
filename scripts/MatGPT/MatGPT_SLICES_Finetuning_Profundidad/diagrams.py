"""Genera diagramas técnicos editables del fine-tuning MatGPT y la fusión energética."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_matgpt_depth"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
OUT = ROOT / "images/MatGPT_SLICES_Finetuning_Profundidad"
SOURCES = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad/diagram_sources"
COLORS = {"freeze": "#DCE6F1", "f2": "#4C78A8", "f4": "#F2A541", "f8": "#59A14F",
          "structure": "#B9D8C2", "formula": "#F5D7A1", "output": "#E5C3E8"}


def box(axis: plt.Axes, x: float, y: float, width: float, height: float, text: str,
        color: str, size: int = 9) -> None:
    """Dibuja un módulo redondeado con texto centrado."""
    shape = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.025,rounding_size=0.05",
                           facecolor=color, edgecolor="#344054", linewidth=1.2)
    axis.add_patch(shape)
    axis.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=size, wrap=True)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Une dos módulos con una flecha simple."""
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                   color="#475467", linewidth=1.3))


def architecture() -> None:
    """Exporta arquitectura de profundidad en Mermaid y PNG."""
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    lines = ["flowchart LR", "  A[SLICES estándar] --> B[Tokenizador MatGPT congelado]",
             "  B --> C[Embeddings de tokens y posiciones congelados]"]
    for i in range(12):
        lines.append(f'  C --> L{i}[Bloque Transformer {i}]')
        if i < 11:
            lines.append(f"  L{i} --> L{i+1}")
    lines += ["  L11 --> D[ln_f congelada]", "  D --> E[Estado contextualizado FINAL_END, 768]",
              "  E --> H[MLP8 entrenable: Linear 768×8, GELU, Dropout 0.1, Linear 8×1]",
              "  H --> R[Residuo por polimorfo, centrado en triplete]",
              "  classDef congelado fill:#DCE6F1,stroke:#344054", "  classDef f2 fill:#4C78A8,color:#fff",
              "  classDef f4 fill:#F2A541", "  classDef f8 fill:#59A14F,color:#fff",
              "  class A,B,C,D,E congelado", "  class L10,L11 f2", "  class L8,L9 f4", "  class L4,L5,L6,L7 f8"]
    (SOURCES / "arquitectura_matgpt_bloques_congelados_y_entrenables.mmd").write_text("\n".join(lines)+"\n")
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("MatGPT: bloques originales actualizados en cada profundidad", fontsize=17, pad=18, weight="bold")
    box(ax,.2,3.3,1.5,1,"Secuencia\nSLICES", "#FFFFFF")
    box(ax,2,3.3,2,1,"Tokenizador y\nembeddings congelados", COLORS["freeze"])
    arrow(ax,(1.7,3.8),(2,3.8))
    # Bloques colocados en una línea, con color acumulativo según el experimento más profundo que los actualiza.
    for i in range(12):
        color = COLORS["f8"] if 4 <= i <= 7 else COLORS["f4"] if 8 <= i <= 9 else COLORS["f2"] if i >= 10 else COLORS["freeze"]
        x=4.35+i*.72
        box(ax,x,3.47,.62,.66,str(i),color,9)
        if i==0: arrow(ax,(4,3.8),(x,3.8))
        if i>0: arrow(ax,(x-.12,3.8),(x,3.8))
    box(ax,13.2,3.3,1.15,1,"ln_f\ncongelada",COLORS["freeze"])
    arrow(ax,(12.65,3.8),(13.2,3.8))
    box(ax,14.65,3.3,1.1,1,"FINAL_END\n768",COLORS["structure"])
    arrow(ax,(14.35,3.8),(14.65,3.8))
    box(ax,12.3,1.25,2.2,.9,"MLP8 residual\n6,161 parámetros",COLORS["output"])
    arrow(ax,(15.2,3.3),(13.4,2.15))
    ax.text(8.4,5.25,"F2 actualiza 10–11     |     F4 actualiza 8–11     |     F8 actualiza 4–11",
            ha="center",fontsize=11,weight="bold")
    ax.text(8.5,.55,"F0: ningún bloque se actualiza. Las incrustaciones de tokens/posiciones, ln_f y cabeza generativa\npermanecen congeladas. Sin LoRA ni cambios de vocabulario.",ha="center",fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/"arquitectura_matgpt_bloques_congelados_y_entrenables.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def hybrid_pipeline() -> None:
    """Exporta esquema paralelo de Ridge estequiométrico y residuo MatGPT."""
    mermaid = """flowchart LR
  F[Fórmula química] --> V[17 fracciones estequiométricas]
  V --> S[StandardScaler fijo]
  S --> R[Ridge por pliegue, pesos congelados]
  R --> M[Media energética por fórmula]
  A[SLICES polimorfo A] --> T[MatGPT parcialmente ajustado]
  B[SLICES polimorfo B] --> T
  C[SLICES polimorfo C] --> T
  T --> H[FINAL_END 768 → MLP8]
  H --> Z[Residuos centrados: suma cero]
  M --> P[Energía predicha por átomo: media + residuo]
  Z --> P
  P --> E[Energía total = N × E/N]
  P --> D[Diferencias ΔE = 1000 × (eB − eA), meV/átomo]
  D --> G[Orden de polimorfos y estado fundamental]
  classDef fijo fill:#F5D7A1,stroke:#344054
  classDef entrenable fill:#B9D8C2,stroke:#344054
  classDef resultado fill:#E5C3E8,stroke:#344054
  class V,S,R fijo
  class T,H entrenable
  class M,Z,P,E,D,G resultado
"""
    (SOURCES / "pipeline_regresion_hibrida_ridge_composicional_finetuning.mmd").write_text(mermaid)
    fig, ax = plt.subplots(figsize=(16, 8)); ax.set_xlim(0,16); ax.set_ylim(0,8); ax.axis("off")
    ax.set_title("Predicción energética: rama composicional y rama estructural en paralelo", fontsize=16, weight="bold", pad=18)
    box(ax,.4,5.2,1.7,1,"Fórmula química",COLORS["formula"])
    box(ax,2.65,5.2,2,1,"17 fracciones\nestequiométricas",COLORS["formula"])
    box(ax,5.2,5.2,1.9,1,"StandardScaler\n+ Ridge por pliegue",COLORS["formula"])
    box(ax,7.6,5.2,2,1,"Media energética\npor fórmula",COLORS["output"])
    arrow(ax,(2.1,5.7),(2.65,5.7)); arrow(ax,(4.65,5.7),(5.2,5.7)); arrow(ax,(7.1,5.7),(7.6,5.7))
    ax.text(6.1,6.45,"Ridge fijo; no se reajusta",ha="center",fontsize=9)
    for y,name in ((3.45,"Polimorfo A"),(2.3,"Polimorfo B"),(1.15,"Polimorfo C")):
        box(ax,.4,y,1.7,.72,"SLICES · "+name,"#FFFFFF")
        arrow(ax,(2.1,y+.36),(2.65,y+.36))
    box(ax,2.65,2.3,2.2,1.25,"MatGPT congelado\no F2 / F4 / F8",COLORS["structure"])
    for y in (3.81,2.66,1.51): arrow(ax,(2.4,y),(2.65,2.92))
    box(ax,5.35,2.45,1.8,.95,"FINAL_END\n768 dimensiones",COLORS["structure"])
    box(ax,7.65,2.45,1.9,.95,"MLP8 residual\ncentrada en triplete",COLORS["output"])
    arrow(ax,(4.85,2.92),(5.35,2.92)); arrow(ax,(7.15,2.92),(7.65,2.92))
    box(ax,10.25,3.75,2.1,1,"Media + residuo\nE/N predicha",COLORS["output"])
    arrow(ax,(9.6,5.5),(10.25,4.35)); arrow(ax,(9.55,2.92),(10.25,4.05))
    box(ax,13,5.15,2.3,.9,"Energía total\nN × E/N", "#FFFFFF")
    box(ax,13,3.75,2.3,.9,"ΔE entre polimorfos\nmeV/átomo", "#FFFFFF")
    box(ax,13,2.35,2.3,.9,"Ranking y estado\nfundamental", "#FFFFFF")
    arrow(ax,(12.35,4.35),(13,5.55)); arrow(ax,(12.35,4.25),(13,4.2)); arrow(ax,(14.15,3.75),(14.15,3.25))
    ax.text(8,.35,"La pérdida de entrenamiento supervisa residuos DFT por átomo normalizados; no se optimiza una pérdida directa de ΔE.",
            ha="center",fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"pipeline_regresion_hibrida_ridge_composicional_finetuning.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); hybrid_pipeline()
