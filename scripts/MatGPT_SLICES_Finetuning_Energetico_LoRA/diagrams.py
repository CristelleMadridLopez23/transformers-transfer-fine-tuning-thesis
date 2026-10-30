"""Dibuja la arquitectura MatGPT/H3-C real con adaptadores LoRA Q/V."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import experiment as ex

FROZEN = "#D9E7F1"
LORA = "#F8DDC9"
HEAD = "#DCEEDB"
MATH = "#EEEAF3"
LOG = logging.getLogger("matgpt_lora_diagrams")


def box(ax: plt.Axes, x: float, y: float, width: float, height: float,
        label: str, color: str) -> None:
    """Dibuja un módulo u operación con texto centrado."""
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.2, edgecolor="#314156", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)


def arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float]) -> None:
    """Conecta dos módulos con flecha visible."""
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.5, color="#34465C"))


def export(fig: plt.Figure, output: Path, stem: str) -> None:
    """Guarda PNG y SVG de diagrama y cierra la figura."""
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def pipeline(output: Path) -> None:
    """Muestra las ramas paralelas Ridge y MLP8 más recombinación física."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set(xlim=(0, 16), ylim=(0, 7)); ax.axis("off")
    fig.suptitle("MatGPT + LoRA: regresión energética híbrida H3-C real", fontsize=17, fontweight="bold")
    box(ax, .3, 3, 1.4, .8, "SLICES\noriginal", MATH)
    box(ax, 2.1, 3, 1.5, .8, "Tokenizador\n172 tokens", FROZEN)
    box(ax, 4, 3, 1.9, .8, "MatGPT bloques\n0–3 congelados", FROZEN)
    box(ax, 6.6, 4.7, 2.1, .85, "L03_MEAN × 3\nmedia fórmula: 768", FROZEN)
    box(ax, 9.3, 4.7, 2, .85, "Scaler + Ridge\ncongelado", FROZEN)
    box(ax, 6.6, 2.3, 2.1, .85, "Bloques 4–9 frozen\n10–11: LoRA Q/V", LORA)
    box(ax, 9.3, 2.3, 2, .85, "FINAL_END 768\nMLP8 768→8→1\nGELU; Dropout 0,1", HEAD)
    box(ax, 12, 3.5, 1.8, .85, "μ Ridge + δ\ncentrado", MATH)
    box(ax, 14.2, 3.5, 1.5, .85, "e = E/N\nE = N·e", MATH)
    for a,b in [((1.7,3.4),(2.1,3.4)),((3.6,3.4),(4,3.4)),
                ((5.9,3.55),(6.6,5.05)),((5.9,3.3),(6.6,2.75)),
                ((8.7,5.1),(9.3,5.1)),((8.7,2.7),(9.3,2.7)),
                ((11.3,5.1),(12.4,4.37)),((11.3,2.7),(12.4,3.5)),
                ((13.8,3.9),(14.2,3.9))]: arrow(ax,a,b)
    ax.text(8, 1.8, "Entrenamiento: SmoothL1 β=1 sobre residuos / desviación residual del train interno",
            ha="center", fontsize=9)
    ax.text(8, 1.2, "Residuo: δᵢ = MLP8(hᵢ) − media(MLP8(h_A), MLP8(h_B), MLP8(h_C))",
            ha="center", fontsize=10)
    ax.text(8, .55, "ΔEᵢⱼ = 1000·(eⱼ−eᵢ); estado fundamental = argmin(e_A,e_B,e_C).  "
            "Ridge y MLP8 son ramas paralelas.", ha="center", fontsize=9)
    fig.text(.5, .02, "Diagrama de la implementación auditada: base 114 026 668 congelados; "
             "LoRA 24 576 + MLP8 6 161 entrenables (0,027 % del sistema).", ha="center", fontsize=8)
    export(fig, output, "arquitectura_regresion_energetica_matgpt_finetuning_lora")
    (output / "arquitectura_regresion_energetica_matgpt_finetuning_lora.mmd").write_text(
        "flowchart LR\n S[SLICES] --> T[Tokenizer 172] --> B[Bloques 0-3 frozen]\n"
        " B --> L[L03_MEAN 768] --> R[Scaler + Ridge frozen]\n"
        " B --> U[Bloques 4-9 frozen; 10-11 LoRA Q/V] --> F[FINAL_END 768] --> H[MLP8 768-8-1]\n"
        " H --> C[Residuo centrado] --> A[Sumar media Ridge]\n R --> A --> E[E/N; E=N*e] --> D[Delta E y ranking]\n")


def polymorphs(output: Path) -> None:
    """Representa tres estructuras, pesos compartidos y diferencias posteriores."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlim=(0,14),ylim=(0,6)); ax.axis("off")
    fig.suptitle("ABS₃: tres polimorfos mediante MatGPT y MLP8 compartidos", fontsize=16, fontweight="bold")
    for row, label in zip((4.4,3,1.6), ("A","B","C")):
        box(ax,.4,row,1.4,.7,f"SLICES {label}",MATH)
        box(ax,2.5,row,2.2,.7,"MatGPT + LoRA\ncompartidos",LORA)
        box(ax,5.4,row,1.9,.7,"FINAL_END 768\nMLP8",HEAD)
        box(ax,8,row,1.9,.7,f"δ_{label} centrado\nμ Ridge común",MATH)
        box(ax,10.6,row,1.5,.7,f"e_{label} = E_{label}/N_{label}",MATH)
        for a,b in [((1.8,row+.35),(2.5,row+.35)),((4.7,row+.35),(5.4,row+.35)),
                    ((7.3,row+.35),(8,row+.35)),((9.9,row+.35),(10.6,row+.35))]: arrow(ax,a,b)
    ax.text(7,.65,"Pérdida train: SmoothL1 sobre residuos centrados; ΔE_AB, ΔE_AC, ΔE_BC y argmin(e) se calculan después.",
            ha="center",fontsize=9)
    fig.text(.5,.02,"Diagrama de la implementación auditada; una MLP8 y un adaptador compartidos, sin pérdida pairwise.",
             ha="center",fontsize=8)
    export(fig,output,"procesamiento_tres_polimorfos_con_finetuning_compartido")
    (output / "procesamiento_tres_polimorfos_con_finetuning_compartido.mmd").write_text(
        "flowchart LR\n A[SLICES A] --> M[MatGPT + LoRA compartidos]\n"
        " B[SLICES B] --> M\n C[SLICES C] --> M\n M --> H[FINAL_END; MLP8 compartida]\n"
        " H --> D[Residuos centrados]\n R[Ridge media comun] --> E[e_A,e_B,e_C]\n"
        " D --> E --> P[Delta E y argmin]\n")


def blocks(output: Path) -> None:
    """Ubica Q/V LoRA en bloques 10 y 11 de doce DecoderLayer reales."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set(xlim=(0,15),ylim=(0,5)); ax.axis("off")
    fig.suptitle("MatGPT: LoRA Q/V solo en los bloques 10 y 11", fontsize=17, fontweight="bold")
    for index in range(12):
        x = .25 + index * 1.22
        color = LORA if index in (10,11) else FROZEN
        label = f"{index}\nQ/V LoRA" if index in (10,11) else f"{index}\nfrozen"
        box(ax,x,2.4,1.1,1,label,color)
        if index < 11: arrow(ax,(x+1.1,2.9),(x+1.22,2.9))
    ax.text(4.45,1.65,"L03_MEAN: bloque 3 original, sin cambio para Ridge",ha="center",fontsize=10)
    ax.text(12.2,1.65,"Q/V: deltas de rango 4 en in_proj_weight fusionado\nK, cross-attention y base sin modificar",
            ha="center",fontsize=10)
    ax.text(7.5,.65,"LoRA: 24 576; MLP8: 6 161 entrenables (0,027 %). Base: 114 026 668 congelados.",
            ha="center",fontsize=10)
    fig.text(.5,.02,"Diagrama de la implementación auditada nn.TransformerDecoderLayer y nn.MultiheadAttention.",
             ha="center",fontsize=8)
    export(fig,output,"bloques_transformer_congelados_y_adaptadores_lora")
    (output / "bloques_transformer_congelados_y_adaptadores_lora.mmd").write_text(
        "flowchart LR\n B0[0 frozen] --> B1[1 frozen] --> B2[2 frozen] --> B3[3 frozen]\n"
        " B3 --> B4[4 frozen] --> B5[5 frozen] --> B6[6 frozen] --> B7[7 frozen]\n"
        " B7 --> B8[8 frozen] --> B9[9 frozen] --> B10[10 LoRA Q/V] --> B11[11 LoRA Q/V]\n"
        " B3 --> R[L03_MEAN to frozen Ridge]\n")


def main() -> None:
    """Exporta tres diagramas PNG/SVG y Mermaid en carpeta de reportes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.REPORT / "figures")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = args.output.resolve()
    pipeline(output); polymorphs(output); blocks(output)
    LOG.info("%s", {"event": "diagrams_complete", "directory": str(output), "diagram_count": 3})


if __name__ == "__main__":
    main()
