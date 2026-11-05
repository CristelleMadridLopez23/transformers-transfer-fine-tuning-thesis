"""Genera esquemas metodológicos y distribuciones del conjunto ABS₃ para tesis.

Los esquemas describen el protocolo vigente de extractores originales y el
fine-tuning supervisado F0/F1/F2. Las distribuciones se calculan solo con las
210 estructuras del conjunto de desarrollo; nunca se lee el test histórico.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
STRUCTURES = ROOT / "experiments/02_complementarios/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl"
OUT = ROOT / "images/thesis_plots"
RESOURCE_OUT = ROOT / "experiments/thesis_plots_generation/resources.json"
LOGGER = logging.getLogger("abs3_thesis_plots")

NAVY = "#17324D"
BLUE = "#2878A5"
TEAL = "#228C83"
GOLD = "#D5A12E"
RED = "#C75B52"
PURPLE = "#79599F"
INK = "#24313B"
MUTED = "#637381"
LIGHT = "#EEF3F6"
GREEN = "#4A8D68"
MODEL_COLORS = {"MatGPT": BLUE, "MatterGPT": TEAL}
ELEMENT_COLORS = {"Ba": NAVY, "Ge": RED, "S": GOLD}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def read_development(path: Path) -> pd.DataFrame:
    """Lee y valida únicamente la cohorte ABS₃ de desarrollo.

    Args:
        path: CSV de desarrollo de 210 estructuras.
    Returns:
        Tabla con tripletes, cadenas SLICES y descriptores geométricos.
    Raises:
        ValueError: Si la cohorte o sus variables no cumplen el protocolo.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Falta el dataset de desarrollo: {path}")
    data = pd.read_csv(path)
    required = {"structure_id", "composition_id", "family", "energy_dft_ev_atom", "SLICES",
                "a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
                "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3", "split"}
    if required - set(data.columns):
        raise ValueError(f"Faltan columnas del dataset: {sorted(required-set(data.columns))}")
    data = data[data["split"].eq("development")].copy()
    if (len(data) != 210 or data.structure_id.nunique() != 210 or
            data.composition_id.nunique() != 70 or data.groupby("composition_id").size().ne(3).any()):
        raise ValueError("El desarrollo debe tener 70 fórmulas, 210 IDs y tripletes completos")
    if not data.SLICES.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Hay SLICES vacías en el conjunto de desarrollo")
    return data


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de una fuente efectivamente usada.

    Args:
        path: Archivo fuente existente.
    Returns:
        Hash SHA-256 hexadecimal.
    """
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, filename: str, *, dpi: int = 300) -> None:
    """Guarda una figura PNG en la carpeta solicitada.

    Args:
        fig: Figura Matplotlib terminada.
        filename: Nombre local PNG.
        dpi: Resolución de exportación.
    Returns:
        None.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    LOGGER.info("Figura escrita: %s", path)


def page(title: str, subtitle: str | None = None, *, size: tuple[float, float] = (14, 7.6)) -> tuple[plt.Figure, plt.Axes]:
    """Crea una página con título y área de dibujo.

    Args:
        title: Título informativo en español.
        subtitle: Alcance o descripción breve.
        size: Tamaño en pulgadas.
    Returns:
        Tupla figura/eje.
    """
    fig, ax = plt.subplots(figsize=size)
    fig.suptitle(title, y=0.975, fontsize=17, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="center", color=MUTED, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def footer(fig: plt.Figure, text: str, y: float = 0.035) -> None:
    """Agrega nota metodológica legible bajo el diagrama.

    Args:
        fig: Figura en construcción.
        text: Texto del pie.
        y: Posición vertical de la nota.
    Returns:
        None.
    """
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color=MUTED, wrap=True)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str,
        *, fc: str = LIGHT, ec: str = NAVY, fontsize: float = 10,
        weight: str = "normal", radius: float = 0.02, textcolor: str = INK,
        z: int = 2) -> None:
    """Dibuja un bloque redondeado centrado.

    Args:
        ax: Eje objetivo.
        x, y, w, h: Geometría normalizada del bloque.
        text: Etiqueta multilínea.
        fc, ec: Relleno y borde.
        fontsize: Tamaño de etiqueta.
        weight: Peso tipográfico.
        radius: Redondeo visual.
        textcolor: Color de texto.
        z: Orden de dibujo.
    Returns:
        None.
    """
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                           linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=weight, wrap=True, zorder=z+1)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          *, color: str = MUTED, label: str | None = None, curve: float = 0.0,
          lw: float = 1.8, style: str = "-|>") -> None:
    """Dibuja una flecha con etiqueta opcional.

    Args:
        ax: Eje.
        start, end: Coordenadas normalizadas.
        color: Color de línea y flecha.
        label: Texto junto a la conexión.
        curve: Curvatura de conexión.
        lw: Ancho de trazo.
        style: Estilo de punta.
    Returns:
        None.
    """
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, linewidth=lw,
                            color=color, connectionstyle=f"arc3,rad={curve}", zorder=1)
    ax.add_patch(patch)
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+0.025, label,
                ha="center", va="center", fontsize=8, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})


def section_label(ax: plt.Axes, x: float, y: float, text: str, color: str = NAVY) -> None:
    """Añade etiqueta de sección.

    Args:
        ax: Eje.
        x, y: Posición.
        text: Nombre de etapa.
        color: Color del encabezado.
    Returns:
        None.
    """
    ax.text(x, y, text.upper(), color=color, fontsize=9, fontweight="bold", ha="left", va="center")


def draw_workflow() -> None:
    """Genera Figura 3.1: flujo del diseño experimental vigente."""
    fig, ax = page("Figura 3.1. Flujo del diseño y desarrollo experimental",
                   "De los datos cristalográficos a la evaluación agrupada por composición")
    labels = [
        ("1 · Entorno y fuentes", "Checkpoints originales\nMatGPT y MatterGPT\nCódigo, versiones y hashes", BLUE),
        ("2 · Cohorte ABS₃", "70 fórmulas\n210 estructuras estáticas\n3 polimorfos por fórmula", TEAL),
        ("3 · Preparación", "SLICES estrategia 4\nGeometría de celda\nFolds por composición", GOLD),
        ("4 · Modelos base", "Ridge M2 fijo\nExtractores Transformer\nTokenizadores originales", PURPLE),
        ("5 · Evaluación", "F0: congelado\nF1: último bloque\nF2: dos bloques finales", RED),
        ("6 · Resultados", "SLICES / SLICES + geometría\nPredicciones OOF\nEnergía, ΔE y ranking", GREEN),
    ]
    xs = [0.025, 0.19, 0.355, 0.52, 0.685, 0.85]
    for idx, ((head, body, color), x) in enumerate(zip(labels, xs)):
        box(ax, x, 0.38, 0.13, 0.32, f"{head}\n\n{body}", fc="white", ec=color, fontsize=9.2, weight="normal")
        if idx < len(xs)-1:
            arrow(ax, (x+0.135, 0.54), (xs[idx+1]-0.008, 0.54), color=color)
    box(ax, 0.20, 0.13, 0.23, 0.13, "Partición outer: 14 fórmulas\n(42 estructuras) por fold", fc="#F7FAFC", ec="#A6B6C2", fontsize=9)
    box(ax, 0.47, 0.13, 0.25, 0.13, "Validación interna: selección\nTrain/validación = 45/11 fórmulas", fc="#F7FAFC", ec="#A6B6C2", fontsize=9)
    arrow(ax, (0.56, 0.38), (0.56, 0.27), color=MUTED)
    arrow(ax, (0.74, 0.195), (0.85, 0.39), color=MUTED, curve=-0.2)
    footer(fig, "Alcance actual: fine-tuning supervisado ABS₃ F0/F1/F2 desde checkpoints originales; adaptación con Alexandria excluida. El test histórico permanece cerrado.")
    save(fig, "figura_3_01_flujo_diseno_experimental.png")


def read_example_structure(structure_id: str) -> dict[str, Any]:
    """Recupera una estructura de desarrollo por ID, sin consultar test.

    Args:
        structure_id: Identificador exacto.
    Returns:
        Registro JSONL con estructura cristalina estática.
    """
    if not STRUCTURES.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de estructuras de desarrollo: {STRUCTURES}")
    with STRUCTURES.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("structure_id") == structure_id:
                return record
    raise KeyError(f"ID no encontrado en las estructuras de desarrollo: {structure_id}")


def draw_structure_conversion(data: pd.DataFrame) -> None:
    """Genera Figura 3.2 con estructura estática real, grafo conceptual y SLICES."""
    row = data[data.structure_id.str.contains("Ba1Ge1S3_dist_perovskite", regex=False)].iloc[0]
    structure_id = str(row.structure_id)
    source = read_example_structure(structure_id)
    struct = source["structure"]
    lattice = np.asarray(struct["lattice"]["matrix"], dtype=float)
    sites = struct["sites"]
    frac = np.asarray([site["abc"] for site in sites], dtype=float)
    xyz = frac @ lattice
    species = [site["label"] for site in sites]
    fig = plt.figure(figsize=(15, 7.4))
    fig.suptitle("Figura 3.2. De la estructura cristalina ABS₃ a SLICES", y=.975,
                 fontsize=17, fontweight="bold", color=NAVY)
    fig.text(.5,.925,"Ejemplo real de BaGeS₃ tipo perovskita distorsionada; el grafo se ilustra de forma conceptual",ha="center",color=MUTED)
    ax3d=fig.add_subplot(1,3,1,projection="3d")
    frac_corners=np.asarray([[i,j,k] for i in (0,1) for j in (0,1) for k in (0,1)],float)
    cart_corners=frac_corners@lattice
    for i,a in enumerate(frac_corners):
        for d in range(3):
            b=a.copy();b[d]=1-b[d]
            if a[d]==0:
                q=cart_corners[np.where((frac_corners==b).all(axis=1))[0][0]]
                p=cart_corners[i]
                ax3d.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],color="#91A3AF",lw=1)
    for element in sorted(set(species)):
        choose=np.asarray(species)==element
        ax3d.scatter(*xyz[choose].T,s=55,depthshade=True,label=element,
                     color=ELEMENT_COLORS.get(element,"#777777"),edgecolor="white",linewidth=.4)
    ax3d.set_title(f"Estructura DFT estática\n{len(sites)} sitios · {row.reduced_formula}",pad=10)
    ax3d.set_xlabel("x (Å)");ax3d.set_ylabel("y (Å)");ax3d.set_zlabel("z (Å)")
    ax3d.legend(loc="upper left",fontsize=8)
    axg=fig.add_subplot(1,3,2);axg.axis("off");axg.set_xlim(0,1);axg.set_ylim(0,1)
    axg.set_title("Grafo periódico codificado",pad=10)
    nodes={"A":(.23,.73,NAVY),"B":(.70,.73,RED),"S₁":(.18,.36,GOLD),"S₂":(.48,.35,GOLD),"S₃":(.78,.36,GOLD)}
    for a,b,label in [("A","S₁","T₁"),("A","S₂","T₂"),("B","S₂","T₃"),("B","S₃","T₄"),("S₁","S₂","n₁"),("S₂","S₃","n₂")]:
        x1,y1,_=nodes[a];x2,y2,_=nodes[b]
        axg.plot([x1,x2],[y1,y2],color="#8D9AA4",lw=1.7,zorder=1)
        axg.text((x1+x2)/2,(y1+y2)/2+.035,label,fontsize=7,color=MUTED,ha="center")
    for name,(x,y,c) in nodes.items():
        circ=Circle((x,y),.075,facecolor=c,edgecolor="white",lw=1.5,zorder=2);axg.add_patch(circ)
        axg.text(x,y,name,ha="center",va="center",color="white",weight="bold",fontsize=10,zorder=3)
    axg.annotate("traslación periódica",xy=(.85,.54),xytext=(.52,.9),ha="center",fontsize=8,color=MUTED,
                 arrowprops={"arrowstyle":"->","color":MUTED,"connectionstyle":"arc3,rad=-.3"})
    axg.text(.5,.08,"Nodos = sitios/especies; aristas = relaciones\nperiódicas. Esquema, no geometría de enlace.",ha="center",fontsize=8,color=MUTED)
    axs=fig.add_subplot(1,3,3);axs.axis("off");axs.set_xlim(0,1);axs.set_ylim(0,1)
    axs.set_title("Secuencia SLICES",pad=10)
    tokens=str(row.SLICES).split()
    lines=[];current=""
    for tok in tokens[:36]:
        if len(current)+len(tok)+1>37: lines.append(current);current=tok
        else: current=(current+" "+tok).strip()
    if current: lines.append(current)
    shown="\n".join(lines)+("\n…" if len(tokens)>36 else "")
    box(axs,.04,.30,.92,.43,shown,fc="#F4F7F9",ec=BLUE,fontsize=8.2)
    axs.text(.5,.20,f"{len(tokens)} tokens SLICES",ha="center",fontsize=9,color=BLUE,weight="bold")
    axs.text(.5,.09,"Representación topológica serializada\npara el tokenizador del Transformer",ha="center",fontsize=8,color=MUTED)
    fig.text(.5,.025,f"ID de desarrollo: {structure_id}. Celda y posiciones tomadas del registro estático GGA; los nodos/aristas centrales son ilustrativos y no se usan como etiquetas energéticas.",ha="center",fontsize=8,color=MUTED)
    save(fig,"figura_3_02_estructura_a_slices.png")


def draw_splits() -> None:
    """Genera Figura 3.3: cohortes agrupadas por composición."""
    fig,ax=page("Figura 3.3. Particiones ABS₃ agrupadas por composición",
                "Validación cruzada externa sobre desarrollo; prueba histórica aislada")
    box(ax,.04,.58,.20,.22,"DESARROLLO ABS₃\n70 fórmulas · 210 estructuras\n3 polimorfos por fórmula",fc="#F2F7FA",ec=BLUE,fontsize=11,weight="bold")
    arrow(ax,(.25,.69),(.34,.69),color=BLUE)
    box(ax,.35,.58,.22,.22,"5 pliegues externos\nEn cada fold: 56 fórmulas\npara desarrollo interno",fc="#F1F7F4",ec=TEAL,fontsize=10)
    arrow(ax,(.58,.69),(.66,.69),color=TEAL)
    box(ax,.67,.70,.27,.17,"ENTRENAMIENTO INTERNO\n45 fórmulas · 135 estructuras",fc="#EDF5EE",ec=GREEN,fontsize=10)
    box(ax,.67,.47,.27,.17,"VALIDACIÓN INTERNA\n11 fórmulas · 33 estructuras",fc="#FFF7E7",ec=GOLD,fontsize=10)
    arrow(ax,(.78,.69),(.78,.65),color=GREEN)
    arrow(ax,(.67,.80),(.58,.70),color=GREEN,curve=.18)
    arrow(ax,(.67,.55),(.58,.66),color=GOLD,curve=-.18)
    box(ax,.35,.27,.22,.15,"EVALUACIÓN OUTER\n14 fórmulas · 42 estructuras\nCada fórmula queda fuera una vez",fc="#FCEFED",ec=RED,fontsize=9.5)
    arrow(ax,(.46,.57),(.46,.43),color=RED)
    box(ax,.68,.18,.26,.16,"PRUEBA HISTÓRICA\nSeparada y cerrada\nNo consultada en este estudio",fc="#F1EFF5",ec=PURPLE,fontsize=10)
    ax.text(.04,.33,"Semillas de entrenamiento: 42, 123 y 2026\nLos tres polimorfos viajan juntos con su fórmula.",fontsize=10,color=INK,va="top")
    footer(fig,"Las particiones se cuentan por fórmulas y estructuras. La prueba histórica no forma parte de los 70 materiales de desarrollo y no se muestra su tamaño.")
    save(fig,"figura_3_03_particiones_abs3.png")


def draw_tokenization(data: pd.DataFrame) -> None:
    """Genera Figura 3.4: secuencias y vocabularios propios."""
    fig,ax=page("Figura 3.4. Preparación de SLICES para cada tokenizador",
                "La misma estructura se codifica con vocabularios y convenciones de entrada propias")
    sample=str(data.iloc[0].SLICES).split()
    seq=" ".join(sample[:13])+" …"
    box(ax,.04,.66,.23,.19,"SLICES de una estructura\n"+seq,fc="#F6F8FA",ec=BLUE,fontsize=9)
    arrow(ax,(.28,.75),(.38,.75),color=BLUE)
    box(ax,.39,.62,.25,.27,"MatGPT\n\n<start> · tokens SLICES · <end>\n\nSeparación por espacios\n→ ID del vocabulario MatGPT\n→ padding de lote",fc="#EDF5FA",ec=BLUE,fontsize=9.5)
    box(ax,.69,.62,.27,.27,"MatterGPT\n\n> · tokens SLICES · <\n\nSeparación por espacios\n→ ID del vocabulario MatterGPT\n→ rama prop_nn([1.0, 2.0])",fc="#EDF7F5",ec=TEAL,fontsize=9.5)
    arrow(ax,(.65,.75),(.68,.75),color=PURPLE)
    box(ax,.17,.25,.30,.17,"Secuencia de enteros\nTokens especiales y máscara\n(longitud válida por estructura)",fc="white",ec="#8394A0",fontsize=9)
    box(ax,.56,.25,.30,.17,"Entrada al Transformer\nCondicionamiento MatterGPT\nindependiente de la etiqueta DFT",fc="white",ec="#8394A0",fontsize=9)
    arrow(ax,(.51,.61),(.37,.43),color=BLUE,curve=.1)
    arrow(ax,(.80,.61),(.69,.43),color=TEAL,curve=-.1)
    footer(fig,"No se cambia la secuencia física ni el vocabulario original. MatterGPT recibe la condición fija [1.0, 2.0]; nunca se introducen energías objetivo como condición.")
    save(fig,"figura_3_04_tokenizacion_modelos.png")


def draw_internal_aggregation() -> None:
    """Genera Figura 3.5: estados contextuales por bloque y opciones de pooling."""
    fig,ax=page("Figura 3.5. Extracción y agregación de representaciones internas",
                "Cada bloque contextualiza la secuencia; el token de cierre y el promedio enmascarado son agregaciones distintas")
    x0=.05; gap=.045; bw=.09
    ax.text(.04,.83,"Secuencia tokenizada",fontsize=9,color=MUTED)
    toks=["<start>","S", "Ge", "Ba", "0", "13", "ooo", "<end>"]
    for i,t in enumerate(toks):
        box(ax,x0+i*.113,.72,.10,.09,t,fc="#F5F7F8",ec="#A4B2BB",fontsize=7.5,radius=.01)
    blocks=[("Bloque 1",BLUE),("Bloque 2",BLUE),("⋯",MUTED),("Bloque L−1",TEAL),("Bloque L",TEAL),("ln_f",PURPLE)]
    xs=[.05,.21,.37,.53,.69,.85]
    for (label,color),x in zip(blocks,xs):
        box(ax,x,.46,.12,.14,label,fc="white",ec=color,fontsize=9.5,weight="bold")
        if x<.85: arrow(ax,(x+.125,.53),(x+.155,.53),color=color,lw=1.4)
    ax.text(.05,.36,"Estados por token: H⁽ˡ⁾ ∈ ℝ⁽T×768⁾",fontsize=10,color=INK)
    box(ax,.08,.12,.36,.15,"FINAL_END\nvector de 768 del último token estructural válido",fc="#EEF5FA",ec=BLUE,fontsize=9.5)
    box(ax,.56,.12,.36,.15,"Promedio enmascarado\nmedia de tokens válidos; excluye inicio, cierre y padding",fc="#F3F0F8",ec=PURPLE,fontsize=9.5)
    arrow(ax,(.91,.45),(.27,.28),color=BLUE,curve=.18)
    arrow(ax,(.78,.45),(.74,.28),color=PURPLE,curve=-.08)
    footer(fig,"La extracción principal del regresor histórico usa FINAL_END posterior a ln_f. Poolings de bloques intermedios y masked mean se tratan como representaciones alternativas de auditoría, no se mezclan silenciosamente.")
    save(fig,"figura_3_05_representaciones_internas.png")


def draw_frozen_extractors() -> None:
    """Genera Figura 3.6: los dos extractores congelados originales."""
    fig,ax=page("Figura 3.6. Extracción congelada en MatGPT y MatterGPT",
                "Puntos de lectura contextual posteriores a todos los bloques y a ln_f")
    box(ax,.04,.66,.19,.17,"SLICES\n<start> … <end>",fc="#F5F8FA",ec=BLUE,fontsize=11)
    box(ax,.29,.58,.37,.31,"MatGPT original · 12 bloques\n\nEmbeddings y Transformer congelados\n\nln_f → estado en <end>\nFINAL_END ∈ ℝ⁷⁶⁸",fc="#EDF5FA",ec=BLUE,fontsize=11)
    arrow(ax,(.24,.75),(.28,.75),color=BLUE)
    box(ax,.70,.66,.24,.17,"Embedding MatGPT\n768 características",fc="white",ec=BLUE,fontsize=10)
    arrow(ax,(.67,.73),(.69,.73),color=BLUE)
    box(ax,.04,.27,.19,.17,"SLICES\n> … <",fc="#F5F8FA",ec=TEAL,fontsize=11)
    box(ax,.29,.19,.37,.31,"MatterGPT original · 12 bloques\n\nprop_nn([1.0, 2.0]) como prefijo\n\nln_f → último token estructural\nFINAL_END ∈ ℝ⁷⁶⁸",fc="#EDF7F5",ec=TEAL,fontsize=10.5)
    arrow(ax,(.24,.36),(.28,.36),color=TEAL)
    box(ax,.70,.27,.24,.17,"Embedding MatterGPT\n768 características",fc="white",ec=TEAL,fontsize=10)
    arrow(ax,(.67,.34),(.69,.34),color=TEAL)
    box(ax,.40,.83,.18,.07,"Parámetros congelados",fc="#F7F7F7",ec="#98A5AD",fontsize=8.5)
    footer(fig,"Extracción evaluación: modo eval, sin actualización de gradientes. MatGPT usa tokens de frontera <start>/<end>; MatterGPT usa [1.0, 2.0] y selecciona la posición estructural terminal tras ln_f.")
    save(fig,"figura_3_06_extractores_congelados.png")


def draw_regression_components() -> None:
    """Genera Figura 3.7: Ridge composicional y cabezas MLP8/MLP32."""
    fig,ax=page("Figura 3.7. Componentes supervisados de regresión energética",
                "Ridge composicional y cabezas estructurales de capacidad distinta")
    box(ax,.04,.59,.20,.21,"Composición química\n17 fracciones elementales",fc="#F5F8FA",ec=GOLD,fontsize=10)
    box(ax,.31,.59,.21,.21,"Ridge M2\nStandardScaler + Ridge\nuna media por fórmula",fc="#FFF8E9",ec=GOLD,fontsize=10)
    arrow(ax,(.25,.69),(.30,.69),color=GOLD)
    box(ax,.60,.67,.18,.17,"μ_c\nbase composicional",fc="white",ec=GOLD,fontsize=10)
    arrow(ax,(.53,.69),(.59,.75),color=GOLD)
    box(ax,.04,.24,.20,.21,"Embedding estructural\nFINAL_END (768)\no fusión (775)",fc="#F4F8FA",ec=BLUE,fontsize=10)
    box(ax,.31,.35,.21,.17,"MLP8\n768/775 → 8 → 1\nGELU · dropout 0.1",fc="#EDF5FA",ec=BLUE,fontsize=9.5)
    box(ax,.31,.09,.21,.17,"MLP32\n768/775 → 32 → 1\nGELU · dropout 0.1",fc="#F2F0F8",ec=PURPLE,fontsize=9.5)
    arrow(ax,(.25,.35),(.30,.43),color=BLUE)
    arrow(ax,(.25,.32),(.30,.18),color=PURPLE)
    box(ax,.60,.34,.18,.17,"δ_c,i\nresiduo polimórfico",fc="white",ec=BLUE,fontsize=10)
    arrow(ax,(.53,.43),(.59,.43),color=BLUE)
    box(ax,.82,.45,.15,.18,"E_pred\n= μ_c + δ_c,i",fc="#EDF5EE",ec=GREEN,fontsize=10,weight="bold")
    arrow(ax,(.79,.74),(.85,.64),color=GOLD)
    arrow(ax,(.79,.42),(.82,.54),color=BLUE)
    ax.text(.41,.02,"La matriz principal vigente usa MLP8; MLP32 se incluye como variante histórica de capacidad.",ha="center",fontsize=8.5,color=MUTED)
    footer(fig,"Ridge no recibe embeddings ni geometría. La cabeza aprende el residuo estructural; la salida energética se expresa en eV/átomo.")
    save(fig,"figura_3_07_regresores_ridge_mlp.png")


def draw_energy_decomposition() -> None:
    """Genera Figura 3.8: media composicional y residuos centrados."""
    fig=plt.figure(figsize=(13.5,7.2));fig.suptitle("Figura 3.8. Descomposición de energía entre composición y polimorfo",y=.97,fontsize=17,fontweight="bold",color=NAVY)
    ax=fig.add_subplot(1,1,1);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    xloc=np.array([.24,.50,.76]); delta=np.array([-.16,.04,.12]); base=.46
    ax.axhline(base,color=GOLD,lw=2,ls="--")
    ax.text(.89,base,"μ_c · Ridge M2\nreferencia común",va="center",color="#9B741A",fontsize=10)
    for x,d,name,color in zip(xloc,delta,["Perovskita","Hexagonal","Tipo aguja"],[BLUE,TEAL,RED]):
        ax.annotate("",xy=(x,base+d),xytext=(x,base),arrowprops={"arrowstyle":"<->","color":color,"lw":2})
        ax.scatter([x],[base+d],s=180,color=color,zorder=3,edgecolor="white")
        ax.text(x,base+d+(0.07 if d>=0 else -0.09),f"δ{['₁','₂','₃'][list(xloc).index(x)]} = {d:+.2f}",ha="center",color=color,fontsize=10)
        ax.text(x,.14,name,ha="center",fontsize=10,color=INK)
        ax.text(x,.08,f"E_pred = μ_c + δ",ha="center",fontsize=8.5,color=MUTED)
    ax.text(.5,.84,"Para una fórmula c, la misma media composicional se comparte entre sus tres polimorfos",ha="center",fontsize=12,color=NAVY)
    ax.text(.5,.72,"Los scores brutos se centran por triplete:  δ_c,i = r_c,i − (1/3) Σⱼ r_c,j",ha="center",fontsize=12)
    ax.text(.5,.62,"Σᵢ δ_c,i = 0  ⇒  la posición absoluta depende de μ_c y el orden intrafórmula depende de δ_c,i",ha="center",fontsize=10,color=MUTED)
    ax.set_ylim(0,1)
    footer(fig,"Esquema didáctico (valores δ ilustrativos, no resultados observados). Las diferencias entre polimorfos cancelan μ_c: ΔEᵢⱼ = δ_c,j − δ_c,i.")
    save(fig,"figura_3_08_descomposicion_energetica.png")


def draw_late_fusion() -> None:
    """Genera Figura 3.9: MLP8 residual con fusión geométrica tardía."""
    fig,ax=page("Figura 3.9. MLP8 residual y fusión geométrica tardía",
                "La geometría se incorpora a la cabeza después del Transformer")
    box(ax,.04,.61,.18,.18,"SLICES\nsecuencia original",fc="#F5F8FA",ec=BLUE,fontsize=10)
    box(ax,.30,.61,.25,.18,"Transformer congelado\nFINAL_END (768)",fc="#EDF5FA",ec=BLUE,fontsize=10)
    arrow(ax,(.23,.70),(.29,.70),color=BLUE)
    box(ax,.04,.25,.18,.19,"Celda DFT estática\n7 descriptores",fc="#FFF8E9",ec=GOLD,fontsize=9.5)
    box(ax,.30,.25,.25,.19,"StandardScaler(7)\nfit solo inner-train",fc="#FFF8E9",ec=GOLD,fontsize=9.5)
    arrow(ax,(.23,.34),(.29,.34),color=GOLD)
    box(ax,.62,.43,.15,.22,"Concatenación\n768 + 7 = 775",fc="#F3F0F8",ec=PURPLE,fontsize=10)
    arrow(ax,(.56,.70),(.65,.66),color=BLUE)
    arrow(ax,(.56,.34),(.65,.45),color=GOLD)
    box(ax,.81,.43,.15,.22,"MLP8\n775 → 8 → 1\nδ_c,i",fc="#EDF5FA",ec=BLUE,fontsize=10)
    arrow(ax,(.78,.54),(.80,.54),color=PURPLE)
    box(ax,.68,.14,.25,.13,"μ_c + δ_c,i → E_pred\nΔE intrafórmula",fc="#EDF5EE",ec=GREEN,fontsize=9.5)
    arrow(ax,(.885,.42),(.82,.28),color=GREEN)
    ax.text(.51,.88,"NO entra en los tokens ni en la atención",ha="center",fontsize=10,color=RED,weight="bold")
    footer(fig,"Descriptores: a/b/c reducidos, α/β/γ reducidos y volumen por átomo. Geometría de celdas finales GGA relajadas; se estandariza sin usar validación ni outer.")
    save(fig,"figura_3_09_fusion_geometrica_tardia.png")


def draw_zero_step() -> None:
    """Genera Figura 3.10: inicialización equivalente de las cabezas."""
    fig,ax=page("Figura 3.10. Inicialización de paso cero con y sin geometría",
                "La cabeza híbrida empieza con la misma función que la cabeza SLICES-only")
    box(ax,.04,.67,.24,.18,"C0 · MLP8 SLICES-only\nW₀ ∈ ℝ⁸ˣ⁷⁶⁸\nb ∈ ℝ⁸",fc="#EDF5FA",ec=BLUE,fontsize=10)
    box(ax,.04,.24,.24,.18,"C1 · MLP8 con geometría\nW₁ ∈ ℝ⁸ˣ⁷⁷⁵\nb ∈ ℝ⁸",fc="#F3F0F8",ec=PURPLE,fontsize=10)
    box(ax,.38,.68,.25,.17,"Columnas estructurales\nW₁[:, 0:768] = W₀\nbias copiado",fc="#EAF4EF",ec=GREEN,fontsize=10)
    box(ax,.38,.24,.25,.17,"Columnas geométricas\nW₁[:, 768:775] = 0\n7 entradas",fc="#FFF6E5",ec=GOLD,fontsize=10)
    arrow(ax,(.29,.75),(.37,.77),color=BLUE)
    arrow(ax,(.29,.33),(.37,.33),color=PURPLE)
    box(ax,.72,.46,.23,.20,"Predicción inicial\nC0(x) = C1([x, 0])\n\n|C0 − C1| ≤ 10⁻⁷",fc="#F5F7F8",ec=NAVY,fontsize=10,weight="bold")
    arrow(ax,(.64,.77),(.73,.63),color=GREEN)
    arrow(ax,(.64,.33),(.73,.48),color=GOLD)
    footer(fig,"Control ejecutado para cada pareja fold/semilla en el protocolo de fusión tardía; la geometría puede aprender peso después de iniciar la optimización.")
    save(fig,"figura_3_10_inicializacion_cero_geometria.png")


def draw_matgpt_layers() -> None:
    """Genera Figura 3.11: extracción capa a capa de MatGPT."""
    fig,ax=page("Figura 3.11. Representaciones intermedias de MatGPT",
                "Cada capa produce un descriptor independiente para auditoría o regresión separada")
    x0=.035; widths=.1
    for i in range(12):
        x=x0+i*.078
        col=BLUE if i in [2,5,8,11] else "#DCE7ED"
        edge=BLUE if i in [2,5,8,11] else "#A8B5BE"
        box(ax,x,.65,widths,.14,f"B{i+1}",fc=col if i in [2,5,8,11] else "white",ec=edge,fontsize=8,weight="bold",textcolor="white" if i in [2,5,8,11] else INK,radius=.01)
        if i<11: arrow(ax,(x+widths,.72),(x+.075,.72),color="#92A2AC",lw=1)
    ax.text(.5,.83,"SLICES → bloques Transformer MatGPT (12) → ln_f",ha="center",fontsize=11,color=NAVY)
    taps=[(.241,.15,"B3 / L03\npre-ln_f · 768"),
          (.475,.38,"B6 / L06\npre-ln_f · 768"),
          (.709,.61,"B9 / L09\npre-ln_f · 768"),
          (.943,.84,"B12 → ln_f\nFINAL_END · 768")]
    for source_x,target_x,label in taps:
        arrow(ax,(source_x,.64),(target_x,.49),color=BLUE)
        box(ax,target_x-.095,.34,.19,.12,label,fc="#EDF5FA",ec=BLUE,fontsize=7.8)
        arrow(ax,(target_x,.33),(target_x,.24),color=BLUE)
        box(ax,target_x-.085,.10,.17,.12,"Regresor\nindependiente",fc="white",ec=PURPLE,fontsize=8.2)
    footer(fig,"Los regresores por capa son análisis separados. No se concatenan ni se promedian en el modelo principal FINAL_END. Las salidas intermedias previas a ln_f se identifican como tales.")
    save(fig,"figura_3_11_capas_intermedias_matgpt.png")


def draw_matgpt_regressor_integration() -> None:
    """Genera Figura 3.12: integración de ramas composicional y estructural."""
    fig,ax=page("Figura 3.12. Integración de los regresores MatGPT",
                "Ridge predice la base química; MatGPT + MLP8 modela la variación estructural intrafórmula")
    box(ax,.05,.67,.20,.17,"Fórmula ABS₃\n17 fracciones químicas",fc="#F8F4E8",ec=GOLD,fontsize=10)
    box(ax,.34,.67,.20,.17,"Ridge M2 fijo\nμ_c",fc="#FFF8E9",ec=GOLD,fontsize=10)
    arrow(ax,(.26,.755),(.33,.755),color=GOLD)
    box(ax,.05,.27,.20,.17,"SLICES de cada\npolimorfo",fc="#F2F7FA",ec=BLUE,fontsize=10)
    box(ax,.34,.27,.20,.17,"MatGPT FINAL_END\n768",fc="#EDF5FA",ec=BLUE,fontsize=10)
    box(ax,.62,.27,.18,.17,"MLP8 residual\nδ_c,i",fc="#F5F1F8",ec=PURPLE,fontsize=10)
    arrow(ax,(.26,.355),(.33,.355),color=BLUE);arrow(ax,(.55,.355),(.61,.355),color=BLUE)
    box(ax,.83,.47,.14,.20,"E_pred,c,i\n= μ_c + δ_c,i",fc="#EDF5EE",ec=GREEN,fontsize=10,weight="bold")
    arrow(ax,(.55,.755),(.85,.66),color=GOLD);arrow(ax,(.81,.355),(.86,.46),color=PURPLE)
    ax.text(.71,.10,"Tres polimorfos comparten μ_c; el residuo se centra dentro de su triplete.",ha="center",fontsize=9,color=MUTED)
    footer(fig,"El módulo de composición y la rama estructural se suman en eV/átomo. Las diferencias ΔE entre polimorfos no dependen de la media común μ_c.")
    save(fig,"figura_3_12_integracion_regresores_matgpt.png")


def draw_matter_hooks() -> None:
    """Genera Figura 3.13: hooks y máscara de posiciones estructurales."""
    fig,ax=page("Figura 3.13. Captura de representaciones internas de MatterGPT",
                "El prefijo de condición participa en el contexto; el vector final se toma de una posición SLICES")
    tokens=[("prop₁",PURPLE),("prop₂",PURPLE),(">",TEAL),("S",BLUE),("Ge",BLUE),("Ba",BLUE),("…",BLUE),("<",RED)]
    x=.055
    for token,color in tokens:
        box(ax,x,.73,.095,.12,token,fc=color,ec=color,fontsize=9,textcolor="white",radius=.01)
        x+=.115
    ax.text(.11,.65,"prefijo de condición",ha="center",fontsize=8,color=PURPLE)
    ax.text(.54,.65,"tokens SLICES / posiciones estructurales",ha="center",fontsize=8,color=BLUE)
    # Hook capture ladder.
    for i,y in enumerate([.50,.39,.28]):
        box(ax,.10,y,.20,.075,f"Hook tras bloque {i+1 if i<2 else 'L'}",fc="#F4F7F9",ec=TEAL,fontsize=8.5)
        ax.text(.36,y+.036,"H⁽ˡ⁾ [lote, posiciones, 768]",va="center",fontsize=9,color=INK)
        arrow(ax,(.31,y+.036),(.35,y+.036),color=TEAL,lw=1.2)
    box(ax,.64,.38,.28,.18,"Máscara de posiciones\ncondition ≠ structure\nselección terminal `<`",fc="#EDF7F5",ec=TEAL,fontsize=9.5)
    arrow(ax,(.71,.50),(.74,.46),color=TEAL)
    box(ax,.64,.13,.28,.15,"ln_f → FINAL_END\n768 características\nbackbone congelado en F0",fc="white",ec=PURPLE,fontsize=9.5)
    arrow(ax,(.78,.37),(.78,.29),color=PURPLE)
    footer(fig,"Los hooks registran estados de capas para inspección; la extracción principal usa la salida posterior a ln_f en el último token estructural válido. El prefijo condicionado no se confunde con el pooling SLICES.")
    save(fig,"figura_3_13_hooks_mattergpt.png")


def draw_geometry_comparison() -> None:
    """Genera Figura 3.14: comparación de modalidades estructurales."""
    fig,ax=page("Figura 3.14. Integración de SLICES y geometría de celda",
                "Comparación de la rama SLICES-only con la fusión geométrica tardía")
    box(ax,.04,.61,.18,.17,"SLICES\nsecuencia",fc="#F2F7FA",ec=BLUE,fontsize=10)
    box(ax,.29,.61,.22,.17,"MatGPT o MatterGPT\nFINAL_END (768)",fc="#EDF5FA",ec=BLUE,fontsize=10)
    arrow(ax,(.23,.695),(.28,.695),color=BLUE)
    box(ax,.62,.67,.19,.16,"SLICES-only\nMLP8 (768 → 8 → 1)",fc="white",ec=BLUE,fontsize=9.5)
    arrow(ax,(.52,.695),(.61,.75),color=BLUE)
    box(ax,.04,.25,.18,.17,"Celda DFT\n7 descriptores",fc="#FFF7E7",ec=GOLD,fontsize=9.5)
    box(ax,.29,.25,.22,.17,"StandardScaler\nfit en inner-train",fc="#FFF7E7",ec=GOLD,fontsize=9.5)
    arrow(ax,(.23,.335),(.28,.335),color=GOLD)
    box(ax,.62,.31,.19,.16,"SLICES + geometría\nconcatena 775\nMLP8 (775 → 8 → 1)",fc="#F3F0F8",ec=PURPLE,fontsize=9)
    arrow(ax,(.52,.695),(.62,.43),color=BLUE,curve=-.1)
    arrow(ax,(.52,.335),(.61,.36),color=GOLD)
    box(ax,.84,.46,.13,.19,"Residuo\n+ Ridge μ_c\n→ energía",fc="#EDF5EE",ec=GREEN,fontsize=9.5)
    arrow(ax,(.82,.75),(.86,.65),color=BLUE);arrow(ax,(.82,.39),(.86,.46),color=PURPLE)
    footer(fig,"Ambas variantes conservan la secuencia original y el mismo Ridge M2. La geometría entra después del Transformer y procede de celdas estáticas relajadas por DFT.")
    save(fig,"figura_3_14_slices_geometria.png")


def draw_current_finetuning() -> None:
    """Genera Figura 3.15: protocolo supervisado vigente sin Alexandria."""
    fig,ax=page("Figura 3.15. Fine-tuning supervisado ABS₃ desde checkpoints originales",
                "Etapas emparejadas F0/F1/F2 para MatGPT y MatterGPT; la adaptación con Alexandria está fuera del protocolo")
    box(ax,.03,.65,.20,.18,"Checkpoint original\nMatGPT o MatterGPT\n12 bloques",fc="#F3F7FA",ec=BLUE,fontsize=10)
    box(ax,.31,.65,.21,.18,"Warmup común\nMLP8 entrenable\nbackbone congelado",fc="#F7F4EA",ec=GOLD,fontsize=10)
    arrow(ax,(.24,.74),(.30,.74),color=BLUE)
    for label,x,color in [("F0 · 0 bloques\nsolo MLP8",.60,BLUE),("F1 · bloque 12\n+ MLP8",.60,TEAL),("F2 · bloques 11–12\n+ MLP8",.60,PURPLE)]:
        y={"F0 · 0 bloques\nsolo MLP8":.73,"F1 · bloque 12\n+ MLP8":.47,"F2 · bloques 11–12\n+ MLP8":.21}[label]
        box(ax,x,y,.18,.14,label,fc="white",ec=color,fontsize=9.2)
        arrow(ax,(.53,.74),(x-.01,y+.07),color=color,curve=(.11 if y<.7 else -.1))
    box(ax,.83,.47,.15,.21,"Dos entradas\nSLICES-only\nSLICES + geometría\n\nOOF por fórmula",fc="#EDF5EE",ec=GREEN,fontsize=9.2)
    for y,color in [(.80,BLUE),(.54,TEAL),(.28,PURPLE)]:arrow(ax,(.79,y),(.82,.60),color=color,curve=.06)
    ax.text(.40,.37,"Misma cabeza inicial\nRidge M2 fijo\nmisma pérdida y folds",ha="center",va="center",fontsize=9,color=MUTED)
    footer(fig,"La comparación actual parte de checkpoints generativos originales. F0 mantiene congelado el Transformer; F1/F2 actualizan uno/dos bloques finales. No se usa Alexandria ni test independiente.")
    save(fig,"figura_3_15_finetuning_actual_abs3.png")


def draw_nested_cv() -> None:
    """Genera Figura 3.16: CV anidada agrupada por composición."""
    fig,ax=page("Figura 3.16. Validación cruzada agrupada por fórmula química",
                "Cinco pliegues externos; validación interna separada en el conjunto de desarrollo")
    colors=[BLUE,TEAL,GOLD,RED,PURPLE]
    x0=.06; y0=.68; w=.17; gap=.018
    for k,color in enumerate(colors):
        x=x0+k*(w+gap)
        ax.add_patch(FancyBboxPatch((x,y0),w,.13,boxstyle="round,pad=.005,rounding_size=.01",facecolor="white",edgecolor=color,lw=1.5))
        ax.text(x+w/2,y0+.065,f"Fold externo {k+1}",ha="center",va="center",fontsize=9,color=color,weight="bold")
        # one row each: one held-out outer block, four remaining development blocks
        block_w=w/5
        for j in range(5):
            fc=color if j==k else "#E7EDF1"
            ax.add_patch(Rectangle((x+j*block_w,y0-.12),block_w-.003,.065,facecolor=fc,edgecolor="white"))
        ax.text(x+w/2,y0-.17,"14 fórmulas outer" if True else "",ha="center",fontsize=7.8,color=MUTED)
    box(ax,.10,.28,.27,.18,"56 fórmulas disponibles\npara entrenamiento/selección\n168 estructuras",fc="#F2F6F8",ec="#8A9AA5",fontsize=9.5)
    box(ax,.45,.32,.22,.14,"Inner-train\n45 fórmulas · 135 estructuras",fc="#EDF5EE",ec=GREEN,fontsize=9)
    box(ax,.45,.13,.22,.14,"Inner-validation\n11 fórmulas · 33 estructuras",fc="#FFF7E7",ec=GOLD,fontsize=9)
    arrow(ax,(.39,.37),(.44,.40),color=GREEN);arrow(ax,(.39,.34),(.44,.20),color=GOLD)
    box(ax,.76,.28,.20,.18,"Outer-evaluation\n14 fórmulas\n42 estructuras",fc="#FCEFED",ec=RED,fontsize=10)
    arrow(ax,(.69,.40),(.75,.39),color=RED)
    ax.text(.5,.05,"Cada composición aparece en outer-evaluation exactamente una vez; sus tres polimorfos permanecen agrupados.",ha="center",fontsize=9,color=MUTED)
    footer(fig,"Cohorte: 70 fórmulas y 210 estructuras estáticas; 5 folds. Semillas 42, 123 y 2026 repiten el entrenamiento dentro de cada asignación. La prueba histórica se mantiene fuera de este esquema.")
    save(fig,"figura_3_16_cv_anidada_abs3.png")


def draw_dataset_distributions(data: pd.DataFrame) -> dict[str, Any]:
    """Genera figura adicional con incidencia química y longitud real de SLICES.

    Args:
        data: Dataset de desarrollo validado.
    Returns:
        Resumen numérico empleado en las barras/histograma.
    """
    formulas=data.drop_duplicates("composition_id").composition_id.astype(str).tolist()
    elements=sorted({symbol for formula in formulas for symbol in re.findall(r"[A-Z][a-z]?",formula)})
    counts={element:sum(element in re.findall(r"[A-Z][a-z]?",formula) for formula in formulas) for element in elements}
    lengths=data.SLICES.astype(str).map(lambda sequence:len(sequence.split())).to_numpy(int)
    fig,axes=plt.subplots(1,2,figsize=(14,6.4))
    fig.suptitle("Figura 3.17. Distribución de especies químicas y longitudes SLICES",y=.98,fontsize=16,fontweight="bold",color=NAVY)
    ax=axes[0]; names=list(counts); vals=[counts[e] for e in names]
    ax.bar(names,vals,color=[ELEMENT_COLORS.get(e,BLUE) for e in names],edgecolor="white")
    ax.set_title("(a) Fórmulas de desarrollo que contienen cada elemento")
    ax.set_ylabel("Número de fórmulas (de 70)");ax.set_xlabel("Especie química")
    ax.set_ylim(0,max(vals)*1.18);ax.grid(axis="y",alpha=.2)
    for x,v in enumerate(vals):ax.text(x,v+max(vals)*.02,str(v),ha="center",fontsize=8)
    ax=axes[1]
    bins=np.arange(lengths.min()-.5,lengths.max()+1.5,1)
    ax.hist(lengths,bins=bins,color=BLUE,alpha=.83,edgecolor="white")
    med=float(np.median(lengths));mean=float(np.mean(lengths))
    ax.axvline(med,color=RED,ls="--",lw=1.8,label=f"Mediana = {med:.0f} tokens")
    ax.axvline(mean,color=GOLD,ls=":",lw=2,label=f"Media = {mean:.1f} tokens")
    ax.set_title("(b) Longitud de las secuencias SLICES")
    ax.set_xlabel("Número de tokens separados por espacios");ax.set_ylabel("Estructuras (de 210)")
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
    fig.text(.5,.02,"Desarrollo ABS₃: 70 fórmulas, 210 estructuras, 3 polimorfos por fórmula. (a) Incidencia por fórmula, no conteo de átomos; (b) longitud léxica de la columna SLICES histórica. Sin registros de prueba.",ha="center",fontsize=8,color=MUTED)
    save(fig,"figura_3_17_distribuciones_abs3.png")
    return {"formula_count_by_element":counts,"sequence_length_tokens":{"n":int(len(lengths),),"min":int(lengths.min()),"median":med,"mean":mean,"max":int(lengths.max())}}


def main() -> None:
    """Valida las fuentes de desarrollo y genera las diecisiete figuras PNG."""
    started=time.monotonic();cpu_start=time.process_time()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    data=read_development(DATA)
    draw_workflow();draw_structure_conversion(data);draw_splits();draw_tokenization(data)
    draw_internal_aggregation();draw_frozen_extractors();draw_regression_components()
    draw_energy_decomposition();draw_late_fusion();draw_zero_step();draw_matgpt_layers()
    draw_matgpt_regressor_integration();draw_matter_hooks();draw_geometry_comparison()
    draw_current_finetuning();draw_nested_cv();distributions=draw_dataset_distributions(data)
    elapsed=time.monotonic()-started
    mem=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024))
    resource_data={"experiment":"thesis_plots_generation","status":"COMPLETE",
        "figures_png":17,"output_directory":"images/thesis_plots",
        "source_dataset":"data/abs3/abs3_development_f4_geometry/dataset.csv",
        "source_dataset_sha256":sha256_file(DATA),
        "example_structure_manifest":"experiments/02_complementarios/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl",
        "example_structure_manifest_sha256":sha256_file(STRUCTURES),
        "source_rows_read":int(len(data)),"formulas":int(data.composition_id.nunique()),
        "structures":int(data.structure_id.nunique()),"historical_test_accessed":False,
        "alexandria_adaptation_included":False,
        "current_ft_protocol":"original checkpoints; supervised F0/F1/F2; SLICES and late-fusion geometry",
        "distributions":distributions,"wall_seconds":elapsed,
        "process_cpu_seconds":time.process_time()-cpu_start,"peak_rss_bytes":mem,
        "cpu_count_logical":os.cpu_count(),"platform":platform.platform(),
        "python":platform.python_version(),"matplotlib":matplotlib.__version__,
        "numpy":np.__version__,"pandas":pd.__version__,"export":"PNG only at 300 dpi"}
    RESOURCE_OUT.parent.mkdir(parents=True,exist_ok=True)
    RESOURCE_OUT.write_text(json.dumps(resource_data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    LOGGER.info("Terminado: 17 PNG; tiempo %.2f s; pico RSS %d bytes",elapsed,mem)


if __name__ == "__main__":
    main()
