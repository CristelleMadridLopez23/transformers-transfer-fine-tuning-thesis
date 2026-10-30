"""Genera figuras de comparación e invariancia para el Ridge composicional.

Las poblaciones son predicciones externas históricas de desarrollo; semillas se
promedian por estructura antes de dibujar puntos físicos únicos.
"""
from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as cm
from analyze_results import MODEL_NAMES, ols_stats

LOG = logging.getLogger("matgpt_hybrid_figures")
COLORS = {"M1": "#315a7d", "M2": "#e07a35", "M3": "#4b8768", "M4": "#a24a56"}
FAMILIES_ES = {"dist_perovskite": "Perovskita distorsionada", "hexagonal": "Hexagonal", "needle": "Tipo aguja"}


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    """Guarda PNG y SVG con márgenes revisables.

    Args:
        fig: Figura Matplotlib terminada.
        directory: Directorio `figures`.
        stem: Nombre descriptivo sin extensión.
    Returns:
        None.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def averaged_predictions(output: Path) -> pd.DataFrame:
    """Promedia semillas por estructura tras comprobar targets y pliegue constantes.

    Args:
        output: Carpeta del experimento.
    Returns:
        Una fila por modelo e ID físico.
    """
    frame = pd.read_csv(output / "oof_predictions_by_structure.csv")
    keys = ["model_id", "structure_id"]
    invariant = ["composition_id", "polymorph_family", "fold", "n_atoms", "energy_total_dft_ev",
                 "energy_per_atom_dft_ev"]
    for column in invariant:
        if frame.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"La columna {column} cambia entre semillas para una estructura física")
    result = frame.groupby(keys + invariant, as_index=False).agg(
        mean_energy_pred_ev_atom=("mean_energy_pred_ev_atom", "mean"),
        residual_pred_ev_atom=("residual_pred_ev_atom", "mean"),
        energy_per_atom_pred_ev=("energy_per_atom_pred_ev", "mean"),
        energy_total_pred_ev=("energy_total_pred_ev", "mean"))
    if result.groupby("model_id").size().to_dict() != {m: 210 for m in ("M1", "M2", "M3", "M4")}:
        raise ValueError("Cada modelo debe mostrar exactamente 210 estructuras físicas")
    return result


def parity_panel(ax: plt.Axes, data: pd.DataFrame, ycol: str, title: str, units: str,
                 reference: str = "energy_per_atom_dft_ev") -> None:
    """Dibuja paridad con OLS libre, referencia y = x, y estadísticos completos.

    Args:
        ax: Ejes destino.
        data: Observaciones físicas promediadas por semilla.
        ycol: Columna de predicción.
        title: Título del panel.
        units: Unidad mostrada.
        reference: Columna DFT para el eje x.
    Returns:
        None.
    """
    x, y = data[reference].to_numpy(float), data[ycol].to_numpy(float)
    shapes = {"dist_perovskite": "o", "hexagonal": "^", "needle": "s"}
    for family, part in data.groupby("polymorph_family"):
        ax.scatter(part[reference], part[ycol], s=21, alpha=.76, marker=shapes[family],
                   color=COLORS.get(str(part.model_id.iloc[0]), "#315a7d"), edgecolor="white", linewidth=.25,
                   label=FAMILIES_ES[family])
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = max((high-low)*.035, 1e-4)
    limits = (low-pad, high+pad)
    ax.plot(limits, limits, color="#555555", linestyle="--", linewidth=1, label="Acuerdo ideal: y = x")
    stat = ols_stats(x, y)
    ax.plot(limits, stat["ols_slope"]*np.asarray(limits)+stat["ols_intercept"], color="#9b3ca6",
            linestyle="-.", linewidth=1.3, label="OLS con intercepto libre")
    mae = float(np.mean(np.abs(y-x)))
    r2pred = float(1.0-np.sum((y-x)**2)/np.sum((x-x.mean())**2))
    ax.text(.035, .965, f"N = {len(x)} estructuras\nm = {stat['ols_slope']:.4f}\n"
            f"b = {stat['ols_intercept']:.4f} {units}\nR² OLS = {stat['ols_r2']:.4f}\n"
            f"MAE = {mae:.4f} {units}\nR² predictivo = {r2pred:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox={"facecolor":"white", "alpha":.87, "edgecolor":"#cccccc"})
    ax.set(xlim=limits, ylim=limits, xlabel=f"DFT ({units})", ylabel=f"Predicción ({units})", title=title)
    ax.grid(alpha=.2)


def figure_one(summary: pd.DataFrame, composition: pd.Series, out: Path) -> None:
    """Compara seis métricas y explicita qué columnas son invariantes por diseño."""
    models = summary.set_index("model_id")
    metrics = [
        ("mae_energy_per_atom_mev_atom", "Error absoluto medio E/N\n(meV/átomo)", 1),
        ("mae_composition_mean_mev_atom", "Error absoluto medio de la media\n(meV/átomo)", 1),
        ("mae_residual_intraformula_mev_atom", "Error absoluto medio residual\n(meV/átomo)", 1),
        ("mae_delta_signed_pairs_mev_atom", "Error absoluto medio de ΔE\n(meV/átomo)", 1),
        ("ground_state_argmin_accuracy", "Estado fundamental\n(%)", 100),
        ("pairwise_accuracy_gt5_predicted_gap_gt1", "Orden de pares elegibles\n(%)", 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, (column, label, scale) in zip(axes.flat, metrics):
        vals = [float(models.loc[m, column])*scale for m in ("M1", "M2", "M3", "M4")]
        bars = ax.bar(range(4), vals, color=[COLORS[m] for m in ("M1", "M2", "M3", "M4")])
        ax.set_xticks(range(4), ["Embeddings,\nresiduo Frozen", "Composición,\nresiduo Frozen",
                                 "Embeddings,\nresiduo LoRA", "Composición,\nresiduo LoRA"], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.bar_label(bars, fmt="%.1f" if scale == 1 else "%.1f%%", padding=2, fontsize=8)
    fig.suptitle("ABS₃: efecto de sustituir Ridge de embeddings por composición", fontsize=15, weight="bold")
    fig.text(.5, .015, "70 fórmulas, 210 estructuras; medias de tres semillas con predicciones fuera del pliegue de entrenamiento. Residuo Frozen idéntico en M1/M2; "
             "residuo LoRA ya entrenado en M3/M4. ΔE y ranking son invariantes al cambiar Ridge. Test histórico cerrado.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94), w_pad=2.2, h_pad=2.0)
    save_figure(fig, out, "comparacion_regresion_hibrida_ridge_embeddings_vs_composicion")


def figure_two(avg: pd.DataFrame, out: Path) -> None:
    """Paridad energía DFT/predicha con cuatro modelos, escalas comunes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    global_low = min(avg.energy_per_atom_dft_ev.min(), avg.energy_per_atom_pred_ev.min())
    global_high = max(avg.energy_per_atom_dft_ev.max(), avg.energy_per_atom_pred_ev.max())
    for ax, model in zip(axes.flat, ("M1", "M2", "M3", "M4")):
        part = avg[avg.model_id.eq(model)].copy()
        parity_panel(ax, part, "energy_per_atom_pred_ev", MODEL_NAMES[model], "eV/átomo")
        ax.set_xlim(global_low-.03, global_high+.03)
        ax.set_ylim(global_low-.03, global_high+.03)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(.5,.045), frameon=False, fontsize=8)
    fig.suptitle("ABS₃: paridad de energía por átomo, predicciones externas", fontsize=15, weight="bold")
    fig.text(.5, .012, "Cada punto representa una estructura física; se promedian sus tres semillas. Mismas 210 estructuras y límites en los cuatro paneles. "
             "La forma identifica familia cristalina. La recta de mínimos cuadrados ordinarios (OLS) y el error absoluto medio usan todos los puntos.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0,.09,1,.94))
    save_figure(fig, out, "paridad_energia_absoluta_ridge_embeddings_vs_ridge_composicional")


def figure_three(comp: pd.DataFrame, out: Path) -> None:
    """Compara la predicción OOF de la media por fórmula para ambos Ridge."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, title, color in ((axes[0], "mean_energy_embedding_ridge_ev_atom", "Ridge sobre embeddings", COLORS["M1"]),
                                  (axes[1], "mean_energy_composition_ridge_ev_atom", "Ridge estequiométrico", COLORS["M2"])):
        x, y = comp.mean_energy_dft_ev_atom.to_numpy(float), comp[col].to_numpy(float)
        ax.scatter(x, y, s=32, color=color, alpha=.82, edgecolor="white", linewidth=.4)
        lim=(min(x.min(),y.min())-.04,max(x.max(),y.max())+.04)
        ax.plot(lim,lim,"--",color="#555555",label="y = x")
        stat=ols_stats(x,y)
        ax.plot(lim,stat["ols_slope"]*np.asarray(lim)+stat["ols_intercept"],"-.",color="#9b3ca6",label="OLS libre")
        ax.text(.04,.96,f"N = {len(x)} fórmulas\nMAE = {np.mean(np.abs(y-x))*1000:.1f} meV/átomo\n"
                f"m = {stat['ols_slope']:.3f}; R² OLS = {stat['ols_r2']:.3f}",transform=ax.transAxes,va="top",
                bbox={"facecolor":"white","alpha":.88,"edgecolor":"#ccc"})
        ax.set_title(title); ax.set_xlabel("Media DFT de los tres polimorfos (eV/átomo)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("Media predicha (eV/átomo)")
    fig.suptitle("ABS₃: energía media por fórmula en predicción externa",fontsize=14,weight="bold")
    fig.text(.5,.01,"70 fórmulas; predicción Ridge externa por pliegue. Cada punto es una composición; no se usan sus etiquetas externas para ajustar el Ridge del pliegue. OLS: mínimos cuadrados ordinarios.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.92)); save_figure(fig,out,"error_energia_media_por_composicion")


def figure_four(decomp: pd.DataFrame, out: Path) -> None:
    """Muestra componentes de MSE, término cruzado y MSE total por modelo."""
    grouped=decomp.groupby("model_id").agg(offset=("mse_mean_component_ev2_atom2","mean"),
        residual=("mse_residual_component_ev2_atom2","mean"),cross=("cross_term_2mean_times_residual_ev2_atom2","mean"),
        total=("mse_total_ev2_atom2","mean"))
    fig,ax=plt.subplots(figsize=(10.5,5.8)); xs=np.arange(4); width=.19
    for j,(col,label,color) in enumerate((
        ("offset","MSE del error de media","#d99b36"),("residual","MSE del residuo","#4f8e79"),
        ("cross","Término cruzado","#8b6ca8"),("total","MSE total observado","#315a7d"))):
        ax.bar(xs+(j-1.5)*width,grouped.loc[["M1","M2","M3","M4"],col],width,label=label,color=color)
    ax.set_xticks(xs,["Emb/Frozen","Comp/Frozen","Emb/LoRA","Comp/LoRA"]);ax.set_ylabel("MSE, (eV/átomo)²")
    ax.set_title("ABS₃: descomposición exacta del error cuadrático medio");ax.grid(axis="y",alpha=.2);ax.legend(ncol=2,fontsize=9)
    fig.text(.5,.01,"Cada valor promedia primero las tres semillas por fórmula y luego las 70 fórmulas. MSE total = MSE media + MSE residuo + término cruzado. "
             "Las MAE de componentes no son aditivas.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,1));save_figure(fig,out,"descomposicion_error_energia_media_y_residuo")


def pair_predictions(avg: pd.DataFrame) -> pd.DataFrame:
    """Construye los tres gaps firmados de cada composición a partir de promedios físicos."""
    rows=[]
    for (model,formula),part in avg.groupby(["model_id","composition_id"]):
        t=part.set_index("polymorph_family").loc[list(cm.FAMILIES)]
        y=t.energy_per_atom_dft_ev.to_numpy(float); p=t.energy_per_atom_pred_ev.to_numpy(float)
        for i,j in combinations(range(3),2):
            rows.append({"model_id":model,"composition_id":formula,"pair":f"{cm.FAMILIES[i]}–{cm.FAMILIES[j]}",
                         "dft":(y[j]-y[i])*1000,"pred":(p[j]-p[i])*1000})
    return pd.DataFrame(rows)


def figure_five(avg: pd.DataFrame, out: Path) -> None:
    """Contrasta los gaps de las condiciones que solo difieren en Ridge."""
    pairs=pair_predictions(avg);fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    for ax,(old,new,title) in zip(axes,(('M1','M2','Residuo Frozen: M1 frente a M2'),('M3','M4','Residuo LoRA: M3 frente a M4'))):
        a=pairs[pairs.model_id.eq(old)].merge(pairs[pairs.model_id.eq(new)],on=["composition_id","pair"],suffixes=("_old","_new"),validate="one_to_one")
        ax.scatter(a.pred_old,a.pred_new,s=17,alpha=.62,color=COLORS[old],edgecolor="none")
        lim=max(float(np.max(np.abs(a[["pred_old","pred_new"]].to_numpy()))),5);ax.plot([-lim,lim],[-lim,lim],"--",color="#333",label="Igualdad exacta esperada")
        maximum=float(np.max(np.abs(a.pred_old-a.pred_new)))
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="ΔE predicha antes (meV/átomo)",ylabel="ΔE predicha después (meV/átomo)",title=title)
        ax.text(.04,.96,f"N = {len(a)} gaps\nMáx. |diferencia| = {maximum:.2g} meV/átomo",transform=ax.transAxes,va="top",bbox={"facecolor":"white","alpha":.9,"edgecolor":"#ccc"});ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("ABS₃: invariancia de diferencias al cambiar solo Ridge",fontsize=14,weight="bold")
    fig.text(.5,.01,"210 gaps en 70 fórmulas, tres pares por fórmula; semillas promediadas por estructura. ΔE_BA = 1000·(e_B − e_A) dentro de una fórmula. "
             "La media común se cancela exactamente y deja el mismo gap.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.92));save_figure(fig,out,"diferencias_energeticas_invariantes_al_cambio_de_ridge")


def figure_six(metrics: pd.DataFrame, out: Path) -> None:
    """Compara selección de estado fundamental y pares por rama residual únicamente."""
    summary=metrics.groupby("model_id").mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,(col,title,denom) in zip(axes,(('ground_state_argmin_accuracy','Estado fundamental exacto','70 fórmulas'),
                                         ('pairwise_accuracy_gt5_predicted_gap_gt1','Orden de pares elegibles','pares |ΔE_DFT| > 5 meV/átomo'))):
        vals=[summary.loc[m,col]*100 for m in ("M1","M3")];bars=ax.bar(["Frozen","LoRA"],vals,color=[COLORS["M1"],COLORS["M3"]]);ax.bar_label(bars,fmt="%.1f%%",padding=3)
        ax.set_ylim(0,105);ax.set_ylabel("Exactitud (%)");ax.set_title(title);ax.grid(axis="y",alpha=.2)
    fig.suptitle("ABS₃: rendimiento polimórfico de las ramas residuales existentes",fontsize=14,weight="bold")
    eligible = int(metrics.loc[metrics.model_id.eq("M1"), "n_pairs_eligible_abs_dft_gap_gt5"].iloc[0])
    fig.text(.5,.005,f"Una vez por rama: reemplazar Ridge no cambia ΔE ni el orden. Estado fundamental: menor E/N, denominador 70 fórmulas/semilla. "
             f"Pares: {eligible} elegibles por semilla (|ΔE_DFT| > 5 meV/átomo). LoRA: adaptadores ya entrenados.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.91));save_figure(fig,out,"comparacion_ground_state_y_ranking_por_rama_residual")


def figure_seven(formula_metrics: pd.DataFrame, out: Path) -> None:
    """Grafica cambios pareados de MAE absoluta por composición."""
    avg=formula_metrics.groupby(["model_id","composition_id"],as_index=False).mae_absolute_ev_atom_mev.mean()
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,(old,new,title,color) in zip(axes,(('M1','M2','Frozen: composicional − embeddings',COLORS['M2']),('M3','M4','LoRA: composicional − embeddings',COLORS['M4']))):
        wide=avg[avg.model_id.isin([old,new])].pivot(index="composition_id",columns="model_id",values="mae_absolute_ev_atom_mev")
        delta=wide[new]-wide[old];order=delta.sort_values().index
        ax.bar(np.arange(len(delta)),delta.loc[order],color=color,alpha=.83,width=.85);ax.axhline(0,color="#333",lw=1)
        ax.set_title(title);ax.set_xlabel("70 fórmulas ordenadas por cambio");ax.grid(axis="y",alpha=.2)
        ax.text(.03,.97,f"Media del cambio = {delta.mean():+.2f} meV/átomo\nNegativo = mejora del MAE",transform=ax.transAxes,va="top",bbox={"facecolor":"white","alpha":.9,"edgecolor":"#ccc"})
    axes[0].set_ylabel("Cambio de MAE E/N (meV/átomo)")
    fig.suptitle("ABS₃: mejora o empeoramiento del MAE por fórmula",fontsize=14,weight="bold")
    fig.text(.5,.01,"Una barra representa una composición; antes se promediaron las tres semillas. La población son las mismas 70 fórmulas evaluadas externamente.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.92));save_figure(fig,out,"mejora_energia_absoluta_por_composicion")


def figure_eight(bootstrap: pd.DataFrame, out: Path) -> None:
    """Muestra IC95% del cambio de MAE apareado por fórmula."""
    chosen=bootstrap[bootstrap.metric.isin(["mae_absolute_ev_atom_mev","composition_mean_absolute_error_mev_atom"])].copy()
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,(metric,title) in zip(axes,(("mae_absolute_ev_atom_mev","MAE energía absoluta"),("composition_mean_absolute_error_mev_atom","MAE media por fórmula"))):
        q=chosen[chosen.metric.eq(metric)].reset_index(drop=True);y=np.arange(len(q));x=q.difference_new_minus_old.to_numpy()
        err=np.vstack((x-q.ci95_lower.to_numpy(),q.ci95_upper.to_numpy()-x));ax.errorbar(x,y,xerr=err,fmt="o",capsize=4,color="#315a7d")
        ax.axvline(0,color="#555",linestyle="--");ax.set_yticks(y,["Residual Frozen","Residual LoRA"]);ax.set_title(title);ax.set_xlabel("Composicional − embeddings (meV/átomo)");ax.grid(axis="x",alpha=.2)
    fig.suptitle("ABS₃: diferencia apareada e intervalo de confianza del 95% por composición",fontsize=14,weight="bold")
    fig.text(.5,.01,"Bootstrap percentile: 10 000 remuestreos de 70 composiciones; las tres semillas se promedian antes del remuestreo. Valores negativos favorecen Ridge composicional.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.91));save_figure(fig,out,"diferencias_pareadas_mae_con_intervalos_confianza")


def figure_nine(avg: pd.DataFrame, out: Path) -> None:
    """Verifica que la salida residual no cambia al sustituir Ridge."""
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    for ax,(old,new,title) in zip(axes,(('M1','M2','Residuo Frozen'),('M3','M4','Residuo LoRA'))):
        a=avg[avg.model_id.eq(old)].merge(avg[avg.model_id.eq(new)],on=["structure_id","composition_id","polymorph_family"],suffixes=("_old","_new"),validate="one_to_one")
        x=(a.energy_per_atom_dft_ev_old-a.groupby("composition_id").energy_per_atom_dft_ev_old.transform("mean"))*1000
        y1=a.residual_pred_ev_atom_old*1000;y2=a.residual_pred_ev_atom_new*1000
        ax.scatter(x,y1,s=18,color=COLORS[old],alpha=.6,label="Ridge de embeddings")
        ax.scatter(x,y2,s=8,color=COLORS[new],alpha=.45,label="Ridge composicional")
        stat=ols_stats(x.to_numpy(),y1.to_numpy());lim=(min(x.min(),y1.min(),y2.min())-2,max(x.max(),y1.max(),y2.max())+2)
        ax.plot(lim,lim,"--",color="#555",label="y = x")
        ax.text(.04,.96,f"N = 210 estructuras\nMáx. diferencia de residuo = {np.max(np.abs(y1-y2)):.2g} meV/átomo\n"
                f"m = {stat['ols_slope']:.3f}; R² OLS = {stat['ols_r2']:.3f}",transform=ax.transAxes,va="top",bbox={"facecolor":"white","alpha":.88,"edgecolor":"#ccc"})
        ax.set(xlim=lim,ylim=lim,xlabel="DFT centrada por fórmula (meV/átomo)",ylabel="Residuo MLP8 predicho (meV/átomo)",title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("ABS₃: residuos intrafórmula conservados al cambiar Ridge",fontsize=14,weight="bold")
    fig.text(.5,.01,"Cada punto es una estructura; residuo DFT = E/N menos media de los tres polimorfos de su fórmula. Semillas promediadas. Los valores se superponen porque la rama MLP8 no cambió.",ha="center",fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,.92));save_figure(fig,out,"paridad_residuos_intraformula_sin_cambio_de_regresor")


def architecture_diagram(out: Path) -> None:
    """Escribe Mermaid editable y renderiza un diagrama paralelo de la arquitectura."""
    source='''flowchart LR
  subgraph C[ Rama composicional: nueva ]
    F[Fórmula química] --> X[17 fracciones estequiométricas]
    X --> S[StandardScaler]
    S --> R[Ridge por pliegue: alpha seleccionado en validación interna]
    R --> MU[μ̂ composición, eV/átomo]
  end
  subgraph E[ Rama estructural: pesos ya existentes ]
    SL[SLICES original] --> T[Tokenizer y MatGPT congelado]
    T --> H[FINAL_END, 768 dimensiones]
    H --> M[MLP8: 768→8, GELU, Dropout 0.1, 8→1]
    M --> RC[Centrado de residuo en los tres polimorfos]
    L[Alternativa: residuos LoRA ya guardados] -. reemplaza MLP8 Frozen .-> RC
  end
  MU --> SUM[μ̂ + residuo centrado]
  RC --> SUM
  SUM --> EA[Predicción E/N, eV/átomo]
  EA --> ET[Multiplicar por N: E total, eV]
  EA --> DE[ΔE intrafórmula, ranking y estado fundamental]
  subgraph Q[Evaluación por cada terna A, B, C]
    A[Polimorfo A] --- B[Polimorfo B] --- Cc[Polimorfo C]
  end
  DE -. compara A, B y C .-> Q
'''
    out.mkdir(parents=True,exist_ok=True)
    (out/"arquitectura_regresion_energetica_hibrida_ridge_composicional_matgpt.mmd").write_text(source,encoding="utf-8")
    # Matplotlib permite producir PNG/SVG sin exigir Graphviz/Mermaid instalado.
    fig,ax=plt.subplots(figsize=(15,7.5));ax.axis("off");ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x: float, y: float, w: float, h: float, text: str, color: str, fontsize: int = 9) -> None:
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize,
                bbox={"boxstyle":"round,pad=.55","facecolor":color,"edgecolor":"#48525c","linewidth":1.1})
    def arrow(x1:float,y1:float,x2:float,y2:float,label:str="")->None:
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops={"arrowstyle":"->","lw":1.4,"color":"#425466"})
        if label: ax.text((x1+x2)/2,(y1+y2)/2+.018,label,ha="center",fontsize=8,color="#333")
    ax.text(.25,.94,"Rama composicional · ajuste Ridge por pliegue",ha="center",fontsize=13,weight="bold",color="#9a5b18")
    ax.text(.73,.94,"Rama estructural · extractor y residuos existentes",ha="center",fontsize=13,weight="bold",color="#315a7d")
    coords=[(.02,.73,.12,.1,"Fórmula química","#fff0d7"),(.18,.73,.16,.1,"17 fracciones\nestequiométricas","#fff0d7"),(.39,.73,.13,.1,"StandardScaler","#fff0d7"),(.57,.73,.16,.1,"Ridge por fold\n(α de inner-val)","#fff0d7"),(.79,.73,.18,.1,"Media μ̂\n(eV/átomo)","#ffe2b0")]
    for item in coords: box(*item)
    for a,b in ((.14,.18),(.34,.39),(.52,.57),(.73,.79)):arrow(a,.78,b,.78)
    coords2=[(.02,.57,.12,.1,"SLICES","#e3edf7"),(.18,.57,.15,.1,"MatGPT pretrained\ncongelado","#e3edf7"),(.38,.57,.13,.1,"FINAL_END\n768","#e3edf7"),(.57,.57,.17,.1,"MLP8 existente\n768→8→1","#d8e9f1"),(.80,.57,.17,.1,"Residuo centrado\npor fórmula","#cce3ec")]
    for item in coords2:box(*item)
    for a,b in ((.14,.18),(.33,.38),(.51,.57),(.74,.80)):arrow(a,.62,b,.62)
    box(.05,.385,.31,.075,"Alternativa: residuo del modelo con LoRA\n(guardado; no se reentrena)","#e8dff0",8)
    box(.43,.245,.27,.10,"E/N por polimorfo\nμ̂ + residuo centrado","#e8f0d5",10)
    ax.text(.56,.385,"Para cada polimorfo: media común + residuo centrado",ha="center",fontsize=9,
            bbox={"facecolor":"white","alpha":.9,"edgecolor":"#cccccc"})
    box(.75,.245,.12,.10,"E_total\n= N·e","#f3e0df",9)
    arrow(.70,.295,.75,.295)
    box(.89,.245,.10,.10,"ΔE, orden\ny estado GS","#f3e0df",8)
    arrow(.87,.295,.89,.295)
    ax.text(.5,.09,"En cada fórmula: polimorfos A, B y C. ΔE_BA = 1000·(e_B − e_A); GS = argmin(e_A, e_B, e_C).",ha="center",fontsize=9)
    fig.suptitle("Arquitectura auditada: ramas paralelas para energía media y residuo polimórfico",fontsize=16,weight="bold",y=.99)
    fig.savefig(out/"arquitectura_regresion_energetica_hibrida_ridge_composicional_matgpt.png",dpi=180,bbox_inches="tight")
    fig.savefig(out/"arquitectura_regresion_energetica_hibrida_ridge_composicional_matgpt.svg",bbox_inches="tight");plt.close(fig)


def generate(output: Path = cm.REPORT) -> dict[str, Any]:
    """Genera las nueve figuras y el esquema fuente de arquitectura.

    Args:
        output: Carpeta con resultados de análisis.
    Returns:
        Estado y lista de figuras.
    """
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True)
    predictions=averaged_predictions(output)
    summary=pd.read_csv(output/"summary_by_model.csv")
    baseline=pd.read_csv(output/"composition_only_metrics.csv").iloc[0]
    comp=pd.read_csv(output/"oof_mean_energy_by_composition.csv")
    decomp=pd.read_csv(output/"mse_error_decomposition_by_composition.csv")
    formula_metrics=pd.read_csv(output/"composition_metrics_by_formula.csv")
    bootstrap=pd.read_csv(output/"paired_bootstrap_by_composition.csv")
    metrics=pd.read_csv(output/"metrics_by_seed.csv")
    funcs=(lambda:figure_one(summary,baseline,figures),lambda:figure_two(predictions,figures),
           lambda:figure_three(comp,figures),lambda:figure_four(decomp,figures),
           lambda:figure_five(predictions,figures),lambda:figure_six(metrics,figures),
           lambda:figure_seven(formula_metrics,figures),lambda:figure_eight(bootstrap,figures),
           lambda:figure_nine(predictions,figures))
    for func in funcs:func()
    architecture_diagram(figures)
    png=sorted(path.name for path in figures.glob("*.png"));svg=sorted(path.name for path in figures.glob("*.svg"))
    if len(png)!=10 or len(svg)!=10:
        raise RuntimeError(f"Se esperaban 9 figuras y un diagrama en PNG/SVG; obtenidos {len(png)} PNG y {len(svg)} SVG")
    LOG.info("Figuras creadas: %d PNG y %d SVG",len(png),len(svg))
    return {"status":"PASS","png":png,"svg":svg}


def main() -> None:
    """CLI para generar figuras a partir de las tablas reproducibles."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=cm.REPORT)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format="%(message)s");generate(args.output.resolve())


if __name__=="__main__":main()
