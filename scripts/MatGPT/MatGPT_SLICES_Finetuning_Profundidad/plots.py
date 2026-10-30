"""Genera diez figuras comparables con métricas agrupadas por material."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_matgpt_depth"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
REPORT = EXPERIMENT
FIG = ROOT / "images/MatGPT_SLICES_Finetuning_Profundidad"
COLORS = {"F0": "#6B7280", "F2": "#4C78A8", "F4": "#F2A541", "F8": "#59A14F"}
ORDER = ("F0", "F2", "F4", "F8")


def save(fig: plt.Figure, name: str) -> None:
    """Guarda una figura PNG de 300 dpi."""
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Construye las figuras 1–10 de resultados sin duplicar gráficos por semilla."""
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((REPORT / "metrics.json").read_text())
    pred = pd.read_csv(REPORT / "predicciones_oof_por_estructura.csv")
    pairs = pd.read_csv(REPORT / "predicciones_oof_por_par.csv")
    comp = pd.read_csv(REPORT / "metricas_por_composicion.csv")
    history = pd.read_csv(REPORT / "historial_entrenamiento.csv")
    summary = metrics["by_condition"]

    # 1. Resumen de magnitud energética y ordenamiento.
    names = ["MAE E/N\n(meV/átomo)", "MAE residuo\n(meV/átomo)", "MAE ΔE\n(meV/átomo)", "Estado fundamental\n(%)", "Pares ordenados\n(%)"]
    fields = ["mae_energy_mev_atom", "mae_residual_mev_atom", "mae_delta_signed_pairs_mev_atom", "ground_state_accuracy_exact", "pairwise_accuracy_eligible"]
    fig, axes = plt.subplots(1,5,figsize=(17,4.5))
    for ax,name,field in zip(axes,names,fields):
        vals=[summary[c].get(field) for c in ORDER]
        vals=[np.nan if v is None else (100*v if field in fields[-2:] else v) for v in vals]
        ax.bar(ORDER,vals,color=[COLORS[c] for c in ORDER]); ax.set_title(name,fontsize=10)
        ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("ABS₃: profundidad de ajuste, energía y orden de polimorfos",weight="bold",fontsize=14)
    fig.text(.5,-.04,"210 estructuras físicas, 70 composiciones, 3 semillas promediadas; ΔE = E_B − E_A dentro de la fórmula. F0 reutiliza el control congelado.",ha="center",fontsize=9)
    save(fig,"comparacion_profundidad_finetuning_mae_y_estabilidad_polimorfica")

    # 2. Bootstrap de diferencias pareadas agrupado por fórmula.
    entries=metrics["bootstrap"]
    selected=[(k,v) for k,v in entries.items() if "MAE energía absoluta" in k]
    fig,ax=plt.subplots(figsize=(8,5)); y=np.arange(len(selected))
    values=[v["mean_difference"] for _,v in selected]
    lo=[v["ci95_low"] for _,v in selected]; hi=[v["ci95_high"] for _,v in selected]
    ax.errorbar(values,y,xerr=[np.asarray(values)-lo,np.asarray(hi)-values],fmt="o",capsize=4,color="#375A7F")
    ax.axvline(0,color="black",lw=1,ls="--"); ax.set_yticks(y,[k.split(" | ")[0] for k,_ in selected]); ax.invert_yaxis()
    ax.set_xlabel("Cambio de MAE frente al control (meV/átomo; negativo favorece la condición izquierda)")
    ax.set_title("Cambio apareado de error absoluto con intervalo del 95%")
    ax.text(.01,-.2,"Remuestreo de 10 000 conjuntos de 70 composiciones; las semillas se promediaron dentro de cada fórmula.",transform=ax.transAxes,fontsize=9)
    ax.grid(axis="x",alpha=.2); save(fig,"diferencias_pareadas_finetuning_intervalos_confianza")

    # Población promediada por estructura antes de los gráficos de paridad.
    avg=pred.groupby(["condition","composition_id","structure_id","polymorph_family"],as_index=False).agg(
        dft=("energy_per_atom_dft_ev","first"), predicted=("energy_per_atom_pred_ev","mean"),
        residual=("residual_pred_ev_atom","mean"), formula=("composition_id","first"))
    avg["dft_residual"] = avg.dft - avg.groupby(["condition","composition_id"]).dft.transform("mean")
    for name, ref, value, title, unit, filename in (
        ("energy", "dft", "predicted", "ABS₃: energía DFT por átomo frente a predicción", "eV/átomo", "paridad_energia_por_atomo_segun_profundidad_finetuning"),
        ("residual", "dft_residual", "residual", "ABS₃: residuos energéticos dentro de cada composición", "meV/átomo", "paridad_residuos_energeticos_segun_bloques_adaptados")):
        fig,axes=plt.subplots(2,2,figsize=(10,9)); all_ref=[]; all_val=[]
        for condition,ax in zip(ORDER,axes.flat):
            data=avg[avg.condition==condition]
            x=data[ref].to_numpy(float); yv=data[value].to_numpy(float)
            if name=="residual": x=x*1000; yv=yv*1000
            all_ref.extend(x); all_val.extend(yv)
            ax.scatter(x,yv,s=13,alpha=.65,color=COLORS[condition],edgecolors="none")
            slope,intercept=np.polyfit(x,yv,1) if np.std(x)>0 else (np.nan,np.nan)
            lim=(min(x.min(),yv.min()),max(x.max(),yv.max())); ax.plot(lim,lim,"--",color="#344054",lw=1)
            if np.isfinite(slope): ax.plot(lim,slope*np.asarray(lim)+intercept,"-.",color="#BA2D8C",lw=1)
            ax.text(.04,.96,f"N = {len(x)}\npendiente = {slope:.3f}\nMAE = {np.mean(np.abs(x-yv)):.2f} {unit}",transform=ax.transAxes,va="top",fontsize=8)
            ax.set_title(condition); ax.set_xlabel(f"Referencia DFT ({unit})"); ax.set_ylabel(f"Predicción ({unit})"); ax.grid(alpha=.2)
        low=min(all_ref+all_val); high=max(all_ref+all_val)
        for ax in axes.flat: ax.set_xlim(low,high); ax.set_ylim(low,high)
        fig.suptitle(title+" · misma escala en los cuatro paneles",weight="bold")
        fig.text(.5,.005,"Cada punto es una estructura evaluada fuera de su pliegue; se promediaron tres repeticiones de entrenamiento. Línea discontinua: y=x; línea punteada: ajuste lineal con intercepto.",ha="center",fontsize=8)
        save(fig,filename)

    # 5. Diferencias firmadas por pares.
    avgpairs=pairs.groupby(["condition","composition_id","structure_id_a","structure_id_b"],as_index=False).mean(numeric_only=True)
    fig,axes=plt.subplots(2,2,figsize=(10,9)); low=min(avgpairs.delta_dft_mev_atom.min(),avgpairs.delta_pred_mev_atom.min()); high=max(avgpairs.delta_dft_mev_atom.max(),avgpairs.delta_pred_mev_atom.max())
    for condition,ax in zip(ORDER,axes.flat):
        d=avgpairs[avgpairs.condition==condition]; ax.scatter(d.delta_dft_mev_atom,d.delta_pred_mev_atom,s=13,alpha=.55,color=COLORS[condition])
        slope,intercept=np.polyfit(d.delta_dft_mev_atom,d.delta_pred_mev_atom,1)
        ax.plot([low,high],[low,high],"--",color="#344054"); ax.plot([low,high],slope*np.array([low,high])+intercept,"-.",color="#BA2D8C")
        ax.text(.04,.96,f"210 pares\npendiente = {slope:.3f}\nrazón de dispersión = {d.delta_pred_mev_atom.std()/d.delta_dft_mev_atom.std():.3f}",transform=ax.transAxes,va="top",fontsize=8)
        ax.set_title(condition); ax.set_xlabel("ΔE DFT, E_B − E_A (meV/átomo)"); ax.set_ylabel("ΔE predicha (meV/átomo)"); ax.set_xlim(low,high); ax.set_ylim(low,high); ax.grid(alpha=.2)
    fig.suptitle("ABS₃: diferencias de energía por pares dentro de cada fórmula",weight="bold")
    fig.text(.5,.005,"Un punto representa un par de polimorfos de una composición; 70 fórmulas × 3 pares, con tres semillas promediadas. ΔE = E_B − E_A.",ha="center",fontsize=8)
    save(fig,"paridad_diferencias_energeticas_segun_profundidad_finetuning")

    # 6 y 7. Error y precisión de ordenamiento por amplitud del gap DFT.
    bins=list(metrics["gap_bins"]["F0"]); labels=list(bins); x=np.arange(len(labels)); width=.19
    fig,ax=plt.subplots(figsize=(10,5))
    for ix,condition in enumerate(ORDER):
        values=[metrics["gap_bins"][condition][label]["mae_delta_mev_atom"] for label in labels]
        ax.bar(x+(ix-1.5)*width,values,width,label=condition,color=COLORS[condition])
    baseline=[metrics["gap_bins"]["F0"][label]["zero_baseline_mae_mev_atom"] for label in labels]
    ax.plot(x,baseline,"k--o",label="ΔE predicha = 0")
    ax.set_xticks(x,labels); ax.set_xlabel("Magnitud del gap DFT, |ΔE| (meV/átomo)"); ax.set_ylabel("MAE de ΔE (meV/átomo)")
    ax.set_title("Error de diferencias según separación energética DFT"); ax.legend(ncol=3); ax.grid(axis="y",alpha=.2)
    fig.text(.5,.005,"Pares dentro de una misma composición; MAE y control cero usan los mismos pares de cada intervalo. El intervalo 0–5 no se usa para evaluar el signo.",ha="center",fontsize=8)
    save(fig,"error_diferencias_energeticas_por_magnitud_gap_dft")
    fig,ax=plt.subplots(figsize=(10,5))
    for ix,condition in enumerate(ORDER):
        vals=[]; counts=[]
        for label in labels:
            item=metrics["gap_bins"][condition][label]
            vals.append(np.nan if item["pairwise_accuracy_resolved"] is None else item["pairwise_accuracy_resolved"]*100)
            counts.append(item["eligible_pairs_gt_5_mev"])
        ax.plot(x,vals,marker="o",label=condition,color=COLORS[condition])
    ax.axhline(50,color="black",ls="--",label="Referencia aleatoria: 50%")
    ax.set_xticks(x,labels); ax.set_xlabel("Magnitud del gap DFT, |ΔE| (meV/átomo)"); ax.set_ylabel("Acierto de orden entre pares resueltos (%)")
    ax.set_ylim(0,105); ax.set_title("Ordenamiento correcto según tamaño del gap DFT"); ax.legend(ncol=3); ax.grid(alpha=.2)
    ax.text(.01,-.22,"Elegibilidad: |ΔE DFT| > 5 meV/átomo. Predicción no resuelta: |ΔE predicha| ≤ 1 meV/átomo; no se cuenta como acierto. Denominadores por condición en metrics.json.",transform=ax.transAxes,fontsize=8)
    save(fig,"exactitud_ordenamiento_por_gap_y_profundidad_finetuning")

    # 8. Curvas internas y épocas seleccionadas.
    full=history[history.run_type=="full"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for c in ("F2","F4","F8"):
        d=full[(full.condition==c)&(full.fold==0)&(full.seed==42)]
        if len(d):
            axes[0].plot(d.epoch,d.train_loss,label=f"{c} · entrenamiento",color=COLORS[c],ls="-")
            axes[0].plot(d.epoch,d.inner_validation_loss,label=f"{c} · validación",color=COLORS[c],ls="--")
    axes[0].set_xlabel("Época local"); axes[0].set_ylabel("SmoothL1 residual normalizada"); axes[0].set_title("Pliegue 0, semilla 42"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    best=REPORT/"resources.json"
    resources=json.loads(best.read_text())["runs"] if best.exists() else {}
    ep={c:[v.get("best_epoch") for k,v in resources.items() if k.startswith(c.lower()+"_fold") and not k.endswith("_smoke") and v.get("status")=="COMPLETE"] for c in ("F2","F4","F8")}
    axes[1].boxplot([ep[c] for c in ("F2","F4","F8")],tick_labels=("F2","F4","F8"),showmeans=True)
    axes[1].set_ylabel("Época elegida por validación interna"); axes[1].set_title("Distribución entre pliegues y semillas"); axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("Entrenamiento y selección interna; no se usaron etiquetas outer para elegir épocas",weight="bold")
    save(fig,"curvas_entrenamiento_validacion_segun_profundidad")

    # 9. Drift de representaciones: pendiente de cálculo hasta que haya checkpoints completos.
    drift=metrics.get("representation_diagnostics",{}).get("by_condition",{})
    fig,axes=plt.subplots(1,2,figsize=(10,4.8)); x=np.arange(3)
    cos=[drift.get(c,{}).get("mean_cosine_to_original",np.nan) for c in ("F2","F4","F8")]
    cos_sd=[drift.get(c,{}).get("std_cosine_to_original",0) for c in ("F2","F4","F8")]
    l2=[drift.get(c,{}).get("mean_relative_l2_drift",np.nan) for c in ("F2","F4","F8")]
    l2_sd=[drift.get(c,{}).get("std_relative_l2_drift",0) for c in ("F2","F4","F8")]
    axes[0].bar(x,cos,yerr=cos_sd,color=[COLORS[c] for c in ("F2","F4","F8")],capsize=4)
    axes[0].set_xticks(x,("F2","F4","F8")); axes[0].set_ylim(-1,1.05)
    axes[0].set_ylabel("Similitud coseno con MatGPT original"); axes[0].set_title("Orientación de FINAL_END")
    axes[1].bar(x,l2,yerr=l2_sd,color=[COLORS[c] for c in ("F2","F4","F8")],capsize=4)
    axes[1].set_xticks(x,("F2","F4","F8")); axes[1].set_ylabel("Cambio L2 relativo medio"); axes[1].set_title("Magnitud del cambio por estructura")
    fig.suptitle("Cambio de representaciones en 168 estructuras de desarrollo del pliegue 0",weight="bold")
    fig.text(.5,.01,"Se comparó FINAL_END sobre las mismas estructuras de entrenamiento/validación interna. Barras: desviación estándar entre estructuras; rango espectral e intra-fórmula en metrics.json.",ha="center",fontsize=8)
    save(fig,"cambio_embeddings_segun_profundidad_adaptada")

    # 10. Número de parámetros y brecha de generalización, descriptivo.
    counts={"F0":6161,"F2":18903552+6161,"F4":37807104+6161,"F8":75614208+6161}
    means=full.groupby("condition").agg(train=("train_loss","mean"),valid=("inner_validation_loss","mean"))
    fig,ax=plt.subplots(figsize=(8,5))
    for c in ("F2","F4","F8"):
        if c in means.index:
            ax.scatter(counts[c],means.loc[c,"valid"]-means.loc[c,"train"],s=100,color=COLORS[c],label=c)
            ax.annotate(c,(counts[c],means.loc[c,"valid"]-means.loc[c,"train"]),xytext=(5,5),textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Parámetros entrenables (escala logarítmica)"); ax.set_ylabel("Brecha media de pérdida: validación − entrenamiento")
    ax.set_title("Capacidad entrenable y brecha interna de generalización"); ax.grid(alpha=.2)
    ax.text(.01,-.18,"Los puntos resumen pliegues y semillas; son asociaciones descriptivas y no prueban causalidad.",transform=ax.transAxes,fontsize=9)
    save(fig,"parametros_entrenables_sobreajuste_y_rendimiento")


if __name__ == "__main__":
    main()
