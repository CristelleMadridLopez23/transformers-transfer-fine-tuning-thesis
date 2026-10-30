#!/usr/bin/env python3
"""Compara D2 centrado con E2-A usando grupos y folds idénticos."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, write_json
from Alternative2.scripts.MatGPT.MatGPT_diagnostics.d2_centered_training import group_metrics

LOG = logging.getLogger("matgpt_diagnostics.d2_summary")
E2_OOF = ALT / "reports/MatGPT_finetuning/matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv"


def run() -> None:
    """Escribe métricas por composición, diferencias pareadas e informe breve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "cv"], default="pilot")
    parser.add_argument("--input", type=Path, default=REPORT / "D2/pilot_v1/oof_predictions_pilot.csv")
    parser.add_argument("--output", type=Path, default=REPORT / "D2/pilot_v1")
    args = parser.parse_args()
    pred = pd.read_csv(args.input)
    if pred.empty or pred.structure_id.astype(str).str.contains("test", case=False).any():
        raise ValueError("Predicciones D2 vacías o incluyen test.")
    e2 = pd.read_csv(E2_OOF)
    seeds, folds = sorted(pred.seed.unique()), sorted(pred.fold.unique())
    base = e2[(e2.configuration == "partial_ft") & e2.seed.isin(seeds) & e2.fold.isin(folds)].copy()
    if args.phase == "pilot":
        base = base[base.seed.eq(seeds[0]) & base.fold.eq(folds[0])]
    else:
        base = base.groupby(["seed", "fold", "structure_id"], as_index=False).agg(
            composition_group=("composition_group", "first"), reduced_formula=("reduced_formula", "first"),
            polymorph_family=("polymorph_family", "first"), energy_per_atom_ev=("energy_per_atom_ev", "first"),
            energy_pred_ev_atom=("energy_pred_ev_atom", "mean"))
    cols = ["structure_id", "composition_group", "reduced_formula", "polymorph_family", "energy_per_atom_ev", "energy_pred_ev_atom", "fold", "seed"]
    base = base[cols]; base["configuration"] = "D2-BASE E2-A partial_ft"
    pred["configuration"] = pred.configuration.astype(str)
    joined = pd.concat([base, pred], ignore_index=True, sort=False)
    rows = []
    for keys, group in joined.groupby(["configuration", "seed", "fold", "composition_group"]):
        rows.append({"configuration": keys[0], "seed": int(keys[1]), "fold": int(keys[2]), "composition_group": keys[3], **group_metrics(group)})
    comp = pd.DataFrame(rows)
    comp.to_csv(args.output / f"metrics_composition_{args.phase}_paired.csv", index=False)
    summary = comp.groupby(["configuration", "seed", "fold"]).agg(
        n_compositions=("composition_group", "nunique"), n_gs_eligible=("ground_state_eligible", "sum"),
        **{m: (m, "mean") for m in ["absolute_mae_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "pairwise_energy_mae_mev_atom", "pairwise_signed_bias_mev_atom", "absolute_rank_error_mean", "signed_rank_error_mean", "spearman_by_composition", "pred_range_mev_atom", "collapsed"]}).reset_index()
    summary.to_csv(args.output / f"metrics_paired_{args.phase}.csv", index=False)
    # Desglosar variación por fold y semilla: folds son grupos distintos, las
    # semillas repiten el mismo split y no deben contarse como nuevas fórmulas.
    fold_summary = summary.groupby(["configuration", "fold"]).agg(
        **{m: (m, "mean") for m in ["absolute_mae_mev_atom", "delta_e_mae_mev_atom",
            "ground_state_accuracy", "pairwise_accuracy"]}).reset_index()
    seed_summary = summary.groupby(["configuration", "seed"]).agg(
        **{m: (m, "mean") for m in ["absolute_mae_mev_atom", "delta_e_mae_mev_atom",
            "ground_state_accuracy", "pairwise_accuracy"]}).reset_index()
    fold_summary.to_csv(args.output / f"metrics_mean_by_fold_{args.phase}.csv", index=False)
    seed_summary.to_csv(args.output / f"metrics_mean_by_seed_{args.phase}.csv", index=False)
    history_path = args.output / f"training_history_{args.phase}.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        timing = history.groupby(["configuration", "fold", "seed"], as_index=False).agg(
            epochs_completed=("epoch", "max"), training_wall_seconds=("elapsed_seconds", "max"))
        timing.to_csv(args.output / f"training_time_by_run_{args.phase}.csv", index=False)
    diffs = []
    boot = {}
    rng = np.random.default_rng(2301026)
    base_name = "D2-BASE E2-A partial_ft"
    for variant in sorted(pred.configuration.unique()):
        a = comp[comp.configuration.eq(variant)].set_index(["seed", "fold", "composition_group"])
        b = comp[comp.configuration.eq(base_name)].set_index(["seed", "fold", "composition_group"])
        common = a.index.intersection(b.index)
        for metric in ["absolute_mae_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy"]:
            delta = (a.loc[common, metric] - b.loc[common, metric]).dropna()
            for key, value in delta.items():
                diffs.append({"configuration": variant, "metric": metric, "seed": key[0], "fold": key[1], "composition_group": key[2], "d2_minus_e2a": float(value)})
            if len(delta):
                # Seeds are averaged within composition before bootstrap, not treated as extra samples.
                by_composition = pd.Series(delta.to_numpy(float), index=pd.MultiIndex.from_tuples(delta.index, names=["seed", "fold", "composition_group"])).groupby(level="composition_group").mean().to_numpy(float)
                draws = np.array([rng.choice(by_composition, len(by_composition), replace=True).mean() for _ in range(2000)])
                boot[f"{variant}|{metric}"] = {"mean_difference": float(by_composition.mean()), "median_difference": float(np.median(by_composition)),
                    "ci95_percentile": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                    "n_compositions": int(len(by_composition)), "resamples": 2000, "seed": 2301026,
                    "note": "Exploratory paired bootstrap; each composition remains grouped."}
    pd.DataFrame(diffs).to_csv(args.output / f"paired_differences_{args.phase}.csv", index=False)
    write_json(args.output / f"paired_bootstrap_{args.phase}.json", boot)
    if args.phase == "cv":
        # Figura 1: error por composición, comparando las mismas 70 fórmulas.
        # Para cada fórmula se promedian las tres semillas antes del punto.
        plotdir = args.output / "figures"
        plotdir.mkdir(parents=True, exist_ok=True)
        colors = {base_name: "#666666", "D2-CENTER": "#1B8A78", "D2-CENTER-ONLY": "#A04D95"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, metric, label in zip(axes,
                ["absolute_mae_mev_atom", "delta_e_mae_mev_atom"],
                ["MAE absoluto (meV/átomo)", "MAE de ΔE (meV/átomo)"]):
            wide = comp.pivot_table(index="composition_group", columns="configuration", values=metric, aggfunc="mean")
            for name in ["D2-CENTER", "D2-CENTER-ONLY"]:
                ax.scatter(wide[base_name], wide[name], s=28, alpha=.72, color=colors[name],
                           edgecolors="white", linewidths=.35, label=name)
            low = float(np.nanmin(wide[[base_name, "D2-CENTER", "D2-CENTER-ONLY"]].to_numpy()))
            high = float(np.nanmax(wide[[base_name, "D2-CENTER", "D2-CENTER-ONLY"]].to_numpy()))
            pad = max((high-low)*.04, 1.0)
            ax.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#333333", label="Igual error")
            ax.set(xlim=(low-pad,high+pad), ylim=(low-pad,high+pad), xlabel=f"E2-A: {label}", ylabel=f"D2: {label}")
            ax.set_title(label); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5,.96), frameon=False)
        fig.suptitle("D2: comparación pareada por composición en validación cruzada", y=1.02, fontsize=13, fontweight="bold")
        fig.text(.5,.01,"Cada punto representa una de las 70 composiciones; primero se promedian las 3 semillas.\nLos 5 folds cubren composiciones de desarrollo sin solapamiento. Bajo la diagonal, D2 tiene menor error que E2-A. Test histórico excluido.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.09,1,.9)); fig.savefig(plotdir/"d2_cv_error_pareado.png",dpi=200,bbox_inches="tight"); plt.close(fig)
        # Figura 2: desempeño y variación entre folds y semillas.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        for ax, metric, label in zip(axes, ["ground_state_accuracy","pairwise_accuracy"],
                                     ["Selección de estado fundamental (%)","Orden de pares (%)"]):
            for name, color in colors.items():
                g = summary[summary.configuration.eq(name)]
                ax.scatter(g.fold + (g.seed.map({42:-.08,123:0,2026:.08})),
                           g[metric]*100, color=color, s=35, alpha=.8, label=name)
            ax.set_xticks(sorted(summary.fold.unique()), [str(int(x)+1) for x in sorted(summary.fold.unique())])
            ax.set_xlabel("Fold de composición"); ax.set_ylabel(label); ax.set_title(label); ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5,.97), frameon=False)
        fig.suptitle("D2: ranking entre folds y semillas", y=1.03, fontsize=13, fontweight="bold")
        fig.text(.5,.01,"Cada marca = un fold y semilla; cada composición aparece en un solo fold. GS excluye tripletes DFT degenerados (gap ≤1 meV/átomo) y predicciones empatadas; pairwise excluye pares DFT empatados.\nLa línea horizontal no se usa: la figura muestra variación, no una garantía estadística. Sin test histórico.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.11,1,.87)); fig.savefig(plotdir/"d2_cv_ranking_folds_semillas.png",dpi=200,bbox_inches="tight"); plt.close(fig)
        # Figura 3: medias y dispersión entre las 15 ejecuciones por pérdida.
        # En cada época se incluyen solo las ejecuciones que aún no habían parado.
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
        for ax, name, color in zip(axes, ["D2-CENTER", "D2-CENTER-ONLY"], [colors["D2-CENTER"], colors["D2-CENTER-ONLY"]]):
            h = history[history.configuration.eq(name)]
            stats = h.groupby("epoch").agg(train_mean=("train_objective", "mean"), train_sd=("train_objective", "std"),
                val_mean=("inner_validation_objective", "mean"), val_sd=("inner_validation_objective", "std"), n_active=("seed", "size"))
            x = stats.index.to_numpy(float)
            ax.plot(x, stats.train_mean, color="#2A6FBB", label="Inner-train: media entre ejecuciones")
            ax.fill_between(x, stats.train_mean-stats.train_sd.fillna(0), stats.train_mean+stats.train_sd.fillna(0), color="#2A6FBB", alpha=.15, label="±1 DE entre ejecuciones")
            ax.plot(x, stats.val_mean, color="#D17A00", label="Inner-validation: media completa")
            ax.fill_between(x, stats.val_mean-stats.val_sd.fillna(0), stats.val_mean+stats.val_sd.fillna(0), color="#D17A00", alpha=.15)
            best = pred.loc[pred.configuration.eq(name)].groupby(["fold","seed"]).best_epoch.first()
            ax.axvline(float(best.mean()), color="#333333", linestyle="--", label="Media de época elegida")
            ax.set(xlabel="Época local", ylabel="Objetivo de entrenamiento (adimensional)", title=name)
            ax.grid(alpha=.2)
            ax.text(.98,.97,f"época elegida media = {best.mean():.1f}\nn activo al final = {int(stats.n_active.iloc[-1])}",transform=ax.transAxes,ha="right",va="top",fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5,.98), frameon=False)
        fig.suptitle("D2: convergencia media y variación entre folds y semillas", y=1.04, fontsize=13, fontweight="bold")
        fig.text(.5,.01,"Cada panel promedia 15 ejecuciones; a medida que opera early stopping, menos corridas contribuyen a épocas tardías. Sombreado = ±1 DE entre ejecuciones.\nLas pérdidas están normalizadas y difieren entre paneles; no son MAE físico. La línea vertical marca la media de las épocas elegidas solo por inner-validation.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.11,1,.88)); fig.savefig(plotdir/"d2_cv_curvas_entrenamiento.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    # Resumen humano: desviación entre folds promediando semillas y entre
    # semillas promediando folds, para no confundir ambas fuentes de variación.
    comparison_rows = []
    for name in [base_name, "D2-CENTER", "D2-CENTER-ONLY"]:
        f = fold_summary[fold_summary.configuration.eq(name)]
        s = seed_summary[seed_summary.configuration.eq(name)]
        vals = {}
        for metric, short, digits in [("absolute_mae_mev_atom","abs",1),("delta_e_mae_mev_atom","delta",2),
                                      ("ground_state_accuracy","gs",3),("pairwise_accuracy","pair",3)]:
            vals[short] = (f[metric].mean(), f[metric].std(), s[metric].mean(), s[metric].std())
        comparison_rows.append((name, vals))
    metric_table = "| Configuración | MAE absoluto: media ± DE fold / semilla (meV/átomo) | MAE ΔE: media ± DE fold / semilla (meV/átomo) | GS (fracción) | Pairwise (fracción) | Colapsos |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, v in comparison_rows:
        g = summary[summary.configuration.eq(name)]
        metric_table += (f"| {name} | {v['abs'][0]:.1f} ± {v['abs'][1]:.1f} / {v['abs'][2]:.1f} ± {v['abs'][3]:.1f} | "
                         f"{v['delta'][0]:.2f} ± {v['delta'][1]:.2f} / {v['delta'][2]:.2f} ± {v['delta'][3]:.2f} | "
                         f"{v['gs'][0]:.3f} | {v['pair'][0]:.3f} | {int(g.collapsed.sum())}/15 |\n")
    ci_lines=[]
    for variant in ["D2-CENTER","D2-CENTER-ONLY"]:
        pieces=[]
        for metric,label in [("absolute_mae_mev_atom","MAE absoluto"),("delta_e_mae_mev_atom","MAE ΔE"),
                             ("ground_state_accuracy","GS"),("pairwise_accuracy","pares")]:
            item=boot[f"{variant}|{metric}"]
            mult=100 if metric in ("ground_state_accuracy","pairwise_accuracy") else 1
            unit="puntos porcentuales" if mult==100 else "meV/átomo"
            pieces.append(f"{label}: {item['mean_difference']*mult:+.2f} [{item['ci95_percentile'][0]*mult:+.2f}, {item['ci95_percentile'][1]*mult:+.2f}] {unit}")
        ci_lines.append(f"- {variant} − E2-A: " + "; ".join(pieces) + ".")
    time_summary = ""
    time_path = args.output / f"training_time_by_run_{args.phase}.csv"
    if time_path.is_file():
        t=pd.read_csv(time_path)
        total=t.training_wall_seconds.sum()
        time_summary=f"\n\nTiempo acumulado de entrenamiento de las 30 cabezas/último bloque: **{total/60:.1f} min de CPU** (suma de duraciones por ejecución; no incluye preparación/análisis). Media por corrida: **{t.training_wall_seconds.mean():.1f} s**."
    text = f"""# D2 — ¿ayuda entrenar la energía relativa por triplete?

Estado: **PASS científico / PARTIAL de procedencia** para CV. Se completaron 30 ajustes; el smoke fue una comprobación técnica, no evidencia. Se compara con E2-A partial_ft en los mismos folds y semillas; test histórico excluido.

`D2-CENTER` conserva la pérdida absoluta normalizada y añade SmoothL1 de energías centradas por composición, sustituyendo pairwise. `D2-CENTER-ONLY` usa solo la pérdida centrada. Escalas de energía y centrado proceden solo de inner-train; early stopping usa inner-validation. Mantiene MatGPT unconditional, último bloque ajustable, pooling `<end>`, MLP 768→8→1, AdamW, batch de cuatro composiciones, clipping 1.0 y paciencia 20.

La configuración YAML histórica dice 200 épocas, pero el código E2-A ejecutado fija un máximo real de 100 (smoke de 3); D2 sigue el código ejecutado, con 100 máximas. CENTER-ONLY no aprende offsets absolutos entre composiciones; su salida absoluta usa como offset la media inner-train para poder computar MAE, y no se debe interpretar como una predicción absoluta supervisada.

## Resultados pareados

{metric_table}

En la tabla, “DE fold” se calcula sobre cinco medias de fold (cada una promedia tres semillas); “DE semilla” se calcula sobre tres medias de semilla (cada una promedia cinco folds). GS es la fracción macro de composiciones elegibles cuyo mínimo predicho coincide con DFT; se excluye gap DFT ≤1 meV/átomo y una predicción empatada no cuenta como éxito. “Pares” es la exactitud macro de orden de polimorfos, excluyendo pares DFT empatados a ≤1 meV/átomo. Ninguna corrida colapsó según rango predicho medio <1 meV/átomo.

El bootstrap pareado remuestrea las mismas composiciones (3 polimorfos juntos) 2,000 veces; cada composición promedia primero sus tres semillas. Diferencia negativa favorece D2 en error; positiva favorece D2 en GS/pares. IC95% exploratorios:

{chr(10).join(ci_lines)}

Un cambio en GS puede coexistir con ΔE y exactitud pairwise casi iguales: acertar el mínimo de un triplete es una pregunta distinta de reproducir el tamaño de todas sus diferencias. Los intervalos son de desarrollo y no confirman superioridad. No se consultó el test histórico.{time_summary}

En CV, `metrics_mean_by_fold_{args.phase}.csv` promedia las tres semillas dentro de cada fold y `metrics_mean_by_seed_{args.phase}.csv` promedia los cinco folds dentro de cada semilla. La desviación entre folds mide sensibilidad a qué composiciones quedaron fuera; la desviación entre semillas mide variación de inicialización/orden de entrenamiento en los mismos grupos. No se tratan 15 ejecuciones como 15 muestras químicas independientes.

**Nota de procedencia:** la corrida CV ya estaba iniciada cuando se añadieron al archivo del trainer una columna auxiliar de duración por predicción y una protección para evitar sobrescribir una carpeta no vacía. La pérdida, el optimizador, los folds, las semillas y el criterio de checkpoint no cambiaron. El SHA del trainer al inicio no quedó guardado, por lo que el SHA actual del manifiesto identifica la versión presente y no certifica el bytecode exacto cargado por aquel proceso. La limitación está registrada en `integrity_checks_cv.json`; para trazabilidad de código exacta habría que repetir la CV en una nueva carpeta versionada.

El archivo `training_history_{args.phase}.csv` separa objetivo promedio de minilotes inner-train y objetivo completo inner-validation. No son MAE: las curvas están normalizadas y CENTER y CENTER-ONLY contienen distintos términos. La época se selecciona con inner-validation antes de evaluar el fold externo. `training_time_by_run_{args.phase}.csv` contiene el tiempo total medido por ejecución y las épocas completadas.

La pérdida centrada y pairwise contienen información diferencial estrechamente relacionada en el límite cuadrático y de pares completos. Cualquier diferencia práctica puede venir de Huber, escalas, reducción, batching o convergencia; no implica señal física nueva.

**Lectura simple:** CENTER prácticamente no cambia el promedio global frente a E2-A. CENTER-ONLY da una mejora pequeña de GS, mientras MAE absoluto y MAE de ΔE cambian poco; el intervalo de GS apenas queda por encima de cero y otras métricas no muestran una mejora clara. La evidencia justifica conservar CENTER-ONLY como hipótesis de ranking para una comparación confirmatoria, pero no como ganador ni como razón suficiente para E2 más complejo. La validation histórica ya intervino en preselecciones anteriores, por lo que un holdout nuevo sería necesario para una afirmación independiente.

## Cómo leer las figuras

![Errores D2 pareados por composición](figures/d2_cv_error_pareado.png)

- `figures/d2_cv_error_pareado.png`: cada punto es una composición; se promedian sus tres semillas y se comparan los mismos 70 grupos. En la figura izquierda cada coordenada es el MAE absoluto de un triplete; a la derecha, el MAE de las tres ΔE respecto al mínimo DFT. La diagonal marca empate entre métodos. Un punto bajo la diagonal favorece D2 en esa métrica; sobre ella favorece E2-A. La nube de MAE absoluto queda casi encima de la diagonal, lo que muestra que CENTER apenas cambia el error absoluto. En ΔE hay variación por composición y no una mejora uniforme.

![Variación del ranking por fold y semilla](figures/d2_cv_ranking_folds_semillas.png)

- `figures/d2_cv_ranking_folds_semillas.png`: cada punto es una combinación de fold y semilla (15 por modelo). El eje horizontal identifica qué 14 composiciones quedaron fuera en cada fold. El panel GS muestra la selección correcta del polimorfo de menor energía; el panel pairwise muestra la fracción de pares de polimorfos ordenados correctamente. El patrón ayuda a ver si una mejora se repite entre composiciones y semillas. No se dibuja una media de referencia para no ocultar la variación.

![Curvas medias de entrenamiento y validación](figures/d2_cv_curvas_entrenamiento.png)

- `figures/d2_cv_curvas_entrenamiento.png`: la línea azul es el objetivo de minilotes de entrenamiento y la naranja es el objetivo calculado sobre todos los grupos de inner-validation. Cada panel promedia 15 corridas; las bandas son ±1 desviación estándar. Al avanzar épocas, las corridas que ya activaron early stopping desaparecen de la media; el número activo se anota a la derecha. La línea vertical es la media de la mejor época seleccionada por validation. Estas pérdidas son adimensionales, no son errores en eV/átomo, y las escalas cambian entre CENTER y CENTER-ONLY. Una curva azul descendente con naranja plana o creciente indica que seguir optimizando train no mejora la validación.
"""
    (args.output / "D2_report.md").write_text(text)
    LOG.info("d2_summary_written phase=%s compositions=%d", args.phase, comp.composition_group.nunique())


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
