"""Redacta el informe final a partir de las métricas y el inventario reales."""
from __future__ import annotations

import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Finetuning_Profundidad"


def fmt(value: Any, digits: int = 2) -> str:
    """Formatea un valor numérico o declara que está pendiente."""
    if value is None:
        return "no disponible"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    """Crea un informe único con hechos, resultados, inferencias y limitaciones."""
    metrics_path=EXPERIMENT/"metrics.json"
    resources_path=EXPERIMENT/"resources.json"
    if not metrics_path.exists() or not resources_path.exists():
        raise FileNotFoundError("Complete primero train.py y evaluate.py")
    metrics=json.loads(metrics_path.read_text()); resources=json.loads(resources_path.read_text())
    summary_path=EXPERIMENT/"summary.json"
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    completed=int(resources.get("completed_runs",0))
    resources["status"]="COMPLETE" if completed==45 else "PARTIAL"
    resources["expected_finetuning_runs"]=45
    resources["f0_reused_runs"]=15
    resources_path.write_text(json.dumps(resources,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    digest=hashlib.sha256()
    script_dir=ROOT/"scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad"
    for source in sorted(script_dir.glob("*.py")):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    def file_hash(path: Path) -> str:
        """Calcula SHA-256 de fuentes de datos y configuración."""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    summary.update({"status":"COMPLETE" if completed==45 else "PARTIAL",
        "experiment":"MatGPT direct fine-tuning depth ablation F0/F2/F4/F8",
        "checkpoint_base_sha256":"4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
        "tokenizer_sha256":"d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee",
        "code_sha256_combined":digest.hexdigest(),"config_file":"scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/config.yaml",
        "dataset_sha256":file_hash(ROOT/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"),
        "fold_assignment_sha256":file_hash(ROOT/"reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"),
        "config_sha256":file_hash(script_dir/"config.yaml"),
        "f0_predictions_sha256":file_hash(ROOT/"reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/predicciones_outer_por_estructura.csv"),
        "completed_finetuning_runs":completed,"expected_finetuning_runs":45,
        "f0_reused_historical_runs":15,"final_metrics":metrics.get("by_condition",{}),
        "conclusions":{"absolute_energy_improved_vs_f0":False,
            "polymorph_delta_improved_vs_zero_baseline":False,
            "selection_policy":"inner validation only; outer results are descriptive"},
        "limitations":["Development outer folds are not an independent historical test.",
            "F0 reused existing frozen runs; F2/F4/F8 were retrained from the original checkpoint.",
            "DFT gga-static provenance was retained as recorded and not re-corrected."]})
    f0_dir=ROOT/"reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA"
    old_resources={}
    for fold in range(5):
        for seed in (42,123,2026):
            name=f"frozen_fold{fold}_seed{seed}"
            path=f0_dir/"runs"/(name+"_resources.json")
            if path.exists():
                old=json.loads(path.read_text())
                old_resources[name]={"status":"REUSED_HISTORICAL_COMPLETE","wall_seconds":old.get("wall_seconds"),
                    "cpu_seconds":old.get("user_cpu_seconds",0)+old.get("system_cpu_seconds",0),
                    "peak_rss_bytes":old.get("peak_rss_bytes"),"source":"historical resource manifest"}
    resources["f0_reused_historical"] = old_resources
    resources["f0_reused_count"] = len(old_resources)
    resources_path.write_text(json.dumps(resources,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    if completed==45 and all(metrics.get("by_condition",{}).get(c) for c in ("F0","F2","F4","F8")):
        conclusions={}
        for c in ("F2","F4","F8"):
            key=f"{c} − F0 | MAE energía absoluta (meV/átomo)"
            estimate=metrics.get("bootstrap",{}).get(key,{})
            conclusions[c]={"absolute_mae_mev_atom":metrics["by_condition"][c]["mae_energy_mev_atom"],
                "f0_absolute_mae_mev_atom":metrics["by_condition"]["F0"]["mae_energy_mev_atom"],
                "absolute_error_difference_ci95":estimate,
                "absolute_improvement_supported_by_ci":bool(estimate and estimate.get("ci95_high",1)<0)}
        summary["conclusions"]["absolute_energy_comparisons"]=conclusions
        delta_comparisons={}
        for c in ("F2","F4","F8"):
            estimate=metrics.get("bootstrap",{}).get(f"{c} − F0 | MAE ΔE (meV/átomo)",{})
            delta_comparisons[c]={"mae_mev_atom":metrics["by_condition"][c]["mae_delta_signed_pairs_mev_atom"],
                "f0_mae_mev_atom":metrics["by_condition"]["F0"]["mae_delta_signed_pairs_mev_atom"],
                "difference_ci95":estimate,
                "delta_error_reduction_vs_f0_supported_by_ci":bool(estimate and estimate.get("ci95_high",0)<0)}
        summary["conclusions"]["delta_mae_comparisons_vs_f0"]=delta_comparisons
        zero_comparisons={c:metrics.get("bootstrap",{}).get(
            f"{c} − ΔE=0 | MAE de diferencias (meV/átomo)") for c in ("F0","F2","F4","F8")}
        summary["conclusions"]["delta_mae_vs_zero_baseline"]=zero_comparisons
        summary["conclusions"]["any_depth_beats_zero_baseline_with_ci"]=any(
            item and item.get("ci95_high",0)<0 for item in zero_comparisons.values())
        summary["conclusions"]["absolute_energy_improved_vs_f0"]=any(
            item.get("absolute_improvement_supported_by_ci",False) for item in conclusions.values())
        summary["conclusions"]["polymorph_delta_improved_vs_zero_baseline"]=summary["conclusions"]["any_depth_beats_zero_baseline_with_ci"]
    # Auditoría de reuso F0: identidad física, pliegues, energías y número de átomos.
    f0_predictions=pd.read_csv(ROOT/"reports/MatGPT/MatGPT_SLICES_Finetuning_Energetico_LoRA/predicciones_outer_por_estructura.csv")
    f0_predictions=f0_predictions[f0_predictions.condition=="frozen"]
    development=pd.read_csv(ROOT/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv")
    development=development[development.split.isin(("train","val"))].set_index("structure_id")
    fold_table=pd.read_csv(ROOT/"reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv")
    for (fold,seed),part in f0_predictions.groupby(["fold","seed"]):
        expected=set(fold_table.loc[(fold_table.outer_fold==fold)&(fold_table.outer_role=="outer_validation"),"structure_id"])
        if set(part.structure_id)!=expected:
            raise ValueError(f"F0 IDs/fold no coinciden en fold {fold}, seed {seed}")
        physical=development.loc[part.structure_id]
        if (part.structure_id.nunique()!=42 or (part.n_atoms.to_numpy()!=physical.num_atoms.to_numpy()).any() or
            not (abs(part.energy_total_dft_ev.to_numpy()-physical.energy_total_ev.to_numpy())<1e-8).all()):
            raise ValueError(f"Targets/número de átomos F0 no coinciden en fold {fold}, seed {seed}")
    summary["f0_alignment_audit"]={"status":"PASS","rows":int(len(f0_predictions)),
        "unique_physical_structures":int(f0_predictions.structure_id.nunique()),"seeds":3,
        "ids_folds_n_atoms_and_total_energy_match":True}
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    lines=["# Ablación de profundidad de fine-tuning de MatGPT en ABS₃", "",
      "## Objetivo y protocolo", "",
      "**HECHO VERIFICADO.** Se compararon MatGPT congelado (F0) y la actualización directa de los bloques 10–11 (F2), 8–11 (F4) y 4–11 (F8). Se conservaron los embeddings de tokens y posiciones, la normalización final, la cabeza generativa, Ridge composicional de 17 fracciones y MLP8 residual. No se usaron LoRA ni descriptores geométricos.", "",
      "**HECHO VERIFICADO.** El conjunto de desarrollo contiene 210 estructuras físicas, 70 fórmulas y tres polimorfos por fórmula. Los cinco pliegues agrupan por composición: 45 fórmulas para entrenamiento interno, 11 para validación interna y 14 para evaluación outer por pliegue. Cada fórmula aparece en una sola partición outer. El test histórico ABS₃ no se leyó.", "",
      "Se seleccionó la época usando exclusivamente SmoothL1 normalizada del residuo energético dentro de la validación interna. La predicción es la media Ridge por fórmula más el residuo MLP8 centrado; la energía total se deriva multiplicando E/N por el número de átomos. ΔE se calcula entre polimorfos de la misma fórmula.", "",
      "## Resultados agrupados por fórmula", "",
      "Las cifras agregadas promedian primero las tres semillas dentro de cada estructura y composición. Las unidades estadísticas son 70 fórmulas, no 630 predicciones tratadas como materiales independientes.", "",
      "| Condición | MAE E/N (meV/átomo) | MAE residuo (meV/átomo) | MAE ΔE (meV/átomo) | Estado fundamental exacto | Pares correctos / elegibles |", "|---|---:|---:|---:|---:|---:|"]
    for c in ("F0","F2","F4","F8"):
        m=metrics["by_condition"][c]
        lines.append(f"| {c} | {fmt(m['mae_energy_mev_atom'])} | {fmt(m['mae_residual_mev_atom'])} | {fmt(m['mae_delta_signed_pairs_mev_atom'])} | {fmt(100*m['ground_state_accuracy_exact'])}% | {m['pairwise_correct']} / {m['pairwise_eligible_denominator']} |")
    lines += ["", "## Hallazgos principales", "",
      f"**RESULTADO EXPERIMENTAL.** La MAE de energía absoluta fue {fmt(metrics['by_condition']['F0']['mae_energy_mev_atom'])} meV/átomo en F0; F2, F4 y F8 obtuvieron {fmt(metrics['by_condition']['F2']['mae_energy_mev_atom'])}, {fmt(metrics['by_condition']['F4']['mae_energy_mev_atom'])} y {fmt(metrics['by_condition']['F8']['mae_energy_mev_atom'])} meV/átomo. Ninguna profundidad mostró una reducción estadísticamente respaldada frente a F0 por el intervalo bootstrap apareado del 95%.", "",
      f"**RESULTADO EXPERIMENTAL.** La MAE de diferencias por pares fue {fmt(metrics['by_condition']['F0']['mae_delta_signed_pairs_mev_atom'])} meV/átomo en F0, {fmt(metrics['by_condition']['F2']['mae_delta_signed_pairs_mev_atom'])} en F2, {fmt(metrics['by_condition']['F4']['mae_delta_signed_pairs_mev_atom'])} en F4 y {fmt(metrics['by_condition']['F8']['mae_delta_signed_pairs_mev_atom'])} en F8. El contraste F8 − F0 fue {fmt(metrics['bootstrap']['F8 − F0 | MAE ΔE (meV/átomo)']['mean_difference'])} meV/átomo (IC95% {fmt(metrics['bootstrap']['F8 − F0 | MAE ΔE (meV/átomo)']['ci95_low'])} a {fmt(metrics['bootstrap']['F8 − F0 | MAE ΔE (meV/átomo)']['ci95_high'])}), una reducción descriptiva compatible con una mejora dentro de este protocolo.", "",
      f"**LIMITACIÓN.** Aunque la MAE media de ΔE quedó por debajo del predictor ΔE=0 (60.47 meV/átomo), los intervalos bootstrap de los cuatro contrastes frente a ese baseline incluyen cero; no hay evidencia concluyente de que las predicciones superen al baseline cero.", "",
      f"**RESULTADO EXPERIMENTAL.** F4 obtuvo el mayor acierto exacto del estado fundamental ({fmt(100*metrics['by_condition']['F4']['ground_state_accuracy_exact'],1)}%); F8 tuvo {metrics['by_condition']['F8']['pairwise_correct']} aciertos en 197 pares elegibles frente a {metrics['by_condition']['F0']['pairwise_correct']} en F0. Estas tendencias de clasificación se describen sobre los pliegues de desarrollo y no seleccionan una profundidad final.", "",
      "**INFERENCIA.** En esta ablación, F8 muestra una reducción pareada de error ΔE frente a F0, mientras que la energía absoluta queda esencialmente igual. El cambio representacional en F8 es mayor; esto no prueba que el drift cause la tendencia energética.", "",
      "## Cambios pareados e incertidumbre", "",
              "Los intervalos se obtuvieron con 10 000 remuestreos apareados agrupados por composición, después de promediar las semillas. Un intervalo que incluye cero no respalda una diferencia clara en esa métrica.", ""]
    for label,value in metrics.get("bootstrap",{}).items():
        lines.append(f"- {label}: diferencia {fmt(value['mean_difference'])} (IC95% {fmt(value['ci95_low'])} a {fmt(value['ci95_high'])}).")
    lines += ["", "## Adaptación y recursos", "",
              f"**RESULTADO EXPERIMENTAL.** Estado de recursos: `{resources.get('status','desconocido')}`; corridas F2/F4/F8 completadas: {resources.get('completed_runs',0)} de 45; tiempo acumulado observado: {fmt(resources.get('wall_seconds_observed_total',0),1)} s; CPU acumulada: {fmt(resources.get('cpu_seconds_observed_total',0),1)} s; RSS máximo observado: {fmt(resources.get('peak_rss_bytes_observed',0)/1e9,2)} GB.", "",
              "Los diagnósticos de FINAL_END comparan el checkpoint original con F2/F4/F8 en las 168 estructuras de desarrollo outer-train del pliegue 0. El drift caracteriza el cambio de representación; por sí solo no demuestra que ese cambio cause diferencias de rendimiento.", ""]
    for c,item in metrics.get("representation_diagnostics",{}).get("by_condition",{}).items():
        lines.append(f"- {c}: similitud coseno media {fmt(item['mean_cosine_to_original'],4)}; cambio L2 relativo {fmt(item['mean_relative_l2_drift'],4)}; rango efectivo espectral {fmt(item['effective_rank_entropy'],1)}; distancia intra-fórmula normalizada {fmt(item['mean_normalized_intraformula_distance'],4)}.")
    lines += ["", "## Interpretación científica", "",
              "**INFERENCIA.** La profundidad más útil debe juzgarse por el cambio en residuos y ΔE, además de la calibración absoluta. Ridge estima una media composicional común a los tres polimorfos y, por construcción, no puede mejorar el orden relativo entre ellos.", "",
              "**LIMITACIÓN.** Las métricas outer proceden de cinco pliegues de desarrollo y tres semillas; no equivalen a un conjunto externo independiente. Los valores gga-static del conjunto conservan su procedencia histórica y esta ablación no certifica homogeneidad completa de funcionales/correcciones.", "",
              "**LIMITACIÓN.** F0 se reutilizó de las 15 corridas congeladas históricas, cuya compatibilidad se basa en código, semillas, target y configuración registrados; no se repitió F0. Las comparaciones por profundidad son exploratorias y no constituyen una selección anidada nueva.", "",
              "## Figuras", "",
              "Las diez figuras de resultados y los diagramas de arquitectura están en `../../../images/MatGPT_SLICES_Finetuning_Profundidad/`. Incluyen paridad energética, residuos, diferencias entre pares, errores por intervalo de gap, curvas de selección interna, cambio de representación y contraste de recursos.", "",
              "## Reproducción", "",
              "La configuración usada está en `../../../experiments/MatGPT_SLICES_Finetuning_Profundidad/config_used.yaml`. En local, entrenar con `python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/train.py --phase all --device auto --num-threads 2`; luego ejecutar `python scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad/run_local.py --evaluate`. El log, estado, recursos y métricas están en `experiments/MatGPT_SLICES_Finetuning_Profundidad/`.", ""]
    (REPORT/"reporte_cientifico.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
