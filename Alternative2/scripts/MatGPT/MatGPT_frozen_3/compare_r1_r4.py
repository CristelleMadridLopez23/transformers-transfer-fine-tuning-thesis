#!/usr/bin/env python3
"""Produce informes R1-R4 y comparación interpretativa con H0-H2/D1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.common import REPORT_DIR, REPORT_ROOT, ExperimentError, log, write_json


def fmt(value: Any, digits: int = 2) -> str:
    """Formatea números o marca NA de forma legible."""
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _summary() -> pd.DataFrame:
    """Carga la tabla de comparación ya recalculada desde OOF."""
    path=REPORT_DIR/"comparisons"/"comparative_metrics.csv"
    if not path.is_file():raise ExperimentError("Ejecutar evaluate_oof.py primero.")
    return pd.read_csv(path)


def _row(frame: pd.DataFrame, model: str) -> pd.Series:
    """Devuelve exactamente una fila de métrica para modelo."""
    data=frame.loc[frame.model.eq(model)]
    if len(data)!=1:raise ExperimentError(f"Modelo ausente/duplicado en comparative_metrics.csv: {model}")
    return data.iloc[0]


def build_reports() -> dict[str, Any]:
    """Escribe reportes R1/R2/R3/R4 y síntesis científica con métricas OOF."""
    summary=_summary()
    boot=json.loads((REPORT_DIR/"comparisons"/"paired_bootstrap.json").read_text())
    ridge_selection=pd.read_csv(REPORT_DIR/"R1_linear_probes"/"validation_selection.csv")
    r4_selection=pd.read_csv(REPORT_DIR/"R4_layers_pooling"/"validation_selection.csv")
    mlp_history=[]
    for path in list((REPORT_DIR/"R2_head_capacity").glob("*/training_history.csv"))+list((REPORT_DIR/"R3_regularization").glob("*/training_history.csv")):
        h=pd.read_csv(path)
        if len(h):
            for (fold,seed),g in h.groupby(["fold","seed"]):
                best=g.loc[g.inner_validation_objective.idxmin()]
                selected=g.loc[g.epoch.eq(best.epoch)].iloc[0]
                mlp_history.append({"model":path.parent.name,"fold":int(fold),"seed":int(seed),"best_epoch":int(best.epoch),
                    "epochs_run":int(g.epoch.max()),"inner_val_objective":float(best.inner_validation_objective),
                    "train_loss_mu_at_best":float(selected.train_loss_mu),"train_loss_delta_at_best":float(selected.train_loss_delta),
                    "val_loss_mu_at_best":float(selected.inner_val_loss_mu),"val_loss_delta_at_best":float(selected.inner_val_loss_delta),
                    "normalized_objective_gap_at_best":float(selected.inner_validation_objective-(selected.train_loss_mu+selected.train_loss_delta))})
    hist=pd.DataFrame(mlp_history)
    if len(hist):hist.to_csv(REPORT_DIR/"comparisons"/"capacity_regularization_audit.csv",index=False)
    r1_models=["R1-A-ABS","R1-B-MU","R1-C-DELTA","R1-D-COMBINED"]
    r2_models=["R2-SMALL4","R2-REF8"]
    r3_models=["R2-REF8","R3-DROP30","R3-WD1E2","R3-EARLY8"]
    r4_models=["R4-SELECTED-ABSOLUTE","R4-SELECTED-MU","R4-SELECTED-DELTA","R4-SELECTED-2COMP","R4-FIXED-FINAL_END","R4-FIXED-FINAL_MEAN"]
    def report_table(models:list[str])->str:
        lines=["| Modelo | Target | MAE E | MAE μ | MAE δ centrada | MAE ΔE anclada | MAE ΔE own-min | GS | Pairwise | Rango error |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in models:
            if name not in set(summary.model):continue
            r=_row(summary,name)
            lines.append(f"| {name} | {r.target_kind} | {fmt(r.MAE_absolute_macro_mev_atom)} | {fmt(r.MAE_mu_mev_atom)} | {fmt(r.MAE_delta_centered_mev_atom)} | {fmt(r.MAE_delta_GS_anchored_mev_atom)} | {fmt(r.MAE_delta_own_min_legacy_mev_atom)} | {fmt(100*r.GS_accuracy,1)}% ({int(r.GS_n_eligible)}) | {fmt(100*r.pairwise_accuracy,1)}% ({int(r.pairwise_n_eligible)}) | {fmt(r.mean_range_error_mev_atom)} |")
        return "\n".join(lines)
    common="\n\nEnergías en meV/átomo. H0/H1/H2 y las cabezas tienen 3 semillas: la métrica principal promedia predicciones por Structure ID y recalcula cada fórmula una vez. Ridge es determinista y no se duplica artificialmente por semilla. MAE E solo aplica a predictores de energía total; MAE μ y δ se reportan para cabezas de componente. GS excluye gap DFT ≤1 meV/átomo; abstención no es acierto. Pairwise excluye empates DFT de ±1 meV/átomo. Todas las cifras son OOF development; test histórico excluido. La selección previa en validation histórica vuelve exploratorios los intervalos.\n"
    report_specs={"R1_linear_probes":("# R1 — sondas lineales MatGPT",r1_models,
        "R1-A predice energía absoluta desde h_end; R1-B predice μ usando la media de tres embeddings; R1-C predice desviaciones DFT centradas y vuelve a centrar las tres predicciones; R1-D suma sondas lineales independientes de μ y δ. Los cuatro usan StandardScaler de inner-train y Ridge alpha seleccionado por MAE inner-validation. R1-B no discrimina polimorfos; R1-C no predice el nivel absoluto. D1 composición-only es un control externo y no entra en estas ecuaciones."),
        "R2_head_capacity":("# R2 — capacidad de las cabezas",r2_models,
        "R2-small reduce ambas MLP de ancho 8 a ancho 4; R2-reference conserva la arquitectura H2 de ancho 8. Se mantiene el mismo masked-mean para μ, h_end para δ, pérdidas SmoothL1 normalizadas, AdamW, LR, WD, folds, semillas y selección interna. Parámetros entrenables: 6,162 frente a 12,322. H2 histórico se muestra como contexto; la referencia R2-REF8 fue reentrenada con el runner nuevo."),
        "R3_regularization":("# R3 — regularización de las cabezas",r3_models,
        "R2-REF8 actúa como baseline compartido. R3-DROP cambia dropout a 0.30; R3-WD cambia weight decay a 0.01; R3-EARLY mantiene regularización y cambia paciencia de 20 a 8 épocas. Se conserva el criterio H2 de suma de SmoothL1 internas normalizadas para seleccionar checkpoints y no se consulta outer-validation para parar."),
        "R4_layers_pooling":("# R4 — capa y pooling",r4_models,
        "R4 comparó END/MEAN de bloques 3, 6, 9 y 12 en salida cruda pre-ln_f, más controles finales post-ln_f END/MEAN. Ridge alpha y representación se escogieron por separado en inner-validation para E, μ y δ; la evaluación outer incluye solo candidatos elegidos internamente y controles finales fijos. H2/R4-SELECTED-2COMP combina las capas elegidas para μ y δ. La cabeza MLP secundaria R4-B no se ejecutó: no es necesaria para completar el diagnóstico lineal y queda como extensión condicionada." )}
    for folder,(title,models,method) in report_specs.items():
        directory=REPORT_DIR/folder
        directory.mkdir(parents=True,exist_ok=True)
        status="PASS" if all(m in set(summary.model) for m in models) else "PARTIAL"
        content=title+"\n\n**Estado: "+status+".**\n\n"+method+common+"\n## Métricas\n\n"+report_table(models)+"\n\n## Lectura\n\nLas diferencias absolutas y relativas deben leerse juntas. Una predicción mejor de μ no implica mejor ordenamiento de polimorfos. La interpretación por pares y fórmulas, con intervalos pareados, está en el informe combinado.\n"
        (directory/f"{folder.split('_')[0]}_report.md").write_text(content,encoding="utf-8")
        write_json(directory/"status.json",{"status":status,"models":models,"test_used":False})
    # Summarize how often the inner-validation selector chose each R4 feature; do not pick by outer scores.
    chosen=r4_selection.loc[r4_selection.model.isin(["R4-SELECTED-ABSOLUTE","R4-SELECTED-MU","R4-SELECTED-DELTA"])]
    chosen_summary=chosen.groupby(["target_kind","feature_variant"]).size().rename("outer_folds_selected_inner_only").reset_index()
    chosen_summary.to_csv(REPORT_DIR/"R4_layers_pooling"/"selected_features_by_fold.csv",index=False)
    alphas=ridge_selection.loc[ridge_selection.selected.astype(bool)].groupby(["model","alpha"]).size().rename("folds_selected").reset_index()
    alphas.to_csv(REPORT_DIR/"R1_linear_probes"/"selected_alpha_counts.csv",index=False)
    # Versioned report root overview, written only under requested report tree.
    h2=_row(summary,"H2") if "H2" in set(summary.model) else None
    d1=json.loads((REPORT_DIR/"baseline_audit"/"d1_external_control.json").read_text())
    def value(model:str,field:str)->float:
        return float(_row(summary,model)[field])
    def interval(model:str,field:str)->str:
        item=boot.get("comparisons",{}).get(f"{model}_minus_H2",{}).get(field,{})
        if item.get("ci95_low") is None:return "no aplica"
        return f"{item['difference_mean_candidate_minus_reference']:+.1f} [{item['ci95_low']:+.1f}, {item['ci95_high']:+.1f}]"
    interpretation=(
        f"**Acceso al nivel absoluto.** H0 registra MAE {value('H0','MAE_absolute_macro_mev_atom'):.1f} meV/átomo; H2 baja a {value('H2','MAE_absolute_macro_mev_atom'):.1f}. "
        f"R4-SELECTED-ABSOLUTE obtiene {value('R4-SELECTED-ABSOLUTE','MAE_absolute_macro_mev_atom'):.1f} meV/átomo y su diferencia pareada candidato−H2 es {interval('R4-SELECTED-ABSOLUTE','MAE_absolute_mev_atom')} meV/átomo. "
        f"La selección interna eligió L03_MEAN para energía absoluta en los cinco folds. Su ranking baja a GS {100*value('R4-SELECTED-ABSOLUTE','GS_accuracy'):.1f}% y pairwise {100*value('R4-SELECTED-ABSOLUTE','pairwise_accuracy'):.1f}%, frente a {100*value('H2','GS_accuracy'):.1f}% y {100*value('H2','pairwise_accuracy'):.1f}% en H2."
        f"\n\n**Nivel medio y residuo.** R1-B (μ-only) obtiene MAE de media {value('R1-B-MU','MAE_mu_mev_atom'):.1f} meV/átomo y no discrimina polimorfos. R1-C, que estima solo δ, obtiene MAE ΔE anclada {value('R1-C-DELTA','MAE_delta_GS_anchored_mev_atom'):.1f}; H2 obtiene {value('H2','MAE_delta_GS_anchored_mev_atom'):.1f}. En H0, MSE del error de media = {value('H0','MSE_mean_mev2_atom2'):.0f}, frente a MSE residual = {value('H0','MSE_residual_mev2_atom2'):.0f} meV²/átomo²: el error absoluto de H0 viene principalmente del offset medio, mientras que sus diferencias centradas son mucho menores. La identidad MSE se verifica por composición; los MAE no se suman."
        f"\n\n**Capacidad y regularización.** R2-small4 da MAE absoluto {value('R2-SMALL4','MAE_absolute_macro_mev_atom'):.1f} (H2 {value('H2','MAE_absolute_macro_mev_atom'):.1f}) con la mitad de parámetros de cabeza; GS/pairwise son {100*value('R2-SMALL4','GS_accuracy'):.1f}%/{100*value('R2-SMALL4','pairwise_accuracy'):.1f}%. R3-dropout 0.30 llega a GS {100*value('R3-DROP30','GS_accuracy'):.1f}% y pairwise {100*value('R3-DROP30','pairwise_accuracy'):.1f}%, con MAE absoluto {value('R3-DROP30','MAE_absolute_macro_mev_atom'):.1f}. Las diferencias de R2/R3 y sus IC pareados están en `paired_bootstrap.json`; intervalos que cruzan cero no respaldan una mejora estable."
        f"\n\n**Referencia externa.** D1 composición-only obtiene MAE {d1['MAE_absolute_mev_atom']:.1f} meV/átomo en los mismos 210 IDs, pero asigna una sola energía a los tres polimorfos, así que su ranking es indeterminado. D1 no forma parte de las cabezas."
    )
    content=["# MatGPT frozen R1–R4: informe científico comparativo\n\n",
        "## Auditoría y alcance\n\n",
        f"Checkpoint E0 SHA-256: `{json.loads((REPORT_DIR/'baseline_audit/provenance.json').read_text())['checkpoint_sha256']}`; commit local `{json.loads((REPORT_DIR/'baseline_audit/provenance.json').read_text())['source_commit']}`. Development: 210 estructuras/70 composiciones, 3 polimorfos por composición; cinco outer folds y 11 grupos inner-validation dentro de cada outer-train. Test histórico (24 estructuras/8 composiciones) fue excluido antes de extraer features. No se descongeló MatGPT ni se usó geometría.\n\n",
        "Las cabezas H2 y los hiperparámetros/pooling ya fueron investigados con parte de la validation histórica. Por ello, aunque la selección nueva ocurre dentro de cada fold y no consulta outer-validation, las comparaciones son exploratorias y no sustituyen un holdout externo nuevo.\n\n",
        "## Métricas OOF pareadas\n\n",report_table([m for m in summary.model if m in ("H0","H1","H2","R1-A-ABS","R1-D-COMBINED","R2-SMALL4","R2-REF8","R3-DROP30","R3-WD1E2","R3-EARLY8","R4-SELECTED-ABSOLUTE","R4-SELECTED-2COMP","R4-FIXED-FINAL_END","R4-FIXED-FINAL_MEAN")]),"\n\n",
        f"D1 Ridge composition-only obtuvo MAE absoluta {d1['MAE_absolute_mev_atom']:.2f} meV/átomo en los mismos IDs, pero da energía idéntica a los tres candidatos de cada fórmula; ranking indeterminado. No se incorporó a R1 ni a las cabezas.\n\n",
        "## Selecciones internas\n\n",
        "R1 alpha por fold/target: `R1_linear_probes/selected_alpha_counts.csv`. Para R4, resumen de capas/poolings elegidos por outer fold: `R4_layers_pooling/selected_features_by_fold.csv`. La evaluación solo puntúa los candidatos seleccionados en inner-validation y los controles finales fijos; la tabla de heatmap contiene únicamente puntuaciones de inner-validation.\n\n",
        "## Incertidumbre y comparación pareada\n\n",
        "Bootstrap pareado por 70 composiciones, 2,000 remuestras, semillas promediadas por estructura. Las tablas `paired_bootstrap.json` comparan candidatos con H2; las entradas no aplicables a objetivos μ-only o δ-only se dejan NA. Los intervalos no son prueba de superioridad.\n\n",
        "## Sobreajuste y selección de época\n\n",
        "Por corrida se guardan loss train por minibatch, validación interna completa por triplete, escalas train-only, gradientes, época seleccionada y segundos de entrenamiento. La regla de parada y selección común es el objetivo H2 de SmoothL1 composicional + residual, normalizado con estadísticas de inner-train. El CSV `capacity_regularization_audit.csv` permite revisar las brechas en la época elegida; la curva no usa outer folds para detener.\n\n",
        "## Interpretación científica\n\n",
        interpretation,"\n\n",
        "R1 cuantifica accesibilidad lineal de E, μ y δ por separado; R2/R3 prueban capacidad y regularización con un backbone idéntico; R4 determina si una lectura END/MEAN de capas previas cambia los probes. No se selecciona un ganador global por promedios outer. R4-B (MLP sobre capas elegidas) queda opcional y no ejecutado.\n\n",
        "Limitaciones: solo 70 composiciones, selección exploratoria histórica, energía DFT estática total por átomo, posible descalibración del nivel absoluto, estrategias Ridge de dimensión alta, y diferencia de objetivo/dominio frente a NequIX. Durante Ridge se imprimieron advertencias numéricas del solver; los chequeos explícitos confirmaron predicciones finitas para todas las selecciones y folds, pero las sondas lineales deben leerse con cautela. No hay fuerzas, tensiones ni phonons en este experimento.\n\n",
        "## Figuras\n\n",
        f"Se generaron {len(list((REPORT_DIR/'comparisons/figures').glob('*.png')))} PNG. El manifiesto de figuras registra que no se exportó SVG/PDF/EPS/JPG/HTML/diapositivas. Consulte `comparisons/figures/` y la interpretación en `R1_report.md`–`R4_report.md`.\n\n",
        "## Referencias\n\n",
        "- MatGPT (código histórico): https://github.com/devalab/matgpt, commit local fijado.\n- SLICES: Xiao et al. (2023), https://doi.org/10.1038/s41467-023-42870-7.\n- MatFold: Witman & Schindler (2025), https://doi.org/10.1039/D4DD00250D.\n- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.\n- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html.\n- PyTorch hooks/autograd: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html and https://docs.pytorch.org/docs/stable/notes/autograd.html.\n- PBE: Perdew, Burke & Ernzerhof (1996), https://doi.org/10.1103/PhysRevLett.77.3865.\n- Métricas H0–H2 de contexto: `Alternative2/reports/MatGPT_frozen_2/v1/H0_H1_H2_report.md`.\n"]
    (REPORT_DIR/"comparisons"/"R1_R4_scientific_report.md").write_text("".join(content),encoding="utf-8")
    overview=REPORT_ROOT/"README.md"
    overview.write_text("# MatGPT_frozen_3 — R1–R4\n\n"+
        f"Corrida activa: `{REPORT_DIR.name}`. Consulte el [informe científico R1–R4]({REPORT_DIR.relative_to(PROJECT)}/comparisons/R1_R4_scientific_report.md), el [audit de referencia]({REPORT_DIR.relative_to(PROJECT)}/baseline_audit/baseline_audit.md), y los reportes individuales R1, R2, R3 y R4.\n\n"
        "Se excluyó el test histórico. Las figuras son PNG exclusivamente. Los resultados son exploratorios por selección histórica previa.\n",encoding="utf-8")
    statuses={folder:json.loads((REPORT_DIR/folder/"status.json").read_text())["status"] for folder in report_specs}
    write_json(REPORT_DIR/"manifests"/"report_status.json",{"status":"PASS" if all(x=="PASS" for x in statuses.values()) else "PARTIAL",
        "R1_R4":statuses,"test_used":False,"R4_b_optional_mlp_run":False})
    return {"status":"PASS" if all(v=="PASS" for v in statuses.values()) else "PARTIAL","statuses":statuses}


def main()->None:
    """Valida salidas requeridas y redacta reportes por experimento."""
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:
        result=build_reports();log("experiment_reports_written",status=result["status"],**result["statuses"])
    except (ExperimentError,OSError,ValueError,KeyError) as exc:
        log("report_generation_failed",error=str(exc));raise SystemExit(2) from exc


if __name__=="__main__":main()
