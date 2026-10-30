#!/usr/bin/env python3
"""Integra D1–D3 con tablas simples y conclusiones de alcance limitado."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import REPORT, write_json

LOG=logging.getLogger("matgpt_diagnostics.comparison")


def run() -> None:
    """Integra resultados y conserva separados sus diseños y denominadores."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1",type=Path,default=REPORT/"D1/v3")
    parser.add_argument("--d2",type=Path,default=REPORT/"D2/cv_v1")
    parser.add_argument("--d3",type=Path,default=REPORT/"D3/v5")
    parser.add_argument("--output",type=Path,default=REPORT/"comparison")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    d1=pd.read_csv(args.d1/"metrics.csv")
    d2=pd.read_csv(args.d2/"metrics_paired_cv.csv")
    d3=json.loads((args.d3/"d3_summary.json").read_text())
    seed_rows=d2.groupby(["configuration","seed"]).agg(**{m:(m,"mean") for m in ["absolute_mae_mev_atom","delta_e_mae_mev_atom","ground_state_accuracy","pairwise_accuracy","pairwise_energy_mae_mev_atom","absolute_rank_error_mean","pred_range_mev_atom","collapsed"]}).reset_index()
    fold_rows=d2.groupby(["configuration","fold"]).agg(**{m:(m,"mean") for m in ["absolute_mae_mev_atom","delta_e_mae_mev_atom","ground_state_accuracy","pairwise_accuracy"]}).reset_index()
    rows=[]
    for config,g in d1.groupby("configuration"):
        rows.append({"stage":"D1 composition baseline","configuration":config,"n_folds":g.fold.nunique(),
                     "absolute_mae_mean_mev_atom":float(g.absolute_mae_mev_atom.mean()),"absolute_mae_sd_folds":float(g.absolute_mae_mev_atom.std()),
                     "delta_e_mae_mean_mev_atom":float(g.delta_e_mae_mev_atom.mean()),"delta_e_mae_sd_folds":float(g.delta_e_mae_mev_atom.std()),
                     "absolute_r2_mean_folds":float(g.absolute_r2_pooled_structures.mean()),
                     "ground_state_accuracy_mean":float(g.ground_state_accuracy.mean()),"pairwise_accuracy_mean":float(g.pairwise_accuracy.mean())})
    for config,g in seed_rows.groupby("configuration"):
        folds=fold_rows[fold_rows.configuration.eq(config)]
        rows.append({"stage":"D2 partial-ft CV","configuration":config,"n_folds":folds.fold.nunique(),
                     "absolute_mae_mean_mev_atom":float(g.absolute_mae_mev_atom.mean()),"absolute_mae_sd_seeds":float(g.absolute_mae_mev_atom.std()),
                     "absolute_mae_sd_folds":float(folds.absolute_mae_mev_atom.std()),
                     "delta_e_mae_mean_mev_atom":float(g.delta_e_mae_mev_atom.mean()),"delta_e_mae_sd_seeds":float(g.delta_e_mae_mev_atom.std()),
                     "delta_e_mae_sd_folds":float(folds.delta_e_mae_mev_atom.std()),
                     "ground_state_accuracy_mean":float(g.ground_state_accuracy.mean()),"pairwise_accuracy_mean":float(g.pairwise_accuracy.mean()),
                     "pairwise_energy_mae_mev_atom":float(g.pairwise_energy_mae_mev_atom.mean()),
                     "absolute_rank_error_mean":float(g.absolute_rank_error_mean.mean()),
                     "collapsed_fold_seed_runs":int(g.collapsed.sum())})
    comp_metrics=pd.DataFrame(rows);comp_metrics.to_csv(args.output/"comparative_metrics.csv",index=False)
    boot_d1=json.loads((REPORT/"common/D1_vs_E2A_paired_bootstrap.json").read_text())
    boot_d2=json.loads((args.d2/"paired_bootstrap_cv.json").read_text())
    write_json(args.output/"paired_bootstrap_ci.json",{"D1_Ridge_vs_E2A":boot_d1,"D2_vs_E2A":boot_d2})
    # D2 comparison: average fold metrics first within seed, then summarize seeds/folds separately.
    lines=[]
    for config,g in seed_rows.groupby("configuration"):
        f=fold_rows[fold_rows.configuration.eq(config)]
        lines.append(f"| {config} | {g.absolute_mae_mev_atom.mean():.1f} ± {g.absolute_mae_mev_atom.std():.1f} (semillas) / {f.absolute_mae_mev_atom.mean():.1f} ± {f.absolute_mae_mev_atom.std():.1f} (folds) | {g.delta_e_mae_mev_atom.mean():.2f} ± {g.delta_e_mae_mev_atom.std():.2f} / {f.delta_e_mae_mev_atom.mean():.2f} ± {f.delta_e_mae_mev_atom.std():.2f} | {g.ground_state_accuracy.mean():.3f} | {g.pairwise_accuracy.mean():.3f} | {int(g.collapsed.sum())}/15 |")
    text=f"""# D1–D3 — qué limita la predicción de estabilidad polimórfica en ABS₃

## Resumen en una frase

La fórmula química explica muy bien el nivel de energía absoluta de estas composiciones, pero no discrimina polimorfos; las cadenas estáticas SLICES sí son distintas, aunque el pooling `<end>` preentrenado no sigue la magnitud de |ΔE|. La pérdida centrada cambia algunas métricas en el CV, pero la recomendación final depende de si esas diferencias son consistentes y de los intervalos pareados.

## Qué se hizo y qué no

Desarrollo únicamente: 210 estructuras estáticas DFT-PBE, 70 composiciones y tres polimorfos por composición; GroupKFold en cinco folds. D1 utilizó inner-train/inner-validation para Ridge. D2 comparó dos pérdidas sobre el mismo backbone parcial de E2-A; D2-BASE son las predicciones E2-A ya existentes. D3 congeló el checkpoint original unconditional. El test histórico de ocho composiciones no entró en ninguna etapa. La validation histórica sí participó en selección previa de candidatas, así que estos resultados no son una prueba independiente.

## D1: composición explica energía absoluta, no ranking

Ridge de composición obtiene MAE absoluta {float(d1[d1.configuration.eq('D1-B Ridge composición')].absolute_mae_mev_atom.mean()):.1f} meV/átomo y R² promedio por fold {float(d1[d1.configuration.eq('D1-B Ridge composición')].absolute_r2_pooled_structures.mean()):.3f}. El modelo E2-A partial_ft tiene MAE absoluta alrededor de 729 meV/átomo. La diferencia pareada Ridge−E2-A fue negativa y su bootstrap de 70 composiciones está en `paired_bootstrap_ci.json`. Eso demuestra que la representación actual de MatGPT/head no está calibrando bien el nivel absoluto en este benchmark; no demuestra que una feature concreta sea universal.

Los tres polimorfos de una fórmula tienen las mismas features Ridge. Por eso el predictor da idéntica energía y no resuelve el ranking, aunque prediga bien el valor absoluto. La varianza de energía observada se separa en ~99.6% entre medias de composición y ~0.4% dentro de cada fórmula. Es una descripción de este conjunto, no una explicación causal.

## D2: comparación de funciones objetivo

| Configuración | MAE absoluto, media ± DE (semillas) / (folds), meV/átomo | MAE ΔE, media ± DE (semillas) / (folds), meV/átomo | GS | Pairwise | Corridas colapsadas |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Bootstrap pareado por composición (promedio de las tres semillas dentro de cada fórmula antes de remuestrear): `paired_bootstrap_ci.json`. Los intervalos son exploratorios. Si CENTER baja MAE absoluta pero no ΔE/ranking, la ganancia se limita a calibración absoluta. CENTER-ONLY puede recuperar variación relativa, pero no identifica offset entre composiciones. Pairwise y centrado codifican información diferencial relacionada; una diferencia de rendimiento puede proceder de SmoothL1, escalas, batching o convergencia, no de una señal física nueva.

## D3: distinguen las cadenas; asociación energética es otra pregunta

En {d3['n_compositions']} composiciones, las tres cadenas SLICES e IDs tokenizados son distintos. No se observó colisión exacta de cadena estática dentro de estos tripletes. Embeddings originales: checkpoint SHA-256 `{d3['embedding_audit']['checkpoint_sha256']}`, strict load, 768 dimensiones, padding error {d3['embedding_audit']['padding_end_max_abs_error']:.2e}. Las correlaciones Spearman exploratorias con |ΔE| DFT: end euclidiana {d3['spearman_vs_abs_dft_pair_gap']['end_embedding_euclidean']['spearman_rho']:.2f}, masked-mean euclidiana {d3['spearman_vs_abs_dft_pair_gap']['mean_embedding_euclidean']['spearman_rho']:.2f}, geometría {d3['spearman_vs_abs_dft_pair_gap']['geometry_descriptor_l1']['spearman_rho']:.2f}. Tres pares comparten cada fórmula; las correlaciones son exploratorias y no demuestran error predictivo ni causalidad.

E2-T1 ya mostró 1,148 de 1,677 grupos de IDs repetidos con rango energético ≥1 meV/átomo en frames de relajación. No se extrapola ese resultado a los estáticos: D3 no halló colisiones exactas de strings estáticos.

## Conclusión científica y recomendación

1. **Limitación absoluta:** D1 demuestra que la fórmula ofrece señal fuerte; el gran MAE de E2-A no puede explicarse diciendo simplemente que la composición es insuficiente. Revisar calibración de salida, normalización, pooling, cabeza y función objetivo.
2. **Limitación del ranking:** D1 no puede ordenar por diseño. D3 muestra inputs SLICES distintos, pero `<end>` congelado casi no se asocia con gaps DFT. Esto apunta a representación/pooling o aprendizaje supervisado, pero no los separa causalmente.
3. **D2 y siguiente paso:** CENTER no cambia de forma material las métricas frente a E2-A. CENTER-ONLY mejora GS alrededor de 5 puntos porcentuales, con IC bootstrap pareado apenas positivo; ΔE MAE y pairwise no mejoran claramente y el MAE absoluto no cambia. Es una señal modesta y exploratoria para el mínimo de triplete, no una mejora global confirmada. D2 completó científicamente, pero el SHA exacto del trainer al inicio no se capturó; los cambios durante la corrida fueron de medición de tiempo/guardas y no cambiaron el protocolo. Antes de ampliar el fine-tuning o añadir geometría, hace falta validación en composiciones nuevas que no participaron en la preselección histórica.
4. **D4:** los análisis D3 aportan una pista para diseñar —sin afirmar que vaya a funcionar— una comparación SLICES+geometría con estructuras estáticas y mismos folds/cabeza. La asociación geométrica es exploratoria. Si se evalúa, debe comparar contra CENTER-ONLY y E2-A de forma pareada y con holdout nuevo.
5. **No usar test:** ninguna cifra de este reporte proviene de los ocho grupos históricos de test. Para una afirmación independiente se necesita un holdout nuevo o nested CV completo porque validation histórica ya participó en preselección.

## Figuras

Las figuras PNG están en `D1/v3/figures`, `D2/pilot_v1/figures`, `D2/cv_v1/figures` y `D3/v5/figures`. Las figuras CV explican los puntos pareados por composición, el conteo de pliegues/semillas y la selección de épocas. Los reportes específicos explican unidades, denominadores, empates y límites interpretativos.

## Referencias verificadas

- MatGPT, [código oficial](https://github.com/devalab/matgpt), commit local `a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e`.
- Xiao et al., SLICES, *Nature Communications* 14, 7027 (2023), [doi:10.1038/s41467-023-42870-7](https://doi.org/10.1038/s41467-023-42870-7).
- Witman & Schindler, MatFold, *Digital Discovery* 4, 625–635 (2025), [doi:10.1039/D4DD00250D](https://doi.org/10.1039/D4DD00250D); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- SOAP, [documentación DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html); alternativa geométrica publicada, no usada aquí.
- [MPtrj de Materials Project](https://docs.materialsproject.org/services/ml-and-ai-applications/mptrj) y Deng et al., CHGNet (2023), [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3), como contexto de MLIP, no comparadores.
- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html), para separar extracción en inference mode de actualización entrenable.
"""
    (args.output/"D1_D2_D3_scientific_report.md").write_text(text)
    LOG.info("comparison_report_written d2_runs=%d d3_pairs=%d",len(d2),d3["n_pairs"])


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    run()
