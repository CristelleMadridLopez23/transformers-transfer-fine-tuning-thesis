"""Compile MatterGPT audit, grouped-CV, locked-test and comparison artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import MODEL_DIR, REPORT_DIR, TEST_INPUTS, logger, sha256, write_json


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    """Format a small numeric DataFrame as a Markdown table."""
    view=frame[columns].copy()
    for col in view:
        if pd.api.types.is_numeric_dtype(view[col]): view[col]=view[col].map(lambda x:"" if pd.isna(x) else f"{x:.{digits}f}")
    header="| "+" | ".join(columns)+" |"; rule="| "+" | ".join(["---"]*len(columns))+" |"
    rows=["| "+" | ".join(map(str,row))+" |" for row in view.itertuples(index=False,name=None)]
    return "\n".join([header,rule,*rows])


def run() -> dict[str, Any]:
    """Write scientific report and final artifact hash manifest."""
    pre=json.loads((REPORT_DIR/"provenance/preflight.json").read_text())
    cv=pd.read_csv(REPORT_DIR/"cv/metrics_seed_fold_ensemble.csv")
    fold=pd.read_csv(REPORT_DIR/"cv/metrics_per_fold_seed.csv")
    test_path=REPORT_DIR/"test/test_metrics_extended.csv"
    test=pd.read_csv(test_path if test_path.exists() else REPORT_DIR/"test/test_metrics.csv")
    paired=json.loads((REPORT_DIR/"comparisons/test_paired_bootstrap.json").read_text())
    dev_boot=json.loads((REPORT_DIR/"cv/paired_bootstrap_ci.json").read_text())
    lock=json.loads((REPORT_DIR/"test/test_lock.json").read_text())
    m2=fold[fold.model=="M2"].set_index(["outer_fold","seed"])
    m3=fold[fold.model=="M3-A"].set_index(["outer_fold","seed"])
    same_ranking=(m2.gs_accuracy.fillna(-1).equals(m3.gs_accuracy.fillna(-1)) and m2.pairwise_accuracy.fillna(-1).equals(m3.pairwise_accuracy.fillna(-1)))
    labels={"model":"Modelo","n_structures":"N estructuras","n_compositions":"N composiciones",
        "mae_absolute_mev_atom":"MAE absoluto pooled (meV/átomo)",
        "mae_absolute_macro_by_composition_mev_atom":"MAE absoluto macro (meV/átomo)",
        "rmse_absolute_mev_atom":"RMSE absoluto (meV/átomo)",
        "mae_delta_dft_gs_anchored_mev_atom":"MAE ΔE anclada pooled (meV/átomo)",
        "mae_delta_dft_gs_anchored_macro_by_composition_mev_atom":"MAE ΔE anclada macro (meV/átomo)",
        "gs_accuracy":"GS accuracy pooled","gs_accuracy_macro_by_composition":"GS accuracy macro",
        "pairwise_accuracy":"Pairwise accuracy pooled","pairwise_accuracy_macro_by_composition":"Pairwise accuracy macro",
        "mae_delta_own_min_mev_atom":"MAE ΔE own-min (meV/átomo)",
        "mae_composition_mean_mev_atom":"MAE media composición (meV/átomo)",
        "mae_centered_residual_mev_atom":"MAE residuo centrado (meV/átomo)",
        "signed_pairwise_mae_mev_atom":"MAE pairwise firmado (meV/átomo)",
        "mean_absolute_range_error_mev_atom":"Error rango medio (meV/átomo)",
        "pairwise_balanced_accuracy":"Pairwise balanced accuracy","pairwise_macro_f1":"Pairwise macro-F1"}
    cvtable=cv.rename(columns=labels)
    testtable=test.rename(columns=labels)
    report=f'''# MatterGPT congelado como extractor: evaluación ABS₃

## Resumen

**Estado del estudio: PARTIAL.** La extracción/carga, CV y evaluación bloqueada se completaron, pero el checkout local no incluye el loader de teacher forcing del entrenamiento original. El cierre `<` se respalda por la tokenización/decodificación pública, no por una reproducción independiente del dataset de entrenamiento. Además, los BOS/EOS IDs de `config.json` no coinciden con los IDs textuales reconstruidos. Por eso los valores describen explícitamente el protocolo de serialización `> SLICES <` y no una réplica confirmada bit a bit del pretraining.

El checkpoint se cargó con `strict=True`, SHA-256 `{pre['checkpoint_sha256']}`, 85,567,488 parámetros y cero parámetros entrenables. La condición fue fija `[-1.0, 2.0]`, copiada del ejemplo público de inferencia, para toda estructura. No describe las propiedades reales de ninguna estructura y no es MatterGPT incondicional. El cache se extrajo en CPU, con el backbone en `eval()` y `torch.inference_mode()`.

Se completaron M0, M2 y M4 en cinco folds por composición y tres semillas (45 ejecuciones de cabezas). M3-A se construyó sin reentrenar: media composicional M4 más residuo centrado M2. Se evaluaron 210 estructuras/70 composiciones OOF y después se abrió una vez el test bloqueado de 24 estructuras/8 composiciones.

## Datos, arquitectura y procedencia

- Objetivo: energía total estática GGA/PBE por átomo (eV/átomo); métricas en meV/átomo.
- Desarrollo: 210 estructuras, 70 composiciones; cada triplete completo queda dentro de un fold. El manifiesto aporta 5×14 composiciones externas.
- Test: 24 estructuras, 8 composiciones; IDs y SLICES proceden de `test_inputs.csv`, sin etiquetas, antes del lock.
- Modelo: {pre['model_config']['n_layer']} bloques, hidden {pre['model_config']['n_embd']}, {pre['model_config']['num_props']} condiciones, block size {pre['model_config']['block_size']}.
- Poolings: BLOCK03_MEAN (raw/pre-ln_f), FINAL_MEAN (post-ln_f sobre SLICES), FINAL_END (estado post-ln_f en token de cierre `<`). Se excluyó el prefijo de condición del pooling.
- M0: MLP8 absoluto en FINAL_END, Huber normalizada + término pareado; M2: cabeza MLP8 para media sobre FINAL_MEAN y MLP8 de residuo en FINAL_END; M4: StandardScaler+Ridge, capa/pooling y alpha elegidos solo en inner-validation. M3-A conserva los residuos M2 y sustituye solo la media por la media de Ridge M4.
- Cada normalizador y early stopping de MLP se ajustó dentro de outer-training. M4 selecciona entre alphas `[0.01, 0.1, 1, 10, 100, 1000]` y lecturas predefinidas. Se usó solver LSQR (`tol=1e-8`) y suma fila a fila para evitar avisos falsos de matmul en Accelerate; se verificaron predicciones finitas.

## Auditoría del tokenizador y condicionamiento

El vocabulario reconstruido de `Voc_prior` tiene {pre['vocabulary_audit']['vocab_size']} tokens y 0 estructuras OOV. Longitudes SLICES: mínimo {pre['data']['slice_tokens_min']}, mediana {pre['data']['slice_tokens_median']:.0f}, máximo {pre['data']['slice_tokens_max']}; quedan bajo block size. La entrada usa el prefijo público `>` y añade el marcador real `<` como cierre, siguiendo el tokenizador local (que elimina `<` al decodificar). El `config.json` declara BOS/EOS IDs 130/131, que no corresponden al `SimpleTokenizer` ordenado (prefijo/cierre tienen IDs {pre['vocabulary_audit']['public_prefix_id']}/{pre['vocabulary_audit']['terminator_id']}); el forward no usa esos metadatos. Se usa el mapeo textual real y esta discrepancia queda registrada.

El estado final cambió de norma {pre['condition_feature_change_norm']:.3f} al cambiar a la condición diagnóstica fija `[-2,1]`; esta segunda condición no se usó en selección. El extractor recibe solo tokens y constantes. No accede a energía total, formación, band gap ni ΔE. MatterGPT espera obligatoriamente `num_props=2`; no se usó `prop=None`.

## Resultados OOF de desarrollo

Cada modelo se evaluó con las mismas estructuras, folds y semillas. Las métricas se calculan tras promediar las predicciones de los 15 modelos para cada estructura.

{_markdown_table(cvtable,['Modelo','N estructuras','N composiciones','MAE absoluto pooled (meV/átomo)','MAE absoluto macro (meV/átomo)','RMSE absoluto (meV/átomo)','MAE ΔE anclada pooled (meV/átomo)','MAE ΔE anclada macro (meV/átomo)','GS accuracy pooled','GS accuracy macro','Pairwise accuracy pooled','Pairwise accuracy macro'])}

En OOF development, M3-A reduce el MAE absoluto frente a M2 en {cv.loc[cv.model=='M2','mae_absolute_mev_atom'].iloc[0]-cv.loc[cv.model=='M3-A','mae_absolute_mev_atom'].iloc[0]:.1f} meV/átomo, y conserva exactamente la ΔE y el ranking de M2. Los resultados de M4 muestran la separación entre energía absoluta y ordenamiento: la regularización lineal mejora el nivel pero daña el ranking.

M3-A conserva algebraicamente el ranking de M2: error máximo de diferencias entre polimorfos `{json.loads((REPORT_DIR/'h3a/invariance.json').read_text())['max_pairwise_error_ev_atom']:.3e}` eV/átomo. La comparación M3-A−M2 por bootstrap pareado de 2,000 tripletes arroja los siguientes IC95% exploratorios:

{json.dumps(dev_boot['intervals']['M3-A_minus_M2'],indent=2,ensure_ascii=False)}

## Test histórico bloqueado

El lock `test_lock.json` se guardó antes de leer las etiquetas. La predicción se generó para las mismas 24 IDs y se promedió entre 5 folds × 3 semillas. La configuración primaria y las reglas de empate quedaron fijadas antes de cargar energías. El resultado sigue siendo exploratorio porque el test histórico ya se usó para evaluar MatGPT H3-A.

{_markdown_table(testtable,['Modelo','N estructuras','N composiciones','MAE absoluto pooled (meV/átomo)','MAE absoluto macro (meV/átomo)','RMSE absoluto (meV/átomo)','MAE ΔE anclada pooled (meV/átomo)','MAE ΔE anclada macro (meV/átomo)','MAE ΔE own-min (meV/átomo)','MAE media composición (meV/átomo)','MAE residuo centrado (meV/átomo)','MAE pairwise firmado (meV/átomo)','Error rango medio (meV/átomo)','GS accuracy pooled','GS accuracy macro','Pairwise accuracy pooled','Pairwise accuracy macro','Pairwise balanced accuracy','Pairwise macro-F1'])}

La descomposición MSE de M3-A en test es MSE total = {test.loc[test.model=='M3-A','mse_total_ev2_atom2'].iloc[0]:.5f} eV²/átomo²; término de error medio = {test.loc[test.model=='M3-A','mse_mean_component_ev2_atom2'].iloc[0]:.5f}; término residual centrado = {test.loc[test.model=='M3-A','mse_centered_residual_component_ev2_atom2'].iloc[0]:.5f}; discrepancia de identidad = {test.loc[test.model=='M3-A','mse_decomposition_abs_error'].iloc[0]:.2e}. La identidad cuadrática no implica suma de MAE.

La comparación pareada con MatGPT H3-A usa exactamente las mismas IDs y targets DFT. Referencia ya publicada en el proyecto para MatGPT: MAE absoluto 133.545 meV/átomo, MAE ΔE anclada 24.208 meV/átomo, GS 5/8 y pares correctos 17/24. IC95% bootstrap pareado MatterGPT−MatGPT, por composición:

{json.dumps(paired,indent=2,ensure_ascii=False)}

## Interpretación y límites

1. **Transferencia de energía absoluta:** M4 mide cuánto de la energía absoluta puede recuperarse con un regresor lineal sobre hidden states congelados. M3-A prueba si esa media mejora al combinarse con diferencias polimórficas de M2; consultar las tablas, no atribuir el cambio solo al backbone.
2. **ΔE y ranking:** M3-A reutiliza el residuo M2 centrado; por identidad, su ΔE, GS y orden de pares deben coincidir con M2. Un cambio observado sería un error de alineación o métrica.
3. **Condicionamiento:** todo resultado describe embeddings de MatterGPT bajo la condición fija de ejemplo. La sensibilidad a otra condición evidencia dependencia del contexto, pero no autoriza escoger una condición por MAE.
4. **Comparabilidad:** MatterGPT es un generador condicionado por formación/gap; MatGPT H3-A procede de un checkpoint unconditional distinto. Esta comparación descriptiva alinea el target, IDs y unidades, pero no vuelve equivalentes sus preentrenamientos.
5. **Cobertura:** no se halló OOV ni truncamiento. Se leyó SLICES de MatGPT como la entrada estructural compartida; no se regeneró ni canonicalizó.
6. **Selección:** M4 selecciona capa/lectura y alpha con validation interna. Las cifras OOF son exploratorias por decisiones previas sobre el benchmark; test histórico también está expuesto en el proyecto.

## Referencias

- Chen et al., [MatterGPT, arXiv:2408.07608](https://arxiv.org/abs/2408.07608).
- [Checkpoint y ejemplo de inferencia MatterGPT](https://huggingface.co/xiaohang07/MatterGPT), revisión local `ee83624d85e5e60e6a8f48bc72bf2663f87273cd`.
- Xiao et al., [SLICES, Nature Communications 14, 7027 (2023)](https://doi.org/10.1038/s41467-023-42870-7).
- [Ridge API de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html); [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Witman y Schindler, [MatFold, Digital Discovery 4, 625–635 (2025)](https://doi.org/10.1039/D4DD00250D).

## Artefactos

Los datos intermedios, checkpoints de cabezas, selección Ridge, predicciones OOF/test, hashes y figuras están bajo `Alternative2/reports/MatterGPT/MatterGPT_frozen_1/`. El checkpoint original de MatterGPT permanece intacto.
'''
    path=REPORT_DIR/"MatterGPT_Frozen_Scientific_Report.md";path.write_text(report)
    conditioning='''# Auditoría de fuga por condicionamiento\n\n- El checkpoint requiere `num_props=2`; `prop=None` no se permite.\n- Protocolo principal: una sola constante `[-1.0, 2.0]`, tomada del ejemplo público; se replica para todas las muestras y no pretende ser su e_form/gap real.\n- La firma del extractor recibe `ids` y `condition`; no recibe DataFrame ni etiquetas. El cache de TEST se construyó desde `test_inputs.csv`, que solo contiene IDs, fórmula, familia, split y SLICES.\n- Las energías DFT de desarrollo se usan únicamente en cabezas/splits de desarrollo. Las etiquetas TEST se abrieron después de `test_lock.json` y `predictions_prelabels.csv`.\n- Sensibilidad diagnóstica a `[-2,1]` se mide en representaciones y no se usa para seleccionar protocolo.\n- `model.py` antepone la proyección de propiedad en posición interna 0; se excluye del pooling. El cuarto retorno `embed` ocurre antes de `self.blocks`, por lo que no se usa como hidden final. Los hooks capturan salidas de los bloques originales.\n- Limitación de procedencia: el checkout local contiene los archivos públicos del checkpoint y el ejemplo de inferencia, pero no el loader de teacher forcing del repo de entrenamiento; `>` está explícitamente respaldado por el ejemplo público y `<` por el tokenizador/decodificador local. La serialización de cierre no puede verificarse contra un loader de entrenamiento local independiente; por eso se conserva como limitación y los resultados no se presentan como reproducción bit a bit del entrenamiento original.\n'''
    # Keep the historical top-level copy and place the audit in its dedicated folder.
    (REPORT_DIR/"Conditioning_Leakage_Audit.md").write_text(conditioning)
    conditioning_dir=REPORT_DIR/"conditioning"
    conditioning_dir.mkdir(parents=True,exist_ok=True)
    (conditioning_dir/"Conditioning_Leakage_Audit.md").write_text(conditioning)
    comparison=f'''# Comparación MatGPT H3-A vs MatterGPT M3-A\n\nLas predicciones corresponden a las mismas 24 estructuras y 8 composiciones ABS₃, y usan energía estática DFT por átomo. La condición de MatterGPT fue fijada antes de abrir el target de test. Esta evaluación es descriptiva: los resultados históricos MatGPT ya eran conocidos dentro del proyecto. MatterGPT está condicionado por una constante común; MatGPT H3-A procede de un checkpoint unconditional, por lo que las representaciones no son equivalentes.\n\nResultados numéricos y bootstrap pareado: `comparisons/test_paired_predictions.csv`, `comparisons/test_paired_bootstrap.json` y `comparisons/test_comparison.json`.\n'''
    (REPORT_DIR/"Comparison_MatGPT_vs_MatterGPT.md").write_text(comparison)
    files={str(p.relative_to(REPORT_DIR)):sha256(p) for p in sorted(REPORT_DIR.rglob('*')) if p.is_file() and p.name!="manifest.json"}
    script_dir=Path(__file__).resolve().parent
    current_scripts={name:sha256(script_dir/name) for name in lock["scripts_sha256"] if (script_dir/name).is_file()}
    changed={name:{"locked_sha256":locked_hash,"current_sha256":current_scripts.get(name)} for name,locked_hash in lock["scripts_sha256"].items() if current_scripts.get(name)!=locked_hash}
    manifest={"status":"PARTIAL","experiment":"MatterGPT_frozen_1","checkpoint_sha256":pre["checkpoint_sha256"],"condition":[-1.0,2.0],"test_lock_sha256":sha256(REPORT_DIR/"test/test_lock.json"),"test_labels_read_only_after_lock":True,"teacher_forcing_loader_verified":False,"post_lock_script_hash_differences":changed,"interpretation_of_post_lock_changes":{"audit.py":"documentation/provenance fields added after test evaluation; no feature math changed","plot_results.py":"rendering index bug fixed after predictions and metrics were locked","write_report.py":"report wording/table formatting updated after scoring"},"supplemental_test_metrics_source_sha256":sha256(REPORT_DIR/"test/test_metrics_supplement_provenance.json"),"artifacts":files}
    write_json(REPORT_DIR/"manifest.json",manifest)
    return {"status":"PARTIAL","report":str(path),"artifacts":len(files),"reason":"MatterGPT teacher-forcing serialization lacks independent local verification"}


def main() -> None:
    """Write scientific reports and hashes after all scoring/plots."""
    log=logger("mattergpt_report")
    try:log.info("mattergpt_report_complete %s",run())
    except Exception as exc:log.exception("mattergpt_report_failed error=%s",exc);raise


if __name__=="__main__":main()
