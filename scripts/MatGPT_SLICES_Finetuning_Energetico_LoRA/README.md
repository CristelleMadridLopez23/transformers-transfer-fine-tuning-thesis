# Scripts de Experimento C

`experiment.py` verifica el conjunto histórico train/val y entrena una corrida única C0/C1. `lora_matgpt.py` añade adaptadores Q/V a las proyecciones fusionadas de `nn.MultiheadAttention`. `evaluate.py` calcula predicciones de los cinco outer folds después de que se completen todas las corridas; `figures.py` genera doce PNG; `diagrams.py` genera las tres arquitecturas en PNG/SVG y conserva sus fuentes Mermaid. `audit_invariance.py` comprueba que LoRA no altera bloque 3 ni la rama Ridge. `measure_run.py` registra tiempo, CPU, RSS y código de salida.

El módulo reutiliza el cargador oficial local de MatGPT para construir el backbone y cargar el checkpoint con coincidencia estricta. El entrenamiento no exporta ni modifica pesos originales; cada checkpoint guarda solo MLP8, factores LoRA, configuración de normalización y estados de optimizador/azar.

Consulte [`auditoria_viabilidad_finetuning_lora.md`](../../../reports/MatGPT_SLICES_Finetuning_Energetico_LoRA/auditoria_viabilidad_finetuning_lora.md) y [`README.md`](../../../experiments/MatGPT_SLICES_Finetuning_Energetico_LoRA/README.md) para procedencia y comandos.
