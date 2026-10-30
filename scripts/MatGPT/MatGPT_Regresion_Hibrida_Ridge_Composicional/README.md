# Ridge composicional + residuo MatGPT

Este módulo ajusta cinco Ridge composicionales sobre las fracciones estequiométricas históricas. Reutiliza las predicciones residuales MLP8 Frozen y LoRA ya verificadas. No llama al Transformer ni actualiza los pesos MLP8/LoRA.

Ejecute desde la raíz del repositorio:

```bash
bash sh_files/MatGPT_Regresion_Hibrida_Ridge_Composicional/run.sh
```

El comando registra tiempo, CPU, RSS máximo y log en `Alternative2/reports/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/`. La salida no se sobrescribe. Consulte `config.yaml` y el directorio de reportes para la definición científica, predicciones, validaciones, figuras y hashes.
