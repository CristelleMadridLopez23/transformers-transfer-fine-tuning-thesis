# Orientación del proyecto

Las normas operativas canónicas están en [AGENTS.md](AGENTS.md).
La documentación y comandos locales de cada experimento se mantienen en
`experiments/<experimento>/README.md`; sus informes, en `reports/`; figuras PNG,
en `images/`; y métricas, tablas, manifiestos, logs y checkpoints derivados, en
`experiments/`. Los datos fuente, embeddings y modelos base se guardan en sus
carpetas homónimas de la raíz. Las rutas heredadas `Alternative2/*` son enlaces
de compatibilidad; el código nuevo debe usar las carpetas raíz.
