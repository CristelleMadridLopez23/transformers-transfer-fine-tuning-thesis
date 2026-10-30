# Organización canónica del proyecto

La raíz se organiza en `data/`, `embeddings/`, `models/`, `reports/`, `scripts/`, `images/` y `experiments/`.

- Fuentes y datos procesados: `data/`.
- Embeddings: `embeddings/`.
- Implementaciones y modelos base: `models/`.
- Código local: `scripts/`.
- Informes científicos Markdown: `reports/`.
- Figuras PNG por experimento: `images/`.
- CSV, JSON, logs, configuraciones ejecutadas y checkpoints derivados: `experiments/`.

Los datos no se eliminaron. Los 1,087 archivos de resultados que estaban físicamente en `reports/` se movieron a `experiments/` conservando la jerarquía de familia/experimento; sus rutas anteriores son enlaces de compatibilidad. Las imágenes PNG se centralizaron bajo `images/` y las carpetas históricas `figures/` se mantienen como enlaces cuando es posible.

`Alternative2/01_data`, `Alternative2/embeddings`, `Alternative2/models`, `Alternative2/reports` y `Alternative2/scripts` son enlaces a las carpetas raíz para compatibilidad. No deben usarse en código nuevo.

No quedan lanzadores `.sh` ni `.sbatch`; la ejecución local se realiza con Python. Los manifiestos `image_migration_manifest.csv`, `report_artifact_migration_manifest.csv` y `cleanup_manifest.csv` documentan los cambios.
