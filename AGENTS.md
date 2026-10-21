# Normas operativas del proyecto

- Respetar la estructura de carpetas. No reorganizar, mover ni borrar datos sin autorización.
- Organizar cada experimento mediante un identificador único.
- Guardar scripts Python en `model-pretrained/scripts/<experimento>/` y archivos `.sh`/`.sbatch` en `sh_files/<experimento>/`.
- Guardar gráficos PNG y reportes científicos en `reports/<experimento>/`.
- Mantener checkpoints originales separados de artefactos entrenados; nunca modificar estructuras ni valores DFT originales.
- No ejecutar procesos pesados ni entrenamientos GPU en nodos de login.
- Documentar versiones, semillas, configuraciones, procedencia y resultados reales.
- Evitar fuga entre entrenamiento, validación y prueba. Agrupar polimorfos por composición.
- Exigir auditorías satisfactorias antes del envío de experimentos. Detener fases dependientes ante controles críticos fallidos.
- Nunca reemplazar resultados reales con datos sintéticos sin identificarlos explícitamente. Nunca inventar métricas, archivos de datos, configuraciones usadas ni resultados.
- Cada script Python debe incluir docstring de módulo, funciones documentadas con entradas/tipos/salidas, type hints, validación de argumentos, logging estructurado, `main()` protegido y rutas configurables relativas al proyecto. Registrar errores descriptivos y comentar transformaciones científicas.
- Preservar cambios del usuario; no usar `git reset`, `git clean` ni operaciones destructivas.
