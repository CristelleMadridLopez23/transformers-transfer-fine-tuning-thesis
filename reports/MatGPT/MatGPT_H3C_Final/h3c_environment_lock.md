# Entorno fijado y límites de determinismo

[HECHO VERIFICADO] Python 3.10.21; PyTorch 2.8.0; scikit-learn 1.7.2; NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.2; joblib 1.6.0; plataforma `macOS-26.6.2-arm64-arm-64bit`. Ejecución CPU con dos threads; CUDA y MPS no se usaron. Semillas 42, 123 y 2026.

[HECHO VERIFICADO] Checkpoint MatGPT SHA-256 `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`; cache de features `2f790a30ba0fa3cd961c12de82b65febf0ae651b1f217efd5df49c8c5a9cfb62`. Los hashes de cada Ridge y MLP8 permanecen en sus manifiestos históricos; la reproducción los carga sin sobrescribirlos.

[LIMITACIÓN] Inicializadores, BLAS, serialización y orden de operaciones pueden producir diferencias float32 pequeñas entre plataformas. La tolerancia de reproducción fijada es 1e−5 eV/átomo; en este entorno se observó 7.08e-07. El lock de pesos aleatorios verifica hashes tensoriales y falla si cambia la inicialización.
