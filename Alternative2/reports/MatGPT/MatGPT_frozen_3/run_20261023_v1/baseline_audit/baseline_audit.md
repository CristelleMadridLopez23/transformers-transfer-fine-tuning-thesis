# Auditoría de referencias MatGPT_frozen_3

Estado: **PASS**. Se confirmó el checkpoint E0 congelado, commit histórico, 210 estructuras/70 composiciones de desarrollo, cinco folds y exclusión de 24 estructuras/8 composiciones del test. H2 OOF coincide por ID, fórmula, familia y energía con el dataset; D1 coincide por ID y etiqueta.

Checkpoint SHA-256: `4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414`. Tokenizer/model-info SHA-256: `d0948d7ba6fe010c23b99755d71dc9546558b950509b02a7cccef73eb2f15dee`. Dataset SHA-256: `0719f496c96297761ede3a7e2e7ef57aa7b62400fb0004e679a66430a70c31e8`. Fold manifest SHA-256: `a5f00a89a41e4645a68d1463f26f9eca10a09bdfd1208193a87e541ff7cd5627`. Git HEAD al auditar: `1d643c9b0b3eacf922e93f99576420d38ada68ef`.

H2 y D1 se leen como referencias históricas, no se usan para escoger los parámetros nuevos. D1 es composición-only y queda como control externo de energía absoluta; no discrimina polimorfos. La validation histórica participó en selección previa, así que las nuevas métricas OOF son exploratorias.
