# Auditoría de tokenización basada en MatText y el paper



Estado científico global: **failed**.



Se comprobaron las 8898 filas de only_hex_new. Estados: {'compatible_with_official_code': 8712, 'conversion_rejected': 186}.

Compatibilidad con la implementación oficial histórica: **passed**.

Procedencia de IDs vinculada al checkpoint específico: **not_established**.



## Comprobaciones observadas



Vocabulario original: 156 IDs. Embeddings: [30522, 512]. Especiales: {'unk_token': 149, 'pad_token': 150, 'cls_token': 151, 'sep_token': 152, 'mask_token': 153, 'eos_token': 154, 'bos_token': 155}.

Envoltura reproducida: single_CLS_SEP; longitud máxima observada: 401.

Todas las entradas compatibles se compararon ID por ID y máscara por máscara con __call__ del tokenizador oficial de 2024 ejecutado con Transformers 4.38.2. Se rechazaron omisiones y truncamientos; se comprobó equivalencia de especies, índices, aristas e imágenes periódicas.

Las estructuras rechazadas por SLICES se conservan con su motivo; no se sustituyen ni se etiquetan como problemas del tokenizador.

Particiones compatibles: {'train': 7027, 'val': 932, 'test': 753}. Sin composición compartida entre train/val/test.



## Colisiones y límite de la representación



2853 entradas tokenizadas únicas; 1119 grupos compartidos por varios frames; 0 grupos con polimorfos estáticos de familias distintas.

Grupos de cadenas SLICES distintas que el tokenizador fusiona en los mismos IDs: 0. Se separan de las colisiones ya presentes en la representación de origen.

Límite descriptivo MAE por entradas idénticas en entrenamiento: 0.002063508780062608 eV/átomo. No es una métrica de generalización.

Las colisiones se registran sin deduplicar. Una topología común puede ocultar geometrías y energías distintas a lo largo de una trayectoria; el fine-tuning no puede separar entradas determinísticamente idénticas.



## Procedencia y bloqueos



El paper y su archivo de código permiten reproducir el flujo BERT/SLICES. El checkpoint publicado solo contiene pesos/configuración/model card; no incluye el tokenizador de aquella ejecución. La documentación por defecto usa otro orden de especiales. El archivo finetune2m/slice.yaml incluso apunta a un checkpoint 30k_rt; no se reutiliza esa ruta como prueba del modelo 2m.

- No released original tokenizer or checkpoint-specific published ID/wrapping artifact was found



El fine-tuning permanece bloqueado hasta satisfacer las comprobaciones críticas. Los IDs se guardan explícitamente como candidatos y no se presenta una reconstrucción del método como prueba de un artefacto original inexistente.



Fuentes: [paper](https://doi.org/10.1038/s43588-026-01045-2), [código archivado](https://doi.org/10.5281/zenodo.21151839), [tokenizador histórico](https://github.com/lamalab-org/xtal2txt/blob/1d9d0c6f8675efea885ef7c9c87b3195cd79aaed/src/xtal2txt/tokenizer.py).
