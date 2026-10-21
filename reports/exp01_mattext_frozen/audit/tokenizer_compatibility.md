# Compatibilidad SLICES / MatText

Estado: **failed**. 225 estructuras auditadas; 225 entradas candidatas sin pérdida de caracteres no blancos.

La fuente previa usó SLICES 2.0.12, econnn, relax_model=None. El adaptador cambia exclusivamente cada vector periódico compacto a tres componentes. Se comparan especies, índices, aristas y vectores de todas las filas. Comprobación adicional de tres grafos con API oficial: True.

Vocabulario candidato: 156 IDs; matriz: 30522 filas. Que los IDs quepan en la matriz no demuestra su correspondencia semántica. La documentación por defecto y la configuración archivada añaden tokens especiales distintos. El checkpoint no incluye un tokenizador serializado. No se amplía el vocabulario ni se altera ninguna matriz de embeddings.

Tokens especiales candidatos: `{'[UNK]': 149, '[PAD]': 150, '[CLS]': 151, '[SEP]': 152, '[MASK]': 153, '[EOS]': 154, '[BOS]': 155}`. Longitud máxima candidata: 352; colisiones de IDs candidatos entre polimorfos: 0.

Las colisiones se conservan en polymorph_collision_audit.csv con diferencias GGA. Igual bolsa de tokens no implica igual entrada para un encoder posicional.

Ejemplo original:
```text
S S S S S S S S S S S S Ge Ge Ge Ge Ba Ba Ba Ba 0 13 ooo 0 16 ooo 0 17 ooo 0 18 ooo 0 19 oo- 1 13 -oo 1 14 --o 1 16 -oo 1 18 ooo 2 12 ooo 2 16 -+o 2 17 ooo 2 18 o+o 2 19 -o- 3 12 o-+ 3 16 -o+ 3 17 o-o 3 18 ooo 3 19 --o 4 15 oo+ 4 16 -o+ 4 17 ooo 4 18 ooo 4 19 -oo 5 13 ooo 5 16 oo+ 5 17 ooo 5 18 ooo 5 19 ooo 6 14 ooo 6 16 o+o 6 17 ooo 6 18 o+o 6 19 oo- 7 15 ooo 7 16 -oo 7 17 ooo 7 18 ooo 7 19 -o- 8 12 oo+ 8 15 oo+ 8 17 ooo 8 19 ooo 9 13 -oo 9 14 -oo 9 17 ooo 9 19 -oo 10 14 o-o 10 16 oo+ 10 17 o-o 10 18 ooo 10 19 o-o 11 12 o-o 11 15 ooo 11 16 ooo 11 18 ooo 12 19 -o- 13 17 ooo 13 18 ooo 14 17 ooo 14 18 o+o 15 19 -o- 16 19 o-- 16 19 oo- 17 18 ooo 17 18 o+o
```
Adaptado:
```text
S S S S S S S S S S S S Ge Ge Ge Ge Ba Ba Ba Ba 0 13 o o o 0 16 o o o 0 17 o o o 0 18 o o o 0 19 o o - 1 13 - o o 1 14 - - o 1 16 - o o 1 18 o o o 2 12 o o o 2 16 - + o 2 17 o o o 2 18 o + o 2 19 - o - 3 12 o - + 3 16 - o + 3 17 o - o 3 18 o o o 3 19 - - o 4 15 o o + 4 16 - o + 4 17 o o o 4 18 o o o 4 19 - o o 5 13 o o o 5 16 o o + 5 17 o o o 5 18 o o o 5 19 o o o 6 14 o o o 6 16 o + o 6 17 o o o 6 18 o + o 6 19 o o - 7 15 o o o 7 16 - o o 7 17 o o o 7 18 o o o 7 19 - o - 8 12 o o + 8 15 o o + 8 17 o o o 8 19 o o o 9 13 - o o 9 14 - o o 9 17 o o o 9 19 - o o 10 14 o - o 10 16 o o + 10 17 o - o 10 18 o o o 10 19 o - o 11 12 o - o 11 15 o o o 11 16 o o o 11 18 o o o 12 19 - o - 13 17 o o o 13 18 o o o 14 17 o o o 14 18 o + o 15 19 - o - 16 19 o - - 16 19 o o - 17 18 o o o 17 18 o + o
```

Problemas críticos:
- No checkpoint-specific original tokenizer artifact proves vocabulary IDs and CLS/SEP wrapping

Conclusión: la adaptación sintáctica y el vocabulario candidato se auditan separadamente de la compatibilidad exacta con el checkpoint. No se autoriza extracción científica ni envío GPU mientras exista un control crítico fallido.

Fuentes y hashes: reference/reference_manifest.json. Código oficial: https://github.com/lamalab-org/MatText; vocabulario original: https://zenodo.org/records/11484062.
