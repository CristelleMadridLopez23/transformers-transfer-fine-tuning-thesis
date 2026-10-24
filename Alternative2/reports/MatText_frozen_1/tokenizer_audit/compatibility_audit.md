# Auditoría de compatibilidad tokenizer–pesos

**Estado: BLOCKED.** El checkpoint tiene 30,522 filas en `embeddings.word_embeddings.weight`; el tokenizer SLICES recuperado tiene 156 IDs. `AutoTokenizer` desde la raíz local: BertTokenizer, vocabulario 5, UNKs del cuerpo SLICES diagnóstico 8. `pad_token_id` modelo=0, tokenizer recuperado=150. Bloqueos: `['recovered_tokenizer_size_156_does_not_match_embedding_rows_30522', 'recovered_pad_token_id_does_not_match_model_config', 'AutoTokenizer_from_checkpoint_root_maps_SLICES_body_to_UNK', 'pinned_checkpoint_snapshot_does_not_contain_original_tokenizer_files']`.

No se extrajeron features ni se leyeron etiquetas DFT. Recuperar el tokenizer original y demostrar la correspondencia exacta token→fila antes de continuar.
