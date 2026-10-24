# MatText frozen experiment — closure log

## Status: BLOCKED before feature extraction

The experiment stopped at Stage A because the exact pretrained tokenizer could not be established from the pinned local checkpoint snapshot. This is a critical integrity condition: the experiment must not feed guessed token IDs to learned embedding rows.

Evidence is recorded in `provenance/preflight.json` and `tokenizer_audit/`. The model config and actual `embeddings.word_embeddings.weight` both specify 30,522 vocabulary rows. The 71 safetensors tensors total 32,962,048 parameters by shape. `AutoTokenizer.from_pretrained` against the local checkpoint directory instantiated a five-token fallback; the diagnostic SLICES input had 8 body tokens mapped to `[UNK]`. The separately recovered MatText SLICES tokenizer has 156 IDs and a different padding ID. The local probe used Transformers 5.17.0; the checkpoint config records 4.38.2. The downloaded checkpoint snapshot contains weights/config/model card only and omits `tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`.

No hidden states, probes, heads, CV metrics, plots, test predictions, or test metrics were produced. No DFT test labels were read. MatGPT and MatterGPT outputs remain unchanged. The model weights were inspected by reading the safetensors header only and were not modified.

### What is needed to resume

Recover the exact tokenizer artifact or training-time token-to-ID mapping for revision `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`, then demonstrate that its tokenizer IDs and special-token IDs correspond to the embedding table rows. Once that check passes, proceed with feature extraction and the pre-registered grouped-CV protocol. Do not substitute the compact tokenizer or a generic BERT tokenizer based solely on vocabulary size.
