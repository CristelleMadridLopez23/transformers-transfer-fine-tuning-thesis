# MatText-slices-2m frozen transfer to ABS₃ static DFT energies

## Study status: BLOCKED at tokenizer/checkpoint preflight

The planned scientific comparison could not start because the local pinned checkpoint does not include its tokenizer and the recovered tokenizer is incompatible by vocabulary dimension and special IDs. This report therefore contains an integrity finding, not model-performance results.

### Checkpoint evidence

- Repository: `n0w0f/MatText-slices-2m`.
- Pinned revision: `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`.
- Architecture from config: BERT, 4 layers, hidden size 512, 8 attention heads, max positions 512.
- The safetensors header contains 71 tensors totaling 32,962,048 parameters by shape; the word embedding accounts for 15,891,456 parameters.
- `config.vocab_size`: 30,522; `embeddings.word_embeddings.weight`: `[30522, 512]`.
- Recovered SLICES `tokenizer.json`: 156 entries; `[PAD]` ID 150 versus model config pad ID 0.
- Root `AutoTokenizer.from_pretrained(..., local_files_only=True)` created a 5-token fallback. In the diagnostic SLICES sample, all body pieces were `[UNK]`.
- The local probe used Transformers 5.17.0, while the checkpoint config records Transformers 4.38.2. This runtime difference is recorded; it does not resolve the missing tokenizer mapping.
- The separately recovered compact tokenizer preserves the 149 documented base SLICES token order, but its mapping to the model's 30,522 trained rows is unknown.
- Exact tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, `vocab.txt`) are absent from the pinned checkpoint manifest.

### Protocol disposition

Per the critical integrity gate, feature extraction, frozen probes, MT1/MT2/MT3-A fitting, development CV, test locking/prediction/scoring, and final cross-model comparison were not run. Test target labels were not read. This prevents unsupported results from being mistaken for a MatText evaluation.

### References

- [MatText official tokenizer documentation](https://lamalab-org.github.io/MatText/docs/tokenizers/).
- [MatText source repository](https://github.com/lamalab-org/MatText).
- [Pinned MatText-slices-2m checkpoint](https://huggingface.co/n0w0f/MatText-slices-2m).
- Alampara et al., [MatText, arXiv:2406.17295](https://arxiv.org/abs/2406.17295).
