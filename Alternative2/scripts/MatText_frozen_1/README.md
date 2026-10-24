# MatText frozen ABS3 transfer

## Current status: BLOCKED at Stage A

The pinned checkpoint's learned input embedding has 30,522 rows. `AutoTokenizer.from_pretrained` on the local checkpoint directory silently falls back to a five-token BERT tokenizer and maps all sample SLICES content to `[UNK]`. The separately recovered `tokenizer/slice/tokenizer.json` has 156 entries and `[PAD]=150`, while the checkpoint config says `pad_token_id=0`. The pinned checkpoint snapshot contains no original tokenizer files. Therefore Stage B feature extraction and every training or evaluation stage are intentionally gated off. No embedding resize, vocabulary remapping, substitute tokenizer, or partial weight load is used.

Run the no-label preflight from the repository root with the project environment:

```bash
python -u Alternative2/scripts/MatText_frozen_1/run_experiment.py
```

It writes the hard-gate evidence to `Alternative2/reports/MatText_frozen_1/provenance/preflight.json` and exits with code 2 while blocked. The experiment will proceed only after recovering the exact tokenizer-to-row mapping used to train the pinned checkpoint and passing the Stage A gate.

The official MatText `SliceTokenizer` documentation describes a compact 149-token base vocabulary plus `[CLS]` and `[SEP]`; that establishes the recovered compact tokenizer as plausible for MatText's tokenizer library, not as compatible with these 30,522 learned embedding rows. See [official tokenizer docs](https://lamalab-org.github.io/MatText/docs/tokenizers/) and the [pinned model repo](https://huggingface.co/n0w0f/MatText-slices-2m).
