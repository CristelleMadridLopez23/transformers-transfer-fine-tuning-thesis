# Special tokens and ID zero

| Token | Documented default | Reconstructed pretraining YAML | Public model config |
|---|---:|---:|---:|
| `o o o` | 0 | 0 | not specified |
| `[UNK]` | absent | 149 | not specified |
| `[PAD]` | absent | 150 | pad_token_id=0 |
| `[CLS]` | 149 | 151 | not specified |
| `[SEP]` | 150 | 152 | not specified |
| `[MASK]` | absent | 153 | not specified |
| `[EOS]` | absent | 154 | not specified |
| `[BOS]` | absent | 155 | not specified |

Source loads vocabulary first and appends source-defined special tokens in dictionary order. These are reproductions of existing historical configurations, not a newly extended tokenizer. Default documentation lacks MASK/PAD/UNK; it is not sufficient for native MLM.

The 2024 TokenizerMixin calls HF padding=max_length before the collator. Exact local probes show structural ID0 remains attended while inserted PAD150 receives attention0. BERT constructs nn.Embedding(..., padding_idx=config.pad_token_id): with this public config the input embedding padding index is0. Masks are passed independently. This observation does not prove the same combination was used during training.

MLM collator gets special-token mask and excludes special IDs from MLM targets; labels for unselected positions are -100. Its 80/10/10 replacement samples random_words using len(tokenizer). Unmodified historical collation result:

```json
{
  "seed": 42,
  "training_performed": false,
  "len_tokenizer": 156,
  "collation": {
    "input_ids": [
      [
        151,
        57,
        57,
        41,
        41,
        139,
        142,
        24,
        139,
        153,
        20,
        139,
        140,
        8,
        152
      ],
      [
        151,
        82,
        42,
        139,
        140,
        0,
        152,
        150,
        150,
        150,
        150,
        150,
        150,
        150,
        150
      ]
    ],
    "token_type_ids": [
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    ],
    "attention_mask": [
      [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ],
      [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    ],
    "labels": [
      [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        141,
        -100,
        -100,
        -100,
        -100,
        -100
      ],
      [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        -100
      ]
    ]
  },
  "status": "passed"
}
```

## Contradictions and hypotheses

| Observed contradiction | Evidence | Possible explanation (unverified hypothesis) |
|---|---|---|
| Public checkpoint PAD=0 versus reconstructed configured tokenizer PAD=150 and structural vector ID0; actual training semantics not attested | checkpoint_identity.json / tokenizer_golden_tests.json / historical_tokenizer_trace.md | Different effective config, local patch, installed revision or export config; none established |
| Archived pretrain2m slice alias is incompatible with selector slices; effective original overrides absent | checkpoint_identity.json / tokenizer_golden_tests.json / historical_tokenizer_trace.md | Different effective config, local patch, installed revision or export config; none established |

## Read-only embedding diagnosis

```json
[
  {
    "row": 0,
    "l2_norm": 1.6188649342181143,
    "zero_components": 0,
    "row_sha256": "f7b5e464ea5ff18c86ba601a578fd49910ddce6eb9a8fecf69e0215417bf6fa8"
  },
  {
    "row": 149,
    "l2_norm": 1.2368264973137475,
    "zero_components": 0,
    "row_sha256": "e464cd8d5a4906d85145cb57dcfa7a71a5b4301d395cb77ffd360452eb66db0d"
  },
  {
    "row": 150,
    "l2_norm": 1.221300005591168,
    "zero_components": 0,
    "row_sha256": "57aedcc7faef7f29366e492466987d3e7c13c27b1367b9afe9642df09cb337cc"
  },
  {
    "row": 151,
    "l2_norm": 1.217919619064858,
    "zero_components": 0,
    "row_sha256": "8c24b4035c8f5c7b858d1a9f6666889cb2a7ffa73eb0cf6310fab20fe7c16843"
  },
  {
    "row": 152,
    "l2_norm": 1.2393542555114472,
    "zero_components": 0,
    "row_sha256": "d59eaf03cfcaba63cc01544d85841cc5d331ee9a8671bd51bdd088d8e572fc94"
  },
  {
    "row": 153,
    "l2_norm": 1.0961585726190468,
    "zero_components": 0,
    "row_sha256": "f21a4d48ad7bb0a331dce551d9d56f22fd04ce6fee823fcefbbccf539ff88166"
  },
  {
    "row": 154,
    "l2_norm": 1.2645254975974138,
    "zero_components": 0,
    "row_sha256": "ff01692ba0f8fc03fac798ffa1f3fdde5ac5f1adc3bd862d169330f583b3a44a"
  },
  {
    "row": 155,
    "l2_norm": 1.1940894257097423,
    "zero_components": 0,
    "row_sha256": "3b175caa6b06f3238483ccc6bd346c8292edc8e42ef8876055d282d4c7c5d885"
  },
  {
    "row": 1000,
    "l2_norm": 2.6419023148380507,
    "zero_components": 0,
    "row_sha256": "b41215a624ce6a13c75c7683951b735cb781718ec0701fae3a5a49db32e88b15"
  },
  {
    "row": 30521,
    "l2_norm": 2.630697646436033,
    "zero_components": 0,
    "row_sha256": "52e115a743e59ef8ad21ee36de365e6632885947c4448ba82b038febde3fd122"
  }
]
```

Nonzero/zero row norms do not prove padding or chemical semantics. MLM output weight tying and optimizer updates can affect rows even when input padding gradients are suppressed. No weights, IDs, padding settings or embedding sizes were changed.

- [tokenization_utils.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils.py), SHA-256 `4aec95fbac425cc84ea83757131b467975949235ade205777f3dcf5a36e392e0`.
- [tokenization_utils_base.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils_base.py), SHA-256 `a16d2155f1b8a28bec95a76b03eed8bb3d43a4c4637e4f545ce0d6e8c5a4e36d`.
- [data/data_collator.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/data/data_collator.py), SHA-256 `11080356facbc57cc366aa0c007c1577ac2414c7f4a637426044709446ff2fec`.
- [models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bert/modeling_bert.py), SHA-256 `fc0eefec60cc38eddfa033c42cd811103ec333c525ecc59843e10dc01bc30bab`.
- [models/bert/configuration_bert.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bert/configuration_bert.py), SHA-256 `a1664419de46982f33a022200efff38f43676d8e4893313d5dd963c5d9bf41bf`.
