# Historical tokenizer trace

Experiment: exp02_only_hex_mattext_finetune / tokenizer_certification. Audit date: 2026-10-21. No training, Slurm or model inference.

Pinned public histories are indirect evidence about the run unless explicitly stated otherwise. Full archives, hashes and acquisition times are in evidence_manifest.json; source_keyword_evidence.csv contains every keyword hit with URL, line and file hash.

## Snapshots

- mattext_preupload: `b7ce8b3e071c18440e6720d682be6bd3e355f9c9`, commit timestamp 2024-06-04T20:48:09Z.
- mattext_june28: `c7ad540d63bcf1d02e0529c26c3a09ec4ded81d5`, commit timestamp 2024-06-27T13:27:38Z.
- xtal_preupload: `3f7081c474ea8ab79f9ed5920cfb754fb14de1cc`, commit timestamp 2024-06-03T18:47:09Z.
- xtal_june5: `1d9d0c6f8675efea885ef7c9c87b3195cd79aaed`, commit timestamp 2024-06-05T16:32:30Z.
- xtal_2026: `c6cf9593596ce99f054f1ebbdc4346d7dbec99fb`, commit timestamp 2026-03-04T01:08:33Z.
- Published model upload: 2024-06-05 03:40:59 UTC. June5 tokenizer revision 1d9d0c6 is later that day; pre-upload bundled vocabulary is now checked directly.
- Zenodo 21151839: code version 1.1.0, archived 2026-07-02. Archive is reused with its published checksum from the prior reference manifest.

## Observations

### Tokenizer before model

[mattext_preupload/src/mattext/models/pretrain.py:26](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L26) — `super().__init__(`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

TokenizerMixin is initialized before dataset mapping and model creation; no exact run identified.

### Configuration-driven specials

[mattext_preupload/src/mattext/models/pretrain.py:28](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L28) — `special_tokens=cfg.model.special_tokens,`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

Candidate uses YAML specials rather than constructor defaults.

### Ordered special dictionary

[mattext_preupload/conf/model/archived_base/pretrain_template.yaml:5](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/conf/model/archived_base/pretrain_template.yaml#L5) — `"unk_token": "[UNK]",`; SHA-256 `047ae0a2f5d391f8ab0bf4c6881e6ead10e571309f2de229dffba64d5092700b`

Seven special tokens in order UNK,PAD,CLS,SEP,MASK,EOS,BOS; no run-specific serialized vocab.

### Vocabulary then specials

[xtal_preupload/src/xtal2txt/tokenizer.py:146](https://github.com/lamalab-org/xtal2txt/blob/3f7081c474ea8ab79f9ed5920cfb754fb14de1cc/src/xtal2txt/tokenizer.py#L146) — `self.vocab = self.load_vocab(vocab_file)`; SHA-256 `eda1bd3b8ccacdb1d4a61019c1409af83fcb3fc2f0be7b84b2525f241f2680e5`

149 bundled structural rows loaded before appending special tokens.

### MLM collator

[mattext_preupload/src/mattext/models/pretrain.py:94](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L94) — `data_collator = DataCollatorForLanguageModeling(`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

MLM probability comes from YAML (0.15); runtime collation tested separately.

### Generic model config

[mattext_preupload/src/mattext/models/pretrain.py:102](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L102) — `config = AutoConfig.from_pretrained(`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

bert-base-uncased configuration overrides hidden/layers/heads/positions; no BERT vocab_size override or embedding resize in this method.

### Random model initialization

[mattext_preupload/src/mattext/models/pretrain.py:106](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L106) — `model = AutoModelForMaskedLM.from_config(config)`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

Public source constructs MLM model from config; released checkpoint contains BertModel encoder/pooler, not MLM head. Extraction/export procedure remains absent.

### 2024 positional override

[mattext_preupload/src/mattext/models/pretrain.py:92](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L92) — `config_model_args["max_position_embeddings"] = self.context_length`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

512 context overrides 1024 template in 2024. This resolves apparent shape mismatch for this snapshot only.

### 2026 archive differs

[zenodo/src/mattext/models/pretrain.py:112](https://github.com/lamalab-org/MatText/blob/2a594a4c69237da2daccab04e0515cf8cdcd715d/src/mattext/models/pretrain.py#L112) — `# config_model_args["max_position_embeddings"] = self.context_length`; SHA-256 `cc8af88202f368e0cbf9963b7ae3f8208f508d34ade5091fc247ee98af86b222`

Same assignment later commented; archive cannot be assumed identical to 2024 run.

### Representation alias mismatch

[mattext_preupload/src/mattext/models/utils.py:19](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/utils.py#L19) — `"slices": SliceTokenizer,`; SHA-256 `0ba53cddc8fd2ad9a831c598bdbe3ee8ca39841dc1b0f04c5bf3e039f0fdeac3`

Archived pretrain2m YAML says slice; selector has slices. No alias conversion in TokenizerMixin.

### Candidate YAML

[mattext_preupload/conf/archived_experiments/archived/pretrain2m/slice.yaml:6](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/conf/archived_experiments/archived/pretrain2m/slice.yaml#L6) — `representation: slice`; SHA-256 `97d4407fc3c9c9a80085837dfd567fea6f03e6d7577cbe6c190d652a73eab770`

Compatible context512/batch128/namept_2m_wes, but alias and run-name/path differ from published checkpoint.

### Tokenizer padding

[mattext_preupload/src/mattext/models/utils.py:132](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/utils.py#L132) — `padding="max_length",`; SHA-256 `0ba53cddc8fd2ad9a831c598bdbe3ee8ca39841dc1b0f04c5bf3e039f0fdeac3`

Tokenization pads/truncates to context before collator; attention masks constructed by HF tokenizer from length.

### Llama resize only

[mattext_preupload/src/mattext/models/utils.py:79](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/utils.py#L79) — `model.resize_token_embeddings(len(llama_tokenizer), pad_to_multiple_of=8)`; SHA-256 `0ba53cddc8fd2ad9a831c598bdbe3ee8ca39841dc1b0f04c5bf3e039f0fdeac3`

Located in smart_tokenizer_and_embedding_resize for llama_tokenizer; not evidence of BERT resizing.

### Saving

[mattext_preupload/src/mattext/models/pretrain.py:134](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/src/mattext/models/pretrain.py#L134) — `model.save_pretrained(self.cfg.path.finetuned_modelname)`; SHA-256 `38cef0f625930a7f0a9967fe492d6e9ca08d21106078185fd81bd0a86dbd6bfe`

Saves model, not tokenizer, in this method. Original HF tree has no tokenizer or training_args.bin.

### Moving dependency

[mattext_preupload/pyproject.toml:34](https://github.com/lamalab-org/MatText/blob/b7ce8b3e071c18440e6720d682be6bd3e355f9c9/pyproject.toml#L34) — `"xtal2txt @ git+https://github.com/lamalab-org/xtal2txt.git",`; SHA-256 `b6e853c29faa5cb3e8b365d639622b4259d68d13b6d359090a8d12bd30ce0ce1`

xtal2txt Git dependency has no pinned commit; source snapshot date cannot identify installed training revision.

### Original tests

[xtal_preupload/tests/tokenizer/test_slice_tokenizer.py:46](https://github.com/lamalab-org/xtal2txt/blob/3f7081c474ea8ab79f9ed5920cfb754fb14de1cc/tests/tokenizer/test_slice_tokenizer.py#L46) — `def test_encode_decode(tokenizer):`; SHA-256 `21dfada18b09da6352f7bab95121fa9a12807fccf3b5845151bd256afd1448d1`

Tests assert round-trip and explicit token sequences; no checkpoint hash, special custom IDs, training batches or MLM output.

## Artifact searches and limits

All file names in all five full source ZIPs and the paper ZIP were searched (including binary names). Keyword inventory scans text sources/configurations/tests, not binary training state. Results below do not establish global nonexistence in private storage.

- mattext_preupload: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.
- mattext_june28: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.
- xtal_preupload: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.
- xtal_june5: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.
- xtal_2026: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.
- zenodo_1.1.0: exact requested filenames found: none. Vocabulary-specific names listed in artifact_search.json.

Existing GitHub issues and HF discussion metadata were inspected; no checkpoint-specific confirmation was identified in those cached results. Issue99 concerns Llama and cannot resolve BERT provenance. No messages were sent.

## Historical library evidence

- [tokenization_utils.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils.py), SHA-256 `4aec95fbac425cc84ea83757131b467975949235ade205777f3dcf5a36e392e0`.
- [tokenization_utils_base.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils_base.py), SHA-256 `a16d2155f1b8a28bec95a76b03eed8bb3d43a4c4637e4f545ce0d6e8c5a4e36d`.
- [data/data_collator.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/data/data_collator.py), SHA-256 `11080356facbc57cc366aa0c007c1577ac2414c7f4a637426044709446ff2fec`.
- [models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bert/modeling_bert.py), SHA-256 `fc0eefec60cc38eddfa033c42cd811103ec333c525ecc59843e10dc01bc30bab`.
- [models/bert/configuration_bert.py](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bert/configuration_bert.py), SHA-256 `a1664419de46982f33a022200efff38f43676d8e4893313d5dd963c5d9bf41bf`.

BertConfig 4.38.2 defaults to vocab_size=30522; archived BERT source inherits bert-base-uncased config without custom vocab override. This supports a generic configuration explanation. It does not prove which effective vocabulary or installed revision produced checkpoint-393000.

## Changes and candidate configurations

Bundled pre-upload vocabulary and later external slice_vocab.txt are byte-identical. Full mapping preserves source chemical order, including Ni53 before Co54; no sorting by atomic number was introduced. Vector tuples occupy0–26 and digits139–148. Multi-digit node indices tokenize digit by digit with special_num_token=False.

Both 2024 sources reproduce the documented default Ga/P IDs exactly. The 2026 source with Transformers4.38.2 instead returns doubled CLS/SEP (17 IDs instead of15); its vocab mapping alone remains unchanged. This is a directly tested version incompatibility, not evidence that the 2024 run used that later code. Full integer sequences and all versions are in tokenizer_golden_tests.json.

The historical default has no MASK and appropriately fails MLM construction. The source-defined seven-special-token candidate passes native CPU MLM collation with seed42; len(tokenizer)=156. Absence of a vocab_size override in the old class does not imply len failure: the installed parent uses len(get_vocab()). No patch was applied.

candidate_configurations.csv lists every slice pretraining YAML and all pretraining templates/examples in the examined snapshots. Naming/path/context similarities are not run provenance; RT configurations change numeric encoding, and 30k/100k/300k variants identify other dataset scales. finetune2m/slice.yaml is a downstream config referencing a30k checkpoint, not proof of this2m run.
