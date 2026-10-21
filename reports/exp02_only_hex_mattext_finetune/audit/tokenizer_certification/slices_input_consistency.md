# Existing SLICES input consistency

Source: `data/only_hex_new/configurations_only_hex_new.extxyz`, SHA-256 `a31f436fd6f44be4f85c5110f8d5ebe37718a5b28eec5453425d6a9724271de4`.
Existing converter: `model-pretrained/scripts/only_hex_new_csv_slices/02_csv_to_slices.py`; audit adapter: `model-pretrained/scripts/exp02_only_hex_mattext_finetune/01_audit_tokenizer.py`; graph parser: `model-pretrained/scripts/exp01_mattext_frozen/utils.py`.

The existing manifest records SLICES2.0.12, econnn neighbor graph, relax_model=None, strategy4 original generation, explicit graph parsing and official strategy3 serialization for historical MatText input. Original compact vectors and graph data are retained. This is syntax adaptation with graph-equivalence checks, not reconstruction of coordinates or relaxation. All signs, node indices and periodic-vector tuples must survive. No indiscriminate string replacement is used.

Original corpus provenance is separate from this thesis dataset: the latter cannot certify the training corpus of MatText.

## Reused evidence

`existing_audit_verification.json` contains current SHA-256 checks, recounts and precise limits. Locally recounted:8898 source rows,8712 accepted,186 graph rejects,2853 ID sequences,1119 shared groups,zero additional tokenizer collisions,zero persisted omissions/UNK/truncation,max401. The five-test success log exists; those tests were not rerun. Per-row graph-equivalence and equality to official encoding are persisted results, not a fresh full-dataset computation. Small new real-sequence comparisons are in tokenizer_golden_tests.json.

The upstream historical regex uses findall and can omit compact `ooo` and unknown `@` without emitting UNK. Zero UNK alone is insufficient. The existing audit's tokenize_lossless uses regex spans and rejects every unmatched nonwhitespace character; graph parser checks representation before official encoding. New negative probes preserve upstream omissions as diagnostics, never accepted scientific inputs. Numeric indices are digit-tokenized with special_num_token=False; periodic vectors are indivisible vocabulary tokens in spaced syntax. Arbitrary extra spacing inside a vector is not assumed equivalent.

Collisions among trajectories/geometries reflect the representation's topology and are separate from checkpoint-tokenizer provenance. They do not prove semantic loss by tokenization or certification. No energy prediction/lower-bound statistic is used in this certification.
