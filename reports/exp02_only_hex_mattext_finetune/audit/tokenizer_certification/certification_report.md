# Certification report — BLOCKED

All twelve requested steps were reviewed. Technical compatibility is supported; historical checkpoint-tokenizer certification remains blocked. No training, inference, Slurm, tokenizer gate changes, embedding resizing or original checkpoint modifications occurred.

Checkpoint `n0w0f/MatText-slices-2m` at `c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e`. Weights SHA-256 `6bef5652d758190e6e6eaf85ccf2a2f584b69255f1be3bb7bdd72cb1f12a74bd`; config SHA-256 `ddd1b4cf361b4fe3a2724563d840e5424bfe05c62aa1ff922a1cc7d7a5da40c0`. Public/local bytes match; 71 tensor shapes and 32962048 parameters match BertModel config. Embeddings30522×512; positions512×512;4layers;8heads;Transformers4.38.2.

Historical tokenizer: xtal2txt `3f7081c474ea8ab79f9ed5920cfb754fb14de1cc` before June5 upload, using its bundled vocabulary SHA-256 `53e2641804ff63c8b1f48b7311a74fd64dafe858d5935e89cb3e50a5c67c60ea` and unchanged AST class bodies. 151/151 default documented token IDs match. Custom YAML reproduces156 entries with shifted special IDs. None of these observations alone binds vocabulary semantics to matrix rows.

Deterministic probes: 22 recorded input/variant cases, repeated integer/mask equality, official Ga/P reference, real ABS3 samples, multi-digit indices, periodic vectors, max-length401 plus padding512, literal specials and deliberate invalid-input diagnostics. Original training reference masks and author CLS vector are absent. MLM diagnostic status: passed.

Favorable evidence, contradictions, missing links and exact unlock conditions are machine-readable in certification_verdict.json. The provenance matrix distinguishes direct evidence, indirect evidence, reconstruction, contradiction and absence for each link.

Frozen extraction and supervised fine-tuning remain unapproved as faithful transfer from the original checkpoint. A future separately authorized exploratory run could be labeled reconstructed-tokenizer exploration, with these uncertainties stated; it would not establish historical reproduction. Predictive results cannot resolve token-ID provenance retrospectively. This audit makes no claim that lack of historical certification prohibits publication of transparently labeled exploratory work.

Thesis disclosure: checkpoint identity is verified; original effective tokenizer/run, custom special IDs and PAD0 handling are unresolved; public code revisions differ; SLICES is a topology representation and geometry collisions require separate interpretation.

Files created: required ten report files plus evidence manifests, artifact and keyword inventories, prior-verification JSON, evidence source caches and working constructor outputs; new Python scripts only under model-pretrained/scripts/tokenizer_certification/. No pre-existing experimental scripts or gates were edited. author_request.md is an English draft and was not sent. Created-file inventory and hashes are recorded in created_files.csv after validation.

Reproduction commands and test protocol: model-pretrained/scripts/tokenizer_certification/README.md. Unit-test execution status is in unit_tests.log; deterministic scientific diagnostics can intentionally fail and must not be relabeled as software-test successes.
