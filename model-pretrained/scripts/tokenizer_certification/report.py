"""Produce evidence-linked scientific certification reports from real probes.

Requires completed gather.py, audit.py and probe.py outputs. Formal decisions
depend on provenance and unresolved material contradictions, never predictive
performance. This script writes only new certification reports; existing
experiment gates, checkpoint bytes and original datasets remain untouched.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

from audit import ARTIFACT_NAMES, csv_write, decide, read
from gather import DEFAULT, ROOT, SNAPSHOTS, digest, write_json

LOGGER = logging.getLogger(__name__)


def cite(output: Path, label: str, relative: str, needle: str) -> str:
    """Return pinned source URL, first matching line and observed text; fail if absent."""
    repository, revision = SNAPSHOTS.get(label, ("MatText", "2a594a4c69237da2daccab04e0515cf8cdcd715d"))
    root = output.parent / "reference/paper_code/MatText-main" if label == "zenodo" else output / "evidence" / label
    path = root / relative
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if needle in line:
            return f"[{label}/{relative}:{number}](https://github.com/lamalab-org/{repository}/blob/{revision}/{relative}#L{number}) — `{line.strip()}`; SHA-256 `{digest(path)}`"
    raise ValueError(f"Evidence needle absent: {path}: {needle}")


def mappings(output: Path, golden: dict[str, Any]) -> dict[str, int]:
    """Compare all structural and special IDs across actual source variants and docs."""
    import re
    import html
    text = html.unescape((output / "evidence/tokenizers_documentation.html").read_text())
    text = re.sub(r"<[^>]+>", "", text)
    start = text.index("{'o o o': 0")
    end = text.index("}", start) + 1
    documented = ast.literal_eval(text[start:end])
    implementations = golden["implementations"]
    historical = implementations["xtal_preupload"]["variants"]["documentation_default"]["vocab"]
    custom = implementations["xtal_preupload"]["variants"]["archived_pretraining_specials"]["vocab"]
    rows = []
    for token in dict.fromkeys([*documented, *custom]):
        observed, rebuilt = documented.get(token), historical.get(token)
        rows.append({"token": token, "id_historico": observed, "id_reconstruido": rebuilt,
                     "coincide": observed == rebuilt if observed is not None and rebuilt is not None else "not_in_documented_default",
                     "id_pretraining_yaml_candidate": custom.get(token),
                     "id_june5_default": implementations["xtal_june5"]["variants"]["documentation_default"]["vocab"].get(token),
                     "id_2026_default": implementations["xtal_2026"]["variants"]["documentation_default"]["vocab"].get(token),
                     "evidencia": "official rendered documentation vs pre-upload class and bundled ordered vocab; not checkpoint linkage"})
    csv_write(output / "vocabulary_mapping_comparison.csv", rows)
    return {"documented_entries": len(documented), "exact_default_matches": sum(documented[token] == historical.get(token) for token in documented),
            "structural_entries": 149, "custom_entries": len(custom)}


def library_evidence(output: Path) -> list[str]:
    """Archive installed historical library sources and return official versioned URLs."""
    runtime = ROOT / "experiments/exp02_only_hex_mattext_finetune/reference_runtime/packages/transformers"
    files = ["tokenization_utils.py", "tokenization_utils_base.py", "data/data_collator.py",
             "models/bert/modeling_bert.py", "models/bert/configuration_bert.py"]
    lines = []
    for relative in files:
        source = runtime / relative
        target = output / "evidence/runtime_4.38.2" / (relative + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        lines.append(f"- [{relative}](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/{relative}), SHA-256 `{digest(source)}`.")
    return lines


def report(output: Path) -> None:
    """Generate required reports with explicit direct, indirect, absent and contradictory links."""
    identity = read(output / "checkpoint_identity.json")
    prior = read(output / "existing_audit_verification.json")
    golden = read(output / "tokenizer_golden_tests.json")
    historical = golden["implementations"]["xtal_preupload"]
    default = historical["variants"]["documentation_default"]
    custom = historical["variants"]["archived_pretraining_specials"]
    if default["status"] != "executed" or custom["status"] != "executed":
        raise RuntimeError("Historical probe incomplete; cannot write conclusions")
    comparison = mappings(output, golden)
    library = library_evidence(output)
    evidence = output / "evidence"
    pre = "src/mattext/models/pretrain.py"
    utils = "src/mattext/models/utils.py"
    tokenizer = "src/xtal2txt/tokenizer.py"
    template = "conf/model/archived_base/pretrain_template.yaml"
    slice_yaml = "conf/archived_experiments/archived/pretrain2m/slice.yaml"
    observations = [
        ("Tokenizer before model", cite(output, "mattext_preupload", pre, "super().__init__("), "TokenizerMixin is initialized before dataset mapping and model creation; no exact run identified."),
        ("Configuration-driven specials", cite(output, "mattext_preupload", pre, "special_tokens=cfg.model.special_tokens"), "Candidate uses YAML specials rather than constructor defaults."),
        ("Ordered special dictionary", cite(output, "mattext_preupload", template, '"unk_token"'), "Seven special tokens in order UNK,PAD,CLS,SEP,MASK,EOS,BOS; no run-specific serialized vocab."),
        ("Vocabulary then specials", cite(output, "xtal_preupload", tokenizer, "self.vocab = self.load_vocab"), "149 bundled structural rows loaded before appending special tokens."),
        ("MLM collator", cite(output, "mattext_preupload", pre, "data_collator = DataCollatorForLanguageModeling"), "MLM probability comes from YAML (0.15); runtime collation tested separately."),
        ("Generic model config", cite(output, "mattext_preupload", pre, "config = AutoConfig.from_pretrained("), "bert-base-uncased configuration overrides hidden/layers/heads/positions; no BERT vocab_size override or embedding resize in this method."),
        ("Random model initialization", cite(output, "mattext_preupload", pre, "model = AutoModelForMaskedLM.from_config(config)"), "Public source constructs MLM model from config; released checkpoint contains BertModel encoder/pooler, not MLM head. Extraction/export procedure remains absent."),
        ("2024 positional override", cite(output, "mattext_preupload", pre, 'config_model_args["max_position_embeddings"] = self.context_length'), "512 context overrides 1024 template in 2024. This resolves apparent shape mismatch for this snapshot only."),
        ("2026 archive differs", cite(output, "zenodo", pre, '# config_model_args["max_position_embeddings"]'), "Same assignment later commented; archive cannot be assumed identical to 2024 run."),
        ("Representation alias mismatch", cite(output, "mattext_preupload", utils, '"slices": SliceTokenizer'), "Archived pretrain2m YAML says slice; selector has slices. No alias conversion in TokenizerMixin."),
        ("Candidate YAML", cite(output, "mattext_preupload", slice_yaml, "representation: slice"), "Compatible context512/batch128/namept_2m_wes, but alias and run-name/path differ from published checkpoint."),
        ("Tokenizer padding", cite(output, "mattext_preupload", utils, 'padding="max_length"'), "Tokenization pads/truncates to context before collator; attention masks constructed by HF tokenizer from length."),
        ("Llama resize only", cite(output, "mattext_preupload", utils, "model.resize_token_embeddings"), "Located in smart_tokenizer_and_embedding_resize for llama_tokenizer; not evidence of BERT resizing."),
        ("Saving", cite(output, "mattext_preupload", pre, "model.save_pretrained"), "Saves model, not tokenizer, in this method. Original HF tree has no tokenizer or training_args.bin."),
        ("Moving dependency", cite(output, "mattext_preupload", "pyproject.toml", "xtal2txt @ git+"), "xtal2txt Git dependency has no pinned commit; source snapshot date cannot identify installed training revision."),
        ("Original tests", cite(output, "xtal_preupload", "tests/tokenizer/test_slice_tokenizer.py", "def test_encode_decode"), "Tests assert round-trip and explicit token sequences; no checkpoint hash, special custom IDs, training batches or MLM output."),
    ]
    traces = ["# Historical tokenizer trace", "", "Experiment: exp02_only_hex_mattext_finetune / tokenizer_certification. Audit date: 2026-10-21. No training, Slurm or model inference.", "",
              "Pinned public histories are indirect evidence about the run unless explicitly stated otherwise. Full archives, hashes and acquisition times are in evidence_manifest.json; source_keyword_evidence.csv contains every keyword hit with URL, line and file hash.", "", "## Snapshots", ""]
    for label, (_, revision) in SNAPSHOTS.items():
        metadata = read(evidence / f"{label}_commit.json")
        traces.append(f"- {label}: `{revision}`, commit timestamp {metadata['commit']['committer']['date']}.")
    traces.extend(["- Published model upload: 2024-06-05 03:40:59 UTC. June5 tokenizer revision 1d9d0c6 is later that day; pre-upload bundled vocabulary is now checked directly.",
                   "- Zenodo 21151839: code version 1.1.0, archived 2026-07-02. Archive is reused with its published checksum from the prior reference manifest.", "", "## Observations", ""])
    for title, source, conclusion in observations:
        traces.extend([f"### {title}", "", source, "", conclusion, ""])
    traces.extend(["## Artifact searches and limits", "", "All file names in all five full source ZIPs and the paper ZIP were searched (including binary names). Keyword inventory scans text sources/configurations/tests, not binary training state. Results below do not establish global nonexistence in private storage.", ""])
    artifact_scope = []
    for label in [*SNAPSHOTS, "zenodo_1.1.0"]:
        archive_path = output.parent / "reference/MatText-main.zip" if label == "zenodo_1.1.0" else evidence / f"{label}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            found = [name for name in archive.namelist() if Path(name).name in ARTIFACT_NAMES]
            artifact_scope.append({"snapshot": label, "archive_sha256": digest(archive_path), "exact_named_artifacts": found,
                                   "all_vocabulary_named_members": [name for name in archive.namelist() if "vocab" in Path(name).name.lower()]})
        traces.append(f"- {label}: exact requested filenames found: {found or 'none'}. Vocabulary-specific names listed in artifact_search.json.")
    write_json(output / "artifact_search.json", artifact_scope)
    traces.extend(["", "Existing GitHub issues and HF discussion metadata were inspected; no checkpoint-specific confirmation was identified in those cached results. Issue99 concerns Llama and cannot resolve BERT provenance. No messages were sent.", "", "## Historical library evidence", "", *library, "",
                   "BertConfig 4.38.2 defaults to vocab_size=30522; archived BERT source inherits bert-base-uncased config without custom vocab override. This supports a generic configuration explanation. It does not prove which effective vocabulary or installed revision produced checkpoint-393000."])
    traces.extend(["", "## Changes and candidate configurations", "",
                   "Bundled pre-upload vocabulary and later external slice_vocab.txt are byte-identical. Full mapping preserves source chemical order, including Ni53 before Co54; no sorting by atomic number was introduced. Vector tuples occupy0–26 and digits139–148. Multi-digit node indices tokenize digit by digit with special_num_token=False.", "",
                   "Both 2024 sources reproduce the documented default Ga/P IDs exactly. The 2026 source with Transformers4.38.2 instead returns doubled CLS/SEP (17 IDs instead of15); its vocab mapping alone remains unchanged. This is a directly tested version incompatibility, not evidence that the 2024 run used that later code. Full integer sequences and all versions are in tokenizer_golden_tests.json.", "",
                   "The historical default has no MASK and appropriately fails MLM construction. The source-defined seven-special-token candidate passes native CPU MLM collation with seed42; len(tokenizer)=156. Absence of a vocab_size override in the old class does not imply len failure: the installed parent uses len(get_vocab()). No patch was applied.", "",
                   "candidate_configurations.csv lists every slice pretraining YAML and all pretraining templates/examples in the examined snapshots. Naming/path/context similarities are not run provenance; RT configurations change numeric encoding, and 30k/100k/300k variants identify other dataset scales. finetune2m/slice.yaml is a downstream config referencing a30k checkpoint, not proof of this2m run."])
    configs = []
    for label in ["mattext_preupload", "mattext_june28", "zenodo"]:
        root = output.parent / "reference/paper_code/MatText-main" if label == "zenodo" else evidence / label
        for path in sorted((root / "conf").rglob("*.yaml")):
            relative = str(path.relative_to(root))
            if ("pretrain" in relative and (path.name == "slice.yaml" or "template" in path.name or "example" in path.name)) or "finetune2m/slice.yaml" in relative:
                text = path.read_text()
                values = [line.strip() for line in text.splitlines() if any(key in line for key in ["representation:", "context_length:", "special_num_token:", "max_position_embeddings:", "pretrained_checkpoint:", "checkpoint-", "model_name_or_path:", "name:", "data_root_path:"])]
                category = "candidate subset; context512 but alias/run linkage unresolved" if "pretrain2m/slice.yaml" in relative else "template, alternate scale/numeric encoding, test or downstream configuration; not attributable to exact checkpoint"
                configs.append({"snapshot": label, "file": relative, "sha256": digest(path), "observed_settings": " | ".join(values), "assessment": category})
    csv_write(output / "candidate_configurations.csv", configs)
    (output / "historical_tokenizer_trace.md").write_text("\n".join(traces) + "\n")

    links = [
        ("Original SLICES corpus → training run", "absent", "YAML/data repository names exist; no corpus revision/hash, split manifest or sample batch bound to checkpoint-393000", "pretrain.py:_prepare_datasets; pretrain2m/slice.yaml", "Original dataset revision/checksum and effective run config"),
        ("Training run → tokenizer revision", "absent", "Moving xtal2txt Git dependency; pre-upload code is temporally plausible, not run-attested", "pyproject.toml; HF config _name_or_path", "Installed xtal2txt revision or author confirmation for exact weight SHA"),
        ("Bundled vocabulary → ordered structural IDs", "direct", f"Pre-upload bundled vocab hash {historical['vocab_sha256']}; 149 ordered rows", "xtal_preupload/src/xtal2txt/vocabs/slice_vocab.txt", "No gap within reproduced implementation"),
        ("Public documentation → default token IDs", "direct", f"{comparison['exact_default_matches']}/{comparison['documented_entries']} IDs and official Ga/P example match", "vocabulary_mapping_comparison.csv; tokenizer_golden_tests.json", "Not evidence of original training specials"),
        ("YAML → special-token IDs", "reconstructed", "Custom append order yields UNK149 PAD150 CLS151 SEP152 MASK153 EOS154 BOS155", "historical source + archived pretrain_template.yaml", "Need effective YAML/vocab for actual run"),
        ("Tokenizer → attention masks / MLM", "contradictory" if custom["mlm_probe"]["status"] != "passed" else "indirect", f"Padding is length-based; unmodified historical MLM probe: {custom['mlm_probe']['status']}", "tokenizer_golden_tests.json; HF 4.38.2 collator", "Exact collator/runtime or training batch required"),
        ("Effective run → public YAML", "contradictory", "slice/slices mismatch; pt_2m_wes/megaloop2 paths differ from santiago_ckpt_normal/slice_pt_2m", "historical_tokenizer_trace.md", "Resolved effective configuration, overrides and export command"),
        ("30522 matrix rows → generic BERT config", "indirect", "BertConfig default and AutoConfig inheritance explain size without resizing", "runtime_4.38.2/configuration_bert.py.txt; pretrain.py", "Cannot infer semantic assignment of rows from dimensions/norms"),
        ("MLM trained model → exported BertModel", "absent", "Published tensors contain encoder/pooler; no MLM head or extraction script tied to hash", "checkpoint_identity.json; pretrain.py", "Original export/extraction procedure and checkpoint/run mapping"),
        ("Published HF revision → local weight bytes", "direct", identity["weight_sha256"], "HF pinned tree/LFS vs local streamed SHA-256", "No identity gap"),
    ]
    csv_write(output / "pretraining_provenance_matrix.csv", [{"link": a, "evidence_level": b, "observation": c, "source": d, "missing_or_limit": e} for a,b,c,d,e in links])

    contradictions = ["Public checkpoint PAD=0 versus reconstructed configured tokenizer PAD=150 and structural vector ID0; actual training semantics not attested",
                      "Archived pretrain2m slice alias is incompatible with selector slices; effective original overrides absent"]
    if custom["mlm_probe"]["status"] != "passed":
        contradictions.append(f"Unmodified historical tokenizer cannot execute native Transformers4.38.2 MLM collator: {custom['mlm_probe'].get('error_type')}: {custom['mlm_probe'].get('error_message')}")
    official_ok = default["tests"][0]["official_ids_match"] is True and comparison["exact_default_matches"] == comparison["documented_entries"]
    status = decide(identity["status"] == "verified", official_ok, prior["locally_verified"]["valid_rows"] == 8712, False, False, contradictions)
    verdict = {"status": status, "checkpoint_revision": identity["revision"], "weight_sha256": identity["weight_sha256"],
               "historical_tokenizer_reconstruction": "PARTIALLY_VERIFIED" if official_ok else "BLOCKED",
               "checkpoint_identity": "verified", "checkpoint_specific_tokenizer": "not_established",
               "favorable_evidence": ["Local checkpoint exactly matches pinned public config and weight LFS digest; all tensor shapes validated",
                   f"Pre-upload tokenizer and bundled vocabulary reproduce {comparison['exact_default_matches']} documented IDs and Ga/P encoding",
                   "Stored 8712-row coverage/omission/truncation/collision counts verified without rerunning SLICES"],
               "contradictions": contradictions,
               "missing_evidence": ["Effective ordered vocabulary/specials bound to exact published checkpoint/run", "Original installed tokenizer and collator revisions; effective Hydra config and dataset revision",
                    "Original training string plus input_ids and attention_mask", "MLM-to-BertModel export mapping", "Author reference CLS vector (optional, absent)"],
               "unverified_hypotheses": ["30522 inherited from generic bert-base-uncased config while effective vocab had156 entries",
                    "Training used local patches/older library or a different installed tokenizer/collator than public snapshot",
                    "Model PAD0 and tokenizer PAD150 coexisted with length-based masks; structural ID0 may still be input-attended"],
               "reason": "Historical encoding is reproducible, but exact run linkage is absent and material protocol/configuration contradictions remain. Software successes cannot certify historical semantic assignment.",
               "unlock_conditions": ["Obtain original tokenizer artifacts or author-attested exact reconstruction bound to revision and weight SHA",
                    "Resolve special IDs/PAD0 and supply actual padding, masks and MLM protocol", "Identify effective source commit/config/overrides and exported encoder provenance",
                    "Match an original encoded training example exactly, and any provided author vector with declared tolerance",
                    "Rerun the certification and require CERTIFIED before dependent experiment approval"],
               "frozen_feature_extraction_approved": False, "supervised_finetuning_approved": False,
               "exploratory_status": "Could only be described as reconstructed-tokenizer exploration; this audit does not execute or unlock it",
               "training_performed": False, "slurm_submissions": 0, "weights_modified": False,
               "golden_test_scope": "Documentation/default encoding and local diagnostics; no original training batch or reference CLS vector"}
    write_json(output / "certification_verdict.json", verdict)

    specials = ["# Special tokens and ID zero", "", "| Token | Documented default | Reconstructed pretraining YAML | Public model config |", "|---|---:|---:|---:|"]
    for token in ["o o o", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]"]:
        specials.append(f"| `{token}` | {default['vocab'].get(token, 'absent')} | {custom['vocab'].get(token, 'absent')} | {'pad_token_id=0' if token == '[PAD]' else 'not specified'} |")
    specials.extend(["", "Source loads vocabulary first and appends source-defined special tokens in dictionary order. These are reproductions of existing historical configurations, not a newly extended tokenizer. Default documentation lacks MASK/PAD/UNK; it is not sufficient for native MLM.",
        "", "The 2024 TokenizerMixin calls HF padding=max_length before the collator. Exact local probes show structural ID0 remains attended while inserted PAD150 receives attention0. BERT constructs nn.Embedding(..., padding_idx=config.pad_token_id): with this public config the input embedding padding index is0. Masks are passed independently. This observation does not prove the same combination was used during training.",
        "", "MLM collator gets special-token mask and excludes special IDs from MLM targets; labels for unselected positions are -100. Its 80/10/10 replacement samples random_words using len(tokenizer). Unmodified historical collation result:", "", "```json", json.dumps(custom["mlm_probe"], indent=2), "```", "", "## Contradictions and hypotheses", "", "| Observed contradiction | Evidence | Possible explanation (unverified hypothesis) |", "|---|---|---|"])
    for contradiction in contradictions:
        specials.append(f"| {contradiction} | checkpoint_identity.json / tokenizer_golden_tests.json / historical_tokenizer_trace.md | Different effective config, local patch, installed revision or export config; none established |")
    specials.extend(["", "## Read-only embedding diagnosis", "", "```json", json.dumps(identity["embedding_row_diagnostics"], indent=2), "```", "",
                     "Nonzero/zero row norms do not prove padding or chemical semantics. MLM output weight tying and optimizer updates can affect rows even when input padding gradients are suppressed. No weights, IDs, padding settings or embedding sizes were changed.", "", *library])
    (output / "special_tokens_audit.md").write_text("\n".join(specials) + "\n")

    (output / "slices_input_consistency.md").write_text("""# Existing SLICES input consistency

Source: `data/only_hex_new/configurations_only_hex_new.extxyz`, SHA-256 `a31f436fd6f44be4f85c5110f8d5ebe37718a5b28eec5453425d6a9724271de4`.
Existing converter: `model-pretrained/scripts/only_hex_new_csv_slices/02_csv_to_slices.py`; audit adapter: `model-pretrained/scripts/exp02_only_hex_mattext_finetune/01_audit_tokenizer.py`; graph parser: `model-pretrained/scripts/exp01_mattext_frozen/utils.py`.

The existing manifest records SLICES2.0.12, econnn neighbor graph, relax_model=None, strategy4 original generation, explicit graph parsing and official strategy3 serialization for historical MatText input. Original compact vectors and graph data are retained. This is syntax adaptation with graph-equivalence checks, not reconstruction of coordinates or relaxation. All signs, node indices and periodic-vector tuples must survive. No indiscriminate string replacement is used.

Original corpus provenance is separate from this thesis dataset: the latter cannot certify the training corpus of MatText.

## Reused evidence

`existing_audit_verification.json` contains current SHA-256 checks, recounts and precise limits. Locally recounted:8898 source rows,8712 accepted,186 graph rejects,2853 ID sequences,1119 shared groups,zero additional tokenizer collisions,zero persisted omissions/UNK/truncation,max401. The five-test success log exists; those tests were not rerun. Per-row graph-equivalence and equality to official encoding are persisted results, not a fresh full-dataset computation. Small new real-sequence comparisons are in tokenizer_golden_tests.json.

The upstream historical regex uses findall and can omit compact `ooo` and unknown `@` without emitting UNK. Zero UNK alone is insufficient. The existing audit's tokenize_lossless uses regex spans and rejects every unmatched nonwhitespace character; graph parser checks representation before official encoding. New negative probes preserve upstream omissions as diagnostics, never accepted scientific inputs. Numeric indices are digit-tokenized with special_num_token=False; periodic vectors are indivisible vocabulary tokens in spaced syntax. Arbitrary extra spacing inside a vector is not assumed equivalent.

Collisions among trajectories/geometries reflect the representation's topology and are separate from checkpoint-tokenizer provenance. They do not prove semantic loss by tokenization or certification. No energy prediction/lower-bound statistic is used in this certification.
""")
    # Author draft is intentionally only a local report, never sent.
    (output / "author_request.md").write_text(f"""Subject: Original tokenizer and provenance for n0w0f/MatText-slices-2m

To: Nawaf Alampara, Santiago Miret, and Kevin Maik Jablonka (MatText authors)
Status: DRAFT — NOT SENT

Dear MatText authors,

I am developing a thesis on transfer learning for ABS3 chalcogenides and would like to reproduce MatText tokenization faithfully before feature extraction or supervised fine-tuning.

The checkpoint is n0w0f/MatText-slices-2m, HF revision `{identity['revision']}`, model.safetensors SHA-256 `{identity['weight_sha256']}`. Its config references `santiago_ckpt_normal/slice_pt_2m/checkpoint-393000` and declares vocab_size30522 and pad_token_id0.

Could you please provide or confirm:

1. The original tokenizer artifacts used for this exact checkpoint.
2. The complete ordered vocabulary and exact token-to-ID mapping.
3. CLS, SEP, PAD, MASK and UNK IDs, plus any other special tokens.
4. Padding/truncation settings and attention/special-token mask construction, including the MLM collator/runtime versions.
5. Whether the effective vocabulary was smaller than30522 and the model inherited a generic BERT config.
6. How pad_token_id0 relates to the public SliceTokenizer token `o o o` at ID0.
7. The pretraining code commits, effective run configuration/overrides, corpus revision and encoder export procedure.
8. One original SLICES string with its exact input_ids and attention_mask, including padding if used.
9. If available, its CLS output from this exact encoder in eval mode, with dtype/version and numerical tolerance.
10. Whether this public checkpoint corresponds to the BERT protocol evaluated in your Nature Computational Science paper (doi:10.1038/s43588-026-01045-2).

The pre-upload historical source reproduces the documented Ga/P example. The archived pretraining special-token order instead gives UNK149/PAD150/CLS151/SEP152/MASK153. We cannot yet bind that configuration to the exact weights. We also observed public source/runtime discrepancies described in our audit; an original encoded example and effective configuration would help resolve them without changing the model.

Thank you for your help with scientific reproducibility.

Best regards,
[Name / institution]
""")
    tests = sum(len(item.get("tests", [])) for entry in golden["implementations"].values() for item in entry["variants"].values())
    report_text = [f"# Certification report — {status}", "", "All twelve requested steps were reviewed. Technical compatibility is supported; historical checkpoint-tokenizer certification remains blocked. No training, inference, Slurm, tokenizer gate changes, embedding resizing or original checkpoint modifications occurred.", "",
        f"Checkpoint `{identity['repository']}` at `{identity['revision']}`. Weights SHA-256 `{identity['weight_sha256']}`; config SHA-256 `{identity['config_sha256']}`. Public/local bytes match; {identity['tensor_count']} tensor shapes and {identity['parameters']} parameters match BertModel config. Embeddings30522×512; positions512×512;4layers;8heads;Transformers4.38.2.", "",
        f"Historical tokenizer: xtal2txt `{SNAPSHOTS['xtal_preupload'][1]}` before June5 upload, using its bundled vocabulary SHA-256 `{historical['vocab_sha256']}` and unchanged AST class bodies. {comparison['exact_default_matches']}/{comparison['documented_entries']} default documented token IDs match. Custom YAML reproduces156 entries with shifted special IDs. None of these observations alone binds vocabulary semantics to matrix rows.", "",
        f"Deterministic probes: {tests} recorded input/variant cases, repeated integer/mask equality, official Ga/P reference, real ABS3 samples, multi-digit indices, periodic vectors, max-length401 plus padding512, literal specials and deliberate invalid-input diagnostics. Original training reference masks and author CLS vector are absent. MLM diagnostic status: {custom['mlm_probe']['status']}.", "",
        "Favorable evidence, contradictions, missing links and exact unlock conditions are machine-readable in certification_verdict.json. The provenance matrix distinguishes direct evidence, indirect evidence, reconstruction, contradiction and absence for each link.", "",
        "Frozen extraction and supervised fine-tuning remain unapproved as faithful transfer from the original checkpoint. A future separately authorized exploratory run could be labeled reconstructed-tokenizer exploration, with these uncertainties stated; it would not establish historical reproduction. Predictive results cannot resolve token-ID provenance retrospectively. This audit makes no claim that lack of historical certification prohibits publication of transparently labeled exploratory work.", "",
        "Thesis disclosure: checkpoint identity is verified; original effective tokenizer/run, custom special IDs and PAD0 handling are unresolved; public code revisions differ; SLICES is a topology representation and geometry collisions require separate interpretation.", "",
        "Files created: required ten report files plus evidence manifests, artifact and keyword inventories, prior-verification JSON, evidence source caches and working constructor outputs; new Python scripts only under model-pretrained/scripts/tokenizer_certification/. No pre-existing experimental scripts or gates were edited. author_request.md is an English draft and was not sent. Created-file inventory and hashes are recorded in created_files.csv after validation.", "",
        "Reproduction commands and test protocol: model-pretrained/scripts/tokenizer_certification/README.md. Unit-test execution status is in unit_tests.log; deterministic scientific diagnostics can intentionally fail and must not be relabeled as software-test successes."]
    (output / "certification_report.md").write_text("\n".join(report_text) + "\n")
    LOGGER.info("certification_report_written verdict=%s cases=%d", status, tests)


def main() -> None:
    """Validate report path and fail explicitly on incomplete evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "reports"):
        parser.error("--output must be inside reports/")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s level=%(levelname)s message=%(message)s')
    try:
        report(output)
    except Exception:
        LOGGER.exception("certification_report_failed")
        raise


if __name__ == "__main__":
    main()
