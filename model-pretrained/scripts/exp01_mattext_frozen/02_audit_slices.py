"""Audit SLICES syntax, lossless strategy conversion and original-tokenizer evidence.

Reads preserved SLICES and immutable upstream references. Writes candidate token
IDs, collision tables and compatibility reports. Candidate IDs alone never unlock
GPU extraction: checkpoint-specific tokenizer provenance is mandatory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import (LOGGER, adapt_slices, audit_fingerprint, cli, load_config,
                   parse_slices, read_json, sha256, tokenize_lossless, write_json)

REFERENCES = {
    "slice_vocab.txt": "https://zenodo.org/records/11484062/files/slice_vocab.txt?download=1",
    "xtal2txt_tokenizer_source.txt": "https://raw.githubusercontent.com/lamalab-org/xtal2txt/c6cf9593596ce99f054f1ebbdc4346d7dbec99fb/src/xtal2txt/tokenizer.py",
    "mattext_pretrain_template.yaml": "https://raw.githubusercontent.com/lamalab-org/MatText/2a594a4c69237da2daccab04e0515cf8cdcd715d/conf/model/archived_base/pretrain_template.yaml",
    "mattext_tokenizers.md": "https://raw.githubusercontent.com/lamalab-org/MatText/2a594a4c69237da2daccab04e0515cf8cdcd715d/docs/tokenizers.md",
}


def fetch_references(directory: Path) -> None:
    """Download fixed official source URLs into directory; verify Zenodo vocab MD5."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in REFERENCES.items():
        destination = directory / name
        if not destination.exists():
            with urllib.request.urlopen(url, timeout=40) as response:
                body = response.read()
            if name == "slice_vocab.txt" and hashlib.md5(body).hexdigest() != "ea6ea28b3e7bf8a5a7e2f4f514968639":
                raise ValueError("Published Zenodo vocabulary MD5 mismatch")
            destination.write_bytes(body)
        if name == "slice_vocab.txt" and hashlib.md5(destination.read_bytes()).hexdigest() != "ea6ea28b3e7bf8a5a7e2f4f514968639":
            raise ValueError("Local reference vocabulary MD5 mismatch")
        records.append({"file": name, "url": url, "sha256": sha256(destination)})
    write_json(directory / "reference_manifest.json", records)


def official_worker(input_file: Path, output_file: Path) -> None:
    """Validate at most three real graph serializations with official SLICES; save evidence."""
    from importlib.metadata import version
    from slices.core import SLICES
    converter = SLICES(graph_method="econnn", check_results=False, relax_model=None)
    rows = []
    for row in pd.read_csv(input_file).head(3).to_dict("records"):
        original = row["slices_original"]
        converted = adapt_slices(original)
        converter.from_SLICES(original, strategy=4, fix_duplicate_edge=False)
        initial = [np.array(item, copy=True) for item in
                   (converter.atom_types, converter.edge_indices, converter.to_jimages)]
        converter.from_SLICES(converted, strategy=3, fix_duplicate_edge=False)
        final = [converter.atom_types, converter.edge_indices, converter.to_jimages]
        equivalent = all(np.array_equal(first, second) for first, second in zip(initial, final))
        rows.append({"structure_id": row["structure_id"], "graph_equivalent": equivalent,
                     "basic_valid": bool(converter.check_SLICES_basic(original, strategy=4))})
    write_json(output_file, {"version": version("slices"), "samples": rows,
                             "passed": all(row["graph_equivalent"] and row["basic_valid"] for row in rows)})


def audit(config: dict[str, Any], fetch: bool = False) -> dict[str, Any]:
    """Audit config's real dataset; save compatibility/collisions and return gate status."""
    paths = config["paths"]
    audit_dir = paths["reports"] / "audit"
    reference = audit_dir / "reference"
    if fetch:
        fetch_references(reference)
    vocabulary_path = reference / "slice_vocab.txt"
    if not vocabulary_path.exists():
        raise FileNotFoundError("Run this audit once with --fetch-reference to retrieve the original vocabulary")
    base = vocabulary_path.read_text().splitlines()
    vocab = {token: index for index, token in enumerate(base)}
    if len(base) != len(vocab):
        raise ValueError("Duplicated tokens in original published vocabulary")
    for token in config["tokenizer"]["special_tokens_order"]:
        if token not in vocab:
            vocab[token] = len(vocab)
    write_json(config["root"] / config["tokenizer"]["vocab"], vocab)
    model_config = read_json(paths["checkpoint"] / "config.json")
    frame = pd.read_csv(paths["processed"] / "dataset.csv")
    results, encoded, problems = [], [], []
    for row in frame.to_dict("records"):
        item = {"structure_id": row["structure_id"], "equivalent": False, "candidate_valid": False}
        try:
            text = adapt_slices(row["slices_original"])
            if parse_slices(text, 3) != parse_slices(row["slices_prior_strategy3"], 3):
                raise ValueError("Adapter disagrees with prior official strategy-3 export")
            tokens, ids = tokenize_lossless(text, vocab, config["tokenizer"]["max_length"])
            if max(ids) >= model_config["vocab_size"]:
                raise ValueError("Token ID exceeds checkpoint embedding matrix")
            item.update(equivalent=True, candidate_valid=True, length=len(ids),
                        token_ids_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        token_bag_hash=hashlib.sha256(json.dumps(sorted(Counter(tokens).items())).encode()).hexdigest(),
                        error="")
            encoded.append({"structure_id": row["structure_id"], "slices_original": row["slices_original"],
                            "slices_model_input": text, "input_ids": ids,
                            "equivalence_status": "species_indices_edges_periodic_vectors_preserved",
                            "candidate_only": True})
        except (ValueError, KeyError) as error:
            item["error"] = str(error)
            problems.append({"structure_id": row["structure_id"], "error": str(error)})
        results.append(item)
    token_audit = pd.DataFrame(results)
    token_audit.to_csv(audit_dir / "tokenizer_per_structure.csv", index=False)
    candidate_path = paths["processed"] / "tokenized_candidate.jsonl"
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in encoded))
    pairs = []
    merged = frame.merge(token_audit, on="structure_id", validate="one_to_one")
    for formula, group in merged.groupby("reduced_formula"):
        for first, second in itertools.combinations(group.to_dict("records"), 2):
            ids_equal = bool(first.get("token_ids_hash") == second.get("token_ids_hash") and first["candidate_valid"] and second["candidate_valid"])
            pairs.append({"reduced_formula": formula, "first_id": first["structure_id"],
                          "second_id": second["structure_id"],
                          "same_slices": first["slices_original"] == second["slices_original"],
                          "same_token_bag": first.get("token_bag_hash") == second.get("token_bag_hash"),
                          "same_candidate_token_ids": ids_equal,
                          "potentially_identical_embeddings": ids_equal,
                          "delta_energy_ev_atom": abs(first["energy_per_atom_ev"] - second["energy_per_atom_ev"])})
    collision_frame = pd.DataFrame(pairs)
    collision_frame.to_csv(audit_dir / "polymorph_collision_audit.csv", index=False)
    # The isolated process prevents SLICES/TensorFlow from altering the encoder's
    # CUDA environment. This only parses graphs; no structure reconstruction.
    worker_file = audit_dir / "official_slices_api.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3")
    command = [sys.executable, str(Path(__file__).resolve()), "--official-worker",
               str(paths["processed"] / "dataset.csv"), str(worker_file)]
    try:
        worker = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90, check=False)
        (audit_dir / "official_slices_api.log").write_text(worker.stdout + worker.stderr)
        official_ok = worker.returncode == 0 and read_json(worker_file).get("passed", False)
    except (subprocess.TimeoutExpired, OSError) as error:
        (audit_dir / "official_slices_api.log").write_text(str(error))
        official_ok = False
    blockers = []
    if problems:
        blockers.append(f"{len(problems)} inputs have omissions, representation mismatch or overflow")
    if not official_ok:
        blockers.append("Official SLICES API sample verification failed; inspect official_slices_api.log")
    # A reference implementation/default vocabulary does not establish which
    # special IDs and wrapping behavior were used for this released checkpoint.
    evidence_path = config["root"] / config["tokenizer"]["evidence"]
    evidence_ok = False
    if evidence_path.exists():
        evidence = read_json(evidence_path)
        artifact_path = config["root"] / evidence.get("original_artifact", "__missing__")
        artifact = read_json(artifact_path) if artifact_path.is_file() else {}
        evidence_ok = (artifact.get("vocab") == vocab
                       and evidence.get("checkpoint_sha256") == sha256(paths["checkpoint"] / "model.safetensors")
                       and evidence.get("artifact_sha256") == sha256(artifact_path)
                       and bool(evidence.get("published_checkpoint_link"))
                       and artifact.get("special_wrapping") == "single_CLS_SEP")
    if not evidence_ok:
        blockers.append("No checkpoint-specific original tokenizer artifact proves vocabulary IDs and CLS/SEP wrapping")
    if not blockers:
        (paths["processed"] / "tokenized.jsonl").write_text("".join(
            json.dumps({**row, "candidate_only": False}) + "\n" for row in encoded))
    report = {"status": "failed" if blockers else "passed", "blockers": blockers,
              "samples": len(frame), "candidate_tokenization_valid": len(encoded),
              "candidate_only": not evidence_ok, "official_api_verified": official_ok,
              "candidate_vocab_size": len(vocab), "embedding_rows": model_config["vocab_size"],
              "special_ids_candidate": {token: vocab[token] for token in config["tokenizer"]["special_tokens_order"]},
              "max_candidate_length": int(token_audit.length.max()) if "length" in token_audit else None,
              "candidate_input_collisions": int(collision_frame.same_candidate_token_ids.sum()),
              "vocab_sha256": sha256(config["root"] / config["tokenizer"]["vocab"]),
              "fingerprint": audit_fingerprint(config)}
    write_json(audit_dir / "tokenizer_audit.json", report)
    example = encoded[0] if encoded else {}
    text = ("# Compatibilidad SLICES / MatText\n\n"
            f"Estado: **{report['status']}**. {len(frame)} estructuras auditadas; "
            f"{len(encoded)} entradas candidatas sin pérdida de caracteres no blancos.\n\n"
            "La fuente previa usó SLICES 2.0.12, econnn, relax_model=None. El adaptador "
            "cambia exclusivamente cada vector periódico compacto a tres componentes. "
            "Se comparan especies, índices, aristas y vectores de todas las filas. "
            f"Comprobación adicional de tres grafos con API oficial: {official_ok}.\n\n"
            f"Vocabulario candidato: {len(vocab)} IDs; matriz: {model_config['vocab_size']} filas. "
            "Que los IDs quepan en la matriz no demuestra su correspondencia semántica. "
            "La documentación por defecto y la configuración archivada añaden tokens especiales "
            "distintos. El checkpoint no incluye un tokenizador serializado. "
            "No se amplía el vocabulario ni se altera ninguna matriz de embeddings.\n\n"
            f"Tokens especiales candidatos: `{report['special_ids_candidate']}`. "
            f"Longitud máxima candidata: {report['max_candidate_length']}; "
            f"colisiones de IDs candidatos entre polimorfos: {report['candidate_input_collisions']}.\n\n"
            "Las colisiones se conservan en polymorph_collision_audit.csv con diferencias GGA. "
            "Igual bolsa de tokens no implica igual entrada para un encoder posicional.\n\n"
            f"Ejemplo original:\n```text\n{example.get('slices_original', '')}\n```\n"
            f"Adaptado:\n```text\n{example.get('slices_model_input', '')}\n```\n\n"
            "Problemas críticos:\n" + "".join(f"- {value}\n" for value in blockers) +
            "\nConclusión: la adaptación sintáctica y el vocabulario candidato se auditan "
            "separadamente de la compatibilidad exacta con el checkpoint. No se autoriza "
            "extracción científica ni envío GPU mientras exista un control crítico fallido.\n\n"
            "Fuentes y hashes: reference/reference_manifest.json. Código oficial: "
            "https://github.com/lamalab-org/MatText; vocabulario original: "
            "https://zenodo.org/records/11484062.\n")
    (audit_dir / "tokenizer_compatibility.md").write_text(text)
    LOGGER.info("tokenizer samples=%d candidate_valid=%d blockers=%s", len(frame), len(encoded), blockers)
    return report


def main() -> None:
    """Run CLI audit/reference retrieval or isolated official parser worker."""
    parser = cli(__doc__)
    parser.add_argument("--fetch-reference", action="store_true")
    parser.add_argument("--official-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    args = parser.parse_args()
    if args.official_worker:
        official_worker(*map(Path, args.official_worker))
        return
    report = audit(load_config(args.config), args.fetch_reference)
    if report["status"] != "passed":
        raise SystemExit("Critical tokenizer audit failed; see tokenizer_compatibility.md")


if __name__ == "__main__":
    main()
