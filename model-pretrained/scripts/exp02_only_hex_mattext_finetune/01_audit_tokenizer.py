"""Audit every only_hex_new SLICES frame against historical official tokenizer behavior.

Check graph-preserving syntax, exact reference IDs, coverage, lengths, masks,
source/split/energy provenance and topology collisions. Separate reproducible
official-code compatibility from unresolved checkpoint-specific training evidence.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from workflow import (LOGGER, ROOT, cli, fingerprint, load_config, read_json,
                      sha256, write_json)


def existing_graph_helpers() -> Any:
    """Load the existing experiment's graph/token helpers by Path; return module without executing main."""
    path = ROOT / "model-pretrained/scripts/exp01_mattext_frozen/utils.py"
    spec = importlib.util.spec_from_file_location("exp01_graph_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_evidence(config: dict[str, Any], vocab: dict[str, int]) -> tuple[bool, str]:
    """Verify optional published tokenizer evidence against exact weights/vocab; return bool and reason."""
    evidence_path = ROOT / config["tokenizer"]["checkpoint_evidence"]
    if not evidence_path.exists():
        return False, "No released original tokenizer or checkpoint-specific published ID/wrapping artifact was found"
    evidence = read_json(evidence_path)
    allowed = lambda url: (url.startswith("https://huggingface.co/n0w0f/MatText-slices-2m/")
                           or url.startswith("https://github.com/lamalab-org/")
                           or url.startswith("https://raw.githubusercontent.com/lamalab-org/")
                           or url.startswith("https://zenodo.org/records/")
                           or url.startswith("https://zenodo.org/api/records/"))
    artifact_url = evidence.get("original_artifact_url", "")
    publication_url = evidence.get("published_checkpoint_link", "")
    if not allowed(artifact_url) or not allowed(publication_url):
        return False, "Evidence must identify public author-owned artifact and checkpoint-link sources"
    artifact_path = ROOT / evidence["original_artifact"]
    artifact = read_json(artifact_path)
    try:
        with urllib.request.urlopen(artifact_url, timeout=45) as response:
            if not allowed(response.url):
                return False, "Published tokenizer artifact redirected outside primary sources"
            published = response.read()
        with urllib.request.urlopen(publication_url, timeout=45) as response:
            if not allowed(response.url):
                return False, "Checkpoint evidence redirected outside primary sources"
            binding = response.read().decode("utf-8", errors="replace")
    except Exception as error:
        return False, f"Could not verify published original tokenizer evidence: {error}"
    weight_hash = sha256(config["paths"]["checkpoint"] / "model.safetensors")
    binds_exact_checkpoint = (weight_hash in binding or
        "n0w0f/MatText-slices-2m/" in binding and "c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e" in binding)
    artifact_vocab = artifact.get("vocab", artifact.get("model", {}).get("vocab", artifact))
    # Existence of a URL/boolean alone never authorizes training: require an
    # archived original artifact tied to these weights and this exact mapping.
    valid = (evidence.get("evidence_type") == "published_checkpoint_tokenizer"
             and evidence.get("checkpoint_revision") == "c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e"
             and evidence.get("checkpoint_sha256") == weight_hash
             and evidence.get("artifact_sha256") == sha256(artifact_path)
             and hashlib.sha256(published).hexdigest() == sha256(artifact_path)
             and artifact_vocab == vocab
             and evidence.get("special_wrapping") == "single_CLS_SEP"
             and binds_exact_checkpoint)
    return valid, "Verified original artifact" if valid else "Published tokenizer evidence does not match this checkpoint"


def audit(config: dict[str, Any]) -> dict[str, Any]:
    """Audit configured complete conversion; write real row-level evidence and return scientific gate dict."""
    paths = config["paths"]
    directory = paths["reports"] / "audit"
    tables = paths["reports"] / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    paths["processed"].mkdir(parents=True, exist_ok=True)
    conversion = read_json(paths["conversion_manifest"])
    if conversion["status"] not in {"completed", "completed_with_rejections"}:
        raise ValueError("Complete SLICES conversion is not available")
    if conversion["output_sha256"][paths["slices_csv"].name] != sha256(paths["slices_csv"]):
        raise ValueError("SLICES output checksum mismatch")
    if conversion["csv_sha256"] != sha256(paths["source_csv"]):
        raise ValueError("Original exported CSV changed")
    reference = read_json(directory / "reference_tokenizer.json")
    if reference["status"] != "passed" or reference["versions"]["transformers"] != "4.38.2":
        raise ValueError("Historical official tokenizer runtime probe has not passed")
    probe = importlib.import_module("00_probe_reference_tokenizer")
    source = directory / "reference" / config["tokenizer"]["source_file"]
    if sha256(source) != reference["source_sha256"]:
        raise ValueError("Historical tokenizer source changed since runtime probe")
    tokenizer = probe.official_class(source, directory / "tokenizer_work" / "slice_vocab.txt")(
        vocab_file=str(directory / "tokenizer_work" / "slice_vocab.txt"),
        special_tokens=reference["special_tokens"], special_num_token=False,
        model_max_length=512, truncation=False, padding=False)
    vocab = reference["vocab"]
    if tokenizer.get_vocab() != vocab:
        raise ValueError("Reference tokenizer mapping differs from reconstructed artifact")
    helpers = existing_graph_helpers()
    model_config = read_json(paths["checkpoint"] / "config.json")
    manifest = read_json(paths["checkpoint_manifest"])
    expected = next(r["sha256"] for r in manifest["files"] if r["path"].endswith("model.safetensors"))
    if sha256(paths["checkpoint"] / "model.safetensors") != expected:
        raise ValueError("Original model weights changed")
    # Inspect safetensors header without loading the 33-million-parameter encoder.
    import struct
    with (paths["checkpoint"] / "model.safetensors").open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    embedding_shape = header["embeddings.word_embeddings.weight"]["shape"]
    if embedding_shape != [model_config["vocab_size"], model_config["hidden_size"]]:
        raise ValueError("Checkpoint embedding shape differs from model configuration")
    csv.field_size_limit(16 * 1024 * 1024)
    with paths["slices_csv"].open() as stream:
        rows = list(csv.DictReader(stream))
    with paths["source_csv"].open() as stream:
        originals = list(csv.DictReader(stream))
    if len(rows) != len(originals) or len(rows) != conversion["input_frames"]:
        raise ValueError("Conversion lost or added frame rows")
    for row, original in zip(rows, originals):
        if any(row[key] != value for key, value in original.items()):
            raise ValueError(f"CSV conversion changed source values: {original['config_id']}")
    results, encoded = [], []
    collisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    formula_splits, static_families = defaultdict(set), defaultdict(set)
    input_ids_seen = set()
    for index, row in enumerate(rows):
        if row["config_id"] in input_ids_seen:
            raise ValueError("Duplicate source frame identifier")
        input_ids_seen.add(row["config_id"])
        formula_splits[row["reduced_formula"]].add(row["split"])
        item = {"structure_id": row["structure_id"], "reduced_formula": row["reduced_formula"],
                "split": row["split"], "polymorph_family": row["polymorph_family"],
                "data_origin": row["data_origin"], "status": "conversion_rejected", "length": "",
                "graph_equivalent": False, "exact_official_ids": False,
                "nonwhitespace_omissions": "", "unknown_tokens": "", "truncated": False,
                "input_ids_sha256": "", "token_bag_sha256": "", "error": row["error"]}
        if row["status"] == "accepted":
            try:
                text = helpers.adapt_slices(row["slices_original"])
                if helpers.parse_slices(text, 3) != helpers.parse_slices(row["slices_model_input"], 3):
                    raise ValueError("Adapted graph differs from official strategy-3 export")
                tokens, ids = helpers.tokenize_lossless(text, vocab, config["tokenizer"]["max_length"])
                actual = tokenizer(text, truncation=False, padding=False)
                if actual["input_ids"] != ids or actual["attention_mask"] != [1] * len(ids):
                    raise ValueError("Historical official HF call disagrees with audited IDs/masks")
                if any(not 0 <= token_id < embedding_shape[0] for token_id in ids):
                    raise ValueError("Token ID does not address an existing checkpoint embedding row")
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                item.update(status="compatible_with_official_code", length=len(ids), graph_equivalent=True,
                            exact_official_ids=True, nonwhitespace_omissions=0, unknown_tokens=0,
                            input_ids_sha256=digest, token_bag_sha256=hashlib.sha256(json.dumps(sorted(Counter(tokens).items())).encode()).hexdigest(), error="")
                encoded.append({"structure_id": row["structure_id"], "reduced_formula": row["reduced_formula"],
                    "polymorph_family": row["polymorph_family"], "data_origin": row["data_origin"],
                    "source_key": row["source_key"], "frame_index": row["frame_index"], "split": row["split"],
                    "energy_per_atom_ev": float(row["energy_per_atom_ev"]),
                    "slices_original": row["slices_original"], "slices_model_input": text,
                    "input_ids": ids, "candidate_only": True})
                collisions[digest].append(encoded[-1])
                if row["data_origin"] == "static":
                    static_families[row["reduced_formula"]].add(row["polymorph_family"])
            except (ValueError, KeyError) as error:
                item.update(status="tokenization_rejected", error=str(error))
        results.append(item)
        if (index + 1) % 1000 == 0:
            LOGGER.info("tokenizer_audit_progress frames=%d compatible=%d", index + 1, len(encoded))
    if any(len(values) != 1 for values in formula_splits.values()):
        raise ValueError("Compositions cross original source partitions")
    with (directory / "tokenizer_per_structure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    candidate = paths["processed"] / "tokenized_candidate.jsonl"
    candidate.write_text("".join(json.dumps(row) + "\n" for row in encoded))
    collision_rows = []
    static_collision_groups = 0
    tokenizer_induced_groups = 0
    for digest, group in collisions.items():
        if len(group) < 2:
            continue
        static = [r for r in group if r["data_origin"] == "static"]
        static_collision = len({r["polymorph_family"] for r in static}) > 1
        static_collision_groups += static_collision
        distinct_slices = len({r["slices_original"] for r in group})
        tokenizer_induced_groups += distinct_slices > 1
        energies = [r["energy_per_atom_ev"] for r in group]
        collision_rows.append({"input_ids_sha256": digest, "frames": len(group),
            "reduced_formula": group[0]["reduced_formula"], "split": group[0]["split"],
            "families_json": json.dumps(sorted({r["polymorph_family"] for r in group})),
            "energy_range_ev_atom": max(energies) - min(energies),
            "different_static_polymorphs": static_collision,
            "distinct_slices": distinct_slices,
            "tokenizer_induced_collision": distinct_slices > 1,
            "ids_json": json.dumps([r["structure_id"] for r in group])})
    with (tables / "tokenized_collisions.csv").open("w", newline="") as stream:
        fields = ["input_ids_sha256", "frames", "reduced_formula", "split", "families_json",
                  "energy_range_ev_atom", "different_static_polymorphs", "distinct_slices",
                  "tokenizer_induced_collision", "ids_json"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(collision_rows)
    expected_families = {"dist_perovskite", "hexagonal", "needle"}
    with (tables / "static_composition_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["reduced_formula", "split", "families", "complete_three_families"])
        writer.writeheader()
        for formula, values in sorted(formula_splits.items()):
            writer.writerow({"reduced_formula": formula, "split": next(iter(values)),
                             "families": json.dumps(sorted(static_families[formula])),
                             "complete_three_families": static_families[formula] == expected_families})
    proof, reason = checkpoint_evidence(config, vocab)
    blockers = []
    statuses = Counter(r["status"] for r in results)
    if statuses["tokenization_rejected"]:
        blockers.append(f"{statuses['tokenization_rejected']} accepted SLICES frames fail tokenization/length checks")
    if tokenizer_induced_groups:
        blockers.append(f"{tokenizer_induced_groups} groups contain different SLICES strings mapped to identical token IDs")
    if not proof:
        blockers.append(reason)
    # Collisions are representation limitations, never grounds for removing
    # physically different frames or selecting a new tokenizer using test labels.
    train_errors = []
    for group in collisions.values():
        if group[0]["split"] == "train":
            energies = sorted(r["energy_per_atom_ev"] for r in group)
            n = len(energies)
            median = (energies[(n - 1) // 2] + energies[n // 2]) / 2
            train_errors.extend(abs(value - median) for value in energies)
    if proof and not blockers:
        for row in encoded:
            row["candidate_only"] = False
        (paths["processed"] / "tokenized.jsonl").write_text("".join(json.dumps(r) + "\n" for r in encoded))
    report = {"status": "failed" if blockers else "passed", "blockers": blockers,
        "official_code_compatibility": "passed" if not statuses["tokenization_rejected"] else "failed",
        "checkpoint_specific_id_provenance": "passed" if proof else "not_established",
        "source_frames": len(rows), "status_counts": dict(statuses), "vocab_size": len(vocab),
        "embedding_shape": embedding_shape, "special_ids": reference["special_ids"],
        "special_wrapping": reference["special_wrapping"], "max_length": max((r["length"] for r in results if r["length"] != ""), default=None),
        "all_source_values_preserved": True, "composition_leakage": False,
        "compatible_split_frames": dict(Counter(r["split"] for r in encoded)),
        "compatible_data_origins": dict(Counter(r["data_origin"] for r in encoded)),
        "unique_tokenized_inputs": len(collisions), "collision_groups": len(collision_rows),
        "static_polymorph_collision_groups": static_collision_groups,
        "tokenizer_induced_collision_groups": tokenizer_induced_groups,
        "training_only_representation_mae_lower_bound_ev_atom": sum(train_errors) / len(train_errors) if train_errors else None,
        "lower_bound_policy": "Train-only descriptive conditional-median bound; not a predictive baseline or test metric",
        "static_compositions_with_all_families": sum(f == expected_families for f in static_families.values()),
        "config_pad_token_id": model_config["pad_token_id"],
        "tokenizer_pad_token_id": reference["special_ids"]["pad_token"],
        "padding_policy": "Explicit length-based attention mask; do not equate structural ID 0 with padding or resize embeddings",
        "paper": "https://doi.org/10.1038/s43588-026-01045-2",
        "paper_code_archive": "https://doi.org/10.5281/zenodo.21151839",
        "candidate_jsonl_sha256": sha256(candidate), "fingerprint": fingerprint(config)}
    write_json(directory / "tokenizer_audit.json", report)
    summary = ["# Auditoría de tokenización basada en MatText y el paper", "",
        f"Estado científico global: **{report['status']}**.", "",
        f"Se comprobaron las {len(rows)} filas de only_hex_new. Estados: {dict(statuses)}.",
        f"Compatibilidad con la implementación oficial histórica: **{report['official_code_compatibility']}**.",
        f"Procedencia de IDs vinculada al checkpoint específico: **{report['checkpoint_specific_id_provenance']}**.", "",
        "## Comprobaciones observadas", "",
        f"Vocabulario original: {len(vocab)} IDs. Embeddings: {embedding_shape}. Especiales: {reference['special_ids']}.",
        f"Envoltura reproducida: {reference['special_wrapping']}; longitud máxima observada: {report['max_length']}.",
        "Todas las entradas compatibles se compararon ID por ID y máscara por máscara con __call__ del tokenizador oficial de 2024 ejecutado con Transformers 4.38.2. Se rechazaron omisiones y truncamientos; se comprobó equivalencia de especies, índices, aristas e imágenes periódicas.",
        "Las estructuras rechazadas por SLICES se conservan con su motivo; no se sustituyen ni se etiquetan como problemas del tokenizador.",
        f"Particiones compatibles: {report['compatible_split_frames']}. Sin composición compartida entre train/val/test.", "",
        "## Colisiones y límite de la representación", "",
        f"{len(collisions)} entradas tokenizadas únicas; {len(collision_rows)} grupos compartidos por varios frames; {static_collision_groups} grupos con polimorfos estáticos de familias distintas.",
        f"Grupos de cadenas SLICES distintas que el tokenizador fusiona en los mismos IDs: {tokenizer_induced_groups}. Se separan de las colisiones ya presentes en la representación de origen.",
        f"Límite descriptivo MAE por entradas idénticas en entrenamiento: {report['training_only_representation_mae_lower_bound_ev_atom']} eV/átomo. No es una métrica de generalización.",
        "Las colisiones se registran sin deduplicar. Una topología común puede ocultar geometrías y energías distintas a lo largo de una trayectoria; el fine-tuning no puede separar entradas determinísticamente idénticas.", "",
        "## Procedencia y bloqueos", "",
        "El paper y su archivo de código permiten reproducir el flujo BERT/SLICES. El checkpoint publicado solo contiene pesos/configuración/model card; no incluye el tokenizador de aquella ejecución. La documentación por defecto usa otro orden de especiales. El archivo finetune2m/slice.yaml incluso apunta a un checkpoint 30k_rt; no se reutiliza esa ruta como prueba del modelo 2m.",
        *[f"- {reason}" for reason in blockers], "",
        "El fine-tuning permanece bloqueado hasta satisfacer las comprobaciones críticas. Los IDs se guardan explícitamente como candidatos y no se presenta una reconstrucción del método como prueba de un artefacto original inexistente.", "",
        "Fuentes: [paper](https://doi.org/10.1038/s43588-026-01045-2), [código archivado](https://doi.org/10.5281/zenodo.21151839), [tokenizador histórico](https://github.com/lamalab-org/xtal2txt/blob/1d9d0c6f8675efea885ef7c9c87b3195cd79aaed/src/xtal2txt/tokenizer.py)."]
    (paths["reports"] / "tokenizer_compatibility.md").write_text("\n\n".join(summary) + "\n")
    LOGGER.info("tokenizer_audit_complete global=%s official=%s checkpoint_proof=%s", report["status"], report["official_code_compatibility"], proof)
    return report


def main() -> None:
    """Run all-frame audit from CLI; log descriptive exceptions and fail on critical blockers."""
    config = load_config(cli(__doc__).parse_args().config)
    try:
        result = audit(config)
    except Exception:
        LOGGER.exception("tokenizer_audit_failed")
        raise
    if result["status"] != "passed":
        raise SystemExit("AUDIT BLOCKED: inspect tokenizer_compatibility.md; no training was submitted")


if __name__ == "__main__":
    main()
