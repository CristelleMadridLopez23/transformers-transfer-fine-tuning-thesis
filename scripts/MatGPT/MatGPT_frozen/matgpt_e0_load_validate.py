#!/usr/bin/env python3
"""Load and audit the original unconditional MatGPT checkpoint without training.

This script uses the authors' ``ModifiedTransformerDecoder`` implementation,
the model metadata JSON distributed with the checkpoint, and the matching
``mat-concat/config/uncond.yaml`` configuration. All paths are configurable
and default to project-relative locations under ``Alternative2/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("matgpt_e0")
ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs "
    "Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    .split()
)
VECTOR_TOKEN = re.compile(r"^[+o-]{3}$")
INTEGER_TOKEN = re.compile(r"^-?\d+$")


class AuditError(RuntimeError):
    """Represent a descriptive error that prevents a requested audit stage."""


class JsonLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record and its structured fields.

        Args:
            record: Python logging record.

        Returns:
            A JSON-encoded log line.
        """
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key not in {
                "message", "asctime"
            }:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(verbose: bool) -> None:
    """Configure structured console logging.

    Args:
        verbose: Enable informational events when true.

    Returns:
        None.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO if verbose else logging.WARNING)
    LOGGER.propagate = False


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it into memory.

    Args:
        path: Existing file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_model_info(path: Path) -> tuple[dict[str, int], dict[int, str], int]:
    """Read and validate the checkpoint's token maps and context length.

    Args:
        path: MatGPT ``model_info_property_0.json`` path.

    Returns:
        ``(stoi, itos, block_size)`` with normalized integer IDs.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        stoi = {str(token): int(index) for token, index in raw["stoi"].items()}
        itos = {int(index): str(token) for index, token in raw["itos"].items()}
        vocab = {int(index): str(token) for index, token in raw["vocab"].items()}
        block_size = int(raw["block_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AuditError(f"No se pudo leer model_info JSON en {path}: {exc}") from exc

    ids = list(stoi.values())
    if len(set(ids)) != len(ids):
        raise AuditError("stoi asigna el mismo ID a más de un token.")
    if len(stoi) != len(itos) or any(itos.get(index) != token for token, index in stoi.items()):
        raise AuditError("stoi e itos no son inversos exactos; se detiene la carga estricta.")
    if vocab != itos:
        raise AuditError("El campo vocab no coincide exactamente con itos en model_info.")
    if set(ids) != set(range(len(stoi))):
        raise AuditError("Los IDs del vocabulario no son contiguos desde cero.")
    if block_size < 2:
        raise AuditError(f"block_size inválido: {block_size}.")
    required = {"<start>", "<end>", "<pad>"}
    missing = sorted(required - set(stoi))
    if missing:
        raise AuditError(f"Faltan tokens especiales requeridos por el código oficial: {missing}.")
    return stoi, itos, block_size


def encode_slices(sequence: str, stoi: dict[str, int]) -> tuple[list[str], list[int]]:
    """Encode SLICES using MatGPT's whitespace split and boundary tokens.

    Args:
        sequence: One SLICES sequence in its original whitespace format.
        stoi: Token-to-integer map from the checkpoint metadata.

    Returns:
        The token list and IDs with ``<start>`` / ``<end>`` boundaries.
    """
    tokens = sequence.split()
    unknown = sorted(set(tokens) - set(stoi))
    if unknown:
        raise AuditError(f"La cadena contiene tokens fuera del vocabulario: {unknown[:20]}.")
    wrapped_tokens = ["<start>", *tokens, "<end>"]
    return wrapped_tokens, [stoi[token] for token in wrapped_tokens]


def classify_tokens(tokens: Iterable[str]) -> dict[str, list[str]]:
    """Group observed SLICES tokens by transparent lexical rules.

    Args:
        tokens: Unique or repeated tokens observed in an input sequence.

    Returns:
        Token examples grouped as elements, integer indices, translation
        vectors, special tokens, and other tokens.
    """
    groups: dict[str, set[str]] = {
        "element_symbols": set(),
        "integer_tokens_indices_or_counts": set(),
        "translation_vector_tokens": set(),
        "special_tokens": set(),
        "other_tokens": set(),
    }
    for token in tokens:
        if token in ELEMENTS:
            groups["element_symbols"].add(token)
        elif token.startswith("<") and token.endswith(">"):
            groups["special_tokens"].add(token)
        elif INTEGER_TOKEN.fullmatch(token):
            groups["integer_tokens_indices_or_counts"].add(token)
        elif VECTOR_TOKEN.fullmatch(token):
            groups["translation_vector_tokens"].add(token)
        else:
            groups["other_tokens"].add(token)
    return {key: sorted(values) for key, values in groups.items()}


def select_real_example(
    csv_path: Path, stoi: dict[str, int], block_size: int, max_rows: int
) -> tuple[str, list[str], list[int]]:
    """Find the first real CSV SLICES row fully supported by this vocabulary.

    Args:
        csv_path: CSV with a column named ``SLICES``.
        stoi: Checkpoint token-to-ID map.
        block_size: Checkpoint's maximum context length.
        max_rows: Maximum number of rows inspected.

    Returns:
        Original sequence, wrapped token list, and encoded IDs.
    """
    if not csv_path.is_file():
        raise AuditError(f"No existe el CSV de ejemplos reales: {csv_path}.")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "SLICES" not in reader.fieldnames:
            raise AuditError(f"El CSV no tiene columna SLICES: {csv_path}.")
        checked = 0
        for row in reader:
            if checked >= max_rows:
                break
            checked += 1
            sequence = (row.get("SLICES") or "").strip()
            if not sequence:
                continue
            try:
                tokens, ids = encode_slices(sequence, stoi)
            except AuditError:
                continue
            if len(ids) <= block_size:
                return sequence, tokens, ids
    raise AuditError(
        f"No se encontró en {max_rows} filas un SLICES sin tokens desconocidos "
        f"y con longitud <= block_size={block_size}."
    )


def load_official_architecture(source_dir: Path) -> type:
    """Import the authors' architecture from the checked-out source copy.

    Args:
        source_dir: Local ``mat-concat`` source directory.

    Returns:
        The official ``ModifiedTransformerDecoder`` class.
    """
    if not (source_dir / "utils" / "models.py").is_file():
        raise AuditError(f"No se encuentra utils/models.py bajo {source_dir}.")
    source_path = str(source_dir.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        module = importlib.import_module("utils.models")
        return module.ModifiedTransformerDecoder
    except (ImportError, AttributeError) as exc:
        raise AuditError(
            "No se pudo importar la arquitectura oficial. Instala las dependencias "
            f"del entorno MatGPT y revisa {source_dir}: {exc}"
        ) from exc


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load the authors' YAML configuration and verify unconditional mode.

    Args:
        path: Official ``uncond.yaml`` path.

    Returns:
        Parsed YAML mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise AuditError("Falta PyYAML; instala las dependencias del entorno MatGPT.") from exc
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AuditError(f"No se pudo leer la configuración YAML {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise AuditError(f"La configuración no es un mapping YAML: {path}.")
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    expected = {
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "dropout": 0.1,
    }
    mismatches = {
        key: {"expected": value, "actual": model_cfg.get(key)}
        for key, value in expected.items()
        if model_cfg.get(key) != value
    }
    if mismatches:
        raise AuditError(f"La configuración no coincide con el MatGPT incondicional esperado: {mismatches}.")
    if train_cfg.get("conditioning_mode") != "unconditional" or int(train_cfg.get("num_properties", -1)) != 0:
        raise AuditError("La configuración local no identifica un modelo incondicional con cero propiedades.")
    if list(train_cfg.get("selected_properties", [])):
        raise AuditError("uncond.yaml contiene selected_properties no vacío.")
    return config


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write the audit report as JSON, creating its parent directory.

    Args:
        path: Report destination.
        report: JSON-serializable audit evidence.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with project-relative defaults.

    Returns:
        Validated command-line arguments.
    """
    project_root = Path(__file__).resolve().parents[2]
    model_root = project_root / "Alternative2" / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=model_root / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "transformer_property_0.pth",
        help="Original unconditional .pth state_dict.",
    )
    parser.add_argument(
        "--model-info", type=Path,
        default=model_root / "MatGPT-source" / "mat-concat" / "mat-uncond" / "weights" / "model_info_property_0.json",
        help="JSON accompanying the original checkpoint (vocab, stoi, itos, block_size).",
    )
    parser.add_argument(
        "--config", type=Path,
        default=model_root / "MatGPT-source" / "mat-concat" / "config" / "uncond.yaml",
        help="Official unconditional YAML configuration.",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        default=model_root / "MatGPT-source" / "mat-concat",
        help="Official mat-concat source directory containing utils/models.py.",
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=model_root / "MatGPT-source.download-manifest.json",
        help="Download manifest recording the official source commit SHA.",
    )
    parser.add_argument(
        "--slices-csv", type=Path,
        default=project_root / "Alternative2" / "01_data" / "toSLICES_output" / "abs3" / "matgpt_unconditional.csv",
        help="Real SLICES input CSV; the script reads its SLICES column.",
    )
    parser.add_argument(
        "--report", type=Path,
        default=project_root / "Alternative2" / "reports" / "matgpt_e0" / "load_validation.json",
        help="JSON audit report destination.",
    )
    parser.add_argument("--max-csv-rows", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_csv_rows < 1:
        parser.error("--max-csv-rows debe ser >= 1")
    return args


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Run strict checkpoint, tokenizer, architecture, and CPU inference checks.

    Args:
        args: Parsed CLI options.

    Returns:
        Audit report with ``PASS``, ``PARTIAL``, or ``BLOCKED`` status.
    """
    report: dict[str, Any] = {
        "experiment": "matgpt_e0_load_validate",
        "status": "BLOCKED",
        "training_performed": False,
        "device": "cpu",
        "inputs": {},
        "checks": {},
        "errors": [],
        "limitations": [
            "El smoke test verifica la interfaz; no valida estabilidad cristalográfica ni calidad generativa.",
            "No se crea una cabeza de regresión y no se actualizan parámetros.",
        ],
    }
    for key in ("checkpoint", "model_info", "config", "source_dir", "source_manifest", "slices_csv"):
        path = Path(getattr(args, key))
        report["inputs"][key] = str(path)
        if not path.exists():
            report["errors"].append(f"No existe {key}: {path}")
    if report["errors"]:
        report["status"] = "BLOCKED"
        return report

    try:
        import torch
    except ImportError as exc:
        raise AuditError(f"PyTorch no está disponible en este entorno: {exc}") from exc
    if not torch.cuda.is_available():
        LOGGER.info("cpu_selected", extra={"device": "cpu"})

    config = load_yaml_config(args.config)
    stoi, itos, block_size = read_model_info(args.model_info)
    vocab_size = len(stoi)
    mode = config["training"]["conditioning_mode"]
    info_json = json.loads(args.model_info.read_text(encoding="utf-8"))
    stored_mode = info_json.get("conditioning_mode")
    if stored_mode is not None and stored_mode != mode:
        raise AuditError(f"model_info declara conditioning_mode={stored_mode!r}; se esperaba {mode!r}.")

    sequence, wrapped_tokens, token_ids = select_real_example(
        args.slices_csv, stoi, block_size, args.max_csv_rows
    )
    restored_tokens = [itos[index] for index in token_ids]
    round_trip = " ".join(restored_tokens[1:-1])
    if round_trip != " ".join(sequence.split()):
        raise AuditError("Falló el round-trip SLICES → tokens → IDs → tokens → SLICES.")
    unknown_rejected = False
    try:
        encode_slices("__MATGPT_UNKNOWN_TOKEN__", stoi)
    except AuditError:
        unknown_rejected = True
    if not unknown_rejected:
        raise AuditError("La prueba de token desconocido no fue rechazada.")
    observed_tokens = sequence.split()
    report["checks"]["tokenizer"] = {
        "vocab_size": vocab_size,
        "vocab_matches_itos": True,
        "stoi_itos_bijective": True,
        "unique_ids": len(set(stoi.values())) == vocab_size,
        "special_token_ids": {token: stoi[token] for token in ("<start>", "<end>", "<pad>")},
        "block_size": block_size,
        "example_source_csv": str(args.slices_csv),
        "example_slices": sequence,
        "example_tokens_with_boundaries": wrapped_tokens,
        "example_ids": token_ids,
        "round_trip_exact_after_whitespace_normalization": round_trip == " ".join(sequence.split()),
        "unknown_token_rejected": unknown_rejected,
        "token_categories_observed": classify_tokens(observed_tokens),
        "sequence_length_including_boundaries": len(token_ids),
    }
    LOGGER.info("tokenizer_audited", extra={"vocab_size": vocab_size, "block_size": block_size})

    model_cfg = config["model"]
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    report["provenance"] = {
        "repository": source_manifest.get("repository"),
        "resolved_commit": source_manifest.get("resolved_commit"),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_info_sha256": sha256_file(args.model_info),
        "config_sha256": sha256_file(args.config),
    }
    model_class = load_official_architecture(args.source_dir)
    model = model_class(
        vocab_size=vocab_size,
        n_embd=int(model_cfg["n_embd"]),
        n_head=int(model_cfg["n_head"]),
        n_layer=int(model_cfg["n_layer"]),
        block_size=block_size,
        dropout=float(model_cfg["dropout"]),
        stoi=stoi,
        num_properties=0,
        conditioning_mode="unconditional",
    ).cpu()

    checkpoint_hash_before = sha256_file(args.checkpoint)
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AuditError(
            "No se pudo leer el .pth con torch.load(weights_only=True). No se intentó "
            f"deserialización insegura: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(state_dict, dict) or not state_dict:
        raise AuditError("El .pth no contiene un state_dict no vacío en el formato esperado.")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise AuditError("El .pth no es un mapeo directo de nombres a tensores; no se desempacó automáticamente.")

    expected_state = model.state_dict()
    missing_keys = sorted(set(expected_state) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(expected_state))
    shape_mismatches = {
        key: {"checkpoint": list(state_dict[key].shape), "model": list(expected_state[key].shape)}
        for key in set(expected_state) & set(state_dict)
        if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise AuditError(
            "El state_dict no coincide con la arquitectura original: "
            + json.dumps({"missing": missing_keys, "unexpected": unexpected_keys, "shape_mismatches": shape_mismatches})
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise AuditError(f"La carga strict=True falló: {exc}") from exc
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    embedding_count = (
        model.token_embedding_table.weight.numel()
        + model.position_embedding_table.weight.numel()
    )
    layers = len(model.transformer_decoder.layers)
    expected_embed_shape = [vocab_size, int(model_cfg["n_embd"])]
    expected_pos_shape = [block_size, int(model_cfg["n_embd"])]
    if list(model.token_embedding_table.weight.shape) != expected_embed_shape:
        raise AuditError("Forma inesperada de token_embedding_table después de cargar el checkpoint.")
    if list(model.position_embedding_table.weight.shape) != expected_pos_shape:
        raise AuditError("Forma inesperada de position_embedding_table después de cargar el checkpoint.")
    if list(model.lm_head.weight.shape) != [vocab_size, int(model_cfg["n_embd"])]:
        raise AuditError("Forma inesperada de lm_head después de cargar el checkpoint.")
    if layers != int(model_cfg["n_layer"]):
        raise AuditError(f"Número inesperado de bloques Transformer: {layers}.")
    if hasattr(model, "property_projection"):
        raise AuditError("La arquitectura cargada contiene property_projection; no parece incondicional.")
    report["checks"]["architecture_and_weights"] = {
        "strict_load": True,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "conditioning_mode": "unconditional",
        "num_properties": 0,
        "n_embd": int(model_cfg["n_embd"]),
        "n_head": int(model_cfg["n_head"]),
        "n_layer": layers,
        "dropout": float(model_cfg["dropout"]),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "parameter_count": parameter_count,
        "embedding_parameter_count": embedding_count,
        "token_embedding_shape": list(model.token_embedding_table.weight.shape),
        "position_embedding_shape": list(model.position_embedding_table.weight.shape),
        "lm_head_weight_shape": list(model.lm_head.weight.shape),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "torch_version": torch.__version__,
    }

    hidden_shapes: list[list[int]] = []
    hook = model.ln_f.register_forward_hook(
        lambda _module, _inputs, output: hidden_shapes.append(list(output.shape))
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
    properties = torch.empty((1, 0), dtype=torch.float32, device="cpu")
    try:
        with torch.inference_mode():
            logits_a, loss_a = model(input_ids, properties)
            logits_b, loss_b = model(input_ids, properties)
    finally:
        hook.remove()
    if logits_a.shape != (1, len(token_ids), vocab_size):
        raise AuditError(f"Forma inesperada de logits: {list(logits_a.shape)}.")
    if not torch.isfinite(logits_a).all() or not torch.isfinite(logits_b).all():
        raise AuditError("El smoke test produjo NaN o Inf en logits.")
    reproducible = torch.allclose(logits_a, logits_b, rtol=1e-6, atol=1e-7)
    if not reproducible:
        raise AuditError("La inferencia repetida en CPU no fue reproducible dentro de tolerancia.")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AuditError("Se detectaron gradientes: la validación no debe actualizar parámetros.")
    checkpoint_hash_after = sha256_file(args.checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise AuditError("El archivo del checkpoint cambió durante la auditoría.")
    report["checks"]["cpu_inference"] = {
        "model_eval": not model.training,
        "inference_mode": True,
        "logits_shape": list(logits_a.shape),
        "logits_finite": True,
        "loss_is_none": loss_a is None and loss_b is None,
        "normalized_hidden_state_shape_before_lm_head": hidden_shapes[-1] if hidden_shapes else None,
        "repeated_forward_allclose": reproducible,
        "reproducibility_rtol": 1e-6,
        "reproducibility_atol": 1e-7,
        "parameter_updates_performed": False,
        "checkpoint_sha256_unchanged": checkpoint_hash_after == checkpoint_hash_before,
    }
    report["status"] = "PASS"
    LOGGER.info("audit_pass", extra={"parameter_count": parameter_count, "vocab_size": vocab_size})
    return report


def main() -> int:
    """Run the audit, save its report, and return an informative exit status.

    Returns:
        0 for PASS, 1 for PARTIAL, or 2 for BLOCKED.
    """
    args = parse_args()
    configure_logging(args.verbose)
    try:
        report = run_audit(args)
    except AuditError as exc:
        report = {
            "experiment": "matgpt_e0_load_validate",
            "status": "BLOCKED",
            "training_performed": False,
            "device": "cpu",
            "errors": [str(exc)],
        }
        LOGGER.error("audit_blocked", extra={"error": str(exc)})
    except Exception as exc:  # preserve an explicit failure report for unexpected runtime issues
        report = {
            "experiment": "matgpt_e0_load_validate",
            "status": "BLOCKED",
            "training_performed": False,
            "device": "cpu",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        LOGGER.exception("unexpected_audit_error")

    try:
        write_report(args.report, report)
    except OSError as exc:
        LOGGER.error("report_write_failed", extra={"path": str(args.report), "error": str(exc)})
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return {"PASS": 0, "PARTIAL": 1, "BLOCKED": 2}.get(report.get("status", "BLOCKED"), 2)


if __name__ == "__main__":
    raise SystemExit(main())
