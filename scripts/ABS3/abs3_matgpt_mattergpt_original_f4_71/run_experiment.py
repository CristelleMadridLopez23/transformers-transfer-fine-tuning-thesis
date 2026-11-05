"""Entrena F0 congelado emparejado y F4 directo sobre ABS3.

Cada fold/semilla comparte un warmup MLP8. MatGPT/MatterGPT se mantienen en
CPU, con sus cuatro bloques superiores entrenables solo durante F4. Las
predicciones outer se guardan únicamente después de seleccionar por inner-val.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
EXP = ROOT / "experiments/abs3_matgpt_mattergpt_original_f4_71"
OUTPUT_ROOT = EXP
DATASET = ROOT / "data/abs3/abs3_development_f4_geometry/dataset.csv"
FOLDS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
EMBEDDINGS = EXP / "embeddings"
GEOMETRY = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg",
            "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
ELEMENTS = ("Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr")
RIDGE_HASHES = {
    0: "99d5114f567450f46322cd39e021405bc4ba9fd396fadeb07ecb62f7a204f3d9",
    1: "48bc05a164936d4c2ef104baa60086a95388121eb95624003aad07388ddcfde1",
    2: "664fc440ce0a699fa5ac5e62f9167e3359682ada5bb63838323cb7a5a1efcf0a",
    3: "e26bae8b47c98b86731c283bcbd69732f6395dea7e66491bad82f08e53005e49",
    4: "680e1266483f838fbc962bed8b0d5458c3e7f1e059cf30a83dafdbf76c56408d",
}
RIDGE_GIT_PATH = "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional/models/ridge_composicional_fold_{}.joblib"
SEEDS = (42, 123, 2026)
FAMILY_ORDER = ("dist_perovskite", "hexagonal", "needle")
EXPECTED_BASE = {"MatGPT": "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414",
                 "MatterGPT": "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"}
LOGGER = logging.getLogger("abs3_original_f4")
STARTED = time.monotonic()
PROCESS_STARTED = time.process_time()


class MLP8(nn.Module):
    """Cabeza residual histórica 775→8→1 con GELU y dropout 0.1."""

    def __init__(self) -> None:
        """Crea la MLP8 sin normalización adicional."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(775, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, values: Tensor) -> Tensor:
        """Devuelve residuos sin centrar, uno por estructura."""
        return self.net(values).squeeze(-1)


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo.

    Args:
        path: Archivo a resumir.
    Returns:
        Digest hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escribe JSON atómicamente.

    Args:
        path: Archivo destino.
        payload: Contenido serializable.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rss_peak_bytes() -> int:
    """Devuelve memoria residente máxima del proceso en bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def sha_bytes(payload: bytes) -> str:
    """Calcula el hash de bytes ya recuperados de Git."""
    return hashlib.sha256(payload).hexdigest()


def import_extractor() -> Any:
    """Importa los cargadores/serializadores compartidos de extractores."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.ABS3.abs3_matgpt_mattergpt_original_f4_71 import extract_embeddings
    return extract_embeddings


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los 210 ejemplos de desarrollo y cinco folds históricos.

    Returns:
        Dataset y asignaciones completas solo para `train`/`val` históricos.
    """
    data = pd.read_csv(DATASET, low_memory=False)
    folds = pd.read_csv(FOLDS, dtype={"structure_id": str})
    folds = folds[folds.historical_split.isin(("train", "val"))].copy()
    if len(data) != 210 or data.structure_id.nunique() != 210 or data.composition_id.nunique() != 70:
        raise ValueError("El dataset ABS3 no tiene 210 IDs únicos / 70 fórmulas")
    if len(folds) != 1050 or set(folds.structure_id.astype(str)) != set(data.structure_id.astype(str)):
        raise ValueError("Los folds no coinciden con los 210 IDs de desarrollo")
    if data.groupby("composition_id").size().ne(3).any():
        raise ValueError("Hay composiciones incompletas")
    if not data.SLICES.eq(data.slices_original).all():
        raise ValueError("La representación SLICES no se preservó byte a byte")
    if data[GEOMETRY].isna().any().any() or not np.isfinite(data[GEOMETRY].to_numpy(float)).all():
        raise ValueError("Descriptores geométricos no finitos")
    if (data.geometry_source != "gga-static_endpoint_cleaned").any():
        raise ValueError("Procedencia geométrica distinta de la histórica")
    for fold in range(5):
        role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
        counts = (int(role.inner_role.eq("inner_train").sum()),
                  int(role.inner_role.eq("inner_validation").sum()),
                  int(role.outer_role.eq("outer_validation").sum()))
        if counts != (45, 11, 14):
            raise ValueError(f"Fold {fold}: conteos no son 45/11/14: {counts}")
        if role.composition_group.duplicated().any():
            raise ValueError(f"Fórmula duplicada en roles de fold {fold}")
    return data.sort_values(["composition_id", "family"], kind="stable").reset_index(drop=True), folds


def load_embedding_cache(model_name: str, ids: np.ndarray) -> np.ndarray:
    """Carga caché fresca y exige alineación explícita por structure_id."""
    path = EMBEDDINGS / f"{model_name.lower()}_original_final_end.npz"
    with np.load(path, allow_pickle=False) as archive:
        stored_ids = archive["structure_id"].astype(str)
        matrix = archive["embeddings"].astype(np.float32)
    if matrix.shape != (210, 768) or len(set(stored_ids)) != 210:
        raise ValueError(f"Caché {model_name} con forma/IDs incorrectos")
    positions = {sid: index for index, sid in enumerate(stored_ids)}
    if set(positions) != set(ids.astype(str)):
        raise ValueError(f"Caché {model_name} IDs no coinciden")
    reordered = matrix[[positions[sid] for sid in ids.astype(str)]]
    if not np.isfinite(reordered).all():
        raise ValueError(f"Caché {model_name} tiene NaN/Inf")
    return reordered


def load_ridge_models(data: pd.DataFrame) -> tuple[dict[int, Any], dict[int, dict[str, float]], dict[int, str]]:
    """Carga M2 serializados desde Git sin ajustar parámetros y los verifica.

    Args:
        data: Dataset de 210 filas.
    Returns:
        Modelos fold, medias por fórmula y hashes SHA-256 de artefactos.
    """
    models: dict[int, Any] = {}
    means: dict[int, dict[str, float]] = {}
    hashes: dict[int, str] = {}
    reference = pd.read_csv(EXP.parent / "matgpt_f0_stoichiometric_ridge_m2_comparison/predicciones_oof_por_estructura.csv")
    reference = reference[reference.model.eq("RETRAINED_M2_C0") & reference.seed.eq(42)]
    element_index = {element: index for index, element in enumerate(ELEMENTS)}
    for fold in range(5):
        path = RIDGE_GIT_PATH.format(fold)
        result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        payload = result.stdout
        digest = sha_bytes(payload)
        if digest != RIDGE_HASHES[fold]:
            raise ValueError(f"Ridge M2 fold {fold} SHA {digest} distinto al manifiesto")
        model = joblib.load(io.BytesIO(payload))
        if not hasattr(model, "predict") or int(getattr(model, "n_features_in_", 17)) != 17:
            raise ValueError(f"Ridge M2 fold {fold} no recibe las 17 fracciones")
        models[fold] = model
        hashes[fold] = digest
        means[fold] = {}
        for formula in sorted(data.composition_id.unique()):
            components = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
            rebuilt = "".join(element + (number if number not in ("", "1") else "") for element, number in components)
            if rebuilt != formula or not components:
                raise ValueError(f"Fórmula no se pudo parsear exactamente: {formula}")
            counts = np.zeros(17, dtype=np.float64)
            for element, amount in components:
                if element not in element_index:
                    raise ValueError(f"Elemento fuera del Ridge M2: {element}")
                counts[element_index[element]] += float(amount or 1.0)
            fraction = counts / counts.sum()
            means[fold][formula] = float(np.asarray(model.predict(fraction.reshape(1, -1))).reshape(-1)[0])
        ref = reference[reference.fold.eq(fold)].drop_duplicates("composition_id").set_index("composition_id")
        matched = 0
        for formula, value in means[fold].items():
            # El OOF conserva solo las 14 fórmulas outer de cada fold.
            if formula not in ref.index:
                continue
            saved = float(ref.loc[formula, "ridge_mean_m2_ev_atom"])
            if abs(value - saved) > 1e-8:
                raise ValueError(f"Ridge M2 calculado difiere del histórico {formula}/fold{fold}: {value} vs {saved}")
            matched += 1
        if matched != 14:
            raise ValueError(f"Ridge M2: se verificaron {matched} y se esperaban 14 salidas outer en fold {fold}")
    return models, means, hashes


def build_triplets(data: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    """Indexa cada fórmula en el orden fijo de los tres polimorfos."""
    groups: dict[str, list[int]] = {}
    targets: dict[str, np.ndarray] = {}
    for formula, part in data.groupby("composition_id", sort=True):
        indexed = part.set_index("family")
        if set(indexed.index) != set(FAMILY_ORDER):
            raise ValueError(f"Familias no canónicas/incompletas: {formula}")
        rows = [int(indexed.loc[name, "_row_index"]) for name in FAMILY_ORDER]
        groups[str(formula)] = rows
        targets[str(formula)] = data.loc[rows, "energy_dft_ev_atom"].to_numpy(np.float32)
    return groups, targets


def load_model_base(model_name: str, extractor: Any) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Carga un Transformer original estricto, preparado para F4."""
    if model_name == "MatGPT":
        source = extractor.MATGPT_SOURCE
        sys.path.insert(0, str(source))
        model, tokenizer, context, audit = extractor.load_matgpt()
        return model, {"stoi": tokenizer, "context": context}, audit
    matter_dir = extractor.MATTER_DIR
    sys.path.insert(0, str(matter_dir))
    model, tokenizer, context, audit = extractor.load_mattergpt()
    return model, {"stoi": tokenizer, "context": context}, audit


def load_tokenizer(model_name: str, extractor: Any) -> tuple[dict[str, int], int]:
    """Carga solo el vocabulario y contexto, sin instanciar el Transformer."""
    if model_name == "MatGPT":
        info = json.loads(extractor.MATGPT_INFO.read_text(encoding="utf-8"))
        return ({str(key): int(value) for key, value in info["stoi"].items()}, int(info["block_size"]))
    config = json.loads((extractor.MATTER_DIR / "config.json").read_text(encoding="utf-8"))
    vocab = sorted(set((extractor.MATTER_DIR / "Voc_prior").read_text(encoding="utf-8").splitlines()) | {"<", ">"})
    if len(vocab) != int(config["vocab_size"]):
        raise ValueError("Voc_prior cambió respecto a config.json")
    return ({token: index for index, token in enumerate(vocab)}, int(config["block_size"]))


def encode_sequences(data: pd.DataFrame, model_name: str, tokenizer: dict[str, int], context: int) -> dict[str, list[int]]:
    """Codifica secuencias preservadas con marcadores propios de cada backbone."""
    encoded: dict[str, list[int]] = {}
    for row in data.itertuples():
        values = str(row.SLICES).split()
        if model_name == "MatGPT":
            values = ["<start>", *values, "<end>"]
        else:
            values = [">", *values, "<"]
        unknown = sorted(set(values) - set(tokenizer))
        if unknown:
            raise ValueError(f"OOV {model_name}, {row.structure_id}: {unknown[:6]}")
        if len(values) > context:
            raise ValueError(f"Truncamiento {model_name}, {row.structure_id}: {len(values)}>{context}")
        encoded[str(row.structure_id)] = [tokenizer[value] for value in values]
    if len(encoded) != 210:
        raise ValueError("No se tokenizaron las 210 estructuras")
    return encoded


def live_embeddings(model: torch.nn.Module, model_name: str, ids: list[str], encoded: dict[str, list[int]],
                    tokenizer: dict[str, int], device: torch.device) -> Tensor:
    """Obtiene FINAL_END en vivo; conserva autograd para el entrenamiento F4."""
    sequences = [encoded[sid] for sid in ids]
    width = max(map(len, sequences))
    pad_id = tokenizer["<pad>"] if model_name == "MatGPT" else 0
    tokens = torch.full((len(ids), width), pad_id, dtype=torch.long, device=device)
    for index, seq in enumerate(sequences):
        tokens[index, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    positions = torch.tensor([len(seq) - 1 for seq in sequences], dtype=torch.long, device=device)
    if model_name == "MatGPT":
        length, batch = tokens.shape[1], tokens.shape[0]
        hidden = model.token_embedding_table(tokens) + model.position_embedding_table(torch.arange(length, device=device))
        hidden = hidden.permute(1, 0, 2)
        mask = model.generate_square_subsequent_mask(length).to(device)
        memory = torch.zeros((length, batch, model.n_embd), device=device)
        hidden = model.transformer_decoder(hidden, memory, tgt_mask=mask).permute(1, 0, 2)
        hidden = model.ln_f(hidden)
        return hidden[torch.arange(batch, device=device), positions]
    property_vector = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device).view(1, 1, 2).expand(len(ids), 1, 2)
    capture: list[Tensor] = []
    hook = model.ln_f.register_forward_hook(lambda _module, _inputs, output: capture.append(output))
    try:
        model(tokens, prop=property_vector)
    finally:
        hook.remove()
    if len(capture) != 1:
        raise RuntimeError("MatterGPT no devolvió una salida ln_f")
    # La condición de propiedades ocupa la posición cero antes de SLICES.
    return capture[0][torch.arange(len(ids), device=device), positions + 1]


def configure_backbone(model: torch.nn.Module, model_name: str, trainable: bool) -> list[nn.Parameter]:
    """Congela todo salvo bloques 8–11 si la condición es F4."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stack = model.transformer_decoder.layers if model_name == "MatGPT" else model.blocks
    if len(stack) != 12:
        raise ValueError(f"{model_name} no tiene 12 bloques")
    active: list[nn.Parameter] = []
    if trainable:
        for index in range(8, 12):
            for parameter in stack[index].parameters():
                parameter.requires_grad_(True)
                active.append(parameter)
    return active


def model_eval_trainable_blocks(model: torch.nn.Module, model_name: str) -> None:
    """Activa dropout solo en los bloques entrenables y mantiene resto en eval."""
    model.eval()
    stack = model.transformer_decoder.layers if model_name == "MatGPT" else model.blocks
    for index in range(8, 12):
        stack[index].train()


def get_features(ids: list[str], data: pd.DataFrame, row_by_id: dict[str, int],
                 scaled_geom: np.ndarray, cache: np.ndarray, model: torch.nn.Module | None,
                 model_name: str, encoded: dict[str, list[int]] | None,
                 tokenizer: dict[str, int] | None, device: torch.device) -> Tensor:
    """Concatena embedding contextualizado con g escalada; F4 usa el grafo vivo."""
    indices = [row_by_id[sid] for sid in ids]
    if model is None:
        hidden = torch.as_tensor(cache[indices], dtype=torch.float32, device=device)
    else:
        assert encoded is not None and tokenizer is not None
        hidden = live_embeddings(model, model_name, ids, encoded, tokenizer, device)
    geometry = torch.as_tensor(scaled_geom[indices], dtype=torch.float32, device=device)
    return torch.cat((hidden, geometry), dim=-1)


def targets_tensor(formulas: list[str], groups: dict[str, list[int]], targets: dict[str, np.ndarray],
                   device: torch.device) -> Tensor:
    """Construye targets DFT centrados por triplete."""
    rows = np.stack([targets[formula] - targets[formula].mean() for formula in formulas]).astype(np.float32)
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def full_loss(head: MLP8, formulas: list[str], groups: dict[str, list[int]], targets: dict[str, np.ndarray],
              data: pd.DataFrame, row_by_id: dict[str, int], scaled_geom: np.ndarray, cache: np.ndarray,
              model: torch.nn.Module | None, model_name: str, encoded: dict[str, list[int]] | None,
              tokenizer: dict[str, int] | None, scale: float, device: torch.device,
              ridge_mean: dict[str, float] | None = None) -> float | tuple[float, float]:
    """Evalúa SmoothL1 y, opcionalmente, MAE de energía absoluta por átomo."""
    head.eval()
    if model is not None:
        model.eval()
    losses = []
    absolute_errors: list[np.ndarray] = []
    with torch.no_grad():
        for formula in formulas:
            ids = [str(data.iloc[i].structure_id) for i in groups[formula]]
            x = get_features(ids, data, row_by_id, scaled_geom, cache, model, model_name, encoded, tokenizer, device)
            pred = head(x).view(1, 3)
            pred = pred - pred.mean(dim=1, keepdim=True)
            y = targets_tensor([formula], groups, targets, device)
            losses.append(F.smooth_l1_loss(pred / scale, y / scale, beta=1.0))
            if ridge_mean is not None:
                energy_prediction = pred.view(-1).cpu().numpy() + float(ridge_mean[formula])
                absolute_errors.extend(np.abs(energy_prediction - targets[formula]).tolist())
    mean_loss = float(torch.stack(losses).mean().cpu())
    if ridge_mean is not None:
        return mean_loss, float(np.mean(absolute_errors))
    return mean_loss


def append_history(row: dict[str, Any]) -> None:
    """Agrega una fila al historial canónico de épocas."""
    path = OUTPUT_ROOT / "train_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_energy_mae_history(row: dict[str, Any]) -> None:
    """Agrega MAE energético por época a la salida diagnóstica separada."""
    path = OUTPUT_ROOT / "training_mae_per_epoch.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_torch(path: Path, state: dict[str, Any]) -> None:
    """Escribe checkpoint temporal y reemplaza atómicamente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run_warmup(model_name: str, fold: int, seed: int, data: pd.DataFrame,
               groups: dict[str, list[int]], targets: dict[str, np.ndarray],
               row_by_id: dict[str, int], scaled_geom: np.ndarray, cache: np.ndarray,
               train_formulas: list[str], val_formulas: list[str], scale: float,
               device: torch.device) -> tuple[MLP8, dict[str, Any]]:
    """Ejecuta y guarda un único warmup compartido por F0 y F4."""
    path = OUTPUT_ROOT / "checkpoints" / model_name / f"fold_{fold}_seed_{seed}_warmup.pt"
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get("dataset_sha256") != sha256(DATASET):
            raise ValueError(f"Warmup incompatible con dataset actual: {path}")
        head = MLP8().to(device)
        head.load_state_dict(saved["head"], strict=True)
        return head, saved["summary"]
    torch.manual_seed(seed)
    head = MLP8().to(device)
    head_initial = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in head.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 1009 * fold)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, 21):
        head.train()
        order = np.asarray(train_formulas)[rng.permutation(len(train_formulas))]
        weighted_loss = 0.0
        n_seen = 0
        for offset in range(0, len(order), 8):
            batch_formulas = [str(value) for value in order[offset:offset + 8]]
            ids = [str(data.iloc[index].structure_id) for formula in batch_formulas for index in groups[formula]]
            x = get_features(ids, data, row_by_id, scaled_geom, cache, None, model_name, None, None, device)
            raw = head(x).view(len(batch_formulas), 3)
            prediction = raw - raw.mean(dim=1, keepdim=True)
            y = targets_tensor(batch_formulas, groups, targets, device)
            loss = F.smooth_l1_loss(prediction / scale, y / scale, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Warmup MLP8 loss no finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
                raise FloatingPointError("Warmup MLP8 recibió gradientes ausentes/no finitos")
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(batch_formulas)
            n_seen += len(batch_formulas)
        train_loss = weighted_loss / n_seen
        val_loss = full_loss(head, val_formulas, groups, targets, data, row_by_id,
                             scaled_geom, cache, None, model_name, None, None, scale, device)
        row = {"model": model_name, "condition": "WARMUP", "fold": fold, "seed": seed,
               "epoch": epoch, "train_minibatch_smoothl1": train_loss,
               "inner_validation_full_smoothl1": val_loss, "best_epoch": best_epoch or epoch}
        append_history(row)
        LOGGER.info("%s", row)
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is None:
        raise RuntimeError("Warmup no generó checkpoint validado")
    head.load_state_dict(best_state, strict=True)
    summary = {"best_epoch": best_epoch, "epochs": epoch, "best_inner_validation_smoothl1": best_loss,
               "initial_head_sha256": head_initial, "residual_scale_ev_atom": scale}
    save_torch(path, {"head": best_state, "summary": summary, "dataset_sha256": sha256(DATASET),
                      "model": model_name, "fold": fold, "seed": seed, "selected_by": "inner_validation"})
    return head, summary


def audit_head_and_gradient(model_name: str, data: pd.DataFrame, folds: pd.DataFrame,
                            cache: np.ndarray, extractor: Any, device: torch.device) -> dict[str, Any]:
    """Audita GELU, minilote memorizable y backward/update F4 end-to-end."""
    role = folds[folds.outer_fold.eq(0)].drop_duplicates("composition_group").set_index("composition_group")
    train = sorted(role.index[role.inner_role.eq("inner_train")].astype(str))
    data = data.copy()
    data["_row_index"] = np.arange(len(data))
    groups, targets = build_triplets(data)
    row_by_id = {str(sid): index for index, sid in enumerate(data.structure_id.astype(str))}
    train_rows = [index for formula in train for index in groups[formula]]
    scaler = StandardScaler().fit(data.loc[train_rows, GEOMETRY].to_numpy(float))
    scaled_geom = scaler.transform(data[GEOMETRY].to_numpy(float)).astype(np.float32)
    tokenizer, context = load_tokenizer(model_name, extractor)
    encoded = encode_sequences(data, model_name, tokenizer, context)
    audit: dict[str, Any] = {"model": model_name, "fold": 0, "seed": 42,
        "inner_train_formulas": len(train), "geometry_fit_rows": len(train_rows)}
    # Random initialization: ensure the hidden layer is not collapsed before warmup.
    torch.manual_seed(42)
    head = MLP8().to(device)
    sample_ids = [str(data.iloc[index].structure_id) for formula in train[:8] for index in groups[formula]]
    sample_idx = [row_by_id[sid] for sid in sample_ids]
    x_cached = torch.cat((torch.as_tensor(cache[sample_idx], dtype=torch.float32, device=device),
                          torch.as_tensor(scaled_geom[sample_idx], dtype=torch.float32, device=device)), dim=-1)
    head.eval()
    with torch.no_grad():
        preactivation = head.net[0](x_cached)
        activated = head.net[1](preactivation)
        initial_output = head(x_cached).view(8, 3)
    audit.update({"preactivation_std": float(preactivation.std().cpu()),
        "gelu_near_zero_fraction": float((activated.abs() < 1e-5).float().mean().cpu()),
        "initial_prediction_std": float(initial_output.std().cpu()),
        "all_units_effectively_saturated": bool((activated.abs() < 1e-5).float().mean() > 0.995)})
    if audit["initial_prediction_std"] <= 1e-10 or audit["all_units_effectively_saturated"]:
        raise FloatingPointError(f"MLP8 inicial colapsada/saturada para {model_name}: {audit}")
    # Isolated fixed-batch memorization, using only a temporary head.
    memorizer = MLP8().to(device)
    memorizer.load_state_dict(head.state_dict(), strict=True)
    y_mini = torch.as_tensor(np.stack([targets[formula] - targets[formula].mean() for formula in train[:8]]),
                             dtype=torch.float32, device=device)
    optimizer_mini = torch.optim.AdamW(memorizer.parameters(), lr=1e-3, weight_decay=0.0)
    memorizer.eval()
    with torch.no_grad():
        pred0 = memorizer(x_cached).view(8, 3)
        loss0 = float(F.smooth_l1_loss((pred0 - pred0.mean(dim=1, keepdim=True)), y_mini))
    for _ in range(50):
        memorizer.train()
        optimizer_mini.zero_grad(set_to_none=True)
        p = memorizer(x_cached).view(8, 3)
        p = p - p.mean(dim=1, keepdim=True)
        loss = F.smooth_l1_loss(p, y_mini)
        loss.backward()
        optimizer_mini.step()
    memorizer.eval()
    with torch.no_grad():
        pred1 = memorizer(x_cached).view(8, 3)
        loss1 = float(F.smooth_l1_loss(pred1 - pred1.mean(dim=1, keepdim=True), y_mini))
    audit.update({"fixed_minibatch_formulas": 8, "memorization_steps": 50,
                  "memorization_loss_start": loss0, "memorization_loss_end": loss1,
                  "memorization_pass": bool(loss1 < loss0)})
    if not audit["memorization_pass"]:
        raise FloatingPointError(f"MLP8 no memoriza el minibatch aislado {model_name}: {audit}")
    # End-to-end F4 backward/update in an isolated fresh backbone.
    model, loaded, model_audit = load_model_base(model_name, extractor)
    tokenizer = loaded["stoi"]
    active = configure_backbone(model, model_name, True)
    if not active:
        raise RuntimeError(f"F4 no configuró parámetros entrenables para {model_name}")
    model.to(device)
    model_eval_trainable_blocks(model, model_name)
    head = MLP8().to(device)
    head.load_state_dict(memorizer.state_dict(), strict=True)
    formula = train[0]
    ids = [str(data.iloc[index].structure_id) for index in groups[formula]]
    before_backbone = [parameter.detach().clone() for parameter in active]
    frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
    before_frozen = [parameter.detach().clone() for parameter in frozen[:8]]
    features = torch.cat((live_embeddings(model, model_name, ids, encoded, tokenizer, device),
                          torch.as_tensor(scaled_geom[[row_by_id[sid] for sid in ids]], dtype=torch.float32, device=device)), dim=-1)
    output = head(features)
    target = torch.as_tensor(targets[formula] - targets[formula].mean(), dtype=torch.float32, device=device)
    scale = float(np.std(np.concatenate([targets[item] - targets[item].mean() for item in train]), ddof=0))
    loss = F.smooth_l1_loss((output - output.mean()) / scale, target / scale, beta=1.0)
    loss.backward()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in active + list(head.parameters())):
        raise FloatingPointError(f"F4 gradiente ausente/no finito {model_name}")
    if any(parameter.grad is not None for parameter in frozen):
        raise ValueError(f"Parámetros congelados reciben gradiente {model_name}")
    optimizer = torch.optim.AdamW([{"params": active, "lr": 1e-6, "weight_decay": 1e-2},
                                   {"params": list(head.parameters()), "lr": 1e-4, "weight_decay": 1e-4}])
    head_before = [parameter.detach().clone() for parameter in head.parameters()]
    optimizer.step()
    updates = [float(torch.linalg.vector_norm(parameter.detach() - before).cpu())
               for parameter, before in zip(active, before_backbone)]
    head_updates = [float(torch.linalg.vector_norm(parameter.detach() - before).cpu())
                    for parameter, before in zip(head.parameters(), head_before)]
    if not any(value > 0 for value in updates) or not any(value > 0 for value in head_updates):
        raise FloatingPointError(f"F4 optimizer no actualizó backbone/head {model_name}")
    if any(not torch.equal(parameter.detach(), before) for parameter, before in zip(frozen[:8], before_frozen)):
        raise ValueError(f"Parámetro congelado cambió en prueba aislada {model_name}")
    audit.update({"checkpoint_sha256": model_audit["checkpoint_sha256"],
        "trainable_blocks": [8, 9, 10, 11], "trainable_parameter_tensors": len(active),
        "trainable_parameter_count": sum(parameter.numel() for parameter in active),
        "gradient_norm_top_blocks": float(torch.sqrt(torch.stack([
            parameter.grad.detach().float().pow(2).sum() for parameter in active]).sum()).cpu()),
        "gradient_norm_head": float(torch.sqrt(torch.stack([
            parameter.grad.detach().float().pow(2).sum() for parameter in head.parameters()]).sum()).cpu()),
        "backbone_update_norm_sum": float(sum(updates)), "head_update_norm_sum": float(sum(head_updates)),
        "frozen_parameters_changed": False, "status": "PASS"})
    del model, memorizer
    return audit


def extract_batch(model: torch.nn.Module, model_name: str, batch_ids: list[str],
                  encoded: dict[str, list[int]], tokenizer: dict[str, int], device: torch.device) -> Tensor:
    """Ejecuta el backbone con o sin grad según contexto de llamada."""
    return live_embeddings(model, model_name, batch_ids, encoded, tokenizer, device)


def train_branch(model_name: str, condition: str, fold: int, seed: int, data: pd.DataFrame,
                 groups: dict[str, list[int]], targets: dict[str, np.ndarray], row_by_id: dict[str, int],
                 scaled_geom: np.ndarray, cache: np.ndarray, train_formulas: list[str],
                 val_formulas: list[str], outer_formulas: list[str], scale: float, head_init: MLP8,
                 device: torch.device, extractor: Any, encoded: dict[str, list[int]],
                 tokenizer: dict[str, int], ridge_mean: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Entrena el control congelado o F4 hasta 71 épocas con selección interna."""
    branch_started = time.monotonic()
    run_name = f"{model_name}_{condition}_fold{fold}_seed{seed}"
    complete_path = OUTPUT_ROOT / "checkpoints" / model_name / f"fold_{fold}_seed_{seed}_{condition}_best.pt"
    is_f4 = condition == "F4"
    model: torch.nn.Module | None = None
    active: list[nn.Parameter] = []
    if is_f4:
        model, loaded, audit = load_model_base(model_name, extractor)
        tokenizer = loaded["stoi"]
        if audit["checkpoint_sha256"] != EXPECTED_BASE[model_name]:
            raise ValueError(f"Checkpoint base cambió durante corrida {run_name}")
        active = configure_backbone(model, model_name, True)
        if not active:
            raise ValueError(f"F4 sin parámetros entrenables {run_name}")
        model.to(device)
    head = MLP8().to(device)
    head.load_state_dict(head_init.state_dict(), strict=True)
    head_initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in head.parameters())).hexdigest()
    if is_f4:
        optimizer = torch.optim.AdamW([
            {"params": active, "lr": 1e-6, "weight_decay": 1e-2},
            {"params": list(head.parameters()), "lr": 1e-4, "weight_decay": 1e-4},])
    else:
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
    stack = None if model is None else (model.transformer_decoder.layers if model_name == "MatGPT" else model.blocks)
    rng = np.random.default_rng(seed + 1009 * fold)
    # The shared warmup is a valid candidate for both branches. In particular,
    # F4 must not replace it unless inner-validation improves after unfreezing.
    warmup_val, warmup_mae = full_loss(head, val_formulas, groups, targets, data, row_by_id, scaled_geom,
                           cache, model, model_name, encoded if model is not None else None,
                           tokenizer if model is not None else None, scale, device, ridge_mean)
    best_loss = warmup_val
    best_epoch = 0
    best_phase = "warmup"
    best_head: dict[str, Tensor] | None = {key: value.detach().cpu().clone()
                                          for key, value in head.state_dict().items()}
    best_blocks: dict[str, dict[str, Tensor]] | None = None
    if model is not None and stack is not None:
        best_blocks = {str(index): {key: value.detach().cpu().clone()
                                    for key, value in stack[index].state_dict().items()}
                       for index in range(8, 12)}
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 72):
        head.train()
        if model is not None:
            model_eval_trainable_blocks(model, model_name)
        order = np.asarray(train_formulas)[rng.permutation(len(train_formulas))]
        weighted_loss = 0.0
        weighted_energy_abs_error = 0.0
        n_seen = 0
        grad_norm_last = 0.0
        for offset in range(0, len(order), 8):
            batch_formulas = [str(value) for value in order[offset:offset + 8]]
            batch_ids = [str(data.iloc[index].structure_id) for formula in batch_formulas for index in groups[formula]]
            indices = [row_by_id[sid] for sid in batch_ids]
            if model is None:
                hidden = torch.as_tensor(cache[indices], dtype=torch.float32, device=device)
            else:
                hidden = extract_batch(model, model_name, batch_ids, encoded, tokenizer, device)
            geo = torch.as_tensor(scaled_geom[indices], dtype=torch.float32, device=device)
            features = torch.cat((hidden, geo), dim=-1)
            raw = head(features).view(len(batch_formulas), 3)
            prediction = raw - raw.mean(dim=1, keepdim=True)
            y = targets_tensor(batch_formulas, groups, targets, device)
            ridge_values = torch.as_tensor([ridge_mean[formula] for formula in batch_formulas],
                                           dtype=prediction.dtype, device=device).view(-1, 1)
            actual_energy = torch.as_tensor(np.stack([targets[formula] for formula in batch_formulas]),
                                            dtype=prediction.dtype, device=device)
            weighted_energy_abs_error += float(torch.abs(ridge_values + prediction - actual_energy).detach().sum())
            loss = F.smooth_l1_loss(prediction / scale, y / scale, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss no finita {run_name}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable_params = active + list(head.parameters())
            gradients = [parameter.grad for parameter in trainable_params]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"Gradientes ausentes/no finitos {run_name}")
            if is_f4 and not any(torch.count_nonzero(gradient).item() for gradient in gradients[:len(active)] if gradient is not None):
                raise FloatingPointError(f"F4 no recibió gradiente no nulo en blocks 8–11: {run_name}")
            grad_norm_last = float(torch.sqrt(torch.stack([gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]).sum()))
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(batch_formulas)
            n_seen += len(batch_formulas)
        train_loss = weighted_loss / n_seen
        train_energy_mae = weighted_energy_abs_error / (3 * n_seen)
        val_loss, val_energy_mae = full_loss(head, val_formulas, groups, targets, data, row_by_id, scaled_geom,
                             cache, model, model_name, encoded if model is not None else None,
                             tokenizer if model is not None else None, scale, device, ridge_mean)
        row = {"model": model_name, "condition": condition, "fold": fold, "seed": seed,
               "epoch": epoch, "train_minibatch_smoothl1": train_loss,
               "inner_validation_full_smoothl1": val_loss, "gradient_norm_last_batch": grad_norm_last,
               "best_epoch": best_epoch, "best_checkpoint_phase": best_phase,
               "warmup_inner_validation_smoothl1": warmup_val,
               "elapsed_seconds": time.monotonic() - branch_started}
        append_history(row)
        append_energy_mae_history({"model": model_name, "condition": condition, "fold": fold, "seed": seed,
            "epoch": epoch, "train_minibatch_energy_mae_ev_atom": train_energy_mae,
            "inner_validation_full_energy_mae_ev_atom": val_energy_mae,
            "warmup_inner_validation_energy_mae_ev_atom": warmup_mae,
            "train_minibatch_smoothl1_normalized": train_loss,
            "inner_validation_smoothl1_normalized": val_loss,
            "best_checkpoint_phase": best_phase, "best_epoch": best_epoch})
        history.append(row)
        LOGGER.info("%s", row)
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_phase = "F4" if is_f4 else "F0"
            best_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            if model is not None and stack is not None:
                best_blocks = {str(index): {key: value.detach().cpu().clone()
                                             for key, value in stack[index].state_dict().items()}
                               for index in range(8, 12)}
            stale = 0
        else:
            stale += 1
            if stale >= 12:
                break
    if best_head is None:
        raise RuntimeError(f"No se seleccionó checkpoint por inner-val: {run_name}")
    last_head = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    last_blocks = None
    if model is not None and stack is not None:
        last_blocks = {str(index): {key: value.detach().cpu().clone()
                                    for key, value in stack[index].state_dict().items()}
                       for index in range(8, 12)}
    if model is not None and best_blocks is not None and stack is not None:
        for index in range(8, 12):
            stack[index].load_state_dict(best_blocks[str(index)], strict=True)
    head.load_state_dict(best_head, strict=True)
    checkpoint_base = {"model": model_name, "condition": condition, "fold": fold, "seed": seed,
        "dataset_sha256": sha256(DATASET), "selected_by": "inner_validation",
        "best_epoch": best_epoch, "best_checkpoint_phase": best_phase,
        "best_validation_smoothl1": best_loss, "head": best_head,
        "base_checkpoint_sha256": EXPECTED_BASE[model_name], "ridge_sha256": RIDGE_HASHES[fold]}
    if best_blocks is not None:
        checkpoint_base["blocks_8_11"] = best_blocks
    save_torch(complete_path, {**checkpoint_base, "status": "COMPLETE",
                               "summary": {"best_epoch": best_epoch, "best_checkpoint_phase": best_phase,
                                           "warmup_validation_smoothl1": warmup_val,
                                           "best_validation_smoothl1": best_loss}})
    final_path = complete_path.with_name(complete_path.name.replace("_best.pt", "_final.pt"))
    save_torch(final_path, {"model": model_name, "condition": condition, "fold": fold, "seed": seed,
        "dataset_sha256": sha256(DATASET), "head": last_head, "blocks_8_11": last_blocks,
        "last_epoch": epoch, "base_checkpoint_sha256": EXPECTED_BASE[model_name]})
    # Rebuild OOF predictions from the selected validation checkpoint.
    output: list[dict[str, Any]] = []
    head.eval()
    if model is not None:
        model.eval()
    with torch.no_grad():
        for formula in outer_formulas:
            ids = [str(data.iloc[index].structure_id) for index in groups[formula]]
            features = get_features(ids, data, row_by_id, scaled_geom, cache, model, model_name,
                                    encoded if model is not None else None,
                                    tokenizer if model is not None else None, device)
            raw = head(features)
            centered = raw - raw.mean()
            if abs(float(centered.sum().cpu())) > 1e-6:
                raise ValueError(f"Residuos no centrados: {formula}")
            for local, row_idx in enumerate(groups[formula]):
                row_data = data.iloc[row_idx]
                mu = ridge_mean[formula]
                residual = float(centered[local].cpu())
                output.append({"model": model_name, "condition": condition, "fold": fold, "seed": seed,
                    "structure_id": row_data.structure_id, "composition_id": formula, "family": row_data.family,
                    "n_atoms": int(row_data.n_atoms), "energy_dft_ev_atom": float(row_data.energy_dft_ev_atom),
                    "ridge_mean_ev_atom": mu, "residual_pred_ev_atom": residual,
                    "energy_pred_ev_atom": mu + residual})
    summary = {"status": "COMPLETE", "model": model_name, "condition": condition, "fold": fold,
        "seed": seed, "warmup_shared_head_sha256": head_initial_hash, "best_epoch": best_epoch,
        "best_checkpoint_phase": best_phase, "warmup_inner_validation_smoothl1": warmup_val,
        "epochs_run": epoch, "best_inner_validation_smoothl1": best_loss,
        "trainable_backbone_parameters": len(active) and sum(value.numel() for value in active) or 0,
        "head_parameters": sum(value.numel() for value in head.parameters()),
        "optimizer_steps": len(history) * math.ceil(45 / 8), "elapsed_seconds": time.monotonic() - branch_started,
        "process_cpu_seconds": time.process_time() - PROCESS_STARTED,
        "peak_rss_bytes": rss_peak_bytes(), "device": str(device),
        "parameter_update_norm": None}
    write_json(OUTPUT_ROOT / "runs" / f"{run_name}.json", summary)
    return output, summary


def run_one(model_name: str, fold: int, seed: int, data: pd.DataFrame, folds: pd.DataFrame,
            cache: np.ndarray, ridge_means: dict[int, dict[str, float]], extractor: Any,
            device: torch.device) -> list[dict[str, Any]]:
    """Ajusta scaler, escala residual y condiciones F0/F4 emparejadas."""
    role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group").set_index("composition_group")
    train = sorted(role.index[role.inner_role.eq("inner_train")].astype(str))
    validation = sorted(role.index[role.inner_role.eq("inner_validation")].astype(str))
    outer = sorted(role.index[role.outer_role.eq("outer_validation")].astype(str))
    if (len(train), len(validation), len(outer)) != (45, 11, 14):
        raise ValueError(f"Roles incorrectos en fold {fold}")
    data = data.copy()
    data["_row_index"] = np.arange(len(data))
    groups, targets = build_triplets(data)
    row_by_id = {str(sid): index for index, sid in enumerate(data.structure_id.astype(str))}
    train_rows = [index for formula in train for index in groups[formula]]
    scaler = StandardScaler().fit(data.loc[train_rows, GEOMETRY].to_numpy(float))
    scaled_geom = scaler.transform(data[GEOMETRY].to_numpy(float)).astype(np.float32)
    residuals = np.concatenate([targets[formula] - targets[formula].mean() for formula in train])
    scale = float(np.std(residuals, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Escala residual no positiva fold {fold}")
    scaler_path = OUTPUT_ROOT / "scalers" / f"fold_{fold}_geometry_scaler.json"
    scaler_obj = {"fold": fold, "fit_compositions": train, "n_structures": len(train_rows),
        "columns": GEOMETRY, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "residual_target_scale_ev_atom": scale}
    write_json(scaler_path, scaler_obj)
    # Fresh embeddings stay frozen for head warmup and F0. F4 always forwards live.
    warm_head, warm_summary = run_warmup(model_name, fold, seed, data, groups, targets, row_by_id,
        scaled_geom, cache, train, validation, scale, device)
    rows: list[dict[str, Any]] = []
    tokenizer, context = load_tokenizer(model_name, extractor)
    encoded = encode_sequences(data, model_name, tokenizer, context)
    for condition in ("F0", "F4"):
        warm_copy = MLP8().to(device)
        warm_copy.load_state_dict(warm_head.state_dict(), strict=True)
        if condition == "F4":
            # The same warmed head state seeds both branches; training order is seed/fold-identical.
            pass
        else:
            branch_tokenizer, branch_encoded = {}, {}
        pred_rows, run_summary = train_branch(model_name, condition, fold, seed, data, groups, targets,
            row_by_id, scaled_geom, cache, train, validation, outer, scale, warm_copy, device,
            extractor, encoded if condition == "F4" else branch_encoded,
            tokenizer if condition == "F4" else branch_tokenizer, ridge_means[fold])
        for row in pred_rows:
            row["warmup_best_epoch"] = warm_summary["best_epoch"]
        rows.extend(pred_rows)
        del warm_copy
    return rows


def pairwise_and_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calcula métricas primarias OOF y pares con reglas históricas."""
    pairs: list[dict[str, Any]] = []
    per_comp: list[dict[str, Any]] = []
    for (model_name, condition, fold, seed, formula), group in predictions.groupby(
            ["model", "condition", "fold", "seed", "composition_id"], sort=True):
        group = group.set_index("family").loc[list(FAMILY_ORDER)].reset_index()
        errors_energy = group.energy_pred_ev_atom.to_numpy() - group.energy_dft_ev_atom.to_numpy()
        rel = group.energy_pred_ev_atom.to_numpy()
        dft = group.energy_dft_ev_atom.to_numpy()
        absolute = np.abs(dft[:, None] - dft[None, :]) * 1000
        dft_gap = float(np.sort(dft)[1] - np.sort(dft)[0]) * 1000
        true_gs = int(np.argmin(dft)); pred_gs = int(np.argmin(rel))
        for i in range(3):
            for j in range(i + 1, 3):
                delta_dft = (dft[j] - dft[i]) * 1000
                delta_pred = (rel[j] - rel[i]) * 1000
                pair_error = delta_pred - delta_dft
                eligible = abs(delta_dft) > 5
                resolved = abs(delta_pred) > 1
                correct = bool(resolved and np.sign(delta_pred) == np.sign(delta_dft))
                pairs.append({"model": model_name, "condition": condition, "fold": fold, "seed": seed,
                    "composition_id": formula, "family_i": group.family.iloc[i], "family_j": group.family.iloc[j],
                    "delta_dft_mev_atom": delta_dft, "delta_pred_mev_atom": delta_pred,
                    "absolute_error_mev_atom": abs(pair_error), "dft_eligible": eligible,
                    "pred_resolved": resolved, "pairwise_correct": correct,
                    "ground_state_dft_gap_mev_atom": dft_gap})
        pair_subset = [row for row in pairs if row["model"] == model_name and row["condition"] == condition
                       and row["fold"] == fold and row["seed"] == seed and row["composition_id"] == formula]
        pair_mae = float(np.mean([row["absolute_error_mev_atom"] for row in pair_subset]))
        dft_rel = (dft - dft.min()) * 1000
        pred_rel = (rel - rel.min()) * 1000
        per_comp.append({"model": model_name, "condition": condition, "fold": fold, "seed": seed,
            "composition_id": formula, "mae_energy_ev_atom": float(np.mean(np.abs(errors_energy))),
            "rmse_energy_ev_atom": float(np.sqrt(np.mean(errors_energy ** 2))),
            "bias_energy_ev_atom": float(np.mean(errors_energy)),
            "mae_residual_mev_atom": float(np.mean(np.abs((rel - group.ridge_mean_ev_atom.to_numpy()) -
                                                   (dft - group.ridge_mean_ev_atom.to_numpy())) * 1000)),
            "mae_delta_signed_est_a_mev_atom": pair_mae,
            "rmse_delta_signed_est_a_mev_atom": float(np.sqrt(np.mean([row["absolute_error_mev_atom"] ** 2 for row in pair_subset]))),
            "ground_state_eligible": dft_gap > 1.0,
            "ground_state_correct": bool(true_gs == pred_gs) if dft_gap > 1.0 else None,
            "ground_state_dft_gap_mev_atom": dft_gap,
            "spearman_by_composition": float(spearmanr(dft, rel).statistic),
            "kendall_by_composition": float(kendalltau(dft, rel).statistic)})
    pair_frame = pd.DataFrame(pairs)
    comp_frame = pd.DataFrame(per_comp)
    metrics: dict[str, Any] = {"primary_estimand": "A; three signed intracomposition pairs; average within composition, then seeds within composition, then 70-formula macro-average"}
    for (model_name, condition), g in comp_frame.groupby(["model", "condition"]):
        seed_agg = g.groupby("composition_id", as_index=False).agg(
            mae_delta=("mae_delta_signed_est_a_mev_atom", "mean"),
            mae_energy=("mae_energy_ev_atom", "mean"),
            mae_residual=("mae_residual_mev_atom", "mean"),
            gs_correct=("ground_state_correct", "mean"),
            gs_eligible=("ground_state_eligible", "max"),
            spearman=("spearman_by_composition", "mean"),
            kendall=("kendall_by_composition", "mean"))
        pg = pair_frame[(pair_frame.model == model_name) & (pair_frame.condition == condition)]
        eligible = pg[pg.dft_eligible]
        resolved = eligible[eligible.pred_resolved]
        both = predictions[(predictions.model == model_name) & (predictions.condition == condition)]
        delta_table = pg[["delta_dft_mev_atom", "delta_pred_mev_atom"]].to_numpy(float)
        if len(delta_table):
            slope, intercept = np.polyfit(delta_table[:, 0], delta_table[:, 1], 1)
            ss_res = float(np.square(delta_table[:, 1] - (slope * delta_table[:, 0] + intercept)).sum())
            ss_tot = float(np.square(delta_table[:, 1] - delta_table[:, 1].mean()).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        else:
            slope, intercept, r2 = None, None, None
        metrics[f"{model_name}_{condition}"] = {
            "n_formulas": int(seed_agg.composition_id.nunique()),
            "mae_delta_signed_est_a_mev_atom": float(seed_agg.mae_delta.mean()),
            "mae_energy_ev_atom": float(seed_agg.mae_energy.mean()),
            "mae_residual_mev_atom": float(seed_agg.mae_residual.mean()),
            "rmse_energy_global_ev_atom": float(np.sqrt(np.mean((both.energy_pred_ev_atom - both.energy_dft_ev_atom) ** 2))),
            "ground_state_accuracy": float(g[g.ground_state_eligible].groupby("composition_id").ground_state_correct.mean().mean()),
            "ground_state_cases": int(g[g.ground_state_eligible].groupby("composition_id").ground_state_correct.count().sum()),
            "pairwise_accuracy_eligible_including_unresolved": float(eligible.pairwise_correct.mean()),
            "pairwise_resolved_accuracy": float(resolved.pairwise_correct.mean()) if len(resolved) else None,
            "unresolved_fraction_eligible_pairs": float((~eligible.pred_resolved).mean()),
            "eligible_pairs": int(len(eligible)), "resolved_pairs": int(len(resolved)),
            "spearman_mean_by_formula": float(seed_agg.spearman.mean()),
            "kendall_mean_by_formula": float(seed_agg.kendall.mean()),
            "ols_delta_slope": None if slope is None else float(slope),
            "ols_delta_intercept_mev_atom": None if intercept is None else float(intercept),
            "ols_delta_r2": r2}
    return pair_frame, comp_frame, metrics


def bootstrap_contrasts(comp: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Bootstrap diferencias pareadas por fórmula, agregando semillas dentro."""
    agg = comp.groupby(["model", "condition", "composition_id"], as_index=False).mae_delta_signed_est_a_mev_atom.mean()
    contrasts = [("MatGPT_F4", "MatGPT_F0"), ("MatterGPT_F4", "MatterGPT_F0"),
                 ("MatterGPT_F4", "MatGPT_F4")]
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(20261103)
    formulas = sorted(agg.composition_id.unique())
    for left, right in contrasts:
        lmodel, lcond = left.rsplit("_", 1); rmodel, rcond = right.rsplit("_", 1)
        l = agg[(agg.model == lmodel) & (agg.condition == lcond)].set_index("composition_id").loc[formulas]
        r = agg[(agg.model == rmodel) & (agg.condition == rcond)].set_index("composition_id").loc[formulas]
        diff = l.mae_delta_signed_est_a_mev_atom.to_numpy() - r.mae_delta_signed_est_a_mev_atom.to_numpy()
        draws = np.mean(diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))], axis=1)
        results.append({"contrast": f"{left} − {right}", "estimate_mev_atom": float(diff.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap_replicates": n_boot, "compositions": len(formulas), "seed": 20261103})
    return pd.DataFrame(results)


def main() -> None:
    """Ejecuta preflight/piloto o evaluación OOF completa y consolidación."""
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full", "mae_curves"), default="pilot")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.threads < 1 or args.bootstrap < 100:
        parser.error("--threads debe ser positivo y --bootstrap >=100")
    if args.stage == "mae_curves":
        OUTPUT_ROOT = EXP / "diagnostics" / "mae_curve_replay"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    EXP.mkdir(parents=True, exist_ok=True)
    data, folds = load_data()
    ids = data.structure_id.to_numpy(str)
    extractor = import_extractor()
    ridge_models, ridge_means, ridge_hashes = load_ridge_models(data)
    ridge_audit = {str(fold): ridge_hashes[fold] for fold in range(5)}
    embedding_caches = {name: load_embedding_cache(name, ids) for name in ("MatGPT", "MatterGPT")}
    gradient_path = EXP / "gradients_audit.csv"
    if gradient_path.exists():
        previous_audit = pd.read_csv(gradient_path)
        if set(previous_audit.model) == {"MatGPT", "MatterGPT"} and previous_audit.status.eq("PASS").all() \
                and previous_audit.dataset_sha256.eq(sha256(DATASET)).all():
            gradient_rows = previous_audit.to_dict("records")
        else:
            raise ValueError("Existe una auditoría de gradiente fallida o de otro dataset; revisar antes de seguir")
    else:
        gradient_rows = []
        for model_name in ("MatGPT", "MatterGPT"):
            audit = audit_head_and_gradient(model_name, data, folds, embedding_caches[model_name], extractor,
                                            torch.device("cpu"))
            audit["dataset_sha256"] = sha256(DATASET)
            gradient_rows.append(audit)
            pd.DataFrame(gradient_rows).to_csv(gradient_path, index=False)
    preflight = {"status": "PASS", "dataset_sha256": sha256(DATASET), "structures": 210,
        "compositions": 70, "complete_triplets": True, "folds": 5, "seeds": list(SEEDS),
        "fold_sizes": {str(f): {"train": 45, "validation": 11, "outer": 14} for f in range(5)},
        "checkpoint_sha256": EXPECTED_BASE, "ridge_sha256": ridge_audit,
        "embedding_cache": {name: {"shape": list(matrix.shape), "finite": bool(np.isfinite(matrix).all())}
                            for name, matrix in embedding_caches.items()},
        "geometry_columns": GEOMETRY, "geometry_fit": "inner-train only (45 formulas / 135 structures per fold)",
        "property_vector_mattergpt": [1.0, 2.0], "historical_test_accessed": False,
        "slices_historically_non_equivalent_graph_count": 3,
        "slices_graph_flag_policy": "preserve archived input verbatim; token and forward tests pass",
        "gradient_audit": gradient_rows, "device": "cpu", "threads": args.threads}
    write_json(EXP / "preflight.json", preflight)
    write_json(OUTPUT_ROOT / "resources.json", {"status": "RUNNING", "stage": args.stage,
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "wall_seconds": 0.0, "process_cpu_seconds": time.process_time() - PROCESS_STARTED,
        "peak_rss_bytes": rss_peak_bytes(), "device": "cpu", "cpu_threads": args.threads,
        "mps": False, "cuda": False, "historical_test_accessed": False,
        "python_version": platform.python_version(), "torch_version": importlib.metadata.version("torch"),
        "numpy_version": np.__version__, "pandas_version": pd.__version__})
    predictions_path = EXP / "predicciones_oof_por_estructura.csv"
    if predictions_path.exists():
        existing = pd.read_csv(predictions_path)
        if existing.dataset_sha256.nunique() != 1 or existing.dataset_sha256.iloc[0] != sha256(DATASET):
            raise ValueError("Predicciones existentes no pertenecen al dataset actual; no se sobrescriben")
    else:
        existing = pd.DataFrame()
    requested = [(model_name, fold, seed) for model_name in ("MatGPT", "MatterGPT")
                 for fold in range(5) for seed in SEEDS]
    if args.stage == "pilot":
        requested = [(model_name, 0, 42) for model_name in ("MatGPT", "MatterGPT")]
    new_predictions: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for model_name, fold, seed in requested:
        key_rows = existing[(existing.model == model_name) & (existing.fold == fold) & (existing.seed == seed)] if not existing.empty else pd.DataFrame()
        if args.stage != "mae_curves" and not key_rows.empty and all(
                int((key_rows.condition == condition).sum()) == 42 for condition in ("F0", "F4")):
            LOGGER.info("skip complete pair %s fold%s seed%s", model_name, fold, seed)
            continue
        # Run each pair together to enforce shared warmup initialization.
        rows = run_one(model_name, fold, seed, data, folds, embedding_caches[model_name],
                       ridge_means, extractor, torch.device("cpu"))
        new_predictions.extend(rows)
        # Commit each complete fold/seed pair immediately so an interruption
        # cannot discard completed OOF predictions or force their retraining.
        pair_frame = pd.DataFrame(rows)
        pair_frame["dataset_sha256"] = sha256(DATASET)
        existing = pd.concat([existing, pair_frame], ignore_index=True) if not existing.empty else pair_frame
        existing = existing.drop_duplicates(
            ["model", "condition", "fold", "seed", "structure_id"], keep="last")
        existing.to_csv(predictions_path, index=False)
        write_json(OUTPUT_ROOT / "resources.json", {"status": "RUNNING", "stage": args.stage,
            "wall_seconds": time.monotonic() - STARTED,
            "process_cpu_seconds": time.process_time() - PROCESS_STARTED,
            "peak_rss_bytes": rss_peak_bytes(), "device": "cpu", "cpu_threads": args.threads,
            "mps": False, "cuda": False, "completed_fold_seed_pairs": int(existing[
                ["model", "fold", "seed"]].drop_duplicates().shape[0]),
            "expected_fold_seed_pairs": 30, "completed_prediction_rows": len(existing),
            "historical_test_accessed": False,
            "python_version": platform.python_version(),
            "torch_version": importlib.metadata.version("torch"),
            "numpy_version": np.__version__, "pandas_version": pd.__version__})
    if args.stage == "mae_curves":
        write_json(OUTPUT_ROOT / "resources.json", {"status": "COMPLETE", "stage": args.stage,
            "wall_seconds": time.monotonic() - STARTED,
            "process_cpu_seconds": time.process_time() - PROCESS_STARTED,
            "peak_rss_bytes": rss_peak_bytes(), "device": "cpu", "cpu_threads": args.threads,
            "mps": False, "cuda": False, "fold_seed_pairs": len(requested),
            "runs": len(requested) * 2, "epochs_recorded": len(pd.read_csv(OUTPUT_ROOT / "training_mae_per_epoch.csv")),
            "historical_test_accessed": False,
            "python_version": platform.python_version(), "torch_version": importlib.metadata.version("torch"),
            "numpy_version": np.__version__, "pandas_version": pd.__version__})
        write_json(OUTPUT_ROOT / "summary.json", {"status": "COMPLETE",
            "purpose": "Replay de train/validation para medir MAE energético por época; no cambia los resultados OOF canónicos.",
            "dataset_sha256": sha256(DATASET), "conditions": ["MatGPT_F0", "MatGPT_F4", "MatterGPT_F0", "MatterGPT_F4"],
            "historical_test_accessed": False})
        LOGGER.info("MAE-per-epoch replay completed; original OOF artifacts were not modified.")
        return
    if new_predictions:
        new_frame = pd.DataFrame(new_predictions)
        new_frame["dataset_sha256"] = sha256(DATASET)
        combined = pd.concat([existing, new_frame], ignore_index=True) if not existing.empty else new_frame
        combined = combined.drop_duplicates(["model", "condition", "fold", "seed", "structure_id"], keep="last")
        combined.to_csv(predictions_path, index=False)
    resources = {"status": "RUNNING" if args.stage == "pilot" else "COMPLETE",
        "stage": args.stage, "wall_seconds": time.monotonic() - STARTED,
        "process_cpu_seconds": time.process_time() - PROCESS_STARTED, "peak_rss_bytes": rss_peak_bytes(),
        "device": "cpu", "cpu_threads": args.threads, "mps": False, "cuda": False,
        "runs_in_this_invocation": len(requested), "historical_test_accessed": False,
        "python_version": platform.python_version(), "torch_version": importlib.metadata.version("torch"),
        "numpy_version": np.__version__, "pandas_version": pd.__version__}
    write_json(EXP / "resources.json", resources)
    if args.stage == "pilot":
        LOGGER.info("Pilot complete; metrics will be finalized only after all OOF folds are complete.")
        return
    predictions = pd.read_csv(predictions_path)
    expected = 2 * 2 * 5 * 3 * 42
    if len(predictions) != expected:
        raise ValueError(f"OOF incompleto: {len(predictions)} filas, esperado {expected}")
    pair_frame, comp_frame, metrics = pairwise_and_metrics(predictions)
    bootstrap = bootstrap_contrasts(comp_frame, args.bootstrap)
    pair_frame.to_csv(EXP / "predicciones_oof_por_par.csv", index=False)
    comp_frame.to_csv(EXP / "metricas_por_composicion.csv", index=False)
    fold_seed_metrics = comp_frame.groupby(["model", "condition", "fold", "seed"], as_index=False).agg(
        mae_delta_signed_est_a_mev_atom=("mae_delta_signed_est_a_mev_atom", "mean"),
        mae_energy_ev_atom=("mae_energy_ev_atom", "mean"),
        mae_residual_mev_atom=("mae_residual_mev_atom", "mean"),
        ground_state_accuracy=("ground_state_correct", "mean"),
        ground_state_eligible_cases=("ground_state_eligible", "sum"),
        n_compositions=("composition_id", "nunique"))
    fold_seed_metrics.to_csv(EXP / "metricas_por_fold_seed.csv", index=False)
    bootstrap.to_csv(EXP / "bootstrap_apareado.csv", index=False)
    checkpoint_rows = []
    for path in sorted((EXP / "checkpoints").rglob("*.pt")):
        checkpoint_rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size, "dataset_sha256": sha256(DATASET)})
    pd.DataFrame(checkpoint_rows).to_csv(EXP / "checkpoint_manifest.csv", index=False)
    write_json(EXP / "metrics.json", {**metrics, "bootstrap_contrasts": bootstrap.to_dict("records")})
    write_json(EXP / "summary.json", {"status": "COMPLETE", "dataset_sha256": sha256(DATASET),
        "primary_metric": "MAE ΔE firmado intrafórmula, estimando A, meV/átomo",
        "models": metrics, "bootstrap": bootstrap.to_dict("records"),
        "historical_test_accessed": False})
    resources["status"] = "COMPLETE"
    resources["wall_seconds"] = time.monotonic() - STARTED
    resources["process_cpu_seconds"] = time.process_time() - PROCESS_STARTED
    resources["peak_rss_bytes"] = rss_peak_bytes()
    write_json(EXP / "resources.json", resources)
    report = ["# F4 directo en ABS₃: MatGPT y MatterGPT", "",
        "## Cohorte y método", "",
        f"Se evaluaron {len(data)} estructuras en {data.composition_id.nunique()} composiciones ABS₃ con tres polimorfos por fórmula. Se usó el dataset `data/abs3/abs3_development_f4_geometry/dataset.csv`, folds agrupados históricos, cinco pliegues y semillas 42/123/2026. El test histórico no se leyó.", "",
        "Los backbones originales se congelaron en F0 y se descongelaron solo los bloques 8–11 en F4. Se compartió un warmup de MLP8 por pareja; la selección de checkpoints usó únicamente inner-validation. La geometría se escaló por fold con inner-train, y el Ridge M2 permaneció fijo.", "",
        "Tres SLICES tienen marca histórica de no equivalencia bajo la validación de grafo estricta; se conservaron exactamente las entradas con las que los extractores congelados habían sido auditados. Tokenización: cero OOV y cero truncamiento.", "",
        "## Resultados OOF", "", json.dumps(metrics, indent=2, ensure_ascii=False), "",
        "## Bootstrap apareado por composición", "", bootstrap.to_markdown(index=False), "",
        "## Limitaciones", "",
        "La evaluación OOF utiliza el conjunto de desarrollo ya explorado y no constituye una prueba ciega. MatGPT y MatterGPT difieren en arquitectura y tokenizador, por lo que la comparación entre familias no aísla causalmente el preentrenamiento.", ""]
    (EXP / "reporte_cientifico.md").write_text("\n".join(report), encoding="utf-8")
    (EXP / "README.md").write_text("# ABS3 MatGPT/MatterGPT F4\n\nEjecución local. Pilot: `python scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py --stage pilot`. Completo: `python scripts/ABS3/abs3_matgpt_mattergpt_original_f4_71/run_experiment.py --stage full`. La ejecución completa omite parejas fold-semilla ya completas; una pareja interrumpida se vuelve a ejecutar desde su warmup compartido. El log `train_history.csv` se actualiza por época. Consulte `preflight.json`, `resources.json` y `reporte_cientifico.md`.\n", encoding="utf-8")
    LOGGER.info("Full OOF completed: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
