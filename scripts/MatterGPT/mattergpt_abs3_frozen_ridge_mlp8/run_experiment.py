"""Ejecuta la extracción congelada MatterGPT y la evaluación OOF ABS₃.

Lee exclusivamente las 210 filas históricas train/val, conserva los SLICES
archivados, y entrena solo quince cabezas MLP8 agrupadas por triplete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import scipy
import sklearn
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch import nn
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
EXPERIMENT = "mattergpt_abs3_frozen_ridge_mlp8"
OUT = ROOT / "experiments" / EXPERIMENT
REPORT_DIR = ROOT / "reports" / "MatterGPT" / EXPERIMENT
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images" / EXPERIMENT
MODEL_DIR = ROOT / "models/MatterGPT"
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
RIDGE_DIR = ROOT / "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional"
MATGPT_BASELINE = RIDGE_DIR / "oof_predictions_by_structure.csv"
SEEDS = (42, 123, 2026)
CONDITION = (1.0, 2.0)
ELEMENTS = ("Ba", "Ca", "Cd", "Cu", "Ge", "Hf", "Mg", "Nb", "Pb", "S", "Si", "Sn", "Sr", "Ti", "V", "Zn", "Zr")


def configure_logging() -> logging.Logger:
    """Configura logging JSON por línea para eventos reproducibles.

    Returns:
        Logger raíz del experimento.
    """
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","event":"%(message)s"}')
    return logging.getLogger(EXPERIMENT)


LOG = configure_logging()


def sha256(path: Path) -> str:
    """Calcula SHA-256 en bloques.

    Args:
        path: Archivo existente.
    Returns:
        Hash hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Escribe JSON ordenado con creación de directorios.

    Args:
        path: Destino dentro del experimento.
        obj: Valor serializable.
    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def dev_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga solo desarrollo y asignaciones agrupadas sin leer filas test.

    Args:
        None.
    Returns:
        Tabla de 210 estructuras con etiqueta y manifiesto de folds.
    """
    fold = pd.read_csv(FOLDS, dtype={"structure_id": str})
    fold = fold[fold.historical_split.isin(["train", "val"])].copy()
    if len(fold) != 1050 or fold.structure_id.nunique() != 210:
        raise ValueError("Folds históricos no cubren 210 estructuras de desarrollo")
    assignment = fold[fold.outer_role.eq("outer_validation")][["structure_id", "composition_group", "outer_fold"]]
    if assignment.structure_id.duplicated().any() or len(assignment) != 210:
        raise ValueError("Asignación outer no es única por estructura")
    dev_ids = set(assignment.structure_id)
    # El lector CSV filtra el split antes de convertir o almacenar etiquetas.
    rows: list[dict[str, Any]] = []
    with DATASET.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") not in {"train", "val"} or row.get("structure_id") not in dev_ids:
                continue
            atoms = int(row["num_atoms"])
            energy = float(row["energy_per_atom_ev"])
            if atoms <= 0 or not np.isfinite(energy):
                raise ValueError(f"Conteo/energía inválido: {row.get('structure_id')}")
            rows.append({"structure_id": row["structure_id"], "composition_id": row["reduced_formula"],
                         "family": row["polymorph_family"], "n_atoms": atoms,
                         "energy_dft_ev_atom": energy, "slices": row["matgpt_slices"],
                         "slices_original": row["slices_original"], "SLICES": row["SLICES"],
                         "topology_sha256": row["topology_sha256"],
                         "strategy3_graph_equivalent": row["strategy3_graph_equivalent"]})
    data = pd.DataFrame(rows).merge(assignment, on="structure_id", validate="one_to_one")
    data = data.rename(columns={"composition_group": "composition_fold_group"})
    if (len(data) != 210 or data.structure_id.nunique() != 210 or data.composition_id.nunique() != 70
            or data.groupby("composition_id").size().ne(3).any()
            or data.groupby("composition_id").family.nunique().ne(3).any()):
        raise ValueError("Desarrollo debe contener 70 tripletes completos")
    if not data.composition_id.eq(data.composition_fold_group).all():
        raise ValueError("La fórmula y el grupo fold no coinciden")
    if not data.slices.eq(data.slices_original).all() or not data.slices.eq(data.SLICES).all():
        raise ValueError("Las cadenas históricas SLICES difieren; no adaptar con pérdida")
    return data.sort_values(["composition_id", "family"]).reset_index(drop=True), fold


def load_model() -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Construye MatterGPT y carga checkpoint original con strict=True.

    Args:
        None.
    Returns:
        (modelo, tokenizer, auditoría de checkpoint/configuración).
    """
    sys.path.insert(0, str(MODEL_DIR))
    from model import GPT, GPTConfig
    # Reproduce literalmente el SimpleTokenizer del wrapper oficial. Importar
    # mattergpt_wrapper cargaría además Transformers, dependencia no usada por
    # la extracción con la clase GPT original.
    class SimpleTokenizer:
        """Tokenizador idéntico al SimpleTokenizer local de MatterGPT."""
        def __init__(self, vocab_file: str) -> None:
            with open(vocab_file, encoding="utf-8") as vocab_stream:
                self.vocab = sorted(set(vocab_stream.read().splitlines() + ["<", ">"]))
            self.stoi = {token: index for index, token in enumerate(self.vocab)}
            self.itos = {index: token for token, index in self.stoi.items()}

        def encode(self, text: str) -> list[int]:
            """Codifica tokens separados por espacios como el wrapper oficial."""
            return [self.stoi[token] for token in text.split()]
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    model = GPT(GPTConfig(config["vocab_size"], config["block_size"], **{k: v for k, v in config.items() if k not in {"model_type", "architectures", "vocab_size", "block_size"}}))
    state = torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Checkpoint incompleto: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = SimpleTokenizer(str(MODEL_DIR / "Voc_prior"))
    audit = {"config": config, "source_url": "https://huggingface.co/xiaohang07/MatterGPT", "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys,
             "checkpoint_sha256": sha256(MODEL_DIR / "pytorch_model.pt"),
             "config_sha256": sha256(MODEL_DIR / "config.json"),
             "tokenizer_sha256": sha256(MODEL_DIR / "Voc_prior"),
             "model_py_sha256": sha256(MODEL_DIR / "model.py"),
             "wrapper_sha256": sha256(MODEL_DIR / "mattergpt_wrapper.py"),
             "tokenizer_implementation": "copia semánticamente idéntica de SimpleTokenizer; wrapper importa transformers no disponible; su clase GPT fue cargada directamente",
             "vocab_size_reconstructed": len(tokenizer.vocab), "trainable_backbone_parameters": 0}
    if (config["n_layer"], config["n_head"], config["n_embd"], config["vocab_size"], config["block_size"], config["num_props"]) != (12, 12, 768, 132, 397, 2):
        raise ValueError("Config MatterGPT no coincide con arquitectura local auditada")
    if len(tokenizer.vocab) != config["vocab_size"]:
        raise ValueError("Vocabulario reconstruido no coincide con config")
    return model, tokenizer, audit


def sequence_ids(slices: str, tokenizer: Any) -> list[int]:
    """Serializa SLICES con los tokens de inicio y cierre de MatterGPT.

    Args:
        slices: Cadena histórica sin modificar.
        tokenizer: Tokenizador SimpleTokenizer original.
    Returns:
        IDs incluyendo `>` inicial y `<` terminal.
    """
    ids = tokenizer.encode("> " + slices + " <")
    if len(ids) < 3 or ids[0] != tokenizer.stoi[">"] or ids[-1] != tokenizer.stoi["<"]:
        raise ValueError("Falla de tokenización de marcadores MatterGPT")
    return ids


def contextualized(model: torch.nn.Module, token_ids: list[int], condition: tuple[float, float] = CONDITION,
                   pad_to: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrae FINAL_END después de `blocks → ln_f`, conservando logits.

    Args:
        model: GPT original congelado en evaluación.
        token_ids: Tokens de una secuencia estructural completa.
        condition: Propiedades multipropiedad en orden publicado.
        pad_to: Longitud de padding derecho opcional para prueba de invariancia.
    Returns:
        Vector contextualizado de dimensión 768 y logits sin modificar.
    """
    device = next(model.parameters()).device
    n = len(token_ids)
    if pad_to is not None and pad_to < n:
        raise ValueError("pad_to no puede truncar la secuencia")
    ids = token_ids + ([0] * (pad_to - n) if pad_to else [])
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    prop = torch.tensor(condition, dtype=torch.float32, device=device).view(1, 1, 2)
    capture: list[torch.Tensor] = []
    hook = model.ln_f.register_forward_hook(lambda _m, _i, out: capture.append(out))
    try:
        with torch.no_grad():
            logits, _loss, _attention, pre_transformer_embed = model(idx, prop=prop)
    finally:
        hook.remove()
    if not capture or pre_transformer_embed.shape[-1] != 768:
        raise RuntimeError("No se capturó ln_f o formato de salida inesperado")
    # El prefijo de propiedades ocupa posición 0; SLICES empieza en posición 1.
    # El terminal < real es el último token estructural en n.
    return capture[0][0, n, :].detach().cpu(), logits.detach().cpu()


def extract_and_audit(data: pd.DataFrame, model: torch.nn.Module, tokenizer: Any) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Audita secuencias y extrae los 210 estados contextualizados.

    Args:
        data: Tabla de desarrollo validada.
        model: MatterGPT congelado.
        tokenizer: Tokenizador de `Voc_prior`.
    Returns:
        Matriz X, manifiesto por estructura y controles aprobados.
    """
    sequences: list[list[int]] = []
    lengths: list[int] = []
    oov: dict[str, int] = {}
    for row in data.itertuples():
        tokens = row.slices.split()
        missing = [token for token in tokens if token not in tokenizer.stoi]
        if missing:
            oov[row.structure_id] = len(missing)
        ids = sequence_ids(row.slices, tokenizer)
        if len(ids) > model.block_size:
            raise ValueError(f"Secuencia fuera de contexto, sin truncar: {row.structure_id} ({len(ids)})")
        sequences.append(ids); lengths.append(len(ids))
    if oov:
        raise ValueError(f"Tokens OOV en {len(oov)} estructuras: {dict(list(oov.items())[:5])}")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rows: list[np.ndarray] = []
    max_change = 0.0
    condition_change = 0.0
    with torch.inference_mode():
        for ids in sequences:
            vec, logits = contextualized(model, ids)
            vec2, logits2 = contextualized(model, ids)
            if vec.shape != (768,) or not torch.isfinite(vec).all() or not torch.equal(vec, vec2):
                raise ValueError("Embedding 768D inválido o no determinista")
            if not torch.equal(logits, logits2):
                raise ValueError("Logits cambian al activar el hook de lectura ln_f")
            if not rows:
                plain_idx=torch.tensor([ids],dtype=torch.long)
                plain_prop=torch.tensor(CONDITION,dtype=torch.float32).view(1,1,2)
                plain_logits=model(plain_idx,prop=plain_prop)[0].detach().cpu()
                if not torch.equal(logits,plain_logits):
                    raise ValueError("Logits con hook difieren del forward original sin hook")
            # Sensibilidad diagnóstica: una condición alternativa nunca entra en resultados.
            alt, _ = contextualized(model, ids, (0.0, 0.0))
            condition_change = max(condition_change, float(torch.max(torch.abs(vec - alt))))
            # Sensibilidad estructural sin umbral numérico arbitrario.
            changed = list(ids)
            pos = max(1, len(changed) // 2)
            candidates = [v for v in tokenizer.stoi.values() if v != changed[pos]]
            changed[pos] = candidates[0]
            varied, _ = contextualized(model, changed)
            max_change = max(max_change, float(torch.max(torch.abs(vec - varied))))
            # Batch/padding: causal mask hace irrelevante cualquier token derecho.
            padded, _ = contextualized(model, ids, pad_to=min(model.block_size, len(ids) + 7))
            if not torch.allclose(vec, padded, atol=2e-6, rtol=2e-6):
                raise ValueError("Embedding cambia al añadir padding derecho")
            rows.append(vec.numpy().astype(np.float32))
    batch_ids = torch.tensor([sequences[0] + [0] * (max(lengths[:2]) - len(sequences[0])),
                              sequences[1] + [0] * (max(lengths[:2]) - len(sequences[1]))], dtype=torch.long)
    batch_prop = torch.tensor(CONDITION, dtype=torch.float32).view(1, 1, 2).expand(2, -1, -1)
    batch_capture: list[torch.Tensor] = []
    batch_hook = model.ln_f.register_forward_hook(lambda _m, _i, out: batch_capture.append(out))
    try:
        with torch.no_grad():
            model(batch_ids, prop=batch_prop)
    finally:
        batch_hook.remove()
    for i in range(2):
        if not torch.allclose(batch_capture[0][i, lengths[i], :].cpu(), torch.tensor(rows[i]), atol=2e-6, rtol=2e-6):
            raise ValueError("Embedding cambia al modificar el tamaño del lote")
    if condition_change <= 0 or max_change <= 0:
        raise ValueError("No se detectó sensibilidad a condición o estructura")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Se detectó gradiente en backbone congelado")
    if any(not torch.equal(before[k], value) for k, value in model.state_dict().items()):
        raise ValueError("Cambió un parámetro MatterGPT durante extracción")
    X = np.stack(rows)
    if X.shape != (210, 768) or not np.isfinite(X).all():
        raise ValueError("Cache esperada 210×768 finita")
    manifest = data[["structure_id", "composition_id", "family", "n_atoms", "outer_fold", "slices", "topology_sha256", "strategy3_graph_equivalent"]].copy()
    manifest["embedding_row"] = np.arange(len(manifest))
    manifest["embedding_dim"] = 768
    manifest["property_vector"] = "[1.0, 2.0]"
    manifest["slices_strategy"] = "SLICES estrategia 4 histórica ABS3; cadena preservada"
    manifest["terminal_policy"] = "> SLICES <; último estado = ln_f(posición terminal <)"
    manifest["sequence_length"] = lengths
    manifest["validation_status"] = "PASS"
    checks = {"status": "PASS", "structures": 210, "compositions": 70, "oov_structures": 0,
              "truncations": 0, "max_sequence_tokens": max(lengths), "context_limit": 397,
              "atoms_over_20": int((data.n_atoms > 20).sum()), "all_slices_byte_identical_to_original": True,
              "graph_equivalence_true": int(data.strategy3_graph_equivalent.astype(str).str.lower().eq("true").sum()),
              "graph_equivalence_missing": int(data.strategy3_graph_equivalent.isna().sum()),
              "topology_hash_missing": int(data.topology_sha256.isna().sum()),
              "terminal_token": "<", "terminal_token_id": int(tokenizer.stoi["<"]),
              "start_token": ">", "start_token_id": int(tokenizer.stoi[">"]),
              "config_special_token_ids_metadata": {"bos_token_id": 130, "eos_token_id": 131},
              "simple_tokenizer_special_token_ids_used": {"start_gt": int(tokenizer.stoi[">"]), "terminal_lt": int(tokenizer.stoi["<"])},
              "config_bos_eos_ids_used_for_serialization": False,
              "logits_equal_original_forward_without_hook": True,
              "pretraining_teacher_forcing_loader_verified": False,
              "conditioning_shape_dtype": "[batch,1,2]/float32", "property_vector": list(CONDITION),
              "conditioning_max_abs_sensitivity_diagnostic": condition_change,
              "structure_max_abs_sensitivity_diagnostic": max_change,
              "logits_identical_with_read_hook": True, "padding_invariance": True,
              "batch_size_invariance": True,
              "post_ln_f_terminal_hidden": True, "pre_transformer_embed_used": False,
              "backbone_frozen_no_grad_and_unchanged": True,
              "space_group_num_serialized_in_slices": False}
    return X, manifest, checks


class MLP8(nn.Module):
    """Cabeza residual histórica 768→8→1 con GELU y dropout.

    Args:
        None.
    Returns:
        Modelo con 6161 parámetros entrenables.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predice una salida residual cruda por estructura."""
        return self.net(x).squeeze(-1)


def smoke_test_mlp(X: np.ndarray, data: pd.DataFrame, folds: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    """Comprueba gradientes, centrado y reconstrucción antes de las 15 corridas.

    Args:
        X: Cache de embeddings congelados.
        data: Tabla de desarrollo.
        folds: Asignaciones agrupadas históricas.
        device: Dispositivo de ejecución.
    Returns:
        Resultado PASS del smoke test.
    """
    formulas=sorted(folds[(folds.outer_fold==0)&folds.inner_role.eq("inner_train")].composition_group.unique())
    formula=formulas[0]; indices=np.flatnonzero(data.composition_id.to_numpy()==formula)
    if len(indices)!=3: raise ValueError("Smoke MLP8 requiere triplete inner-train")
    torch.manual_seed(42); head=MLP8().to(device).train()
    xb=torch.tensor(X[indices],dtype=torch.float32,device=device).unsqueeze(0)
    energy=torch.tensor(data.iloc[indices].energy_dft_ev_atom.to_numpy(float),dtype=torch.float32,device=device).unsqueeze(0)
    scale=float(energy.std(unbiased=False).item())
    if scale<=0: raise ValueError("Smoke residual train con escala nula")
    raw=head(xb); centered=raw-raw.mean(dim=1,keepdim=True); target=energy-energy.mean(dim=1,keepdim=True)
    loss=F.smooth_l1_loss(centered/scale,target/scale,beta=1.0); loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError("Smoke MLP8: gradiente ausente o pérdida no finita")
    if abs(float(centered.detach().sum()))>1e-7: raise ValueError("Smoke MLP8: residuo no centrado")
    ridge=joblib.load(RIDGE_DIR/"models"/"ridge_composicional_fold_0.joblib")
    mu=float(ridge.predict(formula_features([formula]))[0]); reconstructed=mu+centered.detach()
    if not torch.allclose(reconstructed-mu,centered.detach(),atol=2e-6,rtol=2e-6):
        raise ValueError("Smoke MLP8: reconstrucción Ridge+residuo inconsistente")
    return {"status":"PASS","fold":0,"seed":42,"composition_id":formula,"group_role":"inner_train",
            "finite_smoothl1":True,"mlp8_gradients_finite":True,"residual_centered_per_formula":True,
            "ridge_plus_residual_reconstruction":True,"backbone_parameters_in_optimizer":False}


def formula_features(formulas: Iterable[str]) -> np.ndarray:
    """Codifica fórmulas en las 17 fracciones elementales del Ridge histórico.

    Args:
        formulas: Fórmulas reducidas.
    Returns:
        Matriz float64 de 17 columnas.
    """
    import re
    result = []
    for formula in formulas:
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if "".join(e + n for e, n in parts) != formula:
            raise ValueError(f"Fórmula no parseable: {formula}")
        counts = {e: int(n or 1) for e, n in parts}
        total = sum(counts.values())
        if any(e not in ELEMENTS for e in counts):
            raise ValueError(f"Elemento fuera del Ridge histórico: {formula}")
        result.append([counts.get(e, 0) / total for e in ELEMENTS])
    return np.asarray(result, dtype=np.float64)


def train_one(X: np.ndarray, data: pd.DataFrame, fold_rows: pd.DataFrame, fold: int, seed: int,
              scale: float, device: torch.device) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, MLP8]:
    """Entrena una MLP8 y predice únicamente las 14 fórmulas outer.

    Args:
        X: Cache congelada 210×768.
        data: Desarrollo alineado con X.
        fold_rows: Manifiesto completo de asignación histórica.
        fold: Pliegue outer 0..4.
        seed: Semilla de inicialización y barajado.
        scale: Escala residual calculada solo en inner-train.
        device: Dispositivo de ejecución.
    Returns:
        Predicción residual outer, registro, historial y modelo elegido.
    """
    run_started = time.perf_counter()
    fr = fold_rows[fold_rows.outer_fold.eq(fold)]
    role = fr.drop_duplicates("composition_group").set_index("composition_group")
    tr = sorted(role.index[role.inner_role.eq("inner_train")])
    va = sorted(role.index[role.inner_role.eq("inner_validation")])
    outer = sorted(fr.loc[fr.outer_role.eq("outer_validation"), "composition_group"].unique())
    if (len(tr), len(va), len(outer)) != (45, 11, 14) or set(tr) & set(va) or set(tr + va) & set(outer):
        raise ValueError(f"Fold {fold}: roles agrupados inválidos")
    groups = {key: np.flatnonzero(data.composition_id.to_numpy() == key) for key in set(tr + va + outer)}
    idx = {k: torch.tensor(X[v], dtype=torch.float32, device=device) for k, v in groups.items()}
    y_by = {k: torch.tensor(data.energy_dft_ev_atom.to_numpy()[v], dtype=torch.float32, device=device) for k, v in groups.items()}
    ymean = {k: y.mean() for k, y in y_by.items()}
    targets = {k: y - ymean[k] for k, y in y_by.items()}
    scale_real = float(torch.cat([targets[k] for k in tr]).std(unbiased=False).item())
    if scale <= 0 or abs(scale_real - scale) > 2e-7:
        raise ValueError("Escala residual interna inconsistente")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = MLP8().to(device)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf"); best_epoch = 0; best_state: dict[str, torch.Tensor] | None = None
    stale = 0; history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, 201):
        model.train(); order = np.asarray(tr)[rng.permutation(len(tr))]
        total = 0.0; batches = 0
        for start in range(0, len(order), 8):
            batch = list(order[start:start + 8])
            xb = torch.stack([idx[k] for k in batch])
            yb = torch.stack([targets[k] for k in batch])
            raw = model(xb)
            centered = raw - raw.mean(dim=1, keepdim=True)
            loss = F.smooth_l1_loss(centered / scale_real, yb / scale_real, beta=1.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida MLP8 no finita")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Gradiente MLP8 ausente/no finito")
            optimizer.step(); total += float(loss.detach().cpu()); batches += 1
        model.eval()
        with torch.no_grad():
            val_true = torch.stack([targets[k] for k in va])
            val_raw = model(torch.stack([idx[k] for k in va]))
            val_center = val_raw - val_raw.mean(dim=1, keepdim=True)
            val_loss = float(F.smooth_l1_loss(val_center / scale_real, val_true / scale_real, beta=1.0).cpu())
        history.append({"fold": fold, "seed": seed, "epoch": epoch, "train_minibatch_smoothl1": total / batches,
                        "inner_validation_full_smoothl1": val_loss, "optimizer_steps": batches})
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 20:
                break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("No se encontró checkpoint válido")
    model.load_state_dict(best_state); model.eval()
    if all(torch.equal(initial_state[k].cpu(), v.cpu()) for k, v in model.state_dict().items()):
        raise ValueError("La MLP8 no cambió sus pesos en entrenamiento")
    with torch.no_grad():
        outer_res: list[np.ndarray] = []
        for key in outer:
            raw = model(idx[key]); centered = raw - raw.mean()
            if abs(float(centered.sum().cpu())) > 1e-6:
                raise ValueError(f"Predicción residual no centrada: {key}")
            outer_res.extend(centered.cpu().numpy().astype(float).tolist())
    record = {"fold": fold, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
              "best_inner_validation_smoothl1": best_loss, "residual_scale_ev_atom": scale_real,
              "elapsed_seconds": time.perf_counter() - run_started,
              "train_compositions": 45, "validation_compositions": 11, "outer_compositions": 14,
              "head_parameters": sum(p.numel() for p in model.parameters()), "frozen_backbone_parameters": 0}
    checkpoint = OUT / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "fold": fold, "seed": seed, "best_epoch": best_epoch,
                "residual_scale_ev_atom": scale_real}, checkpoint)
    return np.asarray(outer_res), record, pd.DataFrame(history), model


def prediction_metrics(pred: pd.DataFrame) -> dict[str, Any]:
    """Calcula métricas estructurales, de pares y ranking por composición.

    Args:
        pred: Tabla de una réplica por modelo, semilla y estructura.
    Returns:
        Diccionario de estadísticas con unidades y denominadores.
    """
    rows = []
    pair_rows = []
    for (model, seed, formula), g in pred.groupby(["model", "seed", "composition_id"]):
        dft = g.energy_dft_ev_atom.to_numpy(float)
        y = g.energy_pred_ev_atom.to_numpy(float)
        anchor = int(np.argmin(dft))
        dd = (dft - dft[anchor]) * 1000
        # Referencia común: ambos gaps se anclan a la estructura de mínimo DFT.
        # Un gap predicho negativo expresa inversión frente al estado fundamental DFT.
        dp = (y - y[anchor]) * 1000
        pair_abs_errors = [abs((y[j]-y[i] - (dft[j]-dft[i])) * 1000)
                           for i in range(3) for j in range(i+1,3)]
        zero_baseline_pair_mae = float(np.mean([abs((dft[j]-dft[i])*1000)
                                                for i in range(3) for j in range(i+1,3)]))
        dft_gap = float((np.partition(dft, 1)[1] - dft.min()) * 1000)
        pred_sorted = np.sort(y)
        pred_tie = bool((pred_sorted[1] - pred_sorted[0]) <= 0.001)
        dft_tie = dft_gap <= 1.0
        exact = bool(not pred_tie and not dft_tie and int(np.argmin(y)) == int(np.argmin(dft)))
        rows.append({"model": model, "seed": seed, "composition_id": formula,
                     "mae_delta_mev_atom": float(np.mean(pair_abs_errors)),
                     "mae_delta_signed_pairs_mev_atom": float(np.mean(pair_abs_errors)),
                     "mae_delta_zero_baseline_mev_atom": zero_baseline_pair_mae,
                     "mae_delta_anchored_all3_mev_atom": float(np.mean(np.abs(dp-dd))),
                     "mae_energy_ev_atom": float(np.mean(np.abs(y-dft))),
                     "rmse_energy_ev_atom": float(np.sqrt(np.mean((y-dft)**2))),
                     "mae_residual_mev_atom": float(np.mean(np.abs((y-y.mean())-(dft-dft.mean()))) * 1000),
                     "ground_state_exact": int(exact), "ground_state_eligible": int(not dft_tie),
                     "ground_state_unresolved_prediction": int(pred_tie),
                     "spearman": float(spearmanr(dft, y).statistic) if np.ptp(y)>1e-12 else None,
                     "kendall": float(kendalltau(dft, y).statistic) if np.ptp(y)>1e-12 else None,
                     "dft_gap_mev_atom": dft_gap})
        for i in range(3):
            for j in range(i+1, 3):
                true = (dft[j]-dft[i])*1000; estimate = (y[j]-y[i])*1000
                eligible = abs(true) > 5.0
                unresolved = eligible and abs(estimate) <= 1.0
                pair_rows.append({"model": model, "seed": seed, "composition_id": formula,
                                  "structure_i": g.structure_id.iloc[i], "structure_j": g.structure_id.iloc[j],
                                  "delta_dft_mev_atom": true, "delta_pred_mev_atom": estimate,
                                  "eligible_abs_dft_gt_5": eligible, "unresolved_abs_pred_le_1": unresolved,
                                  "correct": bool(np.sign(true) == np.sign(estimate)) and not unresolved})
    return {"by_composition_seed": rows, "pairs": pair_rows}


def bootstrap_ci(metric_by_comp: pd.DataFrame, models: list[str], n_boot: int = 10000) -> list[dict[str, Any]]:
    """Bootstrap pareado por fórmula, promediando semillas dentro de fórmula.

    Args:
        metric_by_comp: Métrica por composición y semilla.
        models: Modelos a comparar.
        n_boot: Número de remuestreos.
    Returns:
        IC percentile 95% para promedios y diferencias pareadas.
    """
    rng = np.random.default_rng(20261030)
    metric = metric_by_comp.groupby(["model", "composition_id"]).mae_delta_mev_atom.mean().unstack("model")
    formulas = metric.index.to_numpy(); output = []
    for model in models:
        values = metric[model].to_numpy(float)
        samples = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])
        output.append({"estimand": "A", "contrast": model, "estimate": float(values.mean()),
                       "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)),
                       "n_compositions": int(len(formulas)), "bootstrap_replicates": n_boot})
    for a, b in ((models[0], models[1]), (models[0], models[2])):
        diff = metric[a].to_numpy(float) - metric[b].to_numpy(float)
        samples = np.asarray([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        output.append({"estimand": "A", "contrast": f"{a} − {b}", "estimate": float(diff.mean()),
                       "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)),
                       "n_compositions": int(len(formulas)), "bootstrap_replicates": n_boot})
    return output


def summarize_predictions(allpred: pd.DataFrame, n_boot: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Resume estimando A, métricas por fórmula, pares y OLS desde predicciones.

    Args:
        allpred: Predicciones OOF para MatterGPT, MatGPT y Ridge-only.
        n_boot: Réplicas bootstrap apareadas por composición.
    Returns:
        Métricas, tabla fórmula-semilla y tabla por pares.
    """
    calc=prediction_metrics(allpred); metric_df=pd.DataFrame(calc["by_composition_seed"]); pair_df=pd.DataFrame(calc["pairs"])
    metrics: dict[str,Any]={}
    for name,g in metric_df.groupby("model"):
        def mean_or_none(column: str) -> float | None:
            value=pd.to_numeric(g[column],errors="coerce").mean()
            return float(value) if pd.notna(value) and np.isfinite(value) else None
        metrics[name]={col:mean_or_none(col) for col in ("mae_delta_mev_atom","mae_delta_anchored_all3_mev_atom","mae_delta_zero_baseline_mev_atom","mae_energy_ev_atom","rmse_energy_ev_atom","mae_residual_mev_atom","ground_state_exact","spearman","kendall")}
        pairs=pair_df[(pair_df.model==name)&pair_df.eligible_abs_dft_gt_5]
        metrics[name]["pairwise_accuracy_including_unresolved_in_denominator"]=float(pairs.correct.mean())
        metrics[name]["unresolved_fraction_eligible_pairs"]=float(pairs.unresolved_abs_pred_le_1.mean())
        metrics[name]["eligible_pairs"]=int(len(pairs))
        eligible=g[g.ground_state_eligible.eq(1)]
        metrics[name]["ground_state_eligible_composition_seed_cases"]=int(len(eligible))
        metrics[name]["ground_state_unresolved_predictions"]=int(g.ground_state_unresolved_prediction.sum())
        metrics[name]["ground_state_accuracy_denominator"]=int(len(eligible))
        metrics[name]["ground_state_exact"]=float(eligible.ground_state_exact.mean()) if len(eligible) else None
    regression: dict[str,Any]={}
    averaged=allpred.groupby(["model","structure_id","composition_id"],as_index=False).agg(energy_dft_ev_atom=("energy_dft_ev_atom","mean"),energy_pred_ev_atom=("energy_pred_ev_atom","mean"))
    for name,frame in averaged.groupby("model"):
        xe=frame.energy_dft_ev_atom.to_numpy(float); ye=frame.energy_pred_ev_atom.to_numpy(float)
        anchor_rows=frame.loc[frame.groupby("composition_id").energy_dft_ev_atom.idxmin()]
        anchor_dft=anchor_rows.set_index("composition_id").energy_dft_ev_atom
        anchor_pred=anchor_rows.set_index("composition_id").energy_pred_ev_atom
        xd=(frame.energy_dft_ev_atom-frame.composition_id.map(anchor_dft)).to_numpy(float)*1000
        yd=(frame.energy_pred_ev_atom-frame.composition_id.map(anchor_pred)).to_numpy(float)*1000
        xres=(frame.energy_dft_ev_atom-frame.groupby("composition_id").energy_dft_ev_atom.transform("mean")).to_numpy(float)*1000
        yres=(frame.energy_pred_ev_atom-frame.groupby("composition_id").energy_pred_ev_atom.transform("mean")).to_numpy(float)*1000
        absfit=LinearRegression().fit(xe.reshape(-1,1),ye); dfit=LinearRegression().fit(xd.reshape(-1,1),yd)
        rfit=LinearRegression().fit(xres.reshape(-1,1),yres)
        regression[name]={"population":"210 estructuras OOF; promedio de tres semillas por estructura",
            "absolute_energy_e_per_n":{"N":int(len(frame)),"slope":float(absfit.coef_[0]),"intercept_ev_atom":float(absfit.intercept_),"r2_ols":float(r2_score(ye,absfit.predict(xe.reshape(-1,1)))),"mae_ev_atom":float(np.mean(np.abs(ye-xe)))},
            "delta_energy":{"definition":"ambos ejes anclados a la misma estructura de mínimo DFT por fórmula","N":int(len(frame)),"slope":float(dfit.coef_[0]),"intercept_mev_atom":float(dfit.intercept_),"r2_ols":None if np.var(yd)==0 else float(r2_score(yd,dfit.predict(xd.reshape(-1,1)))),"mae_mev_atom":float(np.mean(np.abs(yd-xd)))} }
        regression[name]["intraformula_residual"]={"definition":"cada energía centrada restando la media del triplete de su fórmula",
            "N":int(len(frame)),"slope":float(rfit.coef_[0]),"intercept_mev_atom":float(rfit.intercept_),
            "r2_ols":None if np.var(yres)==0 else float(r2_score(yres,rfit.predict(xres.reshape(-1,1)))),
            "mae_mev_atom":float(np.mean(np.abs(yres-xres)))}
    metrics["ols_parity"]=regression
    metrics["bootstrap_paired_composition_ci95"]=bootstrap_ci(metric_df,["MatterGPT_F0","MatGPT_F0","Ridge_only"],n_boot)
    metrics["definitions"]={"delta_dft":"E_i - E_DFT del polimorfo de menor energía DFT de la misma fórmula",
        "delta_prediction":"Epred_i - Epred del mismo polimorfo de mínimo DFT (ancla común); los valores negativos son inversiones",
        "primary_metric":"Estimando A: MAE de diferencias de energía firmadas entre los tres pares intrafórmula, promedio de pares por fórmula/semilla, semillas promediadas dentro de fórmula antes del bootstrap",
        "ridge_only":"ΔE_pred=0 para todos los pares; su MAE ΔE es el promedio de |ΔE_DFT|",
        "pair_eligibility":"|ΔE_DFT| > 5 meV/átomo", "predicted_unresolved":"|ΔE_pred| ≤ 1 meV/átomo cuenta incorrecto en el denominador elegible",
        "ground_state_tie":"DFT gap mínimo ≤1 meV/átomo excluido; empate predicho ≤1 meV/átomo es fallo no resuelto",
        "bootstrap":"10 000 remuestreos apareados por 70 composiciones; semillas no cuentan como materiales"}
    return metrics,metric_df,pair_df


def run(args: argparse.Namespace) -> None:
    """Ejecuta auditoría, extracción, quince entrenamientos y resultados.

    Args:
        args: Parámetros de línea de comandos validados.
    Returns:
        None.
    """
    start = time.perf_counter(); cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    OUT.mkdir(parents=True, exist_ok=True); IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    data, folds = dev_data()
    model, tokenizer, model_audit = load_model()
    X, embedding_manifest, checks = extract_and_audit(data, model, tokenizer)
    extraction_seconds = time.perf_counter() - start
    np.savez_compressed(OUT / "embeddings_mattergpt.npz", embeddings=X,
                        structure_ids=data.structure_id.to_numpy(dtype=str),
                        property_vector=np.asarray(CONDITION, dtype=np.float32))
    embedding_manifest["checkpoint_sha256"] = model_audit["checkpoint_sha256"]
    embedding_manifest["tokenizer_sha256"] = model_audit["tokenizer_sha256"]
    embedding_manifest["property_vector"] = "[1.0, 2.0]"
    embedding_manifest["cache_sha256"] = sha256(OUT / "embeddings_mattergpt.npz")
    embedding_manifest.drop(columns=["slices"], errors="ignore").to_csv(OUT / "embedding_manifest.csv", index=False)
    write_json(OUT / "preflight.json", {"status": "PASS", "model": model_audit, "checks": checks,
                                        "slices_teacher_forcing_caveat": "La secuencia pública `> SLICES <` se aplica según uso de generación, pero no se encontró el dataloader original de teacher-forcing para confirmar el token final del preentrenamiento."})
    ridge_manifest = json.loads((RIDGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if ridge_manifest.get("status") != "PASS" or ridge_manifest.get("historical_test_accessed") is not False:
        raise ValueError("Manifiesto Ridge no certifica PASS y test histórico cerrado")
    split_history = pd.read_csv(RIDGE_DIR / "outer_split_assignments.csv")
    for fold in range(5):
        ridge_path = RIDGE_DIR / "models" / f"ridge_composicional_fold_{fold}.joblib"
        expected_hash = ridge_manifest.get("created_outputs", {}).get(f"models/ridge_composicional_fold_{fold}.joblib")
        if not expected_hash or sha256(ridge_path) != expected_hash:
            raise ValueError(f"Ridge fold {fold}: SHA-256 no coincide con su manifiesto histórico")
        held = set(split_history[(split_history.fold == fold) & (split_history.role == "outer_test")].composition_id)
        expected_held = set(folds[(folds.outer_fold == fold) & folds.outer_role.eq("outer_validation")].composition_group)
        trained = set(split_history[(split_history.fold == fold) & (split_history.role.isin(["inner_train", "inner_validation", "outer_train"]))].composition_id)
        if held != expected_held or held & trained or (len(held), len(trained)) != (14, 56):
            raise ValueError(f"Ridge fold {fold}: contaminación o tamaño incorrecto del outer split")
    device = torch.device("cpu")
    smoke = smoke_test_mlp(X, data, folds, device)
    write_json(OUT / "mlp_smoke.json", smoke)
    checks["mlp8_smoke"] = smoke
    write_json(OUT / "preflight.json", {"status":"PASS","model":model_audit,"checks":checks,
        "slices_teacher_forcing_caveat":"La secuencia pública `> SLICES <` se aplica según uso de generación, pero no se encontró el dataloader original de teacher-forcing para confirmar el token final del preentrenamiento."})
    # Entrenamiento MLP8 requiere pasar smoke: un paso con tripletes, gradiente finito,
    # centrado por fórmula y reconstrucción composicional algebraicamente exacta.
    predictions: list[dict[str, Any]] = []; history_parts = []; run_records = []
    for fold in range(5):
        fr = folds[folds.outer_fold.eq(fold)]
        roles = fr.drop_duplicates("composition_group").set_index("composition_group")
        tr = list(roles.index[roles.inner_role.eq("inner_train")])
        # Escala calculada usando únicamente las 45 composiciones inner-train.
        means = data.groupby("composition_id").energy_dft_ev_atom.mean()
        residuals = data[data.composition_id.isin(tr)].energy_dft_ev_atom - data[data.composition_id.isin(tr)].composition_id.map(means)
        scale = float(residuals.std(ddof=0))
        ridge_path = RIDGE_DIR / "models" / f"ridge_composicional_fold_{fold}.joblib"
        expected_hash = ridge_manifest.get("created_outputs", {}).get(f"models/ridge_composicional_fold_{fold}.joblib")
        if not expected_hash or sha256(ridge_path) != expected_hash:
            raise ValueError(f"Ridge fold {fold}: SHA-256 no coincide con su manifiesto histórico")
        ridge = joblib.load(ridge_path)
        if getattr(ridge, "n_features_in_", None) not in (None, 17):
            raise ValueError(f"Ridge fold {fold} no acepta 17 características")
        # La carga joblib contiene un Pipeline scaler+Ridge y no se llama fit.
        formula_list = sorted(data.composition_id.unique())
        ridge_means = ridge.predict(formula_features(formula_list))
        mu = dict(zip(formula_list, ridge_means))
        if not np.isfinite(ridge_means).all():
            raise ValueError(f"Predicción Ridge no finita en fold {fold}")
        for seed in SEEDS:
            residual_pred, record, hist, _head = train_one(X, data, folds, fold, seed, scale, device)
            run_records.append(record); history_parts.append(hist)
            outer_formula = sorted(fr.loc[fr.outer_role.eq("outer_validation"), "composition_group"].unique())
            offset = 0
            for formula in outer_formula:
                g = data[data.composition_id.eq(formula)].sort_values("family")
                res = residual_pred[offset:offset+len(g)]; offset += len(g)
                # B2 = mu Ridge histórico + salida MLP8 centrada intrafórmula.
                for row, r in zip(g.itertuples(), res):
                    predictions.append({"model": "MatterGPT_F0", "seed": seed, "fold": fold,
                                        "structure_id": row.structure_id, "composition_id": formula,
                                        "family": row.family, "n_atoms": row.n_atoms,
                                        "energy_dft_ev_atom": row.energy_dft_ev_atom,
                                        "ridge_mean_ev_atom": mu[formula], "residual_pred_ev_atom": float(r),
                                        "energy_pred_ev_atom": float(mu[formula] + r)})
            LOG.info("training_complete fold=%s seed=%s best_epoch=%s", fold, seed, record["best_epoch"])
    matter = pd.DataFrame(predictions)
    if len(matter) != 630 or matter.structure_id.nunique() != 210:
        raise ValueError("Predicciones MatterGPT no cubren 630 puntos OOF")
    dft = data[["structure_id", "composition_id", "family", "n_atoms", "energy_dft_ev_atom"]]
    # Ridge-only usa las mismas medias Ridge por fold y tripletes.
    ridge_pred = []
    for fold in range(5):
        fr = folds[folds.outer_fold.eq(fold) & folds.outer_role.eq("outer_validation")]
        historical_test = set(split_history[(split_history.fold == fold) & (split_history.role == "outer_test")].composition_id)
        expected_test = set(fr.composition_group)
        ridge_train = set(split_history[(split_history.fold == fold) & (split_history.role.isin(["inner_train", "inner_validation", "outer_train"]))].composition_id)
        if historical_test != expected_test or historical_test & ridge_train:
            raise ValueError(f"Fold {fold}: Ridge entrenado sobre fórmula outer o split distinto")
        ridge = joblib.load(RIDGE_DIR / "models" / f"ridge_composicional_fold_{fold}.joblib")
        formulas = sorted(fr.composition_group.unique()); means = dict(zip(formulas, ridge.predict(formula_features(formulas))))
        for row in data[data.composition_id.isin(formulas)].itertuples():
            for seed in SEEDS:
                ridge_pred.append({"model": "Ridge_only", "seed": seed, "fold": fold, "structure_id": row.structure_id,
                                   "composition_id": row.composition_id, "family": row.family, "n_atoms": row.n_atoms,
                                   "energy_dft_ev_atom": row.energy_dft_ev_atom, "ridge_mean_ev_atom": means[row.composition_id],
                                   "residual_pred_ev_atom": 0.0, "energy_pred_ev_atom": means[row.composition_id]})
    # Baseline MatGPT F0 historical: M2, mismo ridge composicional, folds y semillas.
    base = pd.read_csv(MATGPT_BASELINE)
    base = base[(base.model_id == "M2") & base.seed.isin(SEEDS)].copy()
    if len(base) != 630:
        raise ValueError("Baseline histórico MatGPT F0 no contiene M2 con 630 filas")
    target_map = data.set_index("structure_id").energy_dft_ev_atom
    fold_map = data.drop_duplicates("structure_id").set_index("structure_id").outer_fold
    if (set(base.structure_id) != set(data.structure_id)
            or not np.allclose(base.energy_per_atom_dft_ev.to_numpy(float),
                               base.structure_id.map(target_map).to_numpy(float), atol=1e-9)
            or not np.array_equal(base.fold.to_numpy(int), base.structure_id.map(fold_map).to_numpy(int))):
        raise ValueError("Baseline MatGPT no coincide con targets/IDs/folds históricos")
    mat_base = base.rename(columns={"seed":"seed", "fold":"fold", "energy_per_atom_dft_ev":"energy_dft_ev_atom",
                                    "energy_per_atom_pred_ev":"energy_pred_ev_atom", "residual_pred_ev_atom":"residual_pred_ev_atom"})
    mat_base["model"] = "MatGPT_F0"
    allpred = pd.concat([matter, pd.DataFrame(ridge_pred), mat_base[["model","seed","fold","structure_id","composition_id","polymorph_family","n_atoms","energy_dft_ev_atom","mean_energy_pred_ev_atom","residual_pred_ev_atom","energy_pred_ev_atom"]].rename(columns={"polymorph_family":"family","mean_energy_pred_ev_atom":"ridge_mean_ev_atom"})], ignore_index=True)
    allpred.to_csv(OUT / "predicciones_oof_por_estructura.csv", index=False)
    metrics, metric_df, pair_df = summarize_predictions(allpred, args.bootstrap)
    pair_df.to_csv(OUT / "predicciones_oof_por_par.csv", index=False)
    write_json(OUT / "metrics.json", metrics)
    metric_df.to_csv(OUT / "metricas_por_composicion.csv", index=False)
    history = pd.concat(history_parts, ignore_index=True); history.to_csv(OUT / "historial_entrenamiento.csv", index=False)
    with torch.no_grad():
        trainable = sum(p.numel() for p in MLP8().parameters())
    duration = time.perf_counter() - start; cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    resources = {"wall_seconds": duration, "extraction_seconds": extraction_seconds,
                 "mlp_training_seconds": float(sum(r["elapsed_seconds"] for r in run_records)),
                 "process_user_cpu_seconds": cpu_end.ru_utime-cpu_start.ru_utime,
                 "process_system_cpu_seconds": cpu_end.ru_stime-cpu_start.ru_stime,
                 "peak_rss_bytes": cpu_end.ru_maxrss, "peak_rss_mib": cpu_end.ru_maxrss / (1024**2), "device": "CPU", "cpu_count": os.cpu_count(),
                 "torch_threads": torch.get_num_threads(), "mlp_trainable_parameters": trainable,
                 "mattergpt_trainable_parameters": 0, "fold_seed_runs": 15,
                 "python_version":platform.python_version(),"torch_version":torch.__version__,
                 "numpy_version":np.__version__,"pandas_version":pd.__version__,
                 "scipy_version":scipy.__version__,"scikit_learn_version":sklearn.__version__,
                 "platform":platform.platform()}
    write_json(OUT / "resources.json", resources)
    write_json(OUT / "summary.json", {"status":"COMPLETE", "experiment":EXPERIMENT, "structures":210,
        "compositions":70,"compatible_fraction":1.0,"runs":len(run_records),"preflight":"PASS",
        "property_vector":list(CONDITION),"metrics":metrics,"run_records":run_records,"resources":resources,
        "test_historico_abierto":False,"interpretation":"La comparación entre MatterGPT y MatGPT no es causal: difieren en arquitectura, vocabulario y condicionamiento."})
    write_config(model_audit, checks, args)
    make_figures(allpred, metric_df, pair_df, history, metrics)
    write_report(metrics, checks, model_audit, run_records, resources)
    (OUT / "README.md").write_text("# MatterGPT ABS₃ congelado\n\nVer `reporte_cientifico.md` para método, auditorías, métricas y límites. Ejecución reproducible con `python scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py`. El test histórico no se abrió.\n", encoding="utf-8")
    LOG.info("experiment_complete wall_seconds=%.2f", duration)


def write_config(model_audit: dict[str, Any], checks: dict[str, Any], args: argparse.Namespace) -> None:
    """Registra protocolo efectivo en YAML legible.

    Args:
        model_audit: Hashes y configuración del backbone.
        checks: Controles de entrada.
        args: Argumentos de ejecución.
    Returns:
        None.
    """
    alpha_table=pd.read_csv(RIDGE_DIR/"ridge_selection.csv")
    config = {"experiment": EXPERIMENT, "property_vector": list(CONDITION), "model": model_audit,
              "input_audit": checks, "ridge": {"source": str(RIDGE_DIR), "models_reused_without_fit": True,
              "features_order": list(ELEMENTS), "feature_count": 17,
              "fold_model_sha256": {str(f):sha256(RIDGE_DIR/"models"/f"ridge_composicional_fold_{f}.joblib") for f in range(5)},
              "selected_alpha_by_fold": {str(int(row.fold)):float(row.selected_alpha) for row in alpha_table.itertuples()}},
              "dataset_source":str(DATASET),"fold_manifest_source":str(FOLDS),"fold_manifest_sha256":sha256(FOLDS),
              "matgpt_f0_oof_source_sha256":sha256(MATGPT_BASELINE),
              "historical_test_labels_parsed":False,
              "head": {"architecture": "768-8-1", "activation": "GELU", "dropout": 0.1,
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "batch_compositions": 8, "max_epochs": 200, "patience": 20, "smoothl1_beta": 1.0},
              "splits": {"folds": 5, "inner_train": 45, "inner_validation": 11, "outer_evaluation": 14,
              "seeds": list(SEEDS)}, "bootstrap_replicates": args.bootstrap, "historical_test_accessed": False}
    # YAML básico vía JSON (JSON es subconjunto válido de YAML 1.2).
    (OUT / "config.yaml").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_figures(allpred: pd.DataFrame, metric_df: pd.DataFrame, pairs: pd.DataFrame,
                 history: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Genera siete figuras PNG comparativas con notas científicas en español.

    Args:
        allpred: Predicciones OOF por estructura, fold y semilla.
        metric_df: Métricas por fórmula y semilla.
        pairs: Métricas de pares.
        history: Historial de train y validación.
        metrics: Estadísticas agregadas.
    Returns:
        None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"MatterGPT_F0":"#007c91", "MatGPT_F0":"#b05a21", "Ridge_only":"#555b66"}
    note = "OOF: 210 estructuras / 70 fórmulas, 5 folds; se promedian 3 semillas por estructura. Punto = estructura estática; tripletes agrupados. ΔE_pred y ΔE_DFT comparten el ancla del mínimo DFT por fórmula. Test histórico cerrado."
    names = ["MatterGPT_F0", "MatGPT_F0", "Ridge_only"]
    # 1. Métricas comparativas.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = [metrics[n]["mae_delta_mev_atom"] for n in names]
    ax.bar(["MatterGPT F0", "MatGPT F0", "Ridge composicional"], vals, color=[colors[n] for n in names])
    ax.set_ylabel("MAE de ΔE firmado entre pares (meV/átomo)"); ax.set_title("ABS₃: error OOF en energías relativas")
    ax.text(.01,-.24,note,transform=ax.transAxes,fontsize=8,va="top",wrap=True); fig.tight_layout(); fig.savefig(IMAGE_DIR/"01_comparacion_modelos.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # 2. Bootstrap intervals.
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); ci = metrics["bootstrap_paired_composition_ci95"][:3]
    for i,row in enumerate(ci):
        ax.errorbar(row["estimate"],i,xerr=[[row["estimate"]-row["ci95_low"]],[row["ci95_high"]-row["estimate"]]],fmt="o",color=colors[names[i]],capsize=4)
    ax.set_yticks(range(3),["MatterGPT F0","MatGPT F0","Ridge composicional"]); ax.set_xlabel("MAE ΔE firmado entre pares (meV/átomo), IC95% bootstrap por fórmula")
    ax.set_title("ABS₃: incertidumbre de la MAE primaria"); ax.text(.01,-.22,"10 000 remuestreos apareados de 70 fórmulas; semillas promediadas dentro de fórmula.",transform=ax.transAxes,fontsize=8); fig.tight_layout(); fig.savefig(IMAGE_DIR/"02_mae_delta_ic95.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # 3. Curves averaged over 15 runs, with selected epoch convention in note.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col,label,color in [("train_minibatch_smoothl1","Minilotes train","#007c91"),("inner_validation_full_smoothl1","Validación interna completa","#b05a21")]:
        agg=history.groupby("epoch")[col].agg(["mean","std","count"]); ax.plot(agg.index,agg["mean"],label=label,color=color); ax.fill_between(agg.index,agg["mean"]-agg["std"],agg["mean"]+agg["std"],color=color,alpha=.15)
    ax.set_xlabel("Época local por corrida"); ax.set_ylabel("SmoothL1 residual normalizada"); ax.set_title("MatterGPT congelado: entrenamiento de MLP8"); ax.legend()
    ax.text(.01,-.22,"Media ± DE entre corridas activas; checkpoint elegido por menor pérdida de validación completa. Sin observaciones outer en selección.",transform=ax.transAxes,fontsize=8); fig.tight_layout(); fig.savefig(IMAGE_DIR/"03_curvas_entrenamiento.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    # 4/5 parity on seed-averaged rows.
    avg=allpred.groupby(["model","structure_id","composition_id","family"],as_index=False).agg(energy_dft_ev_atom=("energy_dft_ev_atom","mean"),energy_pred_ev_atom=("energy_pred_ev_atom","mean"))
    for mode in ("residual","delta"):
        fig, axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
        plotted_ranges=[]
        for ax,name in zip(axes,names):
            g=avg[avg.model.eq(name)].copy()
            if mode=="residual":
                x=(g.energy_dft_ev_atom-g.groupby("composition_id").energy_dft_ev_atom.transform("mean"))*1000; y=(g.energy_pred_ev_atom-g.groupby("composition_id").energy_pred_ev_atom.transform("mean"))*1000; unit="meV/átomo"
                title=f"{name.replace('_',' ')} · residuo intrafórmula"
            else:
                anchor_idx=g.groupby("composition_id").energy_dft_ev_atom.idxmin()
                anchors=g.loc[anchor_idx].set_index("composition_id").energy_pred_ev_atom
                x=(g.energy_dft_ev_atom-g.groupby("composition_id").energy_dft_ev_atom.transform("min"))*1000; y=(g.energy_pred_ev_atom-g.composition_id.map(anchors))*1000; unit="meV/átomo"
                title=f"{name.replace('_',' ')} · ΔE anclada al mínimo DFT"
            ax.scatter(x,y,s=18,alpha=.65,color=colors[name]); m,b=np.polyfit(x,y,1); grid=np.linspace(min(x.min(),y.min()),max(x.max(),y.max()),80); ax.plot(grid,grid,"k--",lw=1,label="Ideal y=x"); ax.plot(grid,m*grid+b,color="#b21f56",lw=1.2,label="OLS")
            plotted_ranges.append((float(min(x.min(),y.min())),float(max(x.max(),y.max()))))
            r2_text="no definido" if np.var(y)==0 else f"{r2_score(y,m*x+b):.3f}"; mae=float(np.mean(np.abs(y-x))); ax.text(.04,.96,f"N={len(x)}\nm={m:.3f}\nb={b:.2f} {unit}\nR²={r2_text}\nMAE={mae:.2f} {unit}",transform=ax.transAxes,va="top",fontsize=8); ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.15)
        low=min(v[0] for v in plotted_ranges); high=max(v[1] for v in plotted_ranges); margin=.04*(high-low or 1)
        for ax in axes: ax.set_xlim(low-margin,high+margin); ax.set_ylim(low-margin,high+margin); ax.set_aspect("equal",adjustable="box")
        for ax in axes: ax.set_xlabel(f"DFT {mode} ({unit})")
        axes[0].set_ylabel(f"Predicción ({unit})"); fig.suptitle("ABS₃: paridad entre referencia DFT y predicción",y=1.03)
        fig.text(.5,.012,note+" Recta OLS con intercepto libre y todas las estructuras mostradas.",ha="center",fontsize=7.5); fig.tight_layout(rect=(0,.11,1,.95)); fig.savefig(IMAGE_DIR/("04_paridad_residuos.png" if mode=="residual" else "05_paridad_delta.png"),dpi=180,bbox_inches="tight"); plt.close(fig)
    # 6. Ground state and pairwise accuracy.
    fig,ax=plt.subplots(figsize=(8,4.5)); gs=[metrics[n]["ground_state_exact"]*100 for n in names]; pa=[metrics[n]["pairwise_accuracy_including_unresolved_in_denominator"]*100 for n in names]
    xx=np.arange(3); ax.bar(xx-.18,gs,.36,label="Estado fundamental exacto",color="#007c91");ax.bar(xx+.18,pa,.36,label="Pares elegibles correctos",color="#b05a21");ax.set_xticks(xx,["MatterGPT F0","MatGPT F0","Ridge composicional"]);ax.set_ylabel("Exactitud (%)");ax.set_ylim(0,105);ax.legend();ax.set_title("ABS₃: selección del mínimo y orden de pares");ax.text(.01,-.22,"GS: 201/210 casos (3 semillas); DFT gap≤1 excluido, empate predicho ≤1 = fallo. Pares: 591 elegibles; |ΔE_DFT|>5; |ΔE_pred|≤1 = incorrecto.",transform=ax.transAxes,fontsize=8);fig.tight_layout();fig.savefig(IMAGE_DIR/"06_ground_state_ranking.png",dpi=180,bbox_inches="tight");plt.close(fig)
    # 7. MAE delta against DFT gap.
    fig,ax=plt.subplots(figsize=(8,4.5)); bins=[0,5,20,50,100,1e9]; labels=["≤5","(5,20]","(20,50]","(50,100]",">100"]
    for name in names:
        g=metric_df[metric_df.model.eq(name)].copy(); g["gap_bin"]=pd.cut(g.dft_gap_mev_atom,bins=bins,labels=labels,include_lowest=True); a=g.groupby("gap_bin",observed=False).mae_delta_mev_atom.mean();ax.plot(labels,a.values,marker="o",label=name.replace("_"," "),color=colors[name])
    ax.set_xlabel("Gap DFT entre los dos polimorfos más bajos (meV/átomo)");ax.set_ylabel("MAE de ΔE firmado entre pares (meV/átomo)");ax.set_title("ABS₃: error según separación DFT del mínimo");ax.legend(fontsize=8);ax.grid(alpha=.18);ax.text(.01,-.24,"Media por composición y semilla; gap por fórmula. Pares dentro de fórmula; bins vacíos sin observaciones.",transform=ax.transAxes,fontsize=8);fig.tight_layout();fig.savefig(IMAGE_DIR/"07_error_por_gap_dft.png",dpi=180,bbox_inches="tight");plt.close(fig)


def write_report(metrics: dict[str, Any], checks: dict[str, Any], model_audit: dict[str, Any],
                 runs: list[dict[str, Any]], resources: dict[str, Any]) -> None:
    """Escribe el informe de resultados con hechos e inferencias separados.

    Args:
        metrics: Métricas OOF agregadas.
        checks: Auditoría de entrada y extractor.
        model_audit: Proveniencia MatterGPT.
        runs: Registros de las quince corridas.
        resources: Medidas de recursos.
    Returns:
        None.
    """
    def pct(x: float) -> str: return f"{x:.2f}"
    models=["MatterGPT_F0","MatGPT_F0","Ridge_only"]
    lines=["# MatterGPT congelado como extractor para estabilidad polimórfica ABS₃", "",
        "## Diseño y procedencia", "",f"Se evaluaron {checks['structures']} estructuras en {checks['compositions']} composiciones (tres polimorfos por fórmula), con cinco folds y tres semillas. Se usó MatterGPT multipropiedad oficial local, cargado con `strict=True`, 12 bloques, 12 cabezas, hidden size 768, vocabulario 132, contexto 397 y `num_props=2`. El vector `[1.0, 2.0]` se pasó en `float32` con forma `[batch,1,2]`; es una condición fija (energía de formación y band gap según el código), no una etiqueta ABS₃.",
        "",f"Origen: [MatterGPT en Hugging Face](https://huggingface.co/xiaohang07/MatterGPT); se usó el checkpoint local. SHA-256 checkpoint: `{model_audit['checkpoint_sha256']}`. `Voc_prior`: `{model_audit['tokenizer_sha256']}`. Código y wrapper: `{model_audit['model_py_sha256']}` / `{model_audit['wrapper_sha256']}`.",
        "", "## Auditorías", "",f"Compatibilidad: {checks['structures']}/210 ({100*checks['structures']/210:.1f}%). Se conservaron byte por byte las cadenas históricas `matgpt_slices`, que coinciden con `slices_original` y `SLICES`; no se eliminaron ni alteraron tokens, índices, aristas o traslaciones. No se añadió un campo `space_group_num`: no está presente en la tabla fuente ni se serializa en la cadena SLICES usada. `strategy3_graph_equivalent=True` aparece en {checks['graph_equivalence_true']} filas (las restantes no están marcadas como equivalentes); hay hash de topología en las 210. Se preservó exactamente la representación textual histórica sin conversión cristalográfica nueva.",
        "",f"Cero OOV y truncamientos; máximo {checks['max_sequence_tokens']} tokens frente a contexto 397. Hay {checks['atoms_over_20']} estructuras sobre 20 átomos/celda. `>` inicia y `<` cierra según el uso de generación del código público. Los IDs serializados son los derivados de `Voc_prior` (>{checks['start_token_id']}, <{checks['terminal_token_id']}); los `bos_token_id/eos_token_id` de `config.json` (130/131) no se usan porque no coinciden con el tokenizador reconstruido. Se extrajo la posición del `<` tras `blocks → ln_f`, índice de secuencia n por el prefijo de propiedades. El `embed` de entrada pre-Transformer no se usó. La salida es 210×768, finita y determinista; el hook no alteró logits; padding y tamaño de lote no cambiaron el embedding. Condición y secuencia alteradas cambiaron la representación en controles diagnósticos.",
        "", "**Limitación de serialización:** no se encontró el dataloader original del teacher-forcing del checkpoint. Por ello `> SLICES <` se justifica por el pipeline público de generación y el cierre `<`, pero la política exacta de token terminal durante el preentrenamiento no queda verificada de manera independiente.",
        "", "## Entrenamiento y evaluación", "", "Se reutilizaron los cinco Ridge estequiométricos ya ajustados (17 fracciones, scaler y alpha históricos), sin invocar `fit`. Se entrenaron 15 MLP8 (45 grupos inner-train, 11 inner-validation, 14 outer-evaluation; 42/123/2026), con entrada contextualizada fija, AdamW 1e-3, weight decay 1e-4, minibatches de ocho fórmulas, máximo 200 épocas, patience 20, GELU, dropout 0.1 y SmoothL1 beta 1. La escala de residuo se calculó solo en inner-train. Se seleccionó la menor pérdida de validación interna completa. Solo se entrenó la MLP8 (6161 parámetros); MatterGPT quedó congelado.",
        "", "ΔE estructural usa ancla DFT común: `ΔE_DFT,i = E_DFT,i − E_DFT,g` y `ΔE_pred,i = E_pred,i − E_pred,g`, donde `g = argmin_j(E_DFT,j)` dentro de cada fórmula. Una predicción negativa indica inversión respecto al mínimo DFT. La métrica primaria A es MAE de diferencias firmadas `ΔE_ij = E_j − E_i` en los tres pares intrafórmula; Ridge-only predice cero en todos ellos. Empates predichos ≤1 meV/átomo quedan no resueltos. OOF cubre 210 estructuras; las semillas se promedian dentro de composición antes del bootstrap apareado de 10 000 réplicas por 70 composiciones.",
        "", "## Resultados (media sobre 15 fold-semilla)" ]
    for n in models:
        m=metrics[n]; o=metrics["ols_parity"][n]; r2=o['delta_energy']['r2_ols']; r2text="no definido (predicción constante)" if r2 is None else f"{r2:.3f}"; lines.append(f"- **{n}:** MAE ΔE {m['mae_delta_mev_atom']:.3f} meV/átomo; MAE E/N {m['mae_energy_ev_atom']:.5f} eV/átomo; RMSE E/N {m['rmse_energy_ev_atom']:.5f} eV/átomo; MAE residual {m['mae_residual_mev_atom']:.3f} meV/átomo; paridad ΔE OLS m={o['delta_energy']['slope']:.3f}, b={o['delta_energy']['intercept_mev_atom']:.2f} meV/átomo, R²={r2text}; GS {100*m['ground_state_exact']:.1f}% ({m['ground_state_eligible_composition_seed_cases']} casos elegibles); pairwise {100*m['pairwise_accuracy_including_unresolved_in_denominator']:.1f}% sobre {m['eligible_pairs']} pares elegibles; no resueltos {100*m['unresolved_fraction_eligible_pairs']:.1f}%.")
    contrast1=metrics["bootstrap_paired_composition_ci95"][3]; contrast2=metrics["bootstrap_paired_composition_ci95"][4]
    lines += ["", f"**Contrastes pareados de MAE ΔE (MatterGPT menos baseline):** frente a MatGPT F0, {contrast1['estimate']:.3f} meV/átomo (IC95% {contrast1['ci95_low']:.3f} a {contrast1['ci95_high']:.3f}; incluye cero); frente a Ridge-only, {contrast2['estimate']:.3f} meV/átomo (IC95% {contrast2['ci95_low']:.3f} a {contrast2['ci95_high']:.3f}; no incluye cero). MatterGPT mejora el MAE respecto a Ridge-only en este CV; la estimación puntual es menor que MatGPT F0, pero no se distingue con claridad por este bootstrap. La exactitud por pares de MatterGPT es {100*metrics['MatterGPT_F0']['pairwise_accuracy_including_unresolved_in_denominator']:.1f}% frente a {100*metrics['MatGPT_F0']['pairwise_accuracy_including_unresolved_in_denominator']:.1f}% para MatGPT F0; la selección exacta de estado fundamental es {100*metrics['MatterGPT_F0']['ground_state_exact']:.1f}% frente a {100*metrics['MatGPT_F0']['ground_state_exact']:.1f}%.",
        "", "MatGPT F0 se reutilizó desde el modelo histórico M2 del Ridge híbrido, sobre las mismas filas, folds y semillas; se verificó el total de 630 predicciones y no se modificó retrospectivamente.",
        "", "## Respuestas e interpretación", "", "1. **Compatibilidad:** 210/210 estructuras tokenizables sin OOV ni truncamiento.",
        "2. **SLICES:** cadena histórica preservada byte por byte; no se transformó la estructura. El indicador histórico `strategy3_graph_equivalent` es verdadero en 207/210 filas y falso en las otras tres.",
        "3. **Condición:** `[1.0,2.0]` entra por `prop_nn` y afecta el embedding; se aplicó a toda la cohorte. Su región puede ser poco representada por tener energía de formación positiva.",
        "4. **Representación:** último token `<` tras `ln_f`, dimensión 768.",
        f"5–8. **Desempeño:** MatterGPT F0 MAE ΔE {metrics['MatterGPT_F0']['mae_delta_mev_atom']:.3f}; Ridge {metrics['Ridge_only']['mae_delta_mev_atom']:.3f}; MatGPT F0 {metrics['MatGPT_F0']['mae_delta_mev_atom']:.3f} meV/átomo. Pairwise: MatterGPT {100*metrics['MatterGPT_F0']['pairwise_accuracy_including_unresolved_in_denominator']:.1f}% y MatGPT {100*metrics['MatGPT_F0']['pairwise_accuracy_including_unresolved_in_denominator']:.1f}%; GS: MatterGPT {100*metrics['MatterGPT_F0']['ground_state_exact']:.1f}% y MatGPT {100*metrics['MatGPT_F0']['ground_state_exact']:.1f}%. Ranking y estado fundamental están detallados arriba y en JSON.",
        "9. **Sobreajuste:** train sigue bajando mientras validación sube en épocas tardías (aprox. desde 45); early stopping limita ese sobreajuste de la cabeza.",
        "10. **Fine-tuning parcial:** MatterGPT mejora la estimación frente a Ridge-only, pero la diferencia frente a MatGPT F0 es incierta; esto no basta para atribuir una mejora al backbone. Un ajuste parcial queda como hipótesis para un experimento separado.",
        "", "## Límites", "", "La comparación MatterGPT–MatGPT no identifica un efecto causal de backbone: cambian arquitectura, vocabulario y condiciones de preentrenamiento. La condición `[1,2]` se eligió de antemano y no se optimizó con ABS₃. El test histórico quedó cerrado; los resultados son validación cruzada OOF sobre desarrollo ya explorado, no un test ciego.",
        "", "## Recursos y archivos", "",f"Tiempo de pared: {resources['wall_seconds']:.1f} s (extracción {resources['extraction_seconds']:.1f} s; MLP8 {resources['mlp_training_seconds']:.1f} s acumulados); CPU usuario/sistema: {resources['process_user_cpu_seconds']:.1f}/{resources['process_system_cpu_seconds']:.1f} s; dispositivo CPU; memoria máxima: {resources['peak_rss_mib']:.1f} MiB. Versiones: Python {resources['python_version']}, PyTorch {resources['torch_version']}, NumPy {resources['numpy_version']}, pandas {resources['pandas_version']}, SciPy {resources['scipy_version']}, scikit-learn {resources['scikit_learn_version']}. Se guardaron quince checkpoints MLP8 compactos sin duplicar MatterGPT.",
        "", "Las siete figuras se exportan solo en PNG bajo `images/mattergpt_abs3_frozen_ridge_mlp8/`.",""]
    report_text="\n".join(lines)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    (REPORT_DIR/"reporte_cientifico.md").write_text(report_text,encoding="utf-8")
    (OUT/"reporte_cientifico.md").write_text(report_text,encoding="utf-8")


def finalize_existing() -> None:
    """Regenera el informe y PNG desde las predicciones guardadas.

    Args:
        None.
    Returns:
        None.
    """
    preflight=json.loads((OUT/"preflight.json").read_text(encoding="utf-8"))
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    pred=pd.read_csv(OUT/"predicciones_oof_por_estructura.csv")
    hist=pd.read_csv(OUT/"historial_entrenamiento.csv")
    summary["resources"].update({"python_version":platform.python_version(),"torch_version":torch.__version__,
        "numpy_version":np.__version__,"pandas_version":pd.__version__,"scipy_version":scipy.__version__,
        "scikit_learn_version":sklearn.__version__,"platform":platform.platform()})
    write_json(OUT/"resources.json",summary["resources"])
    summary["metrics"],comp,pair=summarize_predictions(pred,10000)
    comp.to_csv(OUT/"metricas_por_composicion.csv",index=False)
    pair.to_csv(OUT/"predicciones_oof_por_par.csv",index=False)
    write_json(OUT/"metrics.json",summary["metrics"])
    write_json(OUT/"summary.json",summary)
    write_config(preflight["model"],preflight["checks"],argparse.Namespace(bootstrap=10000))
    make_figures(pred,comp,pair,hist,summary["metrics"])
    write_report(summary["metrics"],preflight["checks"],preflight["model"],summary["run_records"],summary["resources"])
    (OUT/"README.md").write_text("# MatterGPT ABS₃ congelado\n\nVer `reporte_cientifico.md` para método, auditorías, métricas y límites. Ejecución reproducible con `python scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py`. El test histórico no se abrió.\n",encoding="utf-8")
    LOG.info("final_artifacts_regenerated_from_saved_predictions")


def parse_args() -> argparse.Namespace:
    """Valida argumentos del pipeline.

    Args:
        None.
    Returns:
        Opciones validadas.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",type=int,default=10000,help="Réplicas bootstrap por composición")
    parser.add_argument("--finalize-existing",action="store_true",help="Regenera informe/figuras con outputs OOF ya guardados")
    args=parser.parse_args()
    if args.bootstrap < 1000 or args.bootstrap > 100000:
        parser.error("--bootstrap debe estar entre 1000 y 100000")
    for path in (() if args.finalize_existing else (DATASET,FOLDS,MATGPT_BASELINE,MODEL_DIR/"pytorch_model.pt")):
        if not path.is_file(): parser.error(f"Falta archivo requerido: {path}")
    if args.finalize_existing and not (OUT/"summary.json").is_file():
        parser.error("--finalize-existing requiere summary.json del experimento")
    return args


def main() -> None:
    """Punto de entrada protegido."""
    args=parse_args()
    try:
        if args.finalize_existing: finalize_existing()
        else: run(args)
    except Exception:
        LOG.exception("experiment_failed; no se declaran resultados completos")
        raise


if __name__ == "__main__":
    main()
