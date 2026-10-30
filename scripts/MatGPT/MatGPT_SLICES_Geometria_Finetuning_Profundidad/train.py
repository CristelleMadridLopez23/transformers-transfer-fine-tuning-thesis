"""Fine-tuning apareado MatGPT con SLICES y siete descriptores de celda.

Entrena únicamente las condiciones geométricas F2/F4/F8. Reutiliza F0
geométrico validado y controles SLICES-only compatibles; Ridge composicional,
dataset, folds, tokenizador y checkpoint base permanecen inalterados.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
REPORT = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
IMAGES = ROOT / "images/MatGPT_SLICES_Geometria_Finetuning_Profundidad"
SCRIPT = Path(__file__).resolve().parent
DEPTH_SCRIPT = ROOT / "scripts/MatGPT/MatGPT_SLICES_Finetuning_Profundidad"
GEOMETRY_SOURCE = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv"
GEOMETRY_MANIFEST = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.manifest.json"
GEOMETRY_RESULTS = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/results"
DEPTH_EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Finetuning_Profundidad"
GEOMETRY_COLUMNS = ("a_reduced_shape", "b_reduced_shape", "c_reduced_shape",
                    "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg",
                    "volume_per_atom_ang3")
SEEDS = (42, 123, 2026)
DEPTHS = {"F2": (10, 11), "F4": (8, 9, 10, 11),
          "F8": (4, 5, 6, 7, 8, 9, 10, 11)}
BASE_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
LOG = logging.getLogger("matgpt_experiment_d")


def sha256(path: Path) -> str:
    """Calcula SHA-256 sin cargar el archivo completo a memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Guarda el estado consolidado mediante reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def rss_bytes() -> int:
    """Devuelve el máximo RSS observado por el proceso, en bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class GeometryMLP8(nn.Module):
    """MLP residual fijo con entrada concatenada 768+7 y salida centrada."""

    def __init__(self) -> None:
        """Crea la topología 775→8→1, con GELU y dropout 0.1."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(775, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, joined_features: Tensor) -> Tensor:
        """Predice y centra los residuos de cada triplete de polimorfos."""
        raw = self.network(joined_features).squeeze(-1)
        return raw - raw.mean(dim=-1, keepdim=True)


def init_matched_head(seed: int, base_head: nn.Module) -> tuple[GeometryMLP8, str, Tensor]:
    """Copia la cabeza SLICES-only inicial y deja en cero los 56 pesos geométricos.

    Restaura el estado aleatorio inmediatamente posterior a crear la cabeza de
    768 entradas para que dropout y bloques Transformer sigan la secuencia RNG
    de la corrida SLICES-only histórica con la misma semilla.
    """
    torch.manual_seed(seed)
    base_head = base_head.__class__()
    init_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                      for p in base_head.parameters())).hexdigest()
    rng_after_base = torch.get_rng_state()
    geo_head = GeometryMLP8()
    with torch.no_grad():
        geo_head.network[0].weight[:, :768].copy_(base_head.network[0].weight)
        geo_head.network[0].weight[:, 768:].zero_()
        geo_head.network[0].bias.copy_(base_head.network[0].bias)
        geo_head.network[3].load_state_dict(base_head.network[3].state_dict(), strict=True)
    torch.set_rng_state(rng_after_base)
    if sum(p.numel() for p in geo_head.parameters()) != 6217:
        raise ValueError("La MLP geométrica no tiene 6,217 parámetros")
    return geo_head, init_hash, rng_after_base


def import_depth_trainer():
    """Importa funciones auditadas del entrenador de profundidad existente."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("scripts.MatGPT.MatGPT_SLICES_Finetuning_Profundidad.train")


def load_geometry(depth: Any, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Carga geometrías reales, verifica manifiesto, IDs, N y valores finitos."""
    if not GEOMETRY_SOURCE.is_file() or not GEOMETRY_MANIFEST.is_file():
        raise FileNotFoundError("Falta geometry_cell_development.csv o su manifiesto")
    manifest = json.loads(GEOMETRY_MANIFEST.read_text(encoding="utf-8"))
    actual = sha256(GEOMETRY_SOURCE)
    if manifest.get("status") != "PASS" or manifest.get("geometry_csv_sha256") != actual:
        raise ValueError("Hash/manifiesto de las geometrías no coincide")
    if manifest.get("entries") != 210 or manifest.get("source_counts") != {"gga-static_endpoint_cleaned": 210}:
        raise ValueError("La procedencia geométrica no coincide con el conjunto auditado")
    data = pd.read_csv(GEOMETRY_SOURCE)
    expected = set(frame.structure_id.astype(str))
    if len(data) != 210 or data.structure_id.nunique() != 210 or set(data.structure_id.astype(str)) != expected:
        raise ValueError("Geometría no se alinea exactamente con los 210 IDs de desarrollo")
    if data.geometry_source.isna().any() or set(data.geometry_source) != {"gga-static_endpoint_cleaned"}:
        raise ValueError("Procedencia geométrica ausente o heterogénea")
    if data.source_path.isna().any() or data.source_sha256.isna().any():
        raise ValueError("Falta evidencia de archivo fuente geométrico")
    source_jsonl=ROOT/"experiments/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl"
    source_hash=sha256(source_jsonl)
    if set(data.source_sha256.astype(str))!={source_hash}:
        raise ValueError("Hash del JSONL estructural no coincide con los registros geométricos")
    structures=[json.loads(line) for line in source_jsonl.read_text(encoding="utf-8").splitlines()]
    if len(structures)!=210 or {str(row["structure_id"]) for row in structures}!=set(frame.structure_id.astype(str)):
        raise ValueError("JSONL de estructuras no alinea 1:1 con los 210 IDs de desarrollo")
    for row in data.itertuples():
        try: line_number=int(str(row.source_path).rsplit(":line:",1)[1])
        except (IndexError,ValueError) as exc: raise ValueError(f"Referencia de línea inválida: {row.source_path}") from exc
        if not 1<=line_number<=len(structures) or str(structures[line_number-1]["structure_id"])!=str(row.structure_id):
            raise ValueError(f"ID físico no coincide con JSONL en línea {line_number}: {row.structure_id}")
    lookup = data.set_index("structure_id").loc[frame.structure_id]
    if not np.array_equal(lookup.n_atoms.astype(int).to_numpy(), frame.n_atoms.astype(int).to_numpy()):
        raise ValueError("Número de átomos de geometría no coincide con dataset")
    if not set(GEOMETRY_COLUMNS).issubset(data.columns):
        raise ValueError("Faltan columnas de geometría: " + str(set(GEOMETRY_COLUMNS)-set(data.columns)))
    values = lookup.loc[:, list(GEOMETRY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, :3] <= 0).any() or (values[:, 6] <= 0).any():
        raise ValueError("Descriptores geométricos no finitos/no positivos")
    if (values[:, 3:6] <= 0).any() or (values[:, 3:6] >= 180).any():
        raise ValueError("Ángulos cristalográficos fuera del intervalo válido")
    return data, values.astype(np.float32), manifest


def preflight_sources(depth: Any) -> dict[str, Any]:
    """Comprueba geometría, predicciones base, pesos, folds y Ridge sin entrenar."""
    frame, folds, model, stoi, context = depth.load_sources()
    geo_frame, geometry, geo_manifest = load_geometry(depth, frame)
    source_hash=sha256(ROOT/"experiments/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl")
    original_manifest = json.loads((GEOMETRY_RESULTS / "manifest_experimento.json").read_text())
    final_geometry_manifest = json.loads((ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/final_manifest.json").read_text())
    geo_predictions = pd.read_csv(GEOMETRY_RESULTS / "predicciones_oof_por_estructura.csv")
    depth_predictions = pd.read_csv(DEPTH_EXPERIMENT / "predicciones_oof_por_estructura.csv")
    depth_summary = json.loads((DEPTH_EXPERIMENT / "summary.json").read_text())
    resources = json.loads((DEPTH_EXPERIMENT / "resources.json").read_text())
    frozen_training=pd.read_csv(GEOMETRY_RESULTS/"seleccion_y_entrenamiento_por_fold.csv")
    if (original_manifest.get("historical_test_accessed") is not False or
            final_geometry_manifest.get("historical_test_accessed") is not False or
            final_geometry_manifest.get("ridge_retrained") is not False):
        raise ValueError("Manifiesto de control geométrico indica acceso test o reentrenamiento Ridge")
    if sha256(GEOMETRY_RESULTS / "predicciones_oof_por_estructura.csv") != original_manifest.get("output_prediction_sha256"):
        raise ValueError("Hash de predicciones geométricas F0 no coincide con manifiesto")
    if len(geo_predictions) != 1260 or set(geo_predictions.condition) != {"slices_only", "slices_plus_cell_geometry"}:
        raise ValueError("Predicciones geométricas Frozen incompletas")
    if len(frozen_training)!=30 or set(frozen_training.condition)!={"slices_only","slices_plus_cell_geometry"}:
        raise ValueError("No hay 15 pares de corridas Frozen SLICES/geometría verificables")
    if not (frozen_training.groupby("condition").size()==15).all():
        raise ValueError("Conteo de corridas Frozen incompatible")
    if set(frozen_training.loc[frozen_training.condition=="slices_only","parameters"])!={6161} or \
            set(frozen_training.loc[frozen_training.condition=="slices_plus_cell_geometry","parameters"])!={6217}:
        raise ValueError("Las dimensiones de cabezas del baseline congelado no corresponden")
    if len(depth_predictions) != 2520 or set(depth_predictions.condition) != {"F0", "F2", "F4", "F8"}:
        raise ValueError("Predicciones control de profundidad incompletas")
    if depth_summary.get("completed_finetuning_runs") != 45 or depth_summary.get("status") != "COMPLETE":
        raise ValueError("No están completas las 45 corridas históricas de profundidad")
    if depth_summary.get("ridge_refit") is not False or depth_summary.get("checkpoint_sha256") != depth.BASE_SHA:
        raise ValueError("El control de profundidad no coincide en Ridge/checkpoint")
    for condition in ("F2", "F4", "F8"):
        for fold in range(5):
            for seed in SEEDS:
                run = resources.get("runs", {}).get(f"{condition.lower()}_fold{fold}_seed{seed}", {})
                if run.get("status") != "COMPLETE":
                    raise ValueError(f"Falta control SLICES completo {condition}/fold{fold}/seed{seed}")
    # La media Ridge de composición es invariante entre profundidad y semillas.
    spread = depth_predictions.groupby(["fold", "seed", "structure_id"]).ridge_mean_ev_atom.agg(lambda x: x.max()-x.min())
    if spread.max() > 1e-10:
        raise ValueError("Ridge composicional varía entre condiciones de control")
    # Comprueba entradas, targets, unidades e IDs de desarrollo antes de ajustar.
    expected_ids = set(frame.structure_id.astype(str))
    for name, table in (("F0 geometría", geo_predictions), ("profundidad", depth_predictions)):
        if set(table.structure_id.astype(str)) != expected_ids:
            raise ValueError(f"IDs de {name} no coinciden con desarrollo")
        if table.duplicated(["condition", "fold", "seed", "structure_id"]).any():
            raise ValueError(f"Filas duplicadas en {name}")
        if not np.allclose(table.energy_total_dft_ev / table.n_atoms, table.energy_per_atom_dft_ev,
                           atol=1e-8, rtol=0):
            raise ValueError(f"Unidades E_total/N inconsistentes en {name}")
    # Verify exact initialization hashes reported by the completed depth controls.
    initial_hashes = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        head = depth.MLP8()
        initial_hashes[str(seed)] = hashlib.sha256(b"".join(
            p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
        expected_hashes = {resources["runs"][f"{c.lower()}_fold0_seed{seed}"].get("head_initial_sha256")
                           for c in ("F2", "F4", "F8")}
        if expected_hashes != {initial_hashes[str(seed)]}:
            raise ValueError(f"Inicialización MLP8 no coincide con controles de semilla {seed}")
    del model
    ridge_hashes = depth_summary.get("ridge_sha256", {})
    actual_ridge_hashes={str(i):sha256(depth.RIDGE_DIR/f"ridge_composicional_fold_{i}.joblib") for i in range(5)}
    if len(ridge_hashes) != 5 or ridge_hashes!=actual_ridge_hashes:
        raise ValueError("No se verificaron los cinco Ridge composicionales")
    return {"status": "PASS", "structures": len(frame), "compositions": frame.formula.nunique(),
            "geometry_entries": len(geo_frame), "geometry_source": geo_manifest["source_counts"],
            "geometry_sha256": sha256(GEOMETRY_SOURCE), "geometry_manifest_sha256": sha256(GEOMETRY_MANIFEST),
            "feature_columns": list(GEOMETRY_COLUMNS), "depth_oof_sha256": sha256(DEPTH_EXPERIMENT / "predicciones_oof_por_estructura.csv"),
            "geometry_frozen_oof_sha256": sha256(GEOMETRY_RESULTS / "predicciones_oof_por_estructura.csv"),
            "geometry_final_manifest_sha256": sha256(ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/final_manifest.json"),
            "depth_controls_complete": 45, "geometry_frozen_controls_reused": 15,
            "checkpoint_sha256": depth.BASE_SHA, "folds_sha256": depth.FOLD_SHA,
            "tokenizer_sha256": depth.TOKEN_SHA, "ridge_hashes": actual_ridge_hashes,
            "geometry_jsonl_sha256": source_hash,"frozen_geometry_training_rows":len(frozen_training),
            "mlp_initial_hashes": initial_hashes, "max_ridge_spread_ev_atom": float(spread.max()),
            "feature_cache_numeric_audit": final_geometry_manifest.get("feature_cache_policy"),
            "historical_test_accessed": False}


def extract_triplet(depth: Any, frame: pd.DataFrame, geo_by_id: dict[str, np.ndarray],
                    ids: list[str], model: Any, stoi: dict[str, int], context: int,
                    scaler: StandardScaler, device: torch.device, grad: bool) -> Tensor:
    """Extrae FINAL_END y concatena descriptores escalados en orden fijo."""
    seq = [depth.encode_all(frame[frame.structure_id.isin(ids)], stoi, context)[sid] for sid in ids]
    embeddings = depth.final_end(model, seq, stoi["<pad>"], device)
    g = np.stack([geo_by_id[sid] for sid in ids])
    scaled = scaler.transform(g).astype(np.float32)
    geometry = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    joined = torch.cat((embeddings, geometry), dim=-1)
    if joined.shape != (3, 775):
        raise ValueError(f"Vector de triplete inesperado: {tuple(joined.shape)}")
    return joined if grad else joined.detach()


def fit_one(depth: Any, frame: pd.DataFrame, folds: pd.DataFrame, model: Any,
            stoi: dict[str, int], context: int, geometry_values: np.ndarray,
            condition: str, fold: int, seed: int, device: torch.device,
            smoke: bool = False) -> dict[str, Any]:
    """Ajusta una condición geométrica con selección solo inner-validation."""
    run = f"{condition.lower()}_fold{fold}_seed{seed}" + ("_smoke" if smoke else "")
    run_dir = EXPERIMENT / "checkpoints" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "complete.flag").exists() and not smoke:
        LOG.info("%s", {"event": "skip_complete", "run": run})
        return {"status": "COMPLETE", "run": run}
    base_path = depth.SOURCE / "mat-uncond/weights/transformer_property_0.pth"
    model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True), strict=True)
    part = folds[folds.outer_fold == fold]
    train_ids = set(part.loc[part.inner_role == "inner_train", "structure_id"].astype(str))
    val_ids = set(part.loc[part.inner_role == "inner_validation", "structure_id"].astype(str))
    train_groups = depth.groups_for(frame, train_ids)
    val_groups = depth.groups_for(frame, val_ids)
    if smoke:
        train_groups, val_groups = train_groups[:2], val_groups[:1]
    lookup = frame.set_index("structure_id")
    geo_lookup = {str(sid): geometry_values[i] for i, sid in enumerate(frame.structure_id.astype(str))}
    scaler_fit_ids = [sid for _, group in depth.groups_for(frame, train_ids) for sid in group]
    scaler = StandardScaler().fit(np.stack([geo_lookup[sid] for sid in scaler_fit_ids]))
    target_train = depth.target_matrix(frame, train_groups)
    target_val = depth.target_matrix(frame, val_groups)
    scale = float(target_train.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Escala residual de inner-train inválida")
    block_ids = depth.CONDITIONS[condition]
    model.to(device)
    transformer_parameters = depth.configure_trainable(model, block_ids)
    torch.manual_seed(seed)
    base_head = depth.MLP8().to(device)
    base_initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                               for p in base_head.parameters())).hexdigest()
    reference = json.loads((DEPTH_EXPERIMENT / "resources.json").read_text())["runs"][
        f"{condition.lower()}_fold0_seed{seed}"]["head_initial_sha256"]
    if base_initial_hash != reference:
        raise ValueError(f"MLP8 inicial no coincide con control {condition}, semilla {seed}")
    rng_after_head_init = torch.get_rng_state()
    head = GeometryMLP8().to(device)
    with torch.no_grad():
        head.network[0].weight[:, :768].copy_(base_head.network[0].weight)
        head.network[0].weight[:, 768:].zero_()
        head.network[0].bias.copy_(base_head.network[0].bias)
        head.network[3].load_state_dict(base_head.network[3].state_dict(), strict=True)
    del base_head
    # No permitir que la inicialización temporal de 56 pesos extra cambie el
    # dropout del backbone/cabeza frente a su control SLICES-only apareado.
    torch.set_rng_state(rng_after_head_init)
    if sum(p.numel() for p in head.parameters()) != 6217:
        raise ValueError("Conteo de parámetros MLP8 geométrica incorrecto")
    # La única estandarización usada por el modelo se ajustó solo en inner-train.
    joblib.dump(scaler, run_dir / "geometry_scaler_inner_train_only.joblib")
    model.eval()
    for layer_id in block_ids:
        model.transformer_decoder.layers[layer_id].train()
    head.train()
    optimizer = torch.optim.AdamW([
        {"params": transformer_parameters, "lr": 1e-5, "weight_decay": 0.01},
        {"params": list(head.parameters()), "lr": 1e-3, "weight_decay": 1e-4}], betas=(0.9, 0.999))
    # Entradas tokenizadas una sola vez por corrida.
    encoded = depth.encode_all(frame, stoi, context)
    g_scaled = {sid: scaler.transform(value.reshape(1, -1))[0].astype(np.float32)
                for sid, value in geo_lookup.items()}
    def feature(ids: list[str], do_grad: bool) -> Tensor:
        sequences = [encoded[sid] for sid in ids]
        with torch.set_grad_enabled(do_grad):
            emb = depth.final_end(model, sequences, stoi["<pad>"], device)
            geom = torch.as_tensor(np.stack([g_scaled[sid] for sid in ids]), dtype=torch.float32, device=device)
            result = torch.cat((emb, geom), dim=-1)
        return result
    rng = np.random.default_rng(seed)
    max_epochs, patience_limit = (3, 2) if smoke else (200, 20)
    best_loss, best_epoch, patience, steps = float("inf"), 0, 0, 0
    history: list[dict[str, Any]] = []
    started, cpu_started = time.monotonic(), time.process_time()
    prior_wall = prior_cpu = 0.0
    last_path=run_dir/"last.pt"
    start_epoch=1
    history_path=EXPERIMENT/"historial_entrenamiento.csv"
    if last_path.exists() and not smoke:
        resume=torch.load(last_path,map_location="cpu",weights_only=False)
        if resume.get("run")!=run or resume.get("base_sha256")!=BASE_SHA:
            raise ValueError(f"Checkpoint de reanudación incompatible: {run}")
        for layer_id in block_ids:
            model.transformer_decoder.layers[layer_id].load_state_dict(resume["blocks"][str(layer_id)],strict=True)
        head.load_state_dict(resume["head"],strict=True);optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["torch_rng_state"]);rng.bit_generator.state=resume["numpy_rng_state"]
        start_epoch=int(resume["epoch"])+1;steps=int(resume["optimizer_steps"])
        best_loss=float(resume["best_loss"]);best_epoch=int(resume["best_epoch"]);patience=int(resume["patience"])
        prior_wall=float(resume.get("cumulative_wall_seconds",0));prior_cpu=float(resume.get("cumulative_cpu_seconds",0))
        # No se depende del CSV append-only para reanudar: un proceso previo
        # puede haber dejado filas de smoke con ancho distinto. El log y el
        # contador epoch del checkpoint conservan la procedencia reproducible.
    model.train(False)
    for layer_id in block_ids:
        model.transformer_decoder.layers[layer_id].train()
    head.train()
    for epoch in range(start_epoch, max_epochs + 1):
        order = rng.permutation(len(train_groups)); batch_losses=[]; last_geom_grad=0.0
        for offset in range(0, len(order), 8):
            selected = order[offset:offset+8]
            optimizer.zero_grad(set_to_none=True); losses=[]
            for ix in selected:
                _, ids = train_groups[int(ix)]
                x = feature(ids, True)
                y = torch.as_tensor(target_train[int(ix)], dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(head(x) / scale, y / scale, beta=1.0)
                if not torch.isfinite(loss): raise FloatingPointError(f"Loss no finita: {run}")
                losses.append(loss)
            batch_loss = torch.stack(losses).mean(); batch_loss.backward()
            geo_grad = head.network[0].weight.grad[:, 768:]
            if geo_grad is None or not torch.isfinite(geo_grad).all() or not torch.any(geo_grad != 0):
                raise FloatingPointError(f"Pesos geométricos sin gradiente válido: {run}")
            if any(p.grad is not None for p in model.token_embedding_table.parameters()):
                raise ValueError("Token embeddings congelados recibieron gradiente")
            if any(p.grad is not None for p in model.position_embedding_table.parameters()):
                raise ValueError("Positional embeddings congelados recibieron gradiente")
            if not any(p.grad is not None and torch.any(p.grad != 0)
                       for p in transformer_parameters):
                raise FloatingPointError(f"Bloques {block_ids} sin gradiente: {run}")
            last_geom_grad = float(torch.linalg.vector_norm(geo_grad).detach().cpu())
            optimizer.step(); steps += 1; batch_losses.append(float(batch_loss.detach().cpu()))
        model.eval(); head.eval(); val_losses=[]
        with torch.no_grad():
            for index, (_, ids) in enumerate(val_groups):
                pred = head(feature(ids, False))
                target = torch.as_tensor(target_val[index], dtype=torch.float32, device=device)
                val_losses.append(float(F.smooth_l1_loss(pred/scale,target/scale,beta=1.0).cpu()))
        model.eval()
        for layer_id in block_ids: model.transformer_decoder.layers[layer_id].train()
        head.train()
        val_loss=float(np.mean(val_losses))
        elapsed=prior_wall+time.monotonic()-started;cpu_elapsed=prior_cpu+time.process_time()-cpu_started
        row={"condition":condition,"fold":fold,"seed":seed,"run_type":"smoke" if smoke else "full","epoch":epoch,
             "train_loss_minibatch_mean":float(np.mean(batch_losses)),"inner_validation_loss_full":val_loss,
             "optimizer_steps":steps,"geometry_gradient_norm_last_batch":last_geom_grad,
             "elapsed_seconds":elapsed,"peak_rss_bytes":rss_bytes()}
        history.append(row)
        if val_loss < best_loss - 1e-12:
            best_loss,best_epoch,patience=val_loss,epoch,0
            state={"blocks":{str(i):model.transformer_decoder.layers[i].state_dict() for i in block_ids},
                   "head":head.state_dict(),"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_,
                   "condition":condition,"fold":fold,"seed":seed,"blocks_updated":list(block_ids),
                   "best_epoch":best_epoch,"best_inner_loss":best_loss,"residual_scale_ev_atom":scale,"base_sha256":BASE_SHA,
                   "geometry_sha256":sha256(GEOMETRY_SOURCE),"head_initial_sha256":base_initial_hash}
            temp=run_dir/"best.pt.tmp";torch.save(state,temp);temp.replace(run_dir/"best.pt")
        else: patience+=1
        history_path.parent.mkdir(parents=True,exist_ok=True)
        exists=history_path.exists()
        pd.DataFrame([row]).to_csv(history_path,mode="a",header=not exists,index=False)
        resource_data=json.loads((EXPERIMENT/"resources.json").read_text())
        resource_data["runs"][run]={"status":"RUNNING","condition":condition,"fold":fold,"seed":seed,
            "blocks":list(block_ids),"trainable_transformer_parameters":sum(p.numel() for p in transformer_parameters),
            "trainable_head_parameters":6217,"best_epoch":best_epoch,"best_inner_loss":best_loss,
            "epochs":epoch,"optimizer_steps":steps,"wall_seconds":elapsed,
            "cpu_seconds":cpu_elapsed,"peak_rss_bytes":rss_bytes(),"device":str(device),
            "geometry_gradient_norm_last_batch":last_geom_grad}
        resource_data["wall_seconds_observed_total"]=sum(v.get("wall_seconds",0) for v in resource_data["runs"].values())
        resource_data["cpu_seconds_observed_total"]=sum(v.get("cpu_seconds",0) for v in resource_data["runs"].values())
        resource_data["peak_rss_bytes_observed"]=max(v.get("peak_rss_bytes",0) for v in resource_data["runs"].values())
        atomic_json(EXPERIMENT/"resources.json",resource_data)
        last_state={"run":run,"base_sha256":BASE_SHA,
            "blocks":{str(i):model.transformer_decoder.layers[i].state_dict() for i in block_ids},
            "head":head.state_dict(),"optimizer":optimizer.state_dict(),"torch_rng_state":torch.get_rng_state(),
            "numpy_rng_state":rng.bit_generator.state,"epoch":epoch,"optimizer_steps":steps,
            "best_loss":best_loss,"best_epoch":best_epoch,"patience":patience,
            "cumulative_wall_seconds":elapsed,"cumulative_cpu_seconds":cpu_elapsed}
        temp=last_path.with_suffix(".pt.tmp");torch.save(last_state,temp);temp.replace(last_path)
        LOG.info("%s",{"event":"epoch","run":run,**row,"best_epoch":best_epoch})
        if patience>=patience_limit: break
    result={"status":"COMPLETE","condition":condition,"fold":fold,"seed":seed,
            "blocks":list(block_ids),"trainable_transformer_parameters":sum(p.numel() for p in transformer_parameters),
            "trainable_head_parameters":6217,"best_epoch":best_epoch,"best_inner_loss":best_loss,
            "epochs":epoch,"optimizer_steps":steps,"target_scale_ev_atom":scale,
            "wall_seconds":prior_wall+time.monotonic()-started,"cpu_seconds":prior_cpu+time.process_time()-cpu_started,
            "peak_rss_bytes":rss_bytes(),"device":str(device),"head_initial_sha256":base_initial_hash}
    if not smoke:
        (run_dir/"complete.flag").write_text("complete\n",encoding="utf-8")
    else: result["status"]="SMOKE"
    all_resources=json.loads((EXPERIMENT/"resources.json").read_text())
    all_resources["runs"][run]={**result}
    all_resources["completed_runs"]=sum(v.get("status")=="COMPLETE" for v in all_resources["runs"].values())
    all_resources["active_runs"]=sum(v.get("status")=="RUNNING" for v in all_resources["runs"].values())
    all_resources["wall_seconds_observed_total"]=sum(v.get("wall_seconds",0) for v in all_resources["runs"].values())
    all_resources["cpu_seconds_observed_total"]=sum(v.get("cpu_seconds",0) for v in all_resources["runs"].values())
    all_resources["peak_rss_bytes_observed"]=max(v.get("peak_rss_bytes",0) for v in all_resources["runs"].values())
    atomic_json(EXPERIMENT/"resources.json",all_resources)
    return result


def main() -> None:
    """Ejecuta preflight, smoke o las corridas geométricas F2/F4/F8."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("preflight","smoke","all"),required=True)
    parser.add_argument("--device",choices=("cpu","mps","cuda","auto"),default="cpu")
    parser.add_argument("--num-threads",type=int,default=2)
    args=parser.parse_args()
    if args.num_threads<1: parser.error("--num-threads debe ser positivo")
    EXPERIMENT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True); IMAGES.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(EXPERIMENT/"run.log",encoding="utf-8")])
    depth=import_depth_trainer(); torch.set_num_threads(args.num_threads); torch.use_deterministic_algorithms(True)
    device=depth.device_from(args.device)
    audit=preflight_sources(depth)
    if args.phase=="preflight":
        audit.update({"device":str(device),"torch":torch.__version__,"cpu_count":os.cpu_count(),
                      "num_threads":args.num_threads,"mlp8_slices_parameters":6161,"mlp8_geometry_parameters":6217,
                      "new_finetuning_runs":45,"reused_f0_geometry_runs":15,"reused_slices_only_depth_runs":45})
        atomic_json(EXPERIMENT/"summary.json",audit)
        atomic_json(EXPERIMENT/"resources.json",{"status":"PREFLIGHT_PASS","device":str(device),
            "cpu_count":os.cpu_count(),"num_threads":args.num_threads,"runs":{},"completed_runs":0,
            "active_runs":0,"wall_seconds_observed_total":0.0,"cpu_seconds_observed_total":0.0,
            "peak_rss_bytes_observed":rss_bytes()})
        LOG.info("%s",audit); return
    summary=json.loads((EXPERIMENT/"summary.json").read_text())
    if summary.get("status")!="PASS": raise RuntimeError("Primero debe pasar --phase preflight")
    frame,folds,model,stoi,context=depth.load_sources()
    _,geometry,_=load_geometry(depth,frame)
    resources=json.loads((EXPERIMENT/"resources.json").read_text())
    if args.phase=="smoke":
        for condition in DEPTHS:
            fit_one(depth,frame,folds,model,stoi,context,geometry,condition,0,42,device,smoke=True)
        LOG.info("%s",{"event":"all_geometry_smokes_pass"}); return
    for condition in DEPTHS:
        for fold in range(5):
            for seed in SEEDS:
                LOG.info("%s",{"event":"run_start","condition":condition,"fold":fold,"seed":seed})
                fit_one(depth,frame,folds,model,stoi,context,geometry,condition,fold,seed,device)
    final=json.loads((EXPERIMENT/"summary.json").read_text()); final["training_status"]="COMPLETE";final["status"]="COMPLETE"
    atomic_json(EXPERIMENT/"summary.json",final)
    resource_data=json.loads((EXPERIMENT/"resources.json").read_text());resource_data["status"]="COMPLETE"
    resource_data["completed_geometry_runs"]=sum(resource_data["runs"].get(f"{c.lower()}_fold{f}_seed{s}",{}).get("status")=="COMPLETE"
        for c in DEPTHS for f in range(5) for s in SEEDS)
    resource_data["planned_geometry_runs"]=45
    atomic_json(EXPERIMENT/"resources.json",resource_data)
    LOG.info("%s",{"event":"all_geometry_finetuning_complete"})


if __name__=="__main__":
    main()
