"""Entrena MatGPT con geometría inyectada antes de la atención causal.

Los datos, etiquetas, pliegues, Ridge y tokenizador históricos se reutilizan
sin modificaciones. Se entrenan condiciones de fusión temprana y tardía con
una proyección geométrica equiparada y una MLP8 residual fija.
"""
from __future__ import annotations

import argparse
import gc
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

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXPERIMENT = ROOT / "experiments/MatGPT_SLICES_Geometria_Fusion_Temprana"
IMAGES = ROOT / "images/MatGPT_SLICES_Geometria_Fusion_Temprana"
SCRIPT = Path(__file__).resolve().parent
DEPTH_MODULE = "scripts.MatGPT.MatGPT_SLICES_Finetuning_Profundidad.train"
GEOMETRY_CSV = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.csv"
GEOMETRY_MANIFEST = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/geometry_cell_development.manifest.json"
GEOMETRY_JSONL = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria/structures_development.jsonl"
BASE_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
GEOMETRY_COLUMNS = ("a_reduced_shape", "b_reduced_shape", "c_reduced_shape",
                    "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg",
                    "volume_per_atom_ang3")
SEEDS = (42, 123, 2026)
DEPTHS = {"F0": (), "F8": tuple(range(4, 12))}
MODES = ("early", "late")
LOG = logging.getLogger("matgpt_early_geometry_fusion")


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo mediante lectura por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Guarda JSON con sustitución atómica para tolerar interrupciones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def depth_module() -> Any:
    """Carga las funciones auditadas de datos, tokenización y MatGPT."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(DEPTH_MODULE)


class GeometryProjection(nn.Module):
    """Proyecta siete propiedades de celda a la dimensión oculta de MatGPT."""

    def __init__(self, hidden_size: int = 768) -> None:
        """Inicializa una proyección lineal sin sesgo con pesos exactamente cero."""
        super().__init__()
        self.linear = nn.Linear(7, hidden_size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, geometry: Tensor) -> Tensor:
        """Devuelve el vector geométrico proyectado para cada estructura."""
        if geometry.ndim != 2 or geometry.shape[1] != 7:
            raise ValueError(f"Se esperaba geometría [batch,7], se recibió {tuple(geometry.shape)}")
        return self.linear(geometry)


class MLP8(nn.Module):
    """Regresor residual histórico, Linear(768,8)-GELU-Dropout-Linear(8,1)."""

    def __init__(self) -> None:
        """Construye la cabeza fija de 6,161 parámetros."""
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, vectors: Tensor) -> Tensor:
        """Predice residuos y los centra dentro de cada triplete."""
        raw = self.network(vectors).squeeze(-1)
        return raw - raw.mean(dim=-1, keepdim=True)


def geometry_table(depth: Any, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Audita origen e identidad de las 210 celdas y alinea sus descriptores.

    Args:
        depth: Módulo de utilidades históricas MatGPT.
        frame: Dataset físico de desarrollo con IDs y composición.

    Returns:
        Matriz float32 [210,7] en el orden del dataset y manifiesto de procedencia.
    """
    if not GEOMETRY_CSV.is_file() or not GEOMETRY_MANIFEST.is_file() or not GEOMETRY_JSONL.is_file():
        raise FileNotFoundError("Falta el CSV, manifiesto o JSONL de procedencia de geometría")
    manifest = json.loads(GEOMETRY_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("geometry_csv_sha256") != sha256(GEOMETRY_CSV):
        raise ValueError("El manifiesto o hash de la geometría no es válido")
    if manifest.get("entries") != 210 or manifest.get("source_counts") != {"gga-static_endpoint_cleaned": 210}:
        raise ValueError("El manifiesto no certifica 210 endpoints estáticos de la fuente declarada")
    data = pd.read_csv(GEOMETRY_CSV)
    if len(data) != 210 or data.structure_id.nunique() != 210:
        raise ValueError("La geometría no contiene 210 IDs físicos únicos")
    if set(data.geometry_source) != {"gga-static_endpoint_cleaned"}:
        raise ValueError("Procedencia geométrica desconocida o mezclada")
    if data.source_sha256.isna().any() or data.source_path.isna().any():
        raise ValueError("Falta hash o ruta de procedencia de una estructura")
    if set(data.structure_id.astype(str)) != set(frame.structure_id.astype(str)):
        raise ValueError("Los IDs geométricos no coinciden con el dataset ABS3")
    structures = [json.loads(line) for line in GEOMETRY_JSONL.read_text(encoding="utf-8").splitlines()]
    if len(structures) != 210:
        raise ValueError("El JSONL de estructuras no contiene 210 filas")
    if set(data.source_sha256.astype(str)) != {sha256(GEOMETRY_JSONL)}:
        raise ValueError("Hash del JSONL no coincide con la auditoría geométrica")
    structure_by_id = {str(item["structure_id"]): item for item in structures}
    if set(structure_by_id) != set(frame.structure_id.astype(str)):
        raise ValueError("Las estructuras JSONL no alinean exactamente por ID")
    source_paths = data.set_index("structure_id").loc[frame.structure_id].source_path.astype(str)
    for sid, source_path in zip(frame.structure_id.astype(str), source_paths):
        try:
            line_number = int(source_path.rsplit(":line:", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Referencia de línea inválida para {sid}: {source_path}") from error
        if not 1 <= line_number <= len(structures) or str(structures[line_number - 1]["structure_id"]) != sid:
            raise ValueError(f"El JSONL no corresponde al ID físico {sid}")
    # Verifica fórmula reducida con la estructura original, sin usar etiquetas.
    try:
        from pymatgen.core import Composition, Structure
        expected_formula = frame.set_index("structure_id").formula.astype(str)
        for sid, item in structure_by_id.items():
            structure = Structure.from_dict(item["structure"])
            if len(structure) != int(frame.set_index("structure_id").loc[sid, "n_atoms"]):
                raise ValueError(f"Número de átomos incompatible con estructura {sid}")
            if structure.composition.reduced_formula != Composition(expected_formula.loc[sid]).reduced_formula:
                raise ValueError(f"Composición incompatible con estructura {sid}")
    except ImportError as error:
        raise RuntimeError("pymatgen es requerido para verificar composición de celdas") from error
    id_to_row = frame.set_index("structure_id")
    geom = data.set_index("structure_id").loc[frame.structure_id]
    if not np.array_equal(geom.n_atoms.astype(int).to_numpy(), id_to_row.loc[frame.structure_id].n_atoms.astype(int).to_numpy()):
        raise ValueError("Número de átomos difiere entre geometría y dataset")
    if not set(GEOMETRY_COLUMNS).issubset(data.columns):
        raise ValueError("Faltan uno o más de los siete descriptores")
    vals = geom.loc[:, GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all() or (vals[:, :3] <= 0).any() or (vals[:, 6] <= 0).any():
        raise ValueError("Descriptores no finitos o razones/volumen no positivos")
    if (vals[:, 3:6] <= 0).any() or (vals[:, 3:6] >= 180).any():
        raise ValueError("Ángulos fuera del intervalo (0,180)")
    return vals.astype(np.float32), {"status": "PASS", "entries": len(data),
        "source_counts": data.geometry_source.value_counts().to_dict(),
        "csv_sha256": sha256(GEOMETRY_CSV), "manifest_sha256": sha256(GEOMETRY_MANIFEST),
        "structures_jsonl_sha256": sha256(GEOMETRY_JSONL)}


def assert_zero_fusion_equivalence(depth: Any, model: Any, frame: pd.DataFrame,
                                   sequences: dict[str, list[int]], stoi: dict[str, int],
                                   block_size: int, geo_raw: np.ndarray,
                                   device: torch.device) -> dict[str, float]:
    """Exige equivalencia de F0/F8 temprana/tardía con proyección nula.

    También verifica autograd de la proyección en F0/F8, sin hacer optimizer.step.
    """
    del block_size
    ids = frame.structure_id.astype(str).tolist()[:3]
    seqs = [sequences[sid] for sid in ids]
    geometry = torch.as_tensor(geo_raw[:3], dtype=torch.float32, device=device)
    results: dict[str, float] = {}
    checkpoint = depth.SOURCE / "mat-uncond/weights/transformer_property_0.pth"
    for condition, blocks in DEPTHS.items():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        configure(model, blocks)
        with torch.no_grad():
            reference = depth.final_end(model, seqs, stoi["<pad>"], device)
        for mode in MODES:
            projection = GeometryProjection().to(device)
            actual = hidden_end(model, seqs, stoi["<pad>"], device, projection, geometry, mode)
            error = float((actual.detach() - reference).abs().max())
            results[f"{condition}_{mode}_max_abs_error"] = error
            if not torch.allclose(actual.detach(), reference, atol=2e-6, rtol=2e-6):
                raise AssertionError(f"Proyección cero no reproduce baseline {condition}/{mode}: {error}")
            # La pérdida aleatoria evita que suma de LayerNorm cancele el gradiente.
            torch.manual_seed(813)
            weight = torch.randn_like(actual)
            (actual * weight).sum().backward()
            grad = projection.linear.weight.grad
            norm = 0.0 if grad is None else float(grad.norm())
            results[f"{condition}_{mode}_projection_gradient_norm"] = norm
            if grad is None or not torch.isfinite(grad).all() or norm == 0:
                raise AssertionError(f"Proyección {condition}/{mode} no recibe gradiente")
            if condition == "F8" and not any(p.grad is not None and torch.any(p.grad != 0)
                                               for p in model.transformer_decoder.layers[4:12].parameters()):
                raise AssertionError("F8 no transmite gradiente a bloques 4–11")
            model.zero_grad(set_to_none=True)
    return results


def assert_trained_geometry_sensitivity(model: Any, projection: GeometryProjection, head: nn.Module,
                                        mode: str, sequences: list[list[int]], pad_id: int,
                                        geometry: np.ndarray, device: torch.device) -> dict[str, float]:
    """Mide el efecto de cambiar geometría manteniendo SLICES fijo tras updates."""
    if torch.count_nonzero(projection.linear.weight).item() == 0:
        raise AssertionError("La proyección sigue en cero después del entrenamiento smoke")
    sequence = sequences[0]
    geom_a, geom_b = torch.as_tensor(geometry[:1], device=device), torch.as_tensor(geometry[1:2], device=device)
    model.eval(); projection.eval(); head.eval()
    with torch.no_grad():
        h_a = hidden_end(model, [sequence], pad_id, device, projection, geom_a, mode)
        h_b = hidden_end(model, [sequence], pad_id, device, projection, geom_b, mode)
    difference = float((h_a-h_b).norm())
    relative = difference / max(float(h_a.norm()), 1e-12)
    if difference <= 1e-7:
        raise AssertionError(f"FINAL_END no es sensible a geometría ({mode}, norma={difference})")
    return {"final_end_geometry_sensitivity_l2": difference,
            "final_end_geometry_sensitivity_relative": relative,
            "geometry_projection_weight_l2": float(projection.linear.weight.detach().norm())}


def hidden_end(model: Any, sequences: list[list[int]], pad_id: int, device: torch.device,
               projection: GeometryProjection, geometry: Tensor, mode: str) -> Tensor:
    """Ejecuta MatGPT e inyecta geometría antes del bloque cero o tras FINAL_END.

    En modo temprano se suma la proyección a token+posición antes del decoder.
    En modo tardío se suma al vector contextualizado de <end>.
    No se desactiva autograd: en F0 el gradiente debe cruzar el backbone congelado.
    """
    if mode not in MODES:
        raise ValueError(f"Modo de fusión inválido: {mode}")
    length = max(map(len, sequences))
    batch = torch.full((len(sequences), length), pad_id, dtype=torch.long, device=device)
    end_positions = []
    for row, ids in enumerate(sequences):
        if not ids or len(ids) > model.block_size:
            raise ValueError("Secuencia vacía o más larga que el contexto MatGPT")
        batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        end_positions.append(len(ids) - 1)
    x = model.token_embedding_table(batch)
    x = x + model.position_embedding_table(torch.arange(length, device=device))
    geo = projection(geometry).to(dtype=x.dtype)
    if mode == "early":
        x = x + geo[:, None, :]
    x = x.permute(1, 0, 2)
    mask = model.generate_square_subsequent_mask(length).to(device)
    memory = torch.zeros(length, len(sequences), model.n_embd, device=device, dtype=x.dtype)
    contextual = model.transformer_decoder(x, memory, tgt_mask=mask).permute(1, 0, 2)
    contextual = model.ln_f(contextual)
    ends = contextual[torch.arange(len(sequences), device=device), torch.tensor(end_positions, device=device)]
    return ends + geo if mode == "late" else ends


def configure(model: Any, block_ids: tuple[int, ...]) -> list[nn.Parameter]:
    """Congela todos los pesos originales salvo los bloques designados."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for index in block_ids:
        for p in model.transformer_decoder.layers[index].parameters():
            p.requires_grad_(True)
            params.append(p)
    if any(p.requires_grad for p in model.token_embedding_table.parameters()) or any(
        p.requires_grad for p in model.position_embedding_table.parameters()) or any(
        p.requires_grad for p in model.ln_f.parameters()) or any(p.requires_grad for p in model.lm_head.parameters()):
        raise RuntimeError("Se descongeló un módulo que debía permanecer fijo")
    return params


def groups_with_ids(depth: Any, frame: pd.DataFrame, ids: set[str]) -> list[tuple[str, list[str]]]:
    """Crea fórmulas con los tres polimorfos completos en orden histórico."""
    return depth.groups_for(frame, ids)


def init_seed(seed: int) -> None:
    """Fija RNGs Python, NumPy y PyTorch para reproducibilidad local."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_path(mode: str, depth_name: str, fold: int, seed: int) -> Path:
    """Devuelve la carpeta inmutable de la corrida especificada."""
    return EXPERIMENT / "checkpoints" / f"{depth_name.lower()}_{mode}_fold{fold}_seed{seed}"


def record_resource(run_name: str, row: dict[str, Any]) -> None:
    """Actualiza el manifiesto único de recursos por corrida."""
    path = EXPERIMENT / "resources.json"
    value = json.loads(path.read_text()) if path.exists() else {"status": "RUNNING", "runs": {}}
    value.setdefault("runs", {})[run_name] = row
    value["completed_runs"] = sum(r.get("status") == "COMPLETE" for r in value["runs"].values())
    value["active_runs"] = sum(r.get("status") == "RUNNING" for r in value["runs"].values())
    value["wall_seconds_observed_total"] = sum(r.get("wall_seconds", 0.0) for r in value["runs"].values())
    value["cpu_seconds_observed_total"] = sum(r.get("cpu_seconds", 0.0) for r in value["runs"].values())
    value["peak_rss_bytes_observed"] = max((r.get("peak_rss_bytes", 0) for r in value["runs"].values()), default=0)
    atomic_json(path, value)


def train_run(depth: Any, frame: pd.DataFrame, folds: pd.DataFrame, geo_raw: np.ndarray,
              sequences: dict[str, list[int]], model: Any, stoi: dict[str, int], block_size: int,
              mode: str, depth_name: str, fold: int, seed: int, device: torch.device,
              smoke: bool = False) -> dict[str, Any]:
    """Ajusta una condición agrupando tripletes y selecciona por validación interna.

    Args:
        depth: Módulo con las funciones históricas de preparación y targets.
        frame: Datos DFT de desarrollo.
        folds: Asignaciones históricas por estructura, fórmula y fold.
        geo_raw: Siete descriptores físicos en orden de frame.
        sequences: IDs tokenizados con el vocabulario original.
        model: Instancia MatGPT cargada desde checkpoint original.
        stoi: Vocabulario original.
        block_size: Contexto máximo auditado.
        mode: ``early`` o ``late``.
        depth_name: ``F0`` o ``F8``.
        fold: Outer fold 0..4.
        seed: Semilla histórica.
        device: Dispositivo explícito.
        smoke: Limita la corrida a unas pocas fórmulas/épocas.

    Returns:
        Metadatos medidos de ejecución y estado del checkpoint.
    """
    name = f"{depth_name.lower()}_{mode}_fold{fold}_seed{seed}" + ("_smoke" if smoke else "")
    out = run_path(mode, depth_name, fold, seed)
    if not smoke and (out / "complete.flag").is_file():
        LOG.info("%s", {"event": "skip_complete", "run": name})
        return {"status": "COMPLETE", "run": name}
    checkpoint = depth.SOURCE / "mat-uncond/weights/transformer_property_0.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    init_seed(seed)
    part = folds[folds.outer_fold == fold]
    train_ids = set(part.loc[part.inner_role == "inner_train", "structure_id"].astype(str))
    valid_ids = set(part.loc[part.inner_role == "inner_validation", "structure_id"].astype(str))
    train_groups = groups_with_ids(depth, frame, train_ids)
    valid_groups = groups_with_ids(depth, frame, valid_ids)
    if smoke:
        train_groups, valid_groups = train_groups[:2], valid_groups[:1]
    row_lookup = {str(s): i for i, s in enumerate(frame.structure_id.astype(str))}
    scaler = StandardScaler().fit(np.stack([geo_raw[row_lookup[s]] for _, trip in train_groups for s in trip]))
    geom_scaled = scaler.transform(geo_raw).astype(np.float32)
    geometry_by_id = {sid: geom_scaled[ix] for sid, ix in row_lookup.items()}
    targets = depth.target_matrix(frame, train_groups)
    valid_targets = depth.target_matrix(frame, valid_groups)
    scale = float(targets.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Escala residual de inner-train inválida")
    model.to(device)
    block_ids = DEPTHS[depth_name]
    block_params = configure(model, block_ids)
    # Inicializar primero MLP8 con la semilla histórica; así coincide con las
    # cabezas SLICES-only y entre fusión temprana/tardía.
    head = depth.MLP8().to(device)
    rng_after_head = torch.get_rng_state()
    projection = GeometryProjection().to(device)
    # La inicialización temporal de nn.Linear se descarta al fijar pesos a cero;
    # restaurar RNG conserva la secuencia histórica de dropout tras MLP8.
    torch.set_rng_state(rng_after_head)
    if sum(p.numel() for p in projection.parameters()) != 5376 or sum(p.numel() for p in head.parameters()) != 6161:
        raise RuntimeError("Dimensiones/parametrización del modelo cambiaron")
    model.eval()
    for ix in block_ids:
        model.transformer_decoder.layers[ix].train()
    projection.train(); head.train()
    groups = []
    if block_params:
        groups.append({"params": block_params, "lr": 1e-5, "weight_decay": 0.01})
    groups.extend([{"params": projection.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
                   {"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}])
    optimizer = torch.optim.AdamW(groups)
    init_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()
    geo_init_hash = hashlib.sha256(projection.linear.weight.detach().cpu().numpy().tobytes()).hexdigest()
    best_loss, best_epoch, patience, steps = float("inf"), 0, 0, 0
    max_epochs, patience_max = (3, 2) if smoke else (200, 20)
    started, cpu_started = time.monotonic(), time.process_time()
    history: list[dict[str, Any]] = []
    start_epoch = 1
    rng = np.random.default_rng(seed)
    last_path = out / "last.pt"
    if last_path.is_file() and not smoke:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("run_name") != name or saved.get("base_sha256") != BASE_SHA:
            raise ValueError(f"Checkpoint de reanudación incompatible: {name}")
        saved_model = saved["model"]
        for ix in block_ids:
            model.transformer_decoder.layers[ix].load_state_dict(saved_model["blocks"][str(ix)], strict=True)
        projection.load_state_dict(saved_model["projection"], strict=True)
        head.load_state_dict(saved_model["head"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        start_epoch = int(saved["epoch"]) + 1
        best_loss, best_epoch = float(saved["best_loss"]), int(saved["best_epoch"])
        patience, steps = int(saved["patience"]), int(saved["optimizer_steps"])
        history_path = EXPERIMENT / "historial_entrenamiento.csv"
        if history_path.is_file():
            previous = pd.read_csv(history_path)
            selected = previous[(previous["mode"] == mode) & (previous.condition == depth_name) &
                (previous.fold == fold) & (previous.seed == seed) & (previous.run_type == "full")]
            history = selected.to_dict("records")
        LOG.info("%s", {"event": "resume", "run": name, "start_epoch": start_epoch,
                         "best_epoch": best_epoch, "optimizer_steps": steps})

    def get_h(ids: list[str], train: bool) -> Tensor:
        seqs = [sequences[s] for s in ids]
        geo = torch.as_tensor(np.stack([geometry_by_id[s] for s in ids]), dtype=torch.float32, device=device)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            return hidden_end(model, seqs, stoi["<pad>"], device, projection, geo, mode)

    def evaluate() -> float:
        model.eval(); projection.eval(); head.eval()
        losses = []
        with torch.no_grad():
            for j, (_, ids) in enumerate(valid_groups):
                h = get_h(ids, False)
                y = torch.as_tensor(valid_targets[j], dtype=torch.float32, device=device)
                losses.append(float(F.smooth_l1_loss(head(h) / scale, y / scale, beta=1.0)))
        model.eval(); projection.train(); head.train()
        for ix in block_ids:
            model.transformer_decoder.layers[ix].train()
        return float(np.mean(losses))

    for epoch in range(start_epoch, max_epochs + 1):
        model.eval()
        for ix in block_ids:
            model.transformer_decoder.layers[ix].train()
        projection.train(); head.train()
        order = rng.permutation(len(train_groups))
        losses = []
        for start in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            local = []
            for ix in order[start:start + 8]:
                _, ids = train_groups[int(ix)]
                h = get_h(ids, True)
                y = torch.as_tensor(targets[int(ix)], dtype=torch.float32, device=device)
                loss = F.smooth_l1_loss(head(h) / scale, y / scale, beta=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Pérdida no finita")
                local.append(loss)
            batch_loss = torch.stack(local).mean()
            batch_loss.backward()
            if projection.linear.weight.grad is None or not torch.isfinite(projection.linear.weight.grad).all():
                raise FloatingPointError("Gradiente geométrico ausente/no finito")
            if not torch.any(projection.linear.weight.grad != 0):
                raise FloatingPointError("Proyección geométrica no recibió gradiente")
            if depth_name == "F8" and not any(p.grad is not None and torch.any(p.grad != 0) for p in block_params):
                raise FloatingPointError("Los bloques Transformer 4–11 no reciben gradientes")
            if any(p.grad is not None for p in model.token_embedding_table.parameters()):
                raise RuntimeError("Se observó gradiente en token embeddings congelados")
            if any(p.grad is not None for p in model.position_embedding_table.parameters()):
                raise RuntimeError("Se observó gradiente en position embeddings congelados")
            if any(p.grad is not None for p in model.ln_f.parameters()):
                raise RuntimeError("Se observó gradiente en ln_f congelada")
            optimizer.step(); steps += 1
            losses.append(float(batch_loss.detach()))
        val = evaluate()
        hist = {"mode": mode, "condition": depth_name, "fold": fold, "seed": seed,
                "epoch": epoch, "run_type": "smoke" if smoke else "full",
                "train_loss": float(np.mean(losses)), "inner_validation_loss": val,
                "optimizer_steps": steps, "elapsed_seconds": time.monotonic() - started,
                "peak_rss_bytes": _rss()}
        history.append(hist)
        if not smoke:
            append_history(hist)
        if val < best_loss - 1e-12:
            best_loss, best_epoch, patience = val, epoch, 0
            if not smoke:
                _atomic_torch(out / "best.pt", _state(model, block_ids, projection, head, mode, depth_name,
                    fold, seed, scale, best_epoch, best_loss, BASE_SHA, scaler.mean_, scaler.scale_))
        else:
            patience += 1
        _log_resource(name, "RUNNING", started, cpu_started, epoch, steps,
                      _trainable_count(block_params, projection, head), mode, depth_name, fold, seed)
        LOG.info("%s", {"event": "epoch", **hist, "best_epoch": best_epoch})
        if not smoke:
            _atomic_torch(out / "last.pt", {"model": _state(model, block_ids, projection, head, mode,
                depth_name, fold, seed, scale, best_epoch, best_loss, BASE_SHA, scaler.mean_, scaler.scale_),
                "optimizer": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": rng.bit_generator.state, "run_name": name, "base_sha256": BASE_SHA,
                "epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss,
                "patience": patience, "optimizer_steps": steps})
        if patience >= patience_max:
            break
    result = {"status": "SMOKE" if smoke else "COMPLETE", "mode": mode, "condition": depth_name,
              "fold": fold, "seed": seed, "blocks": list(block_ids), "best_epoch": best_epoch,
              "best_inner_loss": best_loss, "epochs": len(history), "optimizer_steps": steps,
              "projection_parameters": sum(p.numel() for p in projection.parameters()),
              "trainable_transformer_parameters": sum(p.numel() for p in block_params),
              "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
              "head_initial_sha256": init_hash, "projection_initial_sha256": geo_init_hash,
              "geometry_scaler_mean": scaler.mean_.tolist(), "geometry_scaler_scale": scaler.scale_.tolist(),
              "scale_residual_ev_atom": scale, "wall_seconds": time.monotonic()-started,
              "cpu_seconds": time.process_time()-cpu_started, "peak_rss_bytes": _rss(),
              "device": str(device), "status_detail": "smoke only" if smoke else "selected by inner validation"}
    if smoke:
        sensitivity = assert_trained_geometry_sensitivity(model, projection, head, mode,
            [sequences[s] for s in frame.structure_id.astype(str).tolist()[:2]], stoi["<pad>"],
            geom_scaled[:2], device)
        result.update(sensitivity)
    if not smoke:
        (out / "complete.flag").write_text("complete\n", encoding="utf-8")
        _log_resource(name, "COMPLETE", started, cpu_started, len(history), steps,
                      _trainable_count(block_params, projection, head), mode, depth_name, fold, seed,
                      result=result)
    else:
        _log_resource(name, "SMOKE", started, cpu_started, len(history), steps,
                      _trainable_count(block_params, projection, head), mode, depth_name, fold, seed,
                      result=result)
    del optimizer, projection, head
    gc.collect()
    return result


def _rss() -> int:
    """Obtiene RSS máximo del proceso en bytes (macOS/Linux)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _trainable_count(blocks: list[nn.Parameter], projection: nn.Module, head: nn.Module) -> int:
    """Cuenta parámetros entrenables de bloques, proyección y MLP8."""
    return sum(p.numel() for p in blocks) + sum(p.numel() for p in projection.parameters()) + sum(p.numel() for p in head.parameters())


def _state(model: Any, block_ids: tuple[int, ...], projection: GeometryProjection, head: nn.Module,
           mode: str, condition: str, fold: int, seed: int, scale: float,
           best_epoch: int, best_loss: float, base_sha: str,
           scaler_mean: np.ndarray, scaler_scale: np.ndarray) -> dict[str, Any]:
    """Empaqueta solamente módulos adaptados, sin duplicar checkpoint base."""
    return {"blocks": {str(i): model.transformer_decoder.layers[i].state_dict() for i in block_ids},
            "projection": projection.state_dict(), "head": head.state_dict(), "mode": mode,
            "condition": condition, "fold": fold, "seed": seed, "scale_ev_atom": scale,
            "best_epoch": best_epoch, "best_inner_loss": best_loss, "base_sha256": base_sha,
            "projection_shape": [7, 768], "head_shape": [768, 8, 1],
            "geometry_scaler_mean": np.asarray(scaler_mean).tolist(),
            "geometry_scaler_scale": np.asarray(scaler_scale).tolist()}


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    """Guarda de forma atómica un checkpoint parcial o seleccionado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _log_resource(name: str, status: str, started: float, cpu_started: float,
                  epochs: int, steps: int, trainable: int, mode: str, condition: str,
                  fold: int, seed: int, result: dict[str, Any] | None = None) -> None:
    """Registra tiempo de pared/CPU, memoria, optimizaciones y estado."""
    path = EXPERIMENT / "resources.json"
    resources = json.loads(path.read_text()) if path.exists() else {"status": "RUNNING", "device": "cpu", "runs": {}}
    row = {"status": status, "mode": mode, "condition": condition, "fold": fold, "seed": seed,
           "epochs": epochs, "optimizer_steps": steps, "trainable_parameters": trainable,
           "wall_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu_started,
           "peak_rss_bytes": _rss(), "device": resources.get("device", "cpu")}
    if result:
        row.update(result)
    resources.setdefault("runs", {})[name] = row
    resources["completed_runs"] = sum(v.get("status") == "COMPLETE" for v in resources["runs"].values())
    resources["active_runs"] = sum(v.get("status") == "RUNNING" for v in resources["runs"].values())
    resources["wall_seconds_observed_total"] = sum(v.get("wall_seconds", 0) for v in resources["runs"].values())
    resources["cpu_seconds_observed_total"] = sum(v.get("cpu_seconds", 0) for v in resources["runs"].values())
    resources["peak_rss_bytes_observed"] = max(v.get("peak_rss_bytes", 0) for v in resources["runs"].values())
    atomic_json(path, resources)


def append_history(row: dict[str, Any]) -> None:
    """Añade una época a la única tabla de historial de entrenamiento."""
    path = EXPERIMENT / "historial_entrenamiento.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def preflight(depth: Any) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, int], int, np.ndarray, dict[str, Any]]:
    """Verifica las fuentes auditadas antes de cualquier entrenamiento."""
    frame, folds, model, stoi, context = depth.load_sources()
    geometry, provenance = geometry_table(depth, frame)
    if depth.sha256(depth.SOURCE / "mat-uncond/weights/transformer_property_0.pth") != BASE_SHA:
        raise ValueError("Checkpoint base MatGPT no coincide con hash bloqueado")
    if len(model.transformer_decoder.layers) != 12 or model.n_embd != 768:
        raise ValueError("La arquitectura cargada no es 12 bloques × 768")
    if "<end>" not in stoi or "<pad>" not in stoi:
        raise ValueError("Tokenizer no contiene los tokens especiales esperados")
    preds = pd.read_csv(depth.EXPERIMENT / "predicciones_oof_por_estructura.csv")
    if not {"F0", "F8"}.issubset(set(preds.condition.astype(str))):
        raise ValueError("Faltan controles SLICES-only F0/F8 para el análisis apareado")
    if len(preds[preds.condition.astype(str).isin(["F0", "F8"])]) != 1260:
        raise ValueError("Controles SLICES-only F0/F8 no contienen 1,260 filas esperadas")
    return frame, folds, model, stoi, context, geometry, provenance


def main() -> None:
    """Ejecuta auditoría, smoke tests o las cuatro condiciones completas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "all"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--conditions", nargs="+", default=["F0_early", "F8_early", "F0_late", "F8_late"])
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads debe ser positivo")
    if any(c.split("_")[0] not in DEPTHS or c.split("_")[-1] not in MODES for c in args.conditions):
        parser.error("Condiciones válidas: F0_early F8_early F0_late F8_late")
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(EXPERIMENT/"run.log", encoding="utf-8")])
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    depth = depth_module()
    frame, folds, model, stoi, block_size, geo_raw, provenance = preflight(depth)
    encoded = depth.encode_all(frame, stoi, block_size)
    prior_summary_path = EXPERIMENT / "summary.json"
    prior_summary = json.loads(prior_summary_path.read_text()) if prior_summary_path.is_file() else {}
    base_meta = {**prior_summary, "status": "PREFLIGHT_PASS" if args.phase == "preflight" else "RUNNING",
        "checkpoint_sha256": BASE_SHA, "geometry": provenance, "structures": len(frame),
        "compositions": frame.formula.nunique(), "folds": 5, "seeds": list(SEEDS),
        "tokenizer_sha256": depth.TOKEN_SHA, "folds_sha256": depth.FOLD_SHA,
        "conditions_requested": args.conditions, "historical_test_accessed": False,
        "ridge_retrained": False, "geometry_parameters": 5376, "mlp8_parameters": 6161,
        "early_injection": "sum geometry_projection(scaled_cell) into token+position embeddings before transformer_decoder",
        "late_injection": "sum same geometry_projection(scaled_cell) into FINAL_END before fixed MLP8",
        "geometry_scaling": "StandardScaler fit per fold using only inner_train triplets"}
    atomic_json(EXPERIMENT/"summary.json", base_meta)
    resource_path = EXPERIMENT / "resources.json"
    old_resources = json.loads(resource_path.read_text()) if resource_path.is_file() and args.phase == "all" else {}
    atomic_json(resource_path, {**old_resources, "status": "RUNNING", "device": args.device,
        "cpu_count": os.cpu_count(), "num_threads": args.num_threads, "runs": old_resources.get("runs", {})})
    if args.phase == "preflight":
        LOG.info("%s", {"event": "preflight_pass", **base_meta})
        return
    run_specs = [(c.split("_")[0], c.split("_")[-1]) for c in args.conditions]
    if args.phase == "smoke":
        run_specs = [("F0", "early"), ("F8", "early"), ("F0", "late"), ("F8", "late")]
    results = []
    if args.phase == "smoke":
        equivalence = assert_zero_fusion_equivalence(depth, model, frame, encoded, stoi,
            block_size, geo_raw, torch.device(args.device))
        base_meta["preflight_model_tests"] = {"status": "PASS", "zero_projection_equivalence": equivalence}
        atomic_json(EXPERIMENT / "summary.json", base_meta)
        LOG.info("%s", {"event": "equivalence_and_gradient_pass", **equivalence})
    for condition, mode in run_specs:
        results.append(train_run(depth, frame, folds, geo_raw, encoded, model, stoi, block_size,
            mode, condition, 0, 42, torch.device(args.device), smoke=args.phase == "smoke"))
    if args.phase == "all":
        for condition, mode in run_specs:
            for fold in range(5):
                for seed in SEEDS:
                    results.append(train_run(depth, frame, folds, geo_raw, encoded, model, stoi, block_size,
                        mode, condition, fold, seed, torch.device(args.device), smoke=False))
        summary = json.loads((EXPERIMENT/"summary.json").read_text())
        completed = sum(r.get("status") == "COMPLETE" for r in results)
        summary.update({"status": "COMPLETE" if completed == 60 else "PARTIAL",
                        "completed_new_runs": completed, "expected_new_runs": 60,
                        "training_status": "COMPLETE" if completed == 60 else "PARTIAL"})
        atomic_json(EXPERIMENT/"summary.json", summary)
        resource_data = json.loads((EXPERIMENT/"resources.json").read_text())
        resource_data["status"] = summary["status"]
        atomic_json(EXPERIMENT/"resources.json", resource_data)
        gc.collect()
    LOG.info("%s", {"event": "phase_done", "phase": args.phase, "results": len(results)})


if __name__ == "__main__":
    main()
