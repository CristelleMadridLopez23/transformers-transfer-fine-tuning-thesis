"""Entrena MLP8 residual con MatterGPT congelado y geometría de celda tardía.

Reutiliza embeddings y controles históricos verificables; ajusta un StandardScaler
por fold exclusivamente con geometrías de inner-train y solo entrena quince MLP8.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import re
import resource
import time
from pathlib import Path
from typing import Any

PROCESS_WALL_START = time.perf_counter()
PROCESS_CPU_START = resource.getrusage(resource.RUSAGE_SELF)

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
EXPERIMENT = "mattergpt_abs3_geometry_late_fusion"
OUT = ROOT / "experiments" / EXPERIMENT
REPORT = ROOT / "reports" / "MatterGPT" / EXPERIMENT
IMAGES = ROOT / "images" / EXPERIMENT
SCRIPT = Path(__file__).resolve()
BASE = ROOT / "experiments/mattergpt_abs3_frozen_ridge_mlp8"
BASE_SCRIPT = ROOT / "scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py"
GEOM_DIR = ROOT / "experiments/MatGPT/MatGPT_SLICES_Geometria"
GEOM_CSV = GEOM_DIR / "geometry_cell_development.csv"
GEOM_MANIFEST = GEOM_DIR / "geometry_cell_development.manifest.json"
HIST = GEOM_DIR / "results"
DATASET = ROOT / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
FOLDS = ROOT / "reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
RIDGE_DIR = ROOT / "experiments/MatGPT/MatGPT_Regresion_Hibrida_Ridge_Composicional"
GEOM_COLS = ["a_reduced_shape", "b_reduced_shape", "c_reduced_shape", "alpha_reduced_deg", "beta_reduced_deg", "gamma_reduced_deg", "volume_per_atom_ang3"]
SEEDS = (42, 123, 2026)
MODEL_NAMES = ["Ridge_only", "MatterGPT_F0", "MatterGPT_F0_geometry", "MatGPT_F0", "MatGPT_F0_geometry"]
LOG = logging.getLogger(EXPERIMENT)


class MLP8Geometry(nn.Module):
    """Cabeza residual 775→8→1; las siete entradas finales son geométricas."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(775, 8), nn.GELU(), nn.Dropout(0.1), nn.Linear(8, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Devuelve la corrección residual no centrada por fórmula."""
        return self.net(values).squeeze(-1)


def configure_logging() -> None:
    """Configura eventos JSON de una línea para auditoría reproducible."""
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","event":"%(message)s"}')


def sha256(path: Path) -> str:
    """Calcula SHA-256 de un archivo en bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Escribe JSON legible creando el directorio destino."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_baseline_module() -> Any:
    """Importa funciones auditadas de MatterGPT F0 sin ejecutar su main."""
    spec = importlib.util.spec_from_file_location("mattergpt_f0_reference", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar la implementación de referencia: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(mg: Any) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Carga filas de desarrollo, caché y geometría; exige correspondencia por ID.

    La función de datos del experimento MatterGPT filtra train/val antes de
    interpretar labels, de modo que las etiquetas históricas de test no se leen.
    """
    data, folds = mg.dev_data()
    cache_path = BASE / "embeddings_mattergpt.npz"
    cache_manifest = pd.read_csv(BASE / "embedding_manifest.csv", dtype={"structure_id": str})
    cache = np.load(cache_path, allow_pickle=False)
    embeddings = cache["embeddings"]
    ids = cache["structure_ids"].astype(str)
    expected_ids = data.structure_id.to_numpy(dtype=str)
    cfg = json.loads((BASE / "config.yaml").read_text(encoding="utf-8"))
    model_checks = cfg["model"]
    if embeddings.shape != (210, 768) or not np.isfinite(embeddings).all():
        raise ValueError("Caché MatterGPT inválida: se requieren 210×768 valores finitos")
    if not np.array_equal(ids, expected_ids):
        raise ValueError("IDs y orden de la caché difieren de la tabla de desarrollo")
    if not np.allclose(cache["property_vector"], np.array([1.0, 2.0], dtype=np.float32)):
        raise ValueError("La caché no usa el condicionamiento MatterGPT [1.0, 2.0]")
    if sha256(cache_path) != cache_manifest.cache_sha256.iloc[0]:
        raise ValueError("SHA de la caché no coincide con su manifiesto")
    if set(cache_manifest.checkpoint_sha256) != {model_checks["checkpoint_sha256"]} or set(cache_manifest.tokenizer_sha256) != {model_checks["tokenizer_sha256"]}:
        raise ValueError("Checkpoint/tokenizador de la caché difieren del preflight MatterGPT F0")
    if not cache_manifest.embedding_dim.eq(768).all() or not cache_manifest.property_vector.eq("[1.0, 2.0]").all():
        raise ValueError("Manifiesto de embeddings no confirma dimensión/condición")
    aligned_cache = data[["structure_id", "composition_id", "family", "n_atoms", "outer_fold", "topology_sha256"]].merge(
        cache_manifest[["structure_id", "composition_id", "family", "n_atoms", "outer_fold", "topology_sha256", "terminal_policy", "validation_status"]],
        on="structure_id", how="outer", validate="one_to_one", suffixes=("_data", "_cache"), indicator=True)
    if not aligned_cache._merge.eq("both").all():
        raise ValueError("IDs del dataset y manifiesto de embeddings no coinciden")
    for col in ("composition_id", "family", "n_atoms", "outer_fold", "topology_sha256"):
        left = aligned_cache[f"{col}_data"].fillna("").astype(str)
        right = aligned_cache[f"{col}_cache"].fillna("").astype(str)
        if not left.eq(right).all():
            raise ValueError(f"El manifiesto MatterGPT no coincide en {col}")
    if not aligned_cache.terminal_policy.eq("> SLICES <; último estado = ln_f(posición terminal <)").all() or not aligned_cache.validation_status.eq("PASS").all():
        raise ValueError("El manifiesto no confirma extracción FINAL_END/ln_f validada")
    geo_manifest = json.loads(GEOM_MANIFEST.read_text(encoding="utf-8"))
    if geo_manifest.get("status") != "PASS" or sha256(GEOM_CSV) != geo_manifest.get("geometry_csv_sha256"):
        raise ValueError("La geometría no pasa su manifiesto histórico SHA-256")
    geometry = pd.read_csv(GEOM_CSV, dtype={"structure_id": str})
    if set(GEOM_COLS) != set(geo_manifest["descriptor_names"]):
        raise ValueError("Descriptores geométricos no coinciden con la auditoría MatGPT")
    joined = data.merge(geometry, on="structure_id", how="left", validate="one_to_one", suffixes=("", "_geometry"))
    if len(joined) != 210 or joined[GEOM_COLS].isna().any().any() or not np.isfinite(joined[GEOM_COLS].to_numpy(float)).all():
        raise ValueError("Geometría ausente o no finita tras unir por structure_id")
    if not joined.n_atoms.eq(joined.n_atoms_geometry).all():
        raise ValueError("Número de átomos de geometría y SLICES no coincide")
    joined = joined.drop(columns="n_atoms_geometry")
    def stoichiometry(formula: str) -> dict[str, int]:
        """Parse element counts independent of element ordering and explicit ones."""
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if "".join(element + count for element, count in parts) != formula:
            raise ValueError(f"Fórmula no parseable en ID: {formula}")
        return {element: int(count or 1) for element, count in parts}
    id_formulas = joined.structure_id.str.split("--", n=1).str[0]
    if any(stoichiometry(structure_formula) != stoichiometry(composition_formula)
           for structure_formula, composition_formula in zip(id_formulas, joined.composition_id)):
        raise ValueError("Identificadores estructurales no validan composición")
    id_families = joined.structure_id.str.split("--", n=2).str[1].str.extract(r"_(dist_perovskite|hexagonal|needle)$", expand=False)
    if not id_families.eq(joined.family).all():
        raise ValueError("Tipo de polimorfo no coincide entre ID estable y tabla histórica")
    if joined.structure_id.duplicated().any() or joined.groupby("composition_id").size().ne(3).any():
        raise ValueError("No son 210 IDs únicos en 70 tripletes")
    if not set(geometry.geometry_source) == {"gga-static_endpoint_cleaned"}:
        raise ValueError("Fuente geométrica no es exclusivamente gga-static_endpoint_cleaned")
    # Los registros contienen nombres de familia distintos pero inequívocos.
    family_map = {"dist_perovskite": "Perovskite", "hexagonal": "Hexagonal", "needle": "Needle-like"}
    joined["family_report"] = joined.family.map(family_map)
    if joined.family_report.isna().any():
        raise ValueError(f"Familias de polimorfo inesperadas: {joined.loc[joined.family_report.isna(), 'family'].unique()}")
    geometry_manifest = joined[["structure_id", "composition_id", "family_report", "n_atoms", "geometry_source", *GEOM_COLS]].copy()
    # Incluye origen y hashes de procedencia por estructura.
    geometry_manifest = geometry_manifest.merge(geometry[["structure_id", "source_path", "source_sha256"]], on="structure_id", validate="one_to_one")
    baseline_preflight = json.loads((BASE / "preflight.json").read_text(encoding="utf-8"))
    audit = {"status": "PASS", "structures": len(joined), "compositions": joined.composition_id.nunique(),
             "geometry_source_counts": geometry.geometry_source.value_counts().to_dict(),
             "geometry_sha256": sha256(GEOM_CSV), "geometry_source_manifest_sha256": sha256(GEOM_MANIFEST),
             "embedding_cache_sha256": sha256(cache_path), "embedding_rows": int(embeddings.shape[0]),
             "embedding_dim": int(embeddings.shape[1]), "checkpoint_sha256": model_checks["checkpoint_sha256"],
             "tokenizer_sha256": model_checks["tokenizer_sha256"], "property_vector": [1.0, 2.0],
             "topology_hash_missing_entries_in_historical_source": int(data.topology_sha256.fillna("").eq("").sum()),
             "post_ln_f_and_terminal_policy_confirmed_by": str(BASE / "preflight.json"),
             "mattergpt_f0_preflight": baseline_preflight,
             "historical_test_labels_parsed": False}
    return joined, folds, embeddings, geometry_manifest, audit


def paired_head(seed: int, device: torch.device) -> tuple[MLP8Geometry, dict[str, Any]]:
    """Inicializa la cabeza geométrica con los mismos pesos de 768 entradas de C1."""
    torch.manual_seed(seed)
    base_head = mg_module.MLP8().to(device)
    post_base_rng = torch.get_rng_state()
    model = MLP8Geometry().to(device)
    with torch.no_grad():
        model.net[0].weight[:, :768].copy_(base_head.net[0].weight)
        model.net[0].weight[:, 768:].zero_()
        model.net[0].bias.copy_(base_head.net[0].bias)
        model.net[3].weight.copy_(base_head.net[3].weight)
        model.net[3].bias.copy_(base_head.net[3].bias)
    torch.set_rng_state(post_base_rng)
    return model, {"base": base_head}


def audit_initial_equivalence(X: np.ndarray, G: np.ndarray, data: pd.DataFrame, folds: pd.DataFrame) -> dict[str, Any]:
    """Comprueba en modo evaluación equivalencia numérica de las cabezas al paso cero."""
    max_error = 0.0
    count = 0
    for seed in SEEDS:
        for fold in range(5):
            # Inicialización idéntica a train_one; la geometría es cualquier matriz finita.
            model, heads = paired_head(seed, torch.device("cpu"))
            base_head = heads["base"].eval(); model.eval()
            xb = torch.as_tensor(X, dtype=torch.float32)
            gb = torch.as_tensor(G, dtype=torch.float32)
            with torch.no_grad():
                base = base_head(xb)
                late = model(torch.cat((xb, gb), dim=1))
            err = float(torch.max(torch.abs(base - late)))
            max_error = max(max_error, err); count += len(xb)
            if err > 1e-7:
                raise ValueError(f"Equivalencia paso cero falló fold={fold}, seed={seed}: {err}")
    return {"status": "PASS", "fold_seed_checks": 15, "vectors_compared": count,
            "max_abs_error": max_error, "tolerance": 1e-7,
            "geometry_first_layer_weights_exactly_zero": True,
            "slice_weights_and_bias_identical_to_c1_initialization": True}


def fit_fold_scalers(data: pd.DataFrame, folds: pd.DataFrame) -> tuple[dict[int, StandardScaler], dict[int, float], dict[int, Any]]:
    """Ajusta scaler geométrico y deriva escala residual solo de inner-train."""
    scalers: dict[int, StandardScaler] = {}; scales: dict[int, float] = {}; ridge_models: dict[int, Any] = {}
    ridge_manifest = json.loads((RIDGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if ridge_manifest.get("status") != "PASS" or ridge_manifest.get("historical_test_accessed") is not False:
        raise ValueError("Manifiesto Ridge no certifica PASS y test histórico cerrado")
    ridge_model_hashes = json.loads((BASE / "config.yaml").read_text(encoding="utf-8"))["ridge"]["fold_model_sha256"]
    geometry = data[GEOM_COLS].to_numpy(dtype=np.float64)
    for fold in range(5):
        role = folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group").set_index("composition_group")
        tr = sorted(role.index[role.inner_role.eq("inner_train")]); va = sorted(role.index[role.inner_role.eq("inner_validation")])
        out = sorted(role.index[role.outer_role.eq("outer_validation")])
        if (len(tr), len(va), len(out)) != (45, 11, 14) or set(tr) & set(va) or set(tr + va) & set(out):
            raise ValueError(f"Split inválido en fold {fold}")
        train_idx = np.flatnonzero(data.composition_id.isin(tr).to_numpy())
        if len(train_idx) != 135:
            raise ValueError(f"Scaler fold {fold}: se esperaban 135 estructuras inner-train")
        scaler = StandardScaler().fit(geometry[train_idx])
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0):
            raise ValueError(f"Scaler inválido fold {fold}")
        scalers[fold] = scaler
        energy_means = data.groupby("composition_id").energy_dft_ev_atom.mean()
        residuals = data[data.composition_id.isin(tr)].energy_dft_ev_atom - data[data.composition_id.isin(tr)].composition_id.map(energy_means)
        scales[fold] = float(residuals.std(ddof=0))
        ridge_path = RIDGE_DIR / "models" / f"ridge_composicional_fold_{fold}.joblib"
        expected = ridge_model_hashes[str(fold)]
        if sha256(ridge_path) != expected:
            raise ValueError(f"Ridge fold {fold} SHA no coincide con MatterGPT F0")
        ridge_models[fold] = joblib.load(ridge_path)
        if not np.isfinite(scales[fold]) or scales[fold] <= 0:
            raise ValueError(f"Escala residual inválida fold {fold}")
    return scalers, scales, ridge_models


def train_one(data: pd.DataFrame, folds: pd.DataFrame, X: np.ndarray, G: np.ndarray,
              fold: int, seed: int, residual_scale: float, scaler: StandardScaler,
              device: torch.device) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, MLP8Geometry]:
    """Entrena MLP8(775→8→1) con early stopping interno y predice 14 fórmulas."""
    fr = folds[folds.outer_fold.eq(fold)]
    roles = fr.drop_duplicates("composition_group").set_index("composition_group")
    train_formulas = sorted(roles.index[roles.inner_role.eq("inner_train")])
    val_formulas = sorted(roles.index[roles.inner_role.eq("inner_validation")])
    outer_formulas = sorted(fr.loc[fr.outer_role.eq("outer_validation"), "composition_group"].unique())
    groups = {f: np.flatnonzero(data.composition_id.to_numpy() == f) for f in train_formulas + val_formulas + outer_formulas}
    rng = np.random.default_rng(seed)
    idx = {f: torch.as_tensor(np.concatenate((X[v], G[v]), axis=1), dtype=torch.float32, device=device) for f, v in groups.items()}
    y = {f: torch.as_tensor(data.energy_dft_ev_atom.to_numpy(float)[v], dtype=torch.float32, device=device) for f, v in groups.items()}
    target = {f: values - values.mean() for f, values in y.items()}
    torch.manual_seed(seed); np.random.seed(seed)
    model, _ = paired_head(seed, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = math.inf; best_epoch = 0; best_state: dict[str, torch.Tensor] | None = None
    stale = 0; history: list[dict[str, Any]] = []
    initial_geom = model.net[0].weight[:, 768:].detach().clone()
    for epoch in range(1, 201):
        model.train(); order = np.asarray(train_formulas)[rng.permutation(len(train_formulas))]
        batch_losses: list[float] = []
        for start in range(0, len(order), 8):
            batch = list(order[start:start + 8])
            xb = torch.stack([idx[f] for f in batch]); yb = torch.stack([target[f] for f in batch])
            raw = model(xb); centered = raw - raw.mean(dim=1, keepdim=True)
            loss = F.smooth_l1_loss(centered / residual_scale, yb / residual_scale, beta=1.0)
            if not torch.isfinite(loss): raise FloatingPointError(f"SmoothL1 no finita fold={fold} seed={seed}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Gradientes MLP8 ausentes/no finitos fold={fold} seed={seed}")
            geom_grad = model.net[0].weight.grad[:, 768:]
            if not torch.isfinite(geom_grad).all(): raise ValueError("Gradientes geométricos no finitos")
            optimizer.step(); batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vy = torch.stack([target[f] for f in val_formulas]); vr = model(torch.stack([idx[f] for f in val_formulas]))
            vc = vr - vr.mean(dim=1, keepdim=True)
            val_loss = float(F.smooth_l1_loss(vc / residual_scale, vy / residual_scale, beta=1.0).cpu())
        history.append({"fold": fold, "seed": seed, "epoch": epoch, "train_minibatch_smoothl1": float(np.mean(batch_losses)),
                        "inner_validation_full_smoothl1": val_loss, "optimizer_steps": len(batch_losses)})
        if val_loss < best_loss - 1e-12:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 20: break
    if best_state is None: raise RuntimeError("No se obtuvo checkpoint seleccionado por validación")
    model.load_state_dict(best_state); model.eval()
    if torch.equal(initial_geom.cpu(), model.net[0].weight[:, 768:].detach().cpu()):
        raise ValueError(f"Pesos geométricos permanecen en cero fold={fold}, seed={seed}")
    output: list[float] = []
    with torch.no_grad():
        for formula in outer_formulas:
            raw = model(idx[formula]); centered = raw - raw.mean()
            if abs(float(centered.sum().cpu())) > 1e-6: raise ValueError(f"Residuo sin centrar: {formula}")
            output.extend(centered.cpu().numpy().astype(float).tolist())
    record = {"fold": fold, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
              "best_inner_validation_smoothl1": best_loss, "residual_scale_ev_atom": residual_scale,
              "train_compositions": 45, "validation_compositions": 11, "outer_compositions": 14,
              "head_parameters": sum(p.numel() for p in model.parameters()), "backbone_trainable_parameters": 0,
              "geometry_weight_norm": float(torch.linalg.vector_norm(model.net[0].weight[:, 768:]).item()),
              "geometry_weight_abs_by_feature": torch.sum(torch.abs(model.net[0].weight[:, 768:]), dim=0).cpu().tolist(),
              "elapsed_seconds": None}
    checkpoint_path = OUT / "checkpoints" / f"fold_{fold}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "fold": fold, "seed": seed, "best_epoch": best_epoch,
                "residual_scale_ev_atom": residual_scale, "input_dim": 775,
                "geometry_descriptors": GEOM_COLS, "geometry_scaler_mean": scaler.mean_.tolist(),
                "geometry_scaler_scale": scaler.scale_.tolist()}, checkpoint_path)
    return np.asarray(output), record, pd.DataFrame(history), model


def diagnostic_predictions(data: pd.DataFrame, folds: pd.DataFrame, X: np.ndarray, G_by_fold: dict[int, np.ndarray],
                           runs: dict[tuple[int, int], MLP8Geometry], ridge: dict[int, Any], mg: Any) -> pd.DataFrame:
    """Calcula salidas outer con geometría original, cero y permutada por triplete."""
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        formulas = sorted(folds[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation")].composition_group.unique())
        for seed in SEEDS:
            model = runs[(fold, seed)].eval()
            for formula in formulas:
                ids = np.flatnonzero(data.composition_id.to_numpy() == formula)
                x = X[ids]; geom = G_by_fold[fold][ids]
                rng = np.random.default_rng(920261 + fold * 100 + seed)
                perm = geom[rng.permutation(len(geom))]
                preds = {}
                with torch.no_grad():
                    for name, g in (("original", geom), ("cero", np.zeros_like(geom)), ("permutada", perm)):
                        xt = torch.as_tensor(np.concatenate((x, g), axis=1), dtype=torch.float32)
                        raw = model(xt); preds[name] = (raw - raw.mean()).cpu().numpy()
                for row_i, source in enumerate(data.iloc[ids].itertuples()):
                    for condition, values in preds.items():
                        rows.append({"fold": fold, "seed": seed, "composition_id": formula,
                                     "structure_id": source.structure_id, "family_report": source.family_report,
                                     "diagnostic": condition, "residual_pred_ev_atom": float(values[row_i])})
    return pd.DataFrame(rows)


def write_geometry_manifest(data: pd.DataFrame, folds: pd.DataFrame, scalers: dict[int, StandardScaler],
                            geometry_manifest: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    """Registra geometría, source ID y escalado fold específico sin target/test leakage."""
    output = geometry_manifest.copy()
    roles = folds[["structure_id", "outer_fold", "inner_role", "outer_role"]]
    assigned = data[["structure_id", "outer_fold"]].merge(roles, on=["structure_id", "outer_fold"], validate="one_to_one")
    output = output.merge(assigned, on="structure_id", validate="one_to_one").set_index("structure_id")
    output["scaler_fit_population"] = output.outer_fold.map({f: "45 fórmulas / 135 estructuras inner-train" for f in range(5)})
    for f in range(5):
        m = output.outer_fold.eq(f)
        transformed = scalers[f].transform(output.loc[m, GEOM_COLS].to_numpy(float))
        output.loc[m, [f"scaled_{col}" for col in GEOM_COLS]] = transformed
        output.loc[m, "scaler_mean_json"] = json.dumps(scalers[f].mean_.tolist())
        output.loc[m, "scaler_scale_json"] = json.dumps(scalers[f].scale_.tolist())
    output = output.reset_index()
    output.to_csv(OUT / "geometry_manifest.csv", index=False)
    return output


def make_pairs(pred: pd.DataFrame, mg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula métricas por fórmula-semilla y pares con la auditoría F0 corregida."""
    calc = mg.prediction_metrics(pred.rename(columns={"family_report": "family"}))
    return pd.DataFrame(calc["by_composition_seed"]), pd.DataFrame(calc["pairs"])


def bootstrap_primary(metric: pd.DataFrame, contrasts: list[tuple[str, str]], n_boot: int) -> pd.DataFrame:
    """Bootstrap apareado A por fórmula con semillas promediadas y 10 000 réplicas."""
    rng = np.random.default_rng(20261030)
    avg = metric.groupby(["model", "composition_id"], as_index=False).mae_delta_mev_atom.mean()
    piv = avg.pivot(index="composition_id", columns="model", values="mae_delta_mev_atom")
    if len(piv) != 70: raise ValueError("Bootstrap no contiene 70 fórmulas completas")
    result = []
    for left, right in contrasts:
        d = (piv[left] - piv[right]).to_numpy(float)
        samples = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        result.append({"estimand": "A: MAE de error absoluto de diferencias firmadas ΔE entre los 3 pares", "contrast": f"{left} − {right}",
                       "estimate_mev_atom": float(d.mean()), "ci95_low_mev_atom": float(np.quantile(samples, .025)),
                       "ci95_high_mev_atom": float(np.quantile(samples, .975)), "n_compositions": 70,
                       "bootstrap_replicates": n_boot, "seed": 20261030,
                       "negative_favors": left})
    return pd.DataFrame(result)


def regression_stats(pred: pd.DataFrame) -> dict[str, Any]:
    """Obtiene OLS con intercepto usando todas las observaciones seed-averaged."""
    avg = pred.groupby(["model", "structure_id", "composition_id"], as_index=False).agg(
        energy_dft_ev_atom=("energy_dft_ev_atom", "mean"), energy_pred_ev_atom=("energy_pred_ev_atom", "mean"))
    result: dict[str, Any] = {}
    for name, frame in avg.groupby("model"):
        dmean = frame.groupby("composition_id").energy_dft_ev_atom.transform("mean")
        pmean = frame.groupby("composition_id").energy_pred_ev_atom.transform("mean")
        dx = frame.energy_dft_ev_atom.to_numpy(float) - frame.groupby("composition_id").energy_dft_ev_atom.transform("min").to_numpy(float)
        dy = frame.energy_pred_ev_atom.to_numpy(float) - frame.loc[frame.groupby("composition_id").energy_dft_ev_atom.idxmin()].set_index("composition_id").energy_pred_ev_atom.reindex(frame.composition_id).to_numpy(float)
        rx = (frame.energy_dft_ev_atom.to_numpy(float) - dmean.to_numpy(float)) * 1000
        ry = (frame.energy_pred_ev_atom.to_numpy(float) - pmean.to_numpy(float)) * 1000
        x = dx * 1000; y = dy * 1000
        def fit(a: np.ndarray, b: np.ndarray, unit: str) -> dict[str, Any]:
            model = LinearRegression().fit(a.reshape(-1, 1), b)
            return {"N": int(len(a)), "slope": float(model.coef_[0]), "intercept": float(model.intercept_),
                    "intercept_units": unit, "r2_ols": None if np.var(b) == 0 else float(r2_score(b, model.predict(a.reshape(-1, 1)))),
                    "mae_vs_dft": float(np.mean(np.abs(b - a))), "mae_units": unit,
                    "population": "210 estructuras OOF, media de 3 semillas por estructura"}
        result[name] = {"delta_energy_dft_anchored": {**fit(x, y, "meV/átomo"), "definition": "E_i−E_g, con g = mínimo DFT de la misma fórmula en ambos ejes"},
                        "intraformula_residual": {**fit(rx, ry, "meV/átomo"), "definition": "E_i menos media del triplete de su fórmula"},
                        "absolute_energy": {**fit(frame.energy_dft_ev_atom.to_numpy(float), frame.energy_pred_ev_atom.to_numpy(float), "eV/átomo")}}
    return result


def aggregate_metrics(metric: pd.DataFrame, pairs: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    """Resume métricas solicitadas, denominadores y regresiones OOF."""
    out: dict[str, Any] = {}
    for model, g in metric.groupby("model"):
        def mean(col: str) -> float | None:
            value = pd.to_numeric(g[col], errors="coerce").mean()
            return float(value) if pd.notna(value) and np.isfinite(value) else None
        p = pairs[(pairs.model == model) & pairs.eligible_abs_dft_gt_5]
        eligible = g[g.ground_state_eligible.eq(1)]
        out[model] = {"n_compositions_seed": int(len(g)), "n_compositions": int(g.composition_id.nunique()),
                      "mae_delta_pairs_mev_atom": mean("mae_delta_mev_atom"), "rmse_delta_pairs_mev_atom": float(np.sqrt(np.mean((pairs.loc[pairs.model.eq(model), "delta_pred_mev_atom"] - pairs.loc[pairs.model.eq(model), "delta_dft_mev_atom"]) ** 2))),
                      "mae_delta_dft_anchored_mev_atom": mean("mae_delta_anchored_all3_mev_atom"),
                      "mae_energy_ev_atom": mean("mae_energy_ev_atom"), "rmse_energy_ev_atom": mean("rmse_energy_ev_atom"),
                      "mae_residual_mev_atom": mean("mae_residual_mev_atom"),
                      "ground_state_accuracy_exact": float(eligible.ground_state_exact.mean()) if len(eligible) else None,
                      "ground_state_denominator_cases": int(len(eligible)), "ground_state_unresolved_cases": int(g.ground_state_unresolved_prediction.sum()),
                      "pairwise_accuracy_eligible_including_unresolved": float(p.correct.mean()) if len(p) else None,
                      "pairwise_accuracy_resolved_only": float(p.loc[~p.unresolved_abs_pred_le_1, "correct"].mean()) if (~p.unresolved_abs_pred_le_1).any() else None,
                      "unresolved_fraction_eligible_pairs": float(p.unresolved_abs_pred_le_1.mean()) if len(p) else None,
                      "eligible_pairs": int(len(p)), "spearman_mean_by_composition": mean("spearman"), "kendall_mean_by_composition": mean("kendall")}
    # Errores por polimorfo en unidades por estructura, semillas promediadas por ID.
    avg = pred.groupby(["model", "structure_id", "composition_id", "family_report"], as_index=False).agg(
        dft=("energy_dft_ev_atom", "mean"), pred=("energy_pred_ev_atom", "mean"))
    avg["abs_energy_error_mev_atom"] = np.abs(avg.pred - avg.dft) * 1000
    avg["residual_abs_error_mev_atom"] = (avg.pred - avg.groupby(["model", "composition_id"]).pred.transform("mean") - avg.dft + avg.groupby(["model", "composition_id"]).dft.transform("mean")).abs() * 1000
    by_family = avg.groupby(["model", "family_report"], as_index=False).agg(n_structures=("structure_id", "nunique"),
        mae_energy_mev_atom=("abs_energy_error_mev_atom", "mean"), mae_residual_mev_atom=("residual_abs_error_mev_atom", "mean"))
    out["by_polymorph"] = by_family.to_dict("records")
    out["ols_regression"] = regression_stats(pred)
    # Bins por |ΔE_DFT|, con unidad observacional = par intrafórmula seed-averaged.
    dft_pairs = pairs.groupby(["model", "composition_id", "structure_i", "structure_j"], as_index=False).agg(
        delta_dft_mev_atom=("delta_dft_mev_atom", "mean"), delta_pred_mev_atom=("delta_pred_mev_atom", "mean"))
    dft_pairs["abs_gap"] = dft_pairs.delta_dft_mev_atom.abs()
    edges = [0, 5, 10, 25, 50, 100, np.inf]
    labels = ["0–5", "5–10", "10–25", "25–50", "50–100", ">100"]
    dft_pairs["gap_bin"] = pd.cut(dft_pairs.abs_gap, bins=edges, labels=labels, right=True, include_lowest=True)
    dft_pairs["abs_error"] = np.abs(dft_pairs.delta_pred_mev_atom - dft_pairs.delta_dft_mev_atom)
    out["gap_bins"] = dft_pairs.groupby(["model", "gap_bin"], observed=False).agg(n_pairs=("abs_error", "size"), mae_delta_mev_atom=("abs_error", "mean")).reset_index().to_dict("records")
    out["definitions"] = {"primary": "MAE de los errores absolutos de ΔE firmado en los tres pares por fórmula; semillas promediadas dentro de fórmula antes de bootstrap",
        "delta_ground_state": "ΔE_i = E_i − E_g, g = menor energía DFT de la misma fórmula; ancla común en referencia y predicción; negativo = inversión",
        "energy": "eV/átomo; residuo centrado por triplete; no se ajusta el Ridge",
        "pair_eligibility": "|ΔE_DFT| > 5 meV/átomo; predicciones con |ΔE_pred| ≤ 1 cuentan incorrectas en denominador elegible",
        "ground_state_eligibility": "DFT gap al segundo menor > 1 meV/átomo; empate predicho ≤1 meV/átomo no exacto",
        "sample_unit": "210 estructuras, 70 fórmulas, 3 polimorfos; 15 corridas; métricas por fórmula y semillas",
        "outer": "OOF histórico; no se abrió el test histórico"}
    return out


def make_figures(pred: pd.DataFrame, metric: pd.DataFrame, bootstrap: pd.DataFrame, history: pd.DataFrame,
                 diagnostics: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Genera ocho PNG científicos en español con muestras, unidades y referencias."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IMAGES.mkdir(parents=True, exist_ok=True)
    colors = {"Ridge_only": "#555b66", "MatterGPT_F0": "#007c91", "MatterGPT_F0_geometry": "#00a884",
              "MatGPT_F0": "#b05a21", "MatGPT_F0_geometry": "#e3a000"}
    names = ["MatterGPT_F0", "MatterGPT_F0_geometry", "Ridge_only"]
    labels = ["MatterGPT: SLICES", "MatterGPT: SLICES + geometría", "Ridge composicional"]
    note = "OOF: 210 estructuras estáticas, 70 fórmulas y 5 folds; 3 semillas promediadas dentro de fórmula. MAE E/N = promedio de errores de las 3 estructuras; MAE residual = error tras centrar cada triplete; MAE ΔE = error en diferencias de energía de los 3 pares intrafórmula. Geometría: 7 métricas de celda final. Test histórico cerrado."
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 110})
    # 1 métricas comparables.
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [[metrics[m][k] for m in names] for k in ("mae_energy_ev_atom", "mae_residual_mev_atom", "mae_delta_pairs_mev_atom")]
    x = np.arange(3); width = .24
    for j, (series, label, unit) in enumerate(zip(vals, ["MAE E/N", "MAE residuo", "MAE ΔE por pares"], ["eV/átomo", "meV/átomo", "meV/átomo"])):
        # Se convierten las tres series a meV/átomo para eje común.
        normalized = [v * 1000 if j == 0 else v for v in series]
        shades = [["#7bb5c0", "#83cbb4", "#aeb2b8"], ["#368b9a", "#36a989", "#7e8791"], ["#005968", "#00795f", "#555b66"]][j]
        legend_unit = "meV/átomo (E/N ×1000)" if j == 0 else unit
        ax.bar(x + (j-1)*width, normalized, width, label=f"{label} ({legend_unit})", color=shades)
    ax.set_xticks(x, labels); ax.set_ylabel("Error absoluto medio (meV/átomo)"); ax.set_title("MatterGPT en ABS₃: SLICES frente a fusión tardía de geometría")
    ax.legend(ncol=3, fontsize=8); fig.text(.02, .01, note + " MAE E/N convertida de eV a meV para compartir eje; valores originales en metrics.json.", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(IMAGES / "mattergpt_slices_vs_geometria_metricas.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 2 paired bootstrap.
    fig, ax = plt.subplots(figsize=(8, 4.4))
    chosen = bootstrap[bootstrap.contrast.isin(["MatterGPT_F0_geometry − MatterGPT_F0", "MatterGPT_F0_geometry − Ridge_only"])]
    for i, row in enumerate(chosen.itertuples()):
        ax.errorbar(row.estimate_mev_atom, i, xerr=[[row.estimate_mev_atom-row.ci95_low_mev_atom],[row.ci95_high_mev_atom-row.estimate_mev_atom]], fmt="o", capsize=5, color="#00a884" if i == 0 else "#555b66")
    ax.axvline(0, color="black", ls="--", lw=1); ax.set_yticks(range(len(chosen)), ["Geometría − SLICES", "Geometría − Ridge"])
    ax.set_xlabel("Diferencia en MAE ΔE (meV/átomo); negativa favorece geometría")
    ax.set_title("ABS₃: bootstrap apareado por fórmula, IC95%")
    fig.text(.02, .01, "Estimando A; 70 fórmulas remuestreadas 10 000 veces; 3 semillas promediadas por fórmula; 3 pares por fórmula.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(IMAGES / "mattergpt_geometria_bootstrap.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 3 parity DeltaE; slope OLS all points.
    avg = pred.groupby(["model", "structure_id", "composition_id", "family_report"], as_index=False).agg(dft=("energy_dft_ev_atom", "mean"), pred=("energy_pred_ev_atom", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for ax, name, label in zip(axes, names, labels):
        g = avg[avg.model.eq(name)].copy(); dmin = g.groupby("composition_id").dft.transform("min")
        pmin = g.loc[g.groupby("composition_id").dft.idxmin()].set_index("composition_id").pred
        xx = (g.dft - dmin).to_numpy() * 1000; yy = (g.pred - g.composition_id.map(pmin)).to_numpy() * 1000
        fit = LinearRegression().fit(xx.reshape(-1, 1), yy); grid = np.linspace(min(xx.min(), yy.min()), max(xx.max(), yy.max()), 100)
        ax.scatter(xx, yy, s=14, alpha=.65, c=[colors[name]])
        ax.plot(grid, grid, color="black", ls="--", lw=1, label="Ideal y=x")
        ax.plot(grid, fit.predict(grid.reshape(-1, 1)), color="#c43c39", lw=1.4, label="OLS")
        ax.set_title(label); ax.set_xlabel("ΔE DFT = Eᵢ−Eₘᵢₙ,DFT (meV/átomo)")
        r2_value = None if np.var(yy) == 0 else float(r2_score(yy, fit.predict(xx.reshape(-1,1))))
        r2_label = "no definido" if r2_value is None else f"{r2_value:.2f}"
        ax.text(.03, .97, f"N=210\nm={fit.coef_[0]:.2f}; b={fit.intercept_:.1f} meV/átomo\nR²={r2_label}; MAE={np.mean(np.abs(yy-xx)):.1f} meV/átomo", transform=ax.transAxes, va="top", fontsize=7)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("ΔE predicha, ancla DFT común (meV/átomo)"); fig.suptitle("ABS₃: paridad de ΔE respecto al mínimo DFT de cada fórmula")
    fig.text(.02, .01, note + " Un punto = estructura, promediada en 3 semillas. ΔE predicha negativa indica inversión del orden DFT.", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, .09, 1, .94)); fig.savefig(IMAGES / "mattergpt_geometria_paridad_deltaE.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 4 ranking.
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(names)); w = .34
    for j, key in enumerate(("ground_state_accuracy_exact", "pairwise_accuracy_eligible_including_unresolved")):
        values = [metrics[n][key] for n in names]
        ax.bar(x+(j-.5)*w, values, w, label="Estado fundamental exacto" if j == 0 else "Exactitud pairwise (pares elegibles)", color="#007c91" if j == 0 else "#00a884")
    ax.set_xticks(x, labels, rotation=8); ax.set_ylim(0, 1); ax.set_ylabel("Exactitud (fracción)"); ax.set_title("ABS₃: ranking polimórfico OOF")
    ax.legend(fontsize=8); fig.text(.02, .01, "Barras = promedio de 15 corridas. GS: 201 casos fórmula-semilla elegibles (gap DFT >1 meV/átomo) por modelo; empate predicho ≤1 cuenta fallo. Pairwise: 591 pares elegibles por modelo (|ΔE_DFT|>5); no resueltos |ΔE_pred|≤1 cuentan incorrectos.", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, .10, 1, 1)); fig.savefig(IMAGES / "mattergpt_geometria_ranking.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 5 overfit curves.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, label, color in (("train_minibatch_smoothl1", "Minilotes de optimización inner-train", "#007c91"), ("inner_validation_full_smoothl1", "Validación completa inner-val", "#b05a21")):
        grp = history.groupby(["model", "epoch"])[col].mean().reset_index()
        for name, style in (("MatterGPT_F0", "--"), ("MatterGPT_F0_geometry", "-")):
            s = grp[grp.model.eq(name)]; ax.plot(s.epoch, s[col], linestyle=style, color=color, alpha=.85, label=f"{label} · {name.replace('_',' ')}")
    ax.set_xlabel("Época local; selección por menor pérdida inner-val"); ax.set_ylabel("SmoothL1 del residuo normalizado")
    ax.set_title("ABS₃: entrenamiento de MLP8 y validación interna"); ax.legend(fontsize=7, ncol=2)
    fig.text(.02, .01, "Media de las corridas que alcanzaron cada época; minilotes inner-train y validación completa de 11 fórmulas. El conteo por época está en historial_entrenamiento.csv.", fontsize=7)
    fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(IMAGES / "mattergpt_geometria_overfitting.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 6 polymorph errors.
    fam = pd.DataFrame(metrics["by_polymorph"]); fam = fam[fam.model.isin(names)]
    fig, ax = plt.subplots(figsize=(9, 4.7)); fam_order = ["Perovskite", "Hexagonal", "Needle-like"]; xx = np.arange(3); w=.25
    for j, name in enumerate(names):
        g=fam[fam.model.eq(name)].set_index("family_report").reindex(fam_order)
        ax.bar(xx+(j-1)*w,g.mae_residual_mev_atom,w,label=name.replace("_"," "),color=colors[name])
    ax.set_xticks(xx,fam_order); ax.set_ylabel("MAE residual intrafórmula (meV/átomo)"); ax.set_title("ABS₃: error residual por familia de polimorfo")
    ax.legend(fontsize=8); fig.text(.02,.01,"Unidad = estructura estática, media de 3 semillas; n=70 por familia y modelo (una por cada una de 70 fórmulas). Error residual = diferencia de energías tras centrar cada triplete; valores individuales en metrics.json.",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(IMAGES/"mattergpt_geometria_error_polimorfo.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 7 ablation.
    diag_pred = diagnostics.copy()
    # Attach labels from the main OOF prediction, retaining one condition's DFT and energy.
    labels_df = pred[pred.model.eq("MatterGPT_F0_geometry")][["fold","seed","structure_id","composition_id","energy_dft_ev_atom","family_report"]]
    diag_pred = diagnostics.merge(labels_df, on=["fold","seed","structure_id","composition_id","family_report"], validate="many_to_one")
    abl_rows=[]
    for (condition, formula, seed), g in diag_pred.groupby(["diagnostic","composition_id","seed"]):
        d=g.energy_dft_ev_atom.to_numpy(float); yp=g.residual_pred_ev_atom.to_numpy(float)
        err=np.mean([abs((yp[j]-yp[i]-(d[j]-d[i]))*1000) for i in range(3) for j in range(i+1,3)])
        abl_rows.append({"diagnostic":condition,"composition_id":formula,"seed":seed,"mae_delta_mev_atom":err})
    abl=pd.DataFrame(abl_rows); abl.to_csv(OUT/"ablation_metricas.csv",index=False)
    order=["original","cero","permutada"]; ablmean=abl.groupby("diagnostic").mae_delta_mev_atom.mean().reindex(order)
    fig, ax=plt.subplots(figsize=(7,4.4)); ax.bar(["Geometría original","Geometría = 0","Geometría permutada"],ablmean.values,color=["#00a884","#778899","#e3a000"])
    ax.set_ylabel("MAE ΔE firmado por pares (meV/átomo)"); ax.set_title("MatterGPT: diagnóstico de ablación de geometría")
    fig.text(.02,.01,"15 cabezas C2; outer OOF, 70 fórmulas y 3 semillas. Permutación reproducible dentro de cada triplete. Diagnóstico no causal; embeddings SLICES fijos.",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(IMAGES/"mattergpt_geometria_ablation.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    # 8 MatGPT versus MatterGPT, frozen F0 historical.
    fig, ax=plt.subplots(figsize=(8,4.7)); compare=["MatGPT_F0","MatGPT_F0_geometry","MatterGPT_F0","MatterGPT_F0_geometry"]
    lab=["MatGPT SLICES","MatGPT + geom.","MatterGPT SLICES","MatterGPT + geom."]
    vals=[metrics[n]["mae_delta_pairs_mev_atom"] for n in compare]
    ax.bar(lab,vals,color=[colors[n] for n in compare]); ax.set_ylabel("MAE ΔE firmado por pares (meV/átomo)")
    ax.set_title("ABS₃: referencia histórica de dos extractores congelados")
    fig.text(.02,.01,note+" Comparación externa no causal: difieren tokenización, arquitectura y condicionamiento.",fontsize=7,wrap=True)
    fig.tight_layout(rect=(0,.08,1,1)); fig.savefig(IMAGES/"matgpt_vs_mattergpt_geometria.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def data_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Helper tipado para mantener explícitas las columnas de unión."""
    return frame[["fold", "seed", "structure_id", "composition_id"]]


def architecture_diagram() -> None:
    """Guarda diagrama editable Mermaid y su render PNG científico."""
    source = """flowchart LR
  C[ABS₃] --> F[17 fracciones estequiométricas]
  F --> R[Ridge histórico congelado · scaler y α originales]
  R --> MU[μ_c]
  S[SLICES] --> M[MatterGPT congelado · prop = 1.0, 2.0]
  M --> H[FINAL_END tras ln_f · 768]
  G[Celda DFT estática final] --> GD[siete descriptores históricos]
  GD --> GS[StandardScaler · ajuste solo inner-train]
  H --> CAT[concat · 775]
  GS --> CAT
  CAT --> N[MLP8 · 775 → 8 → 1 · entrenable]
  N --> D[δ_c,i centrado por fórmula]
  MU --> ADD[μ_c + δ_c,i]
  D --> ADD
  ADD --> E[E_pred]
  E --> DE[ΔE con referencia DFT común]
  DE --> ST[Estabilidad polimórfica]
"""
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "diagrama_arquitectura.mmd").write_text(source, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        fig, ax = plt.subplots(figsize=(14, 7)); ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
        nodes={"ABS₃":(.25,5.35,1.25,.75,"#eeeeee"),"17 fracciones\nestequiométricas":(2.0,5.35,1.8,.75,"#eeeeee"),"Ridge histórico\ncongelado · 17 → 1":(4.35,5.35,2.05,.75,"#d7e8f4"),"μ_c":(6.95,5.35,1.1,.75,"#d7e8f4"),
               "SLICES":(.25,3.0,1.25,.75,"#eeeeee"),"MatterGPT congelado\npropiedad [1.0, 2.0]":(2.0,3.0,2.25,.75,"#f3d7d2"),"FINAL_END tras ln_f\n768 características":(4.85,3.0,1.95,.75,"#f3d7d2"),
               "Celda DFT estática final":(.25,.65,1.8,.75,"#eeeeee"),"Siete descriptores\nhistóricos":(2.55,.65,1.8,.75,"#d9edd9"),"StandardScaler\nfit solo inner-train":(4.85,.65,2.0,.75,"#d9edd9"),
               "Concatenar\n768 + 7 = 775":(7.45,2.35,1.55,.85,"#fff0c2"),"MLP8 residual\n775 → 8 → 1":(9.5,2.35,1.7,.85,"#fff0c2"),"δ_c,i\ncentrado":(11.7,2.35,1.15,.85,"#fff0c2"),"μ_c + δ_c,i\nE_pred → ΔE\nestabilidad":(11.7,4.05,1.75,1.0,"#e4dcf4")}
        centers={}; bounds={}
        for label,(x,y,w,h,color) in nodes.items():
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",facecolor=color,edgecolor="#444")); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8); centers[label]=(x+w/2,y+h/2); bounds[label]=(w/2,h/2)
        arrows=[("ABS₃","17 fracciones\nestequiométricas"),("17 fracciones\nestequiométricas","Ridge histórico\ncongelado · 17 → 1"),("Ridge histórico\ncongelado · 17 → 1","μ_c"),("SLICES","MatterGPT congelado\npropiedad [1.0, 2.0]"),("MatterGPT congelado\npropiedad [1.0, 2.0]","FINAL_END tras ln_f\n768 características"),("FINAL_END tras ln_f\n768 características","Concatenar\n768 + 7 = 775"),("Celda DFT estática final","Siete descriptores\nhistóricos"),("Siete descriptores\nhistóricos","StandardScaler\nfit solo inner-train"),("StandardScaler\nfit solo inner-train","Concatenar\n768 + 7 = 775"),("Concatenar\n768 + 7 = 775","MLP8 residual\n775 → 8 → 1"),("MLP8 residual\n775 → 8 → 1","δ_c,i\ncentrado"),("δ_c,i\ncentrado","μ_c + δ_c,i\nE_pred → ΔE\nestabilidad")]
        def connect(a: str, b: str, color: str = "#555", connection: str = "arc3,rad=0.0") -> None:
            """Conecta bordes de cajas para que ninguna flecha atraviese sus etiquetas."""
            ax0, ay0 = centers[a]; bx0, by0 = centers[b]; dx, dy = bx0-ax0, by0-ay0
            wa, ha = bounds[a]; wb, hb = bounds[b]
            ta = min(wa/abs(dx) if dx else float("inf"), ha/abs(dy) if dy else float("inf"))
            tb = min(wb/abs(dx) if dx else float("inf"), hb/abs(dy) if dy else float("inf"))
            start=(ax0+dx*ta, ay0+dy*ta); end=(bx0-dx*tb, by0-dy*tb)
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=13,color=color,connectionstyle=connection))
        for a,b in arrows: connect(a,b)
        # Ridge skips the fusion head and feeds the final energy reconstruction.
        connect("μ_c","μ_c + δ_c,i\nE_pred → ΔE\nestabilidad",color="#416b91",connection="angle3,angleA=0,angleB=90")
        ax.text(8.2,4.0,"Geometría entra después del Transformer",ha="center",fontsize=10,color="#216b47")
        ax.text(7.5,6.55,"Ridge congelado · MLP8 entrenable · geometría fuera de la atención",ha="center",fontsize=10,color="#444")
        ax.set_title("Fusión tardía MatterGPT + geometría de celda + Ridge estequiométrico",fontsize=14,pad=15)
        fig.tight_layout(); fig.savefig(IMAGES/"mattergpt_geometria_arquitectura.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    except Exception as exc:
        LOG.exception("architecture_png_failed; Mermaid remains editable: %s", exc)
        raise


def report_text(metrics: dict[str, Any], bootstrap: pd.DataFrame, preflight: dict[str, Any], records: list[dict[str, Any]], resources: dict[str, Any]) -> str:
    """Construye reporte autocontenido y separa hechos, resultados e inferencias."""
    c1=metrics["MatterGPT_F0"]; c2=metrics["MatterGPT_F0_geometry"]
    contrast=bootstrap.loc[bootstrap.contrast.eq("MatterGPT_F0_geometry − MatterGPT_F0")].iloc[0]
    sig = "el IC95% excluye cero" if contrast.ci95_low_mev_atom*contrast.ci95_high_mev_atom>0 else "el IC95% incluye cero"
    lines=["# MatterGPT congelado + SLICES + geometría: ABS₃", "",
        "## Hechos verificados", "",
        f"Se reutilizó la caché MatterGPT validada: {preflight['embedding_rows']} embeddings de dimensión {preflight['embedding_dim']}, checkpoint SHA-256 `{preflight['checkpoint_sha256']}`, tokenizador `{preflight['tokenizer_sha256']}`, propiedad fija `[1.0, 2.0]`; extracción histórica documentada tras `blocks → ln_f` en el último token `<`. No se ejecutó ni modificó el backbone.",
        f"La geometría histórica contiene {preflight['structures']} IDs, exclusivamente `gga-static_endpoint_cleaned`, SHA-256 `{preflight['geometry_sha256']}`. Los siete descriptores se recuperaron por `structure_id`: celda primitiva reducida de Niggli; a/b/c divididos por la raíz cúbica del volumen primitivo reducido; ángulos de esa celda; y volumen de celda original por átomo.",
        f"Tres de los 210 registros históricos no tienen hash de topología no vacío ({preflight['topology_hash_missing_entries_in_historical_source']}); la correspondencia se conserva mediante el ID estructural estable, fórmula, polimorfo, número de átomos y join uno-a-uno. El preflight original confirma la política pública `> SLICES <` y extracción tras `ln_f`, aunque el dataloader original de teacher-forcing no está disponible para verificar independientemente el token terminal durante el preentrenamiento.",
        "Se reutilizaron los cinco Ridge de 17 fracciones sin reentrenar, cinco folds agrupados por fórmula, 45 fórmulas inner-train, 11 inner-validation, 14 outer-evaluation y semillas 42, 123, 2026. El `StandardScaler` geométrico de cada fold se ajustó con 135 estructuras pertenecientes solo a las 45 fórmulas inner-train. Test histórico cerrado.",
        f"Control de inicialización: {preflight['initial_equivalence']['fold_seed_checks']} verificaciones fold-semilla sobre {preflight['initial_equivalence']['vectors_compared']} vectores; error absoluto máximo {preflight['initial_equivalence']['max_abs_error']:.3g}, tolerancia 1e-7. Las siete columnas geométricas comenzaron en cero y las cabezas MatterGPT de 768 dimensiones compartieron inicialización exacta.",
        "", "## Resultados experimentales", "",
        f"Métrica primaria A: error absoluto medio de diferencias ΔE firmadas en los tres pares por fórmula; seeds promediadas dentro de fórmula para bootstrap apareado de 10 000 réplicas (70 fórmulas). C2−C1 = **{contrast.estimate_mev_atom:.3f} meV/átomo** (IC95% {contrast.ci95_low_mev_atom:.3f}, {contrast.ci95_high_mev_atom:.3f}); {sig}. Un valor negativo favorece la geometría.",
        f"C1 SLICES-only: MAE ΔE {c1['mae_delta_pairs_mev_atom']:.3f}; MAE residual {c1['mae_residual_mev_atom']:.3f}; MAE E/N {c1['mae_energy_ev_atom']:.5f} eV/átomo; GS exacto {100*c1['ground_state_accuracy_exact']:.1f}%; exactitud pairwise elegible {100*c1['pairwise_accuracy_eligible_including_unresolved']:.1f}%.",
        f"C2 SLICES + geometría: MAE ΔE {c2['mae_delta_pairs_mev_atom']:.3f}; MAE residual {c2['mae_residual_mev_atom']:.3f}; MAE E/N {c2['mae_energy_ev_atom']:.5f} eV/átomo; GS exacto {100*c2['ground_state_accuracy_exact']:.1f}%; exactitud pairwise elegible {100*c2['pairwise_accuracy_eligible_including_unresolved']:.1f}%. Denominadores explícitos en `metrics.json`.",
        f"Bootstrap C2−Ridge: {bootstrap.loc[bootstrap.contrast.eq('MatterGPT_F0_geometry − Ridge_only'),'estimate_mev_atom'].iloc[0]:.3f} meV/átomo. Comparación histórica MatGPT F0 + geometría: MAE ΔE {metrics['MatGPT_F0_geometry']['mae_delta_pairs_mev_atom']:.3f} meV/átomo.",
        "", "## Respuestas, interpretación y límites", "",
        f"1–3. La geometría {'reduce' if c2['mae_delta_pairs_mev_atom'] < c1['mae_delta_pairs_mev_atom'] else 'no reduce'} el MAE ΔE de {c1['mae_delta_pairs_mev_atom']:.3f} a {c2['mae_delta_pairs_mev_atom']:.3f} meV/átomo y el MAE residual de {c1['mae_residual_mev_atom']:.3f} a {c2['mae_residual_mev_atom']:.3f} meV/átomo. {sig} en el contraste primario.",
        f"4–5. Ranking pairwise: {100*c1['pairwise_accuracy_eligible_including_unresolved']:.1f}% → {100*c2['pairwise_accuracy_eligible_including_unresolved']:.1f}%; GS exacto: {100*c1['ground_state_accuracy_exact']:.1f}% → {100*c2['ground_state_accuracy_exact']:.1f}%. Ver denominadores y no resueltos en `metrics.json`.",
        "6. Errores por Perovskite, Hexagonal y Needle-like están tabulados en `metrics.json`; la unidad es estructura estática, media de semillas.",
        "7. La ablación a geometría cero y la permutación dentro de triplete diagnostican dependencia predictiva bajo intervención sintética; no son prueba de causalidad física. La norma y suma absoluta de pesos geométricos por descriptor se guardan en el historial/checkpoints y `weight_diagnostics.csv`.",
        "8. MatGPT F0 congelado se compara en mismos folds, IDs, targets y semillas; diferencias entre extractores no son efecto causal del preentrenamiento, pues también difieren arquitectura, vocabulario y condicionamiento.",
        "9. Las curvas muestran pérdida de minibatches y validación completa. Early stopping elige el menor SmoothL1 normalizado de 11 fórmulas; no se usó el outer para selección.",
        "10. Los resultados solo permiten decidir si conviene estudiar fine-tuning parcial. No demuestran que este sea útil; hace falta experimento predefinido separado y control de leakage.",
        "", "### Definiciones", "",
        "`ΔE_DFT,i = E_DFT,i − E_DFT,g`; `ΔE_pred,i = E_pred,i − E_pred,g`; g es el mínimo DFT de la misma fórmula para ambos ejes. Predicción negativa significa inversión de orden DFT. Pareja elegible si |ΔE_DFT| > 5 meV/átomo; |ΔE_pred| ≤ 1 es no resuelta y cuenta incorrecta. Fórmula elegible para estado fundamental si su gap DFT al segundo mínimo >1 meV/átomo; empate predicho ≤1 es fallo. No se mezclan composiciones.",
        "Los intervalos se comparan por estructura de composición (70 unidades bootstrap); las tres semillas se promedian dentro de fórmula. Todas las observaciones outer OOF se reportan como validación cruzada de desarrollo, no como test ciego. Bootstrap primario sin ajuste multiplicidad; contraste C2−C1 predefinido.",
        "", "## Recursos", "",
        f"Duración de pared {resources['wall_seconds']:.1f} s; entrenamiento acumulado de las 15 cabezas C2 {resources['mlp_training_seconds']:.1f} s; desglose C1/C2 y controles MatGPT en `resources.json`. CPU usuario/sistema {resources['process_user_cpu_seconds']:.1f}/{resources['process_system_cpu_seconds']:.1f} s; pico RSS {resources['peak_rss_mib']:.1f} MiB; dispositivo CPU; versiones Python {resources['python_version']}, PyTorch {resources['torch_version']}, NumPy {resources['numpy_version']}, scikit-learn {resources['scikit_learn_version']}. Se guardaron 15 checkpoints MLP8 compactos; MatterGPT no se duplicó.", "",
        "Figuras PNG 300 dpi en `images/mattergpt_abs3_geometry_late_fusion/`. Diagrama editable: `reports/MatterGPT/mattergpt_abs3_geometry_late_fusion/diagrama_arquitectura.mmd`.", ""]
    return "\n".join(lines)


def run(n_boot: int) -> None:
    """Audita fuentes, entrena quince cabezas, evalúa OOF y exporta resultados."""
    started = PROCESS_WALL_START; cpu0 = PROCESS_CPU_START
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True); IMAGES.mkdir(parents=True, exist_ok=True)
    global mg_module
    mg_module = load_baseline_module()
    data, folds, embeddings, geometry_manifest, audit = load_data(mg_module)
    # Mantiene el orden (ID) compartido entre estructuras, embeddings y geometría.
    G = data[GEOM_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(G).all(): raise ValueError("Geometría no finita")
    initial = audit_initial_equivalence(embeddings, G, data, folds)
    scalers, scales, ridge_models = fit_fold_scalers(data, folds)
    audit["initial_equivalence"] = initial
    audit["ridge_sha256"] = {str(f): sha256(RIDGE_DIR / "models" / f"ridge_composicional_fold_{f}.joblib") for f in range(5)}
    audit["split_counts"] = {str(f): {"inner_train":45,"inner_validation":11,"outer_evaluation":14} for f in range(5)}
    if len(data) != 210 or data.composition_id.nunique() != 70: raise ValueError("Tamaños de dataset históricos no coinciden")
    # Smoke test de C2: finitud de salida, gradientes y actualización de las siete columnas.
    smoke, _ = paired_head(42, torch.device("cpu")); smoke.train()
    xb = torch.as_tensor(np.concatenate((embeddings[:3], scalers[0].transform(G[:3])), axis=1), dtype=torch.float32)
    yb = torch.as_tensor(data.energy_dft_ev_atom.to_numpy(float)[:3], dtype=torch.float32); yb -= yb.mean()
    raw = smoke(xb); cent = raw - raw.mean(); loss = F.smooth_l1_loss(cent / scales[0], yb / scales[0], beta=1.0)
    loss.backward(); geom_grad = smoke.net[0].weight.grad[:,768:]
    if not torch.isfinite(loss) or not torch.isfinite(geom_grad).all() or torch.linalg.vector_norm(geom_grad).item() <= 0:
        raise ValueError("Smoke C2 falla: pérdida/gradiente geométrico no finito o nulo")
    smoke_result={"status":"PASS","loss_finite":True,"all_gradients_finite":all(p.grad is not None and torch.isfinite(p.grad).all() for p in smoke.parameters()),
                  "geometry_gradients_nonzero_at_initialization":True,"geometry_init_zero":True,"head_parameters":sum(p.numel() for p in smoke.parameters()),
                  "output_centered_by_triplet":abs(float(cent.detach().sum()))<1e-7}
    audit["mlp8_geometry_smoke"] = smoke_result
    audit["checks"] = {"checkpoint_y_tokenizador_SHA256_correctos": True, "property_vector_1_0_2_0": True,
        "embeddings_210_por_768_posteriores_a_ln_f": True, "policy_terminal_y_ultimo_token_especial_validada": True,
        "IDs_formula_familia_slices_cache_coinciden": True, "geometria_210_por_7_fuente_estatica_final": True,
        "sin_nan_ni_inf": True, "scaler_inner_train_45_formulas_por_fold": True, "ridge_17_fracciones_sin_reentrenar": True,
        "mlp8_775_entradas": True, "pesos_geometricos_iniciales_cero": True, "equivalencia_paso_cero": initial["status"] == "PASS",
        "gradientes_solo_en_mlp8": True, "backbone_sin_forward_durante_entrenamiento": True,
        "targets_residuales_y_smoothl1_historicos": True, "splits_agrupados_por_formula_45_11_14": True,
        "test_historico_cerrado": True, "checkpoint_reload_verificado_al_final": "pending"}
    audit["status"] = "PASS"
    write_json(OUT / "preflight.json", audit)
    geometry_out = write_geometry_manifest(data, folds, scalers, geometry_manifest, audit)
    # Historical M1 MatterGPT SLICES-only, and Ridge-only from its stored predictions.
    base_pred = pd.read_csv(BASE / "predicciones_oof_por_estructura.csv")
    c1 = base_pred[(base_pred.model.eq("MatterGPT_F0")) & base_pred.seed.isin(SEEDS)].copy()
    if len(c1) != 630 or c1.structure_id.nunique() != 210: raise ValueError("Predicciones MatterGPT C1 no son 630 OOF filas")
    c1["family_report"] = c1.family.map({"dist_perovskite":"Perovskite","hexagonal":"Hexagonal","needle":"Needle-like"})
    c1["model"] = "MatterGPT_F0"
    # Reutiliza control Ridge desde las corridas históricas de MatterGPT, verificando triple repetición exacta.
    ridge_rows=[]
    for fold in range(5):
        fr=folds[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation")]
        formulas=sorted(fr.composition_group.unique()); model=ridge_models[fold]
        means=dict(zip(formulas,model.predict(mg_module.formula_features(formulas))))
        for row in data[data.composition_id.isin(formulas)].itertuples():
            for seed in SEEDS:
                ridge_rows.append({"model":"Ridge_only","seed":seed,"fold":fold,"structure_id":row.structure_id,"composition_id":row.composition_id,
                    "family_report":row.family_report,"n_atoms":row.n_atoms,"energy_dft_ev_atom":row.energy_dft_ev_atom,
                    "ridge_mean_ev_atom":means[row.composition_id],"residual_pred_ev_atom":0.0,"energy_pred_ev_atom":means[row.composition_id]})
    ridge_pred=pd.DataFrame(ridge_rows)
    # Históricos MatGPT F0 con geometría y sin ella: misma cohorte, folds, target y semillas.
    hpred=pd.read_csv(HIST/"predicciones_oof_por_estructura.csv")
    hpred=hpred[hpred.seed.isin(SEEDS)].copy()
    expected_conditions={"slices_only":"MatGPT_F0","slices_plus_cell_geometry":"MatGPT_F0_geometry"}
    hist_parts=[]
    for condition, model_name in expected_conditions.items():
        f=hpred[hpred.condition.eq(condition)].copy()
        if len(f)!=630: raise ValueError(f"MatGPT histórico {condition}: se esperaban 630 filas")
        f=f.rename(columns={"polymorph_family":"family_report","energy_per_atom_dft_ev":"energy_dft_ev_atom","energy_per_atom_pred_ev":"energy_pred_ev_atom","ridge_mean_ev_atom":"ridge_mean_ev_atom","residual_ev_atom":"residual_pred_ev_atom"})
        f["family_report"]=f.family_report.map({"dist_perovskite":"Perovskite","hexagonal":"Hexagonal","needle":"Needle-like"})
        f["model"]=model_name
        hist_parts.append(f[["model","seed","fold","structure_id","composition_id","family_report","n_atoms","energy_dft_ev_atom","ridge_mean_ev_atom","residual_pred_ev_atom","energy_pred_ev_atom"]])
    # Confirma labels/IDs/folds y no admite filas ajenas a las 210 estructuras de desarrollo.
    target_map=data.set_index("structure_id").energy_dft_ev_atom; fold_map=data.set_index("structure_id").outer_fold
    for name, frame in [("MatterGPT_C1",c1),("MatGPT_F0",hist_parts[0]),("MatGPT_F0_geom",hist_parts[1])]:
        if set(frame.structure_id)!=set(data.structure_id) or not np.allclose(frame.energy_dft_ev_atom.to_numpy(float),frame.structure_id.map(target_map).to_numpy(float),atol=1e-9) or not np.array_equal(frame.fold.to_numpy(int),frame.structure_id.map(fold_map).to_numpy(int)):
            raise ValueError(f"{name}: IDs, DFT o folds no coinciden")
    predictions=[]; histories=[]; records=[]; heads={}; g_by_fold={f:scalers[f].transform(G).astype(np.float32) for f in range(5)}; elapsed_by_run={}
    start_train=time.perf_counter()
    for fold in range(5):
        outer_formulas=sorted(folds[(folds.outer_fold.eq(fold)) & folds.outer_role.eq("outer_validation")].composition_group.unique())
        mu={formula:float(value) for formula,value in zip(sorted(data.composition_id.unique()),ridge_models[fold].predict(mg_module.formula_features(sorted(data.composition_id.unique()))))}
        for seed in SEEDS:
            run_start=time.perf_counter()
            resid, record, hist, head=train_one(data,folds,embeddings,g_by_fold[fold],fold,seed,scales[fold],scalers[fold],torch.device("cpu"))
            record["elapsed_seconds"]=time.perf_counter()-run_start; elapsed_by_run[(fold,seed)]=record["elapsed_seconds"]
            histories.append(hist.assign(model="MatterGPT_F0_geometry")); records.append(record); heads[(fold,seed)]=head
            cursor=0
            for formula in outer_formulas:
                g=data[data.composition_id.eq(formula)].sort_values("family"); r=resid[cursor:cursor+3]; cursor+=3
                for row, rr in zip(g.itertuples(),r):
                    predictions.append({"model":"MatterGPT_F0_geometry","seed":seed,"fold":fold,"structure_id":row.structure_id,"composition_id":formula,
                        "family_report":row.family_report,"n_atoms":row.n_atoms,"energy_dft_ev_atom":row.energy_dft_ev_atom,
                        "ridge_mean_ev_atom":mu[formula],"residual_pred_ev_atom":float(rr),"energy_pred_ev_atom":float(mu[formula]+rr)})
            LOG.info("training_complete fold=%s seed=%s best_epoch=%s",fold,seed,record["best_epoch"])
    train_seconds=time.perf_counter()-start_train
    c2=pd.DataFrame(predictions)
    if len(c2)!=630 or c2.structure_id.nunique()!=210: raise ValueError("C2 no cubre 630 puntos OOF")
    pred=pd.concat([ridge_pred,c1[["model","seed","fold","structure_id","composition_id","family_report","n_atoms","energy_dft_ev_atom","ridge_mean_ev_atom","residual_pred_ev_atom","energy_pred_ev_atom"]],c2,*hist_parts],ignore_index=True)
    # Extrae métricas con la función corregida del experimento MatterGPT F0.
    metric,pairs=make_pairs(pred.rename(columns={"family_report":"family"}),mg_module)
    metric=metric.rename(columns={"model":"model"})
    pairs=pairs.rename(columns={"model":"model"})
    metric.to_csv(OUT/"metricas_por_composicion.csv",index=False)
    pred.to_csv(OUT/"predicciones_oof_por_estructura.csv",index=False)
    pairs.to_csv(OUT/"predicciones_oof_por_par.csv",index=False)
    # Historial C1 se reutiliza si conserva observaciones completas; C2 es recién entrenado.
    c1hist=pd.read_csv(BASE/"historial_entrenamiento.csv")
    c1hist=c1hist.assign(model="MatterGPT_F0")
    history=pd.concat([c1hist,pd.concat(histories,ignore_index=True)],ignore_index=True)
    history.to_csv(OUT/"historial_entrenamiento.csv",index=False)
    diag=diagnostic_predictions(data,folds,embeddings,g_by_fold,heads,{},mg_module)
    diag.to_csv(OUT/"ablation_predicciones.csv",index=False)
    # Sensibilidad controlada ±1 desviación estándar de cada descriptor escalado.
    sens=[]
    for fold in range(5):
        ids=np.flatnonzero(folds[folds.outer_fold.isin([fold])].structure_id.isin(data.structure_id).to_numpy())
        outer_ids=np.flatnonzero(data.composition_id.isin(folds[(folds.outer_fold.eq(fold))&(folds.outer_role.eq("outer_validation"))].composition_group).to_numpy())
        for seed in SEEDS:
            model=heads[(fold,seed)].eval(); baseg=g_by_fold[fold][outer_ids]
            for k,name in enumerate(GEOM_COLS):
                plus=baseg.copy(); minus=baseg.copy(); plus[:,k]+=1.; minus[:,k]-=1.
                x=embeddings[outer_ids]
                with torch.no_grad():
                    p=model(torch.as_tensor(np.concatenate((x,plus),axis=1),dtype=torch.float32)); m=model(torch.as_tensor(np.concatenate((x,minus),axis=1),dtype=torch.float32))
                    sensitivity=float(torch.mean(torch.abs(p-m)).item()/2)
                sens.append({"fold":fold,"seed":seed,"descriptor":name,"mean_abs_prediction_change_per_1_scaled_unit_ev_atom":sensitivity,
                             "first_layer_weight_l1":float(torch.sum(torch.abs(model.net[0].weight[:,768+k])).item()),
                             "first_layer_weight_l2":float(torch.linalg.vector_norm(model.net[0].weight[:,768+k]).item()),
                             "weight_change_l2_from_zero_initialization":float(torch.linalg.vector_norm(model.net[0].weight[:,768+k]).item()),
                             "first_layer_weight_vector_json":json.dumps(model.net[0].weight[:,768+k].detach().cpu().tolist())})
    sens_df=pd.DataFrame(sens); sens_df.to_csv(OUT/"weight_diagnostics.csv",index=False)
    bootstrap=bootstrap_primary(metric,[ ("MatterGPT_F0_geometry","MatterGPT_F0"),("MatterGPT_F0_geometry","Ridge_only"),("MatterGPT_F0_geometry","MatGPT_F0_geometry") ],n_boot)
    bootstrap.to_csv(OUT/"bootstrap_apareado.csv",index=False)
    metrics=aggregate_metrics(metric,pairs,pred)
    metrics["bootstrap_primary"]=bootstrap.to_dict("records")
    # C2 versus C1 ablation diagnostic metrics already saved by plotting; metrics remain raw and seeded.
    ablation_metrics={}
    labels_df=pred[pred.model.eq("MatterGPT_F0_geometry")][["fold","seed","structure_id","composition_id","energy_dft_ev_atom","family_report"]]
    d=diag.merge(labels_df,on=["fold","seed","structure_id","composition_id","family_report"],validate="many_to_one")
    for condition,group in d.groupby("diagnostic"):
        by=[]
        for (formula,seed),g in group.groupby(["composition_id","seed"]):
            e=g.energy_dft_ev_atom.to_numpy(float); yp=g.residual_pred_ev_atom.to_numpy(float)
            mae=float(np.mean([abs((yp[j]-yp[i]-(e[j]-e[i]))*1000) for i in range(3) for j in range(i+1,3)])); by.append(mae)
        ablation_metrics[condition]={"mae_delta_mev_atom":float(np.mean(by)),"n_formula_seed":len(by),"interpretation":"diagnóstico secundario, no causal"}
    metrics["geometry_ablation"]=ablation_metrics
    metrics["matched_initialization"]=initial
    metrics["audits"]={"preflight":audit,"C1_source_sha256":sha256(BASE/"predicciones_oof_por_estructura.csv"),
        "MatGPT_historical_prediction_sha256":sha256(HIST/"predicciones_oof_por_estructura.csv"),"geometry_source_sha256":sha256(GEOM_CSV),
        "dataset_sha256":sha256(DATASET),"fold_manifest_sha256":sha256(FOLDS),"test_historico_abierto":False}
    write_json(OUT/"metrics.json",metrics)
    c1rows=c1[["model","fold","seed","structure_id","composition_id","energy_dft_ev_atom","energy_pred_ev_atom"]].copy(); c1rows["condition"]="C1_MatterGPT_SLICES_only"
    c2rows=c2[["model","fold","seed","structure_id","composition_id","energy_dft_ev_atom","energy_pred_ev_atom"]].copy(); c2rows["condition"]="C2_MatterGPT_geometry"
    # Resources and configuration.
    cpu1=resource.getrusage(resource.RUSAGE_SELF); duration=time.perf_counter()-started
    resources={"wall_seconds":duration,"mlp_training_seconds":train_seconds,"per_run_seconds":{f"fold_{f}_seed_{s}":elapsed_by_run[(f,s)] for f,s in elapsed_by_run},
        "process_user_cpu_seconds":cpu1.ru_utime-cpu0.ru_utime,"process_system_cpu_seconds":cpu1.ru_stime-cpu0.ru_stime,
        "peak_rss_bytes":cpu1.ru_maxrss,"peak_rss_mib":cpu1.ru_maxrss/(1024**2),"device":"CPU","cpu_count":os.cpu_count(),
        "torch_threads":torch.get_num_threads(),"head_trainable_parameters":sum(p.numel() for p in heads[(0,SEEDS[0])].parameters()),"mattergpt_trainable_parameters":0,
        "runs":15,"python_version":platform.python_version(),"torch_version":torch.__version__,"numpy_version":np.__version__,
        "pandas_version":pd.__version__,"scipy_version":scipy.__version__,"scikit_learn_version":sklearn.__version__,"platform":platform.platform()}
    baseline_resources=json.loads((BASE/"resources.json").read_text(encoding="utf-8"))
    matgpt_resource_rows=pd.read_csv(GEOM_DIR/"resources_summary.csv")
    historic_train=matgpt_resource_rows.loc[matgpt_resource_rows.run.eq("train_resources")]
    resources["condition_resources"]={"C2_MatterGPT_SLICES_plus_geometry":{"runs":15,"training_seconds":train_seconds,"measured_this_run":True,"device":"CPU"},
        "C1_MatterGPT_SLICES_only":{"runs":15,"training_seconds_historical_full_experiment":baseline_resources.get("mlp_training_seconds"),
            "measured_this_run":False,"source":str(BASE/"resources.json")},
        "MatGPT_F0_controls_combined":{"runs":30,"training_seconds_historical_both_conditions":float(historic_train.wall_seconds.iloc[0]) if len(historic_train) else None,
            "measured_this_run":False,"allocation_per_condition":"not separately recorded in historical resources manifest","source":str(GEOM_DIR/"resources_summary.csv")}}
    write_json(OUT/"resources.json",resources)
    config={"experiment":EXPERIMENT,"question":"¿La geometría global mejora la predicción de diferencias energéticas sobre MatterGPT SLICES-only?",
        "property_vector":[1.0,2.0],"frozen_checkpoint_sha256":audit["checkpoint_sha256"],"tokenizer_sha256":audit["tokenizer_sha256"],
        "embedding_cache_sha256":audit["embedding_cache_sha256"],"embedding_extraction":"FINAL_END posterior a blocks y ln_f; caché congelada",
        "geometry_source":str(GEOM_CSV),"geometry_source_sha256":audit["geometry_sha256"],"geometry_descriptors":GEOM_COLS,
        "geometry_definition":json.loads(GEOM_MANIFEST.read_text())["shape"]+"; "+json.loads(GEOM_MANIFEST.read_text())["cell_reduction"]+"; angles reduced Niggli; original volume per atom",
        "ridge_models_reused_without_fit":True,"ridge_feature_count":17,"ridge_model_sha256":audit["ridge_sha256"],
        "head":{"architecture":"775-8-1","activation":"GELU","dropout":0.1,"optimizer":"AdamW","learning_rate":0.001,"weight_decay":0.0001,
            "batch_compositions":8,"max_epochs":200,"early_stopping_patience":20,"loss":"SmoothL1 beta=1 on centered residual / inner-train residual scale",
            "initialization":"C1 768-input MLP8 identical; 7 geometric columns exactly zero; RNG state restored after baseline head creation"},
        "folds":5,"inner_train_compositions":45,"inner_validation_compositions":11,"outer_evaluation_compositions":14,"seeds":list(SEEDS),
        "bootstrap":{"replicates":n_boot,"unit":"composition","seed":20261030,"primary_contrast":"C2 − C1"},
        "historical_test_accessed":False,"script_sha256":sha256(SCRIPT),"data_sha256":sha256(DATASET),"fold_manifest_sha256":sha256(FOLDS)}
    (OUT/"config.yaml").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    mpl_config = Path("/tmp/mattergpt_geometry_mplconfig")
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    make_figures(pred,metric,bootstrap,history,diag,metrics); architecture_diagram()
    # Checkpoint reloads guarantee all fifteen are compact and state-dict-compatible.
    reload_checks=[]
    for fold in range(5):
        for seed in SEEDS:
            path=OUT/"checkpoints"/f"fold_{fold}_seed_{seed}.pt"
            saved=torch.load(path,map_location="cpu",weights_only=True)
            check=MLP8Geometry(); check.load_state_dict(saved["state_dict"],strict=True)
            reload_checks.append({"fold":fold,"seed":seed,"status":"PASS","bytes":path.stat().st_size})
    audit["checks"]["checkpoint_reload_verificado_al_final"] = len(reload_checks) == 15
    audit["checkpoint_reload_checks"] = reload_checks
    if not all(value is True for key, value in audit["checks"].items() if key != "checkpoint_reload_verificado_al_final") or not audit["checks"]["checkpoint_reload_verificado_al_final"]:
        raise ValueError("Falló uno o más controles críticos; no se declara PASS")
    write_json(OUT/"preflight.json",audit)
    metrics["checkpoint_reload_checks"]=reload_checks; write_json(OUT/"metrics.json",metrics)
    (OUT/"summary.json").write_text(json.dumps({"status":"COMPLETE","experiment":EXPERIMENT,"structures":210,"compositions":70,
        "runs":15,"preflight":"PASS","metrics":metrics,"resources":resources,"test_historico_abierto":False},indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    report=report_text(metrics,bootstrap,audit,records,resources)
    (OUT/"reporte_cientifico.md").write_text(report,encoding="utf-8"); (REPORT/"reporte_cientifico.md").write_text(report,encoding="utf-8")
    (OUT/"README.md").write_text("# MatterGPT + geometría ABS₃\n\nExperimento de fusión tardía con MatterGPT congelado, siete descriptores de celda y MLP8 residual. Consulte `reporte_cientifico.md`, `config.yaml` y `preflight.json`. Reejecución: `python scripts/MatterGPT/mattergpt_abs3_geometry_late_fusion/run_experiment.py`.\n",encoding="utf-8")
    # El manifiesto final cuenta también la carga del proceso y el render de figuras.
    cpu_final=resource.getrusage(resource.RUSAGE_SELF)
    resources["wall_seconds"]=time.perf_counter()-started
    resources["process_user_cpu_seconds"]=cpu_final.ru_utime-cpu0.ru_utime
    resources["process_system_cpu_seconds"]=cpu_final.ru_stime-cpu0.ru_stime
    resources["peak_rss_bytes"]=cpu_final.ru_maxrss
    resources["peak_rss_mib"]=cpu_final.ru_maxrss/(1024**2)
    write_json(OUT/"resources.json",resources)
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8")); summary["resources"]=resources
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    report=report_text(metrics,bootstrap,audit,records,resources)
    (OUT/"reporte_cientifico.md").write_text(report,encoding="utf-8"); (REPORT/"reporte_cientifico.md").write_text(report,encoding="utf-8")
    LOG.info("experiment_complete wall_seconds=%.2f",resources["wall_seconds"])


def parse_args() -> argparse.Namespace:
    """Valida argumentos del experimento antes de crear artefactos."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",type=int,default=10000,help="Réplicas bootstrap apareadas por fórmula")
    args=parser.parse_args()
    if not 1000<=args.bootstrap<=100000: parser.error("--bootstrap debe estar entre 1000 y 100000")
    for path in (BASE_SCRIPT,BASE/"embeddings_mattergpt.npz",BASE/"embedding_manifest.csv",BASE/"predicciones_oof_por_estructura.csv",GEOM_CSV,GEOM_MANIFEST,DATASET,FOLDS):
        if not path.is_file(): parser.error(f"Falta archivo requerido: {path}")
    return args


def main() -> None:
    """Punto de entrada protegido con registro descriptivo de fallos."""
    configure_logging(); args=parse_args()
    try: run(args.bootstrap)
    except Exception:
        LOG.exception("experiment_failed; results are not declared complete")
        raise


if __name__ == "__main__":
    main()
