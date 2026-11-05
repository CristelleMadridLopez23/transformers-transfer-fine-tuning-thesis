"""Compara checkpoints F4 MatGPT/MatterGPT en Alexandria y ABS3 de desarrollo.

La paridad energética se calcula en la intersección de validación Alexandria.
El ranking polimórfico usa solo las 210 estructuras ABS3 train/val del conjunto
de desarrollo, excluidas del fine-tuning; nunca carga el test histórico.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_matgpt_mattergpt_f4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger("f4_model_comparison")


def load_module(path: Path, name: str) -> Any:
    """Carga un módulo Python por ruta para reutilizar el pipeline validado.

    Args:
        path: Archivo Python existente.
        name: Nombre temporal único del módulo.
    Returns:
        Módulo importado.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe módulo requerido: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    """Calcula SHA-256 en bloques.

    Args:
        path: Archivo de entrada.
    Returns:
        Hash SHA-256 hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_model(module: Any, checkpoint_path: Path, backbone_name: str,
                     device: torch.device) -> tuple[Any, Any, dict[str, float]]:
    """Carga el backbone original y aplica solo bloques 8–11 y cabeza F4.

    Args:
        module: Módulo de entrenamiento original del backbone.
        checkpoint_path: Checkpoint F4 seleccionado por validación Alexandria.
        backbone_name: ``matgpt`` o ``mattergpt``.
        device: Dispositivo de inferencia.
    Returns:
        Modelo, cabeza MLP8 y estadísticas de normalización del target.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Falta checkpoint F4: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if backbone_name == "matgpt":
        model, stoi, block_size, _ = module.load_matgpt()
        pad_id = int(stoi["<pad>"])
    elif backbone_name == "mattergpt":
        model, stoi, block_size = module.load_model()
        pad_id = 0
    else:
        raise ValueError(f"Backbone desconocido: {backbone_name}")
    incompatible = model.load_state_dict(state["blocks_8_11"], strict=False)
    loaded_names = set(state["blocks_8_11"])
    if not loaded_names or len(loaded_names) < 60:
        raise ValueError(f"Estado F4 incompleto ({len(loaded_names)} parámetros)")
    if backbone_name == "matgpt":
        head = module.EnergyHead()
    else:
        head = module.EnergyHead()
    head.load_state_dict(state["head_state_dict"], strict=True)
    model.to(device).eval()
    head.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    stats = {"mean": float(state["target_mean_ev_atom"]), "std": float(state["target_std_ev_atom"]),
             "selected_epoch": int(state["selected_epoch"]), "pad_id": pad_id,
             "loaded_f4_tensors": len(loaded_names), "missing_base_tensors_expected": len(incompatible.missing_keys)}
    return (model, (head, stoi, int(block_size)), stats)


def predict_matgpt(module: Any, model: Any, head: Any, frame: pd.DataFrame,
                   stoi: dict[str, int], block_size: int, pad_id: int,
                   target_mean: float, target_std: float, device: torch.device,
                   batch_size: int = 8) -> np.ndarray:
    """Predice energía por átomo con la interfaz FINAL_END histórica MatGPT.

    Args:
        module: Código de entrenamiento MatGPT.
        model: MatGPT con sus pesos F4 restaurados.
        head: Cabeza F4 restaurada.
        frame: Filas con ``slices_matgpt``.
        stoi: Vocabulario MatGPT.
        block_size: Contexto máximo.
        pad_id: Token de padding del checkpoint.
        target_mean: Media del target de train.
        target_std: Desviación poblacional del target de train.
        device: Dispositivo de inferencia.
        batch_size: Filas por minibatch.
    Returns:
        Vector de predicciones eV/átomo.
    """
    seq = module.encode_rows(frame, stoi, block_size)
    outputs: list[np.ndarray] = []
    model.eval(); head.eval()
    with torch.inference_mode():
        for begin in range(0, len(seq), batch_size):
            idx = np.arange(begin, min(begin + batch_size, len(seq)))
            ids, positions = module.make_batch(seq, idx, pad_id)
            hidden = module.final_end(model, ids.to(device), positions.to(device))
            pred = head(hidden).detach().cpu().numpy().astype(np.float64)
            outputs.append(pred)
    return np.concatenate(outputs) * target_std + target_mean


def predict_mattergpt(module: Any, model: Any, head: Any, frame: pd.DataFrame,
                      stoi: dict[str, int], block_size: int, target_mean: float,
                      target_std: float, device: torch.device,
                      batch_size: int = 8) -> np.ndarray:
    """Predice con condición [1,2] y FINAL_END posterior a ln_f de MatterGPT.

    Args:
        module: Código de entrenamiento MatterGPT.
        model: MatterGPT con pesos F4 restaurados.
        head: Cabeza F4 restaurada.
        frame: Filas con SLICES y longitud compatible auditada.
        stoi: Vocabulario MatterGPT.
        block_size: Contexto máximo.
        target_mean: Media del target de train.
        target_std: Desviación poblacional del target de train.
        device: Dispositivo de inferencia.
        batch_size: Filas por minibatch.
    Returns:
        Vector de predicciones eV/átomo.
    """
    seq = module.read_sequences(frame, stoi, block_size)
    outputs: list[np.ndarray] = []
    model.eval(); head.eval()
    with torch.inference_mode():
        for begin in range(0, len(seq), batch_size):
            idx = np.arange(begin, min(begin + batch_size, len(seq)))
            ids, positions = module.make_batch(seq, idx)
            pred = head(module.contextualized_final_end(model, ids.to(device), positions.to(device)))
            outputs.append(pred.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs) * target_std + target_mean


def regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calcula resumen de paridad y OLS con intercepto libre.

    Args:
        actual: DFT por átomo.
        predicted: Predicción por átomo.
    Returns:
        N, MAE, RMSE, sesgo, pendiente, intercepto y R² OLS.
    """
    if len(actual) != len(predicted) or len(actual) < 2:
        raise ValueError("La regresión requiere arrays pareados con al menos dos filas")
    slope, intercept = np.polyfit(actual, predicted, 1)
    fitted = slope * actual + intercept
    residual = predicted - actual
    ss_tot = float(np.square(actual - actual.mean()).sum())
    ss_res = float(np.square(predicted - fitted).sum())
    return {"n": int(len(actual)), "mae_ev_atom": float(np.mean(np.abs(residual))),
            "rmse_ev_atom": float(np.sqrt(np.mean(np.square(residual)))),
            "bias_ev_atom": float(np.mean(residual)), "ols_slope": float(slope),
            "ols_intercept_ev_atom": float(intercept),
            "ols_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def build_abs3_pairs(structures: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computa pares firmados y selecciones del mínimo por fórmula ABS3.

    Args:
        structures: 210 filas de desarrollo con fórmula, familia y DFT.
        predictions: Predicciones MatGPT y MatterGPT por structure_id.
    Returns:
        Tabla pareada por fórmula y tabla de elección de estado fundamental.
    """
    merged = structures.merge(predictions, on="structure_id", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    gs_rows: list[dict[str, Any]] = []
    for formula, group in merged.groupby("composition_id", sort=True):
        group = group.sort_values("structure_id").reset_index(drop=True)
        if len(group) != 3 or group.family.nunique() != 3:
            raise ValueError(f"Triplete ABS3 incompleto para {formula}")
        dft = group.energy_dft_ev_atom.to_numpy(float)
        dft_order = np.argsort(dft)
        dft_gap_mev = (dft[dft_order[1]] - dft[dft_order[0]]) * 1000.0
        eligible_gs = bool(dft_gap_mev > 1.0)
        for model in ("MatGPT", "MatterGPT"):
            pred = group[f"energy_pred_{model.lower()}_ev_atom"].to_numpy(float)
            order = np.argsort(pred)
            pred_ties = np.isclose(pred, pred.min(), rtol=0.0, atol=1e-12).sum() > 1
            gs_correct = bool(not pred_ties and order[0] == dft_order[0]) if eligible_gs else None
            gs_rows.append({"model": model, "composition_id": formula,
                            "dft_ground_state_structure_id": group.iloc[dft_order[0]].structure_id,
                            "predicted_ground_state_structure_id": None if pred_ties else group.iloc[order[0]].structure_id,
                            "dft_gap_to_second_mev_atom": float(dft_gap_mev),
                            "eligible_dft_gap_gt_1mev": eligible_gs,
                            "prediction_tie": bool(pred_ties), "ground_state_correct": gs_correct})
            for i, j in combinations(range(3), 2):
                dd = float((dft[j] - dft[i]) * 1000.0)
                dp = float((pred[j] - pred[i]) * 1000.0)
                eligible = abs(dd) > 5.0
                resolved = abs(dp) > 1.0
                correct = bool(eligible and resolved and np.sign(dd) == np.sign(dp))
                rows.append({"model": model, "composition_id": formula,
                             "structure_id_i": group.iloc[i].structure_id,
                             "structure_id_j": group.iloc[j].structure_id,
                             "family_i": group.iloc[i].family, "family_j": group.iloc[j].family,
                             "delta_dft_signed_mev_atom": dd,
                             "delta_pred_signed_mev_atom": dp,
                             "eligible_abs_dft_gt_5mev": eligible,
                             "pred_unresolved_abs_le_1mev": not resolved,
                             "pair_correct": correct})
    return pd.DataFrame(rows), pd.DataFrame(gs_rows)


def plot_curves(mat_history: pd.DataFrame, matter_history: pd.DataFrame, out: Path) -> None:
    """Grafica entrenamiento y validación por etapa de ambos fine-tunings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), constrained_layout=True)
    specs = [("MatGPT", mat_history, "20.000 entrenamiento · 2.000 validación", 100, "#bd542b"),
             ("MatterGPT", matter_history, "17.245 entrenamiento · 1.718 validación", 71, "#6554a5")]
    for ax, (name, hist, cohort, best, color) in zip(axes, specs):
        warm = hist[hist.stage.eq("warmup_frozen")].sort_values("epoch")
        f4 = hist[hist.stage.eq("finetune_f4")].sort_values("epoch")
        wx = warm.epoch.to_numpy(float)
        fx = 20 + f4.epoch.to_numpy(float)
        ax.plot(wx, warm.train_smoothl1, color=color, lw=1.5, label="Train · encoder congelado")
        ax.plot(wx, warm.validation_smoothl1, color=color, lw=1.5, ls="--", label="Validación · encoder congelado")
        ax.plot(fx, f4.train_smoothl1, color="#238579", lw=1.7, label="Train · F4")
        ax.plot(fx, f4.validation_smoothl1, color="#238579", lw=1.7, ls="--", label="Validación · F4")
        point = f4[f4.epoch.eq(best)]
        if not point.empty:
            ax.scatter(20 + best, float(point.validation_smoothl1.iloc[0]), marker="*", s=170,
                       color="#a51e2d", zorder=5, label=f"Checkpoint seleccionado: F4 {best}")
        ax.axvline(20.5, color="#555555", lw=1, ls=":")
        ax.set_title(f"{name}\n{cohort}")
        ax.set_xlabel("Época acumulada (20 warmup + F4)")
        ax.set_ylabel("SmoothL1 normalizada (menor es mejor)")
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Transferencia de energía desde SLICES: historial de train y validación", fontsize=14)
    fig.text(.01, -.015, "Línea discontinua: pérdida en validación completa; línea sólida: promedio de minibatches de train. "
             "Cada modelo normalizó el target con estadísticas de su propio train; las escalas son comparables de forma cualitativa, no exacta. "
             "Selección por mínima SmoothL1 de validación Alexandria. La cohorte MatterGPT es el subconjunto tokenizer/contexto compatible.",
             ha="left", va="top", fontsize=8.3, wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_parity(data: pd.DataFrame, actual_col: str, pred_cols: tuple[str, str],
                titles: tuple[str, str], title: str, note: str, out: Path) -> dict[str, Any]:
    """Produce dos paneles de paridad con y=x, OLS y métricas por panel."""
    actual = data[actual_col].to_numpy(float)
    all_values = [actual] + [data[c].to_numpy(float) for c in pred_cols]
    low = float(min(x.min() for x in all_values)); high = float(max(x.max() for x in all_values))
    margin = max((high - low) * .04, .01); low -= margin; high += margin
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
    stats: dict[str, Any] = {}
    colors = ["#bd542b", "#6554a5"]
    for ax, column, panel_title, color in zip(axes, pred_cols, titles, colors):
        pred = data[column].to_numpy(float)
        result = regression(actual, pred)
        stats[panel_title] = result
        ax.scatter(actual, pred, s=13, alpha=.52, color=color, edgecolors="none", label="Una estructura por punto")
        xx = np.linspace(low, high, 120)
        ax.plot(xx, xx, color="#303030", lw=1.5, ls="--", label="Ideal y = x")
        ax.plot(xx, result["ols_slope"] * xx + result["ols_intercept_ev_atom"], color="#d39b25", lw=1.6,
                label=f"OLS: y={result['ols_slope']:.3f}x {result['ols_intercept_ev_atom']:+.3f}")
        ax.set(xlim=(low, high), ylim=(low, high), title=panel_title,
               xlabel="Energía DFT (eV/átomo)", ylabel="Energía predicha (eV/átomo)")
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend(fontsize=8, loc="lower right")
        text = (f"N = {result['n']}\nMAE = {result['mae_ev_atom']*1000:.1f} meV/átomo\n"
                f"RMSE global = {result['rmse_ev_atom']*1000:.1f} meV/átomo\n"
                f"Pendiente OLS = {result['ols_slope']:.3f}\nIntercepto = {result['ols_intercept_ev_atom']:.3f} eV/átomo\n"
                f"R² OLS = {result['ols_r2']:.3f}")
        ax.text(.03, .97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"boxstyle":"round,pad=.4","facecolor":"white","alpha":.9,"edgecolor":"#cccccc"})
    fig.suptitle(title, fontsize=14)
    fig.text(.01, -.015, note, ha="left", va="top", fontsize=8.2, wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return stats


def plot_ranking(pairs: pd.DataFrame, ground: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Resume exactitud pairwise y elección del mínimo DFT por modelo."""
    models = ["MatGPT", "MatterGPT"]
    pair_stats: dict[str, Any] = {}; gs_stats: dict[str, Any] = {}
    pair_acc, gs_acc, pair_n, gs_n = [], [], [], []
    for model in models:
        p = pairs[(pairs.model == model) & pairs.eligible_abs_dft_gt_5mev]
        g = ground[(ground.model == model) & ground.eligible_dft_gap_gt_1mev]
        pair_stats[model] = {"eligible_pairs": int(len(p)), "unresolved_pairs": int(p.pred_unresolved_abs_le_1mev.sum()),
                             "correct_pairs": int(p.pair_correct.sum()), "accuracy_eligible": float(p.pair_correct.mean())}
        gs_stats[model] = {"eligible_formulas": int(len(g)), "correct_ground_state_picks": int(g.ground_state_correct.sum()),
                           "accuracy": float(g.ground_state_correct.mean())}
        pair_acc.append(pair_stats[model]["accuracy_eligible"]); gs_acc.append(gs_stats[model]["accuracy"])
        pair_n.append(pair_stats[model]["eligible_pairs"]); gs_n.append(gs_stats[model]["eligible_formulas"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6), constrained_layout=True)
    names = ["MatGPT F4 · 20k", "MatterGPT F4 · 17.245"]
    bars = axes[0].bar(names, pair_acc, color=["#bd542b", "#6554a5"], width=.58)
    axes[0].set_ylim(0, 1); axes[0].set_title("Ranking pairwise polimórfico")
    axes[0].set_ylabel("Pares elegibles ordenados correctamente")
    axes[0].bar_label(bars, labels=[f"{a:.1%}\n{n} pares" for a,n in zip(pair_acc,pair_n)], padding=4, fontsize=9)
    axes[0].grid(axis="y", alpha=.2); axes[0].set_axisbelow(True)
    bars = axes[1].bar(names, gs_acc, color=["#bd542b", "#6554a5"], width=.58)
    axes[1].set_ylim(0, 1); axes[1].set_title("Selección exacta del ground state DFT")
    axes[1].set_ylabel("Fórmulas elegibles seleccionadas correctamente")
    axes[1].bar_label(bars, labels=[f"{a:.1%}\n{n} fórmulas" for a,n in zip(gs_acc,gs_n)], padding=4, fontsize=9)
    axes[1].grid(axis="y", alpha=.2); axes[1].set_axisbelow(True)
    fig.suptitle("ABS₃ de desarrollo: orden de tres polimorfos por fórmula", fontsize=14)
    note = ("Ambos checkpoints F4 se evaluaron sin reentrenar en las mismas 210 estructuras estáticas (70 fórmulas × 3 polimorfos), "
            "que quedaron excluidas del fine-tuning Alexandria. Pairwise: |ΔE_DFT| > 5 meV/átomo; |ΔE_pred| ≤ 1 meV/átomo cuenta incorrecto. "
            "Ground state elegible si la brecha DFT al segundo mínimo > 1 meV/átomo; empate predicho cuenta incorrecto. "
            "Denominadores dependen de esos filtros. Es evaluación de desarrollo, no test externo independiente.")
    fig.text(.01, -.02, note, ha="left", va="top", fontsize=8.2, wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return {"pairwise": pair_stats, "ground_state": gs_stats}


def parse_args() -> argparse.Namespace:
    """Configura rutas y dispositivo de inferencia.

    Returns:
        Argumentos validados por argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--device", choices=["mps", "cpu", "auto"], default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    """Evalúa los dos checkpoints F4, escribe CSV/JSON/PNG y manifiesto de recursos."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size debe ser positivo")
    root = args.project_root.resolve(); started = time.time(); cpu_started = time.process_time()
    experiment = root / "experiments/matgpt_mattergpt_alexandria_20k_f4_comparison"
    image_dir = root / "images/MatGPT/MatterGPT_Alexandria_20k_F4_Comparison"
    report_dir = root / "reports/MatGPT/MatterGPT_Alexandria_20k_F4_Comparison"
    experiment.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Se solicitó MPS pero no está disponible")
    torch.set_num_threads(4)
    LOGGER.info("Inferencia sobre checkpoints F4 en %s", device)

    mat_train = load_module(root / "scripts/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/train_energy.py", "matgpt_f4_train")
    matter_train = load_module(root / "scripts/MatterGPT/MatterGPT_Alexandria_Energy_Transfer_20k_F4_Compatible/train_energy.py", "mattergpt_f4_train")
    frozen_matter = load_module(root / "scripts/MatterGPT/mattergpt_abs3_frozen_ridge_mlp8/run_experiment.py", "mattergpt_abs3_dev_loader")

    mat_checkpoint = root / "experiments/matgpt_alexandria_energy_transfer_20k_f4_local_mps/checkpoints/matgpt_alexandria_f4_energy_best.pt"
    matter_checkpoint = root / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/checkpoints/mattergpt_alexandria_f4_energy_best.pt"
    mat_model, mat_bundle, mat_stats = checkpoint_model(mat_train, mat_checkpoint, "matgpt", device)
    mat_head, mat_stoi, mat_block = mat_bundle
    matter_model, matter_bundle, matter_stats = checkpoint_model(matter_train, matter_checkpoint, "mattergpt", device)
    matter_head, matter_stoi, matter_block = matter_bundle

    alex_mat_path = root / "experiments/matgpt_alexandria_energy_transfer_20k/alexandria_2k_validation.csv"
    alex_matter_path = root / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/alexandria_matter_compatible_validation.csv"
    alex_mat = pd.read_csv(alex_mat_path); alex_matter = pd.read_csv(alex_matter_path)
    if alex_mat.mat_id.duplicated().any() or alex_matter.mat_id.duplicated().any():
        raise ValueError("IDs repetidos en validation Alexandria")
    common_ids = sorted(set(alex_mat.mat_id) & set(alex_matter.mat_id))
    if len(common_ids) != len(alex_matter) or len(common_ids) != 1718:
        raise ValueError(f"Intersección validation inesperada: {len(common_ids)}")
    alex = alex_mat.set_index("mat_id").loc[common_ids].reset_index()
    matter_alex = alex_matter.set_index("mat_id").loc[common_ids].reset_index()
    if not np.array_equal(alex.energy_per_atom_ev.to_numpy(), matter_alex.energy_per_atom_ev.to_numpy()):
        raise ValueError("Targets DFT no coinciden por mat_id en la validación compartida")
    if not alex.slices_matgpt.astype(str).equals(matter_alex.slices_matgpt.astype(str)):
        raise ValueError("Cadenas SLICES difieren entre validaciones emparejadas")
    alex_mat_pred = predict_matgpt(mat_train, mat_model, mat_head, alex, mat_stoi, mat_block,
                                   mat_stats["pad_id"], mat_stats["mean"], mat_stats["std"], device, args.batch_size)
    matter_alex_pred = predict_mattergpt(matter_train, matter_model, matter_head, matter_alex,
                                         matter_stoi, matter_block, matter_stats["mean"], matter_stats["std"],
                                         device, args.batch_size)
    alex_pred = pd.DataFrame({"mat_id": alex.mat_id, "formula": alex.formula,
                              "energy_dft_ev_atom": alex.energy_per_atom_ev,
                              "energy_pred_matgpt_ev_atom": alex_mat_pred,
                              "energy_pred_mattergpt_ev_atom": matter_alex_pred})
    alex_pred.to_csv(experiment / "predicciones_validacion_alexandria_pareadas.csv", index=False)
    alex_stats = plot_parity(alex_pred, "energy_dft_ev_atom",
                             ("energy_pred_matgpt_ev_atom", "energy_pred_mattergpt_ev_atom"),
                             ("MatGPT F4 · 20k", "MatterGPT F4 · 17.245"),
                             "Paridad de energía Alexandria en validation compartida",
                             "Una observación por fórmula y estructura: 1.718 IDs comunes; targets y SLICES verificados por ID. "
                             "Ambos modelos se evaluaron con su propio checkpoint seleccionado solo en validation Alexandria. "
                             "Energías DFT en eV/átomo; el test histórico ABS₃ no se usó.",
                             image_dir / "paridad_energia_alexandria_validation.png")

    abs3, _ = frozen_matter.dev_data()
    abs3 = abs3.copy()
    abs3["slices_matgpt"] = abs3["slices"]
    abs3["matter_sequence_length"] = abs3.slices.astype(str).map(lambda text: len(text.split()) + 2)
    if (len(abs3), abs3.composition_id.nunique()) != (210, 70):
        raise ValueError("La cohorte de desarrollo ABS3 no es 210/70")
    # Solo se admiten asignaciones train/val del manifiesto histórico; dev_data ya excluye test antes de leer labels.
    abs3_mat_pred = predict_matgpt(mat_train, mat_model, mat_head, abs3, mat_stoi, mat_block,
                                   mat_stats["pad_id"], mat_stats["mean"], mat_stats["std"], device, args.batch_size)
    abs3_matter_pred = predict_mattergpt(matter_train, matter_model, matter_head, abs3,
                                         matter_stoi, matter_block, matter_stats["mean"], matter_stats["std"],
                                         device, args.batch_size)
    abs3_pred = pd.DataFrame({"structure_id": abs3.structure_id, "composition_id": abs3.composition_id,
                              "family": abs3.family, "outer_fold": abs3.outer_fold,
                              "energy_dft_ev_atom": abs3.energy_dft_ev_atom,
                              "energy_pred_matgpt_ev_atom": abs3_mat_pred,
                              "energy_pred_mattergpt_ev_atom": abs3_matter_pred})
    abs3_pred.to_csv(experiment / "predicciones_abs3_desarrollo.csv", index=False)
    abs3_stats = plot_parity(abs3_pred, "energy_dft_ev_atom",
                             ("energy_pred_matgpt_ev_atom", "energy_pred_mattergpt_ev_atom"),
                             ("MatGPT F4 · 20k", "MatterGPT F4 · 17.245"),
                             "Transferencia F4: paridad energética en ABS₃ de desarrollo",
                             "210 estructuras estáticas de 70 fórmulas (tres polimorfos por fórmula), excluidas de ambos fine-tunings. "
                             "Se muestra todo el conjunto train/val de desarrollo; no es OOF del checkpoint F4 ni test externo. "
                             "Energía por átomo, eV/átomo; cada punto es una estructura.",
                             image_dir / "paridad_energia_abs3_desarrollo.png")
    pair_df, ground_df = build_abs3_pairs(abs3, abs3_pred.drop(columns=["composition_id", "family", "outer_fold", "energy_dft_ev_atom"]))
    pair_df.to_csv(experiment / "ranking_pairwise_abs3.csv", index=False)
    ground_df.to_csv(experiment / "seleccion_groundstate_abs3_por_formula.csv", index=False)
    ranking_stats = plot_ranking(pair_df, ground_df, image_dir / "ranking_groundstate_abs3.png")

    mat_hist_path = root / "experiments/matgpt_alexandria_energy_transfer_20k_f4_local_mps/historial_entrenamiento.csv"
    matter_hist_path = root / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/historial_entrenamiento.csv"
    mat_hist = pd.read_csv(mat_hist_path); matter_hist = pd.read_csv(matter_hist_path)
    plot_curves(mat_hist, matter_hist, image_dir / "curvas_entrenamiento_validacion_f4.png")
    mat_summary = json.loads((root / "experiments/matgpt_alexandria_energy_transfer_20k_f4_local_mps/summary.json").read_text())
    matter_summary = json.loads((root / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/summary.json").read_text())
    result = {"status":"PASS_WITH_SCOPE_LIMITS", "training_comparison": {
                  "MatGPT": {"train_rows":20000,"validation_rows":2000,"best_epoch":mat_stats["selected_epoch"],
                             "all_validation_metrics_original":mat_summary.get("validation_metrics", mat_summary.get("metrics",{}))},
                  "MatterGPT": {"train_rows":17245,"validation_rows":1718,"best_epoch":matter_stats["selected_epoch"],
                                "all_validation_metrics_original":matter_summary.get("metrics",{})}},
              "alexandria_common_validation": {"n":len(alex_pred),"targets_equal_by_id":True,"slices_equal_by_id":True,
                                                "MatGPT":alex_stats["MatGPT F4 · 20k"],
                                                "MatterGPT":alex_stats["MatterGPT F4 · 17.245"]},
              "abs3_development_transfer": {"n_structures":len(abs3_pred),"n_compositions":int(abs3.composition_id.nunique()),
                                             "n_polymorphs_per_composition":3,"test_accessed":False,
                                             "MatGPT":abs3_stats["MatGPT F4 · 20k"],
                                             "MatterGPT":abs3_stats["MatterGPT F4 · 17.245"],**ranking_stats},
              "limitations":["MatterGPT fine-tuning usa el subconjunto compatible (17,245/1,718), no el conjunto 20k/2k completo.",
                             "ABS3 de desarrollo no es un test externo; los checkpoints se seleccionaron con validación Alexandria.",
                             "No se recalcularon ni modificaron baselines congelados históricos."],
              "checkpoints": {"MatGPT":sha256(mat_checkpoint),"MatterGPT":sha256(matter_checkpoint)}}
    (experiment / "metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    config = {"models":"MatGPT F4 vs MatterGPT F4", "fine_tuning_data":{"MatGPT":{"train":20000,"validation":2000},
               "MatterGPT":{"train":17245,"validation":1718}}, "alexandria_comparison":"intersection by mat_id, 1718 rows",
               "ranking_evaluation":"210 ABS3 static development structures; 70 formulas; excludes historical test",
               "device":str(device),"batch_size":args.batch_size,"condition_mattergpt":[1.0,2.0]}
    (experiment / "config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    sources = [mat_checkpoint,matter_checkpoint,alex_mat_path,alex_matter_path,
               root/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv",
               root/"reports/MatGPT/MatGPT_SLICES_Geometria/staging_repo/Alternative2/reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv",
               mat_hist_path,matter_hist_path]
    resources = {"status":"COMPLETED","device":str(device),"batch_size":args.batch_size,
                 "wall_seconds":time.time()-started,"cpu_seconds":time.process_time()-cpu_started,
                 "max_rss_bytes":int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss),
                 "training_performed":False,"source_sha256":{str(p.relative_to(root)):sha256(p) for p in sources if p.is_file()}}
    (experiment/"resources.json").write_text(json.dumps(resources,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (experiment/"summary.json").write_text(json.dumps({"status":result["status"],"figures":[p.name for p in image_dir.glob("*.png")],
                                                          "metrics":"metrics.json","predictions":["predicciones_validacion_alexandria_pareadas.csv","predicciones_abs3_desarrollo.csv"],
                                                          "ranking":["ranking_pairwise_abs3.csv","seleccion_groundstate_abs3_por_formula.csv"]},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report_dir.mkdir(parents=True,exist_ok=True)
    report = ["# Comparación F4 MatGPT vs MatterGPT con transferencia Alexandria 20k", "",
              "## Cohortes y alcance", "",
              "MatGPT se entrenó con 20.000 fórmulas y validó con 2.000. MatterGPT solo acepta 17.245/20.000 fórmulas de train y 1.718/2.000 de validación por vocabulario y límite de contexto. La paridad Alexandria usa la intersección exacta de 1.718 IDs, con SLICES y targets cotejados por ID.", "",
              "El ranking y la selección del estado fundamental se calcularon con los checkpoints F4 sobre 210 estructuras ABS₃ estáticas de desarrollo (70 tripletes), excluidas del fine-tuning Alexandria. No se leyó el test histórico. Esta es una evaluación de transferencia sobre desarrollo, no un test externo independiente.", "",
              "## Hallazgos", "",
              f"En Alexandria compartida, el MAE fue {alex_stats['MatGPT F4 · 20k']['mae_ev_atom']*1000:.1f} meV/átomo para MatGPT y {alex_stats['MatterGPT F4 · 17.245']['mae_ev_atom']*1000:.1f} para MatterGPT.", "",
              f"En ABS₃ desarrollo, el MAE energético fue {abs3_stats['MatGPT F4 · 20k']['mae_ev_atom']*1000:.1f} meV/átomo para MatGPT y {abs3_stats['MatterGPT F4 · 17.245']['mae_ev_atom']*1000:.1f} para MatterGPT.", "",
              f"Exactitud pairwise: MatGPT {ranking_stats['pairwise']['MatGPT']['accuracy_eligible']:.1%} ({ranking_stats['pairwise']['MatGPT']['correct_pairs']}/{ranking_stats['pairwise']['MatGPT']['eligible_pairs']}); MatterGPT {ranking_stats['pairwise']['MatterGPT']['accuracy_eligible']:.1%} ({ranking_stats['pairwise']['MatterGPT']['correct_pairs']}/{ranking_stats['pairwise']['MatterGPT']['eligible_pairs']}).", "",
              f"Selección exacta del ground state: MatGPT {ranking_stats['ground_state']['MatGPT']['accuracy']:.1%} ({ranking_stats['ground_state']['MatGPT']['correct_ground_state_picks']}/{ranking_stats['ground_state']['MatGPT']['eligible_formulas']}); MatterGPT {ranking_stats['ground_state']['MatterGPT']['accuracy']:.1%} ({ranking_stats['ground_state']['MatterGPT']['correct_ground_state_picks']}/{ranking_stats['ground_state']['MatterGPT']['eligible_formulas']}).", "",
              "## Figuras", "",
              "- `curvas_entrenamiento_validacion_f4.png`", "- `paridad_energia_alexandria_validation.png`", "- `paridad_energia_abs3_desarrollo.png`", "- `ranking_groundstate_abs3.png`", "",
              "Las figuras y los CSV/JSON de predicción y métricas están vinculados desde los directorios images/ y experiments/ del proyecto.", ""]
    (report_dir/"reporte_comparativo.md").write_text("\n".join(report),encoding="utf-8")
    LOGGER.info("Comparación terminada en %.1f s; figuras en %s",resources["wall_seconds"],image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
