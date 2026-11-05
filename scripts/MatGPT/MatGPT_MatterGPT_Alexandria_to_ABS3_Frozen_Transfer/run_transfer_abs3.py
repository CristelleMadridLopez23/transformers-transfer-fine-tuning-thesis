#!/usr/bin/env python3
"""Evalúa la transferencia Alexandria→ABS₃ con extractores congelados y Ridge M2.

Reutiliza los cuatro controles M2 auditados y entrena únicamente las cabezas MLP8
de los extractores MatGPT F4 y MatterGPT F4. El test histórico no se carga.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_abs3_transfer")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EXP = "matgpt_mattergpt_alexandria_to_abs3_frozen_transfer"
OUT = ROOT / "experiments" / EXP
IMAGES = ROOT / "images" / EXP
LOG = logging.getLogger(EXP)
SEEDS = (42, 123, 2026)
MAT_BASE_SHA = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"
MAT_F4_SHA = "15ba8a05c6e55d5154b7ce7f4ecd8764e8bac250b71802c88eb1e0d6631d657f"
MATTER_BASE_SHA = "8d0071fa7c05449273bfaf60357e2c4f9525c8b8f47e6d313856160312a72b21"
MATTER_F4_SHA = "99da365ea1df8091dc6c16cbf370085409883ff0d72b568d531d78276d926ef3"
MAT_F4 = ROOT / "experiments/matgpt_alexandria_energy_transfer_20k_f4_local_mps/checkpoints/matgpt_alexandria_f4_energy_best.pt"
MATTER_F4 = ROOT / "experiments/mattergpt_alexandria_energy_transfer_20k_f4_compatible_subset/checkpoints/mattergpt_alexandria_f4_energy_best.pt"
MATTER_ORIG_FEATURES = ROOT / "experiments/mattergpt_abs3_frozen_ridge_mlp8/embeddings_mattergpt.npz"
BASE_PREDS = ROOT / "experiments/matgpt_f0_stoichiometric_ridge_m2_comparison/predicciones_oof_por_estructura.csv"


def sha(path: Path) -> str:
    """Calcula SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    """Importa un módulo de experimento por ruta sin ejecutar su main."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    """Escribe JSON mediante reemplazo atómico."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def restore_f4(base_model: Any, partial: dict[str, torch.Tensor], block_prefix: str) -> tuple[Any, dict[str, Any]]:
    """Fusiona pesos F4 en base, validando nombre/dimensión y carga final estricta."""
    full = base_model.state_dict()
    expected = {k for k in full if any(k.startswith(f"{block_prefix}.{i}.") for i in range(8, 12))}
    if not partial or not set(partial).issubset(expected):
        raise ValueError("El estado parcial F4 contiene parámetros fuera de bloques 8–11 o está vacío")
    for key, value in partial.items():
        if key not in full or full[key].shape != value.shape:
            raise ValueError(f"Tensor F4 incompatible: {key}, {getattr(value, 'shape', None)}")
    merged = dict(full)
    merged.update(partial)
    result = base_model.load_state_dict(merged, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Carga completa no estricta: {result}")
    for key in expected - set(partial):
        if not torch.equal(base_model.state_dict()[key].cpu(), full[key].cpu()):
            raise ValueError(f"Peso base de bloque no incluido cambió: {key}")
    return base_model, {"expected_block_tensors": len(expected), "updated_tensors": len(partial),
                        "updated_names": sorted(partial), "full_state_strict": True}


def extract_embeddings(backbone: str, adapted: bool, data: pd.DataFrame, device: torch.device,
                       mat_train: Any, matter_train: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Extrae FINAL_END contextualizado para un backbone y estado de pesos."""
    if backbone == "matgpt":
        model, stoi, block_size, _ = mat_train.load_matgpt()
        checkpoint = MAT_F4 if adapted else ROOT / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False) if adapted else None
        update_audit = None
        if adapted:
            model, update_audit = restore_f4(model, state["blocks_8_11"], "transformer_decoder.layers")
            if state.get("base_checkpoint_sha256") != MAT_BASE_SHA or int(state.get("selected_epoch", -1)) != 100:
                raise ValueError("Metadatos de checkpoint MatGPT F4 no coinciden con el F4-100 esperado")
        frame = data.rename(columns={"slices": "slices_matgpt"})
        seq = mat_train.encode_rows(frame, stoi, block_size)
        input_hash = hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest()
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(seq), 8):
                ids, positions = mat_train.make_batch(seq, np.arange(start, min(start + 8, len(seq))), int(stoi["<pad>"]))
                vectors.append(mat_train.final_end(model, ids.to(device), positions.to(device)).cpu().numpy())
        out = np.concatenate(vectors).astype(np.float32)
        meta = {"backbone": backbone, "state": "F4-100" if adapted else "original", "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha(checkpoint), "input_ids_sha256": input_hash, "token_count_min": min(map(len, seq)),
                "token_count_max": max(map(len, seq)), "tokenizer": "MatGPT model_info_property_0.json", "condition": None,
                "extraction": "FINAL_END posterior a TransformerDecoder y ln_f", "partial_restore": update_audit}
    elif backbone == "mattergpt":
        model, stoi, block_size = matter_train.load_model()
        checkpoint = MATTER_F4 if adapted else ROOT / "models/MatterGPT/pytorch_model.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True) if adapted else None
        update_audit = None
        if adapted:
            if tuple(state.get("condition_vector", ())) != (1.0, 2.0) or int(state.get("selected_epoch", -1)) != 71:
                raise ValueError("Metadatos de checkpoint MatterGPT F4 no coinciden con F4-71/[1,2]")
            model, update_audit = restore_f4(model, state["blocks_8_11"], "blocks")
            if state.get("base_checkpoint_sha256") != MATTER_BASE_SHA:
                raise ValueError("F4 MatterGPT no registra el checkpoint base esperado")
        frame = data.copy()
        frame["slices_matgpt"] = frame["slices"].astype(str)
        frame["matter_sequence_length"] = frame.slices_matgpt.map(lambda s: len(s.split()) + 2)
        seq = matter_train.read_sequences(frame, stoi, block_size)
        input_hash = hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest()
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(seq), 8):
                ids, positions = matter_train.make_batch(seq, np.arange(start, min(start + 8, len(seq))))
                vectors.append(matter_train.contextualized_final_end(model, ids.to(device), positions.to(device)).cpu().numpy())
        out = np.concatenate(vectors).astype(np.float32)
        meta = {"backbone": backbone, "state": "F4-71" if adapted else "original", "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha(checkpoint), "input_ids_sha256": input_hash, "token_count_min": min(map(len, seq)),
                "token_count_max": max(map(len, seq)), "tokenizer": "MatterGPT Voc_prior", "condition": [1.0, 2.0],
                "extraction": "terminal structural token posterior a bloques y ln_f", "partial_restore": update_audit}
    else:
        raise ValueError(f"Backbone inválido: {backbone}")
    if out.shape != (210, 768) or not np.isfinite(out).all():
        raise ValueError(f"Embedding {backbone}/{adapted} inválido: {out.shape}")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Backbone no quedó congelado")
    meta.update({"shape": list(out.shape), "finite": True, "backbone_frozen": True})
    return out, meta


def embedding_drift(original: np.ndarray, adapted: np.ndarray, data: pd.DataFrame, backbone: str) -> pd.DataFrame:
    """Resume desplazamiento y separación de representaciones por estructura/fórmula."""
    eps = 1e-12
    cos = np.sum(original * adapted, axis=1) / np.maximum(np.linalg.norm(original, axis=1) * np.linalg.norm(adapted, axis=1), eps)
    rel = np.linalg.norm(adapted - original, axis=1) / np.maximum(np.linalg.norm(original, axis=1), eps)
    rows = []
    for i, row in data.reset_index(drop=True).iterrows():
        rows.append({"backbone": backbone, "structure_id": row.structure_id, "composition_id": row.composition_id,
                     "family": row.family, "cosine_similarity": float(cos[i]), "relative_l2_drift": float(rel[i]),
                     "norm_original": float(np.linalg.norm(original[i])), "norm_adapted": float(np.linalg.norm(adapted[i]))})
    return pd.DataFrame(rows)


def load_baselines(data: pd.DataFrame, m2: Any) -> pd.DataFrame:
    """Carga cuatro controles M2 OOF, cotejando 630 filas por condición."""
    raw = pd.read_csv(BASE_PREDS)
    mapping = {"RETRAINED_M2_C0": "M0-S", "RETRAINED_M2_C1": "M0-G",
               "MatterGPT_F0": "G0-S", "MatterGPT_F0_geometry": "G0-G"}
    chunks = []
    for source, target in mapping.items():
        part = raw[raw.model.eq(source)].copy()
        if len(part) != 630:
            raise ValueError(f"Baseline {source}: se esperaban 630 filas, hay {len(part)}")
        part["model"] = target
        chunks.append(part)
    result = pd.concat(chunks, ignore_index=True)
    m2keys = ["structure_id", "composition_id", "fold", "seed"]
    if result.duplicated(["model", *m2keys]).any():
        raise ValueError("Duplicados en predicciones baseline")
    for name, group in result.groupby("model"):
        if set(group.structure_id) != set(data.structure_id):
            raise ValueError(f"Cohorte de {name} no es idéntica a ABS3")
    return result


def bootstrap_transfers(fs: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap paired por fórmula, semillas agregadas, cuatro contrastes A–D."""
    means = fs.groupby(["model", "composition_id"], as_index=False).mae_delta_signed_pairs_mev_atom.mean()
    pivot = means.pivot(index="composition_id", columns="model", values="mae_delta_signed_pairs_mev_atom")
    contrasts = [("M1-S", "M0-S"), ("M1-G", "M0-G"), ("G1-S", "G0-S"), ("G1-G", "G0-G")]
    rng = np.random.default_rng(20261102); rows = []
    for left, right in contrasts:
        diff = (pivot[left] - pivot[right]).to_numpy(float)
        draw = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        rows.append({"contrast": f"{left} − {right}", "estimate_mev_atom": float(diff.mean()),
                     "ci95_low_mev_atom": float(np.quantile(draw, .025)), "ci95_high_mev_atom": float(np.quantile(draw, .975)),
                     "n_compositions": 70, "bootstrap_replicates": 10000, "seed": 20261102,
                     "inference": "percentile IC sin ajustar; sin p-valores"})
    return pd.DataFrame(rows)


def write_figures(pred: pd.DataFrame, metrics: pd.DataFrame, boots: pd.DataFrame,
                  history: pd.DataFrame, drift: pd.DataFrame) -> list[str]:
    """Genera PNG científicos con notas de cohorte y unidad de observación."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    colors = {"M0-S":"#355f8a", "M0-G":"#d47737", "M1-S":"#4b8b63", "M1-G":"#b34d62",
              "G0-S":"#7564a8", "G0-G":"#ab83bd", "G1-S":"#377d86", "G1-G":"#d1a43b"}
    names = ["M0-S", "M0-G", "M1-S", "M1-G", "G0-S", "G0-G", "G1-S", "G1-G"]
    primary = metrics[metrics.estimand.eq("A")].set_index("model")
    files = []
    fig, ax = plt.subplots(figsize=(12, 5.8)); vals = [primary.loc[n, "mae_delta_signed_mev_atom"] for n in names]
    bars = ax.bar(names, vals, color=[colors[n] for n in names]); ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("MAE ΔE firmado (meV/átomo)"); ax.set_title("ABS₃: transferencia Alexandria y error de diferencias energéticas")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.01, -.01, "Estimando A: tres pares firmados i<j por fórmula y semilla; semillas agrupadas dentro de fórmula; macro de 70 fórmulas. 210 estructuras, cinco pliegues OOF; cada par pertenece a una misma fórmula. Geometría, cuando se indica, se concatena tras FINAL_END.", fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); p=IMAGES/"transferencia_mae_deltaE.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)

    fig, ax = plt.subplots(figsize=(9, 5)); y=np.arange(len(boots)); x=boots.estimate_mev_atom.to_numpy(); lo=boots.ci95_low_mev_atom.to_numpy(); hi=boots.ci95_high_mev_atom.to_numpy()
    ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=4,color="#355f8a"); ax.axvline(0,color="black",lw=1)
    ax.set_yticks(y,boots.contrast); ax.set_xlabel("Diferencia de MAE ΔE (adaptado − original; meV/átomo)"); ax.set_title("ABS₃: bootstrap apareado por composición"); ax.grid(axis="x",alpha=.2)
    fig.text(.01,-.02,"IC percentil 95 %, 10 000 réplicas, 70 fórmulas; se agregan semillas de entrenamiento dentro de fórmula. IC sin ajustar; no se calcularon p-valores. OOF de desarrollo, no test externo.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); p=IMAGES/"transferencia_bootstrap_apareado.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)

    fig, ax = plt.subplots(figsize=(9, 5)); gs=[]; pw=[]
    for n in names:
        r=primary.loc[n]; gs.append(float(r.ground_state_accuracy)); pw.append(float(r.pairwise_accuracy_eligible))
    xx=np.arange(8); ax.bar(xx-.18,gs,.36,label="Ground state",color=[colors[n] for n in names]); ax.bar(xx+.18,pw,.36,label="Pairwise elegible",color=[colors[n] for n in names],alpha=.55)
    ax.set_xticks(xx,names); ax.set_ylim(0,1); ax.set_ylabel("Exactitud"); ax.set_title("ABS₃: selección del mínimo DFT y ranking polimórfico"); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.text(.01,-.02,"OOF en 70 fórmulas × 3 polimorfos; GS: gap DFT al segundo mínimo >1 meV/átomo. Pairwise: |ΔE_DFT|>5 meV/átomo; |ΔE_pred|≤1 no resuelto cuenta incorrecto. Denominadores por condición en metrics.json.",fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1)); p=IMAGES/"transferencia_groundstate_pairwise.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)

    # Signed-pair parity uses ensemble B and fixed common axes across all eight models.
    ens=pred.groupby(["model","structure_id","composition_id","fold"],as_index=False).agg(energy_dft_ev_atom=("energy_dft_ev_atom","mean"),energy_pred_ev_atom=("energy_pred_ev_atom","mean"))
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True); lim=0.0
    allpairs={}
    for n in names:
        q=ens[ens.model.eq(n)]; pairs=[]
        for _,g in q.groupby("composition_id"):
            g=g.sort_values("structure_id"); d=g.energy_dft_ev_atom.to_numpy()*1000; pr=g.energy_pred_ev_atom.to_numpy()*1000
            for i in range(3):
                for j in range(i+1,3): pairs.append((d[j]-d[i],pr[j]-pr[i]))
        a=np.asarray(pairs); allpairs[n]=a; lim=max(lim,float(np.max(np.abs(a))))
    lim=max(lim,1)
    for ax,n in zip(axs.flat,names):
        a=allpairs[n]; m,b=np.polyfit(a[:,0],a[:,1],1); r2=1-np.sum((a[:,1]-(m*a[:,0]+b))**2)/np.sum((a[:,1]-a[:,1].mean())**2); mae=np.mean(np.abs(a[:,1]-a[:,0]))
        ax.scatter(a[:,0],a[:,1],s=9,alpha=.4,color=colors[n]); xx=np.linspace(-lim,lim,100); ax.plot(xx,xx,"k--",lw=.8); ax.plot(xx,m*xx+b,color="#a34a2a",lw=1)
        ax.set_title(f"{n} · N=210\nm={m:.2f}; R²={r2:.2f}; MAE={mae:.1f}",fontsize=9); ax.grid(alpha=.15); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    for ax in axs[-1,:]: ax.set_xlabel("ΔE DFT firmado i<j (meV/átomo)")
    for ax in axs[:,0]: ax.set_ylabel("ΔE predicho firmado i<j (meV/átomo)")
    fig.suptitle("ABS₃: paridad OOF de las tres diferencias firmadas por fórmula",y=1.02)
    fig.text(.01,-.01,"Cada punto = un par de polimorfos de la misma fórmula tras promediar las tres semillas (estimando B, solo para paridad). Ejes x=DFT e y=predicción; línea ideal y=x y OLS y=mx+b con todos los puntos. Cinco pliegues, 70 fórmulas.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.98)); p=IMAGES/"transferencia_paridad_deltaE.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)

    # Curves show train minibatch averages separately from full validation.
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,n in zip(axs.flat,["M1-S","M1-G","G1-S","G1-G"]):
        h=history[history.condition.eq(n)]; agg=h.groupby("epoch").agg(train=("train_minibatch_smoothl1","mean"),val=("inner_validation_full_smoothl1","mean"),active=("seed","size"))
        ax.plot(agg.index,agg.train,label="Train: promedio minilotes",color=colors[n]); ax.plot(agg.index,agg.val,label="Validación completa",color="#a34a2a"); ax.set_title(n); ax.set_xlabel("Época MLP8"); ax.grid(alpha=.2)
    axs[0,0].set_ylabel("SmoothL1 residual normalizada"); axs[1,0].set_ylabel("SmoothL1 residual normalizada"); axs[0,0].legend(fontsize=8)
    fig.suptitle("ABS₃: ajuste de las cabezas nuevas con extractores congelados")
    fig.text(.01,-.01,"Promedio de corridas activas por época (n puede variar por early stopping); train = promedio de minilotes inner-train; validation = pérdida completa de 11 fórmulas. Selección por mínimo validation, paciencia 20. Cada condición tiene 15 corridas; M0/G0 reutilizan controles existentes.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.96)); p=IMAGES/"transferencia_overfitting_mlp8.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)

    fig,axs=plt.subplots(1,2,figsize=(10,4.8))
    for ax,bk in zip(axs,["matgpt","mattergpt"]):
        q=drift[drift.backbone.eq(bk)]; ax.hist(q.cosine_similarity,bins=20,color="#547da5",alpha=.85); ax.set_title(f"{bk}: coseno original/adaptado"); ax.set_xlabel("Similitud coseno por estructura"); ax.set_ylabel("Número de estructuras"); ax.grid(axis="y",alpha=.2)
    fig.suptitle("ABS₃: deriva de FINAL_END tras adaptación Alexandria")
    fig.text(.01,-.02,"210 estructuras por arquitectura; histogramas por estructura, sin comparar coordenadas latentes entre arquitecturas. La similitud mide cambio representacional, no calidad energética.",fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.92)); p=IMAGES/"transferencia_embedding_drift.png"; fig.savefig(p,dpi=300,bbox_inches="tight"); plt.close(fig); files.append(p.name)
    return files


def run(preflight_only: bool = False) -> None:
    """Audita fuentes, extrae cachés y entrena 60 MLP8 de transferencia."""
    started = time.time(); cpu_started = time.process_time(); OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("embeddings", "checkpoints", "logs"):
        (OUT / folder).mkdir(exist_ok=True)
    if sha(MAT_F4) != MAT_F4_SHA or sha(MATTER_F4) != MATTER_F4_SHA:
        raise ValueError("SHA de checkpoint Alexandria no coincide")
    mg = load_module(ROOT / "scripts/MatGPT/MatGPT_F0_Stoichiometric_Ridge_M2_Comparison/run_experiment.py", "m2_transfer_reference")
    baseline = mg.baseline_module()
    data, folds, X_mat, G, ridges, mu, audit = mg.load_and_preflight(baseline)
    if sha(ROOT / "models/MatterGPT/pytorch_model.pt") != MATTER_BASE_SHA:
        raise ValueError("SHA checkpoint MatterGPT original no coincide")
    if int(torch.load(MAT_F4, map_location="cpu", weights_only=False)["selected_epoch"]) != 100:
        raise ValueError("MatGPT checkpoint no es F4-100")
    if int(torch.load(MATTER_F4, map_location="cpu", weights_only=True)["selected_epoch"]) != 71:
        raise ValueError("MatterGPT checkpoint no es F4-71")
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name); torch.set_num_threads(4)
    mat_train = load_module(ROOT / "scripts/MatGPT/MatGPT_Alexandria_Energy_Transfer_20k/train_energy.py", "transfer_matgpt_f4_train")
    matter_train = load_module(ROOT / "scripts/MatterGPT/MatterGPT_Alexandria_Energy_Transfer_20k_F4_Compatible/train_energy.py", "transfer_mattergpt_f4_train")
    # Reuse exact original-state FINAL_END caches that backed the validated baselines.
    mm_cache = np.load(mg.CACHE, allow_pickle=False)
    mm_ids = mm_cache["structure_id"].astype(str); lookup = {s:i for i,s in enumerate(mm_ids)}
    X_mat = np.stack([mm_cache["FINAL_END"][lookup[s]] for s in data.structure_id]).astype(np.float32)
    mo = np.load(MATTER_ORIG_FEATURES, allow_pickle=False); matter_lookup = {s:i for i,s in enumerate(mo["structure_ids"].astype(str))}
    if set(matter_lookup) != set(data.structure_id) or not np.allclose(mo["property_vector"], np.array([1.,2.],np.float32)):
        raise ValueError("Cache original MatterGPT IDs o vector fijo [1,2] incorrectos")
    X_matter = np.stack([mo["embeddings"][matter_lookup[s]] for s in data.structure_id]).astype(np.float32)
    # One fresh extraction per adapted backbone; both use identical sequence input IDs within architecture.
    cache_meta_path = OUT / "embedding_manifest.json"
    saved: dict[str, Any] = {}
    if all((OUT / "embeddings" / f"{label}_FINAL_END.npz").is_file() for label in ("M1", "G1")) and cache_meta_path.is_file():
        saved = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if saved.get("MatGPT_F4", {}).get("checkpoint_sha256") != MAT_F4_SHA or saved.get("MatterGPT_F4", {}).get("checkpoint_sha256") != MATTER_F4_SHA:
            raise ValueError("Cachés adaptadas existentes provienen de otros checkpoints")
        Xa_mat = np.load(OUT / "embeddings/M1_FINAL_END.npz", allow_pickle=False)["FINAL_END"]
        Xa_matter = np.load(OUT / "embeddings/G1_FINAL_END.npz", allow_pickle=False)["FINAL_END"]
        mat_meta, matter_meta = saved["MatGPT_F4"], saved["MatterGPT_F4"]
    else:
        Xa_mat, mat_meta = extract_embeddings("matgpt", True, data, device, mat_train, matter_train)
        Xa_matter, matter_meta = extract_embeddings("mattergpt", True, data, device, mat_train, matter_train)
    if mat_meta["input_ids_sha256"] == "" or matter_meta["input_ids_sha256"] == "":
        raise ValueError("Hash de input IDs vacío")
    # F4 checkpoints retain the original tokenizer; verify this directly for MatGPT.
    mat_info = ROOT / "models/MatGPT-source/mat-concat/mat-uncond/weights/model_info_property_0.json"
    mat_state = torch.load(MAT_F4, map_location="cpu", weights_only=False)
    if mat_state.get("tokenizer_sha256") != sha(mat_info):
        raise ValueError("Tokenizer MatGPT F4 no coincide con el tokenizer original")
    if saved and (mat_meta["input_ids_sha256"] != saved["MatGPT_F4"]["input_ids_sha256"] or
                  matter_meta["input_ids_sha256"] != saved["MatterGPT_F4"]["input_ids_sha256"]):
        raise ValueError("Hash de input IDs cambió entre extracción y caché")
    for label, arr in [("M0",X_mat),("M1",Xa_mat),("G0",X_matter),("G1",Xa_matter)]:
        np.savez_compressed(OUT/"embeddings"/f"{label}_FINAL_END.npz", structure_id=data.structure_id.astype(str).to_numpy(), FINAL_END=arr)
    atomic_json(cache_meta_path, {"MatGPT_F4": mat_meta, "MatterGPT_F4": matter_meta,
                                  "MatGPT_original_cache_sha256": sha(mg.CACHE),
                                  "MatterGPT_original_cache_sha256": sha(MATTER_ORIG_FEATURES),
                                  "structure_ids_sha256": hashlib.sha256("\n".join(data.structure_id.astype(str)).encode()).hexdigest()})
    drift = pd.concat([embedding_drift(X_mat,Xa_mat,data,"matgpt"),embedding_drift(X_matter,Xa_matter,data,"mattergpt")],ignore_index=True)
    drift.to_csv(OUT/"embedding_drift.csv",index=False)
    scalers=mg.fit_scalers(data,folds,G); G_by={f:scalers[f].transform(G).astype(np.float32) for f in range(5)}
    init_audit=mg.check_initialization(X_mat,G,data)
    preflight={"status":"PASS","historical_test_accessed":False,"cohort":{"structures":210,"compositions":70,"polymorphs_per_composition":3,"outer_folds":5,"seeds":list(SEEDS),"inner_train_validation_outer":[45,11,14]},
      "backbones":{"MatGPT_original":{"sha256":MAT_BASE_SHA},"MatGPT_F4":{"sha256":MAT_F4_SHA,"selected_epoch":100,**mat_meta},"MatterGPT_original":{"sha256":MATTER_BASE_SHA},"MatterGPT_F4":{"sha256":MATTER_F4_SHA,"selected_epoch":71,**matter_meta}},
      "ridge_m2":audit["ridge_m2_sha256"],"ridge_fit_called":False,"geometry_scalers":{"fit_only_inner_train":True,"per_fold":{"%d"%f:{"mean":scalers[f].mean_.tolist(),"scale":scalers[f].scale_.tolist()} for f in range(5)}},
      "step_zero_slice_vs_geometry":init_audit,"original_controls_reused":True,"device_for_adapted_embedding_extraction":device_name}
    atomic_json(OUT/"preflight.json",preflight)
    if preflight_only:
        LOG.info("Preflight completado; cachés adaptadas listas. No se entrenaron cabezas.")
        return
    # Original controls: reuse exactly the paired Ridge-M2 OOF predictions.
    predictions=[load_baselines(data,mg)]; histories=[]; records=[]
    conditions=[("M1-S","RETRAINED_M2_C0",Xa_mat),("M1-G","RETRAINED_M2_C1",Xa_mat),
                ("G1-S","RETRAINED_M2_C0",Xa_matter),("G1-G","RETRAINED_M2_C1",Xa_matter)]
    scale_by={}
    for fold in range(5):
        roles=folds[folds.outer_fold.eq(fold)].drop_duplicates("composition_group")
        tr=set(roles.loc[roles.inner_role.eq("inner_train"),"composition_group"])
        centered=data.energy_dft_ev_atom.to_numpy(float)-data.groupby("composition_id").energy_dft_ev_atom.transform("mean").to_numpy(float)
        scale_by[fold]=float(np.std(centered[data.composition_id.isin(tr)],ddof=0))
    total=60; done=0
    for name, inner_cond, X in conditions:
        mg.OUT=OUT/"checkpoints"/name
        for fold in range(5):
            for seed in SEEDS:
                pred,hist,record,_=mg.train_run(inner_cond,fold,seed,data,folds,X,G_by[fold],mu,scale_by[fold])
                pred["model"]=name; hist["condition"]=name; record["condition"]=name
                predictions.append(pred); histories.append(hist); records.append(record); done+=1
                progress={"status":"RUNNING","completed_runs":done,"total_runs":total,"current_condition":name,"fold":fold,"seed":seed,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                atomic_json(OUT/"progress.json",progress)
                pd.concat(predictions,ignore_index=True).to_csv(OUT/"predicciones_oof_por_estructura.csv",index=False)
                pd.concat(histories,ignore_index=True).to_csv(OUT/"historial_entrenamiento.csv",index=False)
                pd.DataFrame(records).to_csv(OUT/"run_manifest.csv",index=False)
                LOG.info("run %d/%d %s fold=%d seed=%d epoch=%d val=%.6f wall=%.1fs",done,total,name,fold,seed,record["best_epoch"],record["best_inner_validation_smoothl1"],record["elapsed_seconds"])
    pred=pd.concat(predictions,ignore_index=True)
    expected={"M0-S","M0-G","M1-S","M1-G","G0-S","G0-G","G1-S","G1-G"}
    if set(pred.model.unique()) != expected or pred.groupby("model").size().to_dict() != {x:630 for x in expected}:
        raise ValueError("Predicciones OOF no contienen 630 filas por las ocho condiciones")
    fs,pairs,metric_pack=mg.metrics_for(pred,baseline)
    metrics=pd.DataFrame(metric_pack["rows"])
    fs.to_csv(OUT/"metricas_por_composicion.csv",index=False); pairs.to_csv(OUT/"predicciones_oof_por_par.csv",index=False)
    pd.DataFrame(records).to_csv(OUT/"historial_corridas.csv",index=False)
    boots=bootstrap_transfers(metric_pack["formula_seed"]); boots.to_csv(OUT/"bootstrap_apareado.csv",index=False)
    # Per-polymorph and per-gap diagnostics from OOF rows/pairs.
    poly=pred.assign(abs_err_mev=(pred.energy_pred_ev_atom-pred.energy_dft_ev_atom).abs()*1000).groupby(["model","family"],as_index=False).agg(mae_EN_mev_atom=("abs_err_mev","mean"),n_rows_structure_seed=("structure_id","size"))
    poly.to_csv(OUT/"metricas_por_polimorfo.csv",index=False)
    gaps=pairs.copy(); gaps["DFT_gap_bin_mev_atom"]=pd.cut(gaps.delta_dft_signed_mev_atom.abs(),[0,5,10,25,50,100,np.inf],labels=["0–5","5–10","10–25","25–50","50–100",">100"],right=False,include_lowest=True)
    gap_table=gaps.groupby(["model","DFT_gap_bin_mev_atom"],observed=False).agg(n_pairs=("abs_error_mev_atom","size"),mae_delta_mev_atom=("abs_error_mev_atom","mean")).reset_index(); gap_table.to_csv(OUT/"metricas_por_gap.csv",index=False)
    figures=write_figures(pred,metrics,boots,pd.concat(histories,ignore_index=True),drift)
    resources={"status":"COMPLETED","elapsed_wall_seconds":time.time()-started,"process_cpu_seconds":time.process_time()-cpu_started,
      "cpu_count_logical":os.cpu_count(),"cpu_threads_torch":torch.get_num_threads(),"device_embedding_extraction":device_name,"training_device":"cpu",
      "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform=="darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
      "training_runs_new":60,"baselines_reused":4,"source_sha256":{"matgpt_f4":sha(MAT_F4),"mattergpt_f4":sha(MATTER_F4),"baseline_predictions":sha(BASE_PREDS)}}
    atomic_json(OUT/"resources.json",resources)
    atomic_json(OUT/"metrics.json",{"status":"PASS","estimand_A":metrics[metrics.estimand.eq("A")].to_dict("records"),"estimand_B":metrics[metrics.estimand.eq("B")].to_dict("records"),"bootstrap":boots.to_dict("records"),"figures":figures,"historical_test_accessed":False})
    atomic_json(OUT/"summary.json",{"status":"PASS","conditions":sorted(expected),"rows_per_condition":630,"new_mlp_runs":60,"original_conditions_reused":4,"primary_metric":"Estimando A: MAE ΔE firmado, tres pares intrafórmula","figures":figures})
    progress={"status":"COMPLETED","completed_runs":60,"total_runs":60,"finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}; atomic_json(OUT/"progress.json",progress)
    report=["# Transferencia energética Alexandria a ABS₃","","**Estado: PASS**","","## Hechos verificados","",f"- Cohorte OOF: 70 fórmulas, 210 estructuras, tres polimorfos, cinco folds y semillas {SEEDS}; grupos 45/11/14.",f"- Checkpoints: MatGPT base `{MAT_BASE_SHA}`, F4-100 `{MAT_F4_SHA}`; MatterGPT base `{MATTER_BASE_SHA}`, F4-71 `{MATTER_F4_SHA}`.","- Los F4 se reconstruyeron desde base + bloques 8–11; nombres/dimensiones se comprobaron y el estado completo se cargó con strict=True. Los cuatro backbones estuvieron congelados durante extracción; solo se optimizaron cabezas MLP8.","- Ridge M2 de 17 fracciones reutilizado sin fit; se reutilizaron los cuatro OOF originales validados. Geometría histórica y scalers exclusivos inner-train por fold.","- No se cargó el test histórico. Este resultado es OOF de desarrollo y no estima un test externo independiente.","","## Resultados OOF","","La métrica primaria es estimando A: tres diferencias firmadas i<j por fórmula/semilla, error absoluto promediado dentro de fórmula, semillas agregadas dentro de fórmula y macro-promedio de 70 fórmulas.","","| Condición | MAE E/N (meV/átomo) | MAE residuo | MAE ΔE A | Pairwise | Ground state |", "|---|---:|---:|---:|---:|---:|"]
    tab=metrics[metrics.estimand.eq("A")].set_index("model")
    for n in ["M0-S","M0-G","M1-S","M1-G","G0-S","G0-G","G1-S","G1-G"]:
        r=tab.loc[n]; report.append(f"| {n} | {r.mae_EN_mev_atom:.2f} | {r.mae_residual_mev_atom:.2f} | {r.mae_delta_signed_mev_atom:.2f} | {r.pairwise_accuracy_eligible:.3f} ({int(r.eligible_pairs)}) | {r.ground_state_accuracy:.3f} ({int(r.ground_state_denominator)}) |")
    report += ["","## Contrastes principales de transferencia","","IC bootstrap percentil apareados por fórmula; semillas agrupadas; IC sin ajustar y sin p-valores. La familia A–D se predefinió; no se atribuye significación familiar a IC individuales.","","| Contraste | ΔMAE ΔE adaptado−original | IC95 % |","|---|---:|---:|"]
    for r in boots.itertuples(): report.append(f"| {r.contrast} | {r.estimate_mev_atom:.2f} | [{r.ci95_low_mev_atom:.2f}, {r.ci95_high_mev_atom:.2f}] |")
    report += ["","## Interpretación y límites","","Las comparaciones dentro de cada arquitectura mantienen M2, cohortes, folds, seeds, entrenamiento de cabeza y geometría constantes; cambios corresponden a la inicialización de pesos del backbone, sujetos a que la cohorte Alexandria no introdujo solapamiento ABS₃. La adaptación no prueba que la supervisión energética sea causalmente responsable de una mejora. MatGPT y MatterGPT difieren también en tokenizer, arquitectura, pretraining y, para MatterGPT, subconjunto compatible Alexandria.","","Las curvas representan promedios de corridas activas; los minilotes de train no son la pérdida del conjunto completo. Los resultados por pliegues comparten el mismo conjunto de 70 composiciones y no constituyen una validación externa independiente.","","## Figuras","",*[f"- `images/{EXP}/{f}`" for f in figures],"","Artefactos OOF, métricas, checkpoints de cabezas, cachés FINAL_END y hashes se guardan en este directorio."]
    (OUT/"reporte_cientifico.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (OUT/"README.md").write_text(f"# {EXP}\n\nEjecución autónoma: `python scripts/MatGPT/MatGPT_MatterGPT_Alexandria_to_ABS3_Frozen_Transfer/run_transfer_abs3.py`.\n\nSupervisar: `tail -f experiments/{EXP}/run.log`; estado: `cat experiments/{EXP}/progress.json`; predicciones incrementales: `experiments/{EXP}/predicciones_oof_por_estructura.csv`.\n",encoding="utf-8")
    LOG.info("Completado; artefactos en %s", OUT)


def main() -> int:
    """Configura logging y ejecuta el experimento, informando errores no recuperables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Valida checkpoints y guarda embeddings; no entrena MLP8")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(preflight_only=args.preflight_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
