"""Reproduce H3-C desde Ridge y MLP8 históricos, sin reentrenar ni abrir test.

Entradas: raíz del proyecto y artefactos de H3-C existentes.
Salidas: CSV de predicciones reproducidas y JSON de diferencias numéricas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import StructuralMLP8

LOG = logging.getLogger("h3c_final_reproduction")
FAMILIES = ("dist_perovskite", "hexagonal", "needle")
SEEDS = (42, 123, 2026)
EXPECTED_CHECKPOINT = "4318e5fa88a5b177606c792c518251588edf446fa5edb13c35faa08e34d0e414"


def project_root() -> Path:
    """Obtiene raíz del proyecto desde el módulo."""
    return next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())


def sha256(path: Path) -> str:
    """Calcula SHA-256 en streaming para entradas verificables."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_development(path: Path) -> pd.DataFrame:
    """Excluye filas test antes de leer sus energías; valida identidad y unidades."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in {"train", "val"}:
                continue
            atoms = int(row["num_atoms"])
            total = float(row["energy_total_ev"])
            atom = float(row["energy_per_atom_ev"])
            if atoms <= 0 or abs(total/atoms-atom) > 1e-8:
                raise ValueError(f"E_total/N inconsistente: {row['structure_id']}")
            rows.append({"structure_id": row["structure_id"],
                         "composition_id": row["reduced_formula"],
                         "polymorph_family": row["polymorph_family"],
                         "n_atoms": atoms, "energy_total_dft_ev": total,
                         "energy_per_atom_dft_ev": atom,
                         "slices": row["matgpt_slices"]})
    result = pd.DataFrame(rows)
    if len(result) != 210 or result.structure_id.nunique() != 210 or result.composition_id.nunique() != 70:
        raise ValueError("Se requieren 210 estructuras de desarrollo y 70 fórmulas")
    if result.groupby("composition_id").size().ne(3).any():
        raise ValueError("Tripletes incompletos")
    return result


def feature_cache(path: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """Carga exactamente las dos representaciones históricas de 768 dimensiones."""
    manifest = json.loads(path.with_name("feature_manifest.json").read_text())
    if manifest["checkpoint_sha256"] != EXPECTED_CHECKPOINT or manifest["historical_test_used_or_extracted"]:
        raise ValueError("Cache no corresponde a MatGPT congelado autorizado")
    if sha256(path) != manifest["cache_sha256"]:
        raise ValueError("Hash de cache de embeddings no coincide")
    with np.load(path, allow_pickle=False) as archive:
        if archive["structure_id"].astype(str).tolist() != ids:
            raise ValueError("Orden de IDs diferente al histórico")
        result = {name: np.array(archive[name], dtype=np.float32, copy=True)
                  for name in ("L03_MEAN", "FINAL_END")}
    if any(v.shape != (210, 768) or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Embeddings no finitos o de dimensión distinta")
    return result


def replay(root: Path) -> dict[str, Any]:
    """Reproduce 15 corridas seleccionadas y compara fila por fila con H3-C."""
    old = root / "reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8"
    new = root / "reports/MatGPT/MatGPT_H3C_Final"
    output = new / "reproduction"
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv"
    fold_path = root / "reports/MatGPT/MatGPT_frozen_2/v1/fold_assignments.csv"
    cache_path = root / "reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
    checkpoint_path = root / "models/MatGPT-source/mat-concat/mat-uncond/weights/transformer_property_0.pth"
    if sha256(checkpoint_path) != EXPECTED_CHECKPOINT:
        raise ValueError("Checkpoint MatGPT original cambió")
    data = read_development(dataset_path)
    features = feature_cache(cache_path, data.structure_id.tolist())
    folds = pd.read_csv(fold_path)
    folds = folds[folds.historical_split.isin(["train", "val"])].copy()
    if len(folds) != 1050 or set(folds.structure_id) != set(data.structure_id):
        raise ValueError("Manifiesto de pliegues incompatible")
    historical = pd.read_csv(old / "H3_C_ridge_retrained_delta/oof_predictions.csv")
    if len(historical) != 630 or historical.structure_id.nunique() != 210:
        raise ValueError("Predicciones históricas incompletas")
    index = data.set_index("structure_id")
    lookup = {sid:i for i,sid in enumerate(data.structure_id)}
    out_rows: list[dict[str, Any]] = []
    parameter_counts: set[int] = set()
    ridge_alphas: dict[str, float] = {}
    for fold in range(5):
        part = folds[folds.outer_fold.eq(fold)]
        outer = part[part.outer_role.eq("outer_validation")]
        formulas = sorted(outer.composition_group.unique())
        if len(formulas) != 14 or len(outer) != 42 or outer.groupby("composition_group").size().ne(3).any():
            raise ValueError(f"Pliegue externo {fold} inválido")
        ridge_path = old / f"H3_B_ridge_fixed_delta/ridge_by_fold/fold_{fold}/ridge.joblib"
        ridge = joblib.load(ridge_path)
        if list(ridge.named_steps) != ["scaler", "ridge"]:
            raise ValueError("Pipeline Ridge histórico distinto")
        ridge_alphas[str(fold)] = float(ridge.named_steps["ridge"].alpha)
        grouped_ids = [[str(index[(index.composition_id == formula) &
                                   (index.polymorph_family == family)].index[0])
                        for family in FAMILIES] for formula in formulas]
        x_ridge = np.stack([features["L03_MEAN"][[lookup[sid] for sid in group]].mean(axis=0)
                            for group in grouped_ids])
        means = ridge.predict(x_ridge)
        for seed in SEEDS:
            model_path = old / f"H3_C_ridge_retrained_delta/runs/fold_{fold}_seed_{seed}/structural_mlp8.pt"
            saved = torch.load(model_path, map_location="cpu", weights_only=True)
            if saved["architecture"] != [768,8,1] or saved["fold"] != fold or saved["seed"] != seed:
                raise ValueError("Checkpoint MLP8 incompatible")
            model = StructuralMLP8(input_dim=768, dropout=0.1)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.eval()
            parameter_counts.add(sum(p.numel() for p in model.parameters()))
            x_structural = torch.tensor(np.stack([features["FINAL_END"][[lookup[sid] for sid in group]]
                                                   for group in grouped_ids]), dtype=torch.float32)
            with torch.inference_mode():
                residual = model(x_structural).cpu().numpy()
            if np.max(np.abs(residual.sum(axis=1))) > 1e-6:
                raise ValueError("Residuos MLP8 no centrados")
            for i, (formula, group) in enumerate(zip(formulas, grouped_ids)):
                for j, sid in enumerate(group):
                    source = index.loc[sid]
                    per_atom = float(means[i] + residual[i,j])
                    out_rows.append({"structure_id":sid,"composition_id":formula,
                                     "polymorph_family":FAMILIES[j],"fold":fold,"seed":seed,
                                     "n_atoms":int(source.n_atoms),
                                     "energy_total_dft_ev":float(source.energy_total_dft_ev),
                                     "energy_per_atom_dft_ev":float(source.energy_per_atom_dft_ev),
                                     "ridge_mean_ev_atom":float(means[i]),
                                     "residual_ev_atom":float(residual[i,j]),
                                     "energy_per_atom_pred_ev":per_atom,
                                     "energy_total_pred_ev":per_atom*int(source.n_atoms)})
    out = pd.DataFrame(out_rows)
    old_aligned = historical.merge(out, on=["structure_id","composition_id","polymorph_family","fold","seed"],
                                   validate="one_to_one")
    if len(old_aligned) != 630 or parameter_counts != {6161}:
        raise ValueError("Reproducción incompleta o arquitectura cambiada")
    differences = {
        "energy_per_atom":float(np.max(np.abs(old_aligned.E_pred_eV_atom-old_aligned.energy_per_atom_pred_ev))),
        "ridge_mean":float(np.max(np.abs(old_aligned.E_pred_mu_eV_atom-old_aligned.ridge_mean_ev_atom))),
        "residual":float(np.max(np.abs(old_aligned.delta_pred_centered_eV_atom-old_aligned.residual_ev_atom)))
    }
    status = "PASS" if max(differences.values()) < 1e-5 else "FAIL"
    out.to_csv(output / "replayed_predictions.csv", index=False)
    result = {"status":status,"rows":len(out),"physical_ids":out.structure_id.nunique(),
              "compositions":out.composition_id.nunique(),"seeds":list(SEEDS),
              "max_abs_error_ev_atom":differences,"structural_trainable_parameters":6161,
              "ridge_alphas_by_fold":ridge_alphas,"historical_test_accessed":False,
              "source_hashes":{str(p.relative_to(root)):sha256(p) for p in
                               (dataset_path,fold_path,cache_path,checkpoint_path,
                                old/"H3_C_ridge_retrained_delta/oof_predictions.csv")},
              "replayed_predictions_sha256":sha256(output/"replayed_predictions.csv")}
    (output/"reproduction_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    LOG.info(json.dumps({"event":"reproduction","status":status,"max_errors":differences}))
    if status != "PASS":raise ValueError(f"H3-C no se reprodujo: {differences}")
    return result


def main() -> None:
    """Ejecuta la reproducción histórica con raíz configurable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=project_root())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    replay(args.repo.resolve())


if __name__ == "__main__":
    main()
