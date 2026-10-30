"""Interfaz estable de datos y predicción para H3-C; sin geometría nueva."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())))
from Alternative2.scripts.MatGPT.MatGPT_frozen_3.hybrid_ridge_mlp8.hybrid_predictor import StructuralMLP8
from Alternative2.scripts.MatGPT.MatGPT_H3C_Final.reproduce import FAMILIES,feature_cache,read_development


@dataclass(frozen=True)
class H3CInputs:
    """Alinea identidad física y dos embeddings congelados de 768 dimensiones."""
    metadata:pd.DataFrame
    l03_mean:np.ndarray
    final_end:np.ndarray


def load_inputs(root:Path,include_dft:bool=False) -> H3CInputs:
    """Devuelve IDs, fórmula, N, SLICES y embeddings; DFT solo si se solicita."""
    root=root.resolve()
    data=read_development(root/"reports/MatGPT/MatGPT_frozen/matgpt_e1_v2/dataset.csv")
    path=root/"reports/MatGPT/MatGPT_frozen_3/run_20261023_v2/features_cache/layer_features.npz"
    features=feature_cache(path,data.structure_id.tolist())
    columns=["structure_id","composition_id","polymorph_family","n_atoms","slices"]
    if include_dft:columns.extend(["energy_total_dft_ev","energy_per_atom_dft_ev"])
    return H3CInputs(data[columns].copy(),features["L03_MEAN"],features["FINAL_END"])


def predict_triplets(root:Path,fold:int,seed:int,ids:list[str],bundle:H3CInputs) -> pd.DataFrame:
    """Aplica checkpoints históricos fijos; no usa DFT para inferencia."""
    if fold not in range(5) or seed not in (42,123,2026):raise ValueError("Fold/semilla inválidos")
    data=bundle.metadata
    byid=data.set_index("structure_id")
    if len(ids)%3 or len(set(ids))!=len(ids) or not set(ids)<=set(data.structure_id):
        raise ValueError("IDs incompletos o repetidos")
    subset=data[data.structure_id.isin(ids)]
    lookup={sid:i for i,sid in enumerate(data.structure_id)}
    formulas=sorted(subset.composition_id.unique())
    groups=[]
    for formula in formulas:
        part=subset[subset.composition_id==formula]
        if set(part.polymorph_family)!=set(FAMILIES):raise ValueError("Triplete incompleto")
        groups.append([part[part.polymorph_family==family].structure_id.iloc[0] for family in FAMILIES])
    old=root/"reports/MatGPT/MatGPT_frozen_3/hybrid_ridge_mlp8"
    ridge=joblib.load(old/f"H3_B_ridge_fixed_delta/ridge_by_fold/fold_{fold}/ridge.joblib")
    checkpoint=torch.load(old/f"H3_C_ridge_retrained_delta/runs/fold_{fold}_seed_{seed}/structural_mlp8.pt",
                          map_location="cpu",weights_only=True)
    mlp=StructuralMLP8(input_dim=768,dropout=.1)
    mlp.load_state_dict(checkpoint["state_dict"],strict=True);mlp.eval()
    mean_features=np.stack([bundle.l03_mean[[lookup[sid] for sid in group]].mean(axis=0) for group in groups])
    final_features=np.stack([bundle.final_end[[lookup[sid] for sid in group]] for group in groups])
    means=ridge.predict(mean_features)
    with torch.inference_mode():residual=mlp(torch.tensor(final_features,dtype=torch.float32)).numpy()
    # H3-C centra la salida cruda de la cabeza dentro de cada triplete.
    residual=residual-residual.mean(axis=1,keepdims=True)
    rows=[]
    for i,group in enumerate(groups):
        for j,sid in enumerate(group):
            n=int(byid.loc[sid,"n_atoms"]);e=float(means[i]+residual[i,j])
            rows.append({"structure_id":sid,"composition_id":formulas[i],"polymorph_family":FAMILIES[j],
                         "n_atoms":n,"energy_per_atom_pred_ev":e,"energy_total_pred_ev":n*e,
                         "ridge_mean_ev_atom":float(means[i]),"residual_ev_atom":float(residual[i,j])})
    return pd.DataFrame(rows)


def evaluate_intraformula(predictions:pd.DataFrame,reference:pd.DataFrame) -> pd.DataFrame:
    """Calcula residuos centrados únicamente tras alinear etiquetas de evaluación."""
    if "energy_per_atom_dft_ev" not in reference:raise ValueError("Falta referencia DFT de evaluación")
    joined=predictions.merge(reference[["structure_id","energy_per_atom_dft_ev"]],on="structure_id",validate="one_to_one")
    if len(joined)!=len(predictions):raise ValueError("Referencias incompletas")
    for name in ("pred","dft"):
        column=f"energy_per_atom_{name}_ev"
        joined[f"centered_{name}_ev_atom"]=joined[column]-joined.groupby("composition_id")[column].transform("mean")
    joined["residual_absolute_error_mev_atom"]=1000*np.abs(joined.centered_pred_ev_atom-joined.centered_dft_ev_atom)
    return joined
