#!/usr/bin/env python3
"""Evalúa baselines de energía DFT que solo reciben composición química."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, spearmanr

from Alternative2.scripts.MatGPT.MatGPT_diagnostics.common_data import ALT, REPORT, load_development, sha256, write_json

LOG = logging.getLogger("matgpt_diagnostics.d1")
E2_OOF = ALT / "reports/MatGPT_finetuning/matgpt_e2/runs_endtoken_v2/cpu/oof_predictions.csv"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def formula_features(formula: str, elements: list[str]) -> dict[str, float]:
    """Convierte fórmula a fracciones elementales y resúmenes conocidos.

    Args:
        formula: Fórmula reducida (ej. BaHfS3).
        elements: Vocabulario elemental descubierto solo en fórmulas.
    Returns:
        Diccionario de features independientes de energía y estructura.
    """
    composition = Composition(formula)
    fractions = composition.fractional_composition
    result = {f"fraction_{el}": float(fractions.get_el_amt_dict().get(el, 0.0)) for el in elements}
    props: list[tuple[float, float]] = []
    for el, amount in fractions.get_el_amt_dict().items():
        elem = Element(el)
        weight = float(amount)
        en = elem.X
        props.append((float(elem.Z), weight))
        if en is not None:
            props.append((float(en), weight))
    zvals = [(float(Element(el).Z), float(n)) for el, n in fractions.get_el_amt_dict().items()]
    envals = [(float(Element(el).X), float(n)) for el, n in fractions.get_el_amt_dict().items() if Element(el).X is not None]
    result.update({
        "n_elements": float(len(fractions.get_el_amt_dict())),
        "n_atoms_reduced": float(composition.num_atoms),
        "mean_atomic_number": float(sum(x * w for x, w in zvals)),
        "std_atomic_number": float(np.sqrt(sum(w * (x - sum(z * v for z, v in zvals)) ** 2 for x, w in zvals))),
        "min_atomic_number": min(x for x, _ in zvals),
        "max_atomic_number": max(x for x, _ in zvals),
    })
    if envals:
        mean_en = sum(x * w for x, w in envals) / sum(w for _, w in envals)
        result.update({"mean_electronegativity": float(mean_en),
                       "std_electronegativity": float(np.sqrt(sum(w * (x - mean_en) ** 2 for x, w in envals) / sum(w for _, w in envals))),
                       "min_electronegativity": min(x for x, _ in envals),
                       "max_electronegativity": max(x for x, _ in envals)})
    else:
        result.update({k: float("nan") for k in ("mean_electronegativity", "std_electronegativity", "min_electronegativity", "max_electronegativity")})
    return result


def metrics_for_group(group: pd.DataFrame) -> dict[str, float]:
    """Calcula métricas absolutas y relativas con abstención en ranking.

    Cada fila representa un polimorfo. GS omite DFT degenerado y predicción empatada.
    """
    y = group.energy_per_atom_ev.to_numpy(float) * 1000.0
    p = group.energy_pred_ev_atom.to_numpy(float) * 1000.0
    dft_order, pred_order = y.argsort(), p.argsort()
    gap_y = float(y[dft_order[1]] - y[dft_order[0]])
    gap_p = float(p[pred_order[1]] - p[pred_order[0]])
    dy, dp = y - y.min(), p - p.min()
    eligible = gap_y > 1.0
    pred_discriminative = gap_p > 1.0
    pairs = [(i, j) for i in range(len(y)) for j in range(i + 1, len(y)) if abs(y[i] - y[j]) > 1.0]
    pair_acc = [np.sign(y[i]-y[j]) == np.sign(p[i]-p[j]) for i, j in pairs]
    pair_errors = [((p[i]-p[j])-(y[i]-y[j])) for i,j in pairs]
    rank_error = rankdata(p, method="average")-rankdata(y, method="average")
    rho = spearmanr(y,p).statistic if np.ptp(p)>1e-12 and np.ptp(y)>1e-12 else np.nan
    return {"absolute_mae_mev_atom": float(np.abs(y-p).mean()),
            "delta_e_mae_mev_atom": float(np.abs(dy-dp).mean()),
            "ground_state_accuracy": float(dft_order[0] == pred_order[0] and pred_discriminative) if eligible else float("nan"),
            "gs_eligible": float(eligible), "pairwise_accuracy": float(np.mean(pair_acc)) if pair_acc else float("nan"),
            "predicted_tie": float(not pred_discriminative),
            "n_eligible_pairs": float(len(pairs)), "dft_range_mev_atom": float(np.ptp(y)),
            "pred_range_mev_atom": float(np.ptp(p)), "ranking_discriminative": float(np.ptp(p) > 1.0),
            "pairwise_energy_mae_mev_atom": float(np.mean(np.abs(pair_errors))) if pair_errors else float("nan"),
            "pairwise_signed_bias_mev_atom": float(np.mean(pair_errors)) if pair_errors else float("nan"),
            "spearman_by_composition": float(rho), "absolute_rank_error_mean": float(np.mean(np.abs(rank_error))),
            "signed_rank_error_mean": float(np.mean(rank_error))}


def run() -> None:
    """Entrena y evalúa cuatro modelos por folds agrupados por composición."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "D1")
    args = parser.parse_args()
    data, folds, meta = load_development()
    args.output.mkdir(parents=True, exist_ok=True)
    formulas = sorted(data.reduced_formula.astype(str).unique())
    elements = sorted({el for formula in formulas for el in Composition(formula).element_composition.get_el_amt_dict()})
    features = pd.DataFrame([formula_features(x, elements) for x in data.reduced_formula], index=data.index)
    features.insert(0, "structure_id", data.structure_id.astype(str))
    features.insert(1, "composition_group", data.composition_group.astype(str))
    features.insert(2, "reduced_formula", data.reduced_formula.astype(str))
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    matrices = features.drop(columns=["structure_id", "composition_group", "reduced_formula"]).replace([np.inf, -np.inf], np.nan)
    matrices = matrices.fillna(matrices.median(numeric_only=True)).to_numpy(float)
    y_all = data.energy_per_atom_ev.to_numpy(float)
    ids = data.structure_id.astype(str).to_numpy()
    groups = data.composition_group.astype(str).to_numpy()
    for fold_id, assignments in folds.groupby("outer_fold", sort=True):
        role = assignments.drop_duplicates("structure_id").set_index("structure_id")
        train_idx = np.flatnonzero(np.array([role.loc[s, "inner_role"] == "inner_train" for s in ids]))
        val_idx = np.flatnonzero(np.array([role.loc[s, "inner_role"] == "inner_validation" for s in ids]))
        outer_idx = np.flatnonzero(np.array([role.loc[s, "outer_role"] == "outer_validation" for s in ids]))
        if set(groups[train_idx]) & (set(groups[val_idx]) | set(groups[outer_idx])) or set(groups[val_idx]) & set(groups[outer_idx]):
            raise ValueError(f"Fuga de composición en fold {fold_id}.")
        if (len(train_idx), len(val_idx), len(outer_idx)) != (135, 33, 42):
            raise ValueError(f"Roles inesperados en fold {fold_id}: {len(train_idx)}/{len(val_idx)}/{len(outer_idx)}.")
        mean = float(y_all[train_idx].mean())
        models: dict[str, np.ndarray] = {"D1-A media train": np.full(len(outer_idx), mean)}
        scaler = StandardScaler().fit(matrices[train_idx])
        x_train, x_val = scaler.transform(matrices[train_idx]), scaler.transform(matrices[val_idx])
        best_alpha, best_mae = None, float("inf")
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, y_all[train_idx])
            val_mae = float(mean_absolute_error(y_all[val_idx], model.predict(x_val)))
            tuning.append({"fold": int(fold_id), "alpha": alpha, "inner_val_mae_ev_atom": val_mae})
            if val_mae < best_mae:
                best_alpha, best_mae = alpha, val_mae
        ridge = Ridge(alpha=float(best_alpha)).fit(x_train, y_all[train_idx])
        models["D1-B Ridge composición"] = ridge.predict(scaler.transform(matrices[outer_idx]))
        rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3,
                                   max_features=0.8, random_state=421 + int(fold_id), n_jobs=1)
        rf.fit(matrices[train_idx], y_all[train_idx])
        models["D1-C RandomForest pequeño"] = rf.predict(matrices[outer_idx])
        for name, pred in models.items():
            block = data.iloc[outer_idx][["structure_id", "composition_group", "reduced_formula", "polymorph_family", "energy_per_atom_ev"]].copy()
            block["configuration"] = name
            block["fold"] = int(fold_id)
            block["energy_pred_ev_atom"] = pred
            predictions.append(block)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    features.to_csv(args.output / "composition_features.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "ridge_inner_validation.csv", index=False)
    per_comp = []
    for (config, fold, comp), group in oof.groupby(["configuration", "fold", "composition_group"]):
        formula_mean_error = abs(float(group.energy_pred_ev_atom.mean())-float(group.energy_per_atom_ev.mean()))*1000.0
        per_comp.append({"configuration": config, "fold": int(fold), "composition_group": comp,
                         **metrics_for_group(group), "composition_mean_absolute_error_mev_atom": formula_mean_error,
                         "n_structures": len(group)})
    comp_metrics = pd.DataFrame(per_comp)
    comp_metrics.to_csv(args.output / "metrics_composition.csv", index=False)
    fold_metrics = comp_metrics.groupby(["configuration", "fold"]).agg(
        n_compositions=("composition_group", "nunique"),
        n_gs_eligible=("gs_eligible", "sum"),
        **{m: (m, "mean") for m in ["absolute_mae_mev_atom", "composition_mean_absolute_error_mev_atom", "delta_e_mae_mev_atom", "ground_state_accuracy", "pairwise_accuracy", "dft_range_mev_atom", "pred_range_mev_atom"]}
    ).reset_index()
    # Pooled absolute R² per outer fold: three polymorphs in one formula are not
    # independent replicates, so this is descriptive and paired with macro MAE.
    pooled_r2 = []
    for (configuration, fold), group in oof.groupby(["configuration", "fold"]):
        pooled_r2.append({"configuration": configuration, "fold": int(fold),
                          "absolute_r2_pooled_structures": float(r2_score(group.energy_per_atom_ev, group.energy_pred_ev_atom))})
    fold_metrics = fold_metrics.merge(pd.DataFrame(pooled_r2), on=["configuration", "fold"], validate="one_to_one")
    fold_metrics.to_csv(args.output / "metrics.csv", index=False)
    # Compare each chemistry-only estimate against the E2-A partial_ft average over its 3 seeds.
    if E2_OOF.is_file():
        e2 = pd.read_csv(E2_OOF)
        e2 = e2.loc[e2.configuration.eq("partial_ft")].groupby("structure_id", as_index=False).energy_pred_ev_atom.mean()
        e2 = e2.rename(columns={"energy_pred_ev_atom": "e2_partial_ft_pred_ev_atom"})
        joined = oof.merge(e2, on="structure_id", validate="many_to_one")
        compare = []
        for (configuration, fold, comp), group in joined.groupby(["configuration", "fold", "composition_group"]):
            dft = group.energy_per_atom_ev.to_numpy(float)
            base = group.e2_partial_ft_pred_ev_atom.to_numpy(float)
            compare.append({"configuration": configuration, "fold": int(fold), "composition_group": comp,
                            "d1_absolute_mae_mev_atom": float(np.abs(group.energy_pred_ev_atom.to_numpy()-dft).mean()*1000),
                            "e2a_absolute_mae_mev_atom": float(np.abs(base-dft).mean()*1000),
                            "d1_minus_e2a_mae_mev_atom": float((np.abs(group.energy_pred_ev_atom.to_numpy()-dft).mean()-np.abs(base-dft).mean())*1000)})
        pd.DataFrame(compare).to_csv(args.output / "paired_e2a_comparison.csv", index=False)
    meta.update({"script_sha256": sha256(Path(__file__)), "features": list(features.columns[3:]),
                 "ridge_alphas_inner_validation_only": list(ALPHAS), "random_forest_fixed": {"trees": 200, "depth": 4, "min_leaf": 3},
                 "no_composition_id_or_energy_features": True, "historical_test_used": False})
    write_json(args.output / "provenance.json", meta)
    LOG.info("d1_complete rows=%d folds=5 models=%d output=%s", len(oof), oof.configuration.nunique(), args.output)


def main() -> None:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
