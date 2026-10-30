"""Reconstruye predicciones outer y compara C0/C1 sin seleccionar con outer.

La unidad estadística para incertidumbre es la fórmula ABS3; las semillas son
repeticiones de entrenamiento sobre las mismas 210 estructuras físicas.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import experiment as ex
from lora_matgpt import insert_qv_lora

LOG = logging.getLogger("matgpt_lora_evaluation")
BINS = ((0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, float("inf")))


def ols(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calcula OLS descriptivo y R² predictivo como estadísticas distintas."""
    x = np.asarray(reference, dtype=float)
    y = np.asarray(prediction, dtype=float)
    if len(x) < 3 or np.std(x) <= 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2_ols": float("nan"), "r2_predictive": float("nan"),
                "std_ratio": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    variance = float(np.sum((y - y.mean()) ** 2))
    truth_variance = float(np.sum((x - x.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2_ols": 1 - float(np.sum((y - fitted) ** 2)) / variance if variance > 0 else float("nan"),
            "r2_predictive": 1 - float(np.sum((y - x) ** 2)) / truth_variance,
            "std_ratio": float(np.std(y) / np.std(x))}


def predict_all(output: Path) -> pd.DataFrame:
    """Evalúa una sola vez los checkpoints seleccionados para 30 corridas completas."""
    torch.set_num_threads(2)
    frame = ex.development(); folds = ex.assignments(frame); cache = ex.feature_cache(frame)
    model, stoi, block_size = ex.model_and_tokens()
    tokens = ex.encode(frame, stoi, block_size)
    lookup = {sid: i for i, sid in enumerate(frame.structure_id)}
    data = frame.set_index("structure_id")
    rows: list[dict[str, object]] = []
    base_checkpoint = ex.sha256(ex.CHECKPOINT)
    for fold in range(5):
        part = folds[(folds.outer_fold == fold) & (folds.outer_role == "outer_validation")]
        formulas = ex.formula_groups(frame, set(part.structure_id))
        ridge_file = ex.RIDGE / f"fold_{fold}/ridge.joblib"
        ridge = joblib.load(ridge_file)
        for seed in (42, 123, 2026):
            for condition in ("frozen", "lora"):
                run = output / "runs" / f"{condition}_fold{fold}_seed{seed}"
                summary_file = run / "summary.json"
                if not summary_file.is_file() or json.loads(summary_file.read_text()).get("status") != "COMPLETE":
                    raise ValueError("Falta corrida completa: " + str(run))
                checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                if checkpoint["checkpoint_sha256"] != base_checkpoint or checkpoint["tokenizer_sha256"] != ex.EXPECTED["tokenizer"]:
                    raise ValueError("Checkpoint base/tokenizer no coincide")
                head = ex.MLP8(); head.load_state_dict(checkpoint["head"], strict=True); head.eval()
                if condition == "lora":
                    adapters = insert_qv_lora(model)
                    for adapter, state in zip(adapters, checkpoint["adapters"]):
                        missing, unexpected = adapter.load_state_dict(state, strict=False)
                        if unexpected or set(missing) != set(k for k in adapter.state_dict() if k.startswith("base.")):
                            raise ValueError("Estado LoRA incompleto/incompatible")
                    model.eval()
                for formula, ids in formulas:
                    ridge_x = np.stack([cache["L03_MEAN"][lookup[sid]] for sid in ids]).mean(axis=0, keepdims=True)
                    ridge_mean = float(ridge.predict(ridge_x)[0])
                    with torch.inference_mode():
                        if condition == "frozen":
                            embedding = torch.from_numpy(np.stack([cache["FINAL_END"][lookup[sid]] for sid in ids]))
                        else:
                            embedding = ex.final_end(model, ids, tokens, stoi["<pad>"])
                        residual = head(embedding).detach().cpu().numpy()
                    for sid, delta in zip(ids, residual):
                        item = data.loc[sid]
                        value = ridge_mean + float(delta)
                        rows.append({"condition": condition, "fold": fold, "seed": seed,
                                     "structure_id": sid, "composition_id": formula,
                                     "polymorph_family": item.family, "n_atoms": int(item.n_atoms),
                                     "energy_total_dft_ev": float(item.total_dft_ev),
                                     "energy_per_atom_dft_ev": float(item.dft_ev_atom),
                                     "ridge_mean_ev_atom": ridge_mean,
                                     "residual_ev_atom": float(delta),
                                     "energy_per_atom_pred_ev": value,
                                     "energy_total_pred_ev": int(item.n_atoms) * value})
                if condition == "lora":
                    for index in (10, 11):
                        model.transformer_decoder.layers[index].self_attn = model.transformer_decoder.layers[index].self_attn.base
                LOG.info("%s", {"event": "outer_prediction", "condition": condition,
                               "fold": fold, "seed": seed, "formulas": len(formulas)})
    pred = pd.DataFrame(rows)
    if len(pred) != 1260:
        raise ValueError("Se esperaban 2 condiciones × 3 semillas × 210 estructuras")
    for (_condition, seed), group in pred.groupby(["condition", "seed"]):
        if len(group) != 210 or group.structure_id.nunique() != 210 or group.composition_id.nunique() != 70:
            raise ValueError("Predicciones outer incompletas")
    ref = pred[pred.condition == "frozen"].sort_values(["seed", "structure_id"])
    alt = pred[pred.condition == "lora"].sort_values(["seed", "structure_id"])
    for field in ("structure_id", "fold", "energy_per_atom_dft_ev", "n_atoms"):
        if not np.array_equal(ref[field].to_numpy(), alt[field].to_numpy()):
            raise ValueError("Controles no apareados: " + field)
    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "predicciones_outer_por_estructura.csv", index=False)
    historical_path = ex.ROOT / "reports/MatGPT/MatGPT_H3C_Final/reproduction/replayed_predictions.csv"
    if historical_path.is_file():
        historical = pd.read_csv(historical_path)
        frozen = pred[pred.condition == "frozen"]
        joined = frozen.merge(historical, on=["structure_id", "composition_id", "fold", "seed"],
                              suffixes=("_new", "_historical"), validate="one_to_one")
        if len(joined) != 630:
            raise ValueError("No se pudo alinear el control congelado con las 630 predicciones H3-C")
        if (not np.array_equal(joined.n_atoms_new.to_numpy(), joined.n_atoms_historical.to_numpy()) or
                not np.allclose(joined.energy_per_atom_dft_ev_new, joined.energy_per_atom_dft_ev_historical,
                                atol=1e-10, rtol=0) or
                not np.allclose(joined.energy_total_dft_ev_new, joined.energy_total_dft_ev_historical,
                                atol=1e-8, rtol=0)):
            raise ValueError("Los targets físicos del replay H3-C no coinciden")
        rows = []
        for seed, part in joined.groupby("seed"):
            difference = part.energy_per_atom_pred_ev_new - part.energy_per_atom_pred_ev_historical
            rows.append({"seed": int(seed), "n_structures": len(part),
                         "mae_prediction_difference_mev_atom": float(np.abs(difference).mean() * 1000),
                         "max_abs_prediction_difference_mev_atom": float(np.abs(difference).max() * 1000),
                         "mae_new_mev_atom": float(np.abs(part.energy_per_atom_pred_ev_new - part.energy_per_atom_dft_ev_new).mean() * 1000),
                         "mae_historical_mev_atom": float(np.abs(part.energy_per_atom_pred_ev_historical - part.energy_per_atom_dft_ev_historical).mean() * 1000)})
        pd.DataFrame(rows).to_csv(output / "reproduccion_control_congelado_h3c.csv", index=False)
    return pred


def embedding_drift(output: Path) -> pd.DataFrame:
    """Compara FINAL_END original y LoRA para 210 estructuras outer, semilla 42."""
    frame = ex.development(); folds = ex.assignments(frame); cache = ex.feature_cache(frame)
    model, stoi, block_size = ex.model_and_tokens()
    tokens = ex.encode(frame, stoi, block_size)
    lookup = {sid: i for i, sid in enumerate(frame.structure_id)}
    rows = []
    formula_rows = []
    global_original: list[np.ndarray] = []
    global_adapted: list[np.ndarray] = []
    for fold in range(5):
        checkpoint = torch.load(output / "runs" / f"lora_fold{fold}_seed42" / "best.pt",
                                map_location="cpu", weights_only=False)
        adapters = insert_qv_lora(model)
        for adapter, state in zip(adapters, checkpoint["adapters"]):
            missing, unexpected = adapter.load_state_dict(state, strict=False)
            if unexpected or set(missing) != set(k for k in adapter.state_dict() if k.startswith("base.")):
                raise ValueError("Estado LoRA incompatible en análisis de embeddings")
        model.eval()
        part = folds[(folds.outer_fold == fold) & (folds.outer_role == "outer_validation")]
        for formula, ids in ex.formula_groups(frame, set(part.structure_id)):
            with torch.inference_mode():
                adapted = ex.final_end(model, ids, tokens, stoi["<pad>"]).numpy()
            original = np.stack([cache["FINAL_END"][lookup[sid]] for sid in ids])
            orig_norm = np.linalg.norm(original, axis=1)
            adapted_norm = np.linalg.norm(adapted, axis=1)
            cosine = np.sum(original * adapted, axis=1) / (orig_norm * adapted_norm)
            global_original.extend(original)
            global_adapted.extend(adapted)
            pair_distance_original = []
            pair_distance_adapted = []
            pair_cosine_original = []
            pair_cosine_adapted = []
            for i, j in itertools.combinations(range(3), 2):
                cos_orig = float(np.dot(original[i], original[j]) / (orig_norm[i] * orig_norm[j]))
                cos_adapt = float(np.dot(adapted[i], adapted[j]) / (adapted_norm[i] * adapted_norm[j]))
                pair_cosine_original.append(cos_orig)
                pair_cosine_adapted.append(cos_adapt)
                pair_distance_original.append(float(np.linalg.norm(original[i] - original[j]) /
                                                   max((orig_norm[i] + orig_norm[j]) / 2, 1e-12)))
                pair_distance_adapted.append(float(np.linalg.norm(adapted[i] - adapted[j]) /
                                                  max((adapted_norm[i] + adapted_norm[j]) / 2, 1e-12)))
            formula_rows.append({"composition_id": formula,
                                 "mean_pairwise_cosine_distance_original": float(np.mean(1 - np.asarray(pair_cosine_original))),
                                 "mean_pairwise_cosine_distance_lora": float(np.mean(1 - np.asarray(pair_cosine_adapted))),
                                 "mean_normalized_euclidean_distance_original": float(np.mean(pair_distance_original)),
                                 "mean_normalized_euclidean_distance_lora": float(np.mean(pair_distance_adapted)),
                                 "variance_intraformula_original": float(np.var(original, axis=0).mean()),
                                 "variance_intraformula_lora": float(np.var(adapted, axis=0).mean())})
            for index, sid in enumerate(ids):
                rows.append({"structure_id": sid, "composition_id": formula, "fold": fold,
                             "seed": 42, "cosine_similarity": float(cosine[index]),
                             "normalized_euclidean_distance": float(np.linalg.norm(adapted[index] - original[index]) /
                                                                    max(orig_norm[index], 1e-12)),
                             "original_norm": float(orig_norm[index]),
                             "adapted_norm": float(adapted_norm[index])})
        for index in (10, 11):
            model.transformer_decoder.layers[index].self_attn = model.transformer_decoder.layers[index].self_attn.base
    result = pd.DataFrame(rows)
    if len(result) != 210 or result.structure_id.nunique() != 210:
        raise ValueError("Drift de embeddings incompleto")
    result.to_csv(output / "cambio_embeddings_outer_semilla42.csv", index=False)
    formula_frame = pd.DataFrame(formula_rows)
    formula_frame.to_csv(output / "distancias_embeddings_intraformula_semilla42.csv", index=False)
    def effective_rank(values: np.ndarray) -> float:
        """Rango efectivo espectral exp(entropía) de la covarianza centrada."""
        centered = np.asarray(values, dtype=np.float64)
        centered -= centered.mean(axis=0, keepdims=True)
        scale = float(np.max(np.abs(centered)))
        if not np.isfinite(scale):
            raise ValueError("Embeddings no finitos al calcular el rango efectivo")
        if scale == 0:
            return 0.0
        # SVD de la matriz escalada evita desbordamientos de centered @ centered.T
        # y conserva las proporciones espectrales usadas por exp(entropía).
        singular_values = np.linalg.svd(centered / scale, compute_uv=False)
        eigenvalues = singular_values ** 2
        probabilities = eigenvalues / eigenvalues.sum()
        probabilities = probabilities[probabilities > 0]
        return float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    original_all = np.stack(global_original)
    adapted_all = np.stack(global_adapted)
    summary = {"n_structures": len(result), "n_formulas": len(formula_frame),
               "seed": 42, "folds": 5,
               "variance_across_structures_original": float(original_all.var(axis=0).mean()),
               "variance_across_structures_lora": float(adapted_all.var(axis=0).mean()),
               "effective_rank_original": effective_rank(original_all),
               "effective_rank_lora": effective_rank(adapted_all),
               "mean_intraformula_cosine_distance_original": float(formula_frame.mean_pairwise_cosine_distance_original.mean()),
               "mean_intraformula_cosine_distance_lora": float(formula_frame.mean_pairwise_cosine_distance_lora.mean()),
               "mean_intraformula_normalized_euclidean_original": float(formula_frame.mean_normalized_euclidean_distance_original.mean()),
               "mean_intraformula_normalized_euclidean_lora": float(formula_frame.mean_normalized_euclidean_distance_lora.mean())}
    ex.atomic_json(output / "resumen_rango_y_varianza_embeddings_semilla42.json", summary)
    return result


def metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mide energía, residuos, pares, GS y bins con denominadores explícitos."""
    summary_rows = []; composition_rows = []; pair_rows = []; bin_rows = []
    for (condition, seed), group in pred.groupby(["condition", "seed"]):
        truth = group.energy_per_atom_dft_ev.to_numpy(float)
        estimated = group.energy_per_atom_pred_ev.to_numpy(float)
        all_residual_truth = []; all_residual_pred = []
        all_gap_truth = []; all_gap_pred = []; anchored = []
        eligible_gs = 0; correct_gs = 0; exact_gs = 0
        for formula, part in group.groupby("composition_id"):
            trip = part.set_index("polymorph_family").loc[list(ex.FAMILIES)]
            y = trip.energy_per_atom_dft_ev.to_numpy(float)
            p = trip.energy_per_atom_pred_ev.to_numpy(float)
            ry = y - y.mean(); rp = p - p.mean()
            all_residual_truth.extend(ry); all_residual_pred.extend(rp)
            gs = int(np.argmin(y)); predicted_gs = int(np.argmin(p))
            anchored.extend(1000 * ((p - p[gs]) - (y - y[gs])))
            dft_margin = float(np.partition(y, 1)[1] - y[gs])
            pred_margin = float(np.partition(p, 1)[1] - p[predicted_gs])
            gs_eligible = dft_margin > 0.001
            gs_correct = gs_eligible and pred_margin > 0.001 and gs == predicted_gs
            eligible_gs += int(gs_eligible); correct_gs += int(gs_correct)
            exact_gs += int(gs == predicted_gs)
            errors = 1000 * (p - y)
            local_pair_errors = []; local_pair_eligible = 0; local_pair_correct = 0
            for i, j in itertools.combinations(range(3), 2):
                dy = 1000 * (y[j] - y[i]); dp = 1000 * (p[j] - p[i])
                eligible = abs(dy) > 5
                correct = eligible and abs(dp) > 1 and np.sign(dy) == np.sign(dp)
                all_gap_truth.append(dy); all_gap_pred.append(dp)
                local_pair_errors.append(abs(dp - dy))
                local_pair_eligible += int(eligible); local_pair_correct += int(correct)
                pair_rows.append({"condition": condition, "seed": seed, "fold": int(trip.fold.iloc[0]),
                                  "composition_id": formula, "family_a": ex.FAMILIES[i], "family_b": ex.FAMILIES[j],
                                  "delta_dft_mev_atom": dy, "delta_pred_mev_atom": dp,
                                  "gap_absolute_dft_mev_atom": abs(dy), "absolute_error_mev_atom": abs(dp - dy),
                                  "zero_baseline_error_mev_atom": abs(dy),
                                  "eligible_gt5": int(eligible), "correct_gt5": int(correct)})
            composition_rows.append({"condition": condition, "seed": seed, "fold": int(trip.fold.iloc[0]),
                                     "composition_id": formula,
                                     "absolute_mae_mev_atom": float(np.abs(errors).mean()),
                                     "composition_mean_error_mev_atom": float(abs(errors.mean())),
                                     "residual_mae_mev_atom": float(np.abs((rp - ry) * 1000).mean()),
                                     "pair_mae_mev_atom": float(np.mean(local_pair_errors)),
                                     "gs_eligible": int(gs_eligible), "gs_correct": int(gs_correct),
                                     "gs_exact_argmin": int(gs == predicted_gs),
                                     "pairs_eligible": local_pair_eligible, "pairs_correct": local_pair_correct})
        gap_y = np.asarray(all_gap_truth); gap_p = np.asarray(all_gap_pred)
        residual_y = np.asarray(all_residual_truth); residual_p = np.asarray(all_residual_pred)
        eligible_pairs = np.abs(gap_y) > 5
        summary_rows.append({"condition": condition, "seed": seed, "n_structures": 210,
                             "n_formulas": 70, "n_pairs": 210,
                             "n_pairs_eligible_gt5": int(eligible_pairs.sum()),
                             "mae_total_ev": float(np.abs(group.energy_total_pred_ev - group.energy_total_dft_ev).mean()),
                             "mae_per_atom_mev": float(np.abs(estimated - truth).mean() * 1000),
                             "rmse_per_atom_mev": float(np.sqrt(np.mean((estimated - truth) ** 2)) * 1000),
                             "bias_per_atom_mev": float(np.mean(estimated - truth) * 1000),
                             "mae_residual_mev": float(np.abs(residual_p - residual_y).mean() * 1000),
                             "rmse_residual_mev": float(np.sqrt(np.mean((residual_p - residual_y) ** 2)) * 1000),
                             "mae_pair_mev": float(np.abs(gap_p - gap_y).mean()),
                             "rmse_pair_mev": float(np.sqrt(np.mean((gap_p - gap_y) ** 2))),
                             "mae_zero_pair_mev": float(np.abs(gap_y).mean()),
                             "mae_anchored_gs_dft_mev": float(np.abs(anchored).mean()),
                             "gs_exact_argmin_accuracy": exact_gs / 70,
                             "gs_eligible_accuracy": correct_gs / eligible_gs if eligible_gs else float("nan"),
                             "n_gs_eligible": eligible_gs,
                             "pairwise_accuracy_gt5": float(np.mean((np.sign(gap_p[eligible_pairs]) ==
                                                                   np.sign(gap_y[eligible_pairs])) &
                                                                  (np.abs(gap_p[eligible_pairs]) > 1))),
                             **{"energy_" + k: v for k, v in ols(truth, estimated).items()},
                             **{"residual_" + k: v for k, v in ols(residual_y, residual_p).items()},
                             **{"gap_" + k: v for k, v in ols(gap_y, gap_p).items()}})
        for lo, hi in BINS:
            mask = (np.abs(gap_y) >= lo if lo == 0 else np.abs(gap_y) > lo) & (np.abs(gap_y) <= hi)
            select = mask & eligible_pairs
            bin_rows.append({"condition": condition, "seed": seed, "gap_min_mev": lo,
                             "gap_max_mev": hi, "n_pairs": int(mask.sum()),
                             "mae_pair_mev": float(np.abs(gap_p[mask] - gap_y[mask]).mean()) if mask.any() else float("nan"),
                             "mae_zero_mev": float(np.abs(gap_y[mask]).mean()) if mask.any() else float("nan"),
                             "n_pairs_ranking": int(select.sum()),
                             "pair_accuracy": float(np.mean((np.sign(gap_p[select]) == np.sign(gap_y[select])) &
                                                            (np.abs(gap_p[select]) > 1))) if select.any() else float("nan")})
    return (pd.DataFrame(summary_rows), pd.DataFrame(composition_rows),
            pd.DataFrame(pair_rows), pd.DataFrame(bin_rows))


def paired_bootstrap(composition: pd.DataFrame, repetitions: int = 4000) -> pd.DataFrame:
    """Remuestrea 70 fórmulas; promedia semillas dentro de cada fórmula."""
    rng = np.random.default_rng(20261029)
    rows = []
    for metric in ("absolute_mae_mev_atom", "composition_mean_error_mev_atom",
                   "residual_mae_mev_atom", "pair_mae_mev_atom", "gs_exact_argmin"):
        wide = composition.pivot_table(index="composition_id", columns="condition", values=metric, aggfunc="mean")
        if len(wide) != 70 or wide.isna().any().any():
            raise ValueError("Bootstrap requiere 70 fórmulas apareadas")
        difference = (wide["lora"] - wide["frozen"]).to_numpy(float)
        samples = difference[rng.integers(0, 70, size=(repetitions, 70))].mean(axis=1)
        rows.append({"metric": metric, "difference_lora_minus_frozen": float(difference.mean()),
                     "ci95_lower": float(np.percentile(samples, 2.5)),
                     "ci95_upper": float(np.percentile(samples, 97.5)),
                     "n_formulas": 70, "seeds_averaged_first": 3})
    for name, numerator, denominator in (("pairwise_accuracy_gt5", "pairs_correct", "pairs_eligible"),
                                         ("gs_eligible_accuracy", "gs_correct", "gs_eligible")):
        wide_num = composition.pivot_table(index="composition_id", columns="condition", values=numerator, aggfunc="mean")
        wide_den = composition.pivot_table(index="composition_id", columns="condition", values=denominator, aggfunc="mean")
        if not np.array_equal(wide_den["frozen"], wide_den["lora"]):
            raise ValueError("Denominadores de ranking desalineados")
        diff = (wide_num["lora"] - wide_num["frozen"]).to_numpy(float)
        den = wide_den["frozen"].to_numpy(float)
        indexes = rng.integers(0, 70, size=(repetitions, 70))
        draws = diff[indexes].sum(axis=1) / den[indexes].sum(axis=1)
        rows.append({"metric": name, "difference_lora_minus_frozen": float(diff.sum() / den.sum()),
                     "ci95_lower": float(np.percentile(draws, 2.5)),
                     "ci95_upper": float(np.percentile(draws, 97.5)),
                     "n_formulas": 70, "seeds_averaged_first": 3})
    return pd.DataFrame(rows)


def main() -> None:
    """Valida corridas, exporta predicciones, métricas y bootstrap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = args.output.resolve()
    pred = predict_all(output)
    summary, composition, pairs, bins = metrics(pred)
    summary.to_csv(output / "metricas_por_semilla.csv", index=False)
    composition.to_csv(output / "metricas_por_composicion.csv", index=False)
    pairs.to_csv(output / "diferencias_por_par.csv", index=False)
    bins.to_csv(output / "intervalos_gap_dft.csv", index=False)
    paired_bootstrap(composition).to_csv(output / "bootstrap_apareado.csv", index=False)
    embedding_drift(output)
    print(summary.groupby("condition").mean(numeric_only=True)[["mae_per_atom_mev", "mae_residual_mev",
                                                              "mae_pair_mev", "gs_eligible_accuracy",
                                                              "pairwise_accuracy_gt5"]].to_string())


if __name__ == "__main__":
    main()
