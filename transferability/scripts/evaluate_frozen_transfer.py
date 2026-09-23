#!/usr/bin/env python3
"""Evaluate frozen embeddings with identical grouped CV and ridge regressors."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TRANSFERABILITY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRANSFERABILITY_ROOT.parent
DEFAULT_TABLE = TRANSFERABILITY_ROOT / "data" / "SMILES" / "composition_smiles.csv"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "artifacts" / "embeddings"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "evaluation" / "frozen_transfer.json"
BRACKET_ATOM = re.compile(r"\[([A-Z][a-z]?)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="*",
        default=[
            DEFAULT_EMBEDDINGS / "chemberta_composition_smiles.npz",
            DEFAULT_EMBEDDINGS / "molformer_composition_smiles.npz",
        ],
    )
    parser.add_argument(
        "--target",
        choices=["energy_per_atom", "relaxed_energy_per_atom", "bandgap"],
        default="energy_per_atom",
    )
    parser.add_argument("--calculation", default="gga-static")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--representation-column",
        default="composition_smiles",
        help="Input-string column used to quantify representation collisions.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metrics(y_true, y_pred, mean_absolute_error, mean_squared_error, r2_score):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def elemental_fractions(smiles_values, np):
    """Create a transparent composition baseline from disconnected atom tokens."""
    counts = [Counter(BRACKET_ATOM.findall(smiles)) for smiles in smiles_values]
    elements = sorted({element for row in counts for element in row})
    element_index = {element: index for index, element in enumerate(elements)}
    matrix = np.zeros((len(counts), len(elements)), dtype=np.float64)
    for row_index, row_counts in enumerate(counts):
        total = sum(row_counts.values())
        if total == 0:
            raise ValueError(f"Could not parse composition SMILES: {smiles_values[row_index]!r}")
        for element, count in row_counts.items():
            matrix[row_index, element_index[element]] = count / total
    return matrix, elements


def cross_validated_ridge(
    x,
    y,
    splits,
    alpha,
    np,
    make_pipeline,
    StandardScaler,
    Ridge,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
):
    predictions = x[:, 0].astype(float, copy=True)
    fold_metrics = []
    for fold, (train, test) in enumerate(splits):
        # LSQR is stable for the high-dimensional, strongly collinear frozen
        # embeddings (many dimensions have nearly zero variance here).
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        regressor.fit(x[train], y[train])
        # Some Accelerate/BLAS combinations emit floating-point warnings from
        # matmul even when inputs, coefficients and outputs are all finite.
        # Suppress that noisy low-level warning, then enforce finiteness.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            fold_predictions = regressor.predict(x[test])
        if not np.isfinite(fold_predictions).all():
            raise FloatingPointError("Non-finite Ridge predictions")
        predictions[test] = fold_predictions
        fold_metrics.append(
            {
                "fold": fold,
                **metrics(y[test], predictions[test], mean_absolute_error, mean_squared_error, r2_score),
            }
        )
    return {
        "overall": metrics(y, predictions, mean_absolute_error, mean_squared_error, r2_score),
        "folds": fold_metrics,
    }


def main() -> None:
    try:
        import numpy as np
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Missing numpy/scikit-learn runtime dependencies.") from exc

    args = parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    selected = [
        (index, row)
        for index, row in enumerate(all_rows)
        if args.calculation == "all" or row["calculation"] == args.calculation
    ]
    if not selected:
        raise SystemExit(f"No rows match calculation={args.calculation!r}")
    if args.representation_column not in selected[0][1]:
        raise SystemExit(
            f"Column {args.representation_column!r} is absent from {args.table}"
        )

    selected_indices = np.asarray([index for index, _ in selected])
    selected_rows = [row for _, row in selected]
    record_ids = np.asarray([row["record_id"] for row in selected_rows])
    groups = np.asarray([row["canonical_reduced_formula"] for row in selected_rows])
    y = np.asarray([float(row[args.target]) for row in selected_rows], dtype=np.float64)
    unique_groups = np.unique(groups)
    if len(unique_groups) < args.folds:
        raise SystemExit(f"Need at least {args.folds} formula groups; found {len(unique_groups)}")

    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    result = {
        "target": args.target,
        "calculation": args.calculation,
        "representation_column": args.representation_column,
        "records": len(selected_rows),
        "unique_formula_groups": len(unique_groups),
        "cv": {
            "type": "GroupKFold",
            "group": "canonical_reduced_formula",
            "folds": args.folds,
            "shuffle": True,
            "seed": args.seed,
        },
        "regressor": {
            "type": "StandardScaler + Ridge",
            "solver": "lsqr",
            "alpha": args.alpha,
        },
        "models": {},
    }

    # The mean predictor establishes whether an embedding adds predictive signal.
    dummy_predictions = np.empty_like(y)
    for train, test in splits:
        dummy = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), y[train])
        dummy_predictions[test] = dummy.predict(np.zeros((len(test), 1)))
    result["dummy_mean"] = metrics(
        y, dummy_predictions, mean_absolute_error, mean_squared_error, r2_score
    )

    representation_groups = np.asarray(
        [row[args.representation_column] for row in selected_rows]
    )
    unique_representation_groups = np.unique(representation_groups)
    group_means = {
        group: y[representation_groups == group].mean()
        for group in unique_representation_groups
    }
    collision_predictions = np.asarray(
        [group_means[group] for group in representation_groups]
    )
    result["representation_collision_floor"] = {
        "description": (
            "In-sample mean for records sharing the exact representation string; "
            "not a predictive baseline. It quantifies target ambiguity caused by "
            "representation collisions."
        ),
        "unique_representation_strings": int(len(unique_representation_groups)),
        **metrics(y, collision_predictions, mean_absolute_error, mean_squared_error, r2_score),
    }

    composition_x, elements = elemental_fractions(
        [row["composition_smiles"] for row in selected_rows], np
    )
    result["elemental_fraction_ridge"] = {
        "dimensions": len(elements),
        "elements": elements,
        **cross_validated_ridge(
            composition_x,
            y,
            splits,
            args.alpha,
            np,
            make_pipeline,
            StandardScaler,
            Ridge,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        ),
    }

    for embedding_path in args.embeddings:
        with np.load(embedding_path, allow_pickle=False) as archive:
            all_record_ids = archive["record_ids"]
            if not np.array_equal(all_record_ids[selected_indices], record_ids):
                raise ValueError(f"Record order mismatch in {embedding_path}")
            # Float64 avoids numerical overflow when StandardScaler encounters
            # embedding dimensions with extremely small variance.
            x = archive["embeddings"][selected_indices].astype(np.float64, copy=False)
            alias = str(archive["model_alias"])

        result["models"][alias] = {
            "embedding_file": str(embedding_path.resolve()),
            "embedding_dimensions": int(x.shape[1]),
            **cross_validated_ridge(
                x,
                y,
                splits,
                args.alpha,
                np,
                make_pipeline,
                StandardScaler,
                Ridge,
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
