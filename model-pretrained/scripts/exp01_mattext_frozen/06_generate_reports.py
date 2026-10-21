"""Generate real-data audit/scientific figures and an honest experiment status report.

Always report completed audits. Embedding/test figures are created only when
corresponding real artifacts exist; missing results are explicitly pending.
"""
from __future__ import annotations

import itertools
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import (LOGGER, audit_fingerprint, cli, load_config, load_embeddings,
                   read_json, sha256, write_json)


def save_figure(directory: Any, name: str, title: str, xlabel: str, ylabel: str) -> None:
    """Label and save current real-data plot at 240 dpi; return None."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(directory / name, dpi=240)
    plt.close()


def generate(config: dict[str, Any]) -> None:
    """Build reports from artifacts that actually exist; never fabricate downstream results."""
    paths = config["paths"]
    report_dir = paths["reports"]
    figures, tables = report_dir / "figures", report_dir / "tables"
    frame = pd.read_csv(paths["processed"] / "dataset.csv")
    dataset = read_json(report_dir / "audit" / "dataset_audit.json")
    tokenizer = read_json(report_dir / "audit" / "tokenizer_audit.json")
    inventory = read_json(report_dir / "audit" / "inventory.json")
    token_rows = pd.read_csv(report_dir / "audit" / "tokenizer_per_structure.csv")
    splits = pd.read_csv(paths["processed"] / "split_assignments.csv")
    plt.figure()
    for family, group in frame.groupby("polymorph_family"):
        plt.hist(group.energy_per_atom_ev, bins=20, alpha=0.5, label=family)
    plt.legend()
    save_figure(figures, "01_energy_distribution.png", "Static GGA energies — provisional audited subset", "Energy (eV/atom)", "Structures")
    plt.figure()
    plt.hist(token_rows.loc[token_rows.candidate_valid, "length"], bins=20)
    plt.axvline(config["tokenizer"]["max_length"], color="red", label="Checkpoint context")
    plt.legend()
    save_figure(figures, "02_slices_sequence_lengths.png", "Candidate tokenization — checkpoint mapping unverified", "Tokens including candidate CLS/SEP", "Structures")
    plt.figure()
    plt.bar(["Lossless candidate", "Rejected"], [int(token_rows.candidate_valid.sum()), int((~token_rows.candidate_valid).sum())])
    save_figure(figures, "03_tokenizer_coverage.png", "Candidate tokenizer coverage — not a checkpoint-ID validation", "Audit outcome", "Structures")
    embeddings_ready = (paths["embeddings"] / "embedding_manifest.json").exists()
    if embeddings_ready:
        metadata, arrays = load_embeddings(config)
        x = arrays[config["extraction"]["pooling"]]
        norms = np.linalg.norm(x, axis=1)
        plt.figure()
        plt.hist(norms, bins=20)
        save_figure(figures, "04_embedding_norm_distribution.png", "Frozen MatText feature norms", "L2 norm (dimensionless)", "Structures")
        similarities = []
        for formula, group in metadata.groupby("reduced_formula"):
            for a, b in itertools.combinations(group.index, 2):
                similarities.append({"reduced_formula": formula,
                    "cosine_similarity": float(x[a] @ x[b] / max(norms[a] * norms[b], 1e-12)),
                    "euclidean_distance": float(np.linalg.norm(x[a] - x[b])),
                    "absolute_energy_difference_ev_atom": abs(float(metadata.loc[a, "energy_per_atom_ev"] - metadata.loc[b, "energy_per_atom_ev"]))})
        similarities = pd.DataFrame(similarities)
        similarities.to_csv(tables / "embedding_polymorph_pairs.csv", index=False)
        plt.figure()
        plt.hist(similarities.cosine_similarity, bins=20)
        save_figure(figures, "05_polymorph_cosine_similarity.png", "Polymorph similarity within composition", "Cosine similarity", "Pairs")
        development = metadata.split.eq("development").to_numpy()
        pca = PCA(n_components=2).fit(x[development])
        coordinates = pca.transform(x)
        plt.figure()
        for split in ["development", "test"]:
            selected = metadata.split.eq(split)
            plt.scatter(*coordinates[selected].T, label=split, s=15)
        plt.legend()
        save_figure(figures, "06_embedding_pca.png", "PCA fitted on development features only — exploratory", "PC1", "PC2")
        plt.figure()
        plt.scatter(similarities.euclidean_distance, similarities.absolute_energy_difference_ev_atom, s=12)
        save_figure(figures, "16_distance_vs_energy.png", "Exploratory feature distance and GGA energy difference", "Feature Euclidean distance", "Absolute difference (eV/atom)")
        plt.figure()
        plt.hist(similarities.euclidean_distance, bins=20)
        save_figure(figures, "17_embedding_distance_distribution.png", "Within-composition feature distances", "Euclidean distance", "Polymorph pairs")
    evaluated = (report_dir / "metrics" / "evaluation_manifest.json").exists()
    if evaluated:
        predictions = pd.read_csv(tables / "predictions_test.csv")
        metrics = pd.read_csv(tables / "metrics_summary.csv")
        by_comp = pd.read_csv(tables / "metrics_by_composition.csv")
        by_family = pd.read_csv(tables / "metrics_by_polymorph.csv")
        cv = pd.read_csv(tables / "cross_validation_results.csv")
        relative = pd.read_csv(tables / "relative_energy_pairs.csv")
        curves = pd.read_csv(paths["experiment"] / "artifacts" / "mlp_training_curves.csv")
        plt.figure()
        for model, group in predictions.groupby("model"):
            plt.scatter(group.energy_per_atom_ev, group.prediction_ev_atom, label=model, s=14)
        limits = [predictions.energy_per_atom_ev.min(), predictions.energy_per_atom_ev.max()]
        plt.plot(limits, limits, color="black", linestyle="--")
        plt.legend()
        save_figure(figures, "07_dft_vs_predicted_energy.png", "Independent test compositions", "Static GGA (eV/atom)", "Prediction (eV/atom)")
        plt.figure()
        for model, group in predictions.groupby("model"):
            plt.hist(group.residual_ev_atom, bins=15, alpha=0.5, label=model)
        plt.legend()
        save_figure(figures, "08_energy_residuals.png", "Independent test residuals", "Prediction − GGA (eV/atom)", "Structures")
        energy_models = metrics.loc[metrics.model.ne("frequent_prototype")].set_index("model")
        energy_models[["mae_ev_atom", "rmse_ev_atom"]].plot.bar(rot=0)
        save_figure(figures, "09_mae_rmse_comparison.png", "Independent test energy errors", "Predictor", "Error (eV/atom)")
        metrics.set_index("model").minimum_selection_accuracy.plot.bar(rot=15)
        save_figure(figures, "10_polymorph_selection_accuracy.png", "Minimum-energy selection, quasi-degeneracy policy", "Predictor / baseline", "Composition accuracy")
        plt.figure()
        for model, group in relative.groupby("model"):
            plt.hist(group.absolute_delta_error_ev_atom * 1000, alpha=0.5, bins=15, label=model)
        plt.legend()
        save_figure(figures, "11_relative_energy_errors.png", "Within-composition energy-difference errors", "Absolute error (meV/atom)", "Pairs")
        by_comp.boxplot(column="selection_regret_mev_atom", by="model", rot=15)
        plt.suptitle("")
        save_figure(figures, "12_selection_regret.png", "GGA penalty of selected polymorph", "Predictor / baseline", "Regret (meV/atom)")
        by_family.pivot(index="polymorph_family", columns="model", values="mae_ev_atom").plot.bar(rot=15)
        save_figure(figures, "13_error_by_polymorph_family.png", "Test energy error by initial prototype family", "Family", "MAE (eV/atom)")
        cv.loc[cv.selected_for_final].boxplot(column="mae_ev_atom", by="model")
        plt.suptitle("")
        save_figure(figures, "14_cross_validation_variability.png", "Development CV — selected hyperparameters", "Predictor", "Fold MAE (eV/atom)")
        curves.plot(x="epoch", y=["train_mse_standardized", "validation_mse_standardized"])
        save_figure(figures, "15_mlp_training_curves.png", "MLP internal development early stopping", "Epoch", "MSE in standardized target units")
    blockers = dataset["blockers"] + tokenizer["blockers"]
    preflight_path = report_dir / "audit" / "preflight.json"
    if preflight_path.exists():
        blockers = list(dict.fromkeys(blockers + read_json(preflight_path).get("blockers", [])))
    manifest = {"experiment_id": config["experiment_id"], "status": "evaluated" if evaluated else "blocked" if blockers else "ready",
                "fingerprint": audit_fingerprint(config), "environment": inventory["packages"],
                "python": inventory["python"], "git_revision": inventory["git_revision"],
                "seed": config["seed"], "blockers": blockers, "embeddings_produced": embeddings_ready,
                "scientific_metrics_produced": evaluated,
                "figures": [str(path.relative_to(config["root"])) for path in sorted(figures.glob("*.png"))],
                "code_sha256": {str(path.relative_to(config["root"])): sha256(path)
                                for path in sorted((config["root"] / "model-pretrained/scripts/exp01_mattext_frozen").glob("*.py"))}}
    jobs = paths["experiment"] / "slurm_jobs.json"
    manifest["slurm"] = read_json(jobs) if jobs.exists() else {"submitted": False, "job_ids": []}
    write_json(paths["experiment"] / "experiment_manifest.json", manifest)
    write_json(report_dir / "metrics" / "experiment_manifest.json", manifest)
    sections = ["# Experimento 01 — MatText congelado", "## Objetivo científico",
        "Evaluar energía GGA por átomo y selección polimórfica en composiciones no vistas, sin actualizar el encoder.",
        "## Procedencia y auditoría de datos",
        f"{dataset['source_entries']} cálculos originales ({dataset['calculation_types']}); {len(frame)} filas provisionales y {frame.reduced_formula.nunique()} composiciones con las tres familias. No constituyen una autorización de entrenamiento.",
        dataset["energy_total_policy"], dataset["family_policy"],
        f"Las relajaciones no son muestras adicionales. Los filtros por grafos/candidatos incompletos se registran por ID. Asociaciones ambiguas: {dataset['energy_ambiguities']}; ver energy_ambiguities.csv. No se cambió la tolerancia para admitir discrepancias.",
        "## SLICES y tokenización",
        f"SLICES 2.0.12 oficial, econnn y relax_model=None. {tokenizer['candidate_tokenization_valid']} conversiones sintácticas sin pérdida; API oficial verificada: {tokenizer['official_api_verified']}. Longitud máxima candidata {tokenizer['max_candidate_length']}; {tokenizer['candidate_input_collisions']} colisiones candidatas de IDs entre polimorfos.",
        "Vocabulario de Zenodo recuperado y hash verificado. No está demostrada su asignación de tokens especiales para este checkpoint. Ver audit/tokenizer_compatibility.md.",
        "## MatText y extracción prevista",
        "BertModel: 4 bloques, 8 cabezas, dimensión 512, FFN 3072, 30522 filas de embeddings, contexto 512. Hash del checkpoint verificado contra el manifiesto original. Código exige requires_grad=False, eval e inference_mode, cero parámetros entrenables. Mean excluye CLS/SEP y padding por máscaras de posición; CLS se conserva como alternativa.",
        f"Embeddings científicos producidos: {embeddings_ready}. No se usa el vocabulario candidato para extraer resultados científicos.",
        "## Predictores y validación prevista",
        "Ridge, Random Forest y MLP 512→32→1 (GELU/dropout, AdamW) sobre los mismos embeddings. Selección de pooling común por MAE de MLP en CV de desarrollo. Hiperparámetros por CV; normalización dentro de cada entrenamiento. MLP usa validación interna agrupada para early stopping y refit por el número de épocas elegido. El test no selecciona pooling, cabeza ni hiperparámetros.",
        f"Partición provisional: {splits.loc[splits.split.eq('development')].reduced_formula.nunique()} composiciones de desarrollo / {splits.loc[splits.split.eq('test')].reduced_formula.nunique()} de test, {config['split']['folds']} pliegues; semilla {config['seed']}. Se deberá regenerar/revalidar si cambia el conjunto admitido.",
        "## Evaluación y estabilidad polimórfica",
        "Se implementan MAE/RMSE (eV/átomo), diferencias entre polimorfos, concordancia por pares, selección del mínimo y regret (meV/átomo). Cuasidegeneración: 1 meV/átomo. Baseline de familia frecuente calculado solo con mínimos GGA del entrenamiento.",
        "## Resultados cuantitativos",
        "No hay métricas supervisadas ni trabajos enviados: los controles críticos no pasaron." if not evaluated else pd.read_csv(tables / "metrics_summary.csv").to_markdown(index=False),
        "## Limitaciones e interpretación",
        "El solapamiento con el corpus NOMAD de preentrenamiento es desconocido. Una energía absoluta precisa no demostraría sensibilidad geométrica ni ordenamiento polimórfico. Las familias corresponden al prototipo inicial y pueden cambiar durante relajación. Igual bolsa de tokens no implica igual entrada posicional.",
        "## Estado y conclusiones",
        "\n".join(f"- {item}" for item in blockers) if blockers else "Controles críticos satisfactorios; consultar estado Slurm y resultados reales.",
        "Se detuvieron extracción, entrenamiento y envío a Slurm. No se generaron figuras de embeddings ni métricas ficticias." if blockers else "Consultar experiment_manifest.json para los artefactos realmente producidos."]
    (report_dir / "experiment_report.md").write_text("\n\n".join(sections) + "\n")
    LOGGER.info("report_saved figures=%d scientific_results=%s", len(manifest["figures"]), evaluated)


def main() -> None:
    """Generate reports from CLI configuration and currently available artifacts."""
    generate(load_config(cli(__doc__).parse_args().config))


if __name__ == "__main__":
    main()
