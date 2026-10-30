import os
import pandas as pd
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import matplotlib.pyplot as plt
import seaborn as sns
import matgl
import matgl.utils.io
# matgl 1.1.3 downloads its pretrained models from the main branch of the matgl GitHub repository, where they no
# longer exist (removed in matgl 3.0). Load the same files from the v1.1.3 tag; models in ~/.cache/matgl are reused.
matgl.utils.io.PRETRAINED_MODELS_BASE_URL = "https://raw.githubusercontent.com/materialsvirtuallab/matgl/v1.1.3/pretrained_models/"
import torch
import hydra
from omegaconf import DictConfig

# Fidelity index of the multi-fidelity MEGNet model; the paper uses PBE (Predicted_PBE)
METHODS = ["PBE", "GLLB-SC", "HSE", "SCAN"]

@hydra.main(config_path="config", config_name="band", version_base=None)
def main(cfg: DictConfig):
    try:
        csv_path = cfg.proxy_model.csv_path
        model_name = cfg.proxy_model.model_band
        plot_path = cfg.proxy_model.plot_path
        training_cifs_path = cfg.proxy_model.training_cifs
        cif_column = cfg.proxy_model.training_cif_column
        load_structures_predict_and_plot_kde(csv_path, model_name, plot_path, training_cifs_path, cif_column,
                                             show_count=cfg.proxy_model.show_count)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def build_training_db(training_cifs_path, cif_column='cif'):
    """Training structures grouped by reduced formula (same procedure as form_proxy.py)."""
    training_db = {}
    if not os.path.exists(training_cifs_path):
        print(f"Warning: Training CIF file '{training_cifs_path}' not found. Novelty checking will be skipped.")
        return training_db
    train_data = pd.read_csv(training_cifs_path)
    if cif_column not in train_data.columns:
        print(f"Column '{cif_column}' not found in {training_cifs_path}. Columns: {list(train_data.columns)}")
    if 'status' in train_data.columns:
        train_data = train_data[train_data['status'] == 'success']
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Building Training DB"):
        try:
            if pd.isna(row[cif_column]):
                continue
            prim = SpacegroupAnalyzer(Structure.from_str(row[cif_column], fmt="cif")).get_primitive_standard_structure()
            training_db.setdefault(prim.composition.reduced_formula, []).append(prim)
        except Exception:
            continue
    print(f"Loaded {len(training_db)} unique chemical compositions from the training set.")
    return training_db

def check_novel(structure, training_db, sm):
    eval_struc = structure
    try:
        eval_struc = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    except Exception:
        pass
    for candidate in training_db.get(eval_struc.composition.reduced_formula, []):
        try:
            if sm.fit(candidate, eval_struc):
                return 0
        except Exception:
            pass
    return 1

def load_structures_predict_and_plot_kde(csv_path, model_name, plot_path, training_cifs_path, cif_column='cif', show_count=True):
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=['CIF Path'])

    # Target column written by sample_property.py / sli_to_cif.py (e.g. band_gap_ind)
    pred_columns = [f"Predicted_{m}" for m in METHODS]
    band_gap_column = [col for col in data.columns if col not in ['CIF Path', 'SLICES', 'Is_Novel'] + pred_columns][0]

    model = matgl.load_model(model_name)
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
                          attempt_supercell=False, comparator=ElementComparator())
    training_db = build_training_db(training_cifs_path, cif_column)

    for col in pred_columns + ['Is_Novel']:
        data[col] = None
    method_bandgap_data = {method: [] for method in METHODS}
    target_bandgap_data = {}

    for index, row in data.iterrows():
        cif_path = row['CIF Path']
        target_bandgap = float(row[band_gap_column])
        try:
            structure = Structure.from_file(cif_path)
            predictions = [float(model.predict_structure(structure=structure, state_attr=torch.tensor([i])))
                           for i in range(len(METHODS))]
            is_novel = check_novel(structure, training_db, sm) if training_db else None
        except Exception as e:
            print(f"Skipping {cif_path}: {e}")
            continue
        # written only after every fidelity succeeded, so a row is either complete or empty
        for method, predicted_bandgap in zip(METHODS, predictions):
            data.at[index, f"Predicted_{method}"] = predicted_bandgap
            method_bandgap_data[method].append(predicted_bandgap)
            target_bandgap_data.setdefault(target_bandgap, {}).setdefault(method, []).append(predicted_bandgap)
        data.at[index, 'Is_Novel'] = is_novel

    # Save predictions and novelty flags back to the CSV (read by metrics.py)
    data.to_csv(csv_path, index=False)
    print(f"Updated CSV saved with band gap predictions and novelty flags at: {csv_path}")

    os.makedirs(plot_path, exist_ok=True)

    # Plot individual properties for each method and target
    for target_bandgap, methods_data in target_bandgap_data.items():
        palette = sns.color_palette("husl", n_colors=len(methods_data))
        for method, values in methods_data.items():
            plt.figure(figsize=(12, 6))
            color = palette.pop(0)
            sns.kdeplot(values, fill=True, color=color, label=f'{method} Band Gap')
            plt.axvline(target_bandgap, color=color, linestyle='dotted', linewidth=2, label='Target Band Gap')
            if show_count:
                plt.text(0.05, 0.95, f'Total reconstructions: {len(values)}',
                         verticalalignment='top', horizontalalignment='left',
                         transform=plt.gca().transAxes, color='green', fontsize=12)
            plt.title(f'Band Gap Distribution for {method} (Target {target_bandgap} eV)')
            plt.xlabel('Band Gap (eV)')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f"{plot_path}/{method}_Band_Gap_Target_{target_bandgap}.png")
            plt.close()

    # Combined KDE plot for each method across all targets
    for method, values in method_bandgap_data.items():
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("husl", n_colors=max(1, len(target_bandgap_data)))
        for target_bandgap, methods_data in target_bandgap_data.items():
            if method in methods_data:
                color = palette.pop(0)
                sns.kdeplot(methods_data[method], fill=True, color=color, label=f'Target {target_bandgap} eV')
                plt.axvline(target_bandgap, color=color, linestyle='dotted', linewidth=1)
        plt.title(f'Combined Band Gap Distribution for {method}')
        plt.xlabel('Band Gap (eV)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f"{plot_path}/Combined_{method}_Band_Gaps.png")
        plt.close()

if __name__ == "__main__":
    main()
