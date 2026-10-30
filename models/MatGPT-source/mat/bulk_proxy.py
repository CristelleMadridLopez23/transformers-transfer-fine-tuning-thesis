import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from megnet.utils.models import load_model

import hydra
from omegaconf import DictConfig
import warnings

warnings.filterwarnings("ignore")

# --- HELPER: Clean SLICES strings for accurate Uniqueness ---
def clean_slices_string(s):
    """Removes padding/start tokens and normalizes spaces for accurate exact-matching."""
    if not isinstance(s, str):
        return ""
    s = s.replace('>', '').replace('<', '')
    return " ".join(s.split())

@hydra.main(config_path="config", config_name="bulk", version_base=None)
def main(cfg: DictConfig):
    try:
        csv_path = cfg.proxy_model.csv_path
        model_name = cfg.proxy_model.model_bulk_modulus
        plot_path = cfg.proxy_model.plot_path

        # Training CSV and its CIF column (proxy_model.training_cif_column), used for the novelty check
        training_cifs_path = cfg.proxy_model.training_cifs
        cif_column = cfg.proxy_model.training_cif_column

        load_structures_predict_and_plot_kde(
            csv_path=csv_path,
            model_name=model_name,
            plot_path=plot_path,
            training_cifs_path=training_cifs_path,
            cif_column=cif_column,
            show_count=cfg.proxy_model.show_count
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def load_structures_predict_and_plot_kde(csv_path, model_name, plot_path, training_cifs_path, cif_column='cif', show_count=True):
    # 1. Load Generated Data & Model
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=['CIF Path'])

    # Dynamically find the property column
    property_column = [col for col in data.columns if col not in ['CIF Path', 'SLICES', 'Predicted_Bulk_Modulus', 'Is_Novel']][0]

    print(f"Loading MEGNet model: {model_name}...")
    model = load_model(model_name)

    # 2. Setup 3D StructureMatcher
    sm = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        comparator=ElementComparator(),
    )

    # ---------------------------------------------------------
    # PART A: Build In-Memory Training Database (The Reference)
    # ---------------------------------------------------------
    training_db = {}

    if os.path.exists(training_cifs_path):
        print(f"Loading training structures from '{training_cifs_path}' into memory for Novelty check...")
        try:
            train_data = pd.read_csv(training_cifs_path)
            if cif_column not in train_data.columns:
                print(f"Column '{cif_column}' not found in {training_cifs_path}. Columns: {list(train_data.columns)}")
            if 'status' in train_data.columns:
                train_data = train_data[train_data['status'] == 'success']

            for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Building Training DB"):
                try:
                    cif_str = row[cif_column]
                    if pd.isna(cif_str): continue

                    train_struc = Structure.from_str(cif_str, fmt="cif")
                    finder = SpacegroupAnalyzer(train_struc)
                    prim_struc = finder.get_primitive_standard_structure()
                    comp = prim_struc.composition.reduced_formula

                    if comp not in training_db:
                        training_db[comp] = []
                    training_db[comp].append(prim_struc)
                except Exception:
                    continue
            print(f"Loaded {len(training_db)} unique chemical compositions from the training set.")
        except Exception as e:
            print(f"Failed to process training CIFs: {e}")
            training_db = {}
    else:
        print(f"Warning: Training CIF file '{training_cifs_path}' not found. Novelty checking will be skipped.")

    # Tracking dictionaries for metrics & plotting
    property_bulk_dict = {}
    property_slices_dict = {}
    property_novelty_dict = {}

    # Initialize New Columns
    data['Predicted_Bulk_Modulus'] = None
    data['Is_Novel'] = None

    # ---------------------------------------------------------
    # PART B: Evaluate Generated Structures (Prediction & Novelty)
    # ---------------------------------------------------------
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Predicting & Checking Novelty"):
        cif_path = row['CIF Path']
        slices_string = clean_slices_string(row.get('SLICES', ""))
        target_val = row[property_column]

        try:
            # Parse structure and predict property
            structure = Structure.from_file(cif_path)

            # MEGNet Bulk Modulus conversion (the model predicts log10 of K in GPa)
            bulk_modulus = 10 ** model.predict_structure(structure).ravel()[0]
            pred_val = float(bulk_modulus)

            # Novel unless a training structure matches (left empty if no training data was loaded)
            is_novel = 1 if training_db else None

            # Novelty check against the training structures
            if training_db:
                eval_struc = structure
                try:
                    finder = SpacegroupAnalyzer(structure)
                    eval_struc = finder.get_primitive_standard_structure()
                except Exception:
                    pass # Fallback: just use the original 'structure'

                try:
                    comp_gen = eval_struc.composition.reduced_formula
                    candidates = training_db.get(comp_gen, [])

                    for candidate in candidates:
                        try:
                            if sm.fit(candidate, eval_struc):
                                is_novel = 0  # Match found, not novel
                                break
                        except Exception:
                            pass
                except Exception:
                    pass

            # Save Results to DataFrame directly
            data.at[index, 'Predicted_Bulk_Modulus'] = pred_val
            data.at[index, 'Is_Novel'] = is_novel

            # Track for plotting/metrics below
            if not pd.isnull(pred_val):
                if target_val not in property_bulk_dict:
                    property_bulk_dict[target_val] = []
                    property_slices_dict[target_val] = []
                    property_novelty_dict[target_val] = []

                property_bulk_dict[target_val].append(pred_val)
                property_slices_dict[target_val].append(slices_string)
                property_novelty_dict[target_val].append(is_novel)

        except Exception as e:
            print(f"Skipping {cif_path}: {e}")
            continue

    # Save predictions and novelty scores back to CSV
    data.to_csv(csv_path, index=False)
    print(f"\nUpdated CSV saved with Bulk Modulus predictions and Novelty (1/0) flags at: {csv_path}")

    if not property_bulk_dict:
        print("No valid predicted bulk modulus values to evaluate.")
        return

    # ---------------------------------------------------------
    # PART C: Quick summary (the table values are computed by metrics.py)
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" QUICK SUMMARY (relative to the evaluated structures; table values: metrics.py)")
    print("="*65)

    for target_val, bulk_list in property_bulk_dict.items():
        valid_count = len(bulk_list)

        # --- Uniqueness ---
        unique_slices = set(property_slices_dict[target_val])
        unique_count = len(unique_slices)
        uniqueness = (unique_count / valid_count * 100) if valid_count > 0 else 0.0

        # --- Novelty ---
        novelty_display = "N/A (Training Data Missing)"
        if training_db:
            novel_count = sum(property_novelty_dict[target_val])
            novelty_rate = (novel_count / valid_count * 100) if valid_count > 0 else 0.0
            novelty_display = f"{novelty_rate:.2f}% ({novel_count} novel out of {valid_count} generated)"

        # --- MAPE & MAE ---
        try:
            numeric_target = float(target_val)
            y_p = np.array(bulk_list)
            mae = np.mean(np.abs(y_p - numeric_target))

            if numeric_target != 0:
                errors = np.abs((y_p - numeric_target) / numeric_target)
                mape = np.mean(errors) * 100
                mape_display = f"{mape:.2f}%"
            else:
                mape_display = "N/A (Target is 0)"
        except ValueError:
            mape_display = "N/A"
            mae = 0.0

        # --- Print Results ---
        print(f"Target {property_column}: {target_val} GPa")
        print(f"  - Valid Samples Analyzed : {valid_count}")
        print(f"  - Uniqueness Rate        : {uniqueness:.2f}% ({unique_count} unique out of {valid_count})")
        print(f"  - 3D Novelty Rate        : {novelty_display}")
        print(f"  - MAPE                   : {mape_display}")
        print(f"  - MAE                    : {mae:.2f} GPa")
        print("-" * 65)
    print("\n")

    # ---------------------------------------------------------
    # PART D: Plotting KDE Distributions
    # ---------------------------------------------------------
    os.makedirs(plot_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')

    colors = plt.cm.viridis(np.linspace(0, 1, len(property_bulk_dict)))

    # 1. Plot individual properties
    for (property_value, bulk_modulus_list), color in zip(property_bulk_dict.items(), colors):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        if len(bulk_modulus_list) > 1:
            sns.kdeplot(bulk_modulus_list, fill=True, color=color, alpha=0.7,
                        label=f'{property_column} {property_value}', ax=ax)
        else:
            ax.axvline(bulk_modulus_list[0], color=color, linestyle='dashed', linewidth=2,
                       label=f'{property_column} {property_value}')

        try:
            ax.axvline(float(property_value), color='r', linestyle='dotted', linewidth=2,
                       label=f'Target {property_column}')
        except ValueError:
            pass

        if show_count:
            ax.text(0.05, 0.95, f'Total valid reconstructions: {len(bulk_modulus_list)}',
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, color='green', fontsize=12)

        ax.set_title(f'Predicted Bulk Modulus for {property_column} {property_value}', fontsize=16)
        ax.set_xlabel('Bulk Modulus (GPa)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{plot_path}/Bulk_Modulus_{property_column}_{property_value}.png", bbox_inches='tight')
        plt.close()

    # 2. Combined KDE plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    for (property_value, bulk_modulus_list), color in zip(property_bulk_dict.items(), colors):
        if len(bulk_modulus_list) > 1:
            sns.kdeplot(bulk_modulus_list, fill=True, color=color, alpha=0.5,
                        label=f'{property_value} GPa', ax=ax)
        try:
            ax.axvline(float(property_value), color=color, linestyle='dotted', linewidth=2)
        except ValueError:
            pass

    ax.set_title('Probability Distribution of Generated Structures for Bulk Modulus', fontsize=16)
    ax.set_xlabel('Bulk Modulus (GPa)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f"{plot_path}/Combined_Predicted_Bulk_Modulus.png", bbox_inches='tight')
    plt.close()
    print(f"Plots saved to directory: {plot_path}/")

if __name__ == "__main__":
    main()
