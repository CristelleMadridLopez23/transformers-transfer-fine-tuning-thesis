import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import matgl
import matgl.utils.io
# matgl 1.1.3 downloads its pretrained models from the main branch of the matgl GitHub repository, where they no
# longer exist (removed in matgl 3.0). Load the same files from the v1.1.3 tag; models in ~/.cache/matgl are reused.
matgl.utils.io.PRETRAINED_MODELS_BASE_URL = "https://raw.githubusercontent.com/materialsvirtuallab/matgl/v1.1.3/pretrained_models/"

import hydra
from omegaconf import DictConfig

# --- HELPER: Clean SLICES strings for accurate Uniqueness ---
def clean_slices_string(s):
    """Removes padding/start tokens and normalizes spaces for accurate exact-matching."""
    if not isinstance(s, str):
        return ""
    s = s.replace('>', '').replace('<', '')
    return " ".join(s.split())

# --- MAIN WORKFLOW ---
@hydra.main(config_path="config", config_name="form", version_base=None)
def main(cfg: DictConfig):
    try:
        csv_path = cfg.proxy_model.csv_path
        model_name = cfg.proxy_model.model_eform
        plot_path = cfg.proxy_model.plot_path

        # Training CSV and its CIF column (proxy_model.training_cif_column), used for the novelty check
        training_cifs_path = cfg.proxy_model.training_cifs
        cif_column = cfg.proxy_model.training_cif_column

        load_structures_predict_and_plot(
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


def load_structures_predict_and_plot(csv_path, model_name, plot_path, training_cifs_path, cif_column='cif', show_count=True):
    # 1. Load Generated Data & Model
    data = pd.read_csv(csv_path)
    target_columns = [col for col in data.columns if col not in ['CIF Path', 'SLICES', 'Predicted_Formation_Energy', 'Is_Novel']]
    property_column = target_columns[0] if target_columns else None   # None for unconditional samples
    property_label = property_column if property_column else "unconditional"
    model = matgl.load_model(model_name)

    # 2. StructureMatcher with the MatterGPT settings
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
    training_db = {} # Format: { "NaCl": [PrimitiveStruc1, PrimitiveStruc2, ...], ... }

    if os.path.exists(training_cifs_path):
        print(f"Loading training structures from '{training_cifs_path}' into memory for Novelty check...")
        try:
            train_data = pd.read_csv(training_cifs_path)

            if cif_column not in train_data.columns:
                print(f"Column '{cif_column}' not found in {training_cifs_path}. Columns: {list(train_data.columns)}")
            # Filter only successful downloads/parsing
            if 'status' in train_data.columns:
                train_data = train_data[train_data['status'] == 'success']

            for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Building Training DB"):
                try:
                    cif_str = row[cif_column]
                    if pd.isna(cif_str):
                        continue

                    train_struc = Structure.from_str(cif_str, fmt="cif")

                    # Standardize to the primitive cell before matching
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

    # Tracking dictionaries for generated structures plotting
    property_eform_dict = {}
    property_slices_dict = {}
    property_novelty_dict = {}

    # Initialize New Columns
    data['Predicted_Formation_Energy'] = None
    data['Is_Novel'] = None

    # ---------------------------------------------------------
    # PART B: Evaluate Generated Structures (Prediction & Novelty)
    # ---------------------------------------------------------
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Predicting & Checking Novelty"):
        cif_path = row['CIF Path']
        slices_string = clean_slices_string(row['SLICES'])
        target_val = row[property_column] if property_column else "all"

        if pd.isnull(cif_path):
            continue

        try:
            # Parse structure and predict property
            structure = Structure.from_file(cif_path)
            eform = float(model.predict_structure(structure))
            if not np.isfinite(eform):
                raise ValueError(f"non-finite prediction {eform}")

            # Novel unless a training structure matches (left empty if no training data was loaded)
            is_novel = 1 if training_db else None

            # Novelty check against the training structures
            if training_db:
                # 1. Try to standardize, fallback to original if it fails
                eval_struc = structure
                try:
                    finder = SpacegroupAnalyzer(structure)
                    eval_struc = finder.get_primitive_standard_structure()
                except Exception:
                    pass # Fallback: just use the original 'structure'

                try:
                    # 2. Get composition and candidates
                    comp_gen = eval_struc.composition.reduced_formula
                    candidates = training_db.get(comp_gen, [])

                    # 3. Attempt structural overlap
                    for candidate in candidates:
                        try:
                            if sm.fit(candidate, eval_struc):
                                is_novel = 0  # Match found, not novel
                                break
                        except Exception:
                            pass # If sm.fit crashes, continue to next candidate
                except Exception:
                    pass # keep the default if the composition lookup fails

            # Save Results to DataFrame directly
            data.at[index, 'Predicted_Formation_Energy'] = eform
            data.at[index, 'Is_Novel'] = is_novel

            # Track for plotting/metrics below
            if target_val not in property_eform_dict:
                property_eform_dict[target_val] = []
                property_slices_dict[target_val] = []
                property_novelty_dict[target_val] = []

            property_eform_dict[target_val].append(eform)
            property_slices_dict[target_val].append(slices_string)
            property_novelty_dict[target_val].append(is_novel)

        except Exception as e:
            print(f"Skipping {cif_path}: {e}")
            continue

    # Save predictions and novelty scores back to CSV
    data.to_csv(csv_path, index=False)
    print(f"\nUpdated CSV saved with Energy predictions and Novelty (1/0) flags at: {csv_path}")

    if not property_eform_dict:
        print("No valid predicted formation energies to evaluate.")
        return

    # ---------------------------------------------------------
    # PART C: Quick summary (the table values are computed by metrics.py)
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" QUICK SUMMARY (relative to the evaluated structures; table values: metrics.py)")
    print("="*65)

    for target_val, eform_list in property_eform_dict.items():
        valid_count = len(eform_list)

        # --- Uniqueness ---
        unique_slices = set(property_slices_dict[target_val])
        unique_count = len(unique_slices)
        uniqueness = (unique_count / valid_count * 100) if valid_count > 0 else 0.0

        # --- Novelty ---
        novelty_display = "N/A (Training Data Missing)"
        if training_db:
            # Count the 1s from the tracked array
            novel_count = sum(property_novelty_dict[target_val])
            novelty_rate = (novel_count / valid_count * 100) if valid_count > 0 else 0.0
            novelty_display = f"{novelty_rate:.2f}% ({novel_count} novel out of {valid_count} generated)"

        # --- MAPE ---
        try:
            numeric_target = float(target_val)
            if numeric_target != 0:
                errors = np.abs((np.array(eform_list) - numeric_target) / numeric_target)
                mape = np.mean(errors) * 100
                mape_display = f"{mape:.2f}%"
            else:
                mape_display = "N/A (Target is 0)"
        except ValueError:
            mape_display = "N/A (Non-numeric target)"

        # --- Print Results ---
        print(f"Target {property_label}: {target_val}")
        print(f"  - Valid Samples Analyzed : {valid_count}")
        print(f"  - Uniqueness Rate        : {uniqueness:.2f}% ({unique_count} unique out of {valid_count})")
        print(f"  - 3D Novelty Rate        : {novelty_display}")
        print(f"  - MAPE                   : {mape_display}")
        print("-" * 65)
    print("\n")

    # ---------------------------------------------------------
    # PART D: Plotting KDE Distributions
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(plot_path, exist_ok=True)

    # 1. Plot individual properties
    for target_val, eform_list in property_eform_dict.items():
        plt.figure(figsize=(12, 6))
        if len(eform_list) > 1:
            sns.kdeplot(eform_list, fill=True, label=f'{property_label} {target_val}')
        else:
            plt.axvline(eform_list[0], color='k', linestyle='dashed', linewidth=1, label=f'{property_label} {target_val}')

        try:
            plt.axvline(float(target_val), color='r', linestyle='dotted', linewidth=2, label=f'Target {property_label}')
        except:
            pass

        if show_count:
            plt.text(0.05, 0.95, f'Total valid reconstructions: {len(eform_list)}',
                     verticalalignment='top', horizontalalignment='left',
                     transform=plt.gca().transAxes, color='green', fontsize=12)

        plt.title(f'Predicted Formation Energies for {property_label} {target_val}')
        plt.xlabel('Formation Energy (eV/atom)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{plot_path}/Formation_Energy_{property_label}_{target_val}.png")
        plt.close()

    # 2. Combined KDE plot
    plt.figure(figsize=(18, 6))
    for target_val, eform_list in property_eform_dict.items():
        if len(eform_list) > 1:
            sns.kdeplot(eform_list, fill=True, label=f'Target: {target_val}')
        try:
            plt.axvline(float(target_val), color='r', linestyle='dotted', linewidth=2)
        except:
            pass

    plt.title('Probability Distribution of Generated Structures Formation Energies', fontsize=16)
    plt.xlabel('Formation Energy (eV/atom)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{plot_path}/Combined_Predicted_Formation_Energies.png", bbox_inches='tight')
    plt.close()
    print(f"Plots saved to directory: {plot_path}/")

if __name__ == "__main__":
    main()
