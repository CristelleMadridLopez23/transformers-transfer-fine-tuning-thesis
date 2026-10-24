from slices.core import SLICES
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="form", version_base=None)
def main(cfg: DictConfig):
    try:
        data_path = cfg.back_to_cif.file_path
        output_folder = cfg.back_to_cif.output_folder
        plot_targeted_reconstructions(data_path, output_folder, cfg)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def convert_and_save_structures(data_path, output_folder, cfg):
    data = pd.read_csv(data_path)
    backend = SLICES()   # SLICES 2.0.4 default reconstruction settings, as used for the paper's runs
    reconstructed_data = []
    property_reconstructions = {}
    cif_folder = os.path.join(output_folder, "cif_files")
    if not os.path.exists(cif_folder):
        os.makedirs(cif_folder)

    # Get the property column names (all columns that aren't 'Sequence')
    property_columns = [col for col in data.columns if col != 'Sequence']

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing sequences"):
        property_values = tuple(row[col] for col in property_columns)
        sequence = row['Sequence']
        try:
            structure, final_energy_per_atom_IAP = backend.SLICES2structure(sequence)
            if final_energy_per_atom_IAP != 0:
                # Save structure to CIF in the specified subfolder
                file_path = os.path.join(cif_folder, f"{index}.cif")
                structure.to(fmt="cif", filename=file_path)
                # Record data for plotting and saving
                reconstructed_data.append({**{col: val for col, val in zip(property_columns, property_values)}, 
                                           'SLICES': sequence, 'CIF Path': file_path})
                property_reconstructions[property_values] = property_reconstructions.get(property_values, 0) + 1
        except Exception as e:
            print(f"An error occurred with sequence: {sequence} - {e}")

    # Save the successful reconstructions to a CSV file (with a header even if nothing was reconstructed)
    reconstructed_df = pd.DataFrame(reconstructed_data, columns=property_columns + ['SLICES', 'CIF Path'])
    reconstructed_df.to_csv(os.path.join(output_folder, cfg.back_to_cif.reconstructed), index=False)
    
    return property_reconstructions, reconstructed_df, property_columns

def plot_targeted_reconstructions(data_path, output_folder, cfg):
    property_reconstructions, _, property_columns = convert_and_save_structures(data_path, output_folder, cfg)

    # Unconditional samples contain only a Sequence column, so there is no
    # target-property reconstruction plot to create.
    if not property_columns:
        print("Unconditional reconstruction complete; skipping the property-count plot.")
        return

    # One bar per target (a target pair for two properties)
    target_properties = sorted(property_reconstructions.keys())
    labels = [", ".join(f"{value:.2f}" for value in prop) for prop in target_properties]
    reconstruction_counts = [property_reconstructions[prop] for prop in target_properties]

    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(labels)), 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), reconstruction_counts, color=colors, width=0.8)

    # Add the counts above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xlabel(", ".join(property_columns))
    ax.set_ylabel('Number of Successful Conversions')
    ax.set_title(f'Successful Structure Conversions by {", ".join(property_columns)}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, cfg.back_to_cif.plot_name))
    plt.close()

if __name__ == "__main__":
    main()
