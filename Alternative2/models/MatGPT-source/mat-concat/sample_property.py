# Sample property-conditioned crystal structures using trained model
import json
import pandas as pd
from utils.models import TransformerDecoderModel
from utils.dataset import decode
import hydra
from omegaconf import DictConfig, ListConfig
import time
import os
import logging
import traceback
from tqdm import tqdm
import gc
import torch
import tensorflow as tf
from slices.core import SLICES
from pymatgen.core.structure import Structure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SLICES structure generation settings
SETTINGS = {
    "bond_scaling": 1.05,
    "delta_theta": 0.005,
    "delta_x": 0.45,
    "lattice_shrink": 1,
    "lattice_expand": 1.25,
    "angle_weight": 0.5,
    "vbond_param_ave_covered": 0,
    "vbond_param_ave": 0.01,
    "repul": True,
    "graph_method": "econnn"
}

def load_model_info(model_info_path):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    model_info['itos'] = {int(k): v for k, v in model_info['itos'].items()}
    return model_info

def find_abnormal_lattices(structure: Structure, threshold: float = 10000.0) -> bool:
    """Check if structure has abnormally large lattice parameters"""
    a, b, c = structure.lattice.abc
    if any(abs(param) > threshold for param in [a, b, c]):
        return True
    return False

def initialize_backend():
    """Initialize SLICES backend for structure generation"""
    return SLICES(graph_method=SETTINGS["graph_method"], check_results=False, relax_model="m3gnet")

def is_valid_sequence(sequence, backend):
    """Validate SLICES sequence by checking and generating structure"""
    try:
        # Check SLICES string validity
        if backend.check_SLICES(sequence, strategy=4, dupli_check=False):
            # Try to generate structure
            structures, energy = backend.to_structures(
                SETTINGS["bond_scaling"], SETTINGS["delta_theta"], SETTINGS["delta_x"],
                SETTINGS["lattice_shrink"], SETTINGS["lattice_expand"], SETTINGS["angle_weight"],
                SETTINGS["vbond_param_ave_covered"], SETTINGS["vbond_param_ave"], SETTINGS["repul"]
            )
            
            if structures:
                structure = structures[-1]
                # Check for abnormal lattices
                if not find_abnormal_lattices(structure):
                    return True
        return False
    except Exception as e:
        logging.error(f"Validation error for sequence: {str(e)}")
        # Clear the TensorFlow session after an error
        try:
            del backend
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
        return False

def generate_single_sequence(model, initial_tokens, property_values, stoi, itos, max_new_tokens, temperature=1.2, top_p=0.9):
    """Generate a single sequence using the transformer model with nucleus sampling"""
    try:
        context = torch.tensor([initial_tokens], dtype=torch.long)
        properties = torch.tensor([property_values], dtype=torch.float)

        with torch.no_grad():
            generated_sequence = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                stoi=stoi,
                itos=itos,
                properties=properties,
                temperature=temperature,
                top_p=top_p
            )

        return generated_sequence[0].tolist()
    except Exception as e:
        logging.error(f"Error in generate_single_sequence: {str(e)}")
        return None

@hydra.main(config_path="config", config_name="form", version_base=None)
def generate_sequences_with_property(cfg: DictConfig):
    logging.info("Using CPU for computations")
    torch.manual_seed(cfg.training.seed)
    property_names = cfg.training.selected_properties
    sequences, prop_values = [], []

    try:
        # Initialize backend
        backend = initialize_backend()

        model_info = load_model_info(cfg.sample.vocab_and_model)
        stoi, itos = model_info['stoi'], model_info['itos']
        block_size, vocab_size = model_info['block_size'], len(stoi)
        # weights trained with ../mat or ../mat-multi have no conditioning_mode entry
        stored_mode = model_info.get('conditioning_mode', cfg.training.conditioning_mode)
        if stored_mode != cfg.training.conditioning_mode:
            raise ValueError(
                f"Checkpoint mode is {stored_mode}, but config requests "
                f"{cfg.training.conditioning_mode}"
            )
        stored_properties = model_info.get('selected_properties')
        if stored_properties is not None and stored_properties != list(property_names):
            raise ValueError(
                f"Checkpoint was trained on {stored_properties}, but config requests {list(property_names)}"
            )

        model = TransformerDecoderModel(
            vocab_size=vocab_size,
            n_embd=cfg.model.n_embd,
            n_head=cfg.model.n_head,
            n_layer=cfg.model.n_layer,
            block_size=block_size,
            dropout=cfg.model.dropout,
            stoi=stoi,
            itos=itos,
            train_dataset=None,
            val_dataset=None,
            num_properties=cfg.training.num_properties,
            conditioning_mode=cfg.training.conditioning_mode
        )
        # Load weights into inner model (matches how we save in train.py)
        model.model.load_state_dict(torch.load(cfg.sample.checkpoint_path, map_location='cpu'))
        model.eval()

        for property_value in cfg.sample.properties:
            # a number for one property, a list for several, [] for unconditional
            property_values = list(property_value) if isinstance(property_value, (list, ListConfig)) else [property_value]
            if len(property_values) != cfg.training.num_properties:
                raise ValueError(
                    f"Expected {cfg.training.num_properties} property values, "
                    f"received {len(property_values)}"
                )
            
            for _ in tqdm(range(cfg.sample.number_sequences), desc=f"Generating sequences for {property_values}"):
                start_time = time.time()
                valid_sequence = False
                attempts = 0
                max_attempts = cfg.sample.max_attempts

                while not valid_sequence and attempts < max_attempts:
                    if time.time() - start_time > cfg.sample.timeout:
                        logging.warning(f"Timeout reached for property values {property_values}")
                        break

                    initial_tokens = [stoi["<start>"]]
                    if cfg.sample.initial_tokens:
                        initial_tokens += [stoi[token] for token in cfg.sample.initial_tokens]

                    generated_sequence = generate_single_sequence(
                        model.model,
                        initial_tokens,
                        property_values,
                        stoi,
                        itos,
                        cfg.sample.max_new_tokens,
                        temperature=cfg.sample.temperature,
                        top_p=cfg.sample.top_p
                    )

                    if generated_sequence is None:
                        logging.warning(f"Failed to generate sequence, attempt {attempts + 1}/{max_attempts}")
                        attempts += 1
                        gc.collect()
                        continue

                    generated_sequence_ids = generated_sequence[1:]   # drop <start>
                    decoded_sequence = decode(generated_sequence_ids, itos, cfg.training.num_properties)

                    try:
                        if is_valid_sequence(decoded_sequence, backend):
                            sequences.append(decoded_sequence)
                            prop_values.append(property_values)
                            logging.info(f"Valid sequence generated: {decoded_sequence[:50]}...")
                            valid_sequence = True
                        else:
                            logging.warning(f"Invalid sequence generated, attempt {attempts + 1}/{max_attempts}")
                            attempts += 1
                            
                    except Exception as e:
                        logging.error(f"Error in sequence validation loop: {str(e)}")
                        # Force re-initialization of backend on critical errors
                        try:
                            del backend
                            tf.keras.backend.clear_session()
                            gc.collect()
                            backend = initialize_backend()
                        except Exception as reinit_error:
                            logging.error(f"Failed to re-initialize backend: {str(reinit_error)}")
                            
                        attempts += 1
                        continue

                    gc.collect()

                # Save periodically (every 10 valid sequences) to avoid data loss
                if valid_sequence and len(sequences) % 10 == 0:
                    save_results(sequences, prop_values, cfg, property_names)
                elif attempts >= max_attempts:
                    logging.warning(f"Max attempts reached for property values {property_values}")

    except Exception as e:
        logging.error(f"An error occurred during generation: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    finally:
        if sequences:   # keep what was generated, never overwrite results with an empty file
            save_results(sequences, prop_values, cfg, property_names)

def save_results(sequences, prop_values, cfg, property_names):
    try:
        output_df = pd.DataFrame({
            **{property_names[i]: [v[i] for v in prop_values] for i in range(len(property_names))},
            "Sequence": sequences
        })
        
        os.makedirs(os.path.dirname(cfg.sample.save_path), exist_ok=True)
        
        output_df.to_csv(cfg.sample.save_path, index=False)
        logging.info(f"Generated sequences have been saved to {cfg.sample.save_path}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    generate_sequences_with_property()
