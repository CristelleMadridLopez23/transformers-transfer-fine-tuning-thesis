# utils/dataset.py
import json
import pandas as pd
import torch

def encode(sequence, stoi, special_tokens):
    tokens = sequence.split()
    unknown = sorted(set(tokens) - set(stoi))
    if unknown:
        raise ValueError(f"Unknown SLICES tokens: {unknown}")

    encoded = [stoi[token] for token in tokens]
    return special_tokens + [stoi["<start>"]] + encoded + [stoi["<end>"]]

def decode(l, itos, num_properties):
    # Find the IDs of special tokens
    pad_id = next((k for k, v in itos.items() if v == "<pad>"), None)
    property_ids = [next((k for k, v in itos.items() if v == f"<{p}>"), None) 
                    for p in ["form", "band", "bulk", "num"][:num_properties]]
    
    # Filter out None values
    special_ids = [i for i in [pad_id] + property_ids if i is not None]
    
    # Decode the sequence, excluding special tokens
    return ' '.join([itos[i] for i in l if i not in special_ids])


def save_vocab_and_model(vocab, block_size, file_name, stoi, itos, metadata=None):
    model_info = {
        'vocab': vocab,
        'block_size': block_size,
        'stoi': stoi,
        'itos': itos
    }
    if metadata:
        model_info.update(metadata)
    with open(file_name, 'w') as f:
        json.dump(model_info, f)

def read_and_preprocess_properties_with_sequences(fname, property_columns, slices_column="SLICES"):
    data = pd.read_csv(fname)
    sequences = []
    properties = []
    for _, row in data.iterrows():
        sequence = row[slices_column]
        # No regex substitution needed for consolidated tokens
        sequences.append(sequence)
        properties.append([row[col] for col in property_columns])
    return sequences, properties

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, properties):
        self.data = data
        self.properties = properties

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.properties[idx]
