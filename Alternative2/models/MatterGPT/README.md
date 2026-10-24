---
language: en
license: mit
library_name: transformers
tags:
- materials-science
- crystal-structure
- inverse-design
pipeline_tag: text-generation
inference: true
---

# MatterGPT

MatterGPT is a generative pre-trained transformer model for inverse design of inorganic materials. It uses the SLICES (Simplified Line-Input Crystal-Encoding System) representation to generate novel crystal structures with targeted properties.

## Model Description

- **Model type:** Generative Pre-trained Transformer (GPT2)
- **Language(s):** SLICES (crystal structure representation)
- **License:** MIT
- **Finetuned from model:** GPT2

## Intended Uses & Limitations

MatterGPT is designed for:
- Generating crystal structures with specified formation energies and band gaps
- Multi-property targeted material design
- Exploring novel inorganic materials

Note: This model is trained on structures with up to 20 atoms per unit cell and may not generalize well to larger structures.

## How to Use

You can use this model directly with the Hugging Face Inference API:

```python
from huggingface_hub import InferenceApi

inference = InferenceApi("your-username/mattergpt")

# Generate a single crystal structure
result = inference({"formation_energy": -1.0, "band_gap": 2.0})
print(result)

# Generate multiple crystal structures
results = inference([
    {"formation_energy": -1.0, "band_gap": 2.0},
    {"formation_energy": -2.0, "band_gap": 3.0}
])
for crystal in results:
    print(crystal)
```

For local usage, please refer to the detailed instructions below.
## How to Use MatterGPT locally

This guide will help you get started with using the MatterGPT model for generating crystal structures.

### Setup

First, ensure you have the necessary dependencies installed:

```bash
pip install torch tqdm
```

You'll also need the `matter_gpt_wrapper` module, which should be provided with the model.

### Loading the Model and Tokenizer

```python
from matter_gpt_wrapper import MatterGPTWrapper, SimpleTokenizer
import torch
import os

# Load the model
model_path = "./"  # Directory containing config.json and pytorch_model.pt
model = MatterGPTWrapper.from_pretrained(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer_path = "Voc_prior"
tokenizer = SimpleTokenizer(tokenizer_path)
```

Make sure the `config.json`, `pytorch_model.pt`, and `Voc_prior` files are in the correct locations.

### Generating a Single Sequence

To generate a single crystal structure:

```python
def generate_single(condition):
    context = '>'
    x = torch.tensor([tokenizer.stoi[context]], dtype=torch.long)[None,...].to(model.device)
    p = torch.tensor([condition]).unsqueeze(1).to(model.device)
    
    generated = model.generate(x, prop=p, max_length=model.config.block_size, 
                               temperature=1.2, do_sample=True, top_k=0, top_p=0.9)
    return tokenizer.decode(generated[0].tolist())

# Example usage
condition = [-1.0, 2.0]  # formation energy and bandgap
single_sequence = generate_single(condition)
print(single_sequence)
```

### Generating Multiple Sequences

To generate multiple crystal structures:

```python
from tqdm import tqdm

def generate_multiple(condition, num_sequences, batch_size=32):
    all_sequences = []
    for _ in tqdm(range(0, num_sequences, batch_size)):
        current_batch_size = min(batch_size, num_sequences - len(all_sequences))
        context = '>'
        x = torch.tensor([tokenizer.stoi[context]], dtype=torch.long)[None,...].repeat(current_batch_size, 1).to(model.device)
        p = torch.tensor([condition]).repeat(current_batch_size, 1).unsqueeze(1).to(model.device)
        
        generated = model.generate(x, prop=p, max_length=model.config.block_size, 
                                   temperature=1.2, do_sample=True, top_k=0, top_p=0.9)
        all_sequences.extend([tokenizer.decode(seq.tolist()) for seq in generated])
        
        if len(all_sequences) >= num_sequences:
            break
    
    return all_sequences[:num_sequences]

# Example usage
condition = [-1.0, 2.0]  # formation energy and bandgap
num_sequences = 10
multiple_sequences = generate_multiple(condition, num_sequences)
for seq in multiple_sequences:
    print(seq)
```

### Notes

- The `condition` parameter is a list containing the desired formation energy and bandgap values.
- The generated sequences are SLICES representations of crystal structures.
- You may need to post-process the generated SLICES to convert them into actual crystal structures.

For more detailed information on the SLICES format and how to convert it to crystal structures, please refer to the full documentation.

## Training Data

The model was trained on the Alex-20 dataset, derived from the Alexandria database, containing 280,033 unique crystal structures with up to 20 atoms per unit cell.

## Training Procedure

MatterGPT was trained for 50 epochs using the Adam optimizer with an initial learning rate of 0.0001 and cosine annealing schedule. The model has approximately 80 million trainable parameters.

## Evaluation Results

Performance metrics on test set:
- Validity: >90%
- Uniqueness: >90%
- Novelty: ~40-60%
- MAPE for formation energy: ~11-13%
- MAPE for band gap: ~31-51%

## Citation

If you use this model in your research, please cite:

[Include citation information when available]

## Contact

[Provide contact information or link to the GitHub repository for issues and questions]