---
license: mit
library_name: transformers
pipeline_tag: fill-mask
tags:
  - materials-science
  - crystal-structure
  - foundation-model
  - chemistry
  - bert
  - scope
  - cloud
language:
  - en
---

# CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning

CLOUD (**C**rystal **L**anguage m**O**del for **U**nified and **D**ifferentiable materials modeling) is a Transformer-based foundation model that learns crystal representations from string encodings of crystal structures. Crystals are serialized with a novel **Symmetry-Consistent Ordered Parameter Encoding (SCOPE)**, a compact, coordinate-free representation that captures space-group symmetry, Wyckoff positions, and composition. The model can be fine-tuned for accurate, generalizable, and scalable property prediction, and can be combined with physics laws (e.g. the Debye model) for thermodynamic-consistent predictions.

- 📄 Paper (*Nature Communications* **17**, 4074, 2026): [CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning](https://doi.org/10.1038/s41467-026-70467-3) ([arXiv preprint](https://arxiv.org/abs/2506.17345))
- 💻 Code: [github.com/BattModels/CLOUD](https://github.com/BattModels/CLOUD)
- 🏛️ Authors: Changwen Xu, Shang Zhu, Venkatasubramanian Viswanathan (University of Michigan)

## Model Details

| | |
|---|---|
| Architecture | BERT encoder (`BertForMaskedLM`) |
| Hidden size | 768 |
| Hidden layers | 12 |
| Attention heads | 12 |
| Intermediate size | 3072 |
| Max sequence length | 64 |
| Vocab size | 30522 (custom SCOPE tokenizer) |
| Parameters | ~110M |
| Precision | float32 |
| Pretraining objective | Masked language modeling on SCOPE strings |
| Pretraining data | ~6M crystal structures from OPTIMADE |

## Repository Layout

```
ckpt/
├── config.json              # BertForMaskedLM config
├── generation_config.json
├── model.safetensors        # Pretrained weights (~530 MB)
├── training_args.bin        # HF Trainer arguments used for pretraining
├── tokenizer_config.json    # SCOPE tokenizer config
├── special_tokens_map.json
├── added_tokens.json
└── vocab.txt                # SCOPE vocabulary
```

## Usage

### Load the pretrained model and tokenizer

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("ChangwenXu/CLOUD", subfolder="ckpt")
model = AutoModelForMaskedLM.from_pretrained("ChangwenXu/CLOUD", subfolder="ckpt")
```

### Encode a crystal structure to a SCOPE string

CLOUD operates on SCOPE string representations. Use the conversion utility from the [code repository](https://github.com/BattModels/CLOUD) to turn a CIF file into a SCOPE string:

```bash
git clone https://github.com/BattModels/CLOUD.git
cd CLOUD
python structure_to_str.py --dir <path_to_cif> --out <output_path> \
    --numproc <num_of_processes> --batchsize <batch_size>
```

### Get crystal embeddings

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("ChangwenXu/CLOUD", subfolder="ckpt")
encoder = AutoModel.from_pretrained("ChangwenXu/CLOUD", subfolder="ckpt")
encoder.eval()

scope_string = "..."  # SCOPE representation produced by structure_to_str.py
inputs = tokenizer(scope_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
with torch.no_grad():
    outputs = encoder(**inputs)
# Pooled [CLS] embedding for the crystal:
crystal_embedding = outputs.last_hidden_state[:, 0]
```

### Fine-tuning

Recipes for fine-tuning on MatBench, UnconvBench, MatBench Discovery / WBM, and the physics-informed CLOUD-DEBYE variant are provided in the [GitHub repository](https://github.com/BattModels/CLOUD) (`train.py`, `train_mp.py`, `wbm_predict.py`, `train_debye.py`).

## Intended Use

- Pretrained backbone for downstream crystal property prediction (formation energy, bandgap, mechanical, thermodynamic properties, etc.)
- Featurizer for materials screening and discovery workflows
- Backbone for physics-informed extensions such as CLOUD-DEBYE

### Out-of-scope

- Direct generation of crystal structures from scratch
- Predicting properties of non-crystalline systems (molecules, amorphous solids)
- Use as a substitute for high-fidelity DFT/MD without task-specific fine-tuning and validation

## Limitations

- Trained on equilibrium / known crystal structures from OPTIMADE; out-of-distribution behavior on highly disordered, defective, or hypothetical structures is not guaranteed.
- Maximum sequence length of 64 tokens; very large or low-symmetry unit cells may be truncated by the SCOPE encoder.
- Property predictions require task-specific fine-tuning; the released checkpoint is the masked-language-model pretrained backbone only.

## Citation

If you find CLOUD useful in your research, please cite:

```bibtex
@article{xu2026cloud,
  title   = {{CLOUD}: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning},
  author  = {Xu, Changwen and Zhu, Shang and Viswanathan, Venkatasubramanian},
  journal = {Nature Communications},
  volume  = {17},
  number  = {1},
  pages   = {4074},
  year    = {2026},
  doi     = {10.1038/s41467-026-70467-3}
}

@inproceedings{xu2024cloud,
  title     = {{CLOUD}: A Scalable Scientific Foundation Model for Crystal Representation Learning},
  author    = {Xu, Changwen and Zhu, Shang and Viswanathan, Venkatasubramanian},
  booktitle = {NeurIPS 2024 Workshop on Foundation Models for Science: Progress, Opportunities, and Challenges},
  year      = {2024}
}
```

## License

Released under the [MIT License](https://github.com/BattModels/CLOUD/blob/master/LICENSE), © 2025 Changwen Xu.
