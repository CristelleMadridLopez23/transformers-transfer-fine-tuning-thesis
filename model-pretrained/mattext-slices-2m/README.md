---
library_name: transformers
tags:
- chemistry
- bert
- materials
- pretrained
license: mit
datasets:
- n0w0f/MatText
language:
- en
---

# Model Card for Model ID

Model Pretrained using Masked Language Modelling on 2 million crystal structures in one of the **MatText** Representation



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
**MatText** model pretrained using Masked Language Modelling on crystal structures mined from NOMAD and represented using MatText - SLICES (The [SLICES representation](https://github.com/xiaohang007/SLICES) of a material ).


- **Developed by:** [Lamalab](https://github.com/lamalab-org)
- **Homepage:** https://github.com/lamalab-org/MatText 
- **Leaderboard:** To be published
- **Point of Contact:** [Nawaf Alampara](https://github.com/n0w0f)
- **Model type:** Pretrained BERT
- **Language(s) (NLP):** This is not a natural language model
- **License:** MIT

### Model Sources 

- **Repository:** https://github.com/lamalab-org/MatText
- **Paper:** To be published

## Uses

### Direct Use

The base model can be used for generating meaningful features/embeddings of bulk structures without further training.
This model is ideal if finetuned for narrowdown tasks.

### Downstream Use

This model can be used with fientuning for property prediction, classification or extractions.


## Bias, Risks, and Limitations

> Model was trained only on bulk structures (**n0w0f/MatText - pretrain2m** - dataset).  

The pertaining dataset is a subset of the materials deposited in the NOMAD archive. We queried only 3D-connected structures (i.e., excluding 2D materials, which often require special treatment) and, for consistency, limited our query to materials for which the bandgap has been computed using the PBE functional and the VASP code.

### Recommendations


## How to Get Started with the Model

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("n0w0f/MatText-slices-2m")
```

## Training Details

### Training Data

**n0w0f/MatText - pretrain2m**
The dataset contains crystal structures in various text representations and labels for some subsets.

https://huggingface.co/datasets/n0w0f/MatText

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->


### Training Procedure


#### Training Hyperparameters

- **Training regime:** fp32 <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->



### Testing Data, Factors & Metrics

#### Testing Data

https://huggingface.co/datasets/n0w0f/MatText/viewer/pretrain2m/test




## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** 8 A100 GPUs with 40GB
- **Hours used:** 72h
- **Cloud Provider:** Private Infrastructure
- **Compute Region:** US/EU
- **Carbon Emitted:** 250W x 72h = 18 kWh x 0.432 kg eq. CO2/kWh = 7.78 kg eq. CO2


## Technical Specifications 

#### Software

Pretrained using https://github.com/lamalab-org/MatText

## Citation 

If you use MatText in your work, please cite 

```
@misc{alampara2024mattextlanguagemodelsneed,
      title={MatText: Do Language Models Need More than Text & Scale for Materials Modeling?}, 
      author={Nawaf Alampara and Santiago Miret and Kevin Maik Jablonka},
      year={2024},
      eprint={2406.17295},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
      url={https://arxiv.org/abs/2406.17295}, 
}
```

## Model Card Authors

The model was trained by Nawaf Alampara ([n0w0f](https://github.com/n0w0f)), Santiago Miret ([LinkedIn]()), and Kevin Maik Jablonka ([kjappelbaum](https://github.com/kjappelbaum)).

## Model Card Contact

 [Nawaf](https://github.com/n0w0f), 
 [Kevin](https://github.com/kjappelbaum)