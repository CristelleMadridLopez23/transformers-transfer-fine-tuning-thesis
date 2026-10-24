import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from model import GPT, GPTConfig  # Import your original model and config classes
import json

class CustomGPTConfig(PretrainedConfig):
    model_type = "gpt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class MatterGPTWrapper(PreTrainedModel):
    config_class = CustomGPTConfig
    base_model_prefix = "gpt"

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(GPTConfig(**config.__dict__))

    def forward(self, input_ids, attention_mask=None, labels=None, prop=None):
        return self.model(input_ids, targets=labels, prop=prop)

    def generate(self, input_ids, prop, max_length, num_return_sequences=1, **kwargs):
        steps = max_length - input_ids.shape[1]
        return self.model.sample(input_ids, steps, prop=prop, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config_file = f"{pretrained_model_path}/config.json"
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = CustomGPTConfig(**config_dict)
        
        model = cls(config)
        
        state_dict = torch.load(f"{pretrained_model_path}/pytorch_model.pt", map_location="cpu")
        model.model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        torch.save(self.model.state_dict(), f"{save_directory}/pytorch_model.pt")

class SimpleTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = f.read().splitlines()
        self.vocab = sorted(set(self.vocab + ['<', '>']))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi[token] for token in text.split()]

    def decode(self, ids):
        return " ".join([self.itos[int(i)] for i in ids if i in self.itos]).replace("<", "").strip()

    def __call__(self, text, return_tensors=None):
        encoded = self.encode(text)
        if return_tensors == 'pt':
            import torch
            return {'input_ids': torch.tensor([encoded])}
        return {'input_ids': [encoded]}