from transformers import Pipeline
import torch
from typing import Dict, List, Union

class MatterGPTPipeline(Pipeline):
    def __init__(self, model, tokenizer, device=-1):
        super().__init__(model=model, tokenizer=tokenizer, device=device)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs: Union[Dict[str, float], List[Dict[str, float]]]) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, dict):
            inputs = [inputs]
        
        conditions = [[input['formation_energy'], input['band_gap']] for input in inputs]
        context = '>'
        x = torch.tensor([self.tokenizer.stoi[context]], dtype=torch.long)[None,...].repeat(len(conditions), 1).to(self.device)
        p = torch.tensor(conditions, dtype=torch.float).unsqueeze(1).to(self.device)
        
        return {"input_ids": x, "prop": p}

    def _forward(self, model_inputs):
        return self.model.generate(
            model_inputs["input_ids"], 
            prop=model_inputs["prop"],
            max_length=self.model.config.block_size, 
            temperature=1.2, 
            do_sample=True, 
            top_k=0, 
            top_p=0.9
        )

    def postprocess(self, model_outputs):
        return [self.tokenizer.decode(seq.tolist()) for seq in model_outputs]

    def __call__(self, inputs: Union[Dict[str, float], List[Dict[str, float]]]):
        pre_processed = self.preprocess(inputs)
        model_outputs = self._forward(pre_processed)
        return self.postprocess(model_outputs)