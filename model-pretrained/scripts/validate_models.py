"""Carga estricta y forward sintético; no evalúa propiedades ni tokenización."""
import ast
import gc
import json
from pathlib import Path
import torch
from transformers import BertConfig, BertModel
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
results = {}
p = ROOT / 'mattext-slices-2m'
model = BertModel(BertConfig.from_json_file(str(p / 'config.json')))
state = load_file(str(p / 'model.safetensors'), device='cpu')
model.load_state_dict(state, strict=True)
model.eval()
with torch.no_grad():
    out = model(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3)).last_hidden_state
assert torch.isfinite(out).all()
results['mattext'] = dict(strict_load=True, parameters=sum(x.numel() for x in model.parameters()), output_shape=list(out.shape), finite=True, tokenizer_validated=False)
print('MatText', results['mattext'], flush=True)
del model, state, out
gc.collect()
for variant, folder in [('unconditional', 'mat-concat'), ('formation-bandgap', 'mat-multi')]:
    p = ROOT / 'matgpt' / 'checkpoints' / variant
    info = json.loads(next(p.rglob('*.json')).read_text())
    weights = next(p.rglob('*.pth'))
    source = ROOT / 'matgpt' / 'source' / folder / 'utils' / 'models.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModifiedTransformerDecoder')
    # Isolate the original nn.Module without importing Lightning/training dependencies.
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                 rearrange=lambda tensor, pattern: tensor.unsqueeze(-1))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    extra = dict(num_properties=0, conditioning_mode='unconditional') if variant == 'unconditional' else {}
    model = scope['ModifiedTransformerDecoder'](vocab_size=len(info['stoi']), n_embd=768, n_head=12, n_layer=12, block_size=info['block_size'], dropout=0.1, stoi=info['stoi'], **extra)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out, _ = model(torch.tensor([[info['stoi']['<start>'], 0, 1]]), torch.ones(1, 3))
    assert torch.isfinite(out).all()
    results[variant] = dict(strict_load=True, parameters=sum(x.numel() for x in model.parameters()), output_shape=list(out.shape), finite=True, block_size=info['block_size'], vocab_size=len(info['stoi']), weights=str(weights.relative_to(ROOT)))
    print(variant, results[variant], flush=True)
    del model, state, out
    gc.collect()
results['note'] = 'CPU, tokens sintéticos. MatGPT: nn.Module original aislado por AST; rearrange de escala sustituido por unsqueeze equivalente. No valida calidad científica ni correspondencia del tokenizador MatText.'
(ROOT / 'new-models-validation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
