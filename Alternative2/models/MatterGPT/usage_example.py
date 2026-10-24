from mattergpt_wrapper import MatterGPTWrapper, SimpleTokenizer
import torch
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = "./"  # Directory containing config.json and pytorch_model.bin
if not os.path.exists(os.path.join(model_path, "config.json")):
    raise FileNotFoundError(f"Config file not found in {model_path}")
if not os.path.exists(os.path.join(model_path, "pytorch_model.pt")):
    raise FileNotFoundError(f"Model weights not found in {model_path}")

model = MatterGPTWrapper.from_pretrained(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Model loaded from {model_path}")

# Load the tokenizer
tokenizer_path = "Voc_prior"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer vocabulary file not found at {tokenizer_path}")
tokenizer = SimpleTokenizer(tokenizer_path)
logger.info(f"Tokenizer loaded from {tokenizer_path}")

# Function to generate a single sequence
def generate_single(condition):
    context = '>'
    x = torch.tensor([tokenizer.stoi[context]], dtype=torch.long)[None,...].to(model.device)
    p = torch.tensor([condition]).unsqueeze(1).to(model.device)
    
    generated = model.generate(x, prop=p, max_length=model.config.block_size, temperature=1.2, do_sample=True, top_k=0, top_p=0.9)
    return tokenizer.decode(generated[0].tolist())

# Function to generate multiple sequences
def generate_multiple(condition, num_sequences, batch_size=32):
    all_sequences = []
    for _ in tqdm(range(0, num_sequences, batch_size)):
        current_batch_size = min(batch_size, num_sequences - len(all_sequences))
        context = '>'
        x = torch.tensor([tokenizer.stoi[context]], dtype=torch.long)[None,...].repeat(current_batch_size, 1).to(model.device)
        p = torch.tensor([condition]).repeat(current_batch_size, 1).unsqueeze(1).to(model.device)
        
        generated = model.generate(x, prop=p, max_length=model.config.block_size, temperature=1.2, do_sample=True, top_k=0, top_p=0.9)
        all_sequences.extend([tokenizer.decode(seq.tolist()) for seq in generated])
        
        if len(all_sequences) >= num_sequences:
            break
    
    return all_sequences[:num_sequences]

# Example usage
condition = [-1.0, 2.0]  # eform and bandgap

# Generate a single sequence
logger.info("Generating a single sequence:")
single_sequence = generate_single(condition)
print(single_sequence)
print()

# Generate multiple sequences
num_sequences = 10
logger.info(f"Generating {num_sequences} sequences:")
multiple_sequences = generate_multiple(condition, num_sequences)
for i, seq in enumerate(multiple_sequences, 1):
    print(seq)

logger.info("Generation complete")