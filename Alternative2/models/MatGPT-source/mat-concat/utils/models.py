# utils/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

pad_token = "<pad>"
end_token = "<end>"
start_token = "<start>"

class ModifiedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout,
                 stoi, num_properties=1, conditioning_mode="scaled"):
        super().__init__()
        # The scaled-token models of the paper are implemented in ../mat and ../mat-multi
        if conditioning_mode not in {"concat", "unconditional"}:
            raise ValueError(f"Unknown conditioning mode: {conditioning_mode} (use concat or unconditional)")
        self.stoi = stoi
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        decoder_layer = nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, dropout=dropout, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.n_embd = n_embd
        self.num_properties = num_properties
        self.conditioning_mode = conditioning_mode
        if conditioning_mode == "concat":
            self.property_projection = nn.Linear(num_properties, n_embd)


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, idx, properties, targets=None):
        B, T = idx.shape
        device = idx.device
        x = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = x + pos_emb

        if self.conditioning_mode == "concat":
            # one embedding for all property values, prepended to the token sequence
            property_embedding = self.property_projection(
                properties.to(dtype=x.dtype)
            ).unsqueeze(1)
            x = torch.cat([property_embedding, x], dim=1)

        x = x.permute(1, 0, 2)
        model_length = x.size(0)
        tgt_mask = self.generate_square_subsequent_mask(model_length).to(device)
        
        # Use a dummy memory tensor
        memory = torch.zeros((model_length, B, self.n_embd), device=device)
        
        x = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if self.conditioning_mode == "concat":
            logits = logits[:, 1:, :]
        
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.stoi[pad_token])
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, stoi, itos, properties,
                 temperature=1.2, top_p=0.9):
        """Generate tokens with temperature and nucleus (top-p) sampling"""
        end_token_id = stoi[end_token]
        pad_token_id = stoi[pad_token]
        properties = properties.to(idx.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond, properties)
            logits = logits[:, -1, :]
            
            # Block padding token
            logits[:, pad_token_id] = float('-inf')
            
            # Apply temperature
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set removed tokens to 0 probability
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            # Re-normalize
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample from filtered distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == end_token_id:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

class TransformerDecoderModel(pl.LightningModule):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout,
                 stoi, itos, train_dataset, val_dataset, learning_rate=1e-4,
                 batch_size=32, num_properties=1, warmup_steps=500,
                 conditioning_mode="scaled"):
        super().__init__()
        self.model = ModifiedTransformerDecoder(
            vocab_size, n_embd, n_head, n_layer, block_size, dropout, stoi,
            num_properties=num_properties, conditioning_mode=conditioning_mode
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stoi = stoi
        self.itos = itos
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_properties = num_properties
        self.conditioning_mode = conditioning_mode
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=["train_dataset", "val_dataset"])


    def forward(self, idx, properties, targets=None):
        return self.model(idx, properties, targets)

    def training_step(self, batch, batch_idx):
        batch, properties = batch
        properties = properties.float()  # Ensure properties is a float tensor
        xb, yb = batch[:, :-1], batch[:, 1:]
        logits, loss = self(xb, properties, yb)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, properties = batch
        properties = properties.float()  # Ensure properties is a float tensor
        xb, yb = batch[:, :-1], batch[:, 1:]
        logits, loss = self(xb, properties, yb)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Total training steps estimation
        if self.train_dataset is not None:
            steps_per_epoch = len(self.train_dataset) // self.batch_size
            total_steps = steps_per_epoch * self.trainer.max_epochs
        else:
            total_steps = 10000  # fallback
        
        # Linear warmup + Cosine decay scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
                return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=9, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=9, persistent_workers=True, pin_memory=True)
