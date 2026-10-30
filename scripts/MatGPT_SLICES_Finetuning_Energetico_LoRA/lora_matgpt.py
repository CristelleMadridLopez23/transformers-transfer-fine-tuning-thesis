"""LoRA Q/V para la autoatención fusionada de los bloques finales de MatGPT.

La implementación conserva todos los pesos originales y la máscara causal.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LOG = logging.getLogger(__name__)


class QVLoRAAttention(nn.Module):
    """Sustituye únicamente la autoatención con deltas LoRA en Query y Value."""

    def __init__(self, base: nn.MultiheadAttention, rank: int = 4,
                 alpha: float = 8.0, dropout: float = 0.05) -> None:
        """Inicializa factores; `base` contiene los pesos pretrained congelados."""
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention) or not base._qkv_same_embed_dim:
            raise TypeError("Se requiere nn.MultiheadAttention con QKV fusionada")
        if base.batch_first or base.bias_k is not None or base.bias_v is not None:
            raise ValueError("Configuración de MHA incompatible con MatGPT oficial")
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Parámetros LoRA inválidos")
        self.base = base
        self.rank, self.alpha, self.lora_dropout = rank, alpha, dropout
        dim = base.embed_dim
        self.q_a = nn.Parameter(torch.empty(rank, dim))
        self.q_b = nn.Parameter(torch.zeros(dim, rank))
        self.v_a = nn.Parameter(torch.empty(rank, dim))
        self.v_b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.q_a, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.v_a, a=5 ** 0.5)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        base.eval()

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Tensor | None = None, need_weights: bool = False,
                attn_mask: Tensor | None = None, average_attn_weights: bool = True,
                is_causal: bool = False) -> tuple[Tensor, None]:
        """Calcula atención causal en formato [secuencia, lote, dimensión]."""
        del average_attn_weights
        if need_weights:
            raise ValueError("La ruta MatGPT requiere need_weights=False")
        dim, heads = self.base.embed_dim, self.base.num_heads
        t, batch, _ = query.shape
        source_t = key.shape[0]
        weight, bias = self.base.in_proj_weight, self.base.in_proj_bias
        bq = None if bias is None else bias[:dim]
        bk = None if bias is None else bias[dim:2 * dim]
        bv = None if bias is None else bias[2 * dim:]
        scale = self.alpha / self.rank
        q = F.linear(query, weight[:dim], bq) + scale * F.linear(
            F.linear(F.dropout(query, self.lora_dropout, self.training), self.q_a), self.q_b)
        k = F.linear(key, weight[dim:2 * dim], bk)
        v = F.linear(value, weight[2 * dim:], bv) + scale * F.linear(
            F.linear(F.dropout(value, self.lora_dropout, self.training), self.v_a), self.v_b)
        shape = (batch, heads, -1, dim // heads)
        q = q.reshape(t, batch, heads, dim // heads).permute(1, 2, 0, 3)
        k = k.reshape(source_t, batch, heads, dim // heads).permute(1, 2, 0, 3)
        v = v.reshape(source_t, batch, heads, dim // heads).permute(1, 2, 0, 3)
        mask: Tensor | None = attn_mask
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float("-inf"))
            mask = mask.to(dtype=q.dtype, device=q.device)
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
        elif is_causal:
            mask = torch.triu(torch.full((t, source_t), float("-inf"),
                                         device=q.device, dtype=q.dtype), diagonal=1)[None, None]
        if key_padding_mask is not None:
            pad = key_padding_mask.to(device=q.device)
            if pad.dtype == torch.bool:
                pad = torch.zeros_like(pad, dtype=q.dtype).masked_fill(pad, float("-inf"))
            pad = pad[:, None, None, :].to(q.dtype)
            mask = pad if mask is None else mask + pad
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.permute(2, 0, 1, 3).contiguous().view(t, batch, dim)
        return self.base.out_proj(attended), None


def insert_qv_lora(model: Any, blocks: tuple[int, int] = (10, 11),
                   rank: int = 4, alpha: float = 8.0,
                   dropout: float = 0.05) -> list[QVLoRAAttention]:
    """Inserta adaptadores solo en la autoatención de los bloques indicados."""
    layers = model.transformer_decoder.layers
    if len(layers) != 12 or blocks != (10, 11):
        raise ValueError("Se esperaban 12 bloques y adaptación de índices 10,11")
    adapters: list[QVLoRAAttention] = []
    for index in blocks:
        layer = layers[index]
        if not isinstance(layer.self_attn, nn.MultiheadAttention):
            raise TypeError(f"Autoatención inesperada en bloque {index}")
        adapter = QVLoRAAttention(layer.self_attn, rank, alpha, dropout)
        layer.self_attn = adapter
        adapters.append(adapter)
    LOG.info("%s", {"event": "lora_inserted", "blocks": blocks,
                   "parameters": sum(p.numel() for a in adapters for p in a.parameters() if p.requires_grad)})
    return adapters
