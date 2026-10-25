"""Baseline Transformer pequeno para comparação controlada com os modelos recorrentes.

Encoder pre-LN com atenção bidirecional e posições rotacionais (RoPE), como
nos modelos de linguagem atuais. A classificação usa a representação do
último token real (<eos>), que enxerga a sequência inteira. A máscara de
padding garante que a saída não depende do padding do batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rope_tables(seq_len: int, dim: int, device, base: float = 10_000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()  # (T, dim/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pad_id: int, d_model: int = 192,
                 layers: int = 4, heads: int = 4, ff: int = 640, dropout: float = 0.1):
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model/heads precisa ser inteiro e par (RoPE)")
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList(Block(d_model, heads, ff, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.head_dim = d_model // heads

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        lengths = lengths.to(tokens.device)
        key_mask = (torch.arange(T, device=tokens.device)[None, :] < lengths[:, None])[:, None, None, :]
        cos, sin = rope_tables(T, self.head_dim, tokens.device)
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, key_mask, cos, sin)
        last = x[torch.arange(B, device=tokens.device), lengths - 1]
        return self.head(self.ln_f(last))
