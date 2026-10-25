"""Baseline de recorrência moderna: blocos no estilo Mamba-2 (SSD), escritos do zero.

Cada bloco (pre-norm, residual):
    in_proj → z, x, B, C, Δ ;  conv1d causal (curta) em (x, B, C) + SiLU
    estado por cabeça h:  S_t = exp(Δ_t A) S_{t−1} + Δ_t B_t x_tᵀ      (A < 0, escalar por cabeça)
    saída:                y_t = C_tᵀ S_t + D x_t ;  y ← y ⊙ SiLU(z) ;  out_proj
Δ, B e C dependem da entrada ("seletivo"): a rede decide o quanto lembrar e
escrever a cada token. É a dualidade estado-espaço/atenção do Mamba-2 (Dao &
Gu, 2024), implementada aqui sem kernels especiais: a varredura em blocos é
exata (verificada contra um laço token a token em teste).

Não é o código oficial do Mamba (que exige kernels CUDA sem suporte no
Windows); é uma reimplementação simplificada para comparação.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def ssd_loop(x, dt, A, Bm, Cm):
    """Referência token a token. x (B,T,H,P); dt (B,T,H); A (H,); Bm, Cm (B,T,N)."""
    B_, T, H, P = x.shape
    S = x.new_zeros(B_, H, Bm.shape[-1], P)
    ys = []
    for t in range(T):
        S = (dt[:, t] * A).exp()[..., None, None] * S + \
            dt[:, t, :, None, None] * Bm[:, t, None, :, None] * x[:, t, :, None, :]
        ys.append(torch.einsum("bn,bhnp->bhp", Cm[:, t], S))
    return torch.stack(ys, 1)


def ssd_chunked(x, dt, A, Bm, Cm, chunk: int = 64):
    """Mesma conta de ssd_loop, em blocos: dentro do bloco é uma 'atenção' com máscara de
    decaimento exp(g_t − g_i) (i ≤ t); entre blocos, o estado S é propagado."""
    B_, T, H, P = x.shape
    S = x.new_zeros(B_, H, Bm.shape[-1], P)
    ys = []
    for s in range(0, T, chunk):
        xc, dtc, Bc, Cc = x[:, s:s + chunk], dt[:, s:s + chunk], Bm[:, s:s + chunk], Cm[:, s:s + chunk]
        C = xc.shape[1]
        g = torch.cumsum(dtc * A, dim=1)  # (B, C, H), log do decaimento acumulado (≤ 0)
        diff = g[:, :, None, :] - g[:, None, :, :]  # (B, t, i, H)
        causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))[None, :, :, None]
        decay = torch.where(causal, diff, torch.full_like(diff, float("-inf"))).exp()
        weights = decay * torch.einsum("btn,bin->bti", Cc, Bc)[..., None] * dtc[:, None]
        y = torch.einsum("btih,bihp->bthp", weights, xc)
        y = y + torch.einsum("btn,bhnp->bthp", Cc, S) * g.exp()[..., None]
        ys.append(y)
        to_end = (g[:, -1:] - g).exp() * dtc  # (B, C, H)
        S = g[:, -1].exp()[..., None, None] * S + torch.einsum("bih,bin,bihp->bhnp", to_end, Bc, xc)
    return torch.cat(ys, 1)


class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, expand: int = 2, head_dim: int = 48,
                 conv: int = 4, chunk: int = 64):
        super().__init__()
        d_inner = expand * d_model
        if d_inner % head_dim:
            raise ValueError("expand*d_model precisa ser múltiplo de head_dim")
        self.d_inner, self.N, self.P, self.H = d_inner, d_state, head_dim, d_inner // head_dim
        self.chunk = chunk
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.H)
        ch = d_inner + 2 * d_state
        self.conv = nn.Conv1d(ch, ch, kernel_size=conv, groups=ch, padding=conv - 1)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, self.H)))
        dt = torch.exp(torch.linspace(math.log(1e-3), math.log(1e-1), self.H))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus⁻¹(dt)
        self.D = nn.Parameter(torch.ones(self.H))
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B_, T, _ = u.shape
        z, xbc, dt = self.in_proj(self.norm(u)).split(
            [self.d_inner, self.d_inner + 2 * self.N, self.H], dim=-1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :T].transpose(1, 2))
        x, Bm, Cm = xbc.split([self.d_inner, self.N, self.N], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias) * valid[..., None]  # padding: estado parado
        A = -self.A_log.float().exp()
        xh = x.float().view(B_, T, self.H, self.P)
        y = ssd_chunked(xh, dt, A, Bm.float(), Cm.float(), self.chunk)
        y = (y + xh * self.D[:, None]).reshape(B_, T, self.d_inner)
        y = y.to(u.dtype) * F.silu(z)
        return u + self.out_proj(self.out_norm(y))


class SSMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pad_id: int, d_model: int = 192,
                 layers: int = 6, d_state: int = 32, expand: int = 2, head_dim: int = 48,
                 dropout: float = 0.1, chunk: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList(Mamba2Block(d_model, d_state, expand, head_dim, chunk=chunk)
                                    for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B_, T = tokens.shape
        lengths = lengths.to(tokens.device)
        valid = (torch.arange(T, device=tokens.device)[None, :] < lengths[:, None]).float()
        x = self.embedding(tokens)
        for block in self.blocks:
            x = self.drop(block(x, valid))
        last = x[torch.arange(B_, device=tokens.device), lengths - 1]
        return self.head(self.norm(last))
