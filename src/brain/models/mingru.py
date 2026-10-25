"""minGRU: recorrência que pode ser calculada em paralelo no comprimento.

Na GRU, a porta e o candidato dependem do estado anterior, então cada token
espera o anterior. Na minGRU (Feng et al., 2024, "Were RNNs All We Needed?"),
eles dependem só da entrada do token:

    z_t = σ(W_z x_t),   h̃_t = g(W_h x_t),   h_t = (1 − z_t) h_{t−1} + z_t h̃_t

Isso é uma recorrência linear h_t = a_t h_{t−1} + b_t, resolvida para todos os
tokens de uma vez em espaço log (a_t e b_t positivos; g mantém h̃ > 0):

    log h_t = A_t + logcumsumexp_s(log b_s − A_s),   A_t = Σ_{s≤t} log a_s

Com várias camadas empilhadas, a dependência entre tokens volta pela camada
seguinte. É a mesma conta do laço token a token (verificado em teste).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def log_g(x: torch.Tensor) -> torch.Tensor:
    """log de g(x) = x + 0,5 (x ≥ 0) ou σ(x) (x < 0): contínua e sempre positiva."""
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def mingru_scan(k: torch.Tensor, h_raw: torch.Tensor) -> torch.Tensor:
    """k, h_raw: (B, T, H) pré-ativações da porta e do candidato. Estado inicial zero."""
    k, h_raw = k.float(), h_raw.float()
    log_a = -F.softplus(k)  # log(1 − σ(k))
    log_b = -F.softplus(-k) + log_g(h_raw)  # log σ(k) + log g(h̃)
    a_star = log_a.cumsum(dim=1)
    return (a_star + torch.logcumsumexp(log_b - a_star, dim=1)).exp()


def mingru_loop(k: torch.Tensor, h_raw: torch.Tensor) -> torch.Tensor:
    """Referência token a token (para testes)."""
    z = torch.sigmoid(k.float())
    cand = log_g(h_raw.float()).exp()
    h = torch.zeros_like(z[:, 0])
    out = []
    for t in range(k.shape[1]):
        h = (1 - z[:, t]) * h + z[:, t] * cand[:, t]
        out.append(h)
    return torch.stack(out, 1)


class MinGRUStack(nn.Module):
    """Camadas minGRU com LayerNorm na saída de cada uma e dropout entre elas.
    Causal: o padding à direita não afeta os tokens reais."""

    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(input_size if i == 0 else hidden, 2 * hidden) for i in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, (proj, norm) in enumerate(zip(self.proj, self.norm)):
            if i > 0:
                out = self.drop(out)
            k, h_raw = proj(out).chunk(2, dim=-1)
            out = norm(mingru_scan(k, h_raw).to(x.dtype))
        return out
