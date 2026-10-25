"""Baseline da Fase 2: embedding treinável + GRU + cabeça de classificação.

O padding é tratado com pack_padded_sequence, de modo que o estado final
corresponde ao último token real de cada sequência (o resultado não depende
do comprimento das outras sequências no batch).
"""

from __future__ import annotations

import re

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class StackedGRU(nn.Module):
    """GRU multicamada com o dropout entre camadas aplicado fora do cuDNN.

    nn.GRU(num_layers>1, dropout>0) usa o estado de dropout do cuDNN, que derruba
    o processo ao sair (0xC0000409) com torch 2.14.1+cu130 no Windows. O bug foi
    reproduzido com um script mínimo e não ocorre com dropout=0. Empilhar camadas
    simples com nn.Dropout entre elas é equivalente (mesmos parâmetros, na mesma
    ordem) e evita esse caminho. Checkpoints no formato de nn.GRU continuam
    carregando: as chaves weight_ih_l1 etc. são renomeadas na carga.
    """

    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.GRU(input_size if i == 0 else hidden, hidden, batch_first=True)
            for i in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, packed: PackedSequence,
                h0: torch.Tensor | None = None) -> tuple[PackedSequence, torch.Tensor]:
        """h0 opcional: (camadas, B, hidden), como em nn.GRU."""
        finals = []
        out = packed
        for i, layer in enumerate(self.layers):
            if i > 0:
                out = PackedSequence(self.drop(out.data), out.batch_sizes,
                                     out.sorted_indices, out.unsorted_indices)
            out, h = layer(out, None if h0 is None else h0[i:i + 1].contiguous())
            finals.append(h)
        return out, torch.cat(finals, dim=0)

    def forward_padded(self, x: torch.Tensor, lengths: torch.Tensor,
                       h0: torch.Tensor | None = None, return_outputs: bool = False):
        """Sem empacotar: roda com padding à direita e pega o estado de cada camada no
        último token real. A GRU é causal, então o padding depois dele não interfere.
        Evita a sincronização CPU↔GPU de pack_padded_sequence. lengths >= 1.
        Com return_outputs, devolve também as saídas da última camada (B, S, H)."""
        idx = torch.arange(x.shape[0], device=x.device)
        last = (lengths - 1).to(x.device)
        finals = []
        out = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                out = self.drop(out)
            out, _ = layer(out, None if h0 is None else h0[i:i + 1].contiguous())
            finals.append(out[idx, last])
        return (torch.stack(finals, 0), out) if return_outputs else torch.stack(finals, 0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        pattern = re.compile(re.escape(prefix) + r"(weight|bias)_(ih|hh)_l(\d+)")
        for key in list(state_dict):
            m = pattern.fullmatch(key)
            if m:
                state_dict[f"{prefix}layers.{m[3]}.{m[1]}_{m[2]}_l0"] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pad_id: int,
                 emb_dim: int = 128, hidden: int = 384, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = StackedGRU(emb_dim, hidden, layers, dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, num_classes))

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])
