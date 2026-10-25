"""Iteração 4: memória episódica gravada nas fronteiras de eventos, com reinstauração.

Inspirado (como mecanismo, não como modelo biológico) na teoria de
segmentação de eventos e nos sistemas complementares de aprendizagem:

- um "córtex" recorrente (GRU de 2 camadas) lê o texto;
- em cada fronteira de evento (token <sep>, fim de uma frase), a rede
  1) consulta a memória rápida com o estado atual e recebe o que estava
     guardado (recuperação);
  2) reinjeta essa leitura no estado recorrente (reinstauração), de modo que
     a escrita seguinte pode depender do valor antigo (ex.: guardar o local
     anterior de um objeto antes de sobrescrevê-lo);
  3) escreve com a regra delta, várias cabeças por evento;
- ao final, a pergunta é respondida com leitura em passos internos (como na
  iteração 2), chaves esparsas e sinal de familiaridade (iteração 3).

Diferença essencial para FastWeightGRU: lá a memória só é lida no fim; aqui
ela é lida durante a leitura da história e alimenta o processamento.

write_mode="boundary" (iteração 4): uma escrita por frase, no <sep>.
write_mode="token" (iteração 5): escrita a cada token, como na iteração 2,
e a reinstauração no <sep> só realimenta o estado (não escreve). Com
reinstate=False, a iteração 5 faz a mesma conta da iteração 2.

Implementação: a sequência é cortada em segmentos (um por frase, terminando
em <sep>; o último é a pergunta). Cada segmento passa pela GRU do cuDNN em
paralelo (com padding, pegando o estado no último token real), começando do
estado deixado pelo anterior; o laço em Python é só sobre segmentos. É a mesma conta de um laço token a token com interação de
memória nos <sep> (verificado em teste).
"""

from __future__ import annotations

import torch
from torch import nn

from brain.models.fastweight import delta_rule_chunked, sparse_code
from brain.models.gru import StackedGRU


def segment(tokens: torch.Tensor, lengths: torch.Tensor, sep_id: int, pad_id: int):
    """Corta (B, T) em (B, L, S): segmentos que terminam em <sep>; o último é o resto.

    Retorna seg_tokens (B, L, S), seg_len (B, L) e n_segs (B,).
    """
    B, T = tokens.shape
    dev = tokens.device
    idx = torch.arange(T, device=dev).expand(B, T)
    valid = idx < lengths[:, None]
    is_sep = (tokens == sep_id) & valid
    seg_id = torch.cumsum(is_sep.long(), 1) - is_sep.long()  # o <sep> pertence à frase que fecha
    prev_sep = torch.where(is_sep, idx, torch.full_like(idx, -1))
    prev_sep = torch.cat([torch.full_like(idx[:, :1], -1), prev_sep[:, :-1]], 1).cummax(1).values
    pos = idx - (prev_sep + 1)
    n_segs = seg_id.masked_fill(~valid, -1).max(1).values + 1
    L, S = int(n_segs.max()), int(pos.masked_fill(~valid, 0).max()) + 1
    seg_tokens = torch.full((B, L, S), pad_id, dtype=tokens.dtype, device=dev)
    b = torch.arange(B, device=dev)[:, None].expand(B, T)
    seg_tokens[b[valid], seg_id[valid], pos[valid]] = tokens[valid]
    seg_len = torch.zeros(B, L, dtype=torch.long, device=dev)
    seg_len.index_put_((b[valid], seg_id[valid]), torch.ones_like(pos[valid]), accumulate=True)
    return seg_tokens, seg_len, n_segs


class EpisodicGRU(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pad_id: int, sep_id: int,
                 emb_dim: int = 128, hidden: int = 304, layers: int = 2, dropout: float = 0.1,
                 mem_heads: int = 4, mem_dim: int = 48, key_dim: int = 128, key_topk: int = 16,
                 read_hops: int = 3, familiarity: bool = True, reinstate: bool = True,
                 write_mode: str = "boundary"):
        super().__init__()
        if write_mode not in ("boundary", "token"):
            raise ValueError(f"write_mode inválido: {write_mode}")
        self.write_mode = write_mode
        self.sep_id, self.pad_id = sep_id, pad_id
        self.M, self.D, self.Dk = mem_heads, mem_dim, key_dim
        self.key_topk, self.read_hops = key_topk, read_hops
        self.familiarity, self.use_reinstate = familiarity, reinstate
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = StackedGRU(emb_dim, hidden, layers, dropout)
        read_dim = mem_heads * mem_dim
        self.to_write = nn.Linear(hidden, mem_heads * (key_dim + mem_dim + 1))  # chave, valor, β
        self.to_query = nn.Linear(hidden, mem_heads * key_dim)
        self.reinstate = nn.Linear(read_dim, hidden)
        self.reinstate_norm = nn.LayerNorm(hidden)
        self.hop_update = nn.Linear(read_dim, hidden)
        self.hop_norm = nn.LayerNorm(hidden)
        head_in = hidden + read_dim + (mem_heads if familiarity else 0)
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout),
                                  nn.Linear(head_in, num_classes))

    def _query(self, state: torch.Tensor) -> torch.Tensor:
        return sparse_code(self.to_query(state).float().view(-1, self.M, self.Dk), self.key_topk)

    @staticmethod
    def _read(query: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bmk,bmkd->bmd", query, W)

    def reinstate_step(self, top: torch.Tensor, W: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
        """Recupera da memória com o estado atual e reinjeta a leitura (só nas linhas `at`)."""
        if not self.use_reinstate:
            return top
        read = self._read(self._query(top), W).reshape(top.shape[0], -1)
        back = self.reinstate_norm(top + self.reinstate(read.to(top.dtype)))
        return torch.where(at[:, None], back, top)

    def token_writes(self, out: torch.Tensor, valid: torch.Tensor, W: torch.Tensor):
        """Escrita com regra delta a cada token de um segmento. out: (B, S, H); valid: (B, S)."""
        B, S, _ = out.shape
        p = self.to_write(out).float().view(B, S, self.M, self.Dk + self.D + 1)
        k = sparse_code(p[..., :self.Dk], self.key_topk)
        v = p[..., self.Dk:self.Dk + self.D]
        beta = torch.sigmoid(p[..., -1]) * valid[..., None]
        W = delta_rule_chunked(k.transpose(1, 2), v.transpose(1, 2), beta.transpose(1, 2), W0=W)
        return W, beta

    def boundary_step(self, top: torch.Tensor, W: torch.Tensor, at: torch.Tensor):
        """Recuperação, reinstauração e escrita nas linhas `at` (B,) que estão numa fronteira."""
        B = top.shape[0]
        top = self.reinstate_step(top, W, at)
        p = self.to_write(top).float().view(B, self.M, self.Dk + self.D + 1)
        k = sparse_code(p[..., :self.Dk], self.key_topk)
        v = p[..., self.Dk:self.Dk + self.D]
        beta = torch.sigmoid(p[..., -1]) * at[:, None]
        old = torch.einsum("bmk,bmkd->bmd", k, W)
        W = W + beta[..., None, None] * torch.einsum("bmk,bmd->bmkd", k, v - old)
        return top, W, beta

    def memory_trace(self, tokens: torch.Tensor, lengths: torch.Tensor):
        B = tokens.shape[0]
        lengths = lengths.to(tokens.device)
        seg_tokens, seg_len, n_segs = segment(tokens, lengths, self.sep_id, self.pad_id)
        layers, H = len(self.gru.layers), self.gru.layers[0].hidden_size
        h = tokens.new_zeros(layers, B, H, dtype=torch.float)
        W = torch.zeros(B, self.M, self.Dk, self.D, device=tokens.device)
        betas = []
        max_len = seg_len.max(0).values.tolist()  # uma única sincronização com a CPU
        for j in range(seg_tokens.shape[1]):
            active = seg_len[:, j] > 0
            x = self.embedding(seg_tokens[:, j, :max_len[j]])
            lens = seg_len[:, j].clamp(min=1)
            if self.write_mode == "token":
                h_new, out = self.gru.forward_padded(x, lens, h.to(x.dtype), return_outputs=True)
                valid = torch.arange(out.shape[1], device=out.device)[None, :] < seg_len[:, j, None]
                W, beta = self.token_writes(out, valid, W)
                betas.append(beta)
            else:
                h_new = self.gru.forward_padded(x, lens, h.to(x.dtype))
            h = torch.where(active[None, :, None], h_new.to(h.dtype), h)
            at = active & (j < n_segs - 1)  # todo segmento menos o último termina em <sep>
            if at.any():
                if self.write_mode == "token":
                    top = self.reinstate_step(h[-1], W, at)
                else:
                    top, W, beta = self.boundary_step(h[-1], W, at)
                    betas.append(beta)
                h = torch.cat([h[:-1], top.to(h.dtype)[None]], 0)

        state = h[-1]
        query = self._query(state)
        for hop in range(self.read_hops):
            read = self._read(query, W)
            if hop < self.read_hops - 1:
                state = self.hop_norm(state + self.hop_update(read.reshape(B, -1).to(state.dtype)))
                query = self._query(state)
        if not betas:
            beta_all = W.new_zeros(B, 0, self.M)
        else:  # forças de escrita por evento (boundary) ou por token, concatenadas no tempo
            beta_all = torch.cat([b if b.dim() == 3 else b[:, None] for b in betas], 1)
        return state, read, beta_all

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state, read, _ = self.memory_trace(tokens, lengths)
        B = read.shape[0]
        feats = [state, read.reshape(B, -1).to(state.dtype)]
        if self.familiarity:
            feats.append(read.norm(dim=-1).to(state.dtype))
        return self.head(torch.cat(feats, dim=-1))
