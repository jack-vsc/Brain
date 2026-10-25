"""Protótipo: GRU + memória rápida associativa atualizada pela regra delta.

Pesos lentos (embedding, GRU, projeções) são aprendidos no treino. A memória
rápida W é uma matriz por episódio, zerada no início, que a própria rede
reescreve enquanto lê. A cada token, a partir do estado da GRU, ela produz
uma chave k, um valor v e uma força de escrita β ∈ (0, 1):

    antigo = Wᵀk                  (o que a memória associa hoje a k)
    W ← W + β · k (v − antigo)ᵀ   (substitui o valor antigo em vez de somar)

No fim, uma consulta q (do último token) lê Wᵀq, e a resposta usa o estado
da GRU junto com essa leitura. A ideia de "fast weights" é antiga (Schmidhuber,
1992; Ba et al., 2016) e a regra delta aparece em trabalhos recentes (DeltaNet,
Titans etc.); aqui é uma combinação experimental, não uma descoberta.

Iteração 2 (read_hops > 1): "passos internos" de leitura. Após cada leitura,
o estado de controle é atualizado com o que foi lido e gera uma nova
consulta, permitindo cadeias como objeto → portador → local do portador.

Varredura em blocos (exata): dentro de um bloco de C tokens, com estado
inicial S, os "pseudo-valores" u_t = β_t (v_t − Sᵀk_t − Σ_{i<t} (k_iᵀk_t) u_i)
satisfazem (I + diag(β)·tril(KKᵀ, −1)) U = diag(β)(V − KS), um sistema
triangular unitário; o estado final é S + KᵀU. É a mesma conta do laço
token a token (verificado em teste), resolvida pela GPU de uma vez por bloco,
como na forma paralela do DeltaNet.

Limitação: a memória não realimenta a GRU durante a leitura do texto.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from brain.models.gru import StackedGRU
from brain.models.mingru import MinGRUStack

TIME_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)  # períodos (em frases) do código de tempo


def time_features(tokens: torch.Tensor, sep_id: int) -> torch.Tensor:
    """Índice da frase de cada token (quantos <sep> vieram antes) em senos/cossenos de várias escalas,
    inspirado nas "células de tempo" do hipocampo: dá à memória um "quando" para cada fato."""
    idx = (tokens == sep_id).long().cumsum(1) - (tokens == sep_id).long()  # o próprio <sep> fecha a frase
    ang = idx[..., None].float() * (2 * torch.pi / torch.tensor(TIME_SCALES, device=tokens.device))
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def delta_rule_loop(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                    log_alpha: torch.Tensor | None = None, W0: torch.Tensor | None = None) -> torch.Tensor:
    """Referência token a token. k, v: (B, M, T, D); beta: (B, M, T). Retorna W (B, M, D, D).

    Com log_alpha (B, M, T), regra delta com porta de esquecimento (Gated DeltaNet):
    a memória decai por α_t antes da escrita: W ← α_t W; W ← W + β k (v − Wᵀk)ᵀ.
    """
    B, M, T, D = k.shape
    W = k.new_zeros(B, M, D, v.shape[-1]) if W0 is None else W0
    for t in range(T):
        if log_alpha is not None:
            W = W * log_alpha[:, :, t, None, None].exp()
        old = torch.einsum("bmd,bmde->bme", k[:, :, t], W)
        W = W + beta[:, :, t, None, None] * torch.einsum("bmd,bme->bmde", k[:, :, t], v[:, :, t] - old)
    return W


def delta_rule_chunked(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                       chunk: int = 64, W0: torch.Tensor | None = None,
                       log_alpha: torch.Tensor | None = None) -> torch.Tensor:
    """Mesma conta de delta_rule_loop, resolvida por blocos com um sistema triangular.
    W0 opcional: estado inicial da memória (padrão: zeros).

    Com porta de esquecimento, seja g_t = Σ_{i≤t} log α_i dentro do bloco. Então
    u_t = β_t (v_t − e^{g_t} Sᵀk_t − Σ_{i<t} e^{g_t−g_i} (k_iᵀk_t) u_i) e o estado
    final é e^{g_C} S + Σ_i e^{g_C−g_i} k_i u_iᵀ. Só aparecem razões ≤ 1 (estável).
    """
    B, M, T, D = k.shape
    W = k.new_zeros(B, M, D, v.shape[-1]) if W0 is None else W0
    for s in range(0, T, chunk):
        kc, vc, bc = k[:, :, s:s + chunk], v[:, :, s:s + chunk], beta[:, :, s:s + chunk]
        C = kc.shape[2]
        gram = torch.tril(kc @ kc.transpose(-1, -2), -1)
        if log_alpha is None:
            rhs = bc[..., None] * (vc - kc @ W)
        else:
            g = torch.cumsum(log_alpha[:, :, s:s + chunk], dim=-1)  # (B, M, C)
            diff = g[..., :, None] - g[..., None, :]
            lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=k.device), -1)
            gram = gram * torch.where(lower, diff, torch.full_like(diff, float("-inf"))).exp()
            rhs = bc[..., None] * (vc - g.exp()[..., None] * (kc @ W))
        A = torch.eye(C, device=k.device) + bc[..., None] * gram
        U = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
        if log_alpha is None:
            W = W + kc.transpose(-1, -2) @ U
        else:
            W = g[..., -1, None, None].exp() * W + kc.transpose(-1, -2) @ ((g[..., -1:] - g).exp()[..., None] * U)
    return W


def sparse_code(x: torch.Tensor, topk: int) -> torch.Tensor:
    """Separação de padrões: mantém só as `topk` maiores ativações (ReLU) e normaliza."""
    if topk > 0:
        vals, idx = torch.relu(x).topk(topk, dim=-1)
        x = torch.zeros_like(x).scatter(-1, idx, vals)
    return F.normalize(x, dim=-1)


class FastWeightGRU(nn.Module):
    """GRU + memória rápida.

    Iteração 3 (opcional): `key_dim` > mem_dim com `key_topk` > 0 dá chaves e
    consultas grandes e esparsas (separação de padrões, inspirada no giro
    denteado); `familiarity` acrescenta à cabeça a força da leitura de cada
    cabeça de memória (sinal de "já vi isto?").

    Iteração 6 (`adaptive_hops`): passos de leitura adaptativos, no estilo
    PonderNet. Após cada leitura n, a rede estima h_n = P(parar aqui | não parou
    antes). A distribuição de parada p_n = h_n ∏_{m<n} (1 − h_m) (o último passo
    recebe o resto) pondera as representações de cada passo, e `ponder_cost` ×
    E[passos] entra na perda (aux_loss). `last_ponder` guarda E[passos] por
    exemplo, para ver se a rede "pensa mais" nas perguntas difíceis.

    Iteração 7 (`forget_gate`): regra delta com porta de esquecimento por cabeça
    (Gated DeltaNet), α_t = sigmoid(...) ∈ (0, 1), iniciada perto de 1.

    Memória sináptica persistente: `encode_story` lê só a história e devolve a
    matriz W; `answer_from_memory` responde à pergunta partindo de W, com o
    estado recorrente zerado (como depois de um reinício). Com
    `reset_question_prob` > 0, o treino usa esse caminho dividido em uma fração
    dos batches, para a rede aprender a guardar o que importa em W. Requer `q_id`.
    """

    def __init__(self, vocab_size: int, num_classes: int, pad_id: int,
                 emb_dim: int = 128, hidden: int = 352, layers: int = 2, dropout: float = 0.1,
                 mem_heads: int = 4, mem_dim: int = 48, read_hops: int = 1,
                 scan: str = "chunked", chunk: int = 64,
                 key_dim: int | None = None, key_topk: int = 0, familiarity: bool = False,
                 adaptive_hops: bool = False, ponder_cost: float = 0.01,
                 forget_gate: bool = False, forget_bias: float = 3.0, reset_question_prob: float = 0.0,
                 q_id: int | None = None, rnn: str = "gru", time_code: bool = False,
                 sep_id: int | None = None):
        super().__init__()
        if rnn not in ("gru", "mingru"):
            raise ValueError(f"rnn desconhecida: {rnn}")
        if time_code and (sep_id is None or key_dim is not None or reset_question_prob > 0):
            raise ValueError("time_code precisa de sep_id e não combina com chaves separadas nem com reinícios")
        self.rnn, self.time_code, self.sep_id = rnn, time_code, sep_id
        if reset_question_prob > 0 and q_id is None:
            raise ValueError("reset_question_prob precisa de q_id (id do token <q>)")
        self.reset_question_prob, self.q_id = reset_question_prob, q_id
        self.mem_heads, self.mem_dim, self.read_hops = mem_heads, mem_dim, read_hops
        self.scan, self.chunk = scan, chunk
        self.separate_keys = key_dim is not None
        self.key_dim = key_dim or mem_dim
        self.key_topk, self.familiarity = key_topk, familiarity
        self.adaptive_hops, self.ponder_cost = adaptive_hops, ponder_cost
        self.forget_gate = forget_gate
        self.aux_loss: torch.Tensor | None = None
        self.last_ponder: torch.Tensor | None = None
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = StackedGRU(emb_dim, hidden, layers, dropout) if rnn == "gru" else \
            MinGRUStack(emb_dim, hidden, layers, dropout)
        if time_code:  # código de tempo da frase → somado à chave e ao valor de cada escrita
            self.time_proj = nn.Linear(2 * len(TIME_SCALES), mem_heads * 2 * mem_dim)
        if self.separate_keys:
            self.to_mem = nn.Linear(hidden, mem_heads * (mem_dim + 1))  # valor + β
            self.to_key = nn.Linear(hidden, mem_heads * self.key_dim)
            self.to_query = nn.Linear(hidden, mem_heads * self.key_dim)
        else:
            self.to_mem = nn.Linear(hidden, mem_heads * (3 * mem_dim + 1))
        read_dim = mem_heads * mem_dim
        if read_hops > 1:
            self.hop_update = nn.Linear(read_dim, hidden)
            self.hop_norm = nn.LayerNorm(hidden)
            self.hop_query = nn.Linear(hidden, mem_heads * self.key_dim)
        if adaptive_hops:
            if read_hops < 2:
                raise ValueError("adaptive_hops precisa de read_hops >= 2")
            self.halt = nn.Linear(hidden, 1)
        if forget_gate:
            self.to_alpha = nn.Linear(hidden, mem_heads)
            # viés 3 → α ≈ 0,95 por token (iteração 7: esquece ~95% em 60 tokens); viés 6 → α ≈ 0,9975
            nn.init.constant_(self.to_alpha.bias, forget_bias)
        head_in = hidden + read_dim + (mem_heads if familiarity else 0)
        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(dropout),
                                  nn.Linear(head_in, num_classes))

    def _keys(self, raw: torch.Tensor) -> torch.Tensor:
        return sparse_code(raw, self.key_topk)

    def memory_trace(self, tokens: torch.Tensor, lengths: torch.Tensor, W0: torch.Tensor | None = None):
        """Executa escrita + leitura e devolve (estados por passo, leituras por passo, forças β).
        W0: memória inicial (padrão: vazia). A memória final fica em self._last_W."""
        B, T = tokens.shape
        x = self.embedding(tokens)
        if self.rnn == "mingru":
            h = self.gru(x)  # causal: o padding à direita não interfere
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=T)

        M, D, Dk = self.mem_heads, self.mem_dim, self.key_dim
        lengths = lengths.to(tokens.device)
        idx = torch.arange(B, device=tokens.device)
        last = lengths - 1
        if self.separate_keys:
            p = self.to_mem(h).float().view(B, T, M, D + 1)  # memória sempre em fp32
            v, beta_raw = p[..., :D], p[..., D]
            k = self._keys(self.to_key(h).float().view(B, T, M, Dk))
            q_last = self._keys(self.to_query(h[idx, last]).float().view(B, M, Dk))
        else:
            p = self.to_mem(h).float().view(B, T, M, 3 * D + 1)
            k_raw, v = p[..., :D], p[..., D:2 * D]
            if self.time_code:
                tk, tv = self.time_proj(time_features(tokens, self.sep_id)).float().view(B, T, M, 2 * D).chunk(2, -1)
                k_raw, v = k_raw + tk, v + tv
            k = F.normalize(k_raw, dim=-1)
            q_last = F.normalize(p[idx, last, :, 2 * D:3 * D], dim=-1)
            beta_raw = p[..., 3 * D]
        valid = torch.arange(T, device=tokens.device)[None, :] < lengths[:, None]
        beta = torch.sigmoid(beta_raw) * valid[..., None]  # padding não escreve
        log_alpha = None
        if self.forget_gate:  # padding não esquece (log α = 0)
            log_alpha = F.logsigmoid(self.to_alpha(h).float()) * valid[..., None]

        Tm = int(lengths.max())
        args = (k[:, :Tm].transpose(1, 2), v[:, :Tm].transpose(1, 2), beta[:, :Tm].transpose(1, 2))
        la = None if log_alpha is None else log_alpha[:, :Tm].transpose(1, 2)
        if self.scan == "chunked":
            W = delta_rule_chunked(*args, chunk=self.chunk, log_alpha=la, W0=W0)
        else:
            W = delta_rule_loop(*args, log_alpha=la, W0=W0)

        state, query = h[idx, last], q_last
        states, reads = [], []
        for hop in range(self.read_hops):
            read = torch.einsum("bmd,bmde->bme", query, W)  # (B, M, D)
            states.append(state)
            reads.append(read)
            if hop < self.read_hops - 1:
                state = self.hop_norm(state + self.hop_update(read.reshape(B, M * D).to(state.dtype)))
                query = self._keys(self.hop_query(state).float().view(B, M, Dk))
        self._last_W = W
        return states, reads, beta

    def split_story_question(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """Separa cada sequência em história (antes de <q>) e pergunta (de <q> ao fim)."""
        lengths = lengths.to(tokens.device)
        is_q = tokens == self.q_id
        if not is_q.any(1).all():
            raise ValueError("toda sequência precisa de um token <q>")
        qpos = is_q.float().argmax(1)
        q_len = lengths - qpos
        Q = int(q_len.max())
        j = torch.arange(Q, device=tokens.device)[None, :]
        gather = (qpos[:, None] + j).clamp(max=tokens.shape[1] - 1)
        question = torch.where(j < q_len[:, None], tokens.gather(1, gather), torch.full_like(gather, 0))
        S = int(qpos.max())
        story = torch.where(torch.arange(S, device=tokens.device)[None, :] < qpos[:, None],
                            tokens[:, :S], torch.zeros_like(tokens[:, :S]))
        return story, qpos, question, q_len

    def encode_story(self, story: torch.Tensor, story_len: torch.Tensor) -> torch.Tensor:
        """Lê só a história e devolve a memória sináptica W (B, M, Dk, D)."""
        self.memory_trace(story, story_len.clamp(min=1))
        return self._last_W

    def answer_from_memory(self, question: torch.Tensor, q_len: torch.Tensor,
                           W0: torch.Tensor) -> torch.Tensor:
        """Responde partindo da memória W0 e com o estado recorrente zerado (reinício)."""
        return self._logits(*self.memory_trace(question, q_len, W0=W0)[:2], question.device)

    def _features(self, state: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        B = read.shape[0]
        feats = [state, read.reshape(B, -1).to(state.dtype)]
        if self.familiarity:
            feats.append(read.norm(dim=-1).to(state.dtype))  # força da leitura por cabeça
        return torch.cat(feats, dim=-1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.training and self.reset_question_prob > 0 and \
                torch.rand(()).item() < self.reset_question_prob:
            story, qpos, question, q_len = self.split_story_question(tokens, lengths)
            return self.answer_from_memory(question, q_len, self.encode_story(story, qpos))
        states, reads, _ = self.memory_trace(tokens, lengths)
        return self._logits(states, reads, tokens.device)

    def _logits(self, states, reads, device) -> torch.Tensor:
        if not self.adaptive_hops:
            self.aux_loss = self.last_ponder = None
            return self.head(self._features(states[-1], reads[-1]))
        remaining = torch.ones(states[0].shape[0], device=device)
        mix, expected = 0.0, 0.0
        for n, (s, r) in enumerate(zip(states, reads)):
            if n < len(states) - 1:
                p = remaining * torch.sigmoid(self.halt(s).float().squeeze(-1))
            else:
                p = remaining  # o último passo recebe o resto da probabilidade
            mix = mix + p[:, None] * self._features(s, r).float()
            expected = expected + (n + 1) * p
            remaining = remaining - p
        self.last_ponder = expected.detach()
        self.aux_loss = self.ponder_cost * expected.mean()
        return self.head(mix.to(states[0].dtype))
