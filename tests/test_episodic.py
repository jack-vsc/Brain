"""Iteração 4: memória episódica nas fronteiras de eventos."""

import pytest
import torch

from brain.models.episodic import EpisodicGRU, segment

SEP, PAD = 2, 0


def tiny(**kw):
    torch.manual_seed(0)
    return EpisodicGRU(20, 5, pad_id=PAD, sep_id=SEP, emb_dim=8, hidden=16, dropout=0.0,
                       mem_heads=2, mem_dim=4, key_dim=12, key_topk=3, read_hops=2, **kw).eval()


def test_segmentacao_por_frases():
    tokens = torch.tensor([[5, 6, SEP, 7, SEP, 8, 9, 0], [5, SEP, 6, 0, 0, 0, 0, 0]])
    seg, seg_len, n = segment(tokens, torch.tensor([7, 3]), SEP, PAD)
    assert n.tolist() == [3, 2]
    assert seg_len.tolist() == [[3, 2, 2], [2, 1, 0]]
    assert seg[0, 0].tolist() == [5, 6, SEP]
    assert seg[0, 2, :2].tolist() == [8, 9]
    assert seg[1, 1, :1].tolist() == [6]


def reference_token_loop(m: EpisodicGRU, ids: list[int]) -> torch.Tensor:
    """Laço token a token com interação de memória em cada <sep> (uma sequência)."""
    H = m.gru.layers[0].hidden_size
    h = torch.zeros(len(m.gru.layers), 1, H)
    W = torch.zeros(1, m.M, m.Dk, m.D)
    yes = torch.tensor([True])
    for tok in ids:
        x = m.embedding(torch.tensor([[tok]]))
        new = []
        for i, layer in enumerate(m.gru.layers):
            x, hi = layer(x, h[i:i + 1])
            new.append(hi)
        h = torch.cat(new, 0)
        if m.write_mode == "token":  # escrita a cada token, antes da reinstauração do <sep>
            W, _ = m.token_writes(h[-1][:, None], torch.tensor([[True]]), W)
            if tok == SEP:
                h = torch.cat([h[:-1], m.reinstate_step(h[-1], W, yes)[None]], 0)
        elif tok == SEP:
            top, W, _ = m.boundary_step(h[-1], W, yes)
            h = torch.cat([h[:-1], top[None]], 0)
    state = h[-1]
    query = m._query(state)
    for hop in range(m.read_hops):
        read = m._read(query, W)
        if hop < m.read_hops - 1:
            state = m.hop_norm(state + m.hop_update(read.reshape(1, -1)))
            query = m._query(state)
    feats = [state, read.reshape(1, -1), read.norm(dim=-1)]
    return m.head(torch.cat(feats, -1))[0]


@pytest.mark.parametrize("write_mode", ["boundary", "token"])
@pytest.mark.parametrize("reinstate", [True, False])
def test_versao_por_frases_igual_ao_laco_token_a_token(reinstate, write_mode):
    m = tiny(reinstate=reinstate, write_mode=write_mode)
    seqs = [[4, 5, 6, SEP, 7, 8, SEP, 9, 10, 11], [4, 5, SEP, 12, 13], [6, 7, 8]]
    L = max(map(len, seqs))
    tokens = torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])
    with torch.no_grad():
        batched = m(tokens, torch.tensor([len(s) for s in seqs]))
        for i, s in enumerate(seqs):
            assert torch.allclose(batched[i], reference_token_loop(m, s), atol=1e-5), i


def test_config_episodica_no_mesmo_orcamento():
    from pathlib import Path

    from brain.models.registry import build_model, count_parameters
    from brain.training.trainer import load_config
    cfg = load_config(Path("configs/b_episodic.toml"))["model"]
    n = count_parameters(build_model(cfg, 72, 28, 0, {"sep": SEP}))
    assert 1_350_000 <= n <= 1_650_000, n


def test_reinstauracao_muda_a_escrita():
    """Com reinstauração, o que é escrito depende do que já estava na memória."""
    tokens = torch.tensor([[4, 5, SEP, 4, 5, SEP, 9]])
    with torch.no_grad():
        on = tiny(reinstate=True)(tokens, torch.tensor([7]))
        off = tiny(reinstate=False)(tokens, torch.tensor([7]))
    assert not torch.allclose(on, off)
