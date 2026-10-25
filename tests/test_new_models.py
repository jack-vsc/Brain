"""Iterações 8 (código de tempo na memória) e 9 (minGRU paralela)."""

import torch

from brain.models.fastweight import FastWeightGRU, time_features
from brain.models.mingru import MinGRUStack, mingru_loop, mingru_scan

SEP = 2


def test_mingru_paralela_igual_ao_laco():
    torch.manual_seed(0)
    k, h = torch.randn(3, 50, 8) * 3, torch.randn(3, 50, 8) * 3
    assert torch.allclose(mingru_scan(k, h), mingru_loop(k, h), atol=1e-4, rtol=1e-4)


def test_mingru_estavel_em_sequencia_longa():
    k, h = torch.full((1, 2000, 4), 4.0), torch.randn(1, 2000, 4)
    out = mingru_scan(k, h)
    assert torch.isfinite(out).all()


def test_mingru_causal_padding_nao_interfere():
    torch.manual_seed(0)
    m = MinGRUStack(6, 10, 2, 0.0).eval()
    x = torch.randn(1, 7, 6)
    padded = torch.cat([x, torch.randn(1, 5, 6)], 1)
    assert torch.allclose(m(x), m(padded)[:, :7], atol=1e-5)


def test_codigo_de_tempo_conta_frases():
    tokens = torch.tensor([[4, 5, SEP, 6, 7, SEP, 8, 3]])
    f = time_features(tokens, SEP)
    idx_equal = [torch.allclose(f[0, i], f[0, j]) for i, j in ((0, 2), (3, 5), (6, 7))]
    assert all(idx_equal) and not torch.allclose(f[0, 0], f[0, 3])


def tiny(**kw):
    torch.manual_seed(0)
    return FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0,
                         mem_heads=2, mem_dim=4, read_hops=2, **kw).eval()


def test_variantes_novas_nao_dependem_do_padding():
    seqs = [[4, 5, SEP, 6, 7, SEP, 3, 9], [4, SEP, 3, 9, 10, 11, 12, 13, 14, 15]]
    for kw in ({"rnn": "mingru"}, {"time_code": True, "sep_id": SEP}, {"rnn": "mingru", "time_code": True, "sep_id": SEP}):
        m = tiny(**kw)
        alone = m(torch.tensor([seqs[0]]), torch.tensor([len(seqs[0])]))
        L = max(map(len, seqs))
        batch = torch.tensor([s + [0] * (L - len(s)) for s in seqs])
        both = m(batch, torch.tensor([len(s) for s in seqs]))
        assert torch.allclose(alone[0], both[0], atol=1e-5), kw
        both.sum().backward()
