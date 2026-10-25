"""Memória sináptica persistente: história → W em disco → reinício → pergunta."""

import torch

from brain.models.fastweight import FastWeightGRU
from brain.training.checkpoint import load_checkpoint, save_checkpoint

Q = 3  # id do token <q> nos testes


def tiny(**kw):
    torch.manual_seed(0)
    return FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0,
                         mem_heads=2, mem_dim=4, read_hops=3, q_id=Q, **kw).eval()


def batch():
    seqs = [[4, 5, 6, 2, 7, 8, 2, Q, 9, 10, 11], [4, 5, 2, Q, 12, 13]]
    L = max(map(len, seqs))
    return torch.tensor([s + [0] * (L - len(s)) for s in seqs]), torch.tensor([len(s) for s in seqs])


def test_separa_historia_e_pergunta():
    tokens, lengths = batch()
    story, qpos, question, q_len = tiny().split_story_question(tokens, lengths)
    assert qpos.tolist() == [7, 3] and q_len.tolist() == [4, 3]
    assert story[1, :3].tolist() == [4, 5, 2] and story[1, 3:].eq(0).all()
    assert question[0].tolist() == [Q, 9, 10, 11] and question[1].tolist() == [Q, 12, 13, 0]


def test_memoria_salva_em_disco_responde_igual_apos_reinicio(tmp_path):
    m = tiny()
    tokens, lengths = batch()
    story, qpos, question, q_len = m.split_story_question(tokens, lengths)
    with torch.no_grad():
        W = m.encode_story(story, qpos)
        direct = m.answer_from_memory(question, q_len, W)
    save_checkpoint(tmp_path / "w.pt", {"W": W})
    m2 = tiny()  # "processo novo": mesmo checkpoint de pesos lentos, nenhum estado recorrente
    with torch.no_grad():
        after = m2.answer_from_memory(question, q_len, load_checkpoint(tmp_path / "w.pt")["W"])
    assert torch.allclose(direct, after)


def test_memoria_vazia_muda_a_resposta():
    """Sem a memória da história, a resposta precisa mudar (o conhecimento estava em W)."""
    m = tiny()
    tokens, lengths = batch()
    story, qpos, question, q_len = m.split_story_question(tokens, lengths)
    with torch.no_grad():
        with_mem = m.answer_from_memory(question, q_len, m.encode_story(story, qpos))
        empty = m.answer_from_memory(question, q_len, torch.zeros_like(m.encode_story(story, qpos)))
    assert not torch.allclose(with_mem, empty)


def test_treino_com_reinicios_simulados_tem_gradiente():
    m = tiny(reset_question_prob=1.0).train()
    tokens, lengths = batch()
    m(tokens, lengths).sum().backward()
    assert m.to_mem.weight.grad is not None and m.to_mem.weight.grad.abs().sum() > 0
