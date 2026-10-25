"""Painel: traços de atividade batem com o modelo e o coletor lê runs/ e filas."""

import json

import torch

from brain.dashboard.collect import collect_state
from brain.dashboard.traces import trace_fastweight, trace_transformer
from brain.models.fastweight import FastWeightGRU
from brain.models.transformer import TransformerClassifier

LABELS = [f"c{i}" for i in range(5)]
IDS = [4, 5, 6, 2, 7, 8, 2, 3, 9, 10, 11]


def test_traco_do_transformer_reconstroi_a_saida():
    torch.manual_seed(0)
    m = TransformerClassifier(20, 5, 0, d_model=16, layers=2, heads=2, ff=32, dropout=0.0).eval()
    t = trace_transformer(m, IDS, LABELS)  # a função confere os logits internamente
    assert len(t["layers"]) == 2 and len(t["layers"][0]["act"]) == len(IDS)
    assert len(t["answer_attention"][0]) == 2 and len(t["answer_attention"][0][0]) == len(IDS)


def test_traco_da_rede_rapida_reconstroi_a_saida_e_soma_da_contribuicao():
    torch.manual_seed(0)
    m = FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0,
                      mem_heads=2, mem_dim=4, read_hops=3, q_id=3).eval()
    t = trace_fastweight(m, IDS, LABELS)
    assert len(t["hops"]) == 3 and len(t["beta"]) == len(IDS)
    assert all(len(h["contrib"]) == 2 and len(h["contrib"][0]) == len(IDS) for h in t["hops"])


def test_coletor_le_execucoes_e_filas(tmp_path):
    run = tmp_path / "runs" / "x-s1"
    run.mkdir(parents=True)
    cfg = {"run": {"name": "x", "data_dir": "data/generated/d"}, "model": {"type": "gru"},
           "train": {"max_steps": 10, "lr": 1e-3, "batch_size": 4}}
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "summary.json").write_text(json.dumps({"steps": 10}))
    (run / "metrics.jsonl").write_text(json.dumps({"type": "train", "step": 10, "loss": 1.0}) + "\n" +
                                       json.dumps({"type": "eval", "step": 10, "val_acc": 0.5}) + "\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "x.toml").write_text('[run]\nname = "x"\n')
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "queue_a.txt").write_text("# fila a\ntrain configs/x.toml 1\ntrain configs/x.toml 2\n")
    s = collect_state(tmp_path)
    assert s["runs"][0]["status"] == "concluído" and s["runs"][0]["best_val"] == 0.5
    q = s["queues"][0]
    assert q["title"] == "fila a" and [j["status"] for j in q["jobs"]] == ["feito", "na fila"]


def test_detalhe_ao_vivo_de_treino_e_do_continuo(tmp_path):
    run = tmp_path / "runs" / "y-s1"
    run.mkdir(parents=True)
    cfg = {"run": {"name": "y", "data_dir": "data/generated/d"}, "model": {"type": "gru"},
           "train": {"max_steps": 1000, "lr": 1e-3, "batch_size": 4}}
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"type": "train", "step": s, "loss": 1.0, "lr": 1e-3, "elapsed_s": s / 10}) + "\n"
        for s in (100, 200, 300)))
    (run / "train.log").write_text("[00:00:01] passo 300: loss 1.0\n", encoding="utf-8")
    cont = tmp_path / "runs" / "c-uniform-s1"
    cont.mkdir()
    (cont / "continual.log").write_text(
        "[23:59:00] bloco a passo 500: loss 1.0\n[00:00:40] bloco a passo 1000: loss 0.5\n"
        "[00:00:41] após a: a=0.900  b=0.100\n[00:01:30] bloco b passo 500: loss 0.4\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "continual.toml").write_text(
        '[run]\nname = "c"\n[continual]\nblocks = ["a", "b"]\nsteps_per_block = 1000\n')
    s = collect_state(tmp_path)
    live = s["runs"][0]["live"]
    assert s["runs"][0]["status"] == "treinando"
    assert live["sec_per_step"] == 0.1 and live["eta_s"] == 70 and live["log_tail"]
    c = s["continual"]["running"][0]
    assert (c["step"], c["total"], c["block"], c["strategy"]) == (1500, 2000, "b", "uniform")
    assert c["after"] == {"a": {"a": 0.9, "b": 0.1}}
    assert c["sec_per_step"] == 0.15 and c["eta_s"] == 75  # 150 s por 1000 passos, atravessando a meia-noite
