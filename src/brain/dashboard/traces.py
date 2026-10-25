"""Gravação de atividade real para o mapa do painel.

Roda checkpoints treinados (na CPU, sem disputar a GPU) em algumas histórias de
teste e registra, token a token:
  Transformer: atividade do fluxo residual por camada e a atenção de cada camada;
  FastWeightGRU: atividade de cada camada recorrente, a força de escrita β na
  memória rápida e, em cada leitura (hop), quanto cada token contribuiu.

A contribuição é exata, não aproximada: na regra delta sem porta de esquecimento,
W_t = W_{t-1} + k_t u_tᵀ com u_t = β_t (v_t − W_{t-1}ᵀ k_t), logo W = Σ_t k_t u_tᵀ e
a leitura qᵀW = Σ_t (q·k_t) u_t. A contribuição do token t é a projeção de
(q·k_t) u_t sobre a leitura. As duas reconstruções são conferidas contra a saída
do próprio modelo (logits iguais) antes de serem gravadas.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from brain.data.generator import load_split
from brain.data.tokenizer import encode_example
from brain.evaluation.predict import load_model
from brain.models.transformer import apply_rope, rope_tables

UNITS = 32  # neurônios mostrados por camada (os de maior variância na história)


def hoyer(x: torch.Tensor) -> torch.Tensor:
    """Esparsidade de Hoyer por linha: 0 = todos ativos por igual, 1 = só um ativo."""
    n = x.shape[-1]
    l1, l2 = x.abs().sum(-1), x.norm(dim=-1).clamp_min(1e-9)
    return (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1)


def _q(x: torch.Tensor, scale: float | None = None) -> list:
    """Quantiza para inteiros em [-99, 99] (JSON compacto)."""
    m = scale or float(x.abs().max().clamp_min(1e-9))
    return (x / m * 99).round().clamp(-99, 99).to(torch.int16).tolist()


def _units(acts: torch.Tensor) -> torch.Tensor:
    return acts.var(0).topk(min(UNITS, acts.shape[1])).indices.sort().values


def _layer(acts: torch.Tensor) -> dict:
    """acts: (T, H). Amostra de neurônios, energia e esparsidade por token."""
    return {"act": _q(acts[:, _units(acts)]),
            "energy": _q(acts.norm(dim=-1)),
            "hoyer": [round(v, 3) for v in hoyer(acts).tolist()]}


def _top3(logits: torch.Tensor, labels: list[str]) -> list:
    p = logits.float().softmax(-1)
    v, i = p.topk(3)
    return [[labels[j], round(float(x), 3)] for x, j in zip(v, i)]


@torch.no_grad()
def trace_transformer(model, ids: list[int], labels: list[str]) -> dict:
    tokens = torch.tensor([ids])
    T = len(ids)
    cos, sin = rope_tables(T, model.head_dim, "cpu")
    x = model.embedding(tokens)
    layers, edges, last_att = [], [], []
    for blk in model.blocks:
        B, _, D = x.shape
        q, k, v = blk.qkv(blk.ln1(x)).view(B, T, 3, blk.heads, D // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)  # (1, H, T, T), sem padding
        x = x + blk.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + blk.ff(blk.ln2(x))
        layers.append(_layer(x[0]))
        mean = att[0].mean(0)  # (T, T) média das cabeças
        w, src = mean.topk(min(2, T), dim=-1)
        edges.append([[t, int(s), int(round(float(a) * 100))] for t in range(T)
                      for a, s in zip(w[t], src[t])])
        last_att.append(_q(att[0, :, -1, :], 1.0))  # (H, T): o que o último token (resposta) olha
    logits = model.head(model.ln_f(x[0, -1]))
    ref = model(tokens, torch.tensor([T]))[0]
    assert torch.allclose(logits, ref, atol=1e-4), "reconstrução do Transformer diverge do modelo"
    return {"layers": layers, "edges": edges, "answer_attention": last_att, "top3": _top3(logits, labels),
            "heads": model.blocks[0].heads,
            "connections": len(model.blocks) * model.blocks[0].heads * T * T}


@torch.no_grad()
def trace_fastweight(model, ids: list[int], labels: list[str]) -> dict:
    if model.separate_keys or model.forget_gate or model.adaptive_hops:
        raise ValueError("o traço exato cobre a variante sem chaves separadas, porta ou passos adaptativos")
    tokens = torch.tensor([ids])
    T = len(ids)
    out = model.embedding(tokens)
    layers = []
    for i, gru in enumerate(model.gru.layers):
        out, _ = gru(out)
        layers.append(_layer(out[0]))
    h = out[0]  # (T, H)
    M, D = model.mem_heads, model.mem_dim
    p = model.to_mem(h).float().view(T, M, 3 * D + 1)
    k = F.normalize(p[..., :D], dim=-1)
    v = p[..., D:2 * D]
    beta = torch.sigmoid(p[..., 3 * D])  # (T, M)
    W = torch.zeros(M, D, D)
    U = torch.zeros(T, M, D)
    for t in range(T):
        u = beta[t, :, None] * (v[t] - torch.einsum("md,mde->me", k[t], W))
        W = W + k[t, :, :, None] * u[:, None, :]
        U[t] = u
    query = F.normalize(p[-1, :, 2 * D:3 * D], dim=-1)
    state = h[-1]
    states, reads, hops = [], [], []
    for hop in range(model.read_hops):
        read = torch.einsum("md,mde->me", query, W)  # (M, D)
        sim = torch.einsum("md,tmd->tm", query, k)  # (T, M)
        unit = read / read.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        contrib = sim * torch.einsum("tmd,md->tm", U, unit)  # projeção na leitura; soma = |leitura|
        hops.append({"contrib": _q(contrib.T), "read_norm": [round(float(x), 3) for x in read.norm(dim=-1)]})
        states.append(state[None])
        reads.append(read[None])
        if hop < model.read_hops - 1:
            state = model.hop_norm(state + model.hop_update(read.reshape(M * D)))
            query = F.normalize(model.hop_query(state).float().view(M, D), dim=-1)
    logits = model._logits(states, reads, "cpu")[0]
    ref = model(tokens, torch.tensor([T]))[0]
    assert torch.allclose(logits, ref, atol=1e-4), "reconstrução da rede rápida diverge do modelo"
    return {"layers": layers, "beta": _q(beta, 1.0), "hops": hops, "top3": _top3(logits, labels),
            "mem_heads": M, "writes_strong": round(float((beta.mean(-1) > 0.5).float().mean()), 3),
            "connections": T * len(model.gru.layers) + T * M + model.read_hops * M * T}


def _pick_examples(data_dir: Path) -> list[tuple[str, dict, int]]:
    """Histórias fixas e variadas: uma pergunta encadeada (portador), uma temporal e uma longa."""
    wants = [("test_iid", "holder"), ("test_iid", "before"), ("test_long", "loc_obj")]
    out = []
    for split, task in wants:
        for ep in load_split(data_dir, split):
            qi = next((i for i, q in enumerate(ep["questions"]) if q["task"] == task
                       and q["answer"] != "UNKNOWN"), None)
            if qi is not None and (split == "test_long" or 4 <= len(ep["story"]) <= 7):
                out.append((split, ep, qi))
                break
    return out


def record_traces(checkpoints: dict[str, Path]) -> dict:
    """checkpoints: {"transformer": caminho, "ours": caminho}."""
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    models = {name: load_model(Path(p), "cpu") for name, p in checkpoints.items()}
    _, vocab, answers, ckpt = models["ours"]
    data_dir = Path(ckpt["config"]["run"]["data_dir"])
    examples = []
    for split, ep, qi in _pick_examples(data_dir):
        q = ep["questions"][qi]
        ex = {"split": split, "story": ep["story"], "question": q["text"], "answer": q["answer"],
              "task": q["task"], "tags": q["tags"]}
        for name, (model, vocab, answers, c) in models.items():
            ids = encode_example(ep["story"], q["text"], vocab, c["config"]["model"]["question_position"])
            ex.setdefault("tokens", [vocab.itos[i] for i in ids])
            kind = c["config"]["model"]["type"]
            fn = trace_transformer if kind == "transformer" else trace_fastweight
            ex[name] = fn(model, ids, answers.labels)
        examples.append(ex)
    meta = {name: {"checkpoint": str(p), "type": models[name][3]["config"]["model"]["type"],
                   "params": sum(x.numel() for x in models[name][0].parameters()),
                   "config": models[name][3]["config"]["model"]} for name, p in checkpoints.items()}
    return {"models": meta, "examples": examples, "units": UNITS}
