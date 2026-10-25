"""H1: onde vive a memória? Quatro condições reportadas separadamente.

1. pesos lentos: a rede vê só a pergunta (o que ficou retido nos pesos);
2. estado temporário: história + pergunta no mesmo contexto (caso normal);
3. memória externa após reinício: a história é gravada no SQLite em duas
   sessões, a conexão é fechada e reaberta ("reinício"), e as frases
   recuperadas por projeto/entidade entram como evidência. A rede calcula a
   resposta; o solucionador simbólico aparece só como diagnóstico de
   suficiência da recuperação, em coluna separada;
4. sinapses persistentes após reinício (só fastweight_gru): a memória rápida W
   de cada história é salva em disco; um modelo recarregado do checkpoint lê W
   e responde com o estado recorrente zerado.

Salvar dados em disco não demonstra que os pesos lentos aprenderam os fatos:
por isso a condição 1 é reportada à parte.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from brain.data.dataset import QADataset
from brain.data.generator import load_split
from brain.data.tokenizer import BOS, EOS, Q, SEP, tokenize
from brain.evaluation.metrics import compute_metrics
from brain.evaluation.predict import load_model, predict
from brain.memory.store import MemoryStore
from brain.symbolic.solver import ParseError, solve_episode
from brain.training.checkpoint import load_checkpoint, save_checkpoint
from brain.training.device import select_device


def _with_story(episodes: list[dict], story_fn) -> list[dict]:
    """Uma cópia de cada episódio por pergunta, com a história trocada por story_fn(ep, qi)."""
    out = []
    for ep in episodes:
        for qi, q in enumerate(ep["questions"]):
            out.append({"id": f"{ep['id']}#{qi}", "story": story_fn(ep, qi), "questions": [q]})
    return out


def _accuracy(model, episodes, vocab, answers, qpos, device):
    ds = QADataset(episodes, vocab, answers, qpos)
    preds, _ = predict(model, ds, device)
    return compute_metrics(ds.meta, [answers.labels[p] for p in preds])


def _build_store(episodes: list[dict], db: Path) -> None:
    if db.exists():
        db.unlink()
    with MemoryStore(db) as store:
        for ep in episodes:
            half = len(ep["story"]) // 2
            for i, line in enumerate(ep["story"]):
                store.ingest(line, ep["id"], source_id="sessao-1" if i < half else "sessao-2")


def _retrieved(db: Path, episodes: list[dict]) -> dict[tuple[str, int], list[str]]:
    out = {}
    with MemoryStore(db) as store:  # conexão nova: o "reinício"
        for ep in episodes:
            for qi, q in enumerate(ep["questions"]):
                out[(ep["id"], qi)] = store.retrieve(q["project"], [q["subject"].lower()], ep["id"])
    return out


def _ids(vocab, toks: list[str]) -> list[int]:
    return vocab.encode_tokens(toks)


@torch.no_grad()
def _synaptic(checkpoint: Path, episodes: list[dict], device: str, tmp: Path, batch: int = 64):
    model, vocab, answers, _ = load_model(checkpoint, device)
    pad = vocab.pad_id
    tmp.mkdir(parents=True, exist_ok=True)
    # sessão 1: ler cada história e salvar só a memória sináptica em disco
    for b in range(0, len(episodes), batch):
        chunk = episodes[b:b + batch]
        stories = [_ids(vocab, [BOS] + [t for line in ep["story"] for t in tokenize(line) + [SEP]])
                   for ep in chunk]
        L = max(map(len, stories))
        toks = torch.tensor([s + [pad] * (L - len(s)) for s in stories], device=device)
        W = model.encode_story(toks, torch.tensor([len(s) for s in stories], device=device))
        save_checkpoint(tmp / f"W_{b:06d}.pt", {"W": W.cpu(), "ids": [ep["id"] for ep in chunk]})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    # reinício: modelo recarregado, sem estado; só os pesos lentos + W do disco
    model, vocab, answers, _ = load_model(checkpoint, device)
    meta, preds = [], []
    for b in range(0, len(episodes), batch):
        saved = load_checkpoint(tmp / f"W_{b:06d}.pt")
        chunk = episodes[b:b + batch]
        rows, which = [], []
        for e, ep in enumerate(chunk):
            for q in ep["questions"]:
                rows.append(_ids(vocab, [Q] + tokenize(q["text"]) + [EOS]))
                which.append(e)
                meta.append({"task": q["task"], "tags": q["tags"], "answer": q["answer"]})
        L = max(map(len, rows))
        qt = torch.tensor([r + [pad] * (L - len(r)) for r in rows], device=device)
        W = saved["W"].to(device)[torch.tensor(which, device=device)]
        logits = model.answer_from_memory(qt, torch.tensor([len(r) for r in rows], device=device), W)
        preds += [answers.labels[i] for i in logits.argmax(-1).tolist()]
    return compute_metrics(meta, preds)


def run_memory_benchmark(checkpoint: Path, splits: tuple[str, ...] = ("test_iid", "test_long"),
                         max_episodes: int = 500, device_pref: str = "auto",
                         out_dir: Path | None = None) -> dict:
    checkpoint = Path(checkpoint)
    device = select_device(device_pref, min_free_vram_mb=200).device
    model, vocab, answers, ckpt = load_model(checkpoint, device)
    cfg = ckpt["config"]
    qpos = cfg["model"]["question_position"]
    data_dir = Path(cfg["run"]["data_dir"])
    out_dir = Path(out_dir or checkpoint.parent / "memory_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint), "model": cfg["model"]["type"],
              "max_episodes": max_episodes, "splits": {}}
    for split in splits:
        eps = load_split(data_dir, split)[:max_episodes]
        db = out_dir / f"{split}.sqlite"
        _build_store(eps, db)
        retrieved = _retrieved(db, eps)
        r = {
            "1_pesos_lentos": _accuracy(model, _with_story(eps, lambda ep, qi: []),
                                        vocab, answers, qpos, device),
            "2_estado_temporario": _accuracy(model, _with_story(eps, lambda ep, qi: ep["story"]),
                                             vocab, answers, qpos, device),
            "3_memoria_externa": _accuracy(model, _with_story(eps, lambda ep, qi: retrieved[(ep["id"], qi)]),
                                           vocab, answers, qpos, device),
        }
        index = {(ep["id"], qi): q for ep in eps for qi, q in enumerate(ep["questions"])}
        # diagnóstico (não é resposta da rede): a evidência recuperada basta para um solucionador exato?
        ok = n = 0
        lines = []
        for key, q in index.items():
            ev = retrieved[key]
            lines.append(len(ev))
            try:
                ok += solve_episode(ev, q["text"]) == q["answer"]
            except ParseError:
                pass
            n += 1
        r["diagnostico_recuperacao"] = {
            "suficiencia_simbolica": round(ok / max(1, n), 4),
            "frases_recuperadas_media": round(sum(lines) / max(1, len(lines)), 2),
            "frases_historia_media": round(sum(len(ep["story"]) for ep in eps) / max(1, len(eps)), 2),
        }
        if cfg["model"]["type"] == "fastweight_gru":
            tmp = out_dir / f"W_{split}"
            r["4_sinapses_persistentes"] = _synaptic(checkpoint, eps, device, tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        report["splits"][split] = r
    (out_dir / "memory_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    (out_dir / "memory_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(r: dict) -> str:
    def f(x):
        return "—" if x is None else f"{x:.3f}"
    lines = [f"# Onde vive a memória? (`{r['checkpoint']}`)", "",
             f"Modelo `{r['model']}`, até {r['max_episodes']} episódios por split.", "",
             "| split | condição | acc | isolamento | UNKNOWN inventado ↓ |", "|---|---|---|---|---|"]
    for split, d in r["splits"].items():
        for cond in ("1_pesos_lentos", "2_estado_temporario", "3_memoria_externa", "4_sinapses_persistentes"):
            if cond in d:
                m = d[cond]
                lines.append(f"| {split} | {cond} | {f(m['acc'])} | {f(m['isolation_acc'])} | "
                             f"{f(m['unknown_false_answer_rate'])} |")
        diag = d["diagnostico_recuperacao"]
        lines.append(f"| {split} | diagnóstico: evidência suficiente (solucionador simbólico) | "
                     f"{f(diag['suficiencia_simbolica'])} | — | — |")
        lines.append(f"| {split} | frases recuperadas / frases na história | "
                     f"{diag['frases_recuperadas_media']} / {diag['frases_historia_media']} | — | — |")
    return "\n".join(lines) + "\n"
