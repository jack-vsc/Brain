"""Fase 4 (H2): aprendizado sequencial por blocos, com e sem replay / consolidação.

Blocos (dados gerados com o mesmo gerador, mudando só as perguntas e as correções):
  localizacao → posse → correcoes
Depois de cada bloco, a rede é avaliada no teste de TODOS os blocos, o que forma a
matriz R[i][j] = acurácia no bloco j após treinar o bloco i.
Esquecimento do bloco j = max_{i < último} R[i][j] − R[último][j].

Estratégias, com o mesmo número de passos e o mesmo batch:
  none      treino sequencial puro;
  uniform   buffer de tamanho fixo (amostragem reservatório sobre tudo o que já
            foi visto); uma fração de cada batch vem do buffer;
  category  mesmo buffer, mas com cotas iguais por tarefa (tipo de pergunta);
  ewc       sem buffer; consolidação sináptica (Kirkpatrick et al., 2017):
            penalidade λ Σ F_i (θ_i − θ*_i)², com F = Fisher diagonal estimada no
            fim de cada bloco (pesos importantes para o passado ficam "rígidos");
  episodic  inspirada no hipocampo: no fim de cada bloco, as experiências mais
            "surpreendentes" (maior perda; metade do espaço) e uma amostra
            aleatória (a outra metade) são gravadas numa memória episódica SQLite
            (brain.memory.store), com cotas iguais por bloco. Durante os blocos
            seguintes, as histórias são relidas da memória (conexão reaberta) e o
            alvo é recalculado pelo solucionador simbólico sobre as frases guardadas;
            a pergunta original fica num registro ao lado. O que difere do replay
            uniforme é a seleção por surpresa e a origem dos alvos; o armazenamento
            em si não muda o aprendizado;
  episodic_ewc  episodic + ewc;
  episodic_diverse  mesma memória SQLite, mas a seleção cobre os tipos de pergunta e
            de marcação (tags) em rodízio, em vez de priorizar a surpresa;
  uniform_sleep  replay uniforme + "sono": os últimos `sleep_steps` passos de cada
            bloco usam só a memória (que já inclui o bloco do dia), sem dados novos.
            O total de passos é o mesmo das outras estratégias.
Cada exemplo de replay é uma história completa com a resposta verdadeira para
aquela história; nada substituído depois é ensinado como atual.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from brain.data.dataset import AnswerSpace, QADataset, make_collate
from brain.data.generator import GeneratorConfig, generate_split, load_split, summarize
from brain.data.tokenizer import Vocab, episode_texts
from brain.data.world import BEFORE, FIRST, HOLDER, LOC_OBJ, LOC_PERSON, YESNO
from brain.data.tokenizer import encode_example
from brain.evaluation.metrics import compute_metrics
from brain.evaluation.predict import predict
from brain.memory.store import MemoryStore
from brain.models.registry import build_model, count_parameters
from brain.symbolic.solver import ParseError, solve_episode
from brain.training.checkpoint import capture_rng_state, save_checkpoint
from brain.training.device import MB, select_device
from brain.training.trainer import WarmupCosine, set_seed

BLOCKS = {
    "localizacao": {"question_weights": {LOC_OBJ: 2.0, LOC_PERSON: 1.0, YESNO: 1.0,
                                         HOLDER: 0, FIRST: 0, BEFORE: 0}, "correction_prob": 0.0},
    "posse": {"question_weights": {HOLDER: 1.0, LOC_OBJ: 0, LOC_PERSON: 0, YESNO: 0, FIRST: 0, BEFORE: 0},
              "correction_prob": 0.0},
    "correcoes": {"question_weights": {LOC_OBJ: 2.0, FIRST: 1.0, BEFORE: 1.0, HOLDER: 0, LOC_PERSON: 0,
                                       YESNO: 0}, "correction_prob": 0.5},
}


def generate_continual_data(out_dir: Path, seed: int, n_train: int, n_test: int,
                            blocks: list[str]) -> dict:
    """Gera treino/teste por bloco (sem histórias repetidas entre blocos e splits).
    O vocabulário é a união dos treinos de todos os blocos (o léxico do micromundo é
    fixo; sem isso, palavras como 'correção' seriam <unk> nos primeiros blocos)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    manifest = {"blocks": {}, "seed": seed}
    texts = []
    for b in blocks:
        cfg = GeneratorConfig(name=b, seed=seed + 1000 * blocks.index(b), heldout_pair_fraction=0.0,
                              **BLOCKS[b])
        for split, n in (("train", n_train), ("test", n_test)):
            eps, stats = generate_split(f"{b}_{split}", n, cfg, set(), seen)
            with open(out_dir / f"{b}_{split}.jsonl", "w", encoding="utf-8") as f:
                for ep in eps:
                    f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            manifest["blocks"].setdefault(b, {})[split] = summarize(eps)
            if split == "train":
                texts += [t for ep in eps for t in episode_texts(ep)]
    Vocab.build(texts).save(out_dir / "vocab.json")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return manifest


class ReplayBuffer:
    def __init__(self, size: int, mode: str, rng: random.Random):
        self.size, self.mode, self.rng = size, mode, rng
        self.items: list[tuple] = []  # (ids, label, task)
        self.seen = 0
        self.by_task: dict[str, list[tuple]] = defaultdict(list)

    def add_block(self, ds: QADataset) -> None:
        for (ids, y), m in zip(ds.items, ds.meta):
            item = (ids, y, m["task"])
            if self.mode == "uniform":  # reservatório: amostra uniforme de tudo o que já foi visto
                self.seen += 1
                if len(self.items) < self.size:
                    self.items.append(item)
                else:
                    j = self.rng.randrange(self.seen)
                    if j < self.size:
                        self.items[j] = item
            else:
                self.by_task[m["task"]].append(item)
        if self.mode == "category":  # cotas iguais por tarefa
            quota = self.size // max(1, len(self.by_task))
            for t, lst in self.by_task.items():
                if len(lst) > quota:
                    self.by_task[t] = self.rng.sample(lst, quota)
            self.items = [x for lst in self.by_task.values() for x in lst]

    def sample(self, n: int) -> list[tuple]:
        return [self.rng.choice(self.items) for _ in range(n)] if self.items else []


@torch.no_grad()
def _per_example_loss(model, ds: QADataset, idx: list[int], device: str, collate, batch_size: int = 256) -> list[float]:
    model.eval()
    out = []
    for b in range(0, len(idx), batch_size):
        tokens, lengths, labels, _ = collate([ds[i] for i in idx[b:b + batch_size]])
        logits = model(tokens.to(device), lengths)
        out += torch.nn.functional.cross_entropy(logits.float(), labels.to(device), reduction="none").tolist()
    model.train()
    return out


class EpisodicReplay:
    """Memória episódica em SQLite + seleção por surpresa (ver docstring do módulo)."""

    def __init__(self, db: Path, size: int, rng: random.Random, vocab: Vocab, answers: AnswerSpace, qpos: str,
                 select: str = "surprise"):
        if db.exists():
            db.unlink()
        self.db, self.size, self.rng, self.select = db, size, rng, select
        self.vocab, self.answers, self.qpos = vocab, answers, qpos
        self.blocks: list[list[tuple[str, str, float]]] = []  # por bloco: (episódio, pergunta, prioridade)
        self.items: list[tuple] = []
        self.label_mismatch = 0  # alvo recalculado ≠ resposta original (deve ficar em 0)
        self.label_checked = 0

    def add_block(self, model, ds: QADataset, episodes: list[dict], device: str, collate) -> None:
        cand = self.rng.sample(range(len(ds)), min(len(ds), 4 * self.size))
        quota = self.size // (len(self.blocks) + 1)
        if self.select == "surprise":
            losses = _per_example_loss(model, ds, cand, device, collate)
            ranked = [i for _, i in sorted(zip(losses, cand), reverse=True)]
            surprising = ranked[: quota // 2]
            top = set(surprising)
            rest = [i for i in cand if i not in top]
            chosen = surprising + self.rng.sample(rest, min(len(rest), quota - len(surprising)))
            pri = dict(zip(cand, losses))
        else:  # diversidade: rodízio entre (tarefa, tags); a ordem do rodízio vira a prioridade
            strata = defaultdict(list)
            for i in cand:
                strata[(ds.meta[i]["task"], tuple(sorted(ds.meta[i]["tags"])))].append(i)
            keys = list(strata)
            self.rng.shuffle(keys)
            chosen = []
            while len(chosen) < quota and any(strata[k] for k in keys):
                for k in keys:
                    if strata[k] and len(chosen) < quota:
                        chosen.append(strata[k].pop())
            pri = {i: -r for r, i in enumerate(chosen)}
        by_id = {ep["id"]: ep for ep in episodes}
        with MemoryStore(self.db) as store:
            stored = {r[0] for r in store.db.execute("SELECT DISTINCT episode_id FROM events")}
            entries = []
            for i in chosen:
                m = ds.meta[i]
                ep = by_id[m["episode_id"]]
                if ep["id"] not in stored:
                    for line in ep["story"]:
                        store.ingest(line, ep["id"], source_id=f"bloco-{len(self.blocks)}")
                    stored.add(ep["id"])
                entries.append((ep["id"], ep["questions"][m["q_index"]]["text"], pri[i],
                                ep["questions"][m["q_index"]]["answer"]))
        self.blocks.append(entries)
        for b in range(len(self.blocks)):  # cotas iguais: corta os blocos antigos, mantendo os mais surpreendentes
            self.blocks[b] = sorted(self.blocks[b], key=lambda e: -e[2])[:quota]
        self._reload()

    def _reload(self) -> None:
        """Relê tudo da memória numa conexão nova; o alvo vem das frases guardadas."""
        self.items = []
        with MemoryStore(self.db) as store:
            for entries in self.blocks:
                for eid, qtext, _, original in entries:
                    lines = store.episode_lines(eid)
                    try:
                        answer = solve_episode(lines, qtext)
                    except ParseError:
                        continue
                    self.label_checked += 1
                    self.label_mismatch += answer != original
                    ids = encode_example(lines, qtext, self.vocab, self.qpos)
                    self.items.append((ids, self.answers.index[answer], "episodic"))

    def sample(self, n: int) -> list[tuple]:
        return [self.rng.choice(self.items) for _ in range(n)] if self.items else []


def _fisher(model, ds: QADataset, device: str, batches: int, batch_size: int, collate) -> dict:
    """Fisher diagonal empírica (gradiente² da perda) em alguns batches do bloco."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    # o backward da RNN do cuDNN exige modo de treino; o dropout fica desligado à parte
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[: batches * batch_size].tolist()
    for b in range(batches):
        chunk = [ds[i] for i in idx[b * batch_size:(b + 1) * batch_size]]
        if not chunk:
            break
        tokens, lengths, labels, _ = collate(chunk)
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(tokens.to(device), lengths).float(), labels.to(device))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 / batches
    model.zero_grad(set_to_none=True)
    model.train()
    return fisher


def run_continual(cfg: dict) -> Path:
    run, mcfg, tcfg, ccfg = cfg["run"], cfg["model"], cfg["train"], cfg["continual"]
    seed, strategy = int(run["seed"]), ccfg["strategy"]
    run_dir = Path(run["output_dir"]) / f"{run['name']}-{strategy}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    rng = random.Random(seed)
    data_dir = Path(run["data_dir"])
    blocks = list(ccfg["blocks"])
    vocab = Vocab.load(data_dir / "vocab.json")
    answers = AnswerSpace()
    qpos = mcfg["question_position"]
    train = {b: QADataset(load_split(data_dir, f"{b}_train"), vocab, answers, qpos) for b in blocks}
    test = {b: QADataset(load_split(data_dir, f"{b}_test"), vocab, answers, qpos) for b in blocks}
    device = select_device(tcfg.get("device", "auto"), tcfg.get("min_free_vram_mb", 400)).device
    model = build_model(mcfg, len(vocab), len(answers), vocab.pad_id, vocab.special_ids).to(device)
    collate = make_collate(vocab.pad_id)
    amp = device == "cuda" and tcfg.get("amp", "auto") is not False
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sleep_steps = int(ccfg.get("sleep_steps", 300)) if strategy == "uniform_sleep" else 0
    if strategy in ("uniform", "category", "uniform_sleep"):
        buffer = ReplayBuffer(int(ccfg["buffer_size"]), "category" if strategy == "category" else "uniform", rng)
    elif strategy in ("episodic", "episodic_ewc", "episodic_diverse"):
        buffer = EpisodicReplay(run_dir / "episodic.sqlite", int(ccfg["buffer_size"]), rng, vocab, answers, qpos,
                                select="diverse" if strategy == "episodic_diverse" else "surprise")
    elif strategy in ("none", "ewc"):
        buffer = None
    else:
        raise ValueError(f"estratégia desconhecida: {strategy}")
    train_eps = {b: load_split(data_dir, f"{b}_train") for b in blocks}
    replay_n = int(round(tcfg["batch_size"] * ccfg["replay_fraction"])) if buffer else 0
    ewc_terms: list[tuple[dict, dict]] = []
    steps = int(ccfg["steps_per_block"])
    R: list[dict[str, float]] = []
    log = open(run_dir / "continual.log", "a", encoding="utf-8")

    def note(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    note(f"estratégia {strategy}, {count_parameters(model):,} parâmetros, {steps} passos por bloco, "
         f"dispositivo {device}")
    t0 = time.perf_counter()
    for bi, block in enumerate(blocks):
        for group in optimizer.param_groups:
            group["lr"] = tcfg["lr"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, WarmupCosine(tcfg["warmup_steps"], steps, tcfg["min_lr_ratio"]))
        ds = train[block]
        model.train()
        added = False

        def add_to_buffer():
            if isinstance(buffer, EpisodicReplay):
                buffer.add_block(model, ds, train_eps[block], device, collate)
            elif buffer:
                buffer.add_block(ds)

        for step in range(steps):
            sleeping = sleep_steps and step >= steps - sleep_steps
            if sleeping and not added:  # o "dia" entra na memória antes do sono
                add_to_buffer()
                added = True
            # buffer vazio (1º bloco): batch inteiro do bloco atual, para o mesmo orçamento de exemplos
            n_rep = tcfg["batch_size"] if sleeping else replay_n
            rep = [(ids, y, -1) for ids, y, _ in buffer.sample(n_rep)] if buffer else []
            cur = [ds[rng.randrange(len(ds))] for _ in range(tcfg["batch_size"] - len(rep))]
            tokens, lengths, labels, _ = collate(cur + rep)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(tokens.to(device), lengths)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels.to(device))
            aux = getattr(model, "aux_loss", None)
            if aux is not None:
                loss = loss + aux
            for fisher, anchor in ewc_terms:
                loss = loss + ccfg["ewc_lambda"] * sum(
                    (fisher[n] * (p - anchor[n]) ** 2).sum() for n, p in model.named_parameters()
                    if n in fisher)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
            if (step + 1) % 500 == 0:
                note(f"bloco {block} passo {step + 1}: loss {loss.item():.4f}")
        row = {}
        for b in blocks:
            preds, _ = predict(model, test[b], device)
            row[b] = compute_metrics(test[b].meta, [answers.labels[p] for p in preds])["acc"]
        R.append(row)
        note(f"após {block}: " + "  ".join(f"{b}={row[b]:.3f}" for b in blocks))
        if not added:
            add_to_buffer()
        if strategy in ("ewc", "episodic_ewc"):
            fisher = _fisher(model, ds, device, int(ccfg.get("fisher_batches", 20)), tcfg["batch_size"], collate)
            ewc_terms.append((fisher, {n: p.detach().clone() for n, p in model.named_parameters()}))
        save_checkpoint(run_dir / f"after_{bi}_{block}.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "block": block,
            "R": R, "rng": capture_rng_state(), "config": cfg, "vocab": vocab.itos,
            "answer_labels": answers.labels, "data_version": "continual"})
        for old in run_dir.glob("after_*.pt"):  # disco: guarda só o último
            if old.name != f"after_{bi}_{block}.pt":
                old.unlink()

    last = R[-1]
    forgetting = {b: round(max(R[i][b] for i in range(j, len(R) - 1)) - last[b], 4)
                  for j, b in enumerate(blocks[:-1])}
    summary = {
        "strategy": strategy, "seed": seed, "blocks": blocks, "steps_per_block": steps,
        "batch_size": tcfg["batch_size"], "replay_per_batch": replay_n,
        "buffer_size": ccfg["buffer_size"] if buffer else 0, "sleep_steps": sleep_steps, "R": R,
        "model_type": mcfg["type"],
        "episodic_label_mismatch": (buffer.label_mismatch, buffer.label_checked)
        if isinstance(buffer, EpisodicReplay) else None,
        "final_avg_acc": round(sum(last.values()) / len(last), 4),
        "forgetting": forgetting,
        "avg_forgetting": round(sum(forgetting.values()) / max(1, len(forgetting)), 4),
        "just_learned_acc": {b: R[i][b] for i, b in enumerate(blocks)},  # plasticidade
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / MB, 1) if device == "cuda" else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    note(f"fim: acurácia média final {summary['final_avg_acc']:.4f}, "
         f"esquecimento médio {summary['avg_forgetting']:.4f}")
    log.close()
    return run_dir
