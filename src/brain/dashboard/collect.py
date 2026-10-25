"""Estado do projeto para o painel: só lê o que já está em disco, nada é estimado.

Fontes: runs/*/ (config, summary, eval-best, metrics.jsonl), experiments/queue_*.txt
e .log, docs/ITERACOES.md, docs/ROADMAP.md, data/generated/*/manifest.json e os
resultados do memory-benchmark.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
import tomllib
from pathlib import Path

from brain.training.control import paused_since

SPLITS = ("val", "test_iid", "test_comp", "test_long")
KEYS = ("acc", "macro_task_acc", "heldout_pair_acc", "isolation_acc", "unknown_false_answer_rate",
        "temporal_acc", "correction_acc")
RUNNING_WINDOW_S = 180  # metrics.jsonl tocado há menos que isso e sem summary → em treino
# Cada iteração do diário e o grupo de execuções que a mede (mesmo orçamento: ~1,5 M parâmetros,
# 8 mil passos, batch 64). Um prefixo terminado em "*" pega o melhor grupo pela validação.
ITERATIONS = [
    ("GRU", "baseline", "b_gru_lr1e-3"),
    ("Transformer", "baseline", "b_transformer_lr5e-4"),
    ("Mamba-2", "baseline", "b_mamba2_lr*"),
    ("It. 1", "memória rápida (regra delta)", "b_fastweight_lr1e-3"),
    ("It. 2", "3 leituras internas", "b_fastweight_hops"),
    ("It. 3", "chaves esparsas", "b_fastweight_sep"),
    ("It. 4", "memória episódica", "b_episodic"),
    ("It. 5", "reinstauração", "b_reinstate"),
    ("It. 5 sem", "ablação: sem reinstauração", "b_reinstate_off"),
    ("It. 6", "leituras adaptativas", "b_adaptive"),
    ("It. 7", "porta de esquecimento", "b_gated"),
    ("It. 7b", "porta iniciada aberta", "b_gated_slow"),
    ("It. 8", "código de tempo na memória", "b_time"),
    ("It. 9", "minGRU paralela", "b_mingru"),
]


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _curve(points: list, n: int = 60) -> list:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]


def _run(d: Path, now: float) -> dict | None:
    cfg = _json(d / "config.json")
    if cfg is None:
        return None
    summary = _json(d / "summary.json")
    ev = _json(d / "eval-best.json")
    train, evals, timing, last_eval = [], [], [], None
    mfile = d / "metrics.jsonl"
    if mfile.exists():
        for line in mfile.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("type") == "train":
                train.append([m["step"], m["loss"]])
                if "elapsed_s" in m:
                    timing.append((m["step"], m["elapsed_s"], m.get("lr")))
            elif m.get("type") == "eval":
                evals.append([m["step"], m["val_acc"], m.get("val_per_task")])
                last_eval = m
    if summary:
        status = "concluído"
    elif d.name.endswith("-pilot"):
        status = "piloto"
    elif mfile.exists() and now - mfile.stat().st_mtime < RUNNING_WINDOW_S:
        status = "treinando"
    else:
        status = "parado"
    m = re.fullmatch(r"(.+)-s(\d+)(-pilot)?", d.name)
    out = {
        "dir": d.name, "group": m.group(1) if m else d.name, "seed": int(m.group(2)) if m else None,
        "pilot": bool(m and m.group(3)), "status": status, "model": cfg["model"],
        "max_steps": cfg["train"]["max_steps"], "lr": cfg["train"]["lr"],
        "batch_size": cfg["train"]["batch_size"], "data": Path(cfg["run"]["data_dir"]).name,
        "step": (summary or {}).get("steps") or (train[-1][0] if train else 0),
        "loss": train[-1][1] if train else None,
        "loss_curve": _curve(train), "val_curve": [[s, a] for s, a, _ in evals],
        "best_val": max((a for _, a, _ in evals), default=None),
        "last_val_per_task": evals[-1][2] if evals else None,
        "summary": summary, "updated": (mfile.stat().st_mtime if mfile.exists() else None),
    }
    out["live"] = _live(d, out, timing, last_eval, now)
    if ev:
        out["eval"] = {s: {k: ev["splits"][s]["neural"].get(k) for k in KEYS} |
                       {"per_task": {t: v["acc"] for t, v in ev["splits"][s]["neural"]["per_task"].items()}}
                       for s in SPLITS if s in ev["splits"]}
        out["latency_ms"] = ev["latency"]["p50_ms"]
        out["n_params"] = ev["n_params"]
        out["eval_vram_mb"] = ev.get("resources", {}).get("peak_vram_mb")
        out["checkpoint_mb"] = ev.get("checkpoint_mb")
    return out


LIVE_WINDOW_S = 6 * 3600  # execuções tocadas há menos que isso levam o detalhe (log, velocidade)


def _tail(p: Path, n: int) -> list[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-n:]
    except OSError:
        return []


def _live(d: Path, run: dict, timing: list, last_eval: dict | None, now: float) -> dict | None:
    """Detalhe para o painel ao clicar numa execução: velocidade medida nos últimos registros,
    tempo restante só para quem está treinando, últimas linhas do train.log."""
    if not run["updated"] or now - run["updated"] > LIVE_WINDOW_S:
        return None
    recent = timing[-11:]
    sps = None
    if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
        sps = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
    left = max(0, run["max_steps"] - run["step"])
    model = {k: v for k, v in run["model"].items() if not isinstance(v, (dict, list))}
    return {
        "elapsed_s": timing[-1][1] if timing else None, "sec_per_step": round(sps, 4) if sps else None,
        "eta_s": round(left * sps) if sps and run["status"] == "treinando" else None,
        "lr_now": timing[-1][2] if timing else None, "age_s": round(now - run["updated"]),
        "val_loss": (last_eval or {}).get("val_loss"), "val_macro": (last_eval or {}).get("val_macro_task_acc"),
        "peak_vram_mb": (last_eval or {}).get("peak_vram_mb"), "ram_mb": (last_eval or {}).get("ram_mb"),
        "model": model, "log_tail": _tail(d / "train.log", 14),
    }


CONT_LINE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\] bloco (\w+) passo (\d+): loss ([\d.]+)")
CONT_AFTER = re.compile(r"após (\w+): (.*)")


def _continual_live(d: Path, cfg: dict | None, now: float) -> dict:
    """Progresso de um benchmark contínuo em andamento, lido do continual.log."""
    c = (cfg or {}).get("continual", {})
    blocks, spb = c.get("blocks", ["localizacao", "posse", "correcoes"]), c.get("steps_per_block", 3000)
    lines = _tail(d / "continual.log", 400)
    points, after, stamps = [], {}, []
    for ln in lines:
        if m := CONT_LINE.search(ln):
            h, mi, s, block, step, loss = m.groups()
            if block in blocks:
                g = blocks.index(block) * spb + int(step)
                points.append([g, float(loss)])
                stamps.append((g, int(h) * 3600 + int(mi) * 60 + int(s)))
        elif m := CONT_AFTER.search(ln):
            after[m.group(1)] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(2))}
    done = points[-1][0] if points else 0
    total = len(blocks) * spb
    sps = None
    if len(stamps) >= 2:
        dt = (stamps[-1][1] - stamps[0][1]) % 86400  # o log só tem a hora; atravessar a meia-noite dá a volta
        if stamps[-1][0] > stamps[0][0] and dt > 0:
            sps = dt / (stamps[-1][0] - stamps[0][0])
    mtime = (d / "continual.log").stat().st_mtime
    return {"dir": d.name, "blocks": blocks, "steps_per_block": spb, "step": done, "total": total,
            "block": blocks[min(len(blocks) - 1, done // spb)] if points else blocks[0],
            "loss_curve": points, "after": after, "sec_per_step": round(sps, 4) if sps else None,
            "eta_s": round((total - done) * sps) if sps else None, "age_s": round(now - mtime),
            "strategy": d.name.split("-")[-2] if d.name.count("-") >= 2 else None,
            "model_type": ((cfg or {}).get("model") or {}).get("type"), "log_tail": lines[-14:]}


def _mean_std(vals: list) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else None, "n": len(vals)}


def _state_cost(m: dict) -> dict:
    """Quanto cada arquitetura guarda para ler uma história (contagem de números, pela configuração).
    fixed: não cresce com o texto; per_token: cresce a cada token (ativações que o Transformer mantém
    para todas as posições); persist_bytes: o que precisa ir para o disco para responder depois de um
    reinício sem reler o texto (só implementado na memória sináptica da fastweight_gru)."""
    t = m.get("type")
    if t == "transformer":
        return {"fixed": 0, "per_token": m.get("d_model", 192) * (m.get("layers", 4) + 1), "persist_bytes": None}
    if t == "fastweight_gru":
        h, L, M, D = m.get("hidden", 352), m.get("layers", 2), m.get("mem_heads", 4), m.get("mem_dim", 48)
        w = M * (m.get("key_dim") or D) * D
        return {"fixed": w + h * L, "per_token": 0, "persist_bytes": w * 4}
    if t == "gru":
        return {"fixed": m.get("hidden", 0) * m.get("layers", 1), "per_token": 0, "persist_bytes": None}
    return {"fixed": None, "per_token": None, "persist_bytes": None}


def _groups(runs: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in runs:
        if not r["pilot"] and r.get("eval"):
            by.setdefault(r["group"], []).append(r)
    out = []
    for name, rs in by.items():
        g = {"name": name, "type": rs[0]["model"]["type"], "seeds": sorted(r["seed"] for r in rs),
             "n_params": rs[0].get("n_params"), "steps": rs[0]["summary"]["steps"] if rs[0]["summary"] else None,
             "batch_size": rs[0]["batch_size"], "lr": rs[0]["lr"], "splits": {},
             "resources": {
                 "weights_mb": round(rs[0].get("n_params", 0) * 4 / 2 ** 20, 2),  # pesos em fp32
                 "checkpoint_mb": _mean_std([r.get("checkpoint_mb") for r in rs]),  # inclui estado do otimizador
                 "vram_train_mb": _mean_std([(r["summary"] or {}).get("peak_vram_mb") for r in rs]),
                 "vram_eval_mb": _mean_std([r.get("eval_vram_mb") for r in rs]),
                 "ram_train_mb": _mean_std([(r["summary"] or {}).get("ram_mb") for r in rs]),
                 "latency_ms": _mean_std([r.get("latency_ms") for r in rs]),
                 "train_s": _mean_std([(r["summary"] or {}).get("elapsed_s") for r in rs]),
                 "state": _state_cost(rs[0]["model"]),
             }}
        for s in SPLITS:
            g["splits"][s] = {k: _mean_std([r["eval"].get(s, {}).get(k) for r in rs]) for k in KEYS}
            tasks = sorted({t for r in rs for t in r["eval"].get(s, {}).get("per_task", {})})
            g["splits"][s]["per_task"] = {t: _mean_std([r["eval"].get(s, {}).get("per_task", {}).get(t)
                                                        for r in rs]) for t in tasks}
        out.append(g)
    return sorted(out, key=lambda g: -(g["splits"]["test_iid"]["acc"] or {"mean": 0})["mean"])


def _iterations(groups: list[dict]) -> list[dict]:
    by = {g["name"]: g for g in groups}
    out = []
    for label, what, name in ITERATIONS:
        if name.endswith("*"):
            cands = [g for g in groups if g["name"].startswith(name[:-1])]
            g = max(cands, key=lambda g: (g["splits"]["val"]["acc"] or {"mean": 0})["mean"], default=None)
        else:
            g = by.get(name)
        out.append({"label": label, "what": what, "group": g["name"] if g else name, "has": g is not None})
    return out


SCALE_ALIASES = {"b_fastweight_hops": ("fw", "1.5m", "8k"), "b_fastweight_hops_x3": ("fw", "1.5m", "24k"),
                 "b_transformer_lr5e-4": ("tf", "1.5m", "8k"), "b_transformer_lr5e-4_x3": ("tf", "1.5m", "24k")}


def _scaling(groups: list[dict]) -> list[dict]:
    """Curva de escala: acerto por tamanho (parâmetros) e orçamento (passos), para as duas redes."""
    out = []
    for g in groups:
        m = re.fullmatch(r"sc_(fw|tf)_([\d.]+m)_(8k|24k)", g["name"])
        key = (m.groups() if m else None) or SCALE_ALIASES.get(g["name"])
        if not key:
            continue
        out.append({"model": key[0], "size": key[1], "budget": key[2], "group": g["name"], "n_params": g["n_params"],
                    "seeds": g["seeds"], "test_iid": g["splits"]["test_iid"]["acc"],
                    "heldout": g["splits"]["test_comp"]["heldout_pair_acc"], "long": g["splits"]["test_long"]["acc"]})
    return out


def _run_dir_for(root: Path, config: str, seed: int | None, sets: list[str]) -> str:
    """Mesma regra de scripts/run_queue.py (nome da execução + seed)."""
    try:
        cfg = tomllib.loads((root / config).read_text(encoding="utf-8"))
    except OSError:
        return "?"
    run = dict(cfg.get("run", {}))
    for item in sets:
        key, _, raw = item.partition("=")
        if key.startswith("run."):
            run[key[4:]] = tomllib.loads(f"v = {raw}")["v"]
    return f"{run.get('name', 'gru_baseline')}-s{seed if seed is not None else run.get('seed', 1)}"


def _queues(root: Path, runs_dir: Path, now: float) -> list[dict]:
    out = []
    for qf in sorted((root / "experiments").glob("queue_*.txt"), key=lambda p: p.stat().st_mtime):
        log = qf.with_suffix(".log")
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        finished = "fila concluída" in log_text
        title, jobs = None, []
        for raw in qf.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("#"):
                title = title or line.lstrip("# ")
                continue
            if not line:
                continue
            kind, *rest = line.split()
            job = {"line": line, "kind": kind}
            if kind == "train":
                seed = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else None
                sets = [x for x in rest[1:] if "=" in x]
                job["run"] = _run_dir_for(root, rest[0], seed, sets)
                d = runs_dir / job["run"]
                if (d / "summary.json").exists():
                    job["status"] = "feito"
                elif (d / "metrics.jsonl").exists() and now - (d / "metrics.jsonl").stat().st_mtime < RUNNING_WINDOW_S:
                    job["status"] = "rodando"
                else:
                    job["status"] = "na fila"
            elif kind == "evaluate":
                ck = Path(rest[0])
                job["run"] = ck.parent.name
                job["status"] = "feito" if (root / ck.parent / f"eval-{ck.stem}.json").exists() else "na fila"
            elif kind == "continual":
                try:
                    cname = tomllib.loads((root / rest[0]).read_text(encoding="utf-8")).get("run", {}).get("name", "continual")
                except OSError:
                    cname = "continual"
                job["run"] = f"{cname}-{rest[1]}-s{rest[2]}"
                d = runs_dir / job["run"]
                if (d / "summary.json").exists():
                    job["status"] = "feito"
                elif (d / "continual.log").exists() and now - (d / "continual.log").stat().st_mtime < 900:
                    job["status"] = "rodando"
                else:
                    job["status"] = "na fila"
            else:
                name = _run_dir_for(root, rest[0], None, []) + "-pilot"
                job["run"] = name
                job["status"] = "feito" if (runs_dir / name / "metrics.jsonl").exists() else "na fila"
            jobs.append(job)
        done = sum(j["status"] == "feito" for j in jobs)
        if finished and done == len(jobs):  # com dois executores, o primeiro a sair também escreve "concluída"
            state = "concluída"
        elif paused_since(runs_dir) and done < len(jobs) and not finished:
            state = "pausada"
        elif any(j["status"] == "rodando" for j in jobs) or (log.exists() and now - log.stat().st_mtime < 3600
                                                            and done < len(jobs)):
            state = "rodando"
        else:
            state = "aguardando" if done < len(jobs) else "concluída"
        tail = log_text.strip().splitlines()[-4:] if log_text else []
        out.append({"name": qf.stem.removeprefix("queue_"), "title": title, "state": state,
                    "done": done, "total": len(jobs), "jobs": jobs, "log_tail": tail})
    return out


def _sections(md: str) -> list[dict]:
    parts = re.split(r"^## ", md, flags=re.M)
    intro = parts[0].split("\n", 1)[1].strip() if "\n" in parts[0] else ""
    out = [{"title": "Introdução", "body": intro}] if intro else []
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        out.append({"title": title.strip(), "body": body.strip()})
    return out


def _data(root: Path) -> list[dict]:
    out = []
    for m in sorted((root / "data" / "generated").glob("*/manifest.json")):
        d = _json(m)
        if d:
            out.append({"name": m.parent.name, "version": d.get("data_version"),
                        "generator": d.get("generator_version"), "vocab": d.get("vocab_size"),
                        "config": d.get("config"), "splits": d.get("splits")})
    return out


def _memory(runs_dir: Path) -> list[dict]:
    out = []
    for p in sorted(runs_dir.glob("*/memory_benchmark*/memory_benchmark.json")) + \
            sorted(runs_dir.glob("_mb_*/memory_benchmark.json")):
        d = _json(p)
        if d:
            sizes = {s: round((p.parent / f"{s}.sqlite").stat().st_size / 1024 / max(1, d.get("max_episodes", 1)), 2)
                 for s in d.get("splits", {}) if (p.parent / f"{s}.sqlite").exists()}
        out.append({"path": str(p.relative_to(runs_dir.parent)).replace("\\", "/"), "sqlite_kb_per_story": sizes, **d})
    return out


def _continual(runs_dir: Path, now: float) -> dict:
    """Fase 4: resultados por estratégia (média entre seeds) e execuções em andamento."""
    by: dict[str, list[dict]] = {}
    running = []
    cfgs = {}  # run.name -> configuração, para saber blocos e passos de quem está rodando
    for c in sorted((runs_dir.parent / "configs").glob("continual*.toml")):
        try:
            t = tomllib.loads(c.read_text(encoding="utf-8"))
            cfgs.setdefault(t.get("run", {}).get("name", "continual"), t)
        except (OSError, ValueError):
            pass
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or "pilot" in d.name or d.name.startswith("_"):
            continue
        s = _json(d / "summary.json")
        if s and "R" in s and "strategy" in s:
            by.setdefault((s.get("model_type", "fastweight_gru"), s["strategy"]), []).append(s)
        elif (d / "continual.log").exists() and now - (d / "continual.log").stat().st_mtime < 900:
            running.append(_continual_live(d, cfgs.get(d.name.split("-")[0]), now))
    out = []
    for (model_type, strategy), rs in by.items():
        blocks = rs[0]["blocks"]
        out.append({
            "strategy": strategy, "model_type": model_type,
            "seeds": sorted(r["seed"] for r in rs), "blocks": blocks, "sleep_steps": rs[0].get("sleep_steps", 0),
            "steps_per_block": rs[0]["steps_per_block"], "batch_size": rs[0]["batch_size"],
            "replay_per_batch": rs[0]["replay_per_batch"], "buffer_size": rs[0]["buffer_size"],
            "final_avg_acc": _mean_std([r["final_avg_acc"] for r in rs]),
            "avg_forgetting": _mean_std([r["avg_forgetting"] for r in rs]),
            "forgetting": {b: _mean_std([r["forgetting"].get(b) for r in rs]) for b in blocks[:-1]},
            "just_learned": {b: _mean_std([r["just_learned_acc"][b] for r in rs]) for b in blocks},
            "R": [[round(sum(r["R"][i][b] for r in rs) / len(rs), 4) for b in blocks] for i in range(len(blocks))],
            "elapsed_s": _mean_std([r["elapsed_s"] for r in rs]),
            "peak_vram_mb": _mean_std([r.get("peak_vram_mb") for r in rs]),
        })
    order = ["none", "uniform", "category", "ewc", "episodic", "episodic_ewc", "episodic_diverse", "uniform_sleep"]
    out.sort(key=lambda g: (g["model_type"] == "transformer",
                            order.index(g["strategy"]) if g["strategy"] in order else 99))
    return {"strategies": out, "running": running}


def _gpu() -> dict | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        name, util, used, total, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "util": int(util), "mem_used": int(used), "mem_total": int(total), "temp": int(temp)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def collect_state(root: Path) -> dict:
    root = Path(root)
    now = time.time()
    runs_dir = root / "runs"
    runs = [r for d in sorted(runs_dir.iterdir()) if d.is_dir() and not d.name.startswith("_")
            if (r := _run(d, now))]
    docs = root / "docs"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs, "groups": (groups := _groups(runs)), "iteration_groups": _iterations(groups), "scaling": _scaling(groups), "queues": _queues(root, runs_dir, now),
        "iterations": _sections((docs / "ITERACOES.md").read_text(encoding="utf-8"))
        if (docs / "ITERACOES.md").exists() else [],
        "roadmap": (docs / "ROADMAP.md").read_text(encoding="utf-8") if (docs / "ROADMAP.md").exists() else "",
        "data": _data(root), "memory": _memory(runs_dir), "gpu": _gpu(),
        "continual": _continual(runs_dir, now),
        "paused_since": paused_since(runs_dir),
    }
