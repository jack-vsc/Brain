"""Avaliação de um checkpoint: neural, simbólico e piso majoritário em colunas separadas."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from brain.data.dataset import QADataset
from brain.data.generator import load_manifest, load_split
from brain.data.tokenizer import UNK
from brain.evaluation.metrics import compute_metrics, majority_baseline
from brain.evaluation.predict import load_model, predict
from brain.models.registry import count_parameters
from brain.symbolic.solver import ParseError, solve_episode
from brain.training.device import MB, process_ram_mb, select_device

def default_splits(manifest: dict) -> tuple[str, ...]:
    return tuple(s for s in manifest["splits"] if s != "train")


def _meta(episodes: list[dict]) -> list[dict]:
    return [{"task": q["task"], "tags": q["tags"], "answer": q["answer"]}
            for ep in episodes for q in ep["questions"]]


def _symbolic_preds(episodes: list[dict]) -> list[str]:
    out = []
    for ep in episodes:
        for q in ep["questions"]:
            try:
                out.append(solve_episode(ep["story"], q["text"]))
            except ParseError:
                out.append("PARSE_ERROR")
    return out


@torch.no_grad()
def measure_latency(model, dataset: QADataset, device: str, n: int) -> dict:
    """Latência por pergunta com batch 1 (inclui transferência para o dispositivo)."""
    model.eval()
    times = []
    n = min(n, len(dataset))
    for i in range(n + 5):
        ids, _, _ = dataset[i % len(dataset)]
        t0 = time.perf_counter()
        tokens = torch.tensor([ids], device=device)
        lengths = torch.tensor([len(ids)])
        model(tokens, lengths).argmax(-1).item()
        if i >= 5:  # aquecimento descartado
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"n": len(times), "p50_ms": round(statistics.median(times), 3),
            "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3)}


def _train_sha(data_dir: Path) -> str | None:
    try:
        return load_manifest(data_dir)["splits"]["train"]["sha256"]
    except (FileNotFoundError, KeyError):
        return None


def run_evaluation(checkpoint: Path, data_dir: Path | None = None,
                   splits: tuple[str, ...] | None = None, device_pref: str = "auto",
                   latency_samples: int = 200, out_dir: Path | None = None) -> dict:
    checkpoint = Path(checkpoint)
    choice = select_device(device_pref, min_free_vram_mb=200)
    device = choice.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, vocab, answers, ckpt = load_model(checkpoint, device)
    cfg = ckpt["config"]
    trained_on = Path(cfg["run"]["data_dir"])
    data_dir = Path(data_dir or trained_on)
    manifest = load_manifest(data_dir)
    splits = splits or default_splits(manifest)
    qpos = cfg["model"]["question_position"]
    train_meta = _meta(load_split(data_dir, "train"))

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt["state"]["step"],
        "checkpoint_mb": round(checkpoint.stat().st_size / MB, 2),
        "n_params": count_parameters(model),
        "model": cfg["model"],
        "seed": cfg["run"]["seed"],
        "data_dir": str(data_dir),
        "data_version": manifest["data_version"],
        "data_version_matches_checkpoint": manifest["data_version"] == ckpt["data_version"],
        # Avaliar em outro diretório só é justo se o split de treino for idêntico.
        "train_split_matches_checkpoint": _train_sha(data_dir) == _train_sha(trained_on),
        "device": choice.to_dict(),
        "splits": {},
    }
    unk_id = vocab.stoi[UNK]
    for split in splits:
        eps = load_split(data_dir, split)
        ds = QADataset(eps, vocab, answers, qpos)
        extras: dict = {}
        preds, loss = predict(model, ds, device, extras=extras)
        neural = compute_metrics(ds.meta, [answers.labels[p] for p in preds])
        neural["loss"] = round(loss, 4)
        if "ponder" in extras:  # passos de leitura esperados, por tarefa
            by_task: dict[str, list[float]] = {}
            for m, v in zip(ds.meta, extras["ponder"]):
                by_task.setdefault(m["task"], []).append(v)
            neural["ponder_mean"] = round(statistics.mean(extras["ponder"]), 3)
            neural["ponder_by_task"] = {t: round(statistics.mean(v), 3) for t, v in sorted(by_task.items())}
        symbolic = compute_metrics(ds.meta, _symbolic_preds(eps))
        report["splits"][split] = {
            "examples": len(ds),
            "story_lines_mean": manifest["splits"][split].get("story_lines_mean"),
            "max_tokens": ds.max_length,
            "examples_with_unk_tokens": sum(unk_id in ids for ids, _ in ds.items),
            "neural": neural,
            "symbolic_reference": symbolic,
            "majority_baseline": majority_baseline(train_meta, ds.meta),
        }
        if split == "test_iid" or (split == splits[-1] and "latency" not in report):
            report["latency"] = measure_latency(model, ds, device, latency_samples)
    report["resources"] = {
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / MB, 1) if device == "cuda" else None,
        "ram_mb": round(process_ram_mb(), 1),
    }
    out_dir = Path(out_dir or checkpoint.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{checkpoint.stem}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _f(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def render_markdown(r: dict) -> str:
    lines = [
        f"# Avaliação de `{r['checkpoint']}`",
        "",
        f"- passo {r['checkpoint_step']}, {r['n_params']:,} parâmetros, seed {r['seed']}, "
        f"checkpoint {r['checkpoint_mb']} MB",
        f"- dados `{r['data_dir']}` v{r['data_version']} "
        f"(versão confere: {r['data_version_matches_checkpoint']}; "
        f"split de treino idêntico ao do checkpoint: {r['train_split_matches_checkpoint']})",
        f"- dispositivo: {r['device']['device']} ({r['device']['reason']})",
        f"- latência batch 1: p50 {r['latency']['p50_ms']} ms, p95 {r['latency']['p95_ms']} ms",
        f"- pico VRAM na avaliação: {r['resources']['peak_vram_mb']} MB; RAM do processo: "
        f"{r['resources']['ram_mb']} MB",
        "",
        "Colunas *simbólico* e *majoritária* são referências, não respostas da rede.",
        "",
        "| split | neural acc | neural macro | UNKNOWN inventado | temporal | correção | isolamento | par inédito | simbólico acc | majoritária acc |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, d in r["splits"].items():
        n, sy, mj = d["neural"], d["symbolic_reference"], d["majority_baseline"]
        lines.append(
            f"| {s} | {_f(n['acc'])} | {_f(n['macro_task_acc'])} | {_f(n['unknown_false_answer_rate'])} "
            f"| {_f(n['temporal_acc'])} | {_f(n['correction_acc'])} | {_f(n['isolation_acc'])} "
            f"| {_f(n['heldout_pair_acc'])} | {_f(sy['acc'])} | {_f(mj['acc'])} |")
    for s, d in r["splits"].items():
        ponder = d["neural"].get("ponder_by_task")
        extra_h = " passos de leitura (média) |" if ponder else ""
        lines += ["", f"## {s}: por tarefa", "", f"| tarefa | n | neural | simbólico |{extra_h}",
                  "|---|---|---|---|" + ("---|" if ponder else "")]
        for t, v in d["neural"]["per_task"].items():
            extra = f" {ponder[t]:.2f} |" if ponder else ""
            lines.append(f"| {t} | {v['n']} | {_f(v['acc'])} | "
                         f"{_f(d['symbolic_reference']['per_task'][t]['acc'])} |{extra}")
        lines += ["", "| tag | n | neural |", "|---|---|---|"]
        for t, v in d["neural"]["per_tag"].items():
            lines.append(f"| {t} | {v['n']} | {_f(v['acc'])} |")
    return "\n".join(lines) + "\n"
