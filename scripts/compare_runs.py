"""Tabela comparativa entre execuções (média ± desvio entre seeds).

Agrupa runs/<nome>-s<seed>/eval-best.json por <nome> e escreve uma tabela
Markdown com as métricas principais por split e por tarefa. Só lê números
já medidos; nada é estimado.

Uso: python scripts/compare_runs.py b_gru b_transformer b_fastweight --out experiments/x.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

METRICS = [("acc", "acc"), ("macro_task_acc", "macro"), ("unknown_false_answer_rate", "UNK inventado↓"),
           ("temporal_acc", "temporal"), ("correction_acc", "correção"), ("isolation_acc", "isolamento"),
           ("heldout_pair_acc", "par inédito")]


def load_group(runs: Path, name: str) -> list[dict]:
    out = []
    for d in sorted(runs.glob(f"{name}-s*")):
        if re.fullmatch(rf"{re.escape(name)}-s\d+", d.name) and (d / "eval-best.json").exists():
            r = json.loads((d / "eval-best.json").read_text(encoding="utf-8"))
            s = d / "summary.json"
            r["_summary"] = json.loads(s.read_text(encoding="utf-8")) if s.exists() else {}
            out.append(r)
    return out


def fmt(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    groups = {n: load_group(args.runs, n) for n in args.names}
    lines = []
    lines += ["| modelo | seeds | parâmetros | passos | exemplos vistos | melhor passo | "
              "tempo treino (s) | s/passo | VRAM pico (MB) | latência p50 (ms) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for n, g in groups.items():
        if not g:
            lines.append(f"| {n} | 0 |" + " — |" * 8)
            continue
        s = [r["_summary"] for r in g]
        lines.append(f"| {n} | {len(g)} | {g[0]['n_params']:,} | "
                     f"{fmt([x.get('steps') for x in s])} | {fmt([x.get('samples_seen') for x in s])} | "
                     f"{fmt([r['checkpoint_step'] for r in g])} | "
                     f"{fmt([x.get('elapsed_s') for x in s])} | {fmt([x.get('sec_per_step') for x in s])} | "
                     f"{fmt([x.get('peak_vram_mb') for x in s])} | "
                     f"{fmt([r['latency']['p50_ms'] for r in g])} |")
    splits = []
    for g in groups.values():
        for r in g:
            splits += [s for s in r["splits"] if s not in splits]
    for split in splits:
        lines += ["", f"### {split}", "",
                  "| modelo | " + " | ".join(lbl for _, lbl in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for n, g in groups.items():
            rs = [r["splits"][split]["neural"] for r in g if split in r["splits"]]
            if rs:
                lines.append(f"| {n} | " + " | ".join(fmt([x[k] for x in rs]) for k, _ in METRICS) + " |")
        tasks = sorted({t for g in groups.values() for r in g if split in r["splits"]
                        for t in r["splits"][split]["neural"]["per_task"]})
        lines += ["", "| modelo | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
        for n, g in groups.items():
            rs = [r["splits"][split]["neural"]["per_task"] for r in g if split in r["splits"]]
            if rs:
                lines.append(f"| {n} | " + " | ".join(fmt([x[t]["acc"] for x in rs]) for t in tasks) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
