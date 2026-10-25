"""Fila de experimentos resistente a falhas.

Cada linha do arquivo de fila é um job:
    train <config> [seed] [seção.chave=valor ...]
                                    treina; se o processo cair (ex.: erro de CUDA),
                                    retoma de last.pt até --retries vezes. Os pares
                                    chave=valor viram --set (ex.: run.name="x" train.lr=5e-4)
    pilot <config> [passos]         piloto para estimar custo
    evaluate <checkpoint> [data_dir]
    continual <config> <estratégia> <seed>
                                    benchmark contínuo (Fase 4); sem retomada
Linhas vazias e começando com # são ignoradas. Cada job roda num processo
novo (um erro de CUDA corrompe o contexto do processo; só um processo novo
recupera). Tudo é registrado em <fila>.log.

Uso: python scripts/run_queue.py experiments/queue_2b.txt [--retries 3]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import psutil

from brain.training.control import PAUSED_EXIT, pause_flag, paused_since

PY =[sys.executable, "-m", "brain.cli"]


def run_dir_for(config: str, seed: int | None, sets: list[str]) -> Path:
    cfg = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    run = dict(cfg.get("run", {}))
    for item in sets:
        key, _, raw = item.partition("=")
        if key.startswith("run."):
            run[key[4:]] = tomllib.loads(f"v = {raw}")["v"]
    s = seed if seed is not None else run.get("seed", 1)
    return Path(run.get("output_dir", "runs")) / f"{run.get('name', 'gru_baseline')}-s{s}"


LOCKS = Path("runs") / "_locks"


def claim(run_name: str) -> bool:
    """Trava por execução: duas filas rodando em paralelo nunca pegam o mesmo job.
    Uma trava cujo processo já morreu é descartada."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = LOCKS / f"{run_name}.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text())
        except (OSError, ValueError):
            pid = -1
        if pid != os.getpid() and psutil.pid_exists(pid):
            return False
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def release(run_name: str) -> None:
    (LOCKS / f"{run_name}.lock").unlink(missing_ok=True)


def wait_if_paused(note) -> None:
    if not pause_flag().exists():
        return
    note(f"fila pausada pelo painel (desde {paused_since()}); esperando o comando de continuar")
    while pause_flag().exists():
        time.sleep(10)
    note("pausa encerrada; continuando de onde parou")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    log = open(args.queue.with_suffix(".log"), "a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    failures = 0
    for raw in args.queue.read_text(encoding="utf-8").splitlines():
        parts = raw.split("#", 1)[0].split()
        if not parts:
            continue
        kind, rest = parts[0], parts[1:]
        wait_if_paused(note)  # pausa pelo painel: nenhuma tarefa nova começa
        if kind == "train":
            config = rest[0]
            seed = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else None
            sets = [x for x in rest[1:] if "=" in x]
            run_dir = run_dir_for(config, seed, sets)
            if (run_dir / "summary.json").exists():
                note(f"pulando {run_dir}: já concluído")
                continue
            if not claim(run_dir.name):
                note(f"pulando {run_dir}: outra fila está rodando este job")
                continue
            base = PY + ["train", "--config", config] + (["--seed", str(seed)] if seed is not None else [])
            for item in sets:
                base += ["--set", item]
            try:
                attempt = 0
                while attempt <= args.retries:
                    cmd = base + (["--resume", str(run_dir / "last.pt")] if (run_dir / "last.pt").exists() else [])
                    note(f"train (tentativa {attempt + 1}): {' '.join(cmd[2:])}")
                    rc = subprocess.call(cmd)
                    if rc == 0:
                        break
                    if rc == PAUSED_EXIT and not (run_dir / "summary.json").exists():
                        note(f"treino pausado pelo painel; estado salvo em {run_dir / 'last.pt'}")
                        wait_if_paused(note)
                        continue  # retoma do mesmo passo, sem gastar tentativa
                    attempt += 1
                    if (run_dir / "summary.json").exists():
                        # O treino terminou e gravou tudo; só a saída do processo falhou.
                        note(f"processo terminou com código {rc} DEPOIS de concluir (summary.json gravado)")
                        break
                    note(f"processo terminou com código {rc}")
                if attempt > args.retries:
                    note(f"FALHOU após {args.retries + 1} tentativas: {config}")
                    failures += 1
            finally:
                release(run_dir.name)
        elif kind == "continual":
            config, strategy, seed = rest[0], rest[1], int(rest[2])
            name = tomllib.loads(Path(config).read_text(encoding="utf-8")).get("run", {}).get("name", "continual")
            run_dir = Path("runs") / f"{name}-{strategy}-s{seed}"
            if (run_dir / "summary.json").exists():
                note(f"pulando {run_dir}: já concluído")
                continue
            if not claim(run_dir.name):
                note(f"pulando {run_dir}: outra fila está rodando este job")
                continue
            cmd = PY + ["continual-benchmark", "--config", config, "--strategy", strategy, "--seed", str(seed)]
            try:
                for attempt in range(args.retries + 1):  # sem retomada: uma tentativa nova recomeça do bloco 1
                    note(f"continual (tentativa {attempt + 1}): {' '.join(cmd[2:])}")
                    rc = subprocess.call(cmd)
                    if rc == 0 or (run_dir / "summary.json").exists():
                        break
                    note(f"processo terminou com código {rc}")
                else:
                    note(f"FALHOU após {args.retries + 1} tentativas: {config} {strategy} {seed}")
                    failures += 1
            finally:
                release(run_dir.name)
        elif kind == "pilot":
            steps = rest[1] if len(rest) > 1 else "300"
            note(f"pilot: {rest[0]} ({steps} passos)")
            failures += subprocess.call(PY + ["train", "--config", rest[0], "--pilot-steps", steps]) != 0
        elif kind == "evaluate":
            ck = Path(rest[0])
            report = ck.parent / f"eval-{ck.stem}.json"
            if report.exists() and ck.exists() and report.stat().st_mtime >= ck.stat().st_mtime:
                note(f"pulando avaliação de {ck}: já feita")
                continue
            if not claim(ck.parent.name):
                # outra fila ainda treina esta execução; ela mesma avalia ao terminar (a linha seguinte dela)
                note(f"pulando avaliação de {ck}: outra fila está com esta execução")
                continue
            release(ck.parent.name)
            cmd = PY + ["evaluate", "--checkpoint", rest[0]] + (["--data-dir", rest[1]] if len(rest) > 1 else [])
            note(f"evaluate: {' '.join(cmd[2:])}")
            for attempt in range(2):
                if subprocess.call(cmd) == 0:
                    break
            else:
                failures += 1
        else:
            note(f"job desconhecido ignorado: {raw}")
    note(f"fila concluída com {failures} falha(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
