"""Pausa dos treinos pelo painel: um arquivo-sinal em runs/_control/pause.

Com o sinal presente, o treino em andamento salva last.pt no passo atual e sai com PAUSED_EXIT
(sem summary.json, então a execução continua "por terminar"); scripts/run_queue.py espera o
sinal sumir e retoma do mesmo passo. Avaliações e benchmarks contínuos não são interrompidos:
a fila pausa antes da tarefa seguinte.
"""

from __future__ import annotations

import time
from pathlib import Path

PAUSED_EXIT = 75


def pause_flag(runs_dir: Path | str = "runs") -> Path:
    return Path(runs_dir) / "_control" / "pause"


def pause(runs_dir: Path | str = "runs") -> None:
    f = pause_flag(runs_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def resume(runs_dir: Path | str = "runs") -> None:
    pause_flag(runs_dir).unlink(missing_ok=True)


def paused_since(runs_dir: Path | str = "runs") -> str | None:
    f = pause_flag(runs_dir)
    try:
        return f.read_text(encoding="utf-8").strip() or "?"
    except OSError:
        return None


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "pause":
        pause()
    elif cmd == "resume":
        resume()
    since = paused_since()
    print(f"pausado desde {since}" if since else "rodando (sem pausa)")
