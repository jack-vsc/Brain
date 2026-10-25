"""Checkpoints neurais: escrita atômica e carregamento sem execução arbitrária.

O checkpoint guarda apenas o estado neural e o que é preciso para retomar o
treino (otimizador, scheduler, scaler, RNGs, passo, configuração, versão dos
dados, vocabulário e métricas). Memória factual não vive aqui.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch


def capture_rng_state() -> dict:
    version, state, gauss = random.getstate()
    rng = {
        "python": [version, list(state), gauss],
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng_state(rng: dict) -> None:
    version, state, gauss = rng["python"]
    random.setstate((version, tuple(state), gauss))
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])


def save_checkpoint(path: Path, payload: dict) -> int:
    """Salva em arquivo temporário e renomeia; nunca deixa um checkpoint pela metade."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    # weights_only=True: o unpickler só aceita tensores e tipos primitivos.
    return torch.load(Path(path), map_location=map_location, weights_only=True)
