"""Inspeção de hardware e escolha explícita de dispositivo (GPU ou CPU)."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass

import psutil
import torch

MB = 1024 * 1024


@dataclass
class DeviceChoice:
    device: str
    reason: str
    free_vram_mb: float | None = None
    total_vram_mb: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cuda_memory_mb() -> tuple[float, float] | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / MB, total / MB


def select_device(preference: str = "auto", min_free_vram_mb: float = 400) -> DeviceChoice:
    """Escolhe o dispositivo e sempre registra o motivo (fallback para CPU nunca é silencioso)."""
    if preference == "cpu":
        return DeviceChoice("cpu", "CPU pedida na configuração")
    mem = cuda_memory_mb()
    if mem is None:
        if preference == "cuda":
            raise RuntimeError("device='cuda' pedido, mas CUDA não está disponível")
        return DeviceChoice("cpu", "CUDA indisponível; usando CPU")
    free, total = mem
    if preference == "cuda" or free >= min_free_vram_mb:
        return DeviceChoice("cuda", f"GPU com {free:.0f} MB livres de {total:.0f} MB", free, total)
    return DeviceChoice(
        "cpu",
        f"VRAM livre ({free:.0f} MB) abaixo do mínimo configurado ({min_free_vram_mb:.0f} MB); usando CPU",
        free, total)


def hardware_report() -> dict:
    vm = psutil.virtual_memory()
    rep = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "torch_threads": torch.get_num_threads(),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = cuda_memory_mb()
        rep.update({
            "gpu": props.name,
            "gpu_capability": f"{props.major}.{props.minor}",
            "vram_total_mb": round(total),
            "vram_free_mb_now": round(free),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "cudnn": torch.backends.cudnn.version(),
        })
    return rep


def process_ram_mb() -> float:
    return psutil.Process().memory_info().rss / MB
