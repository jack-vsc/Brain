"""Construção de modelos a partir da configuração. Todos começam com pesos aleatórios."""

from __future__ import annotations

from torch import nn

from brain.models.episodic import EpisodicGRU
from brain.models.fastweight import FastWeightGRU
from brain.models.gru import GRUClassifier
from brain.models.ssm import SSMClassifier
from brain.models.transformer import TransformerClassifier

_COMMON = {"type", "question_position"}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, vocab_size: int, num_classes: int, pad_id: int,
                special_ids: dict[str, int] | None = None) -> nn.Module:
    """special_ids: ids de tokens especiais do vocabulário (ex.: {"sep": 2}) para modelos que os usam."""
    kind = cfg.get("type", "gru")
    kwargs = {k: v for k, v in cfg.items() if k not in _COMMON}
    if kind == "gru":
        return GRUClassifier(vocab_size, num_classes, pad_id, **kwargs)
    if kind == "transformer":
        return TransformerClassifier(vocab_size, num_classes, pad_id, **kwargs)
    if kind == "mamba2":
        return SSMClassifier(vocab_size, num_classes, pad_id, **kwargs)
    if kind == "fastweight_gru":
        q_id = (special_ids or {}).get("q")
        sep_id = (special_ids or {}).get("sep") if kwargs.get("time_code") else None
        return FastWeightGRU(vocab_size, num_classes, pad_id, q_id=q_id, sep_id=sep_id, **kwargs)
    if kind == "episodic_gru":
        if not special_ids or "sep" not in special_ids:
            raise ValueError("episodic_gru precisa de special_ids['sep']")
        if cfg.get("question_position", "after") != "after":
            raise ValueError("episodic_gru supõe a pergunta no último segmento (question_position='after')")
        return EpisodicGRU(vocab_size, num_classes, pad_id, special_ids["sep"], **kwargs)
    raise ValueError(f"tipo de modelo desconhecido: {kind}")
