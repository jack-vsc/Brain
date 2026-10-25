"""Inferência em lote e reconstrução do modelo a partir de um checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brain.data.dataset import AnswerSpace, QADataset, make_collate
from brain.data.tokenizer import Vocab
from brain.models.registry import build_model
from brain.training.checkpoint import load_checkpoint


@torch.no_grad()
def predict(model, dataset: QADataset, device: str, batch_size: int = 256,
            amp: bool = False, extras: dict | None = None) -> tuple[list[int], float]:
    """Retorna índices previstos (na ordem do dataset) e a perda média.

    Se `extras` for um dict e o modelo expuser `last_ponder` (passos adaptativos),
    extras["ponder"] recebe o número esperado de passos por exemplo.
    """
    model.eval()
    # Gerador próprio: criar o iterador não consome o RNG global (que o dropout do treino usa).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(model.embedding.padding_idx),
                        generator=torch.Generator())
    preds = [0] * len(dataset)
    ponder = [0.0] * len(dataset)
    has_ponder = False
    loss_sum, n = 0.0, 0
    for tokens, lengths, labels, index in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(tokens, lengths)
        loss_sum += torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum").item()
        n += len(labels)
        for i, p in zip(index.tolist(), logits.argmax(-1).tolist()):
            preds[i] = p
        lp = getattr(model, "last_ponder", None)
        if lp is not None:
            has_ponder = True
            for i, v in zip(index.tolist(), lp.tolist()):
                ponder[i] = v
    if extras is not None and has_ponder:
        extras["ponder"] = ponder
    return preds, loss_sum / max(1, n)


def load_model(checkpoint_path: Path, device: str):
    """Reconstrói modelo, vocabulário e espaço de respostas só a partir do checkpoint."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    vocab = Vocab(ckpt["vocab"])
    answers = AnswerSpace(ckpt["answer_labels"])
    model = build_model(ckpt["config"]["model"], len(vocab), len(answers), vocab.pad_id,
                        vocab.special_ids)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, vocab, answers, ckpt
