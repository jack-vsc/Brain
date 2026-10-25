"""Dataset PyTorch: cada pergunta de cada episódio vira um exemplo de classificação."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from brain.data.tokenizer import Vocab, encode_example
from brain.data.world import ANSWER_LABELS


class AnswerSpace:
    """Classes de saída: entidades, lugares, sim/não/ninguém e UNKNOWN."""

    def __init__(self, labels: list[str] | None = None):
        self.labels = list(labels or ANSWER_LABELS)
        self.index = {a: i for i, a in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)


class QADataset(Dataset):
    def __init__(self, episodes: list[dict], vocab: Vocab, answers: AnswerSpace,
                 question_position: str = "after", max_len: int | None = None):
        self.items: list[tuple[list[int], int]] = []
        self.meta: list[dict] = []
        for ep in episodes:
            for qi, q in enumerate(ep["questions"]):
                ids = encode_example(ep["story"], q["text"], vocab, question_position)
                if max_len and len(ids) > max_len:
                    raise ValueError(f"exemplo com {len(ids)} tokens excede max_len={max_len}")
                self.items.append((ids, answers.index[q["answer"]]))
                self.meta.append({"episode_id": ep["id"], "q_index": qi, "task": q["task"],
                                  "tags": q["tags"], "answer": q["answer"]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ids, y = self.items[i]
        return ids, y, i

    @property
    def max_length(self) -> int:
        return max(len(ids) for ids, _ in self.items)


def make_collate(pad_id: int):
    def collate(batch):
        lengths = torch.tensor([len(ids) for ids, _, _ in batch], dtype=torch.long)
        tokens = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long)
        for row, (ids, _, _) in enumerate(batch):
            tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor([y for _, y, _ in batch], dtype=torch.long)
        index = torch.tensor([i for _, _, i in batch], dtype=torch.long)
        return tokens, lengths, labels, index
    return collate
