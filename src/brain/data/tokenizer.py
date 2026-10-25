"""Tokenizer determinístico do micromundo.

Minúsculas, palavras (com hífen interno) e pontuação como tokens. O
vocabulário é construído apenas a partir do split de treino; qualquer token
fora dele vira <unk>. Nenhum embedding ou tokenizer pré-treinado é usado.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)

PAD, UNK, SEP, Q, BOS, EOS = "<pad>", "<unk>", "<sep>", "<q>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, SEP, Q, BOS, EOS]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def episode_texts(ep: dict) -> Iterable[str]:
    yield from ep["story"]
    for q in ep["questions"]:
        yield q["text"]


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        assert self.itos[: len(SPECIALS)] == SPECIALS

    @classmethod
    def build(cls, texts: Iterable[str]) -> "Vocab":
        words = set()
        for t in texts:
            words.update(tokenize(t))
        return cls(SPECIALS + sorted(words - set(SPECIALS)))

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def special_ids(self) -> dict[str, int]:
        return {"pad": self.stoi[PAD], "unk": self.stoi[UNK], "sep": self.stoi[SEP],
                "q": self.stoi[Q], "bos": self.stoi[BOS], "eos": self.stoi[EOS]}

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def unknown_tokens(self, text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in self.stoi]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def encode_example(story: list[str], question: str, vocab: Vocab,
                   question_position: str = "after") -> list[int]:
    """Sequência de entrada: <bos> história (linhas separadas por <sep>) <q> pergunta <eos>.

    Com question_position="before", a pergunta vem antes da história, o que
    permite à rede recorrente saber o que acompanhar enquanto lê.
    """
    story_toks: list[str] = []
    for line in story:
        story_toks += tokenize(line) + [SEP]
    q_toks = [Q] + tokenize(question)
    if question_position == "after":
        toks = [BOS] + story_toks + q_toks + [EOS]
    elif question_position == "before":
        toks = [BOS] + q_toks + [SEP] + story_toks + [EOS]
    else:
        raise ValueError(f"question_position inválido: {question_position}")
    return vocab.encode_tokens(toks)
