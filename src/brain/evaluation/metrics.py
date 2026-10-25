"""Métricas por tarefa, por tag e de comportamento com UNKNOWN."""

from __future__ import annotations

from collections import Counter, defaultdict

from brain.data.world import BEFORE, FIRST, UNKNOWN


def _acc(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    correct = sum(p == g for p, g in pairs)
    return {"n": n, "acc": round(correct / n, 4) if n else None}


def compute_metrics(meta: list[dict], preds: list[str]) -> dict:
    """meta[i] tem task/tags/answer; preds[i] é o rótulo previsto."""
    assert len(meta) == len(preds)
    by_task: dict[str, list] = defaultdict(list)
    by_tag: dict[str, list] = defaultdict(list)
    allp = []
    for m, p in zip(meta, preds):
        pair = (p, m["answer"])
        allp.append(pair)
        by_task[m["task"]].append(pair)
        for t in m["tags"]:
            by_tag[t].append(pair)

    per_task = {t: _acc(v) for t, v in sorted(by_task.items())}
    gold_unknown = [(p, g) for p, g in allp if g == UNKNOWN]
    gold_known = [(p, g) for p, g in allp if g != UNKNOWN]
    temporal = by_task.get(FIRST, []) + by_task.get(BEFORE, [])
    task_accs = [v["acc"] for v in per_task.values() if v["acc"] is not None]
    return {
        **_acc(allp),
        "macro_task_acc": round(sum(task_accs) / len(task_accs), 4) if task_accs else None,
        "per_task": per_task,
        "per_tag": {t: _acc(v) for t, v in sorted(by_tag.items())},
        # Fração das perguntas com resposta esperada UNKNOWN em que algo foi inventado.
        "unknown_false_answer_rate": round(
            sum(p != UNKNOWN for p, _ in gold_unknown) / len(gold_unknown), 4) if gold_unknown else None,
        # Fração das perguntas respondíveis em que o modelo disse UNKNOWN.
        "unknown_overuse_rate": round(
            sum(p == UNKNOWN for p, _ in gold_known) / len(gold_known), 4) if gold_known else None,
        "temporal_acc": _acc(temporal)["acc"],
        "correction_acc": _acc(by_tag.get("correction", []))["acc"],
        "isolation_acc": _acc(by_tag.get("cross_project", []))["acc"],
        "heldout_pair_acc": _acc(by_tag.get("heldout_pair", []))["acc"],
    }


def majority_baseline(train_meta: list[dict], eval_meta: list[dict]) -> dict:
    """Prever sempre a resposta mais comum do treino: piso de referência."""
    majority = Counter(m["answer"] for m in train_meta).most_common(1)[0][0]
    return {"label": majority,
            **_acc([(majority, m["answer"]) for m in eval_meta])}
