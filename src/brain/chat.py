"""Chat do micromundo: a rede responde e o solucionador simbólico aparece separado, como referência."""

from __future__ import annotations

from pathlib import Path

import torch

from brain.data.tokenizer import encode_example
from brain.evaluation.predict import load_model
from brain.symbolic.solver import ParseError, SymbolicSolver, parse_line
from brain.training.device import select_device

HELP = """Digite frases do micromundo (ex.: "Ana deixou a chave na oficina.") e perguntas
terminadas em "?" (ex.: "onde está a chave agora?"). Sem prefixo, vale o projeto alfa.
Comandos: /historia, /reset, /ajuda, /sair.
A história vive só nesta sessão; /reset apaga."""


def _with_project(text: str, project: str) -> str:
    return text if text.lower().startswith("projeto ") else f"Projeto {project}: {text}"


def run_chat(checkpoint: Path, device_pref: str = "auto") -> None:
    device = select_device(device_pref, min_free_vram_mb=200).device
    model, vocab, answers, ckpt = load_model(Path(checkpoint), device)
    qpos = ckpt["config"]["model"]["question_position"]
    story: list[str] = []
    solver = SymbolicSolver()
    print(f"Checkpoint {checkpoint} (passo {ckpt['state']['step']}, dispositivo {device}).")
    print(HELP)
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/sair", "/exit"):
            break
        if text == "/ajuda":
            print(HELP)
            continue
        if text == "/reset":
            story, solver = [], SymbolicSolver()
            print("história apagada.")
            continue
        if text == "/historia":
            print("\n".join(story) or "(vazia)")
            continue
        line = _with_project(text, "alfa")
        unknown = vocab.unknown_tokens(line)
        if unknown:
            print(f"aviso: tokens fora do vocabulário de treino viram <unk>: {unknown}")
        if not line.endswith("?"):
            try:
                solver.observe(line)
            except ParseError as err:
                print(f"aviso: o solucionador simbólico não entendeu a frase ({err}).")
            story.append(line)
            print("anotado.")
            continue
        ids = encode_example(story, line, vocab, qpos)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor([ids], device=device),
                                        torch.tensor([len(ids)])).float(), -1)[0]
        top = probs.topk(3)
        alts = ", ".join(f"{answers.labels[i]} {p:.2f}" for p, i in zip(top.values.tolist(),
                                                                           top.indices.tolist()))
        print(f"  neural:      {answers.labels[top.indices[0]]}   (top-3: {alts})")
        try:
            parse_line(line)
            print(f"  simbólico:   {solver.ask(line)}   (referência, não é a rede)")
        except ParseError as err:
            print(f"  simbólico:   —   (pergunta fora da gramática: {err})")
