"""Semântica do micromundo e concordância entre gerador e solucionador simbólico."""

import pytest

from brain.data.generator import GeneratorConfig, choose_heldout_pairs, generate_split
from brain.symbolic.solver import ParseError, SymbolicSolver, solve_episode


def story(*lines):
    return [f"Projeto alfa: {line}" for line in lines]


def ask(lines, question, project="alfa"):
    return solve_episode(lines, f"Projeto {project}: {question}")


def test_exemplo_da_especificacao():
    s = story("Ana deixou a chave na oficina.", "Ana levou a chave para casa.")
    assert ask(s, "onde está a chave agora?") == "casa"
    assert ask(s, "onde a chave estava primeiro?") == "oficina"
    assert ask(s, "onde a chave estava antes?") == "oficina"
    assert ask(s, "quem está com a chave?") == "ana"


def test_negacao_e_hipotese_nao_alteram_fatos():
    s = story("Ana deixou a chave na oficina.",
              "Ana não levou a chave para a praia.",
              "Imagine que Bruno pegou a chave.")
    assert ask(s, "onde está a chave agora?") == "oficina"
    assert ask(s, "quem está com a chave?") == "ninguém"
    assert ask(s, "onde está Bruno agora?") == "UNKNOWN"


def test_correcao_reescreve_o_passado():
    s = story("Ana deixou a chave na oficina.",
              "Bruno foi para a escola.",
              "Correção: Ana deixou a chave no parque.")
    assert ask(s, "onde está a chave agora?") == "parque"
    assert ask(s, "onde a chave estava primeiro?") == "parque"
    assert ask(s, "a chave está na oficina?") == "não"


def test_desconhecido_e_isolamento_entre_projetos():
    s = ["Projeto alfa: Ana deixou a chave na oficina.",
         "Projeto beta: Bruno deixou o livro na escola."]
    assert ask(s, "onde está o livro agora?", "alfa") == "UNKNOWN"
    assert ask(s, "onde está a chave agora?", "beta") == "UNKNOWN"
    assert ask(s, "onde está a caneca agora?", "alfa") == "UNKNOWN"
    assert ask(s, "onde está o livro agora?", "beta") == "escola"


def test_portador_move_o_objeto_e_entrega():
    s = story("Ana pegou a chave.", "Ana foi para a garagem.", "Ana deu a chave para Bruno.",
              "Bruno foi para a praia.", "Bruno largou a chave.")
    assert ask(s, "onde está a chave agora?") == "praia"
    assert ask(s, "onde a chave estava primeiro?") == "garagem"
    assert ask(s, "onde a chave estava antes?") == "garagem"
    assert ask(s, "onde está Ana agora?") == "garagem"
    assert ask(s, "quem está com a chave?") == "ninguém"


def test_local_desconhecido_com_portador_sem_local():
    s = story("Ana pegou a chave.")
    assert ask(s, "onde está a chave agora?") == "UNKNOWN"
    assert ask(s, "quem está com a chave?") == "ana"
    assert ask(s, "a chave está em casa?") == "UNKNOWN"


def test_frases_fora_da_gramatica_sao_rejeitadas():
    solver = SymbolicSolver()
    with pytest.raises(ParseError):
        solver.observe("Projeto alfa: Ana comprou a chave.")
    with pytest.raises(ParseError):
        solver.ask("Projeto alfa: por que a chave sumiu?")


@pytest.mark.parametrize("split", ["train", "test_comp"])
def test_solucionador_concorda_com_o_gerador(split):
    cfg = GeneratorConfig(seed=5)
    held = choose_heldout_pairs(cfg.seed, cfg.heldout_pair_fraction)
    episodes, _ = generate_split(split, 400, cfg, held, set())
    for ep in episodes:
        for q in ep["questions"]:
            assert solve_episode(ep["story"], q["text"]) == q["answer"], (ep["story"], q)
