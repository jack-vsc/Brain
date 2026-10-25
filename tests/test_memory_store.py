"""Memória persistente em SQLite: log só de acréscimo, correções, expurgo e recuperação."""

import pytest

from brain.memory.store import MemoryStore
from brain.symbolic.solver import solve_episode


def fill(store, lines, episode="ep1"):
    return [store.ingest(line, episode, source_id="sessao-1") for line in lines]


def test_persiste_entre_reinicios(tmp_path):
    db = tmp_path / "mem.sqlite"
    with MemoryStore(db) as s:
        fill(s, ["Projeto alfa: Ana deixou a chave na oficina.", "Projeto alfa: Ana levou a chave para casa."])
    with MemoryStore(db) as s:  # "reinício": nova conexão ao mesmo arquivo
        assert s.episode_lines("ep1") == ["Projeto alfa: Ana deixou a chave na oficina.",
                                          "Projeto alfa: Ana levou a chave para casa."]


def test_correcao_nao_altera_o_log_e_marca_substituido(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s:
        first, _, corr = fill(s, ["Projeto alfa: Ana deixou a chave na oficina.",
                                  "Projeto alfa: Bruno foi para a escola.",
                                  "Projeto alfa: Correção: Ana deixou a chave no parque."])
        rows = dict(s.db.execute("SELECT event_id, current_status FROM event_status").fetchall())
        assert rows[first] == "superseded" and rows[corr] == "observed"
        assert s.db.execute("SELECT supersedes_id FROM events WHERE event_id = ?", (corr,)).fetchone()[0] == first
        assert s.db.execute("SELECT value FROM events WHERE event_id = ?", (first,)).fetchone()[0] == "oficina"


def test_hipotese_e_negacao_nao_viram_fato(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s:
        fill(s, ["Projeto alfa: Imagine que Ana levou a chave para a praia.",
                 "Projeto alfa: Ana não foi para a escola."])
        assert s.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        assert s.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0


def test_campos_minimos_e_ordens_separadas(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s:
        fill(s, ["Projeto beta: Ana deu a chave para Bruno."])
        cols = [r[1] for r in s.db.execute("PRAGMA table_info(events)")]
        for c in ("event_id", "episode_id", "project_id", "entity_ids", "relation", "value",
                  "event_time", "recorded_at", "source_id", "status", "supersedes_id", "schema_version"):
            assert c in cols
        row = s.db.execute("SELECT project_id, entity_ids, value, event_time FROM events").fetchone()
        assert row == ("beta", '["ana", "chave", "bruno"]', "bruno", 0)


def test_recuperacao_por_projeto_e_entidade_com_expansao(tmp_path):
    lines = ["Projeto alfa: Ana pegou a chave.",
             "Projeto alfa: Ana foi para a garagem.",
             "Projeto beta: Ana foi para a praia.",
             "Projeto alfa: Bruno foi para o parque.",
             "Projeto alfa: Carla deixou o livro na escola."]
    with MemoryStore(tmp_path / "m.sqlite") as s:
        fill(s, lines)
        got = s.retrieve("alfa", ["chave"])
        assert got == lines[:2]  # chave → Ana → frases de Ana no alfa; nada do beta nem de Bruno/Carla
        assert solve_episode(got, "Projeto alfa: onde está a chave agora?") == "garagem"
        assert s.retrieve("beta", ["chave"]) == []


def test_expurgo_remove_evento_derivados_e_correcoes(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s:
        fill(s, ["Projeto alfa: Ana deixou a chave na oficina.",
                 "Projeto alfa: Correção: Ana deixou a chave no parque.",
                 "Projeto alfa: Bruno foi para a escola."])
        assert s.purge_entity("alfa", "chave", reason="pedido do usuário") == 2
        assert s.episode_lines("ep1") == ["Projeto alfa: Bruno foi para a escola."]
        assert s.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
        purge = s.db.execute("SELECT removed_events, entity_hash FROM purges").fetchone()
        assert purge[0] == 2 and "chave" not in purge[1]


def test_pergunta_nao_e_gravada(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s, pytest.raises(ValueError):
        s.ingest("Projeto alfa: onde está a chave agora?", "ep1", "sessao-1")


def test_grafo_de_conhecimento_so_tem_fatos_registrados(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite") as s:
        fill(s, ["Projeto alfa: Ana deixou a chave na oficina.",
                 "Projeto alfa: Imagine que Ana levou a chave para a praia."])
        g = s.knowledge_graph()
        assert len(g["edges"]) == 1 and g["edges"][0]["value"] == "oficina"
        assert all("praia" not in n["label"] for n in g["nodes"])
