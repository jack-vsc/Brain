"""Memória persistente: log de eventos só de acréscimo + índice derivado, em SQLite.

Regras:
- `events` é só de acréscimo: nenhuma linha é alterada. Uma correção é um
  evento novo com `supersedes_id` apontando para o corrigido; o status atual
  ("superseded") é derivado na view `event_status`.
- Campos mínimos: event_id, episode_id, project_id, entity_ids, relation,
  value, event_time (ordem do evento na história), recorded_at (relógio de
  registro), source_id, status, supersedes_id, schema_version. Guardamos
  também `modality` (fact/negated/hypothetical/correction) e o texto original.
- Status: observed, user_confirmed, inferred, superseded, retracted. Inferência
  nunca é gravada como fato confirmado. Negações e hipóteses ("Imagine que…")
  ficam no log como registro do que foi dito, mas não entram no índice de fatos.
- Expurgo explícito (`purge_entity`): remove eventos e fatos derivados que
  citam a entidade e registra em `purges` apenas que houve expurgo (não o conteúdo).
- A recuperação é determinística, por projeto e entidade (com expansão por
  entidades citadas juntas). Ela devolve as FRASES ORIGINAIS como evidência para
  a rede, nunca uma resposta pronta.

O parsing usa a gramática do micromundo (brain.symbolic.solver.parse_line) só
para estruturar o evento; nenhuma resposta é calculada aqui.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from brain.symbolic.solver import parse_line

SCHEMA_VERSION = 1
STATUSES = ("observed", "user_confirmed", "inferred", "superseded", "retracted")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    entity_ids TEXT NOT NULL,          -- JSON: lista de entidades citadas
    relation TEXT NOT NULL,            -- move, put, take, carry, give, drop
    value TEXT,                        -- lugar ou pessoa de destino
    modality TEXT NOT NULL,            -- fact, negated, hypothetical, correction
    event_time INTEGER NOT NULL,       -- ordem do evento na história
    recorded_at TEXT NOT NULL,         -- quando foi registrado
    source_id TEXT NOT NULL,           -- de onde veio (sessão, arquivo...)
    text TEXT NOT NULL,                -- frase original
    status TEXT NOT NULL,
    supersedes_id TEXT REFERENCES events(event_id),
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_project ON events(project_id, episode_id, event_time);
CREATE TABLE IF NOT EXISTS event_entities (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    project_id TEXT NOT NULL,
    entity_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_by_name ON event_entities(project_id, entity_id);
CREATE TABLE IF NOT EXISTS facts (      -- índice derivado (reconstruível a partir de events)
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    episode_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT,
    value TEXT,
    event_time INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS purges (
    purge_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    entity_hash TEXT NOT NULL,         -- não guardamos o nome expurgado
    removed_events INTEGER NOT NULL,
    purged_at TEXT NOT NULL,
    reason TEXT
);
CREATE VIEW IF NOT EXISTS event_status AS
    SELECT e.*, CASE WHEN EXISTS (SELECT 1 FROM events c WHERE c.supersedes_id = e.event_id)
                     THEN 'superseded' ELSE e.status END AS current_status
    FROM events e;
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------- escrita ----------
    def _next_time(self, episode_id: str) -> int:
        row = self.db.execute("SELECT COALESCE(MAX(event_time), -1) FROM events WHERE episode_id = ?",
                              (episode_id,)).fetchone()
        return row[0] + 1

    def ingest(self, text: str, episode_id: str, source_id: str,
               default_project: str = "alfa", status: str = "observed") -> str:
        """Registra uma frase do micromundo. Retorna o event_id."""
        if status not in STATUSES:
            raise ValueError(f"status inválido: {status}")
        p = parse_line(text, default_project)
        if p.is_question:
            raise ValueError("perguntas não são gravadas como eventos")
        f = p.fact
        entities = [e for e in (f.person, f.obj, f.target) if e]
        supersedes = None
        if p.mode == "correction":
            supersedes = self._find_corrected(p.project, episode_id, f)
            if supersedes is None:
                raise ValueError("correção sem evento anterior correspondente")
        event_id = uuid.uuid4().hex
        value = f.place or f.target
        self.db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, episode_id, p.project, json.dumps(entities, ensure_ascii=False),
             f.kind.lower(), value, p.mode, self._next_time(episode_id), _now(), source_id,
             text, status, supersedes, SCHEMA_VERSION))
        self.db.executemany("INSERT INTO event_entities VALUES (?,?,?)",
                            [(event_id, p.project, e) for e in entities])
        if p.mode in ("fact", "correction"):
            self.db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?)",
                            (event_id, episode_id, p.project, f.person, f.kind.lower(), f.obj,
                             value, self._next_time(episode_id) - 1))
        self.db.commit()
        return event_id

    def _find_corrected(self, project: str, episode_id: str, f) -> str | None:
        rows = self.db.execute(
            "SELECT event_id, entity_ids FROM event_status WHERE project_id = ? AND episode_id = ? "
            "AND relation = ? AND modality IN ('fact','correction') AND current_status != 'superseded' "
            "ORDER BY event_time DESC", (project, episode_id, f.kind.lower())).fetchall()
        for event_id, ents in rows:
            ents = json.loads(ents)
            if ents[0] == f.person and (f.obj is None or f.obj in ents):
                return event_id
        return None

    def purge_entity(self, project: str, entity: str, reason: str = "") -> int:
        """Expurgo explícito: remove eventos que citam a entidade, seus derivados e correções ligadas."""
        ids = [r[0] for r in self.db.execute(
            "SELECT DISTINCT event_id FROM event_entities WHERE project_id = ? AND entity_id = ?",
            (project, entity))]
        # correções que apontam para eventos removidos também saem (não ficam órfãs)
        frontier = list(ids)
        while frontier:
            q = ",".join("?" * len(frontier))
            frontier = [r[0] for r in self.db.execute(
                f"SELECT event_id FROM events WHERE supersedes_id IN ({q})", frontier) if r[0] not in ids]
            ids += frontier
        if ids:
            q = ",".join("?" * len(ids))
            self.db.execute(f"DELETE FROM facts WHERE event_id IN ({q})", ids)
            self.db.execute(f"DELETE FROM event_entities WHERE event_id IN ({q})", ids)
            self.db.execute(f"DELETE FROM events WHERE event_id IN ({q})", ids)
        import hashlib
        self.db.execute("INSERT INTO purges VALUES (?,?,?,?,?,?)",
                        (uuid.uuid4().hex, project, hashlib.sha256(entity.encode()).hexdigest()[:16],
                         len(ids), _now(), reason))
        self.db.commit()
        return len(ids)

    # ---------- leitura ----------
    def episode_lines(self, episode_id: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT text FROM events WHERE episode_id = ? ORDER BY event_time", (episode_id,))]

    def retrieve(self, project: str, entities: list[str], episode_id: str | None = None,
                 hops: int = 2) -> list[str]:
        """Frases originais do projeto que citam as entidades, expandindo por entidades
        citadas junto (até `hops` saltos), em ordem de evento. Determinístico."""
        where = "e.project_id = ?" + (" AND e.episode_id = ?" if episode_id else "")
        base = [project] + ([episode_id] if episode_id else [])
        found = {x.lower() for x in entities}
        for _ in range(hops):
            q = ",".join("?" * len(found))
            rows = self.db.execute(
                f"SELECT DISTINCT e.entity_ids FROM events e JOIN event_entities x ON x.event_id = e.event_id "
                f"WHERE {where} AND x.entity_id IN ({q})", base + sorted(found)).fetchall()
            new = {ent.lower() for (ents,) in rows for ent in json.loads(ents)}
            if new <= found:
                break
            found |= new
        q = ",".join("?" * len(found))
        rows = self.db.execute(
            f"SELECT DISTINCT e.event_id, e.text, e.episode_id, e.event_time FROM events e "
            f"JOIN event_entities x ON x.event_id = e.event_id "
            f"WHERE {where} AND x.entity_id IN ({q}) ORDER BY e.episode_id, e.event_time",
            base + sorted(found)).fetchall()
        return [r[1] for r in rows]

    def knowledge_graph(self, project: str | None = None) -> dict:
        """Grafo de conhecimento derivado dos fatos: nós = entidades/lugares, arestas = eventos."""
        sql = ("SELECT f.event_id, f.project_id, f.subject, f.relation, f.object, f.value, f.event_time, "
               "s.current_status, s.text FROM facts f JOIN event_status s ON s.event_id = f.event_id")
        rows = self.db.execute(sql + (" WHERE f.project_id = ?" if project else "") + " ORDER BY f.event_time",
                               (project,) if project else ()).fetchall()
        nodes, edges = {}, []
        for ev, proj, subj, rel, obj, val, t, status, text in rows:
            for name, kind in ((subj, "pessoa"), (obj, "objeto"), (val, "lugar_ou_pessoa")):
                if name:
                    nodes.setdefault(f"{proj}:{name}", {"id": f"{proj}:{name}", "label": name,
                                                        "project": proj, "kind": kind})
            target = obj or val
            edges.append({"event_id": ev, "project": proj, "source": f"{proj}:{subj}",
                          "target": f"{proj}:{target}", "relation": rel, "value": val,
                          "event_time": t, "status": status, "evidence": text})
        return {"schema_version": SCHEMA_VERSION, "nodes": list(nodes.values()), "edges": edges}
