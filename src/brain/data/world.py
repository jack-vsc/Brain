"""Léxico, eventos e semântica de estado do micromundo.

Esta é a semântica usada pelo gerador para rotular perguntas. O solucionador
simbólico (brain.symbolic.solver) reimplementa a mesma semântica a partir do
texto, de forma independente; os testes verificam que os dois concordam.

Regras de estado (por projeto; projetos nunca compartilham estado):
  MOVE(p, L)      p vai para L.
  PUT(p, o, L)    o fica em L, sem portador; p está em L.
  TAKE(p, o)      p passa a portar o; se o local de o era conhecido, p está lá.
  CARRY(p, o, L)  p porta o e vai para L.
  GIVE(p, o, q)   q passa a portar o; se o local de p é conhecido, q está lá.
  DROP(p, o)      o fica sem portador, no local de p (talvez desconhecido).
Local de um objeto portado = local do portador. Negações ("não ...") e
hipóteses ("Imagine que ...") não alteram o estado. Uma correção substitui o
local do evento mais recente do mesmo tipo, pessoa e objeto; o estado é então
recalculado do início.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

UNKNOWN = "UNKNOWN"
YES = "sim"
NO = "não"
NOBODY = "ninguém"

PEOPLE = [
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Fábio",
    "Gabi", "Heitor", "Iara", "João", "Laura", "Marcos",
]

# objeto -> artigo definido
OBJECTS = {
    "chave": "a", "livro": "o", "celular": "o", "mochila": "a",
    "caneca": "a", "caderno": "o", "bola": "a", "guarda-chuva": "o",
    "lanterna": "a", "carteira": "a", "martelo": "o", "garrafa": "a",
}

# lugar -> (forma locativa, forma direcional)
PLACES = {
    "casa": ("em casa", "para casa"),
    "oficina": ("na oficina", "para a oficina"),
    "escola": ("na escola", "para a escola"),
    "mercado": ("no mercado", "para o mercado"),
    "parque": ("no parque", "para o parque"),
    "escritório": ("no escritório", "para o escritório"),
    "cozinha": ("na cozinha", "para a cozinha"),
    "garagem": ("na garagem", "para a garagem"),
    "biblioteca": ("na biblioteca", "para a biblioteca"),
    "praia": ("na praia", "para a praia"),
    "jardim": ("no jardim", "para o jardim"),
    "padaria": ("na padaria", "para a padaria"),
}

PROJECTS = ["alfa", "beta"]

# Rótulos de resposta fixos do micromundo (a ordem é a das classes).
ANSWER_LABELS = (
    [p.lower() for p in PEOPLE] + list(PLACES) + [YES, NO, NOBODY, UNKNOWN]
)

MOVE, PUT, TAKE, CARRY, GIVE, DROP = "MOVE", "PUT", "TAKE", "CARRY", "GIVE", "DROP"
STATE_KINDS = (MOVE, PUT, TAKE, CARRY, GIVE, DROP)
CORRECTABLE = (MOVE, PUT, CARRY)

# Modos de um evento no texto.
FACT, NEGATED, HYPOTHETICAL, CORRECTION = "fact", "negated", "hypothetical", "correction"

# Tipos de pergunta (= tarefas reportadas separadamente).
LOC_OBJ, LOC_PERSON, HOLDER, FIRST, BEFORE, YESNO = (
    "loc_obj", "loc_person", "holder", "first", "before", "yesno",
)
TASKS = (LOC_OBJ, LOC_PERSON, HOLDER, FIRST, BEFORE, YESNO)


@dataclass(frozen=True)
class Event:
    project: str
    kind: str
    person: str
    obj: str | None = None
    place: str | None = None
    target: str | None = None  # destinatário em GIVE
    mode: str = FACT


@dataclass(frozen=True)
class Question:
    project: str
    task: str
    subject: str  # objeto ou pessoa perguntado
    place: str | None = None  # lugar perguntado em YESNO


def render_event(e: Event) -> str:
    art = OBJECTS[e.obj] if e.obj else ""
    neg = "não " if e.mode == NEGATED else ""
    if e.kind == MOVE:
        body = f"{e.person} {neg}foi {PLACES[e.place][1]}"
    elif e.kind == PUT:
        body = f"{e.person} {neg}deixou {art} {e.obj} {PLACES[e.place][0]}"
    elif e.kind == TAKE:
        body = f"{e.person} {neg}pegou {art} {e.obj}"
    elif e.kind == CARRY:
        body = f"{e.person} {neg}levou {art} {e.obj} {PLACES[e.place][1]}"
    elif e.kind == GIVE:
        body = f"{e.person} {neg}deu {art} {e.obj} para {e.target}"
    elif e.kind == DROP:
        body = f"{e.person} {neg}largou {art} {e.obj}"
    else:
        raise ValueError(f"tipo de evento desconhecido: {e.kind}")
    if e.mode == HYPOTHETICAL:
        body = f"Imagine que {body}"
    elif e.mode == CORRECTION:
        body = f"Correção: {body}"
    return f"Projeto {e.project}: {body}."


def render_question(q: Question) -> str:
    prefix = f"Projeto {q.project}:"
    if q.task == LOC_PERSON:
        return f"{prefix} onde está {q.subject} agora?"
    art = OBJECTS[q.subject]
    if q.task == LOC_OBJ:
        return f"{prefix} onde está {art} {q.subject} agora?"
    if q.task == HOLDER:
        return f"{prefix} quem está com {art} {q.subject}?"
    if q.task == FIRST:
        return f"{prefix} onde {art} {q.subject} estava primeiro?"
    if q.task == BEFORE:
        return f"{prefix} onde {art} {q.subject} estava antes?"
    if q.task == YESNO:
        return f"{prefix} {art} {q.subject} está {PLACES[q.place][0]}?"
    raise ValueError(f"tarefa desconhecida: {q.task}")


def effective_events(events: list[Event], project: str) -> list[Event]:
    """Eventos factuais do projeto, com correções aplicadas, em ordem."""
    out: list[Event] = []
    for e in events:
        if e.project != project or e.mode in (NEGATED, HYPOTHETICAL):
            continue
        if e.mode == CORRECTION:
            for i in range(len(out) - 1, -1, -1):
                prev = out[i]
                if (prev.kind, prev.person, prev.obj) == (e.kind, e.person, e.obj):
                    out[i] = replace(prev, place=e.place)
                    break
            else:
                raise ValueError(f"correção sem evento anterior correspondente: {e}")
            continue
        out.append(e)
    return out


def find_correctable(events: list[Event], project: str) -> list[Event]:
    """Eventos efetivos que uma correção poderia alterar (o mais recente de cada chave)."""
    latest: dict[tuple, Event] = {}
    for e in effective_events(events, project):
        if e.kind in CORRECTABLE:
            latest[(e.kind, e.person, e.obj)] = e
    return list(latest.values())


@dataclass
class WorldState:
    person_loc: dict[str, str] = field(default_factory=dict)
    holder: dict[str, str | None] = field(default_factory=dict)
    obj_loc: dict[str, str | None] = field(default_factory=dict)
    history: dict[str, list[str]] = field(default_factory=dict)
    people: set[str] = field(default_factory=set)

    def loc_of(self, obj: str) -> str | None:
        h = self.holder.get(obj)
        if h is not None:
            return self.person_loc.get(h)
        return self.obj_loc.get(obj)

    def apply(self, e: Event) -> None:
        self.people.add(e.person)
        if e.kind == MOVE:
            self.person_loc[e.person] = e.place
        elif e.kind == PUT:
            self.holder[e.obj] = None
            self.obj_loc[e.obj] = e.place
            self.person_loc[e.person] = e.place
        elif e.kind == TAKE:
            prev = self.loc_of(e.obj)
            self.holder[e.obj] = e.person
            self.obj_loc[e.obj] = None
            if prev is not None:
                self.person_loc[e.person] = prev
        elif e.kind == CARRY:
            self.holder[e.obj] = e.person
            self.obj_loc[e.obj] = None
            self.person_loc[e.person] = e.place
        elif e.kind == GIVE:
            self.people.add(e.target)
            self.holder[e.obj] = e.target
            if e.person in self.person_loc:
                self.person_loc[e.target] = self.person_loc[e.person]
        elif e.kind == DROP:
            self.holder[e.obj] = None
            self.obj_loc[e.obj] = self.person_loc.get(e.person)
        else:
            raise ValueError(f"tipo de evento desconhecido: {e.kind}")
        for o in self.holder:
            loc = self.loc_of(o)
            hist = self.history.setdefault(o, [])
            if loc is not None and (not hist or hist[-1] != loc):
                hist.append(loc)


def simulate(events: list[Event], project: str) -> WorldState:
    state = WorldState()
    for e in effective_events(events, project):
        state.apply(e)
    return state


def answer(events: list[Event], q: Question) -> str:
    s = simulate(events, q.project)
    if q.task == LOC_PERSON:
        return s.person_loc.get(q.subject, UNKNOWN)
    if q.subject not in s.holder:  # objeto nunca mencionado factualmente
        return UNKNOWN
    if q.task == LOC_OBJ:
        return s.loc_of(q.subject) or UNKNOWN
    if q.task == HOLDER:
        h = s.holder[q.subject]
        return h.lower() if h else NOBODY
    hist = s.history.get(q.subject, [])
    if q.task == FIRST:
        return hist[0] if hist else UNKNOWN
    if q.task == BEFORE:
        if s.loc_of(q.subject) is None or len(hist) < 2:
            return UNKNOWN
        return hist[-2]
    if q.task == YESNO:
        loc = s.loc_of(q.subject)
        if loc is None:
            return UNKNOWN
        return YES if loc == q.place else NO
    raise ValueError(f"tarefa desconhecida: {q.task}")
