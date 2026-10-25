"""Solucionador simbólico do micromundo, a partir do texto.

Faz parsing das frases com uma gramática fixa e reimplementa a semântica de
estado sem reutilizar brain.data.world.simulate, para servir de verificação
cruzada do gerador. É uma referência de correção: nunca alimenta a rede neural
e suas respostas são identificadas como simbólicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brain.data.tokenizer import tokenize
from brain.data.world import NO, NOBODY, OBJECTS, PEOPLE, PLACES, UNKNOWN, YES

SOURCE = "simbólico"

_PEOPLE = {p.lower() for p in PEOPLE}
_OBJECTS = set(OBJECTS)
_PLACES = set(PLACES)
_VERBS = {"foi": "MOVE", "deixou": "PUT", "pegou": "TAKE", "levou": "CARRY",
          "deu": "GIVE", "largou": "DROP"}


class ParseError(ValueError):
    pass


@dataclass
class Fact:
    kind: str
    person: str
    obj: str | None
    place: str | None
    target: str | None


@dataclass
class ParsedLine:
    project: str
    is_question: bool
    mode: str = "fact"  # fact | negated | hypothetical | correction
    fact: Fact | None = None
    task: str | None = None
    subject: str | None = None
    place: str | None = None


def _place(args: list[str], directional: bool) -> tuple[str, list[str]]:
    if directional:
        if not args or args[0] != "para":
            raise ParseError(f"esperado 'para': {args}")
        args = args[1:]
        if args and args[0] in ("a", "o"):
            args = args[1:]
    else:
        if not args or args[0] not in ("em", "na", "no"):
            raise ParseError(f"esperado locativo: {args}")
        args = args[1:]
    if not args or args[0] not in _PLACES:
        raise ParseError(f"lugar desconhecido: {args}")
    return args[0], args[1:]


def _obj(args: list[str]) -> tuple[str, list[str]]:
    if len(args) < 2 or args[0] not in ("a", "o") or args[1] not in _OBJECTS:
        raise ParseError(f"objeto esperado: {args}")
    return args[1], args[2:]


def parse_line(text: str, default_project: str = "alfa") -> ParsedLine:
    toks = tokenize(text)
    if not toks:
        raise ParseError("linha vazia")
    is_q = toks[-1] == "?"
    if toks[-1] in (".", "?"):
        toks = toks[:-1]
    project = default_project
    if toks[:1] == ["projeto"]:
        if len(toks) < 3 or toks[2] != ":":
            raise ParseError("prefixo de projeto malformado")
        project, toks = toks[1], toks[3:]
    if is_q:
        return _parse_question(toks, project)

    mode = "fact"
    if toks[:2] == ["imagine", "que"]:
        mode, toks = "hypothetical", toks[2:]
    elif toks[:2] == ["correção", ":"]:
        mode, toks = "correction", toks[2:]
    if not toks or toks[0] not in _PEOPLE:
        raise ParseError(f"pessoa desconhecida: {toks[:1]}")
    person, rest = toks[0], toks[1:]
    if rest[:1] == ["não"]:
        if mode != "fact":
            raise ParseError("negação combinada com outro modo não é suportada")
        mode, rest = "negated", rest[1:]
    if not rest or rest[0] not in _VERBS:
        raise ParseError(f"verbo desconhecido: {rest[:1]}")
    kind, args = _VERBS[rest[0]], rest[1:]
    obj = place = target = None
    if kind == "MOVE":
        place, args = _place(args, directional=True)
    else:
        obj, args = _obj(args)
        if kind == "PUT":
            place, args = _place(args, directional=False)
        elif kind == "CARRY":
            place, args = _place(args, directional=True)
        elif kind == "GIVE":
            if len(args) < 2 or args[0] != "para" or args[1] not in _PEOPLE:
                raise ParseError(f"destinatário esperado: {args}")
            target, args = args[1], args[2:]
    if args:
        raise ParseError(f"tokens sobrando: {args}")
    if mode == "correction" and kind not in ("MOVE", "PUT", "CARRY"):
        raise ParseError("correção só é suportada para foi/deixou/levou")
    return ParsedLine(project, False, mode, Fact(kind, person, obj, place, target))


def _parse_question(t: list[str], project: str) -> ParsedLine:
    def q(task, subject, place=None):
        return ParsedLine(project, True, task=task, subject=subject, place=place)

    if len(t) == 4 and t[:2] == ["onde", "está"] and t[3] == "agora" and t[2] in _PEOPLE:
        return q("loc_person", t[2])
    if len(t) == 5 and t[:2] == ["onde", "está"] and t[4] == "agora" and t[3] in _OBJECTS:
        return q("loc_obj", t[3])
    if len(t) == 5 and t[:3] == ["quem", "está", "com"] and t[4] in _OBJECTS:
        return q("holder", t[4])
    if len(t) == 5 and t[0] == "onde" and t[2] in _OBJECTS and t[3] == "estava":
        if t[4] == "primeiro":
            return q("first", t[2])
        if t[4] == "antes":
            return q("before", t[2])
    if len(t) >= 4 and t[0] in ("a", "o") and t[1] in _OBJECTS and t[2] == "está":
        place, rest = _place(t[3:], directional=False)
        if not rest:
            return q("yesno", t[1], place)
    raise ParseError(f"pergunta fora da gramática: {' '.join(t)}")


@dataclass
class _Project:
    facts: list[Fact] = field(default_factory=list)


class SymbolicSolver:
    """Mantém a história por projeto e responde perguntas do micromundo."""

    def __init__(self, default_project: str = "alfa"):
        self.default_project = default_project
        self.projects: dict[str, _Project] = {}

    def observe(self, text: str) -> ParsedLine:
        p = parse_line(text, self.default_project)
        if p.is_question:
            raise ParseError("observe() recebeu uma pergunta")
        proj = self.projects.setdefault(p.project, _Project())
        if p.mode == "fact":
            proj.facts.append(p.fact)
        elif p.mode == "correction":
            f = p.fact
            for prev in reversed(proj.facts):
                if (prev.kind, prev.person, prev.obj) == (f.kind, f.person, f.obj):
                    prev.place = f.place
                    break
            else:
                raise ParseError("correção sem evento anterior correspondente")
        return p

    def ask(self, text: str) -> str:
        p = parse_line(text, self.default_project)
        if not p.is_question:
            raise ParseError("ask() recebeu uma afirmação")
        return self._answer(p)

    def _replay(self, project: str):
        where: dict[str, str] = {}  # pessoa -> lugar
        carried_by: dict[str, str | None] = {}  # objeto -> pessoa
        resting: dict[str, str | None] = {}  # objeto -> lugar (quando solto)
        trail: dict[str, list[str]] = {}
        people: set[str] = set()

        def locate(o):
            c = carried_by.get(o)
            return where.get(c) if c else resting.get(o)

        for f in self.projects.get(project, _Project()).facts:
            people.add(f.person)
            if f.kind == "MOVE":
                where[f.person] = f.place
            elif f.kind == "PUT":
                carried_by[f.obj], resting[f.obj] = None, f.place
                where[f.person] = f.place
            elif f.kind == "TAKE":
                before = locate(f.obj)
                carried_by[f.obj], resting[f.obj] = f.person, None
                if before:
                    where[f.person] = before
            elif f.kind == "CARRY":
                carried_by[f.obj], resting[f.obj] = f.person, None
                where[f.person] = f.place
            elif f.kind == "GIVE":
                people.add(f.target)
                carried_by[f.obj] = f.target
                if f.person in where:
                    where[f.target] = where[f.person]
            elif f.kind == "DROP":
                carried_by[f.obj], resting[f.obj] = None, where.get(f.person)
            for o in carried_by:
                now = locate(o)
                tr = trail.setdefault(o, [])
                if now and (not tr or tr[-1] != now):
                    tr.append(now)
        return where, carried_by, locate, trail, people

    def _answer(self, p: ParsedLine) -> str:
        where, carried_by, locate, trail, _ = self._replay(p.project)
        s = p.subject
        if p.task == "loc_person":
            return where.get(s, UNKNOWN)
        if s not in carried_by:
            return UNKNOWN
        now = locate(s)
        if p.task == "loc_obj":
            return now or UNKNOWN
        if p.task == "holder":
            return carried_by[s] or NOBODY
        tr = trail.get(s, [])
        if p.task == "first":
            return tr[0] if tr else UNKNOWN
        if p.task == "before":
            return tr[-2] if now and len(tr) >= 2 else UNKNOWN
        if p.task == "yesno":
            return UNKNOWN if now is None else (YES if now == p.place else NO)
        raise ParseError(f"tarefa desconhecida: {p.task}")


def solve_episode(story: list[str], question: str) -> str:
    solver = SymbolicSolver()
    for line in story:
        solver.observe(line)
    return solver.ask(question)
