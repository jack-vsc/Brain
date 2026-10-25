"""Gerador procedural versionado de episódios do micromundo.

Cada episódio é uma história curta (um ou dois projetos isolados) mais várias
perguntas. Os splits são feitos por episódio: todas as perguntas de um
episódio ficam no mesmo split. Histórias idênticas nunca aparecem em dois
splits. O split test_comp usa pares (objeto, lugar) que nunca aparecem juntos
em train/val/test_iid, nem no texto nem no estado do mundo. O split opcional
test_long tem histórias mais longas que qualquer uma do treino (mesmas regras
e mesmos pares permitidos), isolando a generalização de comprimento.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from brain.data import world
from brain.data.world import (
    BEFORE, CARRY, CORRECTION, DROP, FACT, FIRST, GIVE, HOLDER, HYPOTHETICAL,
    LOC_OBJ, LOC_PERSON, MOVE, NEGATED, OBJECTS, PEOPLE, PLACES, PROJECTS, PUT,
    TAKE, UNKNOWN, YESNO, Event, Question,
)

GENERATOR_VERSION = "1.1.0"
SPLITS = ("train", "val", "test_iid", "test_comp", "test_long")


@dataclass
class GeneratorConfig:
    name: str = "mvp"
    seed: int = 1234
    n_train: int = 20000
    n_val: int = 2000
    n_test_iid: int = 2000
    n_test_comp: int = 2000
    n_test_long: int = 0  # 0 = não gerar
    long_min_events: int = 10  # eventos por projeto no test_long
    long_max_events: int = 16
    questions_per_episode: int = 4
    min_events: int = 3  # eventos por projeto (inclui distratores)
    max_events: int = 8
    two_project_prob: float = 0.4
    shared_object_prob: float = 0.5  # com 2 projetos, mesmo objeto em ambos
    people_per_project: tuple[int, int] = (2, 3)
    objects_per_project: tuple[int, int] = (2, 3)
    places_per_project: tuple[int, int] = (3, 5)
    negation_prob: float = 0.10
    hypothetical_prob: float = 0.08
    correction_prob: float = 0.12
    unknown_question_prob: float = 0.15
    keep_incidental_unknown_prob: float = 0.5
    heldout_pair_fraction: float = 0.15
    event_weights: dict[str, float] = field(default_factory=lambda: {
        MOVE: 1.0, PUT: 1.5, TAKE: 1.0, CARRY: 1.2, GIVE: 0.8, DROP: 0.8,
    })
    question_weights: dict[str, float] = field(default_factory=lambda: {
        LOC_OBJ: 2.0, LOC_PERSON: 1.0, HOLDER: 1.0, FIRST: 0.8, BEFORE: 0.8, YESNO: 1.0,
    })

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratorConfig":
        d = dict(d)
        for k in ("people_per_project", "objects_per_project", "places_per_project"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def split_sizes(self) -> dict[str, int]:
        sizes = {"train": self.n_train, "val": self.n_val,
                 "test_iid": self.n_test_iid, "test_comp": self.n_test_comp}
        if self.n_test_long:
            if self.long_min_events <= self.max_events:
                raise ValueError("long_min_events deve ser maior que max_events do treino")
            sizes["test_long"] = self.n_test_long
        return sizes

    def events_range(self, split: str) -> tuple[int, int]:
        if split == "test_long":
            return self.long_min_events, self.long_max_events
        return self.min_events, self.max_events


def choose_heldout_pairs(seed: int, fraction: float) -> set[tuple[str, str]]:
    """Pares (objeto, lugar) reservados ao teste composicional.

    Cada objeto e cada lugar continuam aparecendo no treino com a maioria dos
    outros parceiros, então o vocabulário do teste é todo conhecido.
    """
    rng = random.Random(f"{seed}:heldout")
    pairs = [(o, p) for o in OBJECTS for p in PLACES]
    rng.shuffle(pairs)
    target = round(len(pairs) * fraction)
    max_per = max(1, round(len(PLACES) * fraction) + 1)
    per_obj: Counter = Counter()
    per_place: Counter = Counter()
    chosen: set[tuple[str, str]] = set()
    for o, p in pairs:
        if len(chosen) >= target:
            break
        if per_obj[o] < max_per and per_place[p] < max_per:
            chosen.add((o, p))
            per_obj[o] += 1
            per_place[p] += 1
    return chosen


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _event_pairs(e: Event) -> set[tuple[str, str]]:
    return {(e.obj, e.place)} if e.obj and e.place else set()


def _state_pairs(events: list[Event], projects: list[str]) -> set[tuple[str, str]]:
    pairs = set()
    for p in projects:
        s = world.simulate(events, p)
        for o, hist in s.history.items():
            pairs.update((o, loc) for loc in hist)
    return pairs


class _Cast:
    def __init__(self, rng: random.Random, cfg: GeneratorConfig, shared_obj: str | None):
        self.people = rng.sample(PEOPLE, rng.randint(*cfg.people_per_project))
        objs = list(OBJECTS)
        k = rng.randint(*cfg.objects_per_project)
        if shared_obj:
            objs.remove(shared_obj)
            self.objects = [shared_obj] + rng.sample(objs, k - 1)
        else:
            self.objects = rng.sample(objs, k)
        self.places = rng.sample(list(PLACES), rng.randint(*cfg.places_per_project))


def _candidate_events(rng, project: str, cast: _Cast, state: world.WorldState,
                      kind: str, prefer_places: list[str]) -> list[Event]:
    """Eventos factuais válidos de um tipo, dado o estado atual do projeto."""
    out: list[Event] = []
    places = prefer_places or cast.places
    for person in cast.people:
        if kind == MOVE:
            for pl in places:
                if state.person_loc.get(person) != pl:
                    out.append(Event(project, MOVE, person, place=pl))
            continue
        for obj in cast.objects:
            h = state.holder.get(obj)
            if kind in (PUT, CARRY) and h in (None, person):
                out.extend(Event(project, kind, person, obj, place=pl) for pl in places)
            elif kind == TAKE and h is None:
                out.append(Event(project, TAKE, person, obj))
            elif kind == GIVE and h == person:
                out.extend(Event(project, GIVE, person, obj, target=t)
                           for t in cast.people if t != person)
            elif kind == DROP and h == person:
                out.append(Event(project, DROP, person, obj))
    return out


def _next_event(rng, cfg, events, project, cast, heldout, comp_mode) -> Event | None:
    state = world.simulate(events, project)
    has_fact = any(e.project == project and e.mode == FACT for e in events)
    prefer: list[str] = []
    if comp_mode and rng.random() < 0.5:
        prefer = [pl for pl in cast.places if any((o, pl) in heldout for o in cast.objects)]

    r = rng.random()
    if has_fact and r < cfg.correction_prob:
        options = world.find_correctable(events, project)
        rng.shuffle(options)
        for old in options:
            new_places = [pl for pl in (prefer or cast.places) if pl != old.place]
            if new_places:
                return Event(project, old.kind, old.person, old.obj,
                             place=rng.choice(new_places), mode=CORRECTION)
    mode = FACT
    r = rng.random()
    if has_fact and r < cfg.negation_prob:
        mode = NEGATED
    elif has_fact and r < cfg.negation_prob + cfg.hypothetical_prob:
        mode = HYPOTHETICAL

    weights = dict(cfg.event_weights)
    for _ in range(len(weights)):
        kind = _weighted(rng, weights)
        cands = _candidate_events(rng, project, cast, state, kind, prefer)
        if not cands and prefer:
            cands = _candidate_events(rng, project, cast, state, kind, [])
        if cands:
            e = rng.choice(cands)
            return Event(e.project, e.kind, e.person, e.obj, e.place, e.target, mode)
        weights[kind] = 0
        if not any(weights.values()):
            break
    return None


def _subject_mentions(events: list[Event], project: str, subject: str, mode: str) -> bool:
    return any(e.project == project and e.mode == mode
               and subject in (e.person, e.obj, e.target) for e in events)


def _factual_subjects(events: list[Event], project: str) -> tuple[set[str], set[str]]:
    s = world.simulate(events, project)
    return set(s.holder), set(s.people)


def _make_questions(rng, cfg, events, projects, heldout, comp_mode) -> list[dict]:
    questions: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(questions) < cfg.questions_per_episode and attempts < 60:
        attempts += 1
        project = rng.choice(projects)
        task = _weighted(rng, cfg.question_weights)
        objs, people = _factual_subjects(events, project)
        intentional_unknown = rng.random() < cfg.unknown_question_prob
        if task == LOC_PERSON:
            pool = sorted(people)
            if intentional_unknown or not pool:
                pool = [p for p in PEOPLE if p not in people]
        else:
            pool = sorted(objs)
            if intentional_unknown or not pool:
                other = set()
                for p2 in projects:
                    if p2 != project:
                        other |= _factual_subjects(events, p2)[0]
                other -= objs
                pool = sorted(other) if other and rng.random() < 0.7 else \
                    [o for o in OBJECTS if o not in objs]
        subject = rng.choice(pool)
        place = None
        if task == YESNO:
            loc = world.simulate(events, project).loc_of(subject)
            if loc and rng.random() < 0.5:
                place = loc
            else:
                place = rng.choice(list(PLACES))
            if not comp_mode and (subject, place) in heldout:
                continue
        q = Question(project, task, subject, place)
        text = world.render_question(q)
        if text in seen:
            continue
        gold = world.answer(events, q)
        if gold == UNKNOWN and not intentional_unknown \
                and rng.random() > cfg.keep_incidental_unknown_prob:
            continue

        mentioned_anywhere = any(subject in (e.person, e.obj, e.target) for e in events)
        in_other = any(subject in _factual_subjects(events, p2)[0] | _factual_subjects(events, p2)[1]
                       for p2 in projects if p2 != project)
        in_this = subject in objs or subject in people
        tags = []
        if len(projects) == 2:
            tags.append("two_projects")
        if in_other and not in_this:
            tags.append("cross_project")
        if not mentioned_anywhere:
            tags.append("unmentioned")
        if any(e.project == project and e.mode == CORRECTION for e in events):
            tags.append("correction")
        if _subject_mentions(events, project, subject, NEGATED):
            tags.append("negation")
        if _subject_mentions(events, project, subject, HYPOTHETICAL):
            tags.append("hypothetical")
        if gold == UNKNOWN:
            tags.append("unknown_expected")
        if task != LOC_PERSON and gold in PLACES and (subject, gold) in heldout:
            tags.append("heldout_pair")
        seen.add(text)
        questions.append({"project": project, "task": task, "subject": subject,
                          "place": place, "text": text, "answer": gold, "tags": tags})
    return questions


def generate_episode(seed_str: str, cfg: GeneratorConfig,
                     heldout: set[tuple[str, str]], comp_mode: bool,
                     events_range: tuple[int, int] | None = None) -> dict | None:
    rng = random.Random(seed_str)
    min_events, max_events = events_range or (cfg.min_events, cfg.max_events)
    if rng.random() < cfg.two_project_prob:
        projects = list(PROJECTS)
        rng.shuffle(projects)
    else:
        projects = [rng.choice(PROJECTS)]
    shared = rng.choice(list(OBJECTS)) if len(projects) == 2 and \
        rng.random() < cfg.shared_object_prob else None
    casts = {p: _Cast(rng, cfg, shared) for p in projects}

    slots = []
    for p in projects:
        slots += [p] * rng.randint(min_events, max_events)
    first = {p: slots.index(p) for p in projects}
    rest = [s for i, s in enumerate(slots) if i not in first.values()]
    rng.shuffle(rest)
    slots = list(projects) + rest  # cada projeto começa com um evento

    events: list[Event] = []
    for p in slots:
        for _ in range(8):
            e = _next_event(rng, cfg, events, p, casts[p], heldout, comp_mode)
            if e is None:
                break
            if not comp_mode:
                if _event_pairs(e) & heldout:
                    continue
                if _state_pairs(events + [e], projects) & heldout:
                    continue
            events.append(e)
            break

    if not any(e.mode == FACT for e in events):
        return None
    questions = _make_questions(rng, cfg, events, projects, heldout, comp_mode)
    if len(questions) < cfg.questions_per_episode:
        return None
    if comp_mode and not any("heldout_pair" in q["tags"] for q in questions):
        return None
    return {
        "seed": seed_str,
        "projects": projects,
        "story": [world.render_event(e) for e in events],
        "events": [asdict(e) for e in events],
        "questions": questions,
    }


def story_hash(ep: dict) -> str:
    return hashlib.sha256("\n".join(ep["story"]).encode("utf-8")).hexdigest()


def generate_split(split: str, n: int, cfg: GeneratorConfig, heldout, seen_hashes: set[str],
                   max_attempts_factor: int = 50) -> tuple[list[dict], dict]:
    comp_mode = split == "test_comp"
    episodes: list[dict] = []
    stats = Counter()
    attempt = 0
    while len(episodes) < n:
        if attempt > n * max_attempts_factor:
            raise RuntimeError(f"split {split}: não foi possível gerar {n} episódios únicos")
        seed_str = f"{cfg.seed}:{split}:{attempt}"
        attempt += 1
        ep = generate_episode(seed_str, cfg, heldout, comp_mode, cfg.events_range(split))
        if ep is None:
            stats["rejected_invalid"] += 1
            continue
        h = story_hash(ep)
        if h in seen_hashes:
            stats["rejected_duplicate_story"] += 1
            continue
        seen_hashes.add(h)
        ep["id"] = f"{split}-{len(episodes):06d}"
        episodes.append(ep)
    stats["attempts"] = attempt
    return episodes, dict(stats)


def summarize(episodes: list[dict]) -> dict:
    tasks, answers, tags = Counter(), Counter(), Counter()
    for ep in episodes:
        for q in ep["questions"]:
            tasks[q["task"]] += 1
            answers[q["answer"]] += 1
            tags.update(q["tags"])
    n_q = sum(tasks.values())
    return {
        "episodes": len(episodes),
        "questions": n_q,
        "tasks": dict(sorted(tasks.items())),
        "tags": dict(sorted(tags.items())),
        "unknown_fraction": round(answers[UNKNOWN] / max(1, n_q), 4),
        "majority_answer": answers.most_common(1)[0] if answers else None,
        "story_lines_mean": round(sum(len(e["story"]) for e in episodes) / max(1, len(episodes)), 2),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset(cfg: GeneratorConfig, out_dir: Path) -> dict:
    """Gera todos os splits, o vocabulário (só do treino) e o manifesto."""
    from brain.data.tokenizer import Vocab, episode_texts

    out_dir.mkdir(parents=True, exist_ok=True)
    heldout = choose_heldout_pairs(cfg.seed, cfg.heldout_pair_fraction)
    seen: set[str] = set()
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "config": asdict(cfg),
        "heldout_pairs": sorted(map(list, heldout)),
        "splits": {},
    }
    for split, n in cfg.split_sizes().items():
        episodes, stats = generate_split(split, n, cfg, heldout, seen)
        path = out_dir / f"{split}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        tmp.replace(path)
        manifest["splits"][split] = {**summarize(episodes), "generation": stats,
                                     "sha256": _sha256_file(path)}
        if split == "train":
            vocab = Vocab.build(t for ep in episodes for t in episode_texts(ep))
            vocab.save(out_dir / "vocab.json")
            manifest["vocab_size"] = len(vocab)
    joined = "".join(s["sha256"] for s in manifest["splits"].values())
    manifest["data_version"] = hashlib.sha256(
        (GENERATOR_VERSION + joined).encode()).hexdigest()[:12]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_split(data_dir: Path, split: str) -> list[dict]:
    with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
