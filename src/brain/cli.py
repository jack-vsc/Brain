"""Interface de linha de comando: `brain <comando> ...`."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def _parse_overrides(items: list[str]) -> dict:
    """--set train.lr=0.001 --set model.question_position="before" """
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise SystemExit(f"--set espera seção.chave=valor, recebeu {item!r}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw
        out.setdefault(section, {})[name] = value
    return out


def cmd_hardware(args) -> int:
    from brain.training.device import hardware_report
    print(json.dumps(hardware_report(), indent=2, ensure_ascii=False))
    return 0


def cmd_generate(args) -> int:
    from brain.data.generator import GeneratorConfig, generate_dataset
    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    data = raw.get("data", {})
    out = Path(args.output or raw.get("output", {}).get("dir", f"data/generated/{data.get('name', 'mvp')}"))
    manifest = generate_dataset(GeneratorConfig.from_dict(data), out)
    print(f"dados gravados em {out} (versão {manifest['data_version']}, vocab {manifest['vocab_size']})")
    for split, s in manifest["splits"].items():
        print(f"  {split:9s} {s['episodes']:6d} episódios {s['questions']:6d} perguntas "
              f"UNKNOWN {s['unknown_fraction']:.1%}")
    return 0


def cmd_train(args) -> int:
    from brain.training.trainer import load_config, train
    overrides = _parse_overrides(args.set)
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    run_dir = train(cfg, resume=args.resume, pilot_steps=args.pilot_steps)
    print(f"execução em {run_dir}")
    return 0


def cmd_evaluate(args) -> int:
    from brain.evaluation.evaluate import run_evaluation
    splits = tuple(args.splits.split(",")) if args.splits else None
    report = run_evaluation(Path(args.checkpoint), args.data_dir, splits, args.device,
                            args.latency_samples, args.output)
    for split, d in report["splits"].items():
        print(f"{split:9s} neural {d['neural']['acc']:.4f}  simbólico {d['symbolic_reference']['acc']:.4f}"
              f"  majoritária {d['majority_baseline']['acc']:.4f}")
    print(f"latência p50 {report['latency']['p50_ms']} ms, p95 {report['latency']['p95_ms']} ms")
    return 0


def cmd_memory_benchmark(args) -> int:
    from brain.evaluation.memory_benchmark import run_memory_benchmark
    r = run_memory_benchmark(Path(args.checkpoint), tuple(args.splits.split(",")), args.max_episodes,
                             args.device, args.output)
    for split, d in r["splits"].items():
        for cond, m in d.items():
            if "acc" in m:
                print(f"{split:9s} {cond:24s} {m['acc']:.4f}")
    return 0


def cmd_export_graph(args) -> int:
    from brain.memory.store import MemoryStore
    with MemoryStore(Path(args.store)) as store:
        graph = store.knowledge_graph(args.project)
    Path(args.output).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(graph['nodes'])} nós e {len(graph['edges'])} arestas em {args.output}")
    return 0


def cmd_continual(args) -> int:
    from brain.training.continual import generate_continual_data, run_continual
    from brain.training.trainer import load_config
    overrides = _parse_overrides(args.set)
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if args.strategy:
        overrides.setdefault("continual", {})["strategy"] = args.strategy
    cfg = load_config(args.config, overrides)
    c, data_dir = cfg["continual"], Path(cfg["run"]["data_dir"])
    if not (data_dir / "manifest.json").exists():
        generate_continual_data(data_dir, int(c["data_seed"]), int(c["n_train"]), int(c["n_test"]), list(c["blocks"]))
        print(f"dados dos blocos gravados em {data_dir}")
    run_dir = run_continual(cfg)
    print(f"execução em {run_dir}")
    return 0


def cmd_chat(args) -> int:
    from brain.chat import run_chat
    run_chat(Path(args.checkpoint), args.device)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="brain", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("hardware", help="mostra o hardware e o ambiente medidos")
    s.set_defaults(func=cmd_hardware)

    s = sub.add_parser("generate-data", help="gera os splits do micromundo")
    s.add_argument("--config", required=True)
    s.add_argument("--output")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("train", help="treina o baseline neural")
    s.add_argument("--config", required=True)
    s.add_argument("--resume", type=Path)
    s.add_argument("--pilot-steps", type=int, default=0,
                   help="roda N passos e estima o custo do treino completo")
    s.add_argument("--seed", type=int)
    s.add_argument("--set", action="append", metavar="SEÇÃO.CHAVE=VALOR")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="avalia um checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--data-dir", type=Path)
    s.add_argument("--splits", help="lista separada por vírgulas (padrão: todos menos train)")
    s.add_argument("--device", default="auto")
    s.add_argument("--latency-samples", type=int, default=200)
    s.add_argument("--output", type=Path)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("chat", help="conversa no micromundo com um checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("continual-benchmark", help="Fase 4: aprendizado sequencial por blocos, replay e consolidação")
    s.add_argument("--config", required=True)
    s.add_argument("--strategy", choices=["none", "uniform", "category", "ewc", "episodic", "episodic_ewc",
                                          "episodic_diverse", "uniform_sleep"])
    s.add_argument("--seed", type=int)
    s.add_argument("--set", action="append", metavar="SEÇÃO.CHAVE=VALOR")
    s.set_defaults(func=cmd_continual)

    s = sub.add_parser("memory-benchmark", help="H1: pesos lentos, contexto, SQLite e sinapses após reinício")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--splits", default="test_iid,test_long")
    s.add_argument("--max-episodes", type=int, default=500)
    s.add_argument("--device", default="auto")
    s.add_argument("--output", type=Path)
    s.set_defaults(func=cmd_memory_benchmark)

    s = sub.add_parser("export-graph", help="exporta o grafo de conhecimento de uma memória SQLite em JSON")
    s.add_argument("--store", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--project")
    s.set_defaults(func=cmd_export_graph)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
