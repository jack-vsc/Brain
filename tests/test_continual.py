"""Fase 4: benchmark contínuo em miniatura (CPU)."""

import json
import random

import pytest

from brain.training.continual import ReplayBuffer, generate_continual_data, run_continual
from brain.training.trainer import load_config

BLOCKS = ["localizacao", "posse", "correcoes"]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("continual")
    generate_continual_data(d, seed=7, n_train=40, n_test=12, blocks=BLOCKS)
    return d


def cfg_for(data_dir, out, strategy):
    return load_config(None, {
        "run": {"name": "t", "seed": 1, "output_dir": str(out), "data_dir": str(data_dir)},
        "model": {"type": "fastweight_gru", "emb_dim": 8, "hidden": 16, "layers": 1, "dropout": 0.0,
                  "mem_heads": 2, "mem_dim": 4, "read_hops": 2},
        "train": {"batch_size": 8, "warmup_steps": 2, "device": "cpu"},
        "continual": {"strategy": strategy, "blocks": BLOCKS, "steps_per_block": 3, "buffer_size": 20,
                      "replay_fraction": 0.25, "ewc_lambda": 10.0, "fisher_batches": 2,
                      "sleep_steps": 1},
    })


def test_blocos_tem_as_perguntas_certas(data_dir):
    m = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(m["blocks"]["posse"]["train"]["tasks"]) == {"holder"}
    assert "holder" not in m["blocks"]["localizacao"]["train"]["tasks"]
    assert m["blocks"]["correcoes"]["train"]["tags"].get("correction", 0) > 0


@pytest.mark.parametrize("strategy", ["none", "category", "ewc", "episodic_ewc", "episodic_diverse", "uniform_sleep"])
def test_matriz_de_acuracia_e_esquecimento(data_dir, tmp_path, strategy):
    run_dir = run_continual(cfg_for(data_dir, tmp_path, strategy))
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(s["R"]) == 3 and all(set(r) == set(BLOCKS) for r in s["R"])
    assert set(s["forgetting"]) == {"localizacao", "posse"}
    if strategy.startswith("episodic"):
        mismatch, checked = s["episodic_label_mismatch"]
        assert checked > 0 and mismatch == 0  # o alvo relido da memória bate com a resposta original


def test_buffer_por_categoria_divide_cotas():
    class DS:
        items = [([1], 0)] * 30 + [([2], 1)] * 10
        meta = [{"task": "a"}] * 30 + [{"task": "b"}] * 10
    buf = ReplayBuffer(10, "category", random.Random(0))
    buf.add_block(DS())
    tasks = [t for _, _, t in buf.items]
    assert tasks.count("a") == 5 and tasks.count("b") == 5
