"""Modelo, checkpoints e treino de ponta a ponta (em CPU, para rodar em qualquer máquina)."""

import json
from pathlib import Path

import pytest
import torch

from brain.data.dataset import AnswerSpace, QADataset, make_collate
from brain.data.generator import load_split
from brain.data.tokenizer import Vocab
from brain.evaluation.predict import load_model, predict
from brain.models.fastweight import FastWeightGRU
from brain.models.gru import GRUClassifier
from brain.models.registry import build_model, count_parameters
from brain.models.ssm import SSMClassifier
from brain.models.transformer import TransformerClassifier
from brain.training.checkpoint import load_checkpoint, save_checkpoint
from brain.training.control import PAUSED_EXIT, pause, resume
from brain.training.trainer import load_config, train


def test_baseline_fica_na_faixa_de_parametros():
    m = build_model(load_config(None)["model"], vocab_size=80, num_classes=30, pad_id=0)
    assert 1_000_000 <= count_parameters(m) <= 10_000_000


TINY_MODELS = {
    "gru": lambda: GRUClassifier(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0),
    "transformer": lambda: TransformerClassifier(20, 5, pad_id=0, d_model=16, layers=2, heads=2,
                                                 ff=32, dropout=0.0),
    "fastweight_gru": lambda: FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2,
                                            dropout=0.0, mem_heads=2, mem_dim=4),
    "fastweight_hops": lambda: FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2,
                                             dropout=0.0, mem_heads=2, mem_dim=4, read_hops=3),
    "fastweight_sep": lambda: FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2,
                                            dropout=0.0, mem_heads=2, mem_dim=4, key_dim=12,
                                            key_topk=3, familiarity=True, read_hops=2),
    "fastweight_act": lambda: FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2,
                                            dropout=0.0, mem_heads=2, mem_dim=4, read_hops=4,
                                            adaptive_hops=True),
    "fastweight_gate": lambda: FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2,
                                             dropout=0.0, mem_heads=2, mem_dim=4, read_hops=3,
                                             forget_gate=True),
    "mamba2": lambda: SSMClassifier(20, 5, pad_id=0, d_model=16, layers=2, d_state=4,
                                    expand=2, head_dim=8, dropout=0.0, chunk=4),
}
CONFIGS = {"gru": "b_gru", "transformer": "b_transformer", "fastweight_gru": "b_fastweight",
           "fastweight_hops": "b_fastweight_hops", "fastweight_sep": "b_fastweight_sep",
           "fastweight_act": "b_adaptive", "fastweight_gate": "b_gated", "mamba2": "b_mamba2"}


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_ssd_em_blocos_igual_ao_laco(chunk):
    from brain.models.ssm import ssd_chunked, ssd_loop
    torch.manual_seed(0)
    x = torch.randn(2, 23, 3, 5, dtype=torch.float64)
    dt = torch.nn.functional.softplus(torch.randn(2, 23, 3, dtype=torch.float64))
    A = -torch.rand(3, dtype=torch.float64) * 4
    Bm, Cm = torch.randn(2, 23, 6, dtype=torch.float64), torch.randn(2, 23, 6, dtype=torch.float64)
    assert torch.allclose(ssd_chunked(x, dt, A, Bm, Cm, chunk), ssd_loop(x, dt, A, Bm, Cm), atol=1e-9)


def test_codigo_esparso_tem_no_maximo_topk_ativos():
    from brain.models.fastweight import sparse_code
    x = sparse_code(torch.randn(5, 3, 64), topk=8)
    assert ((x != 0).sum(-1) <= 8).all()
    assert torch.allclose(x.norm(dim=-1), torch.ones(5, 3), atol=1e-5)


@pytest.mark.parametrize("kind", list(TINY_MODELS))
def test_modelos_na_mesma_faixa_de_parametros(kind):
    cfg = load_config(Path(f"configs/{CONFIGS[kind]}.toml"))["model"]
    n = count_parameters(build_model(cfg, vocab_size=72, num_classes=28, pad_id=0))
    assert 1_350_000 <= n <= 1_650_000, n


def test_memoria_rapida_sobrescreve_com_regra_delta():
    """Com β=1 e chave unitária, escrever (k, v1) e depois (k, v2) deixa só v2."""
    k = torch.nn.functional.normalize(torch.randn(4), dim=0)
    v1, v2 = torch.randn(4), torch.randn(4)
    W = torch.zeros(4, 4)
    for v in (v1, v2):
        W = W + torch.outer(k, v - k @ W)
    assert torch.allclose(k @ W, v2, atol=1e-5)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_regra_delta_em_blocos_igual_ao_laco(chunk, gated):
    from brain.models.fastweight import delta_rule_chunked, delta_rule_loop
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(3, 2, 37, 8, dtype=torch.float64), dim=-1)
    v = torch.randn(3, 2, 37, 8, dtype=torch.float64)
    beta = torch.rand(3, 2, 37, dtype=torch.float64)
    beta[0, :, 30:] = 0  # padding
    la = torch.nn.functional.logsigmoid(torch.randn(3, 2, 37, dtype=torch.float64) + 1) if gated else None
    assert torch.allclose(delta_rule_chunked(k, v, beta, chunk, log_alpha=la),
                          delta_rule_loop(k, v, beta, log_alpha=la), atol=1e-9)


def test_regra_delta_com_porta_estavel_em_sequencia_longa():
    """Esquecimento forte numa sequência longa não pode gerar inf/NaN na forma em blocos."""
    from brain.models.fastweight import delta_rule_chunked
    k = torch.nn.functional.normalize(torch.randn(2, 2, 400, 8), dim=-1)
    W = delta_rule_chunked(k, torch.randn(2, 2, 400, 8), torch.rand(2, 2, 400),
                           log_alpha=torch.full((2, 2, 400), -3.0))
    assert torch.isfinite(W).all()


def test_passos_adaptativos_distribuicao_e_custo():
    torch.manual_seed(0)
    m = FastWeightGRU(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0,
                      mem_heads=2, mem_dim=4, read_hops=4, adaptive_hops=True, ponder_cost=0.1)
    out = m(torch.randint(1, 20, (3, 11)), torch.tensor([11, 8, 5]))
    assert out.shape == (3, 5)
    assert ((m.last_ponder >= 1) & (m.last_ponder <= 4)).all()
    assert m.aux_loss is not None and m.aux_loss.requires_grad
    out.sum().add(m.aux_loss).backward()
    assert m.halt.weight.grad is not None


@pytest.mark.parametrize("kind", ["fastweight_hops", "fastweight_sep", "fastweight_act",
                                  "fastweight_gate"])
def test_modelo_memoria_rapida_blocos_igual_ao_laco(kind):
    torch.manual_seed(0)
    m = TINY_MODELS[kind]().eval()
    tokens = torch.randint(1, 20, (3, 23))
    lengths = torch.tensor([23, 17, 9])
    with torch.no_grad():
        m.scan = "loop"
        a = m(tokens, lengths)
        m.scan = "chunked"
        b = m(tokens, lengths)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("kind", list(TINY_MODELS))
def test_padding_nao_altera_a_saida(kind):
    torch.manual_seed(0)
    m = TINY_MODELS[kind]().eval()
    short = [3, 4, 5]
    long = [6, 7, 8, 9, 10, 11, 12]
    batch = make_collate(0)([(short, 0, 0), (long, 1, 1)])
    alone = make_collate(0)([(short, 0, 0)])
    with torch.no_grad():
        together = m(batch[0], batch[1])[0]
        single = m(alone[0], alone[1])[0]
    assert torch.allclose(together, single, atol=1e-5)


def test_checkpoint_atomico_e_carregamento_seguro(tmp_path):
    p = tmp_path / "c.pt"
    save_checkpoint(p, {"x": torch.ones(2), "meta": {"a": [1, 2]}})
    assert not (tmp_path / "c.pt.tmp").exists()
    assert load_checkpoint(p)["meta"] == {"a": [1, 2]}

    class Evil:
        def __reduce__(self):
            return (print, ("executou código",))
    torch.save({"x": Evil()}, tmp_path / "evil.pt")
    with pytest.raises(Exception):
        load_checkpoint(tmp_path / "evil.pt")


def _tiny_cfg(data_dir, out_dir, **train):
    return load_config(None, {
        "run": {"name": "t", "seed": 3, "data_dir": str(data_dir), "output_dir": str(out_dir)},
        "model": {"emb_dim": 16, "hidden": 32, "layers": 1},
        "train": {"device": "cpu", "batch_size": 16, "max_steps": 20, "warmup_steps": 2,
                  "eval_every": 10, "log_every": 5, **train},
    })


def test_treino_curto_salva_checkpoint_que_reproduz_previsoes(tiny_data, tmp_path):
    data_dir, _ = tiny_data
    run_dir = train(_tiny_cfg(data_dir, tmp_path))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 20 and summary["device"] == "cpu"
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()

    model, vocab, answers, ckpt = load_model(run_dir / "last.pt", "cpu")
    ds = QADataset(load_split(data_dir, "val"), vocab, answers, ckpt["config"]["model"]["question_position"])
    p1, _ = predict(model, ds, "cpu")
    model2, *_ = load_model(run_dir / "last.pt", "cpu")
    p2, _ = predict(model2, ds, "cpu")
    assert p1 == p2


def test_retomada_continua_do_mesmo_ponto(tiny_data, tmp_path):
    data_dir, _ = tiny_data
    full = train(_tiny_cfg(data_dir, tmp_path / "full"))
    half = train(_tiny_cfg(data_dir, tmp_path / "half", max_steps=20, max_minutes=0.0))
    # max_minutes=0 interrompe após o 1º passo; a retomada deve chegar ao mesmo estado final.
    resumed = train(_tiny_cfg(data_dir, tmp_path / "half"), resume=half / "last.pt")
    a = load_checkpoint(full / "last.pt")["model"]
    b = load_checkpoint(resumed / "last.pt")["model"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_pausa_salva_o_passo_e_retoma_no_mesmo_ponto(tiny_data, tmp_path):
    data_dir, _ = tiny_data
    full = train(_tiny_cfg(data_dir, tmp_path / "full"))
    pause(tmp_path / "half")  # sinal já presente: pausa depois do 1º passo
    with pytest.raises(SystemExit) as exc:
        train(_tiny_cfg(data_dir, tmp_path / "half"))
    assert exc.value.code == PAUSED_EXIT
    run_dir = tmp_path / "half" / "t-s3"
    assert not (run_dir / "summary.json").exists()  # a fila ainda vê a execução como por terminar
    assert load_checkpoint(run_dir / "last.pt")["state"]["step"] == 1
    resume(tmp_path / "half")
    resumed = train(_tiny_cfg(data_dir, tmp_path / "half"), resume=run_dir / "last.pt")
    a = load_checkpoint(full / "last.pt")["model"]
    b = load_checkpoint(resumed / "last.pt")["model"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_checkpoint_no_formato_antigo_de_gru_carrega():
    """Checkpoints salvos com nn.GRU(num_layers=2) carregam no StackedGRU."""
    torch.manual_seed(0)
    old = torch.nn.GRU(8, 16, num_layers=2, batch_first=True)
    m = GRUClassifier(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.0).eval()
    sd = {f"gru.{k}": v for k, v in old.state_dict().items()}
    sd.update({k: v for k, v in m.state_dict().items() if not k.startswith("gru.")})
    m.load_state_dict(sd)
    x = torch.randn(2, 7, 8)
    with torch.no_grad():
        _, h_old = old(x)
        _, h_new = m.gru(torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor([7, 7]),
                                                                batch_first=True))
    assert torch.allclose(h_old, h_new, atol=1e-6)


def test_retomar_treino_concluido_nao_roda_passos_extras(tiny_data, tmp_path):
    data_dir, _ = tiny_data
    run_dir = train(_tiny_cfg(data_dir, tmp_path, max_steps=10))
    again = train(_tiny_cfg(data_dir, tmp_path, max_steps=10), resume=run_dir / "last.pt")
    assert json.loads((again / "summary.json").read_text(encoding="utf-8"))["steps"] == 10
    assert load_checkpoint(run_dir / "last.pt")["state"]["step"] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de GPU")
def test_processo_com_gru_em_gpu_sai_sem_crash(tmp_path):
    """Regressão: nn.GRU com dropout no cuDNN derrubava o processo na saída (0xC0000409)."""
    import subprocess
    import sys
    script = tmp_path / "s.py"
    script.write_text(
        "import torch\n"
        "from brain.models.gru import GRUClassifier\n"
        "m = GRUClassifier(20, 5, pad_id=0, emb_dim=8, hidden=16, layers=2, dropout=0.1).cuda()\n"
        "x = torch.randint(1, 20, (4, 9), device='cuda')\n"
        "m(x, torch.tensor([9, 8, 7, 6])).sum().backward()\n", encoding="utf-8")
    assert subprocess.run([sys.executable, str(script)]).returncode == 0


def test_espaco_de_respostas_inclui_unknown():
    assert "UNKNOWN" in AnswerSpace().labels


def test_vocab_round_trip(tmp_path):
    v = Vocab.build(["Projeto alfa: Ana foi para casa."])
    v.save(tmp_path / "v.json")
    assert Vocab.load(tmp_path / "v.json").itos == v.itos
