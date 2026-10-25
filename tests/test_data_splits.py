"""Determinismo do gerador, splits sem vazamento e vocabulário só do treino."""

from brain.data import world
from brain.data.generator import (
    SPLITS, GeneratorConfig, generate_episode, load_split, story_hash,
)
from brain.data.tokenizer import UNK, Vocab, episode_texts, tokenize
from brain.data.world import Event


def test_gerador_deterministico():
    cfg = GeneratorConfig()
    a = generate_episode("1:train:0", cfg, set(), False)
    b = generate_episode("1:train:0", cfg, set(), False)
    assert a == b


def test_mesmo_conteudo_gera_mesma_versao(tiny_data, tmp_path):
    from brain.data.generator import generate_dataset
    _, manifest = tiny_data
    again = generate_dataset(GeneratorConfig(name="tiny", seed=99, n_train=200, n_val=40,
                                             n_test_iid=40, n_test_comp=40, n_test_long=20),
                             tmp_path / "again")
    assert again["data_version"] == manifest["data_version"]


def test_test_long_e_mais_longo_que_todo_o_treino(tiny_data):
    data_dir, _ = tiny_data
    longest_train = max(len(ep["story"]) for ep in load_split(data_dir, "train"))
    long_eps = load_split(data_dir, "test_long")
    assert min(len(ep["story"]) for ep in long_eps) > longest_train // 2
    assert sum(len(ep["story"]) for ep in long_eps) / len(long_eps) > longest_train


def test_historias_nao_se_repetem_entre_splits(tiny_data):
    data_dir, _ = tiny_data
    seen = {}
    for split in SPLITS:
        for ep in load_split(data_dir, split):
            h = story_hash(ep)
            assert h not in seen, f"história de {split} já aparece em {seen.get(h)}"
            seen[h] = split


def test_episodio_inteiro_fica_em_um_split(tiny_data):
    data_dir, _ = tiny_data
    ids = [ep["id"] for s in SPLITS for ep in load_split(data_dir, s)]
    assert len(ids) == len(set(ids))


def test_pares_reservados_nunca_aparecem_fora_do_test_comp(tiny_data):
    data_dir, manifest = tiny_data
    held = {tuple(p) for p in manifest["heldout_pairs"]}
    assert held
    for split in ("train", "val", "test_iid"):
        for ep in load_split(data_dir, split):
            events = [Event(**e) for e in ep["events"]]
            for e in events:
                assert (e.obj, e.place) not in held
            for p in ep["projects"]:
                for o, hist in world.simulate(events, p).history.items():
                    assert not {(o, loc) for loc in hist} & held
            for q in ep["questions"]:
                assert (q["subject"], q["place"]) not in held
                assert "heldout_pair" not in q["tags"]
    comp = load_split(data_dir, "test_comp")
    assert all(any("heldout_pair" in q["tags"] for q in ep["questions"]) for ep in comp)


def test_vocabulario_vem_so_do_treino(tiny_data):
    data_dir, _ = tiny_data
    vocab = Vocab.load(data_dir / "vocab.json")
    train_tokens = {t for ep in load_split(data_dir, "train") for x in episode_texts(ep)
                    for t in tokenize(x)}
    assert set(vocab.itos[6:]) == train_tokens
    assert vocab.encode_tokens(["palavrainexistente"]) == [vocab.stoi[UNK]]


def test_tokenizer_mantem_hifen_e_acentos():
    assert tokenize("Projeto beta: o guarda-chuva está no escritório?") == [
        "projeto", "beta", ":", "o", "guarda-chuva", "está", "no", "escritório", "?"]
