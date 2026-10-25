import pytest

from brain.data.generator import GeneratorConfig, generate_dataset


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """Dataset pequeno gerado uma vez por sessão de testes."""
    out = tmp_path_factory.mktemp("data") / "tiny"
    cfg = GeneratorConfig(name="tiny", seed=99, n_train=200, n_val=40, n_test_iid=40,
                          n_test_comp=40, n_test_long=20)
    manifest = generate_dataset(cfg, out)
    return out, manifest
