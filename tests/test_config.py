import numpy as np

from src.config import load_config
from src.data.real import is_usable
from src.train.finetune import length_grouped_batches
from src.utils import resolve


def test_line_config_inherits_and_overrides():
    cfg = load_config(resolve("configs/lines/dual_human.yaml"), ["train.lr=0.001", "experiment.seeds=[7]"])
    assert cfg["selection"] == {"method": "dual", "real_frac": 0.3}
    assert cfg["experiment"]["line"] == "dual_human"
    assert cfg["experiment"]["seeds"] == [7]
    assert cfg["train"]["lr"] == 0.001
    assert cfg["model"]["base"] == "HuggingFaceTB/SmolLM-135M"  # inherited


def test_every_line_config_loads():
    for path in resolve("configs/lines").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg["experiment"]["line"] == path.stem


def test_env_overrides_output_dir(monkeypatch):
    monkeypatch.setenv("MC_OUTPUT_DIR", "D:/mc_outputs")
    assert load_config(resolve("configs/base.yaml"))["experiment"]["output_dir"] == "D:/mc_outputs"


def test_is_usable():
    assert is_usable("word " * 50, 40, 220)
    assert not is_usable("word " * 10, 40, 220)
    assert not is_usable("bad â€™ text " + "word " * 50, 40, 220)


def test_length_grouped_batches_cover_everything_once():
    lengths = list(np.random.default_rng(0).integers(5, 300, size=1000))
    batches = length_grouped_batches(lengths, 16, np.random.default_rng(1), group=10)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(1000))
    assert all(len(b) <= 16 for b in batches)


def test_check_signature_refuses_changed_config(tmp_path):
    import pytest

    from src.pipeline import check_signature

    path = tmp_path / "shared_config.json"
    check_signature(path, {"train": {"lr": 1e-4}})
    check_signature(path, {"train": {"lr": 1e-4}})  # same config resumes
    with pytest.raises(RuntimeError, match="different configuration"):
        check_signature(path, {"train": {"lr": 2e-4}})
