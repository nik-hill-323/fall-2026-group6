"""Tests for the common diagnostic interface (src/diagnostics/interface.py).

They use a tiny transformer built by src/zoo/train.py, so they run anywhere (including GitHub
Actions). The reference Othello GPT test is skipped when transformer_lens or the weights are missing.

Run from the repo root:
    python -m pytest src/tests/test_interface.py -q
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("interface", ROOT / "src/diagnostics/interface.py")
itf = importlib.util.module_from_spec(_spec)
sys.modules["interface"] = itf
_spec.loader.exec_module(itf)
oth = itf.oth
train = itf.load_module("zoo_train", "src/zoo/train.py")

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row; weights are random here


def tiny(arch: str = "transformer", model_id: str = ZOO_ID) -> "itf.ZooModel":
    torch.manual_seed(0)
    cls = train.ARCHS[arch]
    module = cls(n_layers=2, d_model=16, n_heads=2)
    return itf.TrainPyModel(module, model_id, arch, 2, 16, "cpu")


def tokens(n: int = 3) -> torch.Tensor:
    return torch.as_tensor(oth.build(oth.random_games(n, seed=1))["tokens"])


# ==========================================================================
# ZooModel
# ==========================================================================

def test_shapes() -> None:
    m, tok = tiny(), tokens(3)
    assert m.logits(tok).shape == (3, oth.N_CTX, oth.VOCAB)
    assert m.activations(tok).shape == (2, 3, oth.N_CTX, 16)


def test_callable_works_with_legal_rate() -> None:
    m = tiny()
    data = oth.build(oth.random_games(5, seed=2), legal=True)
    rate = oth.legal_rate(m, data, "cpu", 4)
    assert 0.0 <= rate <= 1.0


@pytest.mark.parametrize("arch", ["transformer", "lstm", "mamba"])
def test_every_architecture_has_layer_activations(arch: str) -> None:
    if arch == "mamba":
        pytest.importorskip("mambapy")
    m, tok = tiny(arch, f"othello_{arch}_small_synthetic_s0"), tokens(2)
    acts = m.activations(tok)
    assert acts.shape == (2, 2, oth.N_CTX, 16)
    # the last layer through the readout gives exactly the logits, for every architecture
    assert torch.allclose(m.module.readout(acts[-1]), m.logits(tok), atol=1e-5)
    # layers differ from each other (each layer is really a different vector)
    assert not torch.allclose(acts[0], acts[1])


def test_lstm_is_a_stack_of_single_layers() -> None:
    m = tiny("lstm", "othello_lstm_small_synthetic_s0").module
    assert len(m.layers) == 2 and all(layer.num_layers == 1 for layer in m.layers)
    # same parameter count as one 2 layer nn.LSTM
    ref = torch.nn.LSTM(16, 16, num_layers=2, batch_first=True)
    assert sum(p.numel() for p in m.layers.parameters()) == sum(p.numel() for p in ref.parameters())


# ==========================================================================
# Contract and runner
# ==========================================================================

def legal_rate_diagnostic() -> SimpleNamespace:
    """A minimal diagnostic module following the contract (D1 style legal move rate)."""
    def score(model, split, n_games, data_seed, **config):
        data = oth.build(oth.random_games(n_games, seed=data_seed), legal=True)
        rate = oth.legal_rate(model, data, model.device, 8)
        return itf.result_for(model, "D1", method="legal move rate, argmax, higher is better",
                              split=split, n_games=n_games, data_seed=data_seed, score=rate)
    return SimpleNamespace(__name__="legal_rate_diagnostic", DIAGNOSTIC="D1", score=score)


def test_run_saves_valid_results(tmp_path: Path) -> None:
    out = itf.run(legal_rate_diagnostic(), [tiny()], split="test", n_games=6, data_seed=0, results_dir=tmp_path)
    assert out[0].status == "ok" and 0.0 <= out[0].score <= 1.0
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D1", tmp_path))
    assert saved.score == out[0].score and saved.arch == "transformer"


def test_same_seed_same_score(tmp_path: Path) -> None:
    a = itf.run(legal_rate_diagnostic(), [tiny()], n_games=6, data_seed=3, save=False)[0].score
    b = itf.run(legal_rate_diagnostic(), [tiny()], n_games=6, data_seed=3, save=False)[0].score
    assert a == b


def test_crash_becomes_failed_result(tmp_path: Path) -> None:
    def boom(model, split, n_games, data_seed, **config):
        raise RuntimeError("out of memory")
    diag = SimpleNamespace(__name__="boom", DIAGNOSTIC="D4", score=boom)
    out = itf.run(diag, [tiny(), tiny("transformer", "othello_transformer_small_synthetic_s1")],
                  n_games=4, results_dir=tmp_path)
    assert [r.status for r in out] == ["failed", "failed"]  # the sweep went on after the first crash
    assert "RuntimeError: out of memory" in out[0].reason
    assert itf.results.load(itf.results.result_path(ZOO_ID, "D4", tmp_path)).status == "failed"


def test_bad_diagnostic_module_rejected() -> None:
    with pytest.raises(TypeError):
        itf.check_diagnostic(SimpleNamespace(__name__="x", DIAGNOSTIC="D7", score=lambda *a, **k: None))
    with pytest.raises(TypeError):
        itf.check_diagnostic(SimpleNamespace(__name__="x", DIAGNOSTIC="D1"))


def test_reference_model_result_uses_meta() -> None:
    m = tiny("transformer", "othello_gpt_synthetic")
    m.meta = {"domain": "othello", "arch": "transformer", "scale": "othello_gpt",
              "distribution": "synthetic", "seed": 0}
    r = itf.result_for(m, "D1", method="m", split="test", n_games=1, data_seed=0, score=0.5)
    assert r.scale == "othello_gpt" and r.model_id == "othello_gpt_synthetic"


# ==========================================================================
# Reference Othello GPT (EC2 only)
# ==========================================================================

HAVE_GPT = os.path.exists(os.path.join(itf.GPT_DIR, "synthetic_model.pth"))


@pytest.mark.skipif(not HAVE_GPT, reason="Othello GPT weights not found")
def test_othello_gpt_through_interface() -> None:
    pytest.importorskip("transformer_lens")
    m = itf.othello_gpt("synthetic")
    tok = tokens(2)
    assert m.logits(tok).shape == (2, oth.N_CTX, oth.VOCAB)
    assert m.activations(tok).shape == (8, 2, oth.N_CTX, 512)
    data = oth.build(oth.random_games(50, seed=4), legal=True)
    assert oth.legal_rate(m, data, m.device, 25) > 0.99
