"""Tests for Diagnostic 3, the inductive bias probe (src/diagnostics/d3/inductive_bias.py).

They use random legal games and small models, so they run anywhere (including GitHub Actions)
with no data or checkpoints:
    planted   a model whose last layer vector is the true mine/theirs board: it should learn
              "is square X mine" from a handful of examples, and its random weight twin, which
              only sees the last move, should not
    tiny GPT  a real src/zoo/train.py transformer (small scale), to check the contract, that the
              original weights are never changed, and that the same seed gives the same score

Run from the repo root:
    python -m pytest src/tests/test_d3_inductive_bias.py -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d3_inductive_bias", ROOT / "src/diagnostics/d3/inductive_bias.py")
d3 = importlib.util.module_from_spec(_spec)
sys.modules["d3_inductive_bias"] = d3
_spec.loader.exec_module(d3)
oth, itf = d3.oth, d3.itf
train = d3._load("zoo_train", "src/zoo/train.py")

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row (synthetic distribution)
FAST = {"steps": 60, "batch": 16, "head_lr": 5e-2, "body_lr": 1e-3}


class Planted(nn.Module):
    """planted=True: the vector after the last layer is the one hot mine/theirs board (64 x 3),
    computed from the rules, so the board is already there. planted=False (the random weight
    twin): a fixed random embedding of the last move only, which cannot know any disc's colour."""

    def __init__(self, planted: bool, seed: int = 0) -> None:
        super().__init__()
        self.planted = planted
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("emb", torch.randn(oth.VOCAB, 3 * oth.N_SQ, generator=g))
        self.scale = nn.Parameter(torch.ones(1))     # one parameter, so the body has something to adapt

    def random_copy(self, seed: int) -> "Planted":
        return Planted(False, seed)

    def layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        if not self.planted:
            return [self.emb[x] * self.scale]
        out = torch.zeros(*x.shape, 3 * oth.N_SQ)
        for b, row in enumerate(x.cpu().numpy()):
            moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in row if t > 0])
            boards, movers = oth.replay(moves)
            for t in range(len(moves)):
                rel = np.where(boards[t] == movers[t], 1, np.where(boards[t] == 0, 0, 2))
                out[b, t, np.arange(oth.N_SQ) * 3 + rel] = 1.0
        return [out * self.scale]


def wrap(module: nn.Module, d_model: int, n_layers: int = 1) -> "itf.ZooModel":
    return itf.TrainPyModel(module, ZOO_ID, "transformer", n_layers, d_model, "cpu")


@pytest.fixture(scope="module")
def data():
    return d3.examples(oth.random_games(64, seed=1)), d3.examples(oth.random_games(300, seed=2))


@pytest.fixture()
def fake_data(monkeypatch):
    monkeypatch.setattr(oth, "load_distribution", lambda name, split, n, seed=0, **kw: oth.random_games(n, seed=seed + (0 if split == "val" else 7)))
    d3._CACHE.clear()
    yield
    d3._CACHE.clear()


# ==========================================================================
# Pieces
# ==========================================================================

def test_square_names() -> None:
    assert d3.square("A1") == 0 and d3.square("H8") == 63 and d3.square("C4") == 26 and d3.square("F5") == 37


def test_examples_labels_match_the_board(data) -> None:
    ex, _ = data
    assert ex["labels"].shape == (len(ex["tokens"]), len(d3.TASK_SQUARES))
    for g in range(len(ex["tokens"])):
        moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in ex["tokens"][g] if t > 0])
        boards, movers = oth.replay(moves)
        t = ex["pos"][g]
        assert 0 <= t < len(moves)
        for k, sq in enumerate(d3.TASK_SQUARES):
            assert ex["labels"][g, k] == int(boards[t][d3.square(sq)] == movers[t])


def test_examples_are_reproducible() -> None:
    games = oth.random_games(30, seed=3)
    a, b = d3.examples(games), d3.examples(games)
    assert (a["pos"] == b["pos"]).all() and (a["labels"] == b["labels"]).all()


def test_balanced_accuracy() -> None:
    y = np.array([0, 0, 0, 1])
    assert d3.balanced_accuracy(np.zeros(4, int), y) == 0.5        # constant guess is chance, not 0.75
    assert d3.balanced_accuracy(y.copy(), y) == 1.0


def test_last_layer_has_gradients() -> None:
    gpt = train.build_model("transformer", "small")
    h = d3.last_layer(gpt, torch.randint(1, oth.VOCAB, (2, oth.N_CTX)))
    assert h.shape == (2, oth.N_CTX, 128) and h.requires_grad


# ==========================================================================
# The diagnostic measures what it should
# ==========================================================================

def test_planted_board_gains_more_than_random_weights(data) -> None:
    ex_train, ex_test = data
    out = d3.evaluate(wrap(Planted(True), 3 * oth.N_SQ), ex_train, ex_test, seeds=(0,), **FAST)
    # 64 examples in 192 one hot features: many features fit them, so even a perfect board
    # reaches about 0.8 balanced accuracy, not 1. The random twin stays near chance (0.5).
    assert out["pretrained_balanced_accuracy"] > 0.7
    assert out["random_balanced_accuracy"] < 0.6
    assert out["score"] > 0.15
    assert all(gap > 0.05 for gap in out["gap_by_task"].values())


def test_identical_models_have_no_gap(data) -> None:
    ex_train, ex_test = data
    twin = wrap(Planted(False, seed=5), 3 * oth.N_SQ)
    out = d3.evaluate(twin, ex_train, ex_test, seeds=(0,), make_random=lambda m, s: Planted(False, seed=5), **FAST)
    assert abs(out["score"]) < 1e-9


# ==========================================================================
# The contract, and safety
# ==========================================================================

def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d3)


def test_run_saves_a_valid_result(fake_data, tmp_path: Path) -> None:
    out = itf.run(d3, [wrap(Planted(True), 3 * oth.N_SQ)], split="test", n_games=80, data_seed=0,
                  results_dir=tmp_path, n_adapt=32, seeds=(0, 1), **FAST)
    r = out[0]
    assert r.status == "ok", r.reason
    assert r.score > 0.05 and r.n_games == 80 and r.diagnostic == "D3"
    assert r.details["distribution"] == "synthetic" and r.details["n_adapt_games"] == 32
    assert len(r.details["cells"]) == 2 * len(d3.TASK_SQUARES) and len(r.details["gap_by_seed"]) == 2
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D3", tmp_path))
    assert saved.score == r.score


def test_original_weights_are_unchanged(fake_data) -> None:
    gpt = train.build_model("transformer", "small")
    before = {k: v.clone() for k, v in gpt.state_dict().items()}
    d3.score(wrap(gpt, 128, 4), n_games=20, n_adapt=16, seeds=(0,), steps=3, batch=8)
    after = gpt.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_same_seed_same_score(fake_data) -> None:
    torch.manual_seed(0)
    gpt = train.build_model("transformer", "small")
    a = d3.score(wrap(gpt, 128, 4), n_games=20, n_adapt=16, seeds=(0,), steps=3, batch=8)
    b = d3.score(wrap(gpt, 128, 4), n_games=20, n_adapt=16, seeds=(0,), steps=3, batch=8)
    assert a.score == b.score and a.details["cells"] == b.details["cells"]


def test_random_baseline_matches_the_zoo_architecture() -> None:
    gpt = train.build_model("transformer", "small")
    rnd = d3.random_like(wrap(gpt, 128, 4), seed=0)
    assert type(rnd) is type(gpt)
    assert [p.shape for p in rnd.parameters()] == [p.shape for p in gpt.parameters()]
    assert not all(torch.equal(p, q) for p, q in zip(rnd.parameters(), gpt.parameters(), strict=True))
