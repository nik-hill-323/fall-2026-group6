"""Tests for Diagnostic 3, the inductive bias probe (src/diagnostics/d3/inductive_bias.py).

They use random legal games and small models, so they run anywhere (including GitHub Actions)
with no data or checkpoints:
    planted   a model whose last layer vector is the true mine/theirs board: it should learn
              "is square X mine" from a handful of examples, and its random weight twin, which
              only sees the last move, should not
    tiny GPT  a real src/zoo/train.py transformer (small scale), to check the contract, that the
              original weights are never changed, and that the same seed gives the same score
    symbolic  the symbolic regression arm (Amendment 2). PySR needs Julia, which CI does not
              install, so these tests swap in a least squares fit with the same interface; one
              test runs real PySR when it is installed (the EC2)

Run from the repo root:
    python -m pytest src/tests/test_d3_inductive_bias.py -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d3_inductive_bias", ROOT / "src/diagnostics/d3/inductive_bias.py")
d3 = importlib.util.module_from_spec(_spec)
sys.modules["d3_inductive_bias"] = d3
_spec.loader.exec_module(d3)   # first, so PySR (Julia) loads before torch where it is installed

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
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


REAL_PYSR = d3.pysr_fit   # kept before the autouse fixture below swaps it out


def least_squares_fit(X: np.ndarray, y: np.ndarray, seed: int):
    """Stand in for pysr_fit in CI: same (predict, equation, complexity) interface."""
    A = np.c_[X, np.ones(len(X))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return (lambda Z: np.c_[Z, np.ones(len(Z))] @ coef), "least squares", len(coef)


@pytest.fixture(autouse=True)
def no_julia(monkeypatch):
    monkeypatch.setattr(d3, "pysr_fit", least_squares_fit)


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


# ==========================================================================
# Symbolic regression arm (Amendment 2)
# ==========================================================================

def test_square_name_is_the_inverse_of_square() -> None:
    assert all(d3.square(d3.square_name(s)) == s for s in range(oth.N_SQ))
    assert d3.square_name(d3.square("C4")) == "C4"


def test_describe_law_finds_the_square_that_matters() -> None:
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 0.0, 1.0], size=(500, oth.N_SQ))
    c4, f6 = d3.square("C4"), d3.square("F6")
    y_true = (X[:, c4] == 1).astype(np.int64)
    good = d3.describe_law(lambda Z: 2.0 * Z[:, c4] + 0.2 * Z[:, f6], X, 2.0 * X[:, c4], y_true, c4)
    assert good["recovered"] and good["top_square"] == "C4" and set(good["squares_used"]) == {"C4", "F6"}
    assert good["own_square_contrast_share"] > 0.8 and good["law_agreement"] > 0.7
    assert good["linear_r2"] == pytest.approx(1.0)        # 2 C4 is linear in the squares
    noisy = 2.0 * X[:, c4] + rng.normal(0, 3.0, 500)
    assert 0.0 < d3.linear_r2(X, noisy) < 0.5              # most of it is not a board formula
    shortcut = d3.describe_law(lambda Z: Z[:, f6], X, X[:, f6], y_true, c4)
    assert not shortcut["recovered"] and shortcut["top_square"] == "F6"
    flat = d3.describe_law(lambda Z: np.zeros(len(Z)), X, np.zeros(500), y_true, c4)
    assert not flat["recovered"] and flat["top_square"] is None and flat["squares_used"] == []


def test_occupancy_is_not_the_law() -> None:
    """Nikhil's review of #94: x squared is 1 for mine and for theirs, so a formula in C4 squared
    only knows whether C4 is occupied (the shortcut A7 excluded). It must not count as recovered."""
    rng = np.random.default_rng(1)
    X = rng.choice([-1.0, 0.0, 1.0], size=(500, oth.N_SQ))
    c4 = d3.square("C4")
    y_true = (X[:, c4] == 1).astype(np.int64)
    occ = d3.describe_law(lambda Z: 2.0 * Z[:, c4] ** 2 - 1, X, 2.0 * X[:, c4] ** 2 - 1, y_true, c4)
    assert occ["recovered"] is False and occ["occupancy_shortcut"] is True
    assert occ["own_square_contrast"] == pytest.approx(0.0) and occ["top_square"] is None
    law = d3.describe_law(lambda Z: 2.0 * Z[:, c4], X, 2.0 * X[:, c4], y_true, c4)
    assert law["recovered"] is True and law["occupancy_shortcut"] is False and law["law_agreement"] == 1.0
    wrong_way = d3.describe_law(lambda Z: -2.0 * Z[:, c4], X, -2.0 * X[:, c4], y_true, c4)
    assert wrong_way["recovered"] is False                 # theirs raises the margin: the wrong sign
    always_mine = d3.describe_law(lambda Z: Z[:, c4] + 3.0, X, X[:, c4] + 3.0, y_true, c4)
    assert always_mine["recovered"] is False and always_mine["law_agreement"] == 0.5   # never says "not mine"
    assert d3.linear_r2(X, 2.0 * X[:, c4] ** 2 - 1) == pytest.approx(1.0)              # the reference sees occupancy


def test_pysr_variable_names_are_valid_symbols() -> None:
    """PySR refuses names that are sympy functions (E1 is one); the prefixed names are not."""
    sympy = pytest.importorskip("sympy")
    clash = [n for n in d3.SQUARE_NAMES if n in sympy.__dict__]
    assert "E1" in clash                                   # why the prefix is needed
    assert not [n for n in d3.SR_VARIABLE_NAMES if n in sympy.__dict__]
    assert all(n.isidentifier() for n in d3.SR_VARIABLE_NAMES) and len(set(d3.SR_VARIABLE_NAMES)) == 64


def test_jaccard() -> None:
    assert d3.jaccard(["A1", "B2"], ["B2", "C3"]) == pytest.approx(1 / 3)
    assert d3.jaccard([], []) == 1.0


def test_planted_board_recovers_the_law_and_random_weights_do_not(data) -> None:
    ex_train, ex_test = data
    out = d3.evaluate(wrap(Planted(True), 3 * oth.N_SQ), ex_train, ex_test, seeds=(0, 1), **FAST)
    sr = out["symbolic_regression"]
    assert sr["recovery_rate_pretrained"] >= 0.75          # the adapted head reads the task square
    assert sr["recovery_rate_random"] <= 0.25              # the last move only: no law to recover
    assert len(sr["cells"]) == 2 * 4 * 2                   # (pretrained, random) x 4 tasks x 2 seeds
    assert set(sr["stability_by_task_pretrained"]) == {f"is {s} mine" for s in d3.TASK_SQUARES}


def test_symbolic_arm_can_be_turned_off_and_leaves_the_score_alone(data) -> None:
    ex_train, ex_test = data
    on = d3.evaluate(wrap(Planted(True), 3 * oth.N_SQ), ex_train, ex_test, seeds=(0,), **FAST)
    off = d3.evaluate(wrap(Planted(True), 3 * oth.N_SQ), ex_train, ex_test, seeds=(0,), symbolic=False, **FAST)
    assert "symbolic_regression" in on and "symbolic_regression" not in off
    assert on["score"] == pytest.approx(off["score"], abs=1e-9)


def test_real_pysr_recovers_a_one_square_law() -> None:
    """Runs only where PySR and Julia are installed (the EC2)."""
    if d3.pysr is None:
        pytest.skip("pysr not installed")
    rng = np.random.default_rng(0)
    X = rng.choice([-1.0, 0.0, 1.0], size=(300, oth.N_SQ))
    c4 = d3.square("C4")
    y = 1.5 * X[:, c4] - 0.5
    predict, eq, _ = REAL_PYSR(X, y, 0)
    d = d3.describe_law(predict, X, y, (X[:, c4] == 1).astype(np.int64), c4)
    assert d["recovered"] and d["fit_r2"] > 0.99, eq
