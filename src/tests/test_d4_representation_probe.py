"""Tests for Diagnostic 4, representation probing (src/diagnostics/d4/representation_probe.py).

They use random legal games and small models, so they run anywhere (including GitHub Actions):
    planted    layer 0 is noise, layer 1 holds the true board as a one hot code: the probe must
               read layer 1 almost perfectly, pick it as the best layer, and beat the control
    tiny       a real (untrained) train.py transformer through the full score() contract

Run from the repo root:
    python -m pytest src/tests/test_d4_representation_probe.py -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d4_probe", ROOT / "src/diagnostics/d4/representation_probe.py")
d4 = importlib.util.module_from_spec(_spec)
sys.modules["d4_probe"] = d4
_spec.loader.exec_module(d4)
oth, itf = d4.oth, d4.itf
train = itf.load_module("zoo_train", "src/zoo/train.py")

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row (synthetic distribution)


class Planted(itf.ZooModel):
    """Layer 0: random noise. Layer 1: the true mine / theirs / empty board, one hot (64 x 3)."""

    def __init__(self) -> None:
        super().__init__(nn.Identity(), ZOO_ID, "transformer", 2, oth.N_SQ * 3, "cpu")

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        tok = tokens.cpu().numpy()
        B, T = tok.shape
        board = np.zeros((B, T, oth.N_SQ), dtype=np.int64)
        for b in range(B):
            moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in tok[b] if t > 0])
            boards, movers = oth.replay(moves)
            rel = np.where(boards == 0, 0, np.where(boards == movers[:, None], 1, 2))
            board[b, : len(moves)] = rel
        onehot = torch.nn.functional.one_hot(torch.as_tensor(board), 3).float().view(B, T, -1)
        noise = torch.randn(B, T, oth.N_SQ * 3, generator=torch.Generator().manual_seed(int(tok.sum())))
        return torch.stack([noise, onehot])

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*tokens.shape, oth.VOCAB)


def tiny() -> "itf.ZooModel":
    torch.manual_seed(0)
    module = train.ARCHS["transformer"](n_layers=2, d_model=16, n_heads=2)
    return itf.TrainPyModel(module, ZOO_ID, "transformer", 2, 16, "cpu")


@pytest.fixture()
def fake_data(monkeypatch):
    def fake_load(name, split, n_games, seed=0, **kw):
        return oth.random_games(n_games, seed=seed + (0 if split == "val" else 1000))
    monkeypatch.setattr(oth, "load_distribution", fake_load)


SMALL = dict(n_fit=150, n_choose=50, epochs=3, batch=50, pos_batch=32)   # small data, so small steps


# ==========================================================================
# Pieces
# ==========================================================================

def test_type_ids_are_unique_per_square_and_position() -> None:
    tok = np.array([[1, 2, 60], [60, 2, 1]])
    ids = d4.type_ids(tok)
    assert ids.max() < d4.N_TYPES and ids.min() >= 0
    assert ids[0, 1] == ids[1, 1]                  # same square at the same position: same type
    assert ids[0, 0] != ids[1, 2]                  # same square at another position: different type


def test_control_table_is_fixed_and_follows_frequencies() -> None:
    freq = np.tile([0.6, 0.3, 0.1], (oth.N_SQ, 1))
    a, b = d4.control_table(freq, 0), d4.control_table(freq, 0)
    assert np.array_equal(a, b) and a.shape == (d4.N_TYPES, oth.N_SQ)
    share = np.bincount(a.ravel(), minlength=3) / a.size
    assert np.allclose(share, [0.6, 0.3, 0.1], atol=0.02)


def test_occupancy_baseline_knows_empty_squares_only() -> None:
    labels = np.array([[[0, 1, 2, 1]]])            # one position, four squares
    mask = np.array([[True]])
    rule = np.array([1, 1, 1, 1])                  # occupied squares always called mine
    assert d4.occupancy_accuracy(labels, mask, rule) == pytest.approx(3 / 4)   # wrong only on the theirs square


# ==========================================================================
# Full runs
# ==========================================================================

def test_planted_board_is_read_at_the_right_layer(fake_data) -> None:
    r = d4.score(Planted(), split="test", n_games=50, data_seed=0, **SMALL)
    d = r.details
    assert d["best_layer"] == 1
    assert d["probe_acc_by_layer"][1] > 0.95                   # the board is linearly readable
    assert d["probe_acc_occupied_by_layer"][1] > 0.9           # including whose disc it is
    assert d["probe_acc_by_layer"][1] > d["occupancy_baseline_acc"]
    assert r.score == pytest.approx(d["selectivity_by_layer"][1], abs=1e-4)
    assert r.score > 0.2                                       # far above the control


def test_tiny_model_follows_contract_and_saves(fake_data, tmp_path: Path) -> None:
    out = itf.run(d4, [tiny()], split="test", n_games=40, data_seed=0, results_dir=tmp_path, **SMALL)
    r = out[0]
    assert r.status == "ok"
    d = r.details
    assert len(d["probe_acc_by_layer"]) == len(d["selectivity_by_layer"]) == 2
    assert d["gain_over_layer0_by_layer"][0] == pytest.approx(0.0)
    assert 0.0 <= d["occupancy_baseline_acc"] <= 1.0
    best = d["best_layer"]
    assert best == int(np.argmax(d["selectivity_by_layer_choose"]))   # chosen on val, not on test
    assert r.score == pytest.approx(d["selectivity_by_layer"][best], abs=1e-4)
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D4", tmp_path))
    assert saved.diagnostic == "D4" and saved.score == r.score


def test_same_seed_same_score(fake_data) -> None:
    a = d4.score(tiny(), n_games=40, data_seed=0, **SMALL).score
    b = d4.score(tiny(), n_games=40, data_seed=0, **SMALL).score
    assert a == pytest.approx(b, abs=1e-6)


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d4)
