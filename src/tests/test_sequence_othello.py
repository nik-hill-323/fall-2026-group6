"""Tests for the exploratory whole game measures (src/analysis/sequence_othello.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


seq = _load("sequence_othello", "src/analysis/sequence_othello.py")
oth = seq.oth


class Oracle:
    """Replays the tokens with the true rules and puts the highest score on a legal next move,
    so every measure must come out perfect. With bad=True it always proposes square A1's token
    on top of that, which is illegal at the start of every game, so no move is ever legal."""

    device = "cpu"

    def __init__(self, bad: bool = False) -> None:
        self.bad = bad

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = torch.as_tensor(tokens)
        B, T = tokens.shape
        out = torch.zeros(B, T, oth.VOCAB)
        for b in range(B):
            moves = [oth.TOKEN_TO_SQ[int(x)] for x in tokens[b] if int(x) > 0]
            boards, movers = oth.replay(np.array(moves))
            for t in range(len(moves)):
                legal = oth.legal_next(boards[t], int(movers[t]))
                if legal:
                    out[b, t, oth.SQ_TO_TOKEN[legal[0]]] = 10.0
                if self.bad:
                    out[b, t, 0] = 20.0       # padding token: never a legal move
        return out


def test_oracle_is_perfect_in_both_modes():
    data = oth.build(oth.random_games(20, seed=1), legal=True)
    tf = seq.teacher_forced(Oracle(), data, batch=7)
    assert tf["overall"] == 1.0 and tf["full_game_rate"] == 1.0
    fp = seq.free_play(Oracle(), n_games=10, seed=0)
    assert fp["overall"] == 1.0 and fp["full_game_rate"] == 1.0
    assert fp["mean_errors_per_game"] == 0.0 and fp["mean_first_error_move"] is None
    assert 10 < fp["mean_game_length"] <= 60


def test_always_illegal_model_fails_from_move_two():
    fp = seq.free_play(Oracle(bad=True), n_games=5, seed=0)
    assert fp["overall"] == 0.0 and fp["full_game_rate"] == 0.0
    assert fp["mean_first_error_move"] == 2.0          # move 1 is random; the model first predicts move 2
    assert fp["mean_errors_per_game"] == fp["n_predictions"] / 5


def test_teacher_forced_overall_equals_d1_rate():
    train = _load("zoo_train", "src/zoo/train.py")
    torch.manual_seed(0)
    model = train.GPT(n_layers=1, d_model=16, n_heads=2).eval()
    data = oth.build(oth.random_games(30, seed=2), legal=True)
    tf = seq.teacher_forced(model, data, batch=8)
    assert abs(tf["overall"] - oth.legal_rate(model, data, "cpu", 8)) < 1e-12
    by = tf["by_move"]
    assert sum(by["n"]) == tf["n_positions"] and by["n"][0] == by["n"][1] == 0   # first prediction is move 2


def test_free_play_is_reproducible():
    train = _load("zoo_train", "src/zoo/train.py")
    torch.manual_seed(0)
    model = train.GPT(n_layers=1, d_model=16, n_heads=2).eval()
    a = seq.free_play(model, n_games=8, seed=3)
    b = seq.free_play(model, n_games=8, seed=3)
    assert a == b


def test_batch_size_does_not_change_results():
    train = _load("zoo_train", "src/zoo/train.py")
    torch.manual_seed(0)
    model = train.GPT(n_layers=1, d_model=16, n_heads=2).eval()
    data = oth.build(oth.random_games(12, seed=4), legal=True)
    assert seq.teacher_forced(model, data, batch=3) == seq.teacher_forced(model, data, batch=100)
    assert seq.free_play(model, n_games=12, seed=5, batch=5) == seq.free_play(model, n_games=12, seed=5, batch=100)


def test_free_play_games_are_diverse():
    """v1 played the top legal move, so 1,000 games were at most 4 distinct games. v2 samples."""
    train = _load("zoo_train", "src/zoo/train.py")
    torch.manual_seed(0)
    model = train.GPT(n_layers=1, d_model=16, n_heads=2).eval()
    fp = seq.free_play(model, n_games=40, seed=0)
    assert fp["n_distinct_games"] > 30
