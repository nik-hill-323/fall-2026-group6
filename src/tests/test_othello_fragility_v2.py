"""Tests for the Othello fragility suite, version 2 (src/fragility/othello_v2.py).

The first test is the freeze: v2 must hash to the value recorded when it was tagged, and the v1
suite it builds on must be unchanged. The others check the two parts v2 replaces (Amendment 3):
the after pass comparison at the same move number, and detour rollouts played to the end.

Run from the repo root:
    python -m pytest src/tests/test_othello_fragility_v2.py -q
"""

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


v2 = _load("fragility_othello_v2", "src/fragility/othello_v2.py")
v1 = v2.v1
rules = v2.rules

FROZEN_VERSION = "othello-fragility-v2"
FROZEN_HASH = "a5f70d4658a5"
V1_FROZEN_HASH = "644671bfb322"


class Oracle(torch.nn.Module):
    """A perfect world model: high logits on exactly the legal next moves (as in the v1 tests)."""

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, T = tok.shape
        out = torch.full((B, T, rules.VOCAB), -10.0)
        for b in range(B):
            moves = [rules.TOKEN_TO_SQ[int(t)] for t in tok[b] if int(t) > 0]
            if not moves:
                continue
            boards, movers = rules.replay(np.array(moves))
            for t in range(len(moves)):
                for s in rules.legal_next(boards[t], int(movers[t])):
                    token = rules.SQ_TO_TOKEN[s]
                    out[b, t, token] = 10.0 + 0.01 * token
        return out


class AlwaysFirst(torch.nn.Module):
    """A broken model: always proposes token 1 (square A1)."""

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(tok.shape[0], tok.shape[1], rules.VOCAB)
        out[..., 1] = 5.0
        return out


def test_suite_is_frozen() -> None:
    assert v2.SUITE_VERSION == FROZEN_VERSION
    assert v2.suite_hash() == FROZEN_HASH, "the v2 suite changed after it was frozen"
    assert v1.suite_hash() == V1_FROZEN_HASH, "v1 must stay exactly as tagged"


def test_after_pass_baseline_removes_easier_positions() -> None:
    """The legal rate depends on the move number and on how many moves are legal, and pass
    positions sit late in the game with few legal moves, but within each (move number, number of
    legal moves) cell pass positions are as often legal as non pass ones: no real pass effect.
    v2 must give a drop near 0. Matching on the move number alone, or comparing with all positions
    (v1), shows a large negative drop."""
    rng = np.random.default_rng(0)
    G, T = 6000, rules.N_CTX
    scored = np.ones((G, T), dtype=bool)
    n_legal = rng.integers(1, 12, size=(G, T))
    where = np.zeros((G, T), dtype=bool)
    where[:600, 45:] = True
    n_legal[:600, 45:] = rng.integers(1, 3, size=(600, T - 45))      # passes: few legal moves
    p_ok = np.linspace(0.6, 0.8, T)[None, :] + 0.15 * (n_legal <= 2)  # few legal moves: easier
    correct = rng.random((G, T)) < p_ok
    ap = v2.after_pass_matched(correct, scored, where, n_legal)
    assert abs(ap["baseline"] - ap["rate"]) < 0.02
    same_t = (~where)[:, 45:]
    assert correct[:, 45:][same_t].mean() - ap["rate"] < -0.1                   # move number only
    assert correct[scored].mean() - ap["rate"] < -0.1                           # v1 style
    assert ap["n_left_out"] == 0 and ap["n_positions"] == 600 * (T - 45)


def test_after_pass_finds_a_real_weakness() -> None:
    G, T = 50, rules.N_CTX
    scored = np.ones((G, T), dtype=bool)
    where = np.zeros((G, T), dtype=bool)
    where[:10, 40] = True
    n_legal = np.full((G, T), 3)
    correct = ~where                        # always legal, except right after a pass
    ap = v2.after_pass_matched(correct, scored, where, n_legal)
    assert ap["rate"] == 0.0 and ap["baseline"] == 1.0 and ap["mean_move_number"] == 40.0


def test_after_pass_left_out_when_no_match() -> None:
    G, T = 20, rules.N_CTX
    scored = np.ones((G, T), dtype=bool)
    where = np.zeros((G, T), dtype=bool)
    where[0, 30] = where[1, 31] = True
    n_legal = np.full((G, T), 5)
    n_legal[0, 30] = 2                      # no non pass position at move 30 has 2 legal moves
    ap = v2.after_pass_matched(np.ones((G, T), bool), scored, where, n_legal)
    assert ap["n_positions"] == 1 and ap["n_left_out"] == 1


def test_after_pass_with_no_pass() -> None:
    G, T = 5, rules.N_CTX
    ap = v2.after_pass_matched(np.ones((G, T), bool), np.ones((G, T), bool), np.zeros((G, T), bool),
                               np.ones((G, T), int))
    assert ap["n_positions"] == 0 and ap["rate"] is None


def test_correct_at_matches_v1() -> None:
    data = rules.build(rules.random_games(30, seed=6), legal=True)
    for model in (Oracle(), AlwaysFirst()):
        correct, scored = v2.correct_at(model, data, "cpu", 16)
        rate, n = v1.legal_rate_at(model, data, "cpu", 16)
        assert int(scored.sum()) == n and abs(correct.sum() / n - rate) < 1e-12


def test_broken_model_plays_full_games() -> None:
    """v1 ends a game at the first illegal proposal; v2 keeps playing the model's best legal move,
    so a weak model is still scored on every stage of the game."""
    out = v2.detour_rollout(AlwaysFirst(), 6, 0.0, "random", 0, "cpu")
    old = v1.detour_rollout(AlwaysFirst(), 6, 0.0, "random", 0, "cpu")
    assert out["legal_rate"] < 0.2
    assert out["mean_game_length"] > 50 and out["n_predictions"] > 6 * 45
    assert old["n_predictions"] <= 6        # v1 stops at the first illegal proposal


def test_detours_happen_and_are_reproducible() -> None:
    oracle = Oracle()
    none = v2.detour_rollout(oracle, 4, 0.0, "random", 0, "cpu")
    assert none["n_detours"] == 0 and none["legal_rate"] == 1.0
    for mode in ("random", "adversarial"):
        a = v2.detour_rollout(oracle, 4, 0.5, mode, 0, "cpu")
        b = v2.detour_rollout(oracle, 4, 0.5, mode, 0, "cpu")
        assert a == b, "same seed must give the same rollout"
        assert a["n_detours"] > 0 and a["legal_rate"] == 1.0


def test_perfect_model_is_not_fragile() -> None:
    games = {"synthetic": rules.random_games(30, seed=3), "championship": rules.random_games(30, seed=4)}
    res = v2.score_model(Oracle(), "cpu", "synthetic", batch=16, games_by_distribution=games, n_detour_games=3)
    assert res["suite_hash"] == FROZEN_HASH and res["base_suite_hash"] == V1_FROZEN_HASH
    assert res["in_distribution_legal_rate"] == 1.0
    assert set(res["drops"]) >= {"cross_distribution/championship", "symmetry/rot180", "symmetry/transpose",
                                 "symmetry/anti_transpose", "detour/random/p=0.1", "detour/random/p=0.5",
                                 "detour/adversarial/p=0.1", "detour/adversarial/p=0.5"}
    assert all(abs(v) < 1e-12 for v in res["drops"].values())
    assert res["fragility"] == 0.0


def test_v1_and_v2_agree_on_the_unchanged_variants() -> None:
    games = {"synthetic": rules.random_games(20, seed=7), "championship": rules.random_games(20, seed=8)}
    model = AlwaysFirst()
    a = v1.score_model(model, "cpu", "synthetic", batch=16, games_by_distribution=games, n_detour_games=2)
    b = v2.score_model(model, "cpu", "synthetic", batch=16, games_by_distribution=games, n_detour_games=2)
    assert abs(a["in_distribution_legal_rate"] - b["in_distribution_legal_rate"]) < 1e-12
    for k in ("cross_distribution/championship", "symmetry/rot180", "symmetry/transpose", "symmetry/anti_transpose"):
        assert abs(a["legal_rate"][k] - b["legal_rate"][k]) < 1e-12
