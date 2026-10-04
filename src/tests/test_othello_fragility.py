"""Tests for the Othello fragility suite (src/fragility/othello.py).

The first test is the freeze: the suite definition must hash to the value recorded
when it was tagged. If you need a different suite, add a v2; do not edit v1.

The rest check the suite against two models whose answers are known in advance:
a perfect world model (always proposes a legal move) must score zero fragility,
and a broken one (always proposes the same square) must fail.

Run from the repo root:
    python -m pytest src/tests/test_othello_fragility.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fragility_othello", ROOT / "src/fragility/othello.py")
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)
rules = fr.rules

FROZEN_VERSION = "othello-fragility-v1"
FROZEN_HASH = "644671bfb322"


class Oracle(torch.nn.Module):
    """A perfect world model: replays the moves with the rules and puts high logits on
    exactly the legal next moves. Higher token = slightly higher logit, so the top move
    and the least likely legal move are both well defined."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[bytes, int], list[int]] = {}

    def _legal(self, board: np.ndarray, mover: int) -> list[int]:
        key = (board.tobytes(), mover)
        if key not in self._cache:
            self._cache[key] = rules.legal_next(board, mover)
        return self._cache[key]

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, T = tok.shape
        out = torch.full((B, T, rules.VOCAB), -10.0)
        for b in range(B):
            moves = [rules.TOKEN_TO_SQ[int(t)] for t in tok[b] if int(t) > 0]
            if not moves:
                continue
            boards, movers = rules.replay(np.array(moves))
            for t in range(len(moves)):
                for s in self._legal(boards[t], int(movers[t])):
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
    assert fr.SUITE_VERSION == FROZEN_VERSION
    assert fr.suite_hash() == FROZEN_HASH, "the fragility suite changed after it was frozen"


def test_symmetries_keep_the_start_board() -> None:
    start = rules.start_board()
    for name in fr.SUITE["symmetries"]:
        m = fr.symmetry_map(name)
        assert sorted(m.tolist()) == list(range(rules.N_SQ)), f"{name} is not a permutation"
        moved = np.zeros_like(start)
        moved[m] = start
        assert (moved == start).all(), f"{name} changes the start position"


def test_symmetric_games_are_the_same_game_relabelled() -> None:
    games = rules.random_games(25, seed=1)
    for name in fr.SUITE["symmetries"]:
        m = fr.symmetry_map(name)
        mapped = fr.transform_games(games, name)
        assert ((mapped == -1) == (games == -1)).all(), "padding must stay padding"
        for g, h in zip(games, mapped, strict=True):
            boards, movers = rules.replay(g[g >= 0])
            out = rules.replay(h[h >= 0])
            assert out is not None, f"{name} produced an illegal game"
            boards_m, movers_m = out
            assert (movers_m == movers).all(), f"{name} changed who moves"
            assert (boards_m[:, m] == boards).all(), f"{name} board is not the relabelled board"


def test_after_pass_positions_are_real_passes() -> None:
    data = rules.build(rules.random_games(120, seed=2), legal=True)
    where = fr.after_pass_positions(data)
    assert where.sum() > 0, "no pass in 120 random games; raise the number of games"
    has_next = np.zeros_like(data["mask"])
    has_next[:, :-1] = data["mask"][:, 1:]
    checked = 0
    for g, t in zip(*np.nonzero(has_next), strict=True):
        board, mover = data["black_white"][g, t], int(data["movers"][g, t])
        opponent_stuck = not rules.legal_moves(board, rules.other(mover))
        assert opponent_stuck == bool(where[g, t])
        checked += 1
        if checked >= 1500:
            break


def test_perfect_model_is_not_fragile() -> None:
    games = {"synthetic": rules.random_games(30, seed=3), "championship": rules.random_games(30, seed=4)}
    res = fr.score_model(Oracle(), "cpu", "synthetic", batch=16,
                         games_by_distribution=games, n_detour_games=4)
    assert res["suite_hash"] == FROZEN_HASH
    assert res["in_distribution_legal_rate"] == 1.0
    assert set(res["legal_rate"]) >= {"cross_distribution/championship", "symmetry/rot180",
                                      "symmetry/transpose", "symmetry/anti_transpose"}
    assert all(r == 1.0 for r in res["legal_rate"].values())
    assert all(d["completion_rate"] == 1.0 for d in res["detour"].values())
    assert all(abs(v) < 1e-12 for v in res["drops"].values())
    assert res["fragility"] == 0.0


def test_broken_model_fails() -> None:
    data = rules.build(rules.random_games(30, seed=5), legal=True)
    rate, n = fr.legal_rate_at(AlwaysFirst(), data, "cpu", 16)
    assert n > 0 and rate < 0.2
    out = fr.detour_rollout(AlwaysFirst(), 20, 0.0, "random", 0, "cpu")
    assert out["completion_rate"] < 0.2


def test_legal_rate_at_matches_the_domain_module() -> None:
    data = rules.build(rules.random_games(30, seed=6), legal=True)
    for model in (Oracle(), AlwaysFirst()):
        mine, _ = fr.legal_rate_at(model, data, "cpu", 16)
        assert abs(mine - rules.legal_rate(model, data, "cpu", 16)) < 1e-12


def test_detours_happen_and_are_reproducible() -> None:
    oracle = Oracle()
    none = fr.detour_rollout(oracle, 4, 0.0, "random", 0, "cpu")
    assert none["n_detours"] == 0
    for mode in ("random", "adversarial"):
        a = fr.detour_rollout(oracle, 4, 0.5, mode, 0, "cpu")
        b = fr.detour_rollout(oracle, 4, 0.5, mode, 0, "cpu")
        assert a == b, "same seed must give the same rollout"
        assert a["n_detours"] > 0
