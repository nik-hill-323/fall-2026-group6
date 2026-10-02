"""Tests for the shared 8x8 Othello domain (src/domains/othello.py).

Every diagnostic and the zoo trainer use this module, so these check the rules, the tokens,
the true boards, and the legal next moves.

Tests on generated games run everywhere (including GitHub Actions). Tests on the real
othello_world .bin files are skipped when the files are not there (they only exist on the EC2).

Run from the repo root:
    python -m pytest src/tests/test_othello_domain.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("othello", ROOT / "src/domains/othello.py")
oth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oth)

HAVE_DATA = bool(oth.list_files(oth.DATA_DIR, "train")) and bool(oth.list_files(oth.DATA_DIR, "val"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason=f"no .bin files under {oth.DATA_DIR}")


# ==========================================================================
# Rules
# ==========================================================================

def test_start_board() -> None:
    b = oth.start_board()
    assert (b == oth.BLACK).sum() == 2 and (b == oth.WHITE).sum() == 2
    assert b[27] == oth.WHITE and b[36] == oth.WHITE and b[28] == oth.BLACK and b[35] == oth.BLACK


def test_first_moves_for_black() -> None:
    # Standard opening: black can play d3, c4, f5, e6 = squares 19, 26, 37, 44, each flipping one stone.
    b = oth.start_board()
    assert oth.legal_moves(b, oth.BLACK) == [19, 26, 37, 44]
    assert oth.flips_for(b, 19, oth.BLACK) == [27]
    assert oth.flips_for(b, 0, oth.BLACK) == []  # corner is not legal at the start


def test_tokens_round_trip() -> None:
    assert len(oth.PLAYABLE) == 60 and oth.VOCAB == 61
    assert sorted(oth.SQ_TO_TOKEN.values()) == list(range(1, 61))
    assert all(oth.TOKEN_TO_SQ[oth.SQ_TO_TOKEN[s]] == s for s in oth.PLAYABLE)
    assert not any(c in oth.SQ_TO_TOKEN for c in oth.CENTER)


def test_random_games_replay_legally() -> None:
    games = oth.random_games(200, seed=0)
    assert games.shape == (200, 60) and games.dtype == np.int8
    for row in games:
        moves = row[row >= 0]
        out = oth.replay(moves)
        assert out is not None, "a generated game must replay without illegal moves"
        boards, movers = out
        before = oth.start_board()
        for t, (sq, after, mover) in enumerate(zip(moves, boards, movers)):
            flips = oth.flips_for(before, int(sq), int(mover))
            assert flips, "every move must flip at least one stone"
            expected = before.copy()
            expected[int(sq)] = mover
            expected[flips] = mover
            assert np.array_equal(after, expected)
            if t + 1 < len(moves):  # the next recorded move is in the legal next set
                assert int(moves[t + 1]) in oth.legal_next(after, int(mover))
            before = after
        # the game ends only when neither player can move, or the board is full
        assert len(moves) == 60 or oth.legal_next(before, int(movers[-1])) == []


def test_same_seed_same_games() -> None:
    assert np.array_equal(oth.random_games(20, seed=7), oth.random_games(20, seed=7))
    assert not np.array_equal(oth.random_games(20, seed=7), oth.random_games(20, seed=8))


def test_illegal_game_is_rejected() -> None:
    assert oth.replay(np.array([0], dtype=np.int8)) is None  # corner on move 1 is illegal for both


# ==========================================================================
# build(): the triple every diagnostic uses
# ==========================================================================

def test_build_shapes_and_labels() -> None:
    games = oth.random_games(50, seed=1)
    d = oth.build(games, legal=True)
    G = len(games)
    assert d["n_illegal_games"] == 0
    assert d["tokens"].shape == (G, oth.N_CTX) and d["mask"].shape == (G, oth.N_CTX)
    assert d["black_white"].shape == d["mine_theirs"].shape == (G, oth.N_CTX, 64)
    assert d["legal_next"].shape == (G, oth.N_CTX, oth.VOCAB)
    assert not d["legal_next"][..., 0].any(), "padding is never a legal move"
    for g in range(G):
        T = int(d["mask"][g].sum())
        assert (d["tokens"][g, :T] > 0).all() and (d["tokens"][g, T:] == 0).all()
        for t in range(T - 1):  # the recorded next move is always legal
            assert d["legal_next"][g, t, d["tokens"][g, t + 1]]
        # mine_theirs is the colour board seen from the mover
        bw, mt, mv = d["black_white"][g, :T], d["mine_theirs"][g, :T], d["movers"][g, :T]
        assert np.array_equal(mt == 1, bw == mv[:, None])
        assert np.array_equal(mt == 0, bw == oth.EMPTY)


def test_build_drops_illegal_games() -> None:
    games = oth.random_games(3, seed=2)
    bad = games[1].copy()
    bad[0] = 0  # corner first: illegal
    d = oth.build(np.stack([games[0], bad, games[2]]))
    assert d["n_illegal_games"] == 1 and len(d["tokens"]) == 2


class _Oracle(torch.nn.Module):
    """Fake model that always predicts the recorded next move: legal rate must be 1."""

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        nxt = torch.zeros_like(tok)
        nxt[:, :-1] = tok[:, 1:]
        return torch.nn.functional.one_hot(nxt, oth.VOCAB).float()


class _Pad(torch.nn.Module):
    """Fake model that always predicts padding: legal rate must be 0."""

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.zeros_like(tok), oth.VOCAB).float()


def test_legal_rate_bounds() -> None:
    d = oth.build(oth.random_games(30, seed=3), legal=True)
    assert oth.legal_rate(_Oracle(), d, "cpu", 8) == 1.0
    assert oth.legal_rate(_Pad(), d, "cpu", 8) == 0.0
    with pytest.raises(ValueError):
        oth.legal_rate(_Oracle(), oth.build(oth.random_games(2, seed=3)), "cpu", 8)


# ==========================================================================
# Real data (EC2 only)
# ==========================================================================

@needs_data
def test_real_files_format() -> None:
    games = oth.load_split(oth.DATA_DIR, "train", 5000)
    assert games.shape == (5000, 60)
    vals = np.unique(games)
    assert set(vals.tolist()) <= set(range(-1, 64)), "values must be squares 0 to 63 or -1 padding"
    assert not np.isin(vals, oth.CENTER).any(), "no move can be on a start square"
    # padding only at the end of a game
    for row in games:
        pad = np.where(row < 0)[0]
        assert len(pad) == 0 or (pad == np.arange(pad[0], 60)).all()


@needs_data
def test_real_games_replay_legally() -> None:
    d = oth.build(oth.load_split(oth.DATA_DIR, "val", 2000), legal=True)
    assert d["n_illegal_games"] == 0
    assert len(d["tokens"]) == 2000
