"""Tests for the 4x4 Othello rules used by the D4 toy probe.

Every diagnostic depends on the rules being right, so these check that
play_random_game only ever produces legal moves and legal boards.

Run from the repo root:
    python -m pytest src/tests/test_othello_rules.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("toy_probe", ROOT / "src/diagnostics/d4/toy_probe.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


def test_start_board() -> None:
    b = tp.start_board()
    assert (b == tp.BLACK).sum() == 2 and (b == tp.WHITE).sum() == 2
    assert b[5] == tp.WHITE and b[6] == tp.BLACK and b[9] == tp.BLACK and b[10] == tp.WHITE


def test_first_moves_for_black() -> None:
    # On the start board black can only play squares 1, 4, 11 and 14, each flipping one stone.
    b = tp.start_board()
    assert tp.legal_moves(b, tp.BLACK) == [1, 4, 11, 14]
    assert tp.flips_for(b, 1, tp.BLACK) == [5]
    assert tp.flips_for(b, 0, tp.BLACK) == []  # corner is not legal at the start


def test_random_games_are_legal() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        moves, boards, movers, legal_sets = tp.play_random_game(rng)
        assert len(moves) == len(boards) == len(movers) == len(legal_sets)
        before = tp.start_board()
        for sq, after, mover, legal in zip(moves, boards, movers, legal_sets):
            # the move was legal for the player who made it
            assert sq in legal
            assert sorted(legal) == tp.legal_moves(before, mover)
            flips = tp.flips_for(before, sq, mover)
            assert flips, "a legal move must flip at least one stone"
            # the new board is exactly: old board + the placed stone + the flipped stones
            expected = before.copy()
            expected[sq] = mover
            expected[flips] = mover
            assert np.array_equal(after, expected)
            # one new stone per move
            assert (after != tp.EMPTY).sum() == (before != tp.EMPTY).sum() + 1
            before = after
        # the game only ends when neither player can move
        assert not tp.legal_moves(before, tp.BLACK) and not tp.legal_moves(before, tp.WHITE)


def test_relative_labels() -> None:
    b = tp.start_board()
    lab = tp.relative_labels(b, tp.BLACK)
    assert lab[6] == 1 and lab[9] == 1  # black stones are "mine" for black
    assert lab[5] == 2 and lab[10] == 2  # white stones are "theirs"
    assert (lab == 0).sum() == 12
