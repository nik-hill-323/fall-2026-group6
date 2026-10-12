"""Tests for Diagnostic 2, state equivalence (src/diagnostics/d2/state_equiv.py).

They use random legal games and small fake models, so they run anywhere (including GitHub
Actions) with no data or checkpoints:
    oracle     accepts exactly the legal moves of the true board: both errors 0, score 1
    constant   ignores the history and always accepts the same squares: never confuses two
               histories of the same board, but cannot tell different boards apart
    noise      a random choice per exact move sequence: confuses histories of the same board

Run from the repo root:
    python -m pytest src/tests/test_d2_state_equiv.py -q
"""

import importlib.util
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d2_state_equiv", ROOT / "src/diagnostics/d2/state_equiv.py")
d2 = importlib.util.module_from_spec(_spec)
sys.modules["d2_state_equiv"] = d2
_spec.loader.exec_module(d2)
oth, itf = d2.oth, d2.itf

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row (synthetic distribution)


class FakeModel(itf.ZooModel):
    """A ZooModel whose logits come from a rule, not a network."""

    def __init__(self, kind: str) -> None:
        super().__init__(nn.Identity(), ZOO_ID, "transformer", 1, 1, "cpu")
        self.kind = kind

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        tok = tokens.cpu().numpy()
        out = np.zeros((*tok.shape, oth.VOCAB), dtype=np.float32)
        for b in range(len(tok)):
            moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in tok[b] if t > 0])
            if self.kind == "constant":
                out[b, :, 1:] = np.arange(oth.VOCAB - 1, 0, -1)    # always prefers squares 1, 2, 3, ...
                continue
            boards, movers = oth.replay(moves)
            for t in range(len(moves)):
                if self.kind == "oracle":
                    for s in oth.legal_next(boards[t], int(movers[t])):
                        out[b, t, oth.SQ_TO_TOKEN[s]] = 1.0
                else:   # noise: scores seeded by the exact move sequence, blind to the board
                    rng = np.random.default_rng(zlib.crc32(tok[b, : t + 1].tobytes()))
                    out[b, t, 1:] = rng.random(oth.VOCAB - 1)
        return torch.as_tensor(out)


@pytest.fixture(scope="module")
def prepared():
    games = oth.random_games(400, seed=0)
    h = d2.build_histories(games)
    return h, d2.same_state_pairs(h, 300), d2.different_state_pairs(h, 300)


# ==========================================================================
# Histories and pairs
# ==========================================================================

def test_histories_are_distinct_prefixes_with_legal_moves(prepared) -> None:
    h, _, _ = prepared
    prefixes = {h.tokens[g, : t + 1].tobytes() for g, t in zip(h.game, h.pos, strict=True)}
    assert len(prefixes) == len(h)                  # every prefix kept once
    assert h.legal.any(1).all()                     # no history without a legal move
    assert not h.legal[:, 0].any()                  # padding is never legal


def test_same_state_pairs_share_board_but_not_moves(prepared) -> None:
    h, same, _ = prepared
    assert len(same) > 0
    for a, b in same:
        assert h.state[a] == h.state[b]
        assert h.pos[a] == h.pos[b]                 # same board means the same number of moves
        pa = h.tokens[h.game[a], : h.pos[a] + 1]
        pb = h.tokens[h.game[b], : h.pos[b] + 1]
        assert not np.array_equal(pa, pb)
        assert np.array_equal(h.legal[a], h.legal[b])


def test_different_state_pairs_same_length_different_legal_moves(prepared) -> None:
    h, _, diff = prepared
    assert len(diff) == 300
    assert len({tuple(p) for p in diff}) == len(diff)   # no pair twice
    for a, b in diff:
        assert h.pos[a] == h.pos[b]
        assert not np.array_equal(h.legal[a], h.legal[b])


def test_pairs_are_reproducible(prepared) -> None:
    h, same, diff = prepared
    assert np.array_equal(same, d2.same_state_pairs(h, 300))
    assert np.array_equal(diff, d2.different_state_pairs(h, 300))


def test_player_to_move_handles_a_pass() -> None:
    # white to move after black, unless white has no legal move: then black moves again
    board = np.zeros(64, dtype=np.int8)
    board[[0, 1]] = [oth.BLACK, oth.WHITE]          # only black can play (square 2 flips white)
    assert d2.player_to_move(board, oth.BLACK, oth.legal_next(board, oth.BLACK)) == oth.BLACK


# ==========================================================================
# Errors on hand made sets
# ==========================================================================

def test_errors_on_small_hand_made_sets() -> None:
    acc = np.zeros((4, oth.VOCAB), dtype=bool)
    legal = np.zeros((4, oth.VOCAB), dtype=bool)
    acc[0, [1, 2]] = True
    acc[1, [1, 3]] = True                            # shares 1 of 3 accepted moves with row 0
    assert d2.compression_errors(acc, np.array([[0, 1]]))[0] == pytest.approx(2 / 3)
    legal[2, [1, 2]] = True                          # boundary: 2 is legal only after row 2
    legal[3, [1, 4]] = True                          #           4 is legal only after row 3
    acc[2, [1, 2]] = True
    acc[3, [1, 2]] = True                            # model separates neither boundary move
    assert d2.distinction_errors(acc, legal, np.array([[2, 3]]))[0] == pytest.approx(1.0)
    acc[3] = False
    acc[3, [1, 4]] = True                            # now it separates both
    assert d2.distinction_errors(acc, legal, np.array([[2, 3]]))[0] == pytest.approx(0.0)


# ==========================================================================
# Fake models
# ==========================================================================

def test_oracle_scores_one(prepared) -> None:
    out = d2.evaluate(FakeModel("oracle"), *prepared, batch=64)
    assert out["compression_error"] == pytest.approx(0.0)
    assert out["distinction_error"] == pytest.approx(0.0)
    assert out["score"] == pytest.approx(1.0)


def test_constant_model_compresses_but_cannot_distinguish(prepared) -> None:
    out = d2.evaluate(FakeModel("constant"), *prepared, batch=64)
    assert out["compression_error"] == pytest.approx(0.0)   # same k for the same board, same squares
    assert out["distinction_error"] > 0.3
    assert out["score"] < 0.9


def test_noise_model_fails_compression(prepared) -> None:
    out = d2.evaluate(FakeModel("noise"), *prepared, batch=64)
    assert out["compression_error"] > 0.3
    assert out["score"] < 0.8


def test_too_few_pairs_raises(prepared) -> None:
    h, same, diff = prepared
    with pytest.raises(ValueError):
        d2.evaluate(FakeModel("oracle"), h, same[:0], diff, batch=64)


# ==========================================================================
# The diagnostic contract
# ==========================================================================

def test_score_follows_contract_and_adds_games(monkeypatch, tmp_path: Path) -> None:
    asked = []

    def fake_load(name, split, n_games, seed=0, **kw):
        asked.append(n_games)
        return oth.random_games(n_games, seed=seed)

    monkeypatch.setattr(oth, "load_distribution", fake_load)
    d2._CACHE.clear()
    out = itf.run(d2, [FakeModel("oracle")], split="test", n_games=100, data_seed=0, results_dir=tmp_path,
                  max_pairs=200, min_same_pairs=30, game_step=100, max_games=400)
    r = out[0]
    assert r.status == "ok" and r.score == pytest.approx(1.0)
    assert asked[0] == 100 and asked == sorted(asked) and asked[-1] <= 400   # more games only while too few pairs
    assert r.n_games == asked[-1] == r.details["games_used"]
    assert r.details["distribution"] == "synthetic"
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D2", tmp_path))
    assert saved.score == r.score and saved.diagnostic == "D2"
    d2._CACHE.clear()


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d2)
