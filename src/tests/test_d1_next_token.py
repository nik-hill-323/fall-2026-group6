"""Tests for Diagnostic 1, next token validity (src/diagnostics/d1/next_token.py).

They use random legal games and small fake models, so they run anywhere (including GitHub Actions):
    oracle    puts all its weight on the legal moves of the true board: both rates are 1
    illegal   puts all its weight on padding (token 0), which is never legal: both rates are 0

Run from the repo root:
    python -m pytest src/tests/test_d1_next_token.py -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d1_next_token", ROOT / "src/diagnostics/d1/next_token.py")
d1 = importlib.util.module_from_spec(_spec)
sys.modules["d1_next_token"] = d1
_spec.loader.exec_module(d1)
oth, itf = d1.oth, d1.itf

ZOO_ID = "othello_transformer_small_synthetic_s0"   # a real models.csv row (synthetic distribution)


class FakeModel(itf.ZooModel):
    """A ZooModel whose logits come from a rule, not a network."""

    def __init__(self, kind: str) -> None:
        super().__init__(nn.Identity(), ZOO_ID, "transformer", 1, 1, "cpu")
        self.kind = kind

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        tok = tokens.cpu().numpy()
        out = np.full((*tok.shape, oth.VOCAB), -30.0, dtype=np.float32)
        for b in range(len(tok)):
            if self.kind == "illegal":
                out[b, :, 0] = 30.0
                continue
            moves = np.array([oth.TOKEN_TO_SQ[int(t)] for t in tok[b] if t > 0])
            boards, movers = oth.replay(moves)
            for t in range(len(moves)):
                for s in oth.legal_next(boards[t], int(movers[t])):
                    out[b, t, oth.SQ_TO_TOKEN[s]] = 30.0
        return torch.as_tensor(out)


@pytest.fixture()
def fake_data(monkeypatch):
    monkeypatch.setattr(oth, "load_distribution", lambda name, split, n, seed=0, **kw: oth.random_games(n, seed=seed))


def test_oracle_scores_one(fake_data) -> None:
    r = d1.score(FakeModel("oracle"), n_games=20, data_seed=0)
    assert r.score == pytest.approx(1.0)
    assert r.details["sampled_legal_rate"] == pytest.approx(1.0)


def test_illegal_model_scores_zero(fake_data) -> None:
    r = d1.score(FakeModel("illegal"), n_games=20, data_seed=0)
    assert r.score == pytest.approx(0.0)
    assert r.details["sampled_legal_rate"] == pytest.approx(0.0)


def test_positions_match_legal_rate(fake_data) -> None:
    r = d1.score(FakeModel("oracle"), n_games=20, data_seed=0)
    data = oth.build(oth.random_games(20, seed=0), legal=True)
    assert r.details["n_positions"] == int((data["mask"].sum(1) - 1).sum())   # last move of each game skipped


def test_run_saves_valid_result_and_is_reproducible(fake_data, tmp_path: Path) -> None:
    torch.manual_seed(0)
    module = itf.load_module("zoo_train", "src/zoo/train.py").ARCHS["transformer"](n_layers=2, d_model=16, n_heads=2)
    model = itf.TrainPyModel(module, ZOO_ID, "transformer", 2, 16, "cpu")
    a = itf.run(d1, [model], n_games=15, data_seed=0, results_dir=tmp_path)[0]
    b = itf.run(d1, [model], n_games=15, data_seed=0, save=False)[0]
    assert a.status == "ok" and 0.0 <= a.score <= 1.0
    assert a.score == b.score and a.details["sampled_legal_rate"] == b.details["sampled_legal_rate"]
    saved = itf.results.load(itf.results.result_path(ZOO_ID, "D1", tmp_path))
    assert saved.diagnostic == "D1" and saved.score == a.score


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d1)
