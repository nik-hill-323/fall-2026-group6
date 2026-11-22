"""Tests for D1 on navigation (src/diagnostics/d1/next_token.py, score_maps; Amendment 5, G4; ticket N4).

No data or checkpoints: a fake data folder of random legal routes on the real map and models whose answers are
known (src/tests/maps_scoring_fixtures.py).

Run from the repo root:
    python -m pytest src/tests/test_d1_maps.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
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


fx = _load("maps_scoring_fixtures", "src/tests/maps_scoring_fixtures.py")
d1 = _load("d1_next_token", "src/diagnostics/d1/next_token.py")
nav, itf, maps, results = fx.nav, fx.itf, fx.maps, fx.nav.results


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    return fx.use_fake_data(monkeypatch, tmp_path / "data", n=120)


def test_oracle_scores_one(data_dir) -> None:
    res = d1.score(fx.MapsOracle("legal"), n_games=100)
    assert res.score == 1.0 and res.details["argmax_legal_rate"] == 1.0
    assert res.details["sampled_legal_rate"] > 0.999
    assert res.domain == "maps" and res.distribution == "shortest_paths" and res.method == d1.MAPS_METHOD


def test_fixed_tokens(data_dir) -> None:
    d = nav.routes("shortest_paths", "test", 100, seed=0)
    sc = d["score"][:, :maps.N_CTX]
    at_dest = ((d["current"] == d["dest"][:, None])[:, :maps.N_CTX] & sc).sum() / sc.sum()
    res = d1.score(fx.MapsOracle("fixed", fixed=maps.END), n_games=100)
    assert res.score == pytest.approx(at_dest) and res.details["sampled_legal_rate"] == pytest.approx(at_dest)
    for never in (maps.PAD, 1, maps.N_NODES):        # padding and intersections are never legal
        assert d1.score(fx.MapsOracle("fixed", fixed=never), n_games=100).score == 0.0
    assert res.details["n_positions"] == int(sc.sum())


class _Coin(itf.ZooModel):
    """Equal weight on N (token TURN0) and on padding at every position: about half the samples are N."""

    def __init__(self) -> None:
        super().__init__(torch.nn.Identity(), fx.ZOO_ID, "transformer", 1, 1, "cpu")

    def _logits(self, tokens):
        out = torch.full((*tokens.shape, maps.VOCAB), -1e4)
        out[..., maps.TURN0] = 0.0
        out[..., maps.PAD] = 0.0
        return out


def test_sampled_rate_follows_the_probabilities(data_dir) -> None:
    d = nav.routes("shortest_paths", "test", 100, seed=0)
    sc = d["score"][:, :maps.N_CTX]
    n_has_north = (d["legal"][:, :maps.N_CTX, 0] & sc).sum()
    expected = 0.5 * n_has_north / sc.sum()
    got = d1.maps_sampled_legal_rate(_Coin(), d, batch=32)
    assert got == pytest.approx(expected, abs=0.01)


def test_contract_save_and_repeatability(data_dir, tmp_path: Path) -> None:
    model = fx.tiny("transformer")
    out = itf.run(d1, [model], n_games=60, results_dir=tmp_path / "res")
    assert out[0].status == "ok" and 0.0 <= out[0].score <= 1.0
    saved = json.loads((tmp_path / "res" / "D1_next_token" / f"{fx.ZOO_ID}.json").read_text())
    assert saved["score"] == out[0].score and saved["n_games"] == 60
    again = d1.score(model, n_games=60)
    assert again.score == out[0].score and again.details["sampled_legal_rate"] == out[0].details["sampled_legal_rate"]


def test_untrained_models_are_near_the_floor(data_dir) -> None:
    for arch in ("transformer", "lstm"):
        assert d1.score(fx.tiny(arch), n_games=60).score < 0.05


def test_batch_size_does_not_change_the_score(data_dir) -> None:
    m = fx.tiny("lstm")
    a = d1.score_maps(m, n_games=50, batch=7)
    b = d1.score_maps(m, n_games=50, batch=50)
    assert a.score == b.score


def test_pilot_result_uses_its_meta(data_dir) -> None:
    pilot = fx.MapsOracle(model_id=fx.PILOT_ID, meta=nav.parse_id(fx.PILOT_ID))
    res = d1.score(pilot, n_games=50)
    assert res.seed == 100 and res.scale == "small" and res.distribution == "shortest_paths"


def test_othello_models_still_take_the_othello_path() -> None:
    class _Oth(itf.ZooModel):
        def _logits(self, tokens):
            return torch.zeros((*tokens.shape, itf.oth.VOCAB))
    m = _Oth(torch.nn.Identity(), "othello_transformer_small_synthetic_s0", "transformer", 1, 1, "cpu")
    assert not nav.is_maps(m)


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d1)
