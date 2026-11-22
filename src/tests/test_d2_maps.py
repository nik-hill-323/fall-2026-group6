"""Tests for D2 on navigation (src/diagnostics/d2/state_equiv.py, score_maps; Amendment 5, G5; ticket N5).

No data or checkpoints: a fake data folder of random legal routes on the real map (random walks revisit
intersections, so they hold same state pairs) and models whose answers are known.

Run from the repo root:
    python -m pytest src/tests/test_d2_maps.py -q
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

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
d2 = _load("d2_state_equiv", "src/diagnostics/d2/state_equiv.py")
nav, itf, maps = fx.nav, fx.itf, fx.maps
SMALL = {"max_pairs": 300, "min_same_pairs": 20, "game_step": 50, "max_games": 200}


@pytest.fixture(scope="module")
def prepared():
    data = maps.build(fx.dom.random_routes(150, seed=11))
    h = d2.maps_build_histories(data)
    return h, d2.maps_same_state_pairs(h, 300), d2.maps_different_state_pairs(h, 300)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    d2._MAPS_CACHE.clear()
    return fx.use_fake_data(monkeypatch, tmp_path / "data", n=200)


def test_histories_are_distinct_prefixes_at_scored_positions(prepared) -> None:
    h, _, _ = prepared
    d = h.data
    assert d["score"][h.route, h.pos].all()
    prefixes = {d["tokens"][g, :p + 1].tobytes() for g, p in zip(h.route, h.pos)}
    assert len(prefixes) == len(h)
    cur = d["current"][h.route, h.pos]
    assert np.array_equal(h.state, cur * (maps.N_NODES + 1) + d["dest"][h.route])
    assert np.array_equal(h.n_turns, h.pos - 1)
    assert h.legal.any(1).all()


def test_same_state_pairs_share_the_state_not_the_prefix(prepared) -> None:
    h, same, _ = prepared
    assert len(same) > 0
    d = h.data
    for a, b in same:
        assert h.state[a] == h.state[b]
        pa, pb = d["tokens"][h.route[a], :h.pos[a] + 1], d["tokens"][h.route[b], :h.pos[b] + 1]
        assert pa.tobytes() != pb.tobytes()
        assert np.array_equal(h.legal[a], h.legal[b])          # same state, same legal next set


def _all_pairs(h) -> list[tuple[int, int]]:
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(h.state.tolist()):
        groups.setdefault(s, []).append(i)
    return [(a, b) for idx in groups.values() if len(idx) > 1 for x, a in enumerate(idx) for b in idx[x + 1:]]


def test_same_state_sampling_equals_listing_all_pairs_then_drawing(prepared) -> None:
    h, _, _ = prepared
    every = np.array(_all_pairs(h), dtype=np.int64)
    assert np.array_equal(d2.maps_same_state_pairs(h, max_pairs=10 ** 9), every)     # under the cap: all
    cap = len(every) // 3
    idx = np.sort(np.random.default_rng(0).choice(len(every), cap, replace=False))
    assert np.array_equal(d2.maps_same_state_pairs(h, max_pairs=cap, seed=0), every[idx])


def test_pair_in_group_enumerates_in_order() -> None:
    for n in range(2, 9):
        want = list(itertools.combinations(range(n), 2))
        assert [d2._pair_in_group(r, n) for r in range(len(want))] == want


def test_different_state_pairs(prepared) -> None:
    h, _, diff = prepared
    assert len(diff) > 0
    for a, b in diff:
        assert h.n_turns[a] == h.n_turns[b] and not np.array_equal(h.legal[a], h.legal[b])
    assert np.array_equal(diff, d2.maps_different_state_pairs(h, 300))          # reproducible


def test_errors_on_hand_made_sets() -> None:
    n, e, end = maps.TURN0, maps.TURN0 + 2, maps.END
    acc = [frozenset({n, e}), frozenset({n, e}), frozenset({n, end}), frozenset({1, 2})]
    assert d2.maps_compression_errors(acc, np.array([[0, 1], [0, 2]])).tolist() == [0.0, 2 / 3]
    legal = np.zeros((4, 9), dtype=bool)
    legal[0, [0, 2]] = True        # N, E
    legal[2, [0, 8]] = True        # N, end
    legal[3, [0, 8]] = True
    # pair (0, 2): boundary E (only 0) and end (only 2); model separates both
    assert d2.maps_distinction_errors(acc, legal, np.array([[0, 2]])).tolist() == [0.0]
    # pair (0, 3): boundary E and end; model 3 accepts neither (intersections only): E is right, end is wrong
    assert d2.maps_distinction_errors(acc, legal, np.array([[0, 3]])).tolist() == [0.5]


def test_oracle_scores_one(prepared) -> None:
    h, same, diff = prepared
    out = d2.maps_evaluate(fx.MapsOracle("legal"), h, same, diff, batch=32)
    assert out["score"] == 1.0 and out["compression_error"] == 0.0 and out["distinction_error"] == 0.0


def test_constant_model_compresses_but_cannot_distinguish(prepared) -> None:
    h, same, diff = prepared
    out = d2.maps_evaluate(fx.MapsOracle("fixed", fixed=maps.END), h, same, diff, batch=32)
    assert out["compression_error"] == 0.0 and out["distinction_error"] > 0.6


def test_noise_model_fails_compression(prepared) -> None:
    h, same, diff = prepared
    out = d2.maps_evaluate(fx.MapsOracle("noise"), h, same, diff, batch=32)
    assert out["compression_error"] > 0.5


def test_too_few_pairs_raises(prepared) -> None:
    h, same, _ = prepared
    with pytest.raises(ValueError, match="not enough pairs"):
        d2.maps_evaluate(fx.MapsOracle("legal"), h, same, np.zeros((0, 2), dtype=np.int64))


def test_score_follows_contract_and_adds_routes(data_dir, tmp_path: Path) -> None:
    res = itf.run(d2, [fx.MapsOracle("legal")], n_games=50, results_dir=tmp_path / "res",
                  min_same_pairs=10 ** 6, game_step=50, max_games=150, max_pairs=300)[0]
    assert res.status == "ok" and res.score == 1.0
    assert res.n_games == 150 and res.details["routes_used"] == 150      # grew 50 -> 100 -> 150 and stopped
    assert res.method == d2.MAPS_METHOD and res.domain == "maps"
    assert (tmp_path / "res" / "D2_state_equiv" / f"{fx.ZOO_ID}.json").exists()


def test_untrained_model_scores_and_repeats(data_dir) -> None:
    m = fx.tiny("transformer")
    a = d2.score(m, n_games=100, **SMALL)
    b = d2.score(m, n_games=100, **SMALL)
    assert a.status == "ok" and a.score == b.score and 0.0 <= a.score <= 1.0
