"""Tests for src/domains/maps_tasks.py: the D3 task labels and shortcut rules and the rare intersection sets
that Amendment 5 fixes from the data.

Run from the repo root:
    python -m pytest src/tests/test_maps_tasks.py -q
"""

from __future__ import annotations

import importlib.util
import json
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


maps = _load("maps", "src/domains/maps.py")
tasks = _load("maps_tasks", "src/domains/maps_tasks.py")
dom = _load("test_maps_domain", "src/tests/test_maps_domain.py")
build_mod = sys.modules["maps_build"]


def _data(n: int = 300, seed: int = 0) -> dict:
    return maps.build(dom.random_routes(n, seed))


def test_random_positions_are_scored_and_repeatable() -> None:
    d = _data()
    p = tasks.random_positions(d)
    assert ((p >= 1) & (p <= d["length"] - 2)).all()
    assert np.array_equal(p, tasks.random_positions(d))
    assert d["score"][np.arange(len(p)), p].all()


def test_labels_follow_their_definitions() -> None:
    d = _data()
    s = tasks.state_at(d, tasks.random_positions(d))
    D = maps.turn_distance()
    lat, lon = maps.coordinates()
    lab = tasks.labels(s, h_turns=5, r_metres=800.0)
    assert np.array_equal(lab["T1"], D[s["current"], s["dest"]] <= 5)
    assert np.array_equal(lab["T2"], tasks.haversine(s["current"], s["dest"]) <= 800.0)
    assert np.array_equal(lab["T3"], lat[s["current"]] > lat[s["dest"]])
    assert np.array_equal(lab["T4"], lon[s["current"]] > lon[s["dest"]])
    at = s["current"] == s["dest"]
    assert lab["T1"][at].all() and lab["T2"][at].all() and not lab["T3"][at].any() and not lab["T4"][at].any()
    with pytest.raises(ValueError):
        tasks.labels(s)                                         # thresholds not fixed until Amendment 5


def test_haversine() -> None:
    a = np.arange(1, 50)
    assert np.allclose(tasks.haversine(a, a), 0.0)
    b = a[::-1]
    assert np.allclose(tasks.haversine(a, b), tasks.haversine(b, a))
    assert abs(tasks.EARTH_RADIUS_M * np.radians(1.0) - 111_195) < 10   # one degree of latitude is about 111.2 km


def test_shortcut_rules() -> None:
    d = _data()
    pos = np.ones(len(d["tokens"]), dtype=np.int64)              # at position 1 the car is at the start
    s = tasks.state_at(d, pos)
    lab = tasks.labels(s, 5, 800.0)
    sc = tasks.start_shortcuts(s)
    assert np.array_equal(sc["T3"], lab["T3"]) and np.array_equal(sc["T4"], lab["T4"])
    x = np.arange(100)
    assert tasks.best_threshold_balanced(x, x >= 37)[0] == 1.0
    assert tasks.best_threshold_balanced(x, x < 37)[0] == 1.0
    assert tasks.balanced_accuracy(np.ones(10, bool), np.arange(10) < 3) == 0.5


def test_rare_set_and_visit_counts() -> None:
    d = _data(200, seed=3)
    counts = tasks.visit_counts(d["tokens"], chunk=64)
    manual = np.bincount(d["current"][d["score"]], minlength=maps.N_NODES + 1)
    assert np.array_equal(counts, manual)
    test = _data(100, seed=4)
    test_counts = tasks.visit_counts(test["tokens"])
    rare = tasks.rare_set(counts, test_counts)
    held = test_counts[rare].sum() / test_counts[1:].sum()
    assert held >= 0.10
    # the smallest such set: without its most visited member it would hold less than 10%
    worst = rare[np.argmax(counts[rare] * (maps.N_NODES + 1) + rare)]
    assert test_counts[np.setdiff1d(rare, [worst])].sum() / test_counts[1:].sum() < 0.10
    # every intersection outside the set was visited at least as often in training as every one inside
    outside = np.setdiff1d(np.arange(1, maps.N_NODES + 1), rare)
    assert counts[outside].min() >= counts[rare].max()


def test_rare_set_ties_are_broken_by_token_id() -> None:
    flat = np.zeros(maps.N_NODES + 1, dtype=np.int64)
    test_counts = np.zeros(maps.N_NODES + 1, dtype=np.int64)
    test_counts[1:11] = 1                                        # 10 test positions, on tokens 1..10
    assert list(tasks.rare_set(flat, test_counts)) == [1]       # token 1 alone holds 10%


def test_matched_cells_separate_what_they_should() -> None:
    d = _data()
    cells = tasks.matched_cells(d)
    n_legal = d["legal"].sum(-1)
    at = d["current"] == d["dest"][:, None]
    g, p = np.nonzero(d["score"])
    key = set(zip(n_legal[g, p], at[g, p], d["n_turns"][g, p] // 10))
    assert len(set(cells[g, p].tolist())) == len(key)


def test_compute_on_a_small_built_dataset(tmp_path: Path, monkeypatch) -> None:
    vafa = dom._fake_vafa(tmp_path / "vafa")
    out = tmp_path / "maps"
    held = set()
    for name in ("shortest-paths", "noisy-shortest-paths", "random-walks"):
        for line in (vafa / name / "heldout_sequences.txt").read_text().split("\n"):
            if line:
                w = line.split()
                held.add((w[0].replace("42438066", "7732571914"), w[1].replace("42438066", "7732571914")))
    monkeypatch.setitem(build_mod.EXPECTED, "heldout_pairs", len(held))
    build_mod.build_splits(str(vafa), str(out))
    for dist in maps.DISTRIBUTIONS:
        build_mod.build_cache(dist, str(vafa), str(out), n_cache=1000)
    monkeypatch.setattr(maps, "DATA_DIR", str(out))
    orig = maps.load_distribution
    monkeypatch.setattr(maps, "load_distribution",
                        lambda name, split, n=None, seed=0, data_dir=str(out): orig(name, split, n, seed, data_dir))
    monkeypatch.setattr(tasks, "N_THRESHOLD_ROUTES", 100)
    res = tasks.compute(tmp_path / "numbers.json", tmp_path / "rare.json")
    assert isinstance(res["H_TURNS"], int) and res["R_METRES"] > 0
    for t in tasks.TASKS:
        per = res["tasks"][t]["per_distribution"]
        assert set(per) == set(maps.DISTRIBUTIONS)
        assert all(0 <= p["positive_rate"] <= 1 for p in per.values())
        assert res["tasks"][t]["kept"] == (not res["tasks"][t]["drop_reasons"])
    rare = json.loads((tmp_path / "rare.json").read_text())["rare"]
    assert all(len(v) == res["rare"][d]["n_rare_intersections"] for d, v in rare.items())
    assert all(r["share_of_test_positions_held"] >= 0.10 for r in res["rare"].values())
