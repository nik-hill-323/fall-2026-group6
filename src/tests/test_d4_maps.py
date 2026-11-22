"""Tests for D4 on navigation (src/diagnostics/d4/representation_probe.py, score_maps; Amendment 5, G7; ticket N7).

No data or checkpoints: a fake data folder of random legal routes on the real map, a model whose second layer
holds a fixed random vector of the current intersection (src/tests/maps_scoring_fixtures.py), and untrained
small zoo models.

Run from the repo root:
    python -m pytest src/tests/test_d4_maps.py -q
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
d4 = _load("d4_representation_probe", "src/diagnostics/d4/representation_probe.py")
nav, itf, maps = fx.nav, fx.itf, fx.maps
SMALL = {"n_fit": 120, "n_choose": 30, "epochs": 2, "mlp_hidden": 16, "causal_cases_max": 25, "batch": 32}


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    return fx.use_fake_data(monkeypatch, tmp_path / "data", n=200)


@pytest.fixture(scope="module")
def routes():
    return maps.build(fx.dom.random_routes(120, seed=31))


# ==========================================================================
# Labels, types, control, baselines
# ==========================================================================

def test_labels_and_types_follow_the_scored_positions(routes) -> None:
    m = d4.maps_mask(routes)
    y = d4.maps_labels(routes)
    assert len(y) == m.sum() and y.min() >= 0 and y.max() < d4.MAPS_N_CLASSES
    assert np.array_equal(y + 1, routes["current"][:, :maps.N_CTX][m])
    t = d4.maps_types(routes)
    tok = routes["tokens"][:, :maps.N_CTX][m]
    turns = routes["n_turns"][:, :maps.N_CTX][m]
    assert np.array_equal(t // d4.MAPS_TURN_SLOTS, tok) and np.array_equal(t % d4.MAPS_TURN_SLOTS, turns)


def test_control_table_is_fixed_and_follows_the_frequencies() -> None:
    freq = np.zeros(d4.MAPS_N_CLASSES)
    freq[[3, 10, 99]] = [0.5, 0.3, 0.2]
    a = d4.maps_control_table(freq, 0)
    assert np.array_equal(a, d4.maps_control_table(freq, 0)) and a.shape == (d4.MAPS_N_TYPES,)
    assert set(np.unique(a)) == {3, 10, 99}
    share = np.bincount(a, minlength=d4.MAPS_N_CLASSES)[[3, 10, 99]] / len(a)
    assert np.allclose(share, [0.5, 0.3, 0.2], atol=0.01)


def test_baselines(routes) -> None:
    b = d4.maps_baselines(routes)
    m = d4.maps_mask(routes)
    cur = routes["current"][:, :maps.N_CTX]
    assert b["start"] == pytest.approx((cur == routes["tokens"][:, :1])[m].mean())
    assert b["destination"] == pytest.approx((cur == routes["dest"][:, None])[m].mean())


def test_scored_acts_do_not_depend_on_the_chunk(routes) -> None:
    m = fx.tiny("lstm")
    idx = np.arange(20)
    a = d4.maps_scored_acts(m, routes, idx)
    b = d4.maps_scored_acts(m, routes, idx, chunk=7)
    assert a.shape == (4, d4.maps_mask(routes, idx).sum(), 128) and torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(d4.maps_scored_acts(m, routes, idx, layers=[2]), a[[2]], atol=1e-6)


# ==========================================================================
# Probes and the epoch rule
# ==========================================================================

def test_planted_intersection_is_read_at_the_right_layer(routes) -> None:
    model = fx.MapsOracle("legal", d_model=64)          # layer 0: token just read; layer 1: current intersection
    fit, test = d4._take(routes, slice(0, 100)), d4._take(routes, slice(100, 120))
    ctrl = d4.maps_control_table(d4.maps_class_freq(fit))
    mu, sd = d4.maps_activation_stats(model, fit, 32)
    probes = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=20, batch=32, lr=1e-2, mlp_hidden=0)
    ev = d4.maps_evaluate_probes(model, probes["real"], probes["control"], fit, ctrl, mu, sd, 32)
    sel = ev["probe_acc"] - ev["control_acc"]
    assert ev["probe_acc"][1] > 0.95 and sel[1] > sel[0] + 0.3


def test_epoch_checkpoints_equal_shorter_runs(routes) -> None:
    model = fx.tiny("transformer")
    fit = d4._take(routes, slice(0, 60))
    ctrl = d4.maps_control_table(d4.maps_class_freq(fit))
    mu, sd = d4.maps_activation_stats(model, fit, 16)
    snap = {}

    def on_epoch(e, probes):
        if e == 2:
            snap.update({k: v.clone() for k, v in probes["real"].state_dict().items()})

    d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=3, batch=16, mlp_hidden=0, on_epoch=on_epoch)
    two = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=2, batch=16, mlp_hidden=0)
    for k, v in two["real"].state_dict().items():
        assert torch.allclose(v, snap[k], atol=1e-6), k


def test_layers_per_pass_gives_the_same_probes(routes) -> None:
    model = fx.tiny("transformer")
    fit = d4._take(routes, slice(0, 40))
    ctrl = d4.maps_control_table(d4.maps_class_freq(fit))
    mu, sd = d4.maps_activation_stats(model, fit, 16)
    a = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=1, batch=16, mlp_hidden=8)
    b = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=1, batch=16, mlp_hidden=8, layers_per_pass=1)
    for kind in ("real", "control", "mlp_real", "mlp_control"):
        for x, y in zip(a[kind].state_dict().values(), b[kind].state_dict().values(), strict=True):
            assert torch.allclose(x, y, atol=1e-5), kind


def test_mlp_probe_does_not_change_the_linear_probes(routes) -> None:
    model = fx.tiny("lstm")
    fit = d4._take(routes, slice(0, 40))
    ctrl = d4.maps_control_table(d4.maps_class_freq(fit))
    mu, sd = d4.maps_activation_stats(model, fit, 16)
    a = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=1, batch=16, mlp_hidden=0)
    b = d4.maps_fit_probes(model, fit, ctrl, mu, sd, epochs=1, batch=16, mlp_hidden=8)
    for x, y in zip(a["real"].state_dict().values(), b["real"].state_dict().values(), strict=True):
        assert torch.equal(x, y)


def test_synthetic_layer_holds_the_current_intersection(routes) -> None:
    layer = d4.SyntheticLayer(device="cpu")
    acts = layer.activations(nav.inputs(routes, slice(0, 5)))
    assert acts.shape == (1, 5, maps.N_CTX, 128)
    cur = routes["current"][0, 3]
    assert torch.equal(acts[0, 0, 3], layer.table[cur])
    g = np.random.default_rng(0).standard_normal((maps.N_NODES + 1, 128)).astype(np.float32)
    assert torch.equal(layer.table, torch.as_tensor(g))


def test_epoch_rule_runs_and_picks_the_smallest_count(data_dir) -> None:
    out = d4.maps_epoch_rule(options=(1, 2), target=0.0, n_fit=60, n_choose=10, n_test=30, device="cpu", batch=32)
    assert out["epochs"] == 1 and set(out["accuracy"]) == set(maps.DISTRIBUTIONS)
    out = d4.maps_epoch_rule(options=(1, 2), target=1.01, n_fit=60, n_choose=10, n_test=30, device="cpu", batch=32)
    assert out["epochs"] == 2                                           # none reaches it: the largest


def test_navigation_is_refused_until_the_epoch_count_is_fixed(monkeypatch) -> None:
    monkeypatch.setattr(d4, "MAPS_EPOCHS", None)
    with pytest.raises(RuntimeError, match="epoch_rule"):
        d4.maps_epochs({})
    assert d4.maps_epochs({"epochs": 3}) == 3
    monkeypatch.setattr(d4, "MAPS_EPOCHS", 5)
    assert d4.maps_epochs({}) == 5


def test_fixed_epoch_count_matches_the_rule_file() -> None:
    if d4.MAPS_EPOCHS is None:
        pytest.skip("the epoch rule has not been run yet")
    out = json.loads(d4.MAPS_EPOCH_RULE_FILE.read_text())
    assert d4.MAPS_EPOCHS == out["epochs"] and d4.MAPS_EPOCHS in d4.MAPS_EPOCH_OPTIONS


# ==========================================================================
# Causal check
# ==========================================================================

def test_causal_cases_move_to_a_neighbour_with_other_turns(routes) -> None:
    c = d4.maps_causal_cases(routes, seed=0)
    assert len(c["route"]) > 0.9 * len(routes["tokens"])
    for i in range(len(c["route"])):
        g, p, nbr = c["route"][i], c["pos"][i], c["target"][i] + 1
        assert routes["score"][g, p]
        cur, dest = routes["current"][g, p], routes["dest"][g]
        assert nbr in maps.NEXT[cur]                                     # one legal turn away
        first = next(int(maps.NEXT[cur, j]) for j in range(8) if maps.NEXT[cur, j] >= 0
                     and maps.legal_next(int(maps.NEXT[cur, j]), dest)
                     and set(maps.legal_next(int(maps.NEXT[cur, j]), dest)) != set(maps.legal_next(cur, dest)))
        assert nbr == first                                              # the first such neighbour in turn order
        assert np.array_equal(c["legal_after"][i], nav.legal_token_mask(nbr, dest))
        assert c["legal_after"][i].any() and not np.array_equal(c["legal_after"][i], c["legal_before"][i])
    again = d4.maps_causal_cases(routes, seed=0)
    assert np.array_equal(again["pos"], c["pos"])


def test_probe_edit_reaches_its_target() -> None:
    """At width 128 (the smallest zoo width) every class of a linear probe can be made its reading with 0.99."""
    torch.manual_seed(0)
    probe = torch.nn.Linear(128, d4.MAPS_N_CLASSES)
    z = torch.randn(6, 128)
    target = torch.tensor([5, 77, 1000, 3, 4500, 12])
    delta, ok, used = d4.maps_probe_edit(probe, z, target)
    assert ok.all() and (used <= d4.EDIT_STEPS).all()
    assert d4.maps_edit_reached(probe(z + delta), target).all()


def test_untrained_model_end_to_end(data_dir, tmp_path: Path) -> None:
    model = fx.tiny("transformer")
    res = itf.run(d4, [model], n_games=40, results_dir=tmp_path / "res", **SMALL)[0]
    assert res.status == "ok", res.reason
    d = res.details
    assert res.method == d4.MAPS_METHOD and d["n_classes"] == 4579 and d["epochs"] == 2
    assert len(d["probe_acc_by_layer"]) == 4 and "mlp" in d and "causal" in d
    c = d["causal"]
    assert c["layers"] == d4.edit_layers(d["best_layer"], 4) and c["n_cases"] == 25
    assert c["causal_label"] in ("not decodable", "decodable but not causal", "decodable and causal")
    assert 0.0 <= d["start_baseline_acc"] <= 1.0
    again = d4.score(model, n_games=40, **SMALL)
    assert again.score == pytest.approx(res.score, abs=1e-9)


@pytest.mark.parametrize("arch", ["lstm", "mamba"])
def test_other_architectures(data_dir, arch: str) -> None:
    if arch == "mamba":
        pytest.importorskip("mambapy")
    res = d4.score(fx.tiny(arch), n_games=30, **{**SMALL, "causal_cases_max": 8, "mlp_hidden": 0})
    assert res.status == "ok" and "causal" in res.details


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d4)
