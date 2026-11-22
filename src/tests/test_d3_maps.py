"""Tests for D3 on navigation (src/diagnostics/d3/inductive_bias.py, score_maps; Amendment 5, G6; ticket N6).

No data or checkpoints: a fake data folder of random legal routes on the real map (maps_scoring_fixtures) and
small models whose answers are known:
    planted   a model whose last layer vector holds the state the tasks ask about (how far the destination is
              in turns and metres, and how far east of it the car is), computed from the map: it should learn
              T1, T2 and T4 from 64 routes; its random weight twin only sees the token just read
    tiny      untrained src/zoo/train.py models, for the contract, the random baseline and that the original
              weights never change
    symbolic  PySR needs Julia, which CI does not install, so a least squares fit with the same interface stands
              in; one test runs real PySR when it is installed (the EC2)

Run from the repo root:
    python -m pytest src/tests/test_d3_maps.py -q
"""

from __future__ import annotations

import importlib.util
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


d3 = _load("d3_inductive_bias", "src/diagnostics/d3/inductive_bias.py")   # first: PySR before torch
fx = _load("maps_scoring_fixtures", "src/tests/maps_scoring_fixtures.py")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

nav, itf, maps, mt = fx.nav, fx.itf, fx.maps, d3.mt
FAST = {"steps": 60, "batch": 16, "head_lr": 5e-2, "body_lr": 1e-3}
D_PLANTED = 8


def least_squares_fit(X: np.ndarray, y: np.ndarray, seed: int):
    """Stand in for maps_pysr_fit in CI: same (predict, equation, complexity) interface."""
    A = np.c_[X, np.ones(len(X))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return (lambda Z: np.c_[Z, np.ones(len(Z))] @ coef), "least squares", len(coef)


REAL_PYSR = d3.maps_pysr_fit


@pytest.fixture(autouse=True)
def no_julia(monkeypatch):
    monkeypatch.setattr(d3, "maps_pysr_fit", least_squares_fit)


class MapsPlanted(nn.Module):
    """planted=True: the vector after the last layer holds, at every position, the state the tasks ask about,
    computed from the map: turn distance to the destination, straight line distance, and longitude and latitude
    of the current intersection minus those of the destination. planted=False (the random weight twin): a fixed
    random embedding of the token just read, which does not know where the car is."""

    def __init__(self, planted: bool, seed: int = 0) -> None:
        super().__init__()
        self.planted = planted
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("emb", torch.randn(maps.VOCAB, D_PLANTED, generator=g))
        self.scale = nn.Parameter(torch.ones(1))     # one parameter, so the body has something to adapt

    def random_copy(self, seed: int) -> "MapsPlanted":
        return MapsPlanted(False, seed)

    def layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        if not self.planted:
            return [self.emb[x] * self.scale]
        tok = x.cpu().numpy()
        cur = nav.current_intersections(tok)
        live = cur > 0
        c = np.where(live, cur, 1)
        dest = np.broadcast_to(np.maximum(tok[:, 1:2], 1), c.shape)
        lat, lon = maps.coordinates()
        turns = maps.turn_distance()[c, dest].astype(np.float64)
        feats = np.zeros((*tok.shape, D_PLANTED))
        feats[..., 0] = (turns - mt.H_TURNS - 0.5) / 5
        feats[..., 1] = (mt.haversine(c, dest) - mt.R_METRES) / 500
        feats[..., 2] = (lon[c] - lon[dest]) / np.nanstd(lon[1:])
        feats[..., 3] = (lat[c] - lat[dest]) / np.nanstd(lat[1:])
        feats[~live] = 0.0
        return [torch.as_tensor(feats, dtype=torch.float32) * self.scale]


def wrap(module: nn.Module, model_id: str = fx.ZOO_ID, d_model: int = D_PLANTED, n_layers: int = 1,
         meta: dict | None = None) -> "itf.ZooModel":
    return itf.TrainPyModel(module, model_id, "transformer", n_layers, d_model, "cpu", meta)


def pilot(module: nn.Module, d_model: int, n_layers: int, arch: str = "transformer") -> "itf.ZooModel":
    pid = fx.PILOT_ID.replace("transformer", arch)
    return itf.TrainPyModel(module, pid, arch, n_layers, d_model, "cpu", nav.parse_id(pid))


@pytest.fixture(scope="module")
def examples():
    """Adaptation and test examples from random legal routes (no data folder needed)."""
    a = d3.maps_examples(maps.build(fx.dom.random_routes(64, seed=21)), "shortest_paths")
    b = d3.maps_examples(maps.build(fx.dom.random_routes(300, seed=22)), "shortest_paths")
    return a, b


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    d3._CACHE.clear()
    yield fx.use_fake_data(monkeypatch, tmp_path / "data", n=200)
    d3._CACHE.clear()


# ==========================================================================
# Tasks, labels and shortcut rules (G6)
# ==========================================================================

def test_tasks_are_g6s() -> None:
    assert d3.MAPS_TASKS == ("T1", "T2", "T4")            # T3 dropped by the drop rule
    assert (mt.H_TURNS, mt.R_METRES) == (10, 823.9)


def test_shortcut_rules_come_from_the_frozen_numbers_file() -> None:
    assert d3.shortcut_rule("T1", "shortest_paths") == "turns so far > 4"
    assert d3.shortcut_rule("T2", "noisy_shortest_paths") == "turns so far > 20"
    assert d3.shortcut_rule("T1", "random_walks") == "turns so far > 30"
    assert d3.shortcut_rule("T4", "random_walks") == "start instead of current"


def test_shortcut_answers() -> None:
    data = maps.build(fx.dom.random_routes(120, seed=5))
    s = mt.state_at(data, mt.random_positions(data))
    ans = d3.shortcut_answers(s, "shortest_paths")
    assert ans.shape == (len(s["turns"]), 3)
    assert np.array_equal(ans[:, 0], s["turns"] > 4) and np.array_equal(ans[:, 1], s["turns"] > 5)
    assert np.array_equal(ans[:, 2], mt.start_shortcuts(s)["T4"])


def test_an_unknown_shortcut_rule_is_refused(monkeypatch) -> None:
    data = maps.build(fx.dom.random_routes(20, seed=5))
    s = mt.state_at(data, mt.random_positions(data))
    monkeypatch.setattr(d3, "shortcut_rule", lambda t, dist: "something else")
    with pytest.raises(ValueError):
        d3.shortcut_answers(s, "shortest_paths")


def test_examples_follow_maps_tasks(examples) -> None:
    _, test = examples
    data = maps.build(fx.dom.random_routes(300, seed=22))
    pos = mt.random_positions(data, 0)
    s = mt.state_at(data, pos)
    lab = mt.labels(s)
    assert np.array_equal(test["pos"], pos)
    assert data["score"][np.arange(len(pos)), pos].all()
    assert test["tokens"].shape[1] == maps.N_CTX and (test["pos"] < maps.N_CTX).all()
    assert np.array_equal(test["labels"], np.stack([lab["T1"], lab["T2"], lab["T4"]], 1).astype(np.int64))
    X = test["features"]
    assert X.shape == (len(pos), 7)
    assert np.allclose(X.mean(0), 0, atol=1e-9) and np.allclose(X.std(0), 1, atol=1e-9)
    lat, lon = maps.coordinates()
    assert np.corrcoef(X[:, 1], lon[s["current"]])[0, 1] == pytest.approx(1.0)
    assert np.corrcoef(X[:, 6], s["turns"])[0, 1] == pytest.approx(1.0)


# ==========================================================================
# Symbolic regression arm (G6)
# ==========================================================================

def _X(n: int = 600, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 7))


def test_t4_law_recovered_only_with_the_right_inputs_and_signs() -> None:
    X = _X()
    c = {n: j for j, n in enumerate(d3.MAPS_SR_NAMES)}
    y_true = (X[:, c["cur_lon"]] > X[:, c["dest_lon"]]).astype(np.int64)

    def law(Z):
        return 2.0 * Z[:, c["cur_lon"]] - 1.5 * Z[:, c["dest_lon"]]
    good = d3.describe_maps_law(law, X, law(X), y_true, "T4")
    assert good["recovered"] is True and good["start_shortcut"] is False and good["top_input"] == "cur_lon"
    assert good["signed_contrast"]["cur_lon"] > 0 > good["signed_contrast"]["dest_lon"]
    assert set(good["inputs_used"]) == {"cur_lon", "dest_lon"} and good["fit_r2"] == 1.0
    assert good["linear_r2"] == pytest.approx(1.0) and good["law_agreement"] > 0.9

    def start(Z):
        return 2.0 * Z[:, c["start_lon"]] - Z[:, c["dest_lon"]]
    short = d3.describe_maps_law(start, X, start(X), y_true, "T4")
    assert short["recovered"] is False and short["start_shortcut"] is True and short["top_input"] == "start_lon"

    def wrong_way(Z):
        return -law(Z)
    assert d3.describe_maps_law(wrong_way, X, wrong_way(X), y_true, "T4")["recovered"] is False

    def no_dest(Z):
        return 2.0 * Z[:, c["cur_lon"]]                   # destination longitude does not lower the margin
    assert d3.describe_maps_law(no_dest, X, no_dest(X), y_true, "T4")["recovered"] is False

    def always_yes(Z):
        return law(Z) * 0.01 + 100.0                       # right shape, but never says no: no better than chance
    assert d3.describe_maps_law(always_yes, X, always_yes(X), y_true, "T4")["recovered"] is False

    flat = d3.describe_maps_law(lambda Z: np.zeros(len(Z)), X, np.zeros(len(X)), y_true, "T4")
    assert flat["recovered"] is False and flat["top_input"] is None and flat["inputs_used"] == []


def test_contrast_is_the_10th_to_90th_percentile_effect() -> None:
    X = _X(seed=1)
    j = d3.MAPS_SR_NAMES.index("turns")
    lo, hi = np.percentile(X[:, j], [10, 90])
    d = d3.describe_maps_law(lambda Z: 3.0 * Z[:, j], X, 3.0 * X[:, j], (X[:, j] > 0).astype(np.int64), "T1")
    assert d["contrast"]["turns"] == pytest.approx(round(3.0 * (hi - lo), 4), abs=1e-4)
    assert d["signed_contrast"]["turns"] == d["contrast"]["turns"]
    assert d["recovered"] is None and d["start_shortcut"] is None     # T1, T2: no recovery rule


def test_pysr_variable_names_are_valid_symbols() -> None:
    sympy = pytest.importorskip("sympy")
    assert not [n for n in d3.MAPS_SR_NAMES if n in sympy.__dict__]
    assert all(n.isidentifier() for n in d3.MAPS_SR_NAMES) and len(set(d3.MAPS_SR_NAMES)) == 7


def test_real_pysr_recovers_the_t4_law() -> None:
    """Runs only where PySR and Julia are installed (the EC2)."""
    if d3.pysr is None:
        pytest.skip("pysr not installed")
    X = _X(300, seed=2)
    c = {n: j for j, n in enumerate(d3.MAPS_SR_NAMES)}
    y = 1.5 * X[:, c["cur_lon"]] - 1.2 * X[:, c["dest_lon"]]
    predict, eq, _ = REAL_PYSR(X, y, 0)
    d = d3.describe_maps_law(predict, X, y, (y > 0).astype(np.int64), "T4")
    assert d["recovered"] and d["fit_r2"] > 0.99, eq


# ==========================================================================
# The probe
# ==========================================================================

def test_planted_state_gains_more_than_random_weights(examples) -> None:
    train, test = examples
    out = d3.maps_evaluate(wrap(MapsPlanted(True)), train, test, seeds=(0, 1), **FAST)
    assert out["score"] > 0.15, out["gap_by_task"]
    assert all(v > 0.05 for v in out["gap_by_task"].values()), out["gap_by_task"]
    assert len(out["cells"]) == 2 * 3 and len(out["gap_by_seed"]) == 2
    assert set(out["shortcut_agreement_by_task_pretrained"]) == {"T1", "T2", "T4"}
    for c in out["cells"]:
        assert 0.0 <= c["pretrained_shortcut_agreement"] <= 1.0 and 0.0 <= c["random_shortcut_agreement"] <= 1.0
    sr = out["symbolic_regression"]
    assert len(sr["cells"]) == 2 * 3 * 2                  # (pretrained, random) x 3 tasks x 2 seeds
    t4 = [x for x in sr["cells"] if x["model"] == "pretrained" and x["task"] == "T4"]
    assert all(x["law_agreement"] > 0.8 for x in t4), t4
    assert sr["settings"]["select_k_features"] is None and sr["settings"]["inputs"] == d3.MAPS_SR_NAMES


def test_identical_models_have_no_gap(examples) -> None:
    train, test = examples

    class Same(MapsPlanted):
        def random_copy(self, seed):
            return MapsPlanted(True)
    out = d3.maps_evaluate(wrap(Same(True)), train, test, seeds=(0,), symbolic=False, **FAST)
    assert out["score"] == pytest.approx(0.0, abs=1e-9)


def test_symbolic_arm_leaves_the_score_alone(examples) -> None:
    train, test = examples
    on = d3.maps_evaluate(wrap(MapsPlanted(True)), train, test, seeds=(0,), **FAST)
    off = d3.maps_evaluate(wrap(MapsPlanted(True)), train, test, seeds=(0,), symbolic=False, **FAST)
    assert "symbolic_regression" in on and "symbolic_regression" not in off
    assert on["score"] == pytest.approx(off["score"], abs=1e-9)


# ==========================================================================
# The contract, the random baseline and safety
# ==========================================================================

def test_score_sends_navigation_models_to_score_maps(data_dir) -> None:
    r = d3.score(wrap(MapsPlanted(True)), n_games=80, n_adapt=32, seeds=(0,), **FAST)
    assert r.method == d3.MAPS_METHOD and r.diagnostic == "D3" and r.n_games == 80
    d = r.details
    assert d["distribution"] == "shortest_paths" and d["n_adapt_routes"] == 32
    assert d["shortcut_rules"] == {"T1": "turns so far > 4", "T2": "turns so far > 5", "T4": "start instead of current"}
    assert d["protocol"]["h_turns"] == 10 and d["protocol"]["r_metres"] == 823.9


def test_pilot_result_is_saved_in_the_pilot_folder(data_dir, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nav, "PILOT_RESULTS_DIR", tmp_path / "pilot_scoring")
    model = pilot(MapsPlanted(True), D_PLANTED, 1)
    out, not_loaded = nav.run(d3, [model.model_id], save=True, load_fn=lambda m: model, n_games=60,
                              n_adapt=32, seeds=(0,), **FAST)
    assert not not_loaded and out[0].status == "ok", out[0].reason
    saved = itf.results.load(itf.results.result_path(model.model_id, "D3", tmp_path / "pilot_scoring"))
    assert saved.score == out[0].score and saved.details["distribution"] == "shortest_paths"


def test_large_anchors_are_skipped(data_dir) -> None:
    for aid in ("maps_vafa_noisy_shortest_paths", "maps_vafa_random_walks"):
        meta = {"domain": "maps", "arch": "transformer", "scale": "vafa_gpt2",
                "distribution": aid.removeprefix("maps_vafa_"), "seed": 0}
        res = d3.score(fx.MapsOracle(model_id=aid, meta=meta))
        assert res.status == "skipped" and "23 GB" in res.reason and res.score is None


def test_module_passes_interface_check() -> None:
    itf.check_diagnostic(d3)


@pytest.mark.parametrize("arch", ["transformer", "lstm"])
def test_random_baseline_of_a_pilot_matches_its_architecture(arch: str) -> None:
    model = fx.tiny(arch)
    p = pilot(model.module, model.d_model, model.n_layers, arch)
    rnd = d3.maps_random_like(p, seed=1000)
    assert type(rnd) is type(model.module)
    assert [q.shape for q in rnd.parameters()] == [q.shape for q in model.module.parameters()]
    assert not all(torch.equal(a, b) for a, b in zip(rnd.parameters(), model.module.parameters(), strict=True))
    again = d3.maps_random_like(p, seed=1000)
    assert all(torch.equal(a, b) for a, b in zip(rnd.parameters(), again.parameters(), strict=True))


def test_original_weights_are_unchanged_and_same_seed_same_score(data_dir) -> None:
    model = fx.tiny("transformer")
    before = {k: v.clone() for k, v in model.module.state_dict().items()}
    a = d3.score(model, n_games=20, n_adapt=16, seeds=(0,), steps=3, batch=8, symbolic=False)
    b = d3.score(model, n_games=20, n_adapt=16, seeds=(0,), steps=3, batch=8, symbolic=False)
    after = model.module.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert a.status == "ok" and a.score == b.score and a.details["cells"] == b.details["cells"]


def test_vafa_style_gpt2_body() -> None:
    """The anchors' wrapper: same last layer as HFGPT2Model.activations, dropout off while adapting, and a
    random twin of the same configuration."""
    pytest.importorskip("transformers")
    gpt2 = itf.build_gpt2({"vocab_size": maps.VOCAB, "n_embd": 16, "n_layer": 2, "n_head": 2})
    ids = np.arange(maps.VOCAB)
    meta = {"domain": "maps", "arch": "transformer", "scale": "vafa_gpt2", "distribution": "random_walks", "seed": 0}
    model = itf.HFGPT2Model(gpt2, "maps_vafa_random_walks", 2, 16, ids, ids, "cpu", meta)
    tokens = torch.as_tensor(maps.build(fx.dom.random_routes(4, seed=3))["tokens"][:, :maps.N_CTX])
    body = d3.maps_body(model)
    body.train()
    assert not body.gpt2.training                          # dropout off
    with torch.no_grad():
        mine = body.layer_outputs(tokens)
    assert len(mine) == 2 and torch.allclose(mine[-1], model.activations(tokens)[-1], atol=1e-5)
    assert torch.allclose(d3.last_layer(body, tokens), d3.last_layer(body, tokens))    # deterministic in train mode
    rnd = d3.maps_random_like(model, seed=1000)
    assert isinstance(rnd, d3.HFBody) and rnd.gpt2 is not gpt2
    assert not torch.equal(rnd.gpt2.transformer.h[0].attn.c_attn.weight, gpt2.transformer.h[0].attn.c_attn.weight)


def test_command_line_exits_1_when_a_model_is_not_loaded(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["inductive_bias.py", "--models", fx.PILOT_ID])
    monkeypatch.setattr(nav, "load", lambda m, device=None: (_ for _ in ()).throw(FileNotFoundError("no checkpoint")))
    with pytest.raises(SystemExit) as e:
        d3.main()
    assert e.value.code == 1 and "NOT SCORED" in capsys.readouterr().out


def test_command_line_refuses_zoo_models_before_the_tags(monkeypatch) -> None:
    monkeypatch.setattr(nav, "missing_tags", lambda tags=nav.GATE_TAGS: [nav.FRAGILITY_TAG])
    monkeypatch.setattr(sys, "argv", ["inductive_bias.py", "--models", fx.ZOO_ID])
    with pytest.raises(SystemExit) as e:
        d3.main()
    assert nav.FRAGILITY_TAG in str(e.value.code)
