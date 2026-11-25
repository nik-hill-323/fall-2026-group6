"""Tests for the navigation analysis (src/analysis/hypotheses_maps.py; Amendments 5 to 7; ticket N14).

The whole pipeline runs on fake navigation zoos written in the real file formats (models.csv, training records,
D1 to D4 results, maps-fragility-v1 results, G9 results, anchors), built so each rule has a known outcome. Written
and run before any H1 to H3 statistic was computed on the real navigation results.

Run from the repo root:
    python -m pytest src/tests/test_hypotheses_maps.py -q
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hypotheses_maps", ROOT / "src/analysis/hypotheses_maps.py")
hm = importlib.util.module_from_spec(_spec)
sys.modules["hypotheses_maps"] = hm
_spec.loader.exec_module(hm)
hy = hm.hy

FOLDERS = hm.DIAG_FOLDERS
N_BOOT_TEST = 300        # fewer resamples than the preregistered 2,000, only to keep the tests short


@pytest.fixture(autouse=True)
def fewer_resamples(monkeypatch):
    """Every bootstrap in these tests uses 300 resamples (the planted effects are large); the real run uses 2,000."""
    orig = hy.boot_indices
    monkeypatch.setattr(hy, "N_BOOT", N_BOOT_TEST)
    monkeypatch.setattr(hy, "boot_indices", lambda rows, n_boot=N_BOOT_TEST, seed=hy.BOOT_SEED: orig(rows, n_boot, seed))


def test_real_run_uses_2000_resamples() -> None:
    import inspect
    assert hm.hy.BOOT_SEED == 0
    src = inspect.getsource(sys.modules["hypotheses"])
    assert "N_BOOT = 2000" in src


def _write(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _record(m: dict, comp: float) -> dict:
    return {"model_id": m["model_id"], "domain": "maps", "arch": m["arch"], "scale": m["scale"],
            "distribution": m["distribution"], "seed": int(m["seed"]), "n_params": 1,
            "config": {"distribution": m["distribution"], "seed": int(m["seed"])},
            "train": {"steps_run": 5000, "final_loss": 1.0, "wall_sec": 60.0},
            "eval": {"legal_move_rate": comp, "next_token_acc": 0.5}}


def _fields(mid: str, d: str) -> dict:
    """The model fields and run id every result file carries (src/component/results.py)."""
    if mid.startswith("maps_vafa_"):
        f = {"arch": "transformer", "scale": "vafa", "distribution": mid.removeprefix("maps_vafa_"), "seed": 0}
    else:
        body, seed = mid.removeprefix("maps_").rsplit("_s", 1)
        arch, scale, dist = body.split("_", 2)
        f = {"arch": arch, "scale": scale, "distribution": dist, "seed": int(seed)}
    return {"domain": "maps", **f, "run_id": f"{mid}_{d}"}


def write_model(root: Path, mid: str, sc: dict, frag: float, in_dist: float, g9: float, rng,
                d4_extra: dict | None = None, label: str = "decodable but not causal") -> None:
    for d, folder in FOLDERS.items():
        det: dict = {}
        if d == "D4":
            det = {"best_layer": 1, "probe_acc_by_layer": [0.1, 0.3], "start_baseline_acc": 0.05,
                   "destination_baseline_acc": 0.12, "causal": {"causal_label": label, "causal_score": 0.1}}
            det.update(d4_extra or {})
        if d == "D3":
            det = {"shortcut_agreement_by_task_pretrained": {"T1": 0.6, "T2": 0.7, "T4": float(rng.random())}}
        _write(root / "results" / folder / f"{mid}.json",
               {"model_id": mid, "diagnostic": d, "method": "fake", "split": "test", "n_games": 1000,
                "data_seed": 0, "status": "ok", "score": float(sc[d]), "details": det, **_fields(mid, d)})
    _write(root / "results" / "fragility_maps" / f"{mid}.json",
           {"model_id": mid, "suite_version": hm.FRAGILITY_SUITE, "suite_hash": hm.FRAGILITY_HASH,
            "fragility": float(frag), "in_distribution_legal_rate": float(in_dist)})
    _write(root / "results" / "whole_route_maps" / f"{mid}.json",
           {"model_id": mid, "status": "ok", "teacher_forced": {"share_routes_all_legal": 0.5},
            "free_play": {"share_drives_no_illegal": float(g9), "share_reached_destination": 0.3}})


def make_zoo(tmp: Path, scores_fn, frag_fn, comp_fn=lambda i, m, rng: 0.9, g9_fn=None, label_fn=None) -> Path:
    """A fake navigation zoo: training records and every result file for the 81 maps models of models.csv, and
    full results for the shortest paths anchor only."""
    root = tmp / "repo"
    (root / "src/docs").mkdir(parents=True)
    shutil.copy(ROOT / "src/docs/models.csv", root / "src/docs/models.csv")
    with open(ROOT / "src/docs/models.csv", newline="") as f:
        models = [r for r in csv.DictReader(f) if r["domain"] == "maps"]
    assert len(models) == 81
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        comp = comp_fn(i, m, rng)
        _write(root / "results" / "zoo_training" / f"{m['model_id']}.json", _record(m, comp))
        sc = scores_fn(i, m, rng, comp)
        g9 = g9_fn(i, m, rng, sc) if g9_fn else float(rng.random())
        lab = label_fn(i, m, rng) if label_fn else "decodable but not causal"
        write_model(root, m["model_id"], sc, frag_fn(i, m, rng, sc), comp, g9, rng, label=lab)
    write_model(root, "maps_vafa_shortest_paths", {d: 0.9 for d in hm.DIAGS}, 0.01, 0.999, 0.98, rng)
    return root


def rand_scores(i, m, rng, c):
    return {d: float(rng.random()) for d in hm.DIAGS}


def noise_frag(i, m, rng, s):
    return float(rng.normal())


def comp_spread(i, m, rng):
    return float(np.clip(0.55 + 0.42 * rng.random(), 0.5, 0.99))


def low_lstm(i, m, rng):
    """The three seeds of LSTM small random walks below admission, as on the real zoo."""
    return 0.48 if (m["arch"], m["scale"], m["distribution"]) == ("lstm", "small", "random_walks") else 0.9


# ==========================================================================
# Reading files and G12
# ==========================================================================

def test_everything_scored_all_three_architectures(tmp_path: Path) -> None:
    root = make_zoo(tmp_path, rand_scores, noise_frag, low_lstm)
    s = hm.analyse(root)
    assert s["G12_completeness"]["complete_architectures"] == ["transformer", "lstm", "mamba"]
    assert not s["architecture_left_out_of_H2_and_regressions"]
    c = s["counts"]
    assert (c["planned"], c["trained"], c["admitted"], c["all_four_scores"], c["with_fragility"]) == (81, 81, 78, 78, 78)
    assert [x["model_id"] for x in s["section5"]["not_admitted"]] == [
        f"maps_lstm_small_random_walks_s{k}" for k in range(3)]
    assert s["section5"]["runs_not_ok"] == []
    adm = {(t["arch"], t["scale"]): t for t in s["section5"]["admission_by_arch_scale"]}
    assert adm[("lstm", "small")]["not_admitted"] == 3 and adm[("all", "all")]["admitted"] == 78
    assert s["H1"]["n_models"] == 78 and s["bootstrap"]["n_configurations"] == 26
    assert s["H3"]["n_models"] == 78


def test_an_architecture_missing_one_g9_file_is_left_out(tmp_path: Path) -> None:
    """G12: one Mamba model without its G9 file makes Mamba incomplete. The analysis runs on transformers and
    LSTMs; architecture still varies, so it stays in H2."""
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    (root / "results/whole_route_maps/maps_mamba_large_random_walks_s2.json").unlink()
    s = hm.analyse(root)
    comp = s["G12_completeness"]
    assert comp["complete_architectures"] == ["transformer", "lstm"]
    assert comp["by_arch"]["mamba"]["not_scored"] == ["maps_mamba_large_random_walks_s2"]
    assert s["counts"]["all_four_scores"] == 54 and s["H1"]["n_models"] == 54
    assert {"run": "G9", "status": "missing"}.items() <= next(
        x for x in s["section5"]["runs_not_ok"] if x["model_id"] == "maps_mamba_large_random_walks_s2").items()
    assert {t["factor"] for t in s["H2"]["factors"]} == {"arch", "scale", "distribution"}


def test_one_architecture_leaves_architecture_out_of_h2(tmp_path: Path) -> None:
    """G10: with a single architecture complete, the architecture factor is left out of H2 and Holm runs over two."""
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    (root / "results/whole_route_maps/maps_lstm_small_shortest_paths_s0.json").unlink()
    (root / "results/D2_state_equiv/maps_mamba_small_shortest_paths_s0.json").unlink()
    s = hm.analyse(root)
    assert s["architectures_analysed"] == ["transformer"] and s["architecture_left_out_of_H2_and_regressions"]
    assert s["H2"]["factors_left_out"] == ["arch"]
    assert {t["factor"] for t in s["H2"]["factors"]} == {"scale", "distribution"}
    assert "D2" in s["H3"] and s["H3"]["n_models"] == 27


def test_no_architecture_complete_reports_without_h(tmp_path: Path) -> None:
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    for a in ("transformer", "lstm", "mamba"):
        (root / f"results/fragility_maps/maps_{a}_small_shortest_paths_s0.json").unlink()
    s = hm.analyse(root)
    assert s["architectures_analysed"] == [] and "H1" not in s and "without H1 to H3" in s["note"]
    assert len(s["anchors"]) == 3


def test_fragility_from_another_suite_does_not_count(tmp_path: Path) -> None:
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    p = root / "results/fragility_maps/maps_transformer_small_shortest_paths_s0.json"
    d = json.loads(p.read_text())
    d["suite_hash"] = "000000000000"
    p.write_text(json.dumps(d))
    s = hm.analyse(root)
    assert s["G12_completeness"]["by_arch"]["transformer"]["not_scored"] == ["maps_transformer_small_shortest_paths_s0"]
    assert "transformer" not in s["architectures_analysed"]


def test_failed_run_with_reason_is_scored_but_not_in_h(tmp_path: Path) -> None:
    """Section 5 and G12: a failed D3 with its reason counts as scored; the model is left out of H1 to H3 and its
    run is listed with the reason."""
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    p = root / "results/D3_inductive_bias/maps_lstm_large_shortest_paths_s1.json"
    _write(p, {"model_id": "maps_lstm_large_shortest_paths_s1", "diagnostic": "D3", "method": "fake",
               "split": "test", "n_games": 1000, "data_seed": 0, "status": "failed", "score": None,
               "reason": "CUDA out of memory", **_fields("maps_lstm_large_shortest_paths_s1", "D3")})
    s = hm.analyse(root)
    assert s["G12_completeness"]["complete_architectures"] == ["transformer", "lstm", "mamba"]
    assert s["counts"]["all_four_scores"] == 80 and s["missing_scores"]["D3"] == ["maps_lstm_large_shortest_paths_s1"]
    bad = s["section5"]["runs_not_ok"]
    assert bad == [{"model_id": "maps_lstm_large_shortest_paths_s1", "run": "D3", "status": "failed",
                    "reason": "CUDA out of memory"}]


def test_d4_secondary_number() -> None:
    """Amendment 6 H6: probe accuracy at best_layer minus the larger baseline."""
    res = {"details": {"best_layer": 1, "probe_acc_by_layer": [0.1, 0.3], "start_baseline_acc": 0.05,
                       "destination_baseline_acc": 0.12}}
    assert hm.d4_secondary(res) == pytest.approx(0.18)
    assert hm.d4_secondary({"details": {}}) is None


# ==========================================================================
# H1 to H3 and the navigation additions
# ==========================================================================

def test_h3_finds_the_planted_predictor_and_g10_within_distributions(tmp_path: Path) -> None:
    """Relative fragility depends on D3 only: D3 is the best predictor and adds to D1, also with training
    distribution as a covariate (G10 a) and within each distribution (G10 b)."""
    def frag(i, m, rng, s):
        return (-s["D3"] + 0.1 * rng.normal()) * 0.8
    root = make_zoo(tmp_path, rand_scores, frag)
    s = hm.analyse(root)
    h3 = s["H3"]
    assert h3["best_predictor"] == "D3" and h3["D3"]["adds_to_D1"] and not h3["D2"]["adds_to_D1"]
    g = s["G10_training_distribution_control"]
    assert g["a_regression_with_distribution"]["D3"]["adds_to_D1_within_distributions"]
    assert g["a_regression_with_distribution"]["D3"]["ci"][1] < 0
    assert set(g["b_within_distribution"]) == {"shortest_paths", "noisy_shortest_paths", "random_walks"}
    for dist, w in g["b_within_distribution"].items():
        assert w["n_models"] == 27 and w["n_configurations"] == 9 and w["D3"]["adds_to_D1"]
    sec = s["H3_secondary"]
    assert "relative_fragility" in sec and "fragility" in sec and "rel_frag_v2" not in json.dumps(sec)
    assert sec["relative_fragility"]["best_predictor_section4"] == "D3"
    assert set(s["fragility_distribution"]) == {"fragility", "relative_fragility"}


def test_g10_link_that_is_only_between_distributions(tmp_path: Path) -> None:
    """D4 and fragility both differ by training distribution and nothing else (the Othello pattern): the
    within distribution partial correlations do not find a link."""
    shift = {"shortest_paths": 0.0, "noisy_shortest_paths": 0.5, "random_walks": 1.0}
    def scores(i, m, rng, c):
        return {"D1": rng.random(), "D2": rng.random(), "D3": rng.random(),
                "D4": shift[m["distribution"]] + 0.05 * rng.random()}
    def frag(i, m, rng, s):
        return -shift[m["distribution"]] + 0.3 * rng.normal()
    s = hm.analyse(make_zoo(tmp_path, scores, frag))
    g = s["G10_training_distribution_control"]["b_within_distribution"]
    assert sum(w["D4"]["adds_to_D1"] for w in g.values()) <= 1


def test_h6_uses_the_secondary_d4(tmp_path: Path) -> None:
    """H6: with the secondary D4 equal to the primary, the sensitivity H1 equals the primary H1; with a different
    secondary, the D4 pairs change and the other pairs do not."""
    root = make_zoo(tmp_path, rand_scores, noise_frag)
    s = hm.analyse(root)
    h6 = s["H6_sensitivity_secondary_D4"]
    assert "A3" not in h6 and h6["H1"]["n_models"] == 81
    prim = {q["pair"]: q["tau"] for q in s["H1"]["pairs"]}
    sens = {q["pair"]: q["tau"] for q in h6["H1"]["pairs"]}
    # every model has the same secondary D4 (0.18) in this fake: D4 pairs are not computable, the rest equal
    for pair in prim:
        if "D4" in pair:
            assert not np.isfinite(sens[pair])
        else:
            assert sens[pair] == pytest.approx(prim[pair])


def test_g9_secondary_and_d3_shortcut(tmp_path: Path) -> None:
    def g9(i, m, rng, s):
        return s["D2"] + 0.01 * rng.normal()
    s = hm.analyse(make_zoo(tmp_path, rand_scores, noise_frag, g9_fn=g9))
    g = s["G9_secondary"]
    assert g["D2"]["spearman"] > 0.95 and g["D2"]["ci"][0] > 0.9 and "not used in any H decision" in g["note"]
    assert s["D3_shortcut"]["n_models"] == 81 and "descriptive" in s["D3_shortcut"]["note"]


def test_causal_flag_on_navigation(tmp_path: Path) -> None:
    def label(i, m, rng):
        return "decodable and causal" if i % 3 == 0 else "decodable but not causal"
    def frag(i, m, rng, s):
        return -0.5 * (i % 3 == 0) + 0.05 * rng.normal()
    s = hm.analyse(make_zoo(tmp_path, rand_scores, frag, label_fn=label))
    f = s["H3_causal_flag"]["flag_beyond_D4"]
    assert f["adds_to_D4"] and f["n_causal"] == 27
    assert "D3_vs_occupancy_shortcut" not in s["H3_causal_flag"]       # Othello only


def test_anchors_are_description_only(tmp_path: Path) -> None:
    s = hm.analyse(make_zoo(tmp_path, rand_scores, noise_frag))
    a = {r["model_id"]: r for r in s["anchors"]}
    assert set(a) == set(hm.ANCHORS)
    assert a["maps_vafa_shortest_paths"]["D1"] == 0.9 and a["maps_vafa_shortest_paths"]["g9_no_illegal"] == 0.98
    assert a["maps_vafa_random_walks"]["D1_status"] == "missing" and a["maps_vafa_random_walks"]["fragility_status"] == "missing"
    assert s["H1"]["n_models"] == 81                                    # anchors are never in an H statistic


def test_outputs(tmp_path: Path) -> None:
    root = make_zoo(tmp_path, rand_scores, noise_frag, low_lstm)
    s = hm.analyse(root)
    oth = {"label": "EXPLORATORY", "n_models": 0}
    paths = hm.write_outputs(s, tmp_path / "out", tmp_path / "fig", oth, tmp_path / "oth" / "control.json", root)
    names = {p.name for p in paths}
    assert {"summary.json", "model_table.csv", "anchors.csv", "h1_pairs.csv", "h3.csv", "master_table.csv",
            "control.json", "fragility_distribution.svg"} <= names
    summ = json.loads((tmp_path / "out" / "summary.json").read_text())     # valid JSON, no NaN
    assert summ["counts"]["all_four_scores"] == 78
    with open(tmp_path / "out" / "model_table.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 81 and all(r["scored"] == "True" for r in table)
    hm._print(s)


# ==========================================================================
# Othello, exploratory (G10)
# ==========================================================================

def test_othello_control_is_labelled_exploratory() -> None:
    """Runs on the real Othello results on main (already final and seen, hence exploratory)."""
    o = hm.othello_distribution_control(ROOT)
    assert o["label"].startswith("EXPLORATORY") and o["domain"] == "othello"
    assert o["n_models"] >= 70
    assert set(o["b_within_distribution"]) == {"synthetic", "championship", "mixed"}
