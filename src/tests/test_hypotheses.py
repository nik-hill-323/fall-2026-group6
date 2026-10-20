"""Tests for the H1 to H3 analysis (src/analysis/hypotheses.py).

The statistics are checked on cases whose answer is known, and the whole pipeline is run on a fake
zoo written in the real file formats (models.csv, zoo_registry.csv, result JSONs, fragility JSONs),
built so that each hypothesis has a known outcome.

Run from the repo root:
    python -m pytest src/tests/test_hypotheses.py -q
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
_spec = importlib.util.spec_from_file_location("hypotheses", ROOT / "src/analysis/hypotheses.py")
hy = importlib.util.module_from_spec(_spec)
sys.modules["hypotheses"] = hy
_spec.loader.exec_module(hy)


# --------------------------------------------------------------------------
# Statistics on known cases
# --------------------------------------------------------------------------

def test_kendall_w_known_values() -> None:
    x = np.arange(10.0)
    assert hy.kendall_w(np.column_stack([x, x, x, x])) == pytest.approx(1.0)
    assert hy.kendall_w(np.column_stack([x, -x])) == pytest.approx(0.0)


def test_partial_tau_formula() -> None:
    rng = np.random.default_rng(0)
    c = rng.normal(size=400)
    x, y = c + 0.3 * rng.normal(size=400), c + 0.3 * rng.normal(size=400)
    assert hy.kendall_tau(x, y) > 0.5
    # Kendall's partial tau does not fall all the way to 0 when x and y share only c (a known
    # property of this measure), but it falls below the 0.30 "disagree" line of Section 4.
    assert abs(hy.partial_tau(x, y, c)) < 0.30
    tx, ty, txy = hy.kendall_tau(x, c), hy.kendall_tau(y, c), hy.kendall_tau(x, y)
    want = (txy - tx * ty) / np.sqrt((1 - tx ** 2) * (1 - ty ** 2))
    assert hy.partial_tau(x, y, c) == pytest.approx(want)


def test_categories_follow_section_4() -> None:
    assert hy.category(0.6, [0.4, 0.8]) == "agree"
    assert hy.category(0.6, [-0.1, 0.8]) == "disagree"   # interval includes 0
    assert hy.category(0.25, [0.1, 0.4]) == "disagree"   # tau below 0.30
    assert hy.category(0.4, [0.2, 0.6]) == "partial"
    assert hy.category(float("nan"), None) == "not computable"


def test_holm() -> None:
    assert hy.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_partial_spearman_removes_the_control() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=300)
    x, y = z + 0.2 * rng.normal(size=300), z + 0.2 * rng.normal(size=300)
    assert hy.spearman(x, y) > 0.8 and abs(hy.partial_spearman(x, y, z)) < 0.15


def test_residual_rank_w_drops_shared_competence() -> None:
    rng = np.random.default_rng(2)
    c = rng.normal(size=200)
    M = np.column_stack([c + 0.2 * rng.normal(size=200) for _ in range(4)])
    assert hy.kendall_w(M) > 0.8 and hy.residual_rank_w(M, c) < 0.35


def fake_rows(n_config: int = 27, seeds: int = 3) -> list[dict]:
    archs, scales, dists = ["transformer", "lstm", "mamba"], ["small", "medium", "large"], ["a", "b", "c"]
    keys = [(a, s, d) for a in archs for s in scales for d in dists][:n_config]
    return [{"arch": a, "scale": s, "distribution": d, "seed": k} for (a, s, d) in keys for k in range(seeds)]


def test_cluster_bootstrap_keeps_seeds_together() -> None:
    """Amendment 4: every resample is made of whole configurations (all 3 seeds of each)."""
    rows = fake_rows()
    groups = {tuple(g.tolist()) for g in hy.clusters(rows)}
    assert len(groups) == 27 and all(len(g) == 3 for g in groups)
    idx = hy.boot_indices(rows, n_boot=50)
    assert len(idx) == 50
    for i in idx:
        assert len(i) == 81
        for k in range(0, 81, 3):
            assert tuple(i[k:k + 3].tolist()) in groups
    assert hy.boot_indices(rows, n_boot=5)[0].tolist() == hy.boot_indices(rows, n_boot=5)[0].tolist()


def test_cluster_intervals_are_wider_for_near_duplicate_seeds() -> None:
    """When seeds of a configuration are near copies, resampling models treats 81 points as
    independent and gives too narrow an interval; the configuration bootstrap does not."""
    rows = fake_rows()
    rng = np.random.default_rng(3)
    base_x, base_y = rng.normal(size=27), rng.normal(size=27)
    base_y = 0.4 * base_x + base_y
    x = np.repeat(base_x, 3) + 0.01 * rng.normal(size=81)
    y = np.repeat(base_y, 3) + 0.01 * rng.normal(size=81)
    fn = lambda i: hy.kendall_tau(x[i], y[i])
    cl = hy.interval(hy.boot(fn, hy.boot_indices(rows)))
    model_level = [np.random.default_rng(0).integers(0, 81, 81) for _ in range(hy.N_BOOT)]
    ml = hy.interval(hy.boot(fn, model_level))
    assert (cl[1] - cl[0]) > 1.4 * (ml[1] - ml[0])


# --------------------------------------------------------------------------
# A fake zoo in the real file formats
# --------------------------------------------------------------------------

def make_zoo(tmp: Path, scores_fn, frag_fn, competence_fn, labels_fn=None) -> Path:
    """Copy models.csv; write the registry, D1 to D4 results and fragility v1 and v2 for every
    Othello model. Each *_fn takes (i, row, rng) and returns the value(s)."""
    root = tmp / "repo"
    (root / "src/docs").mkdir(parents=True)
    shutil.copy(ROOT / "src/docs/models.csv", root / "src/docs/models.csv")
    with open(ROOT / "src/docs/models.csv", newline="") as f:
        models = [r for r in csv.DictReader(f) if r["domain"] == "othello"]
    rng = np.random.default_rng(0)
    reg_rows = []
    folders = {"D1": "D1_next_token", "D2": "D2_state_equiv", "D3": "D3_inductive_bias", "D4": "D4_probing"}
    for i, m in enumerate(models):
        comp = competence_fn(i, m, rng)
        reg_rows.append({"model_id": m["model_id"], "val_legal_rate": comp, "admitted": "yes" if comp >= 0.5 else "no"})
        sc = scores_fn(i, m, rng, comp)
        for d, folder in folders.items():
            det = {}
            if d == "D4":
                lab = labels_fn(i, m, rng) if labels_fn else "decodable but not causal"
                det = {"causal": {"causal_label": lab, "causal_score": 0.1}}
            if d == "D3":
                det = {"symbolic_regression": {"occupancy_shortcut_rate_pretrained": float(rng.random())}}
            p = root / "results" / folder / f"{m['model_id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"model_id": m["model_id"], "diagnostic": d, "status": "ok",
                                     "score": float(sc[d]), "details": det}))
        fr = frag_fn(i, m, rng, sc)
        for folder in ("fragility", "fragility_v2"):
            p = root / "results" / folder / f"{m['model_id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"model_id": m["model_id"], "fragility": float(fr),
                                     "in_distribution_legal_rate": comp}))
    with open(root / "src/docs/zoo_registry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "val_legal_rate", "admitted"])
        w.writeheader()
        w.writerows(reg_rows)
    return root


def comp_spread(i, m, rng):
    return float(np.clip(0.55 + 0.42 * rng.random(), 0.5, 0.99))


def test_agreement_through_competence_and_a5(tmp_path: Path) -> None:
    """Every diagnostic follows competence plus independent noise; D1 is competence itself. Raw
    taus agree, partial taus disagree, so every D2 to D4 pair is agreement through competence,
    and D1 pairs are marked not interpretable (A5)."""
    def scores(i, m, rng, c):
        return {"D1": c, **{d: c + 0.03 * rng.normal() for d in ("D2", "D3", "D4")}}
    root = make_zoo(tmp_path, scores, lambda i, m, rng, s: rng.normal(), comp_spread)
    s = hy.analyse("othello", root)
    h = s["H1"]
    assert s["counts"]["all_four_scores"] == 81 and h["a5_d1_not_interpretable"]
    for q in h["pairs"]:
        assert q["category"] == "agree"
        if "D1" in q["pair"]:
            assert q["partial_category"] == "not interpretable (A5)"
        else:
            assert q["partial_category"] == "disagree" and q["agreement_through_competence"]
    assert h["kendall_w"] > 0.8 and h["kendall_w_residual"] < 0.4


def test_h3_finds_the_planted_predictor(tmp_path: Path) -> None:
    """Relative fragility depends on D3 only (negatively): D3 is the best predictor and adds to
    D1; D2 and D4 do not."""
    def scores(i, m, rng, c):
        return {d: rng.random() for d in ("D1", "D2", "D3", "D4")}
    def frag(i, m, rng, s):
        return (-s["D3"] + 0.1 * rng.normal()) * 0.8
    root = make_zoo(tmp_path, scores, frag, lambda i, m, rng: 0.8)
    s = hy.analyse("othello", root)
    h3 = s["H3"]
    assert h3["best_predictor"] == "D3" and h3["D3"]["adds_to_D1"]
    assert not h3["D2"]["adds_to_D1"] and not h3["D4"]["adds_to_D1"]
    assert h3["D3"]["regression_ci"][1] < 0 and h3["n_configurations"] == 27
    without = s["H3_without_test_model"]
    assert without["n_models"] == 80 and without["best_predictor"] == "D3"
    fd = s["fragility_distribution"]["rel_frag_v2"]
    assert fd["n"] == 81 and set(fd["mean_by_arch"]) == {"transformer", "lstm", "mamba"}
    assert s["bootstrap"]["n_configurations"] == 27
    assert s["H3_secondary"]["rel_frag_v2"]["best_predictor_section4"] == "D3"


def test_h3_no_winner_when_nothing_predicts(tmp_path: Path) -> None:
    def scores(i, m, rng, c):
        return {d: rng.random() for d in ("D1", "D2", "D3", "D4")}
    root = make_zoo(tmp_path, scores, lambda i, m, rng, s: rng.normal(), lambda i, m, rng: 0.8)
    assert hy.analyse("othello", root)["H3"]["best_predictor"] == "no single diagnostic wins"


def test_h2_systematic_by_architecture(tmp_path: Path) -> None:
    """LSTM models: D2 and D3 rank them in reverse of D1 and D4; elsewhere all four agree. The
    architecture factor must be significant and larger than the seed noise floor."""
    def scores(i, m, rng, c):
        base = rng.random()
        flip = 1 - base if m["arch"] == "lstm" else base
        return {"D1": base, "D2": flip + 0.01 * rng.normal(), "D3": flip + 0.01 * rng.normal(), "D4": base}
    root = make_zoo(tmp_path, scores, lambda i, m, rng, s: rng.normal(), comp_spread)
    h2 = hy.analyse("othello", root)["H2"]
    arch = next(t for t in h2["factors"] if t["factor"] == "arch")
    assert arch["p_holm"] < 0.05 and arch["systematic"] and h2["verdict"] == "systematic"
    assert h2["n_configs_with_seeds"] == 27


def test_causal_flag_and_outputs(tmp_path: Path) -> None:
    """The flag adds to D4: causal models are less fragile at the same D4. Also checks that every
    output file and figure is written, and that missing models are counted, not dropped silently."""
    def labels(i, m, rng):
        return "decodable and causal" if i % 3 == 0 else "decodable but not causal"
    def scores(i, m, rng, c):
        return {d: rng.random() for d in ("D1", "D2", "D3", "D4")}
    def frag(i, m, rng, s):
        return -0.5 * (i % 3 == 0) + 0.05 * rng.normal()
    root = make_zoo(tmp_path, scores, frag, comp_spread, labels)
    for d in ("D2_state_equiv/othello_lstm_large_mixed_s2.json",):
        (root / "results" / d).unlink()
    s = hy.analyse("othello", root)
    assert s["counts"]["all_four_scores"] == 80 and s["missing_scores"]["D2"] == ["othello_lstm_large_mixed_s2"]
    c = s["H3_causal_and_occupancy"]
    assert c["flag_beyond_D4"]["adds_to_D4"] and c["flag_beyond_D4"]["regression_ci"][1] < 0
    assert "D3_vs_occupancy_shortcut" in c
    paths = hy.write_outputs(s, tmp_path / "out", tmp_path / "fig")
    names = {p.name for p in paths}
    assert {"summary.json", "model_table.csv", "h1_pairs.csv", "h3.csv", "h1_concordance.png",
            "a3_competence.png", "h3_scatter.png", "h2_disagreement.png", "fragility_distribution.png"} <= names
    json.loads((tmp_path / "out" / "summary.json").read_text())   # valid JSON, no NaN


def test_a3_sensitivity_threshold(tmp_path: Path) -> None:
    def scores(i, m, rng, c):
        return {d: c + 0.05 * rng.normal() for d in ("D1", "D2", "D3", "D4")}
    few = make_zoo(tmp_path / "a", scores, lambda i, m, rng, s: rng.normal(), lambda i, m, rng: 0.7 + 0.25 * (i < 5))
    assert isinstance(hy.analyse("othello", few)["A3"]["sensitivity_h1"], str)
    many = make_zoo(tmp_path / "b", scores, lambda i, m, rng, s: rng.normal(), lambda i, m, rng: 0.7 + 0.25 * (i < 20))
    assert hy.analyse("othello", many)["A3"]["sensitivity_h1"]["n_models"] == 20
