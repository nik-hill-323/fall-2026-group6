"""Exploratory analyses of the Othello zoo. NOT preregistered.

Everything here was decided after the preregistered analysis (src/analysis/hypotheses.py, tag
prereg-v1.4) had been run and read. It is reported as exploratory and never replaces a preregistered
result. Intervals use the same configuration bootstrap as Amendment 4 (27 configurations, 2,000
resamples, seed 0).

What it computes
    descriptives     mean competence, D1 to D4, relative fragility v2 and the number of "decodable and
                     causal" models, per architecture x scale, per architecture, per scale and per
                     training distribution
    fragility_parts  mean drop of every fragility v2 variant, per training distribution (which shifts
                     make which models fragile)
    h3_by_distribution  the H3 statistic of A2 (partial Spearman of D2, D3, D4 with relative fragility
                     v2 given D1), within each training distribution, and over all models with the
                     training distribution added as a control (the preregistered A2 controls only for
                     D1 in the partial correlation and for architecture and scale in the regression)
    diagnostic_details  D2 compression and distinction errors; D3 balanced accuracies, symbolic recovery
                     and occupancy shortcut rates; D4 linear vs MLP selectivity, best layer depth, gain
                     over the occupancy only baseline, causal scores

Run (repo root, after hypotheses.py):
    python src/analysis/exploratory_othello.py
Writes results/analysis/othello/exploratory.json and results/analysis/othello/descriptives.csv.
"""

from __future__ import annotations

import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("hypotheses", ROOT / "src/analysis/hypotheses.py")
hy = importlib.util.module_from_spec(_spec)
sys.modules["hypotheses"] = hy
_spec.loader.exec_module(hy)

OUT = ROOT / "results" / "analysis" / "othello"
COLS = ["competence", "D1", "D2", "D3", "D4", "rel_frag_v2"]
DISTS = ["synthetic", "championship", "mixed"]


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def descriptives(rows):
    groups = {"arch_scale": lambda r: f"{r['arch']}/{r['scale']}", "arch": lambda r: r["arch"],
              "scale": lambda r: r["scale"], "distribution": lambda r: r["distribution"]}
    out = {}
    table = []
    for name, key in groups.items():
        g = collections.defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        out[name] = {}
        for level, rs in sorted(g.items()):
            entry = {c: _mean(rs, c) for c in COLS}
            entry["n"] = len(rs)
            entry["n_decodable_and_causal"] = sum(r["causal_label"] == hy.CAUSAL for r in rs)
            out[name][level] = entry
            table.append({"grouping": name, "level": level, **entry})
    return out, table


def fragility_parts(rows):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        f = json.loads((ROOT / "results" / "fragility_v2" / f"{r['model_id']}.json").read_text())
        for k, v in f["drops"].items():
            agg[r["distribution"]][k].append(v)
    return {d: {k: float(np.mean(v)) for k, v in sorted(agg[d].items())} for d in DISTS if d in agg}


def _partial_given(x, y, controls):
    ex, ey = hy.residual(hy.ranks(x), controls), hy.residual(hy.ranks(y), controls)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def h3_by_distribution(rows):
    out = {"within_distribution": {}, "controlling_distribution": {}}
    for d in DISTS:
        rs = [r for r in rows if r["distribution"] == d]
        y, d1 = hy.column(rs, "rel_frag_v2"), hy.column(rs, "D1")
        idx = hy.boot_indices(rs)
        entry = {"n_models": len(rs), "n_configurations": len(hy.clusters(rs)),
                 "D1_spearman": hy.spearman(d1, y)}
        for name in ("D2", "D3", "D4"):
            x = hy.column(rs, name)
            entry[name] = {"partial_given_D1": hy.partial_spearman(x, y, d1),
                           "ci": hy.interval(hy.boot(lambda i: hy.partial_spearman(x[i], y[i], d1[i]), idx)),
                           "raw_spearman": hy.spearman(x, y)}
        out["within_distribution"][d] = entry
    y, d1 = hy.column(rows, "rel_frag_v2"), hy.column(rows, "D1")
    dist = np.array([r["distribution"] for r in rows])
    idx = hy.boot_indices(rows)

    def ctrl(i):
        return np.column_stack([hy.ranks(d1[i]), hy.design([dist[i]])])

    for name in ("D2", "D3", "D4"):
        x = hy.column(rows, name)
        out["controlling_distribution"][name] = {
            "partial_given_D1_and_distribution": _partial_given(x, y, ctrl(np.arange(len(rows)))),
            "ci": hy.interval(hy.boot(lambda i: _partial_given(x[i], y[i], ctrl(i)), idx))}
    return out


def diagnostic_details(rows):
    det = collections.defaultdict(list)
    for r in rows:
        mid = r["model_id"]
        d2 = json.loads((ROOT / "results/D2_state_equiv" / f"{mid}.json").read_text())["details"]
        d3 = json.loads((ROOT / "results/D3_inductive_bias" / f"{mid}.json").read_text())["details"]
        d4 = json.loads((ROOT / "results/D4_probing" / f"{mid}.json").read_text())["details"]
        sr = d3["symbolic_regression"]
        b = d4["best_layer"]
        det["D2_compression_error"].append(d2["compression_error"])
        det["D2_distinction_error"].append(d2["distinction_error"])
        det["D3_pretrained_balanced_accuracy"].append(d3["pretrained_balanced_accuracy"])
        det["D3_random_balanced_accuracy"].append(d3["random_balanced_accuracy"])
        det["D3_law_recovered_pretrained"].append(sr["recovery_rate_pretrained"])
        det["D3_law_recovered_random"].append(sr["recovery_rate_random"])
        det["D3_occupancy_shortcut_pretrained"].append(sr["occupancy_shortcut_rate_pretrained"])
        det["D3_occupancy_shortcut_random"].append(sr["occupancy_shortcut_rate_random"])
        det["D4_linear_selectivity"].append(r["D4"])
        det["D4_mlp_selectivity"].append(d4["mlp"]["selectivity_at_best"])
        det["D4_best_layer_depth"].append(d4["best_layer_depth"])
        det["D4_probe_minus_occupancy_baseline"].append(d4["probe_acc_by_layer"][b] - d4["occupancy_baseline_acc"])
        det["D4_causal_score"].append(d4["causal"]["causal_score"])
    return {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in det.items()}


def main():
    rows = [r for r in hy.load_table("othello") if r["admitted"] and r["rel_frag_v2"] is not None
            and all(r[d] is not None for d in hy.DIAGS)]
    desc, table = descriptives(rows)
    out = {"note": "Exploratory, not preregistered; decided after the preregistered analysis was read.",
           "n_models": len(rows), "descriptives": desc, "fragility_v2_mean_drop_by_training_distribution":
           fragility_parts(rows), "h3_by_distribution": h3_by_distribution(rows),
           "diagnostic_details": diagnostic_details(rows),
           "decodable_and_causal_models": [r["model_id"] for r in rows if r["causal_label"] == hy.CAUSAL]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exploratory.json").write_text(json.dumps(hy._clean(out), indent=2) + "\n")
    with open(OUT / "descriptives.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grouping", "level", "n", *COLS, "n_decodable_and_causal"])
        w.writeheader()
        for t in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
    h = out["h3_by_distribution"]
    print("exploratory, not preregistered")
    for d, e in h["within_distribution"].items():
        print(f"  within {d:12s} D4 partial given D1 {e['D4']['partial_given_D1']:+.3f} {e['D4']['ci']}")
    c = h["controlling_distribution"]["D4"]
    print(f"  D4 partial given D1 and distribution {c['partial_given_D1_and_distribution']:+.3f} {c['ci']}")
    print("wrote", (OUT / "exploratory.json").relative_to(ROOT), "and", (OUT / "descriptives.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
