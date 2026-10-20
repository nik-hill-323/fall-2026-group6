"""Vector (PDF) figures for the LaTeX report and the paper, made from the committed Othello results.

Reads only files written by the analysis (no model is run):
    results/analysis/othello/h1_pairs.csv, summary.json, model_table.csv, exploratory.json
Writes, into reports/Latex_report/fig/:
    othello_h1.pdf          Kendall tau and partial tau (competence controlled) for every pair (H1)
    othello_h3_d4.pdf       D4 against relative fragility v2, by training distribution (H3 and the
                            exploratory check)
    othello_shifts.pdf      mean drop in legal move rate per fragility v2 shift, by training distribution
                            (exploratory)

Run from the repo root:
    python reports/Latex_report/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "analysis" / "othello"
OUT = ROOT / "reports" / "Latex_report" / "fig"
DIAGS = ["D1", "D2", "D3", "D4"]
DIST_COLOR = {"synthetic": "#033C5A", "championship": "#C0392B", "mixed": "#9AA5AE"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def h1_figure() -> Path:
    rows = list(csv.DictReader(open(RES / "h1_pairs.csv")))
    raw, part = np.eye(4), np.eye(4)
    for r in rows:
        a, b = (DIAGS.index(x) for x in r["pair"].split("-"))
        raw[a, b] = raw[b, a] = float(r["tau"])
        part[a, b] = part[b, a] = float(r["partial_tau"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, m, title in [(axes[0], raw, "Kendall tau"), (axes[1], part, "Partial tau, competence controlled")]:
        im = ax.imshow(m, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(4), DIAGS)
        ax.set_yticks(range(4), DIAGS)
        ax.set_title(title, fontsize=9)
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(m[i, j]) > 0.6 else "black")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="rank correlation")
    p = OUT / "othello_h1.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def h3_figure() -> Path:
    rows = list(csv.DictReader(open(RES / "model_table.csv")))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for d, c in DIST_COLOR.items():
        x = [float(r["D4"]) for r in rows if r["distribution"] == d]
        y = [float(r["rel_frag_v2"]) for r in rows if r["distribution"] == d]
        ax.scatter(x, y, s=22, color=c, edgecolor="white", linewidth=0.5, label=d)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("D4 score (linear probe selectivity)")
    ax.set_ylabel("relative fragility v2 (higher breaks more)")
    ax.legend(frameon=False, fontsize=8)
    p = OUT / "othello_h3_d4.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def shifts_figure() -> Path:
    e = json.load(open(RES / "exploratory.json"))["fragility_v2_mean_drop_by_training_distribution"]

    def other(d):
        vals = [v for k, v in e[d].items() if k.startswith("cross_distribution")]
        return float(np.mean(vals))

    shifts = [("other\ndistribution", other), ("rotate 180", "symmetry/rot180"), ("transpose", "symmetry/transpose"),
              ("anti\ntranspose", "symmetry/anti_transpose"), ("after pass", "after_pass"),
              ("detour\nrandom 0.5", "detour/random/p=0.5"), ("detour\nadvers. 0.5", "detour/adversarial/p=0.5")]
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    w = 0.26
    for i, (d, c) in enumerate(DIST_COLOR.items()):
        vals = [k(d) if callable(k) else e[d][k] for _, k in shifts]
        ax.bar(np.arange(len(shifts)) + (i - 1) * w, vals, w, color=c, label=d)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(shifts)), [s for s, _ in shifts], fontsize=8)
    ax.set_ylabel("mean drop in legal move rate")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    p = OUT / "othello_shifts.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (h1_figure(), h3_figure(), shifts_figure()):
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
