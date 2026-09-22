"""Generate src/docs/run_matrix.csv, the fixed list of every model to train and
every (model x diagnostic) evaluation to run.

Edit the AXES block, then run from the repo root:

    python src/component/run_matrix.py

The file it writes is the contract for the whole project. Once Week 1 closes,
rows are only ever marked done or skipped, never added.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

# --------------------------------------------------------------------------- #
# AXES
# --------------------------------------------------------------------------- #

DOMAINS: list[str] = ["navigation", "lattice", "othello", "connect4", "interpreter", "chemistry"]
ARCHS: list[str] = ["transformer", "ssm", "lstm"]
SCALES: list[str] = ["small", "medium", "large"]
DISTS: list[str] = ["expert", "noisy", "random"]
SEEDS: list[int] = [0, 1, 2]

# Tier 4 hold-outs: kept out of every phase until Week 14.
HELD_OUT_DOMAIN = "chemistry"
HELD_OUT_ARCH = "lstm"

# What "expert / noisy / random" means in each domain.
DIST_MEANING: dict[str, dict[str, str]] = {
    "navigation": {"expert": "shortest paths", "noisy": "noisy shortest paths", "random": "random walks"},
    "lattice": {"expert": "shortest paths", "noisy": "noisy shortest paths", "random": "random walks"},
    "othello": {"expert": "championship games", "noisy": "championship + random moves", "random": "random legal games"},
    "connect4": {"expert": "minimax play", "noisy": "minimax + random moves", "random": "random legal games"},
    "interpreter": {"expert": "structured programs", "noisy": "structured + random ops", "random": "random valid ops"},
    "chemistry": {"expert": "USPTO / ORD routes", "noisy": "routes + random steps", "random": "random valid steps"},
}

DIAGNOSTICS: list[str] = ["d1_next_token", "d2_myhill_nerode", "d3_inductive_bias", "d4_probing", "fragility"]

# Training priority. Tier 1 trains first; if the Week 5 gate trips, tier 3 is
# dropped first, then tier 2 (this is the "cut scale first, then seeds" rule).
def tier(scale: str, seed: int) -> int:
    if scale == "medium" and seed == 0:
        return 1
    if scale in ("small", "medium") and seed in (0, 1):
        return 2
    return 3


# --------------------------------------------------------------------------- #
# GENERATION
# --------------------------------------------------------------------------- #

def model_rows() -> list[dict]:
    rows = []
    for domain, arch, scale, dist, seed in itertools.product(DOMAINS, ARCHS, SCALES, DISTS, SEEDS):
        held_out = domain == HELD_OUT_DOMAIN or arch == HELD_OUT_ARCH
        rows.append(
            {
                "run_id": f"{domain}-{arch}-{scale}-{dist}-s{seed}",
                "kind": "train",
                "domain": domain,
                "arch": arch,
                "scale": scale,
                "train_dist": dist,
                "train_dist_meaning": DIST_MEANING[domain][dist],
                "seed": seed,
                "diagnostic": "",
                "held_out": "yes" if held_out else "no",
                "tier": 4 if held_out else tier(scale, seed),
                "status": "planned",
                "notes": "",
            }
        )
    return rows


def eval_rows(models: list[dict]) -> list[dict]:
    rows = []
    for m in models:
        for diag in DIAGNOSTICS:
            r = dict(m)
            r["run_id"] = f"{m['run_id']}::{diag}"
            r["kind"] = "eval"
            r["diagnostic"] = diag
            rows.append(r)
    return rows


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "run_matrix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    models = model_rows()
    evals = eval_rows(models)
    rows = models + evals

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dev = [m for m in models if m["held_out"] == "no"]
    by_tier = {t: sum(1 for m in dev if m["tier"] == t) for t in (1, 2, 3)}
    print(f"wrote {out}")
    print(f"models total: {len(models)}  (development: {len(dev)}, held-out: {len(models) - len(dev)})")
    print(f"development models by tier: {by_tier}")
    print(f"evaluations: {len(evals)}")


if __name__ == "__main__":
    main()
