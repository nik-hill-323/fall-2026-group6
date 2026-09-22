"""Generate the fixed run matrix: every model to train and every diagnostic
run to perform. Two files:

    src/docs/models.csv      one row per model      (243 rows)
    src/docs/run_matrix.csv  one row per evaluation (972 rows, 4 per model)

Scope as agreed at the Sep 22 session:
    domains        othello, maps, interpreter
    architectures  transformer, lstm, mamba
    scales         small, medium, large
    distributions  3 per domain (see DISTRIBUTIONS)
    seeds          0, 1, 2
    diagnostics    D1, D2, D3, D4

Naming (used for configs/, checkpoints/, results/):
    model_id = {domain}_{arch}_{scale}_{distribution}_s{seed}
    run_id   = {model_id}_{diagnostic}

Run from the repo root:

    python src/component/run_matrix.py

Once frozen, rows are only ever marked done or skipped, never added.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

DOMAINS: list[str] = ["othello", "maps", "interpreter"]
ARCHS: list[str] = ["transformer", "lstm", "mamba"]
SCALES: list[str] = ["small", "medium", "large"]
SEEDS: list[int] = [0, 1, 2]
DIAGNOSTICS: list[str] = ["D1", "D2", "D3", "D4"]

# Three training distributions per domain. Othello and interpreter are
# tentative until their generators exist.
DISTRIBUTIONS: dict[str, list[str]] = {
    "maps": ["shortest_paths", "noisy_shortest_paths", "random_walks"],
    "othello": ["synthetic", "championship", "mixed"],
    "interpreter": ["random", "structured_short", "structured_long"],
}

# H4 hold-outs. Not decided yet; these are the suggested values from the
# handoff and are flagged in the held_out column so they can be filtered.
HELD_OUT_DOMAIN = "interpreter"
HELD_OUT_ARCH = "mamba"

DIAG_NAMES = {
    "D1": "next_token",
    "D2": "state_equiv",
    "D3": "inductive_bias",
    "D4": "probing",
}


def model_rows() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for arch, scale, dist, seed in itertools.product(ARCHS, SCALES, DISTRIBUTIONS[domain], SEEDS):
            model_id = f"{domain}_{arch}_{scale}_{dist}_s{seed}"
            rows.append(
                {
                    "model_id": model_id,
                    "domain": domain,
                    "arch": arch,
                    "scale": scale,
                    "distribution": dist,
                    "seed": seed,
                    "held_out": "yes" if (domain == HELD_OUT_DOMAIN or arch == HELD_OUT_ARCH) else "no",
                    "config": f"configs/runs/{model_id}.yaml",
                    "checkpoint": f"checkpoints/{model_id}/model.pt",
                    "status": "planned",
                    "notes": "",
                }
            )
    return rows


def eval_rows(models: list[dict]) -> list[dict]:
    rows = []
    for m in models:
        for d in DIAGNOSTICS:
            rows.append(
                {
                    "run_id": f"{m['model_id']}_{d}",
                    "model_id": m["model_id"],
                    "diagnostic": d,
                    "domain": m["domain"],
                    "arch": m["arch"],
                    "scale": m["scale"],
                    "distribution": m["distribution"],
                    "seed": m["seed"],
                    "held_out": m["held_out"],
                    "result": f"results/{d}_{DIAG_NAMES[d]}/{m['model_id']}.json",
                    "status": "planned",
                    "notes": "",
                }
            )
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    docs = Path(__file__).resolve().parents[1] / "docs"
    models = model_rows()
    evals = eval_rows(models)
    write(docs / "models.csv", models)
    write(docs / "run_matrix.csv", evals)

    per_domain = {d: sum(1 for m in models if m["domain"] == d) for d in DOMAINS}
    dev = sum(1 for m in models if m["held_out"] == "no")
    print(f"models.csv     : {len(models)} models  {per_domain}")
    print(f"run_matrix.csv : {len(evals)} evaluations ({len(DIAGNOSTICS)} per model)")
    print(f"development    : {dev} models   held-out (H4): {len(models) - dev}")


if __name__ == "__main__":
    main()
