"""Generate the fixed run matrix: every model to train and every diagnostic
run to perform. Two files:

    src/docs/models.csv      one row per model      (243 rows)
    src/docs/run_matrix.csv  one row per evaluation (972 rows, 4 per model)

Scope as agreed at the Sep 22 session:
    domains        othello, maps, interpreter
    architectures  transformer, lstm, mamba
    scales         small, medium, large   (defined in SCALE_CONFIG below)
    distributions  3 per domain (see DISTRIBUTIONS)
    seeds          0, 1, 2
    diagnostics    D1, D2, D3, D4

One training budget for every model (TRAINING_BUDGET below), so that scale is
the only thing that changes between small, medium and large.

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

# What small / medium / large mean. d_model is the residual width; the LSTM
# and Mamba variants use the same n_layers and d_model so that parameter
# counts are comparable across architectures at each scale.
# "large" matches Othello-GPT (Li et al. 2023): 8 layers, 512 wide, 8 heads.
SCALE_CONFIG: dict[str, dict[str, int]] = {
    "small":  {"n_layers": 4, "d_model": 128, "n_heads": 4},
    "medium": {"n_layers": 6, "d_model": 256, "n_heads": 8},
    "large":  {"n_layers": 8, "d_model": 512, "n_heads": 8},
}

# One budget for every model in the zoo. The training-budget axis was cut on
# Sep 22, so this is fixed, not varied. Throughput numbers in PR #19 are
# measured against this budget.
TRAINING_BUDGET: dict[str, float] = {
    "steps": 5000,          # optimizer steps
    "batch_games": 256,     # games per step; each game is up to 59 tokens
    "lr": 3e-4,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "n_train_games": 200_000,
    "seq_len": 59,
}

# Three training distributions per domain. Othello and interpreter are
# tentative until their generators exist.
DISTRIBUTIONS: dict[str, list[str]] = {
    "maps": ["shortest_paths", "noisy_shortest_paths", "random_walks"],
    "othello": ["synthetic", "championship", "mixed"],
    "interpreter": ["random", "structured_short", "structured_long"],
}

# H4 hold-outs: one domain and one architecture kept out of development
# until Week 14. NOT decided yet. Until the team picks them, every row is
# marked "tbd" so nothing is accidentally excluded. Set these two and
# regenerate once the choice is made.
HELD_OUT_DOMAIN: str | None = None
HELD_OUT_ARCH: str | None = None

DIAG_NAMES = {
    "D1": "next_token",
    "D2": "state_equiv",
    "D3": "inductive_bias",
    "D4": "probing",
}


def held_out_flag(domain: str, arch: str) -> str:
    if HELD_OUT_DOMAIN is None or HELD_OUT_ARCH is None:
        return "tbd"
    return "yes" if (domain == HELD_OUT_DOMAIN or arch == HELD_OUT_ARCH) else "no"


def model_rows() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for arch, scale, dist, seed in itertools.product(ARCHS, SCALES, DISTRIBUTIONS[domain], SEEDS):
            model_id = f"{domain}_{arch}_{scale}_{dist}_s{seed}"
            cfg = SCALE_CONFIG[scale]
            rows.append(
                {
                    "model_id": model_id,
                    "domain": domain,
                    "arch": arch,
                    "scale": scale,
                    "n_layers": cfg["n_layers"],
                    "d_model": cfg["d_model"],
                    "distribution": dist,
                    "seed": seed,
                    "held_out": held_out_flag(domain, arch),
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
    per_scale = {s: sum(1 for m in models if m["scale"] == s) for s in SCALES}
    flags = {f: sum(1 for m in models if m["held_out"] == f) for f in ("no", "yes", "tbd")}
    print(f"models.csv     : {len(models)} models  by domain {per_domain}  by scale {per_scale}")
    print(f"run_matrix.csv : {len(evals)} evaluations ({len(DIAGNOSTICS)} per model)")
    print(f"held_out       : {flags}")
    print(f"budget         : {TRAINING_BUDGET['steps']} steps x {TRAINING_BUDGET['batch_games']} games per model")


if __name__ == "__main__":
    main()
