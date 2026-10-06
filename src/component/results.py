"""Results schema: one format for every diagnostic result, and the master table.

Every diagnostic (D1 to D4) writes one JSON file per model in this format, so the master
table for H1 to H3 is a single join. The format is fixed here, before any diagnostic module
is written; changing it later would mean changing all four.

The one rule every diagnostic must follow
    `score` is the single number that goes into the master table, and HIGHER ALWAYS MEANS A
    MORE WORLD MODEL LIKE MODEL. A diagnostic whose natural number is an error (D2 compression
    and distinction error) stores 1 minus the error, or another higher is better form, and
    says how in `method`. Raw numbers (per layer accuracies, both D2 errors, control task
    accuracy, ...) go in `details`. Without this rule, rank correlations between diagnostics
    would have the wrong sign.

Where results go (paths come from src/component/run_matrix.py)
    results/D1_next_token/{model_id}.json
    results/D2_state_equiv/{model_id}.json
    results/D3_inductive_bias/{model_id}.json
    results/D4_probing/{model_id}.json

Typical use inside a diagnostic:
    res = make_result("othello_transformer_small_synthetic_s0", "D4",
                      method="linear probe, mine/theirs, best layer selectivity",
                      split="test", n_games=1000, data_seed=0,
                      score=0.31, details={"selectivity_by_layer": [...]})
    save(res)

Then for the analysis:
    rows = master_table()            # one row per model, one score column per diagnostic
    write_master_table("results/master_table.csv")
"""


import csv
import datetime as _dt
import importlib.util
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1

_spec = importlib.util.spec_from_file_location("run_matrix", ROOT / "src/component/run_matrix.py")
matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matrix)

DIAGNOSTICS: list[str] = list(matrix.DIAGNOSTICS)          # D1, D2, D3, D4
SPLITS = ("train", "val", "test")
STATUSES = ("ok", "failed", "skipped")
MODELS_CSV = ROOT / "src/docs/models.csv"
RESULTS_DIR = ROOT / "results"
MODEL_FIELDS = ("domain", "arch", "scale", "distribution", "seed")


def diag_folder(diagnostic: str) -> str:
    """D1 -> D1_next_token, the folder name used in run_matrix.csv."""
    return f"{diagnostic}_{matrix.DIAG_NAMES[diagnostic]}"


def result_path(model_id: str, diagnostic: str, results_dir: Path | str = RESULTS_DIR) -> Path:
    return Path(results_dir) / diag_folder(diagnostic) / f"{model_id}.json"


# ==========================================================================
# The result record
# ==========================================================================

@dataclass
class Result:
    """One diagnostic run on one model. Field order is the order in the JSON file."""

    model_id: str
    diagnostic: str                      # D1, D2, D3, D4
    method: str                          # short description: what the score is and how it is oriented
    split: str                           # data split scored on: train, val or test
    n_games: int                         # games (sequences) used
    data_seed: int
    score: float | None                  # higher = more world model like; None only when status is not ok
    score_std: float | None = None       # spread of the score (seeds, bootstrap, folds), if known
    score_ci: list[float] | None = None  # [low, high] interval, if known
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"                   # ok, failed, skipped
    reason: str = ""                     # required when status is failed or skipped
    domain: str = ""
    arch: str = ""
    scale: str = ""
    distribution: str = ""
    seed: int | None = None
    run_id: str = ""
    schema_version: int = SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance(runtime_sec: float | None = None) -> dict[str, Any]:
    """Where and how a result was produced, so any number traces back to code and machine."""
    info: dict[str, Any] = {
        "git_commit": _git_commit(),
        "time_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "host": platform.node(),
        "runtime_sec": runtime_sec,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["torch"] = None
        info["gpu"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def load_models(models_csv: Path | str = MODELS_CSV) -> dict[str, dict[str, str]]:
    """models.csv as {model_id: row}."""
    with open(models_csv, newline="") as f:
        return {row["model_id"]: row for row in csv.DictReader(f)}


def make_result(model_id: str, diagnostic: str, *, method: str, split: str, n_games: int,
                data_seed: int, score: float | None, runtime_sec: float | None = None,
                models_csv: Path | str = MODELS_CSV, **kwargs: Any) -> Result:
    """Build a Result. Model fields (domain, arch, scale, distribution, seed) are filled from
    models.csv. A model that is not in the zoo (for example the reference Othello GPT) must
    pass them as keyword arguments."""
    res = Result(model_id=model_id, diagnostic=diagnostic, method=method, split=split,
                 n_games=n_games, data_seed=data_seed, score=score, **kwargs)
    models = load_models(models_csv)
    if model_id in models:
        row = models[model_id]
        res.domain, res.arch, res.scale, res.distribution = (row["domain"], row["arch"],
                                                             row["scale"], row["distribution"])
        res.seed = int(row["seed"])
    res.run_id = f"{model_id}_{diagnostic}"
    if not res.provenance:
        res.provenance = provenance(runtime_sec)
    validate(res)
    return res


# ==========================================================================
# Validation
# ==========================================================================

class SchemaError(ValueError):
    """A result that does not follow the schema. Raised before anything is written."""


def validate(res: Result) -> None:
    problems: list[str] = []
    if not res.model_id:
        problems.append("model_id is empty")
    if res.diagnostic not in DIAGNOSTICS:
        problems.append(f"diagnostic must be one of {DIAGNOSTICS}, not {res.diagnostic!r}")
    if not res.method.strip():
        problems.append("method is empty: say what the score is and how it is oriented")
    if res.split not in SPLITS:
        problems.append(f"split must be one of {SPLITS}, not {res.split!r}")
    if res.status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}, not {res.status!r}")
    if not isinstance(res.n_games, int) or res.n_games < 0:
        problems.append("n_games must be a non negative int")
    if res.status == "ok":
        if res.score is None or not isinstance(res.score, (int, float)) or not math.isfinite(res.score):
            problems.append("status ok needs a finite score")
    else:
        if not res.reason.strip():
            problems.append(f"status {res.status} needs a reason")
    if res.score_ci is not None:
        if len(res.score_ci) != 2 or res.score_ci[0] > res.score_ci[1]:
            problems.append("score_ci must be [low, high] with low <= high")
    for name in MODEL_FIELDS:
        value = getattr(res, name)
        if value is None or value == "":
            problems.append(f"{name} is empty (fill it from models.csv or pass it)")
    if res.run_id != f"{res.model_id}_{res.diagnostic}":
        problems.append("run_id must be {model_id}_{diagnostic}")
    try:
        json.dumps(res.details)
    except (TypeError, ValueError):
        problems.append("details must be JSON serialisable (convert numpy values with float() or .tolist())")
    if problems:
        raise SchemaError(f"{res.run_id or res.model_id}: " + "; ".join(problems))


# ==========================================================================
# Save and load
# ==========================================================================

def save(res: Result, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Validate and write one result. Overwrites an earlier result for the same run."""
    validate(res)
    path = result_path(res.model_id, res.diagnostic, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(res), indent=2) + "\n")
    return path


def load(path: Path | str) -> Result:
    data = json.loads(Path(path).read_text())
    known = {f for f in Result.__dataclass_fields__}
    extra = set(data) - known
    if extra:
        raise SchemaError(f"{path}: unknown fields {sorted(extra)}")
    res = Result(**data)
    validate(res)
    return res


def load_results(results_dir: Path | str = RESULTS_DIR) -> list[Result]:
    """Every result file of every diagnostic. A file that does not follow the schema raises."""
    out: list[Result] = []
    for d in DIAGNOSTICS:
        folder = Path(results_dir) / diag_folder(d)
        for path in sorted(folder.glob("*.json")):
            out.append(load(path))
    return out


# ==========================================================================
# Master table
# ==========================================================================

def master_table(results_dir: Path | str = RESULTS_DIR, models_csv: Path | str = MODELS_CSV,
                 domain: str | None = None) -> list[dict[str, Any]]:
    """One row per model in models.csv (optionally one domain): the model fields, then for each
    diagnostic its score and status. A run with no result file gets status "missing"."""
    models = load_models(models_csv)
    by_run = {(r.model_id, r.diagnostic): r for r in load_results(results_dir)}
    rows = []
    for model_id, m in models.items():
        if domain is not None and m["domain"] != domain:
            continue
        row: dict[str, Any] = {"model_id": model_id, **{k: m[k] for k in MODEL_FIELDS},
                               "held_out": m["held_out"]}
        for d in DIAGNOSTICS:
            r = by_run.get((model_id, d))
            row[f"{d}_score"] = r.score if r is not None else None
            row[f"{d}_status"] = r.status if r is not None else "missing"
        rows.append(row)
    return rows


def missing_report(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Runs with no score and no written reason, per diagnostic. Review 3 rule: every planned
    run has a score or a written reason, zero silent gaps."""
    return {d: [r["model_id"] for r in rows if r[f"{d}_status"] == "missing"] for d in DIAGNOSTICS}


def write_master_table(path: Path | str, results_dir: Path | str = RESULTS_DIR,
                       models_csv: Path | str = MODELS_CSV, domain: str | None = None) -> Path:
    rows = master_table(results_dir, models_csv, domain)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model_id"])
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the master table from results/ and report gaps.")
    p.add_argument("--domain", default=None)
    p.add_argument("--out", default=str(RESULTS_DIR / "master_table.csv"))
    a = p.parse_args()
    out = write_master_table(a.out, domain=a.domain)
    rows = master_table(domain=a.domain)
    gaps = missing_report(rows)
    print(f"wrote {out}: {len(rows)} models")
    for d, ids in gaps.items():
        print(f"  {d}: {len(rows) - len(ids)} with a result, {len(ids)} missing")
