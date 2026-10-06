"""Zoo registry: which zoo models are trained, how well, and where their checkpoints are.

Source of truth: one small training record per model in results/zoo_training/{model_id}.json,
written by src/zoo/train.py after every full budget run on real data. Each instance commits only
its own records, so four people training different slices never edit the same file.

The registry table src/docs/zoo_registry.csv is rebuilt from those records:
    python src/zoo/registry.py              # rebuild src/docs/zoo_registry.csv and print a summary

Loading a model for a diagnostic:
    model = load_model("othello_transformer_large_synthetic_s0")    # a ZooModel

Admission (H1 to H3 use only admitted models) is decided by the threshold in the preregistration.
Until that threshold is fixed, `admitted` is "tbd" for every model.
"""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = ROOT / "results" / "zoo_training"
REGISTRY_CSV = ROOT / "src" / "docs" / "zoo_registry.csv"

# Minimum legal move rate (validation split) for a model to count in H1 to H3. None until the
# preregistration fixes it; then every row gets admitted yes or no.
ADMISSION_MIN_LEGAL_RATE: float | None = None

COLUMNS = ["model_id", "domain", "arch", "scale", "distribution", "seed", "n_params", "steps",
           "final_loss", "val_legal_rate", "val_next_token_acc", "train_minutes", "gpu", "host",
           "git_commit", "time_utc", "checkpoint", "checkpoint_sha256", "admitted"]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def admitted(val_legal_rate: float, threshold: float | None = ADMISSION_MIN_LEGAL_RATE) -> str:
    if threshold is None:
        return "tbd"
    return "yes" if val_legal_rate >= threshold else "no"


def record_to_row(rec: dict[str, Any], threshold: float | None = ADMISSION_MIN_LEGAL_RATE) -> dict[str, Any]:
    """One training record (train.py results.json plus record fields) as a registry row."""
    legal = rec["eval"]["legal_move_rate"]
    return {
        "model_id": rec["model_id"], "domain": rec.get("domain", "othello"), "arch": rec["arch"],
        "scale": rec["scale"], "distribution": rec.get("distribution", rec["config"]["distribution"]),
        "seed": rec.get("seed", rec["config"]["seed"]), "n_params": rec["n_params"],
        "steps": rec["train"]["steps_run"], "final_loss": round(rec["train"]["final_loss"], 4),
        "val_legal_rate": round(legal, 4), "val_next_token_acc": round(rec["eval"]["next_token_acc"], 4),
        "train_minutes": round(rec["train"]["wall_sec"] / 60, 1), "gpu": rec.get("gpu", ""),
        "host": rec.get("host", ""), "git_commit": rec.get("git_commit", ""),
        "time_utc": rec.get("time_utc", ""),
        "checkpoint": rec.get("checkpoint", f"outputs/zoo/{rec['model_id']}/model.pt"),
        "checkpoint_sha256": rec.get("checkpoint_sha256", ""),
        "admitted": admitted(legal, threshold),
    }


def load_records(records_dir: Path | str = RECORDS_DIR) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(records_dir).glob("*.json")):
        out.append(json.loads(path.read_text()))
    return out


def build_registry(records_dir: Path | str = RECORDS_DIR, threshold: float | None = ADMISSION_MIN_LEGAL_RATE,
                   models_csv: Path | str | None = None) -> list[dict[str, Any]]:
    """Registry rows, one per trained model, in models.csv order. A record whose model_id is not
    in models.csv is rejected: the zoo is exactly the run matrix."""
    results = _load("results", "src/component/results.py")
    models = results.load_models(models_csv) if models_csv else results.load_models()
    rows = {}
    for rec in load_records(records_dir):
        if rec["model_id"] not in models:
            raise ValueError(f"{rec['model_id']} has a training record but is not in models.csv")
        rows[rec["model_id"]] = record_to_row(rec, threshold)
    return [rows[m] for m in models if m in rows]


def write_registry(path: Path | str = REGISTRY_CSV, **kwargs: Any) -> list[dict[str, Any]]:
    rows = build_registry(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_model(model_id: str, device: str | None = None, check_hash: bool = True,
               records_dir: Path | str = RECORDS_DIR):
    """A trained zoo model as a ZooModel (src/diagnostics/interface.py). Checks that a training
    record exists and, by default, that the checkpoint file is the one that was trained."""
    itf = _load("interface", "src/diagnostics/interface.py")
    rec_path = Path(records_dir) / f"{model_id}.json"
    if not rec_path.exists():
        raise FileNotFoundError(f"{model_id} is not trained yet (no {rec_path})")
    rec = json.loads(rec_path.read_text())
    ckpt = ROOT / rec.get("checkpoint", f"outputs/zoo/{model_id}/model.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"{model_id}: checkpoint {ckpt} is not on this machine "
                                f"(trained on {rec.get('host', '?')}; copy it from there or from S3)")
    if check_hash and rec.get("checkpoint_sha256"):
        digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if digest != rec["checkpoint_sha256"]:
            raise ValueError(f"{model_id}: checkpoint does not match its training record (sha256)")
    return itf.from_checkpoint(model_id, device=device, zoo_dir=ckpt.parent.parent)


def import_outputs(zoo_dir: Path | str = ROOT / "outputs" / "zoo", records_dir: Path | str = RECORDS_DIR,
                   budget_steps: int | None = None) -> list[str]:
    """Write training records for full budget runs that finished before train.py wrote records
    itself (for example the first zoo models and the large transformer sanity run). Skips runs
    that already have a record, short runs (throughput), and synthetic smoke tests."""
    if budget_steps is None:
        budget_steps = int(_load("run_matrix", "src/component/run_matrix.py").TRAINING_BUDGET["steps"])
    written = []
    for res_path in sorted(Path(zoo_dir).glob("*/results.json")):
        res = json.loads(res_path.read_text())
        rec_path = Path(records_dir) / f"{res['model_id']}.json"
        cfg = res["config"]
        if rec_path.exists() or cfg.get("synthetic") or res["train"]["steps_run"] != budget_steps:
            continue
        ckpt = res_path.parent / "model.pt"
        if not ckpt.exists():
            continue
        rec = {**res, "domain": "othello", "distribution": cfg["distribution"], "seed": cfg["seed"],
               "checkpoint": str(ckpt.resolve().relative_to(ROOT)),
               "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
               "git_commit": "", "host": "", "time_utc": ""}
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(rec, indent=2) + "\n")
        written.append(res["model_id"])
    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Rebuild src/docs/zoo_registry.csv from results/zoo_training/.")
    p.add_argument("--import_outputs", action="store_true",
                   help="first write records for finished full budget runs in outputs/zoo that have none")
    a = p.parse_args()
    if a.import_outputs:
        for m in import_outputs():
            print(f"  record written for {m}")
    rows = write_registry()
    print(f"wrote {REGISTRY_CSV.relative_to(ROOT)}: {len(rows)} trained models")
    by = {}
    for r in rows:
        by.setdefault((r["arch"], r["scale"]), []).append(r["val_legal_rate"])
    for (arch, scale), rates in sorted(by.items()):
        print(f"  {arch:12s} {scale:7s} {len(rates):3d} models  legal rate {min(rates):.3f} to {max(rates):.3f}")
