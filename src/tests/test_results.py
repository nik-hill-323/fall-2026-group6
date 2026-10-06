"""Tests for the results schema (src/component/results.py).

Run from the repo root:
    python -m pytest src/tests/test_results.py -q
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("results", ROOT / "src/component/results.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

MODEL = "othello_transformer_small_synthetic_s0"
OTHER = "othello_lstm_small_synthetic_s1"


def ok_result(model_id: str = MODEL, diagnostic: str = "D4", score: float = 0.31, **kw):
    return R.make_result(model_id, diagnostic, method="test method, higher is better", split="test",
                         n_games=1000, data_seed=0, score=score, **kw)


def test_make_result_fills_model_fields() -> None:
    r = ok_result()
    assert (r.domain, r.arch, r.scale, r.distribution, r.seed) == ("othello", "transformer", "small", "synthetic", 0)
    assert r.run_id == f"{MODEL}_D4" and r.schema_version == R.SCHEMA_VERSION
    assert "git_commit" in r.provenance and "time_utc" in r.provenance


def test_paths_match_run_matrix() -> None:
    assert R.diag_folder("D1") == "D1_next_token" and R.diag_folder("D4") == "D4_probing"
    with open(ROOT / "src/docs/run_matrix.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["run_id"] == f"{MODEL}_D2")
    assert str(R.result_path(MODEL, "D2", "results")) == row["result"]


def test_round_trip(tmp_path: Path) -> None:
    r = ok_result(details={"selectivity_by_layer": [0.1, 0.2]}, score_std=0.02, score_ci=[0.27, 0.35])
    path = R.save(r, tmp_path)
    assert path == tmp_path / "D4_probing" / f"{MODEL}.json"
    back = R.load(path)
    assert back == r


def test_reference_model_needs_fields() -> None:
    with pytest.raises(R.SchemaError):
        R.make_result("othello_gpt_synthetic", "D4", method="m", split="test", n_games=10,
                      data_seed=0, score=0.9)
    r = R.make_result("othello_gpt_synthetic", "D4", method="m", split="test", n_games=10, data_seed=0,
                      score=0.9, domain="othello", arch="transformer", scale="othello_gpt",
                      distribution="synthetic", seed=0)
    assert r.arch == "transformer"


@pytest.mark.parametrize("change, message", [
    ({"diagnostic": "D9"}, "diagnostic"),
    ({"split": "holdout"}, "split"),
    ({"score": float("nan")}, "finite score"),
    ({"score": None}, "finite score"),
    ({"status": "failed"}, "needs a reason"),
    ({"method": "  "}, "method"),
    ({"score_ci": [0.5, 0.1]}, "score_ci"),
    ({"details": {"x": object()}}, "JSON"),
])
def test_validation_rejects(change: dict, message: str) -> None:
    r = ok_result()
    for k, v in change.items():
        setattr(r, k, v)
    if "diagnostic" in change:
        r.run_id = f"{r.model_id}_{r.diagnostic}"
    with pytest.raises(R.SchemaError, match=message):
        R.validate(r)


def test_failed_run_with_reason_is_valid(tmp_path: Path) -> None:
    r = ok_result(score=None, status="failed", reason="out of GPU memory")
    R.save(r, tmp_path)
    assert R.load(R.result_path(MODEL, "D4", tmp_path)).status == "failed"


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = R.save(ok_result(), tmp_path)
    data = json.loads(path.read_text())
    data["accuracy"] = 0.9
    path.write_text(json.dumps(data))
    with pytest.raises(R.SchemaError, match="unknown fields"):
        R.load(path)


def test_master_table_and_missing_report(tmp_path: Path) -> None:
    R.save(ok_result(MODEL, "D1", 0.84), tmp_path)
    R.save(ok_result(MODEL, "D4", 0.31), tmp_path)
    R.save(ok_result(OTHER, "D1", score=None, status="skipped", reason="below admission threshold"), tmp_path)
    rows = R.master_table(tmp_path, domain="othello")
    assert len(rows) == 81
    row = next(r for r in rows if r["model_id"] == MODEL)
    assert row["D1_score"] == 0.84 and row["D4_score"] == 0.31 and row["D2_status"] == "missing"
    other = next(r for r in rows if r["model_id"] == OTHER)
    assert other["D1_status"] == "skipped" and other["D1_score"] is None
    gaps = R.missing_report(rows)
    assert MODEL not in gaps["D1"] and OTHER not in gaps["D1"]  # a skipped run has a reason, so it is not a gap
    assert len(gaps["D1"]) == 79 and len(gaps["D2"]) == 81
    out = R.write_master_table(tmp_path / "master.csv", tmp_path, domain="othello")
    assert out.read_text().splitlines()[0].startswith("model_id,domain,arch,scale,distribution,seed,held_out,D1_score")
