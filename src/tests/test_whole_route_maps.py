"""Tests for the navigation whole route measure (src/analysis/whole_route_maps.py; Amendment 5, G9; ticket N12).

No data or checkpoints: a fake data folder of random legal routes on the real map, models whose answers are
known (src/tests/maps_scoring_fixtures.py) and untrained small zoo models.

Run from the repo root:
    python -m pytest src/tests/test_whole_route_maps.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load("maps_scoring_fixtures", "src/tests/maps_scoring_fixtures.py")
wr = _load("whole_route_maps", "src/analysis/whole_route_maps.py")
nav, maps = fx.nav, fx.maps

N = 60


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    return fx.use_fake_data(monkeypatch, tmp_path / "data", n=200)


def _pairs(n: int = 40, seed: int = 41):
    d = maps.build(fx.dom.random_routes(n, seed=seed))
    return d["tokens"][:, 0], d["dest"]


class _Recorder(fx.MapsOracle):
    """Records every prefix it is shown (the drive so far)."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.seen: list[np.ndarray] = []

    def _logits(self, tokens):
        self.seen.append(tokens.numpy().copy())
        return super()._logits(tokens)


def _check_counts(fp: dict) -> None:
    assert sum(fp["outcome_counts"].values()) == fp["n_drives"]
    assert sum(n for _, _, n in fp["legal_rate_by_step"]) == fp["n_predictions"]
    ok = sum(k for _, k, _ in fp["legal_rate_by_step"])
    assert fp["legal_rate"] == pytest.approx(ok / fp["n_predictions"])
    assert fp["mean_turns"] <= fp["longest_drive_turns"] <= maps.MAX_TURNS
    assert all(1 <= s <= maps.MAX_TURNS + 1 for s, _, _ in fp["legal_rate_by_step"])


# ==========================================================================
# Teacher forced
# ==========================================================================

def test_teacher_forced_perfect_model() -> None:
    d = maps.build(fx.dom.random_routes(N, seed=5))
    tf = wr.teacher_forced(fx.MapsOracle("legal"), d, batch=16)
    assert tf["legal_rate"] == 1.0 and tf["share_routes_all_legal"] == 1.0 and tf["errors_per_route"] == 0.0
    assert tf["first_error_step"]["n_with_error"] == 0
    assert tf["n_predictions"] == int(d["score"][:, :maps.N_CTX].sum())


@pytest.mark.parametrize("model", [fx.MapsOracle("noise"), fx.MapsOracle("fixed", fixed=maps.END)])
def test_teacher_forced_equals_d1(model) -> None:
    d = maps.build(fx.dom.random_routes(N, seed=6))
    tf = wr.teacher_forced(model, d, batch=16)
    assert tf["legal_rate"] == maps.legal_rate(model, d, "cpu", 16) == tf["d1_rate_recomputed"]
    ok = sum(k for _, k, _ in tf["legal_rate_by_step"])
    assert ok / tf["n_predictions"] == pytest.approx(tf["legal_rate"])
    errs = int(round(tf["errors_per_route"] * tf["n_routes"]))
    assert errs == tf["n_predictions"] - ok


def test_teacher_forced_first_error_and_all_legal_by_hand() -> None:
    """A model that always says end: end is legal only where the car is at the destination (the last scored
    position, and any earlier pass through it), so every route has an error at step 1 and none is all legal."""
    d = maps.build(fx.dom.random_routes(N, seed=7))
    tf = wr.teacher_forced(fx.MapsOracle("fixed", fixed=maps.END), d, batch=16)
    assert tf["share_routes_all_legal"] == 0.0
    assert tf["first_error_step"]["histogram"] == {"1": N}
    sc = d["score"][:, :maps.N_CTX]
    want = int((sc & (d["current"][:, :maps.N_CTX] != d["dest"][:, None])).sum())
    assert tf["errors_per_route"] == pytest.approx(want / N)


def test_teacher_forced_refuses_a_rate_that_is_not_d1(monkeypatch) -> None:
    d = maps.build(fx.dom.random_routes(20, seed=8))
    monkeypatch.setattr(maps, "legal_rate", lambda *a, **k: 0.5)
    with pytest.raises(ValueError, match="differs from the D1"):
        wr.teacher_forced(fx.MapsOracle("legal"), d, batch=16)


def test_teacher_forced_untrained_models() -> None:
    d = maps.build(fx.dom.random_routes(24, seed=9))
    for arch in ("transformer", "lstm"):
        m = fx.tiny(arch)
        tf = wr.teacher_forced(m, d, batch=8)
        assert tf["legal_rate"] == maps.legal_rate(m, d, "cpu", 8)


# ==========================================================================
# Free play
# ==========================================================================

def test_free_play_perfect_model_drives_legal_routes() -> None:
    s, t = _pairs()
    model = _Recorder("legal")
    fp = wr.free_play(model, s, t, batch=16)
    assert fp["legal_rate"] == 1.0 and fp["share_drives_no_illegal"] == 1.0 and fp["errors_per_drive"] == 0.0
    _check_counts(fp)
    for x in model.seen:                      # every prefix the model was shown is a legal route so far
        for row in x:
            cur = int(row[0])
            for tok in row[2:]:
                assert maps.NEXT[cur, tok - maps.TURN0] >= 0
                cur = int(maps.NEXT[cur, tok - maps.TURN0])
    assert max(x.shape[1] for x in model.seen) <= maps.N_CTX


def test_free_play_is_seeded_and_the_seed_matters() -> None:
    s, t = _pairs()
    a = wr.free_play(fx.MapsOracle("legal"), s, t, seed=0, batch=16)
    assert wr.free_play(fx.MapsOracle("legal"), s, t, seed=0, batch=16) == a
    assert wr.free_play(fx.MapsOracle("legal"), s, t, seed=0, batch=7) == a        # batch does not change drives
    assert wr.free_play(fx.MapsOracle("legal"), s, t, seed=1, batch=16) != a


def test_free_play_ends_only_at_the_destination_and_at_96_turns() -> None:
    """A model that always says end: end is legal only at the destination, so away from it every top token is
    wrong, and the moves drawn over the legal set are uniform turns (a random walk). Drives end at the destination
    with end, at a dead end, or at 96 turns, and no drive makes a 97th turn."""
    s, t = _pairs(40, seed=42)
    model = _Recorder("fixed", fixed=maps.END)
    fp = wr.free_play(model, s, t, batch=16)
    _check_counts(fp)
    assert fp["share_drives_no_illegal"] == 0.0
    assert fp["first_error_step"]["histogram"] == {"1": 40}
    assert fp["outcome_counts"]["length_limit"] > 0 and fp["longest_drive_turns"] == maps.MAX_TURNS
    assert max(x.shape[1] for x in model.seen) == maps.N_CTX           # the step at 96 turns is read and scored
    steps = [s_ for s_, _, _ in fp["legal_rate_by_step"]]
    assert max(steps) == maps.MAX_TURNS + 1
    # a drive that reached its destination played end there: every other step of it was a wrong top token
    reached = fp["outcome_counts"]["reached_destination"]
    assert fp["share_reached_destination"] == pytest.approx(reached / 40)


def test_free_play_stops_at_a_dead_end_before_predicting() -> None:
    """From an intersection whose only turn leads into a dead end, the first move is forced; the drive is then
    stuck, with exactly one prediction."""
    nxt = maps.NEXT
    cands = [u for u in range(1, maps.N_NODES + 1)
             if (nxt[u] >= 0).sum() == 1 and maps.IS_DEAD_END[nxt[u][nxt[u] >= 0][0]]]
    if not cands:
        pytest.skip("no intersection on the map whose only turn leads into a dead end")
    start = cands[0]
    dest = 1 if start != 1 else 2
    model = _Recorder("legal")
    fp = wr.free_play(model, np.array([start]), np.array([dest]), batch=4)
    assert fp["outcome_counts"]["stuck_at_dead_end"] == 1 and fp["n_predictions"] == 1
    assert fp["mean_turns"] == 1.0 and len(model.seen) == 1


def test_sampling_follows_the_model_over_the_legal_moves_only() -> None:
    rng = np.random.default_rng(0)
    logits = np.log(np.array([[0.1, 0.2, 0.3, 0.4, 9.0, 1.0, 1.0, 1.0, 5.0]]))
    legal = np.array([[True, True, True, True, False, False, False, False, False]])
    n = 200_000
    picks = wr.sample_legal(np.repeat(logits, n, 0), np.repeat(legal, n, 0), rng.random(n))
    freq = np.bincount(picks, minlength=9) / n
    assert freq[4:].sum() == 0.0
    assert freq[:4] == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=0.005)
    # a uniform draw just below 1 still lands on a legal move (rounding of the cumulative sum)
    edge = wr.sample_legal(np.array([[0.0, 50.0, -50.0, 0, 0, 0, 0, 0, 0]]),
                           np.array([[True, False, True, False, False, False, False, False, False]]),
                           np.array([1 - 1e-17]))
    assert edge[0] in (0, 2)


def test_free_play_untrained_models() -> None:
    s, t = _pairs(12, seed=43)
    for arch in ("transformer", "lstm"):
        fp = wr.free_play(fx.tiny(arch), s, t, batch=8)
        _check_counts(fp)


# ==========================================================================
# One model, files and the runner
# ==========================================================================

def test_score_model_end_to_end(data_dir) -> None:
    res = wr.score_model(fx.MapsOracle("legal"), batch=32, n_routes=N)
    assert res["status"] == "ok" and res["train_distribution"] == "shortest_paths"
    assert res["teacher_forced"]["n_routes"] == N and res["free_play"]["n_drives"] == N
    assert res["teacher_forced"]["share_routes_all_legal"] == 1.0
    assert res["free_play"]["share_drives_no_illegal"] == 1.0
    rec = wr.result_record(fx.ZOO_ID, nav.model_fields(fx.MapsOracle("legal")), res, 1.0)
    assert rec["model_id"] == fx.ZOO_ID and rec["distribution"] == "shortest_paths" and "git_commit" in rec["provenance"]
    json.dumps(rec)


def test_free_play_pairs_are_d1_routes(data_dir) -> None:
    """The free play pairs are the starts and destinations of D1's 1,000 routes (here N): same seed, same order."""
    model = _Recorder("legal")
    wr.score_model(model, batch=1000, n_routes=N)
    d = nav.routes("shortest_paths", "test", N, seed=0)
    first_drive_call = next(x for x in model.seen if x.shape[1] == 2)
    assert (first_drive_call[:, 0] == d["tokens"][:, 0]).all() and (first_drive_call[:, 1] == d["dest"]).all()


def test_d1_file_check(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(nav.results, "RESULTS_DIR", tmp_path)
    assert wr.d1_file_check(fx.ZOO_ID, 0.9) == {"d1_file": None}
    (tmp_path / "D1_next_token").mkdir()
    (tmp_path / "D1_next_token" / f"{fx.ZOO_ID}.json").write_text(json.dumps({"status": "ok", "score": 0.9}))
    assert wr.d1_file_check(fx.ZOO_ID, 0.9)["matches_d1_file"] is True
    assert wr.d1_file_check(fx.ZOO_ID, 0.8)["matches_d1_file"] is False


def test_failure_is_written_with_its_reason_and_raised(tmp_path, monkeypatch, data_dir) -> None:
    monkeypatch.setattr(nav.results, "RESULTS_DIR", tmp_path)

    class Broken(fx.MapsOracle):
        def _logits(self, tokens):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        wr.score_one(fx.ZOO_ID, save=True, batch=16, load_fn=lambda m: Broken())
    rec = json.loads((tmp_path / "whole_route_maps" / f"{fx.ZOO_ID}.json").read_text())
    assert rec["status"] == "failed" and "out of memory" in rec["reason"] and rec["model_id"] == fx.ZOO_ID


def test_result_paths() -> None:
    assert wr.result_path(fx.PILOT_ID) == nav.PILOT_RESULTS_DIR / "whole_route_maps" / f"{fx.PILOT_ID}.json"
    assert wr.result_path(fx.ZOO_ID) == nav.results.RESULTS_DIR / "whole_route_maps" / f"{fx.ZOO_ID}.json"


def _runner(*args: str, jobs: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "JOBS": jobs, "PY": sys.executable}
    return subprocess.run(["bash", "src/shellscripts/maps_score_slice.sh", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_runner_plans_g9() -> None:
    out = _runner("pilots", "--dry_run", jobs="G9")
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    plans = [line for line in lines if line.startswith("[plan]") and " G9 " in line]
    assert all("src/analysis/whole_route_maps.py" in p for p in plans)
    # a pilot whose G9 result is committed is skipped, not planned; together they must cover all 9 pilots
    pilots = [p for p in plans if "_s100 " in p]
    done = [line.split()[1] for line in lines if line.startswith("[skip]") and line.endswith(" G9 (result exists)")]
    done_pilots = [m for m in done if m.endswith("_s100")]
    assert len(pilots) + len(done_pilots) == 9
    assert all("results/pilot_scoring/whole_route_maps/" in p for p in pilots)
    assert all((ROOT / "results/pilot_scoring/whole_route_maps" / f"{m}.json").exists() for m in done_pilots)
    anchors = [p for p in plans if "maps_vafa_" in p]                 # only on a machine with their weights
    assert len(plans) == len(pilots) + len(anchors)
    assert all("-> results/whole_route_maps/" in p for p in anchors)
    big = [p for p in anchors if "maps_vafa_shortest_paths" not in p]
    assert all("--batch 32" in p for p in big)
    assert not any("--batch" in p for p in pilots + [p for p in anchors if p not in big])
    assert _runner("pilots", "--dry_run", jobs="G10").returncode == 1


def test_runner_default_jobs_include_g9() -> None:
    text = (ROOT / "src/shellscripts/maps_score_slice.sh").read_text()
    assert 'JOBS="${JOBS:-D1 D2 D4 fragility G9 D3}"' in text
