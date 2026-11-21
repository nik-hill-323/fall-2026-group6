"""Tests for the navigation scoring foundation (src/diagnostics/navigation.py, N0).

They need no data and no checkpoints: a small fake data folder is built from random legal routes on the real map
(src/tests/maps_scoring_fixtures.py), and models are fakes or untrained small zoo models.

Run from the repo root:
    python -m pytest src/tests/test_navigation_scoring.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("maps_scoring_fixtures", ROOT / "src/tests/maps_scoring_fixtures.py")
fx = importlib.util.module_from_spec(_spec)
sys.modules["maps_scoring_fixtures"] = fx
_spec.loader.exec_module(fx)
nav, itf, maps, results, train = fx.nav, fx.itf, fx.maps, fx.nav.results, fx.train


# ==========================================================================
# Which model is which
# ==========================================================================

def test_pilot_ids_are_the_trainers() -> None:
    assert nav.pilot_ids() == train.pilot_ids()
    assert len(nav.pilot_ids()) == 9


def test_zoo_ids_are_the_81_navigation_rows() -> None:
    ids = nav.zoo_ids()
    assert len(ids) == 81 and all(m.startswith("maps_") for m in ids)
    assert not set(ids) & set(nav.pilot_ids())


def test_kind() -> None:
    assert nav.kind(fx.ZOO_ID) == "zoo"
    assert nav.kind(fx.PILOT_ID) == "pilot"
    assert nav.kind("maps_vafa_random_walks") == "anchor"
    for bad in ("othello_transformer_small_synthetic_s0", "maps_transformer_small_shortest_paths_s7",
                "maps_vafa_other"):
        with pytest.raises(ValueError):
            nav.kind(bad)


def test_parse_id_matches_models_csv() -> None:
    rows = results.load_models()
    for m in nav.zoo_ids():
        f = nav.parse_id(m)
        r = rows[m]
        assert (f["arch"], f["scale"], f["distribution"], f["seed"]) == (r["arch"], r["scale"], r["distribution"], int(r["seed"]))
    for m in nav.pilot_ids():
        f = nav.parse_id(m)
        assert f["scale"] == "small" and f["seed"] == 100 and f["domain"] == "maps"


def test_model_fields_from_csv_and_from_meta() -> None:
    assert nav.model_fields(fx.MapsOracle())["distribution"] == "shortest_paths"
    pilot = fx.MapsOracle(model_id=fx.PILOT_ID, meta=nav.parse_id(fx.PILOT_ID))
    assert nav.model_fields(pilot)["seed"] == 100
    assert nav.is_maps(pilot) and nav.is_maps(fx.MapsOracle())
    with pytest.raises(ValueError):
        nav.model_fields(fx.MapsOracle(model_id=fx.PILOT_ID))     # no meta, not in models.csv


# ==========================================================================
# Gates
# ==========================================================================

def test_gate_refuses_zoo_models_without_both_tags(monkeypatch) -> None:
    monkeypatch.setattr(nav, "missing_tags", lambda tags=nav.GATE_TAGS: ["maps-fragility-v1"])
    monkeypatch.setattr(nav, "uncommitted_src", lambda: [])
    with pytest.raises(SystemExit, match="maps-fragility-v1"):
        nav.check_can_score([fx.ZOO_ID], save=False)
    nav.check_can_score([fx.PILOT_ID, "maps_vafa_shortest_paths"], save=True)     # pilots and anchors: allowed
    monkeypatch.setattr(nav, "missing_tags", lambda tags=nav.GATE_TAGS: [])
    nav.check_can_score([fx.ZOO_ID], save=True)


def test_gate_refuses_mixed_unknown_and_uncommitted(monkeypatch) -> None:
    monkeypatch.setattr(nav, "missing_tags", lambda tags=nav.GATE_TAGS: [])
    with pytest.raises(SystemExit, match="separate commands"):
        nav.check_can_score([fx.ZOO_ID, "othello_transformer_small_synthetic_s0"], save=False)
    with pytest.raises(SystemExit, match="not a navigation"):
        nav.check_can_score(["maps_transformer_small_shortest_paths_s9"], save=False)
    monkeypatch.setattr(nav, "uncommitted_src", lambda: ["src/diagnostics/d1/next_token.py"])
    with pytest.raises(SystemExit, match="not committed"):
        nav.check_can_score([fx.PILOT_ID], save=True)
    nav.check_can_score([fx.PILOT_ID], save=False)                 # a dry look without saving is fine
    nav.check_can_score(["maps_vafa_random_walks"], save=True)      # anchors have no training record to protect


# ==========================================================================
# Loading
# ==========================================================================

def _write_pilot(tmp: Path, monkeypatch, model_id: str = fx.PILOT_ID) -> Path:
    module = train.build_model("transformer", "small", "maps")
    ck = tmp / "pilot" / model_id / "model.pt"
    ck.parent.mkdir(parents=True)
    torch.save(module.state_dict(), ck)
    rec = tmp / "records"
    rec.mkdir()
    (rec / f"{model_id}.json").write_text(json.dumps({"checkpoint_sha256": nav.sha256_file(ck)}))
    monkeypatch.setattr(nav, "PILOT_DIR", tmp / "pilot")
    monkeypatch.setattr(nav, "PILOT_RECORDS", rec)
    return ck


def test_load_pilot_checks_its_checkpoint(tmp_path: Path, monkeypatch) -> None:
    ck = _write_pilot(tmp_path, monkeypatch)
    m = nav.load(fx.PILOT_ID, "cpu")
    assert m.meta == nav.parse_id(fx.PILOT_ID) and m.n_layers == 4 and m.d_model == 128
    assert m.logits(torch.zeros(2, 5, dtype=torch.long)).shape == (2, 5, maps.VOCAB)
    with open(ck, "ab") as f:                                    # any change to the file is caught
        f.write(b"x")
    with pytest.raises(ValueError, match="not the trained model"):
        nav.load(fx.PILOT_ID, "cpu")
    assert nav.load(fx.PILOT_ID, "cpu", verify=False).model_id == fx.PILOT_ID   # tests only


def test_load_without_record_or_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nav, "PILOT_DIR", tmp_path / "none")
    monkeypatch.setattr(nav, "PILOT_RECORDS", tmp_path / "none")
    with pytest.raises(FileNotFoundError, match="training record"):
        nav.load(fx.PILOT_ID, "cpu")


def test_pilot_results_go_apart_from_the_analysis(tmp_path: Path, monkeypatch) -> None:
    assert nav.results_dir(fx.PILOT_ID) == nav.PILOT_RESULTS_DIR
    assert nav.results_dir(fx.ZOO_ID) == results.RESULTS_DIR
    assert nav.results_dir("maps_vafa_random_walks") == results.RESULTS_DIR
    assert nav.PILOT_RESULTS_DIR.name == "pilot_scoring"
    assert "pilot_scoring" not in [results.diag_folder(d) for d in results.DIAGNOSTICS]


# ==========================================================================
# Data and positions
# ==========================================================================

def test_routes_from_a_data_folder(tmp_path: Path, monkeypatch) -> None:
    fx.use_fake_data(monkeypatch, tmp_path / "data", n=50)
    d = nav.routes("random_walks", "test", 20, seed=0)
    assert len(d["tokens"]) == 20 and d["n_illegal"] == 0
    d2 = nav.routes("random_walks", "test", 20, seed=0)
    assert np.array_equal(d["tokens"], d2["tokens"])


def test_data_is_checked_against_the_manifest(tmp_path: Path, monkeypatch) -> None:
    path = fx.fake_data_dir(tmp_path / "data", n=20)
    man = tmp_path / "manifest.json"
    good = maps.sha256(path / "random_walks_test.npy")
    man.write_text(json.dumps({"files": {"random_walks_test.npy": good, "random_walks_val.npy": "0" * 64}}))
    monkeypatch.setattr(maps, "MANIFEST", man)
    monkeypatch.setattr(nav, "VERIFY_DATA", True)
    monkeypatch.setattr(nav, "_VERIFIED", set())
    nav.routes("random_walks", "test", 5, data_dir=path)
    with pytest.raises(ValueError, match="does not match"):
        nav.routes("random_walks", "val", 5, data_dir=path)


def test_current_intersections_equal_the_replay() -> None:
    d = maps.build(fx.dom.random_routes(80, seed=4))
    cur = nav.current_intersections(d["tokens"])
    pos = np.arange(maps.MAX_TOKENS)[None, :]
    alive = (pos >= 1) & (pos <= d["length"][:, None] - 1)
    assert np.array_equal(cur[alive], d["current"][alive])
    # on a prefix cut before the end, the same answer
    cut = nav.current_intersections(d["tokens"][:, :10])
    assert np.array_equal(cut[:, 1:], np.where(alive[:, 1:10], d["current"][:, 1:10], 0))


def test_slots_to_tokens_and_legal_token_mask() -> None:
    d = maps.build(fx.dom.random_routes(30, seed=5))
    tok = nav.slots_to_tokens(d["legal"])
    for g in range(5):
        for p in range(1, int(d["length"][g]) - 1):
            want = nav.legal_token_mask(d["current"][g, p], d["dest"][g])
            assert np.array_equal(tok[g, p], want)


def test_correct_at_adds_up_to_legal_rate() -> None:
    d = maps.build(fx.dom.random_routes(40, seed=6))
    sc = d["score"][:, :maps.N_CTX]
    oracle = fx.MapsOracle("legal")
    assert nav.correct_at(oracle, d, batch=16)[sc].all()
    for model in (fx.tiny("transformer"), fx.tiny("lstm"), fx.MapsOracle("fixed", fixed=maps.END)):
        c = nav.correct_at(model, d, batch=16)
        assert not c[~sc].any()
        assert c.sum() / sc.sum() == pytest.approx(maps.legal_rate(model, d, "cpu", 16))


def test_top_tokens_are_the_best_and_never_padding() -> None:
    d = maps.build(fx.dom.random_routes(10, seed=8))
    model = fx.MapsOracle("fixed", fixed=maps.PAD)
    top = nav.top_tokens(model, d, k=3, batch=4)
    assert (top != maps.PAD).all()
    m = fx.tiny("transformer")
    top = nav.top_tokens(m, d, k=5, batch=4)
    logits = m.logits(nav.inputs(d)).numpy()
    logits[..., 0] = -np.inf
    assert np.array_equal(top[..., 0], logits.argmax(-1))


# ==========================================================================
# Running a scorer
# ==========================================================================

class _Toy:
    """A minimal diagnostic module: the share of legal top tokens on its own little data."""
    DIAGNOSTIC = "D1"
    METHOD = "toy"

    @staticmethod
    def score(model, split="test", n_games=10, data_seed=0, **config):
        d = maps.build(fx.dom.random_routes(10, seed=9))
        return itf.result_for(model, "D1", method="toy", split=split, n_games=10, data_seed=data_seed,
                              score=float(maps.legal_rate(model, d, "cpu", 8)))


def test_run_saves_pilots_apart_and_reports_unloadable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(nav, "PILOT_RESULTS_DIR", tmp_path / "pilot_scoring")
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    pilot = fx.MapsOracle(model_id=fx.PILOT_ID, meta=nav.parse_id(fx.PILOT_ID))

    def fake_load(m):
        if m == fx.PILOT_ID:
            return pilot
        raise FileNotFoundError("no checkpoint here")

    out, not_loaded = nav.run(_Toy, [fx.PILOT_ID, fx.ZOO_ID], save=True, load_fn=fake_load)
    assert [r.model_id for r in out] == [fx.PILOT_ID] and out[0].score == 1.0
    assert out[0].seed == 100 and out[0].distribution == "shortest_paths"
    assert (tmp_path / "pilot_scoring" / "D1_next_token" / f"{fx.PILOT_ID}.json").exists()
    assert not (tmp_path / "results").exists()                     # nothing where the analysis reads
    assert not_loaded and not_loaded[0][0] == fx.ZOO_ID
    assert results.load_results(tmp_path / "results") == []
