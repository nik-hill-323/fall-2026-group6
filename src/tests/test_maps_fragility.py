"""Tests for the navigation fragility suite maps-fragility-v1 (src/fragility/maps.py; Amendment 5, G8; ticket N8).

No data or checkpoints: a fake data folder of random legal routes on the real map, models whose answers are
known (src/tests/maps_scoring_fixtures.py) and untrained small zoo models.

The freeze test is skipped until the suite is tagged: in the commit that makes the tag maps-fragility-v1, set
FROZEN_HASH below to suite_hash() (after EXCLUDED_BY_PILOT_RULE holds the pilot rule's outcome).

Run from the repo root:
    python -m pytest src/tests/test_maps_fragility.py -q
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


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load("maps_scoring_fixtures", "src/tests/maps_scoring_fixtures.py")
fr = _load("fragility_maps", "src/fragility/maps.py")
nav, itf, maps, tasks = fx.nav, fx.itf, fx.maps, fr.maps_tasks

FROZEN_VERSION = "maps-fragility-v1"
FROZEN_HASH: str | None = None          # set in the commit tagged maps-fragility-v1
SMALL = {"n_detour_routes": 30, "n_test_routes": 60, "n_rare_routes": 150}


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    return fx.use_fake_data(monkeypatch, tmp_path / "data", n=200)


# ==========================================================================
# Definition and freeze
# ==========================================================================

def test_freeze() -> None:
    assert fr.SUITE_VERSION == FROZEN_VERSION
    if FROZEN_HASH is None:
        pytest.skip("not tagged yet: set FROZEN_HASH in the commit tagged maps-fragility-v1")
    assert fr.EXCLUDED_BY_PILOT_RULE is not None, "the pilot rule's outcome is written before the tag"
    assert fr.suite_hash() == FROZEN_HASH, "the suite changed after it was frozen"


def test_seven_drops_per_model() -> None:
    for d in maps.DISTRIBUTIONS:
        names = fr.drop_names(d)
        assert len(names) == 7 and f"cross_distribution/{d}" not in names
    assert len(fr.all_drop_names()) == 8          # 3 cross distribution, 4 detour, 1 least visited


def test_rare_file_is_the_preregistered_one() -> None:
    assert fr.SUITE["least_visited"]["rare_file_sha256"] == maps.sha256(tasks.RARE_FILE)
    sizes = {d: len(tasks.load_rare(d)) for d in maps.DISTRIBUTIONS}
    assert sizes == {"shortest_paths": 2749, "noisy_shortest_paths": 2816, "random_walks": 1130}


def test_hash_follows_the_definition(monkeypatch) -> None:
    h = fr.suite_hash()
    monkeypatch.setitem(fr.SUITE, "data_seed", 1)
    assert fr.suite_hash() != h
    monkeypatch.setitem(fr.SUITE, "data_seed", 0)
    monkeypatch.setattr(fr, "EXCLUDED_BY_PILOT_RULE", ["least_visited"])
    assert fr.suite_hash() != h


# ==========================================================================
# Detour drives
# ==========================================================================

def _pairs(n: int = 30, seed: int = 41):
    d = maps.build(fx.dom.random_routes(n, seed=seed))
    return d["tokens"][:, 0], d["dest"]


def test_perfect_model_drives_legally_and_arrives() -> None:
    s, t = _pairs()
    for mode in ("random", "adversarial"):
        for p in (0.0, 0.5):
            out = fr.detour_drive(fx.MapsOracle("legal"), s, t, p, mode, 0, "cpu", batch=16)
            assert out["legal_rate"] == 1.0 and out["reached_destination_with_end"] == 1.0 and out["stuck"] == 0
    assert fr.detour_drive(fx.MapsOracle("legal"), s, t, 0.0, "random", 0, "cpu")["n_detours"] == 0


class _Recorder(fx.MapsOracle):
    """Records every prefix it is shown (the drive so far)."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.seen: list[np.ndarray] = []

    def _logits(self, tokens):
        self.seen.append(tokens.numpy().copy())
        return super()._logits(tokens)


def _full_drive(seen: list[np.ndarray], start: int) -> np.ndarray:
    """The longest prefix shown for the drive that starts at `start` (its whole drive before the last move)."""
    return max((row for x in seen for row in x if row[0] == start), key=len)


def test_every_drive_is_a_legal_route_ending_at_its_destination() -> None:
    s, t = _pairs()
    model = _Recorder("noise")
    out = fr.detour_drive(model, s, t, 0.5, "random", 0, "cpu")
    assert out["reached_destination_with_end"] == 1.0 and out["stuck"] == 0 and out["mean_turns"] <= maps.MAX_TURNS
    for x in model.seen:                      # every prefix the model was shown is legal on the map
        for row in x:
            cur = row[0]
            for tok in row[2:]:
                assert maps.NEXT[cur, tok - maps.TURN0] >= 0
                cur = maps.NEXT[cur, tok - maps.TURN0]


def test_detours_happen_at_about_p_and_never_at_the_destination() -> None:
    s, t = _pairs(40)
    out = fr.detour_drive(fx.MapsOracle("legal"), s, t, 0.5, "random", 0, "cpu")
    assert out["n_detours"] > 0.3 * out["n_predictions"] * 0.5
    assert fr.detour_drive(fx.MapsOracle("legal"), s, t, 0.5, "random", 0, "cpu") == out      # seeded


class _Prefers(fx.MapsOracle):
    """Logits rise with the token id, so among turns NW is rated highest and N lowest."""

    def _logits(self, tokens):
        return torch.arange(maps.VOCAB, dtype=torch.float32).expand(*tokens.shape, maps.VOCAB).clone()


def test_adversarial_detour_takes_the_lowest_rated_safe_turn() -> None:
    """With p = 1, at every step away from the destination that has another safe turn, the move played is the safe
    turn the model rates lowest (other than its best); otherwise its best safe move."""
    s, t = _pairs(20)

    class Rec(_Prefers):
        seen: list[np.ndarray] = []

        def _logits(self, tokens):
            Rec.seen.append(tokens.numpy().copy())
            return super()._logits(tokens)

    fr.detour_drive(Rec(), s, t, 1.0, "adversarial", 0, "cpu")
    for start in s[:5]:
        drive = _full_drive(Rec.seen, int(start))
        cur, dest, turns = int(drive[0]), int(drive[1]), 0
        for move in drive[2:]:
            safe = maps.safe_moves(cur, dest, maps.MAX_TURNS - turns)
            best = max(safe)                                              # highest token = highest rated
            alts = [k for k in safe if k != best and k != maps.END]
            assert move == (min(alts) if cur != dest and alts else best)
            cur = int(maps.NEXT[cur, move - maps.TURN0])
            turns += 1


# ==========================================================================
# Least visited intersections
# ==========================================================================

def test_least_visited_matched_drop() -> None:
    d = maps.build(fx.dom.random_routes(300, seed=43))
    rare = tasks.load_rare("shortest_paths")
    sc = d["score"][:, :maps.N_CTX]
    is_rare = np.isin(d["current"][:, :maps.N_CTX], rare) & sc
    assert is_rare.any()
    perfect = fr.least_visited_matched(sc.copy(), d, rare)
    assert perfect["drop"] == 0.0 and perfect["rate"] == 1.0
    worse = fr.least_visited_matched(sc & ~is_rare, d, rare)          # wrong only at rare intersections
    assert worse["drop"] == 1.0 and worse["rate"] == 0.0
    assert worse["n_positions"] + worse["n_left_out"] == int(is_rare.sum())
    better = fr.least_visited_matched(is_rare.copy(), d, rare)         # right only at rare intersections
    assert better["drop"] == -1.0


def test_least_visited_leaves_out_unmatched_cells() -> None:
    d = maps.build(fx.dom.random_routes(40, seed=44))
    sc = d["score"][:, :maps.N_CTX]
    every = np.unique(d["current"][:, :maps.N_CTX][sc])
    out = fr.least_visited_matched(sc.copy(), d, every)                # every position is rare: no baseline
    assert out["n_positions"] == 0 and out["n_left_out"] == int(sc.sum()) and out["drop"] is None


# ==========================================================================
# The whole suite
# ==========================================================================

def test_perfect_model_has_no_drop(data_dir) -> None:
    res = fr.score_model(fx.MapsOracle("legal"), "cpu", "shortest_paths", batch=32, **SMALL)
    assert res["in_distribution_legal_rate"] == 1.0
    assert set(res["drops"]) == set(fr.drop_names("shortest_paths"))
    assert all(v == 0.0 for v in res["drops"].values()) and res["fragility"] == 0.0
    assert res["relative_fragility"] == 0.0 and res["suite_hash"] == fr.suite_hash()


def test_pilot_rule_outcome_leaves_drops_out_of_fragility(data_dir) -> None:
    model = fx.MapsOracle("fixed", fixed=maps.END)
    full = fr.score_model(model, "cpu", "random_walks", batch=32, excluded=[], **SMALL)
    cut = fr.score_model(model, "cpu", "random_walks", batch=32, excluded=["least_visited"], **SMALL)
    assert full["drops"] == cut["drops"] and "least_visited" not in cut["drops_used"]
    want = np.mean([v for k, v in full["drops"].items() if k != "least_visited"])
    assert cut["fragility"] == pytest.approx(want) and cut["pilot_rule_applied"]
    assert cut["relative_fragility"] == pytest.approx(cut["fragility"] / cut["in_distribution_legal_rate"])


def test_untrained_model_end_to_end(data_dir) -> None:
    m = fx.tiny("transformer")
    res = fr.score_model(m, "cpu", "noisy_shortest_paths", batch=32, **SMALL)
    assert not res["pilot_rule_applied"] and len(res["drops"]) == 7
    assert res["detour"]["p=0.0"]["reached_destination_with_end"] == 1.0
    rec = fr.result_record(m, res, 1.0)
    assert rec["model_id"] == fx.ZOO_ID and rec["distribution"] == "shortest_paths" and "git_commit" in rec["provenance"]
    json.dumps(rec)


def test_result_paths() -> None:
    assert fr.result_path(fx.PILOT_ID) == nav.PILOT_RESULTS_DIR / "fragility_maps" / f"{fx.PILOT_ID}.json"
    assert fr.result_path(fx.ZOO_ID) == nav.results.RESULTS_DIR / "fragility_maps" / f"{fx.ZOO_ID}.json"


# ==========================================================================
# Pilot rule
# ==========================================================================

def test_pilot_rule(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for m in nav.pilot_ids():
        dist = nav.parse_id(m)["distribution"]
        drops = {n: float(rng.normal(0, 0.05)) for n in fr.drop_names(dist)}
        drops["detour/random/p=0.1"] = 0.0002 * (hash(m) % 2)          # sd about 0.0001: left out
        (tmp_path / f"{m}.json").write_text(json.dumps({"drops": drops, "suite_hash": fr.suite_hash()}))
    out = fr.pilot_rule(tmp_path)
    assert out["excluded"] == ["detour/random/p=0.1"]
    assert len(out["values"]["cross_distribution/shortest_paths"]) == 6     # pilots of the other two distributions
    v = out["values"]["least_visited"]
    assert out["sd"]["least_visited"] == pytest.approx(np.std(v, ddof=1), abs=1e-6)
    (tmp_path / f"{nav.pilot_ids()[0]}.json").unlink()
    with pytest.raises(SystemExit, match="all 9 pilots"):
        fr.pilot_rule(tmp_path)


def test_zoo_models_need_the_pilot_rule(monkeypatch) -> None:
    monkeypatch.setattr(nav, "missing_tags", lambda tags=nav.GATE_TAGS: [])
    monkeypatch.setattr(nav, "uncommitted_src", lambda: [])
    monkeypatch.setattr(fr, "EXCLUDED_BY_PILOT_RULE", None)
    monkeypatch.setattr(sys, "argv", ["maps.py", "--models", fx.ZOO_ID])
    with pytest.raises(SystemExit, match="pilot rule"):
        fr.main()
