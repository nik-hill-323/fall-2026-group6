"""Navigation definitions that Amendment 5 fixes from the data alone, before any navigation model exists:
the D3 task labels and their shortcut rules, and the rare intersection sets of the fragility suite.

N6 (D3) and N8 (fragility) import these functions, so the preregistered definitions and the scored ones are
the same code. Design: src/docs/navigation_design.md, sections 4 (D3) and 5.1 (rare intersections).

D3 tasks, about the state at one scored position (current intersection c, destination d):
    T1  the destination is at most H_TURNS turns away on the canonical map:     turn_distance(c, d) <= H_TURNS
    T2  the destination is at most R_METRES away in a straight line:            haversine(c, d) <= R_METRES
    T3  the current intersection is north of the destination:                   lat(c) > lat(d)
    T4  the current intersection is east of the destination:                    lon(c) > lon(d)
Shortcut rules (answers that do not need the current intersection):
    T1, T2  a threshold on the number of turns so far (the best threshold, either direction: an upper bound)
    T3, T4  the same comparison for the start instead of the current intersection
Drop rule, fixed before any number below was computed: a task is dropped if, in any distribution, its
positive rate is below 0.10 or above 0.90, or its shortcut rule reaches a balanced accuracy of at least 0.90.

H_TURNS and R_METRES are the medians over 30,000 positions: 10,000 validation routes per distribution (seed 0),
one uniformly random scored position per route (seed 0), the way D3 draws its positions.

Rare intersections (fragility): per distribution, the least visited intersections of that distribution's
training cache (a visit = the current intersection at a scored position), taken in order of visits (ties by token
id) until together they hold at least 10% of the scored positions of that distribution's test split. Fixed per
distribution, so every model of a distribution uses the same set. A first definition, the 10% of the 4,579
intersections with the fewest visits, was replaced before any navigation model existed: those intersections are
almost never on a route (6 such positions in 5,000 shortest paths test routes, 38 for noisy shortest paths), so
the variant could not be measured.

Run on the EC2 (needs the built data), from the repo root:
    python src/domains/maps_tasks.py
It writes results/prereg/maps_amendment5_numbers.json and src/domains/maps_rare_intersections.json.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("maps_tasks")


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


maps = _load("maps", "src/domains/maps.py")

TASKS = ["T1", "T2", "T3", "T4"]
TASK_SEED = 0              # route draw and position draw for the thresholds (D3 uses seed 0 as well)
N_THRESHOLD_ROUTES = 10_000
RARE_SHARE = 0.10          # share of the scored test positions the rare set must hold
N_RARE_TEST_ROUTES = 5_000
MIN_RARE_POSITIONS = 200
DROP_POSITIVE_RATE = (0.10, 0.90)
DROP_SHORTCUT_BALANCED = 0.90
EARTH_RADIUS_M = 6_371_008.8

# Computed by compute() at commit 7ed6331 (results/prereg/maps_amendment5_numbers.json) and frozen by Amendment 5,
# G6 (prereg-v1.5). A test checks that these equal the numbers file.
H_TURNS: int | None = 10
R_METRES: float | None = 823.9
D3_TASKS = ["T1", "T2", "T4"]   # T3 dropped by the drop rule (shortcut 0.922 shortest paths, 0.908 noisy)
RARE_FILE = ROOT / "src/domains/maps_rare_intersections.json"
NUMBERS_FILE = ROOT / "results/prereg/maps_amendment5_numbers.json"


# ==========================================================================
# Positions and labels
# ==========================================================================

def random_positions(data: dict, seed: int = TASK_SEED) -> np.ndarray:
    """One uniformly random scored position per route (1 .. length - 2)."""
    rng = np.random.default_rng(seed)
    n_scored = data["length"] - 2
    if (n_scored < 1).any():
        raise ValueError("every route needs at least one scored position")
    return 1 + (rng.random(len(n_scored)) * n_scored).astype(np.int64)


def haversine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Straight line distance in metres between intersection tokens a and b."""
    lat, lon = maps.coordinates()
    p1, p2 = np.radians(lat[a]), np.radians(lat[b])
    dp, dl = p2 - p1, np.radians(lon[b] - lon[a])
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


def state_at(data: dict, pos: np.ndarray) -> dict[str, np.ndarray]:
    """Current intersection, destination, start and turns so far at one position per route."""
    g = np.arange(len(pos))
    cur = data["current"][g, pos]
    if (cur == 0).any():
        raise ValueError("a position is not a scored position")
    return {"current": cur, "dest": data["dest"], "start": data["tokens"][:, 0], "turns": data["n_turns"][g, pos]}


def turn_distance_to_destination(s: dict) -> np.ndarray:
    d = maps.turn_distance()[s["current"], s["dest"]].astype(np.int64)
    if (d < 0).any():
        raise ValueError("a destination cannot be reached from a scored position: the route is not legal")
    return d


def labels(s: dict, h_turns: int | None = None, r_metres: float | None = None) -> dict[str, np.ndarray]:
    """Boolean labels of T1 to T4 at the given states."""
    h = H_TURNS if h_turns is None else h_turns
    r = R_METRES if r_metres is None else r_metres
    if h is None or r is None:
        raise ValueError("H_TURNS and R_METRES are not fixed yet (Amendment 5)")
    lat, lon = maps.coordinates()
    return {"T1": turn_distance_to_destination(s) <= h,
            "T2": haversine(s["current"], s["dest"]) <= r,
            "T3": lat[s["current"]] > lat[s["dest"]],
            "T4": lon[s["current"]] > lon[s["dest"]]}


def start_shortcuts(s: dict) -> dict[str, np.ndarray]:
    """The T3 and T4 shortcut answers: the start instead of the current intersection."""
    lat, lon = maps.coordinates()
    return {"T3": lat[s["start"]] > lat[s["dest"]], "T4": lon[s["start"]] > lon[s["dest"]]}


def balanced_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = truth, ~truth
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (float(pred[pos].mean()) + float((~pred[neg]).mean()))


def best_threshold_balanced(x: np.ndarray, truth: np.ndarray) -> tuple[float, int, str]:
    """Best balanced accuracy of a rule 'x <= t' or 'x > t' over every threshold t (an upper bound)."""
    best = (0.0, 0, "<=")
    for t in np.unique(x):
        for side, pred in (("<=", x <= t), (">", x > t)):
            b = balanced_accuracy(pred, truth)
            if b == b and b > best[0]:
                best = (b, int(t), side)
    return best


# ==========================================================================
# Rare intersections
# ==========================================================================

def visit_counts(cache, chunk: int = 100_000) -> np.ndarray:
    """(N_NODES + 1,) visits of each intersection token at scored positions over a training cache."""
    counts = np.zeros(maps.N_NODES + 1, dtype=np.int64)
    for i in range(0, len(cache), chunk):
        d = maps.build(np.asarray(cache[i:i + chunk], dtype=np.int64))
        if d["n_illegal"]:
            raise ValueError("the cache holds illegal routes")
        counts += np.bincount(d["current"][d["score"]], minlength=maps.N_NODES + 1)
    return counts


def rare_set(counts: np.ndarray, test_counts: np.ndarray, share: float = RARE_SHARE) -> np.ndarray:
    """Token ids of the least visited intersections (training visits `counts`, ties by token id), taken in that
    order until they hold at least `share` of the scored test positions (`test_counts`, visits in the test split)."""
    tokens = np.arange(1, maps.N_NODES + 1)
    order = np.lexsort((tokens, counts[1:]))      # by training visits, then by token id
    held = np.cumsum(test_counts[1:][order])
    need = share * test_counts[1:].sum()
    n = int(np.searchsorted(held, need, side="left")) + 1
    return np.sort(tokens[order[:n]])


def load_rare(distribution: str) -> np.ndarray:
    return np.array(json.loads(RARE_FILE.read_text())["rare"][distribution], dtype=np.int64)


def matched_cells(data: dict) -> np.ndarray:
    """(G, MAX_TOKENS) cell id of every position for the rare intersection comparison: (number of legal next
    tokens, at the destination, block of 10 turns so far). Only meaningful where data['score'] is True."""
    n_legal = data["legal"].sum(-1)
    at_dest = data["current"] == data["dest"][:, None]
    block = data["n_turns"] // 10
    return (n_legal * 2 + at_dest) * 100 + block


# ==========================================================================
# The numbers Amendment 5 fixes
# ==========================================================================

def compute(out_numbers: Path = NUMBERS_FILE, out_rare: Path = RARE_FILE) -> dict:
    t0 = time.time()
    states = {}
    for dist in maps.DISTRIBUTIONS:
        val = maps.load_distribution(dist, "val", N_THRESHOLD_ROUTES, seed=TASK_SEED)
        data = maps.build(val)
        if data["n_illegal"]:
            raise SystemExit(f"{dist} val holds illegal routes")
        states[dist] = state_at(data, random_positions(data))
    pooled_turns = np.concatenate([turn_distance_to_destination(s) for s in states.values()])
    pooled_metres = np.concatenate([haversine(s["current"], s["dest"]) for s in states.values()])
    h = int(np.median(pooled_turns))
    r = float(np.round(np.median(pooled_metres), 1))
    if H_TURNS is not None and (h, r) != (H_TURNS, R_METRES):
        log.warning("recomputed thresholds (%s, %s) differ from the frozen ones (%s, %s)", h, r, H_TURNS, R_METRES)
    out: dict = {"H_TURNS": h, "R_METRES": r, "positions_per_distribution": N_THRESHOLD_ROUTES,
                 "drop_rule": {"positive_rate_outside": DROP_POSITIVE_RATE, "shortcut_balanced_at_least": DROP_SHORTCUT_BALANCED},
                 "tasks": {}, "rare": {}}
    for t in TASKS:
        out["tasks"][t] = {"per_distribution": {}}
    for dist, s in states.items():
        lab = labels(s, h, r)
        sc = start_shortcuts(s)
        for t in TASKS:
            truth = lab[t]
            if t in ("T1", "T2"):
                b, thr, side = best_threshold_balanced(s["turns"], truth)
                shortcut = {"rule": f"turns so far {side} {thr}", "balanced_accuracy": round(b, 4)}
            else:
                shortcut = {"rule": "start instead of current", "balanced_accuracy": round(balanced_accuracy(sc[t], truth), 4)}
            out["tasks"][t]["per_distribution"][dist] = {"positive_rate": round(float(truth.mean()), 4), "shortcut": shortcut}
    for t in TASKS:
        per = out["tasks"][t]["per_distribution"].values()
        reasons = []
        for p in per:
            if not DROP_POSITIVE_RATE[0] <= p["positive_rate"] <= DROP_POSITIVE_RATE[1]:
                reasons.append(f"positive rate {p['positive_rate']}")
            if p["shortcut"]["balanced_accuracy"] >= DROP_SHORTCUT_BALANCED:
                reasons.append(f"shortcut {p['shortcut']['balanced_accuracy']}")
        out["tasks"][t]["kept"] = not reasons
        out["tasks"][t]["drop_reasons"] = reasons
    rare_sets = {}
    for dist in maps.DISTRIBUTIONS:
        counts = visit_counts(maps.load_distribution(dist, "train"))
        test_counts = visit_counts(maps.load_distribution(dist, "test"))
        rare = rare_set(counts, test_counts)
        rare_sets[dist] = rare.tolist()
        test = maps.build(maps.load_distribution(dist, "test", N_RARE_TEST_ROUTES, seed=0))
        is_rare = np.isin(test["current"], rare) & test["score"]
        cells = matched_cells(test)
        base_cells = set(np.unique(cells[test["score"] & ~is_rare]).tolist())
        covered = np.array([c in base_cells for c in cells[is_rare]])
        out["rare"][dist] = {"n_rare_intersections": int(len(rare)),
                             "share_of_test_positions_held": round(float(test_counts[rare].sum() / test_counts[1:].sum()), 4),
                             "max_visits_in_rare_set": int(counts[rare].max()),
                             "median_visits_all": float(np.median(counts[1:])),
                             "never_visited": int((counts[1:] == 0).sum()),
                             "rare_positions_in_5000_test_routes": int(is_rare.sum()),
                             "share_with_a_matched_cell": round(float(covered.mean()) if len(covered) else 0.0, 4),
                             "enough": bool(is_rare.sum() >= MIN_RARE_POSITIONS)}
        log.info("%s: %d rare intersections (max %d visits), %d rare test positions", dist, len(rare),
                 counts[rare].max(), is_rare.sum())
    mb = _load("maps_build", "src/domains/maps_build.py")
    out["git_commit"] = mb.git_commit()
    out["data_manifest_sha256"] = maps.sha256(maps.MANIFEST) if maps.MANIFEST.exists() else None
    out["seconds"] = round(time.time() - t0, 1)
    out_numbers.parent.mkdir(parents=True, exist_ok=True)
    out_numbers.write_text(json.dumps(out, indent=2) + "\n")
    out_rare.write_text(json.dumps({"share": RARE_SHARE, "rare": rare_sets}, indent=1) + "\n")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dirty = _load("maps_build", "src/domains/maps_build.py").uncommitted_code()
    if dirty:
        sys.exit(f"refusing to run: these files under src/ are not committed: {dirty}")
    bad = [f for f, ok in maps.verify_data().items() if not ok]
    if bad:
        sys.exit(f"these data files do not match the manifest: {bad}")
    res = compute()
    print(f"H_TURNS = {res['H_TURNS']}, R_METRES = {res['R_METRES']}, commit {res['git_commit']}")
    for t in TASKS:
        per = res["tasks"][t]["per_distribution"]
        cells = "; ".join(f"{d}: positive {v['positive_rate']}, shortcut {v['shortcut']['balanced_accuracy']} ({v['shortcut']['rule']})"
                          for d, v in per.items())
        print(f"{t} {'KEPT' if res['tasks'][t]['kept'] else 'DROPPED ' + str(res['tasks'][t]['drop_reasons'])} | {cells}")
    for d, v in res["rare"].items():
        print(f"rare {d}: {v}")
