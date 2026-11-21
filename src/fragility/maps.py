"""Navigation fragility suite maps-fragility-v1: the H3 outcome for navigation (preregistration Amendment 5, G8;
ticket N8).

Built on what Othello fragility v2 taught (Amendment 3, E2): every comparison is matched, no score can sit at a
floor, and the route the model drives always stays legal, so one wrong proposal never ends the measurement.
Nothing in the suite retrains or adapts the model; every right answer comes from the map (src/domains/maps.py).

Variants and drops (7 drops)
    in distribution     legal rate (D1 rule) on 1,000 test routes of the model's own distribution, data seed 0:
                        the reference
    cross distribution  legal rate on 1,000 test routes of each other distribution (pairs no model trained on);
                        drop = in distribution minus it (2 drops)
    detour              the model drives from the start to the destination of 500 test routes of its own
                        distribution (seed 0). At every step its top token is scored legal or not. The move played
                        is its highest rated SAFE move (legal, and the destination still reachable within 96 turns
                        in all). With probability p (detour draws seed 0; never at the destination) that move is
                        replaced by another safe turn: a uniformly random one ("random") or the one the model rates
                        lowest ("adversarial"); it is kept when there is no other safe turn. The drive ends at end
                        or at 96 turns. drop = legal rate at p = 0 minus legal rate at p, p = 0.1 and 0.5 (4 drops)
    least visited       positions at the least visited intersections of the model's distribution
                        (src/domains/maps_rare_intersections.json) on 5,000 test routes (seed 0). Each is compared
                        with the other positions of the same cell (number of legal next tokens, at the destination
                        or not, block of 10 turns so far) in the same routes: drop = mean over those positions of
                        the cell's legal rate minus their own; positions whose cell has no other position are left
                        out and counted (1 drop)
    fragility           mean of the drops that the pilot rule keeps; relative fragility = fragility divided by the
                        in distribution legal rate (the H3 outcome, as for Othello)
    descriptive         "reached the destination with end" for every drive (safe moves make it 1 by construction)

Pilot rule (G8), fixed before any number was seen
    the suite is run on the 9 pilots; a variant whose drop has a standard deviation (sample, ddof 1) below 0.001
    across them is reported for every model but left out of `fragility`. `--pilot_rule` computes it and writes
    results/prereg/maps_fragility_pilot_rule.json; its outcome is then written into EXCLUDED_BY_PILOT_RULE below
    and into src/docs/maps_fragility_suite.md, and only then is the tag maps-fragility-v1 made. Zoo models are
    refused while EXCLUDED_BY_PILOT_RULE is None, and (src/diagnostics/navigation.py) until both tags exist.

Results (one JSON per model)
    results/fragility_maps/{model_id}.json; pilots: results/pilot_scoring/fragility_maps/{model_id}.json

Usage (EC2, repo root):
    python src/fragility/maps.py --models maps_transformer_small_shortest_paths_s100 --save
    python src/fragility/maps.py --pilot_rule
Doc: src/docs/maps_fragility_suite.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("fragility_maps")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nav = _load("navigation", "src/diagnostics/navigation.py")
maps = nav.maps
maps_tasks = _load("maps_tasks", "src/domains/maps_tasks.py")
results = nav.results

# ==========================================================================
# The frozen definition. Changing anything here changes suite_hash() and fails the freeze test once the tag
# maps-fragility-v1 exists. After the tag, add a v2 instead of editing.
# ==========================================================================

SUITE_VERSION = "maps-fragility-v1"
SUITE: dict = {
    "n_test_routes": 1000,
    "data_seed": 0,
    "cross_distribution": list(maps.DISTRIBUTIONS),
    "detour": {
        "n_routes": 500,
        "seed": 0,
        "modes": ["random", "adversarial"],
        "probabilities": [0.0, 0.1, 0.5],
        "max_turns": maps.MAX_TURNS,
        "score": "legal rate of the top token at every step of the drive",
        "move_played": "the highest rated safe move (legal, destination reachable within max_turns in all); with "
                       "probability p, never at the destination, replaced by another safe turn (random, or the one "
                       "rated lowest), kept when there is no other",
    },
    "least_visited": {
        "n_test_routes": 5000,
        "seed": 0,
        "rare_file": "src/domains/maps_rare_intersections.json",
        "rare_file_sha256": maps.sha256(maps_tasks.RARE_FILE),
        "cell": "(number of legal next tokens, at the destination, block of 10 turns so far)",
        "drop": "mean over rare positions of (legal rate of the non rare positions of the same cell minus their own)",
    },
    "pilot_rule": {"sd_below": 0.001, "ddof": 1, "n_pilots": 9},
    "fragility": "mean of the drops not excluded by the pilot rule",
    "relative_fragility": "fragility / in distribution legal rate",
}

# The pilot rule's outcome (G8): the drop names left out of `fragility`. None until the rule has been run on the
# 9 pilots; then a list (possibly empty), written here and in src/docs/maps_fragility_suite.md before the tag.
EXCLUDED_BY_PILOT_RULE: list[str] | None = None

RESULTS_FOLDER = "fragility_maps"
PILOT_RULE_FILE = ROOT / "results" / "prereg" / "maps_fragility_pilot_rule.json"


def drop_names(train_distribution: str) -> list[str]:
    """The 7 drops of a model trained on `train_distribution`, in order."""
    out = [f"cross_distribution/{d}" for d in SUITE["cross_distribution"] if d != train_distribution]
    det = SUITE["detour"]
    out += [f"detour/{m}/p={p}" for m in det["modes"] for p in det["probabilities"] if p > 0]
    return out + ["least_visited"]


def all_drop_names() -> list[str]:
    """Every drop name over the three distributions (a cross distribution drop exists for 2 of them each)."""
    names: list[str] = []
    for d in maps.DISTRIBUTIONS:
        names += [n for n in drop_names(d) if n not in names]
    return names


def suite_hash() -> str:
    blob = json.dumps({"version": SUITE_VERSION, "suite": SUITE, "excluded": EXCLUDED_BY_PILOT_RULE}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ==========================================================================
# Detour drives
# ==========================================================================

@torch.no_grad()
def _last_logits(model, tokens: np.ndarray, device: str, batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(tokens), batch):
        x = torch.as_tensor(tokens[i:i + batch], device=device)
        out.append(model(x)[:, -1].float().cpu().numpy())
    return np.concatenate(out)


def detour_drive(model, starts: np.ndarray, dests: np.ndarray, p: float, mode: str, seed: int, device: str,
                 batch: int = nav.BATCH, max_turns: int = maps.MAX_TURNS) -> dict[str, Any]:
    """The model drives from each start to its destination (tokens of the canonical map). At every step its top
    token is scored legal or not; the move played is its highest rated safe move, replaced with probability p by
    another safe turn (mode random or adversarial), never at the destination. Drives run side by side; the uniform
    draws of one rollout come from one generator seeded `seed`, in drive order at each step, and are made only
    when another safe turn exists (as Othello v2 draws only when another legal move exists)."""
    if mode not in ("random", "adversarial"):
        raise ValueError(f"mode must be random or adversarial, not {mode!r}")
    n = len(starts)
    rng = np.random.default_rng(seed)
    tokens = np.zeros((n, max_turns + 3), dtype=np.int64)
    tokens[:, 0], tokens[:, 1] = starts, dests
    cur = np.asarray(starts, dtype=np.int64).copy()
    turns = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    reached = np.zeros(n, dtype=bool)
    stuck = np.zeros(n, dtype=bool)
    n_pred = n_legal = n_detours = 0
    legal_by_step: list[tuple[int, int]] = []
    for step in range(max_turns + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        t = 2 + step
        logits = _last_logits(model, tokens[idx, :t], device, batch)
        top = logits.argmax(-1)
        ok_step = 0
        for row, g in enumerate(idx):
            c, d = int(cur[g]), int(dests[g])
            n_pred += 1
            if int(top[row]) in maps.legal_next(c, d):
                n_legal += 1
                ok_step += 1
            safe = maps.safe_moves(c, d, max_turns - int(turns[g]))
            if not safe:                       # cannot happen from a test pair (it is reachable); kept as a guard
                alive[g] = False
                stuck[g] = True
                continue
            best = max(safe, key=lambda tk: logits[row, tk])
            move = best
            if c != d:
                alts = [tk for tk in safe if tk != best and tk != maps.END]
                if alts and rng.random() < p:
                    move = int(rng.choice(alts)) if mode == "random" else min(alts, key=lambda tk: logits[row, tk])
                    n_detours += 1
            tokens[g, t] = move
            if move == maps.END:
                alive[g] = False
                reached[g] = c == d
            else:
                cur[g] = int(maps.NEXT[c, move - maps.TURN0])
                turns[g] += 1
        legal_by_step.append((ok_step, len(idx)))
    return {"legal_rate": n_legal / max(n_pred, 1), "n_drives": int(n), "n_predictions": int(n_pred),
            "n_detours": int(n_detours), "mean_turns": float(turns.mean()),
            "reached_destination_with_end": float(reached.mean()), "stuck": int(stuck.sum()),
            "legal_rate_by_step": [round(a / b, 5) for a, b in legal_by_step]}


# ==========================================================================
# Least visited intersections, matched by cell
# ==========================================================================

def least_visited_matched(correct: np.ndarray, data: dict, rare: np.ndarray) -> dict[str, Any]:
    """Legal rate at the rare positions and its matched baseline. correct (G, N_CTX) is nav.correct_at. A rare
    position's baseline is the legal rate of the non rare scored positions of its cell (maps_tasks.matched_cells)
    in the same routes; the drop is the mean over rare positions of baseline minus own; rare positions whose cell
    has no non rare position are left out and counted."""
    sc = data["score"][:, :maps.N_CTX]
    is_rare = np.isin(data["current"][:, :maps.N_CTX], rare) & sc
    non = sc & ~is_rare
    cells = maps_tasks.matched_cells(data)[:, :maps.N_CTX]
    own = correct[is_rare].astype(float)
    if not non.any() or not is_rare.any():
        return {"rate": None, "baseline": None, "drop": None, "n_positions": 0, "n_left_out": int(len(own))}
    keys, inv = np.unique(cells[non], return_inverse=True)
    rate = np.bincount(inv, weights=correct[non].astype(float)) / np.bincount(inv)
    rare_cells = cells[is_rare]
    where = np.searchsorted(keys, rare_cells)
    has = (where < len(keys)) & (keys[np.minimum(where, len(keys) - 1)] == rare_cells)
    if not has.any():
        return {"rate": None, "baseline": None, "drop": None, "n_positions": 0, "n_left_out": int(len(own))}
    base = rate[where[has]]
    return {"rate": float(own[has].mean()), "baseline": float(base.mean()),
            "drop": float((base - own[has]).mean()), "n_positions": int(has.sum()),
            "n_left_out": int((~has).sum()), "n_rare_intersections": int(len(rare))}


# ==========================================================================
# The whole suite for one model
# ==========================================================================

def score_model(model, device: str, train_distribution: str, batch: int = nav.BATCH,
                excluded: list[str] | None = None, n_detour_routes: int | None = None,
                n_test_routes: int | None = None, n_rare_routes: int | None = None) -> dict[str, Any]:
    """Every variant on one model. excluded = the pilot rule's outcome (EXCLUDED_BY_PILOT_RULE; None before the rule,
    when every drop counts). The three n_ arguments are for tests and smoke runs only."""
    if excluded is None:
        excluded = EXCLUDED_BY_PILOT_RULE
    applied = excluded is not None
    seed = SUITE["data_seed"]
    n = n_test_routes or SUITE["n_test_routes"]
    own = nav.routes(train_distribution, "test", n, seed=seed)
    ref = maps.legal_rate(model, own, device, batch)
    rates: dict[str, float] = {"in_distribution": ref}
    counts: dict[str, int] = {"in_distribution": int(own["score"][:, :maps.N_CTX].sum())}
    drops: dict[str, float] = {}
    for dist in SUITE["cross_distribution"]:
        if dist == train_distribution:
            continue
        data = nav.routes(dist, "test", n, seed=seed)
        name = f"cross_distribution/{dist}"
        rates[name] = maps.legal_rate(model, data, device, batch)
        counts[name] = int(data["score"][:, :maps.N_CTX].sum())
        drops[name] = ref - rates[name]

    det = SUITE["detour"]
    pairs = nav.routes(train_distribution, "test", n_detour_routes or det["n_routes"], seed=det["seed"])
    starts, dests = pairs["tokens"][:, 0], pairs["dest"]
    detour: dict[str, dict] = {"p=0.0": detour_drive(model, starts, dests, 0.0, "random", det["seed"], device, batch)}
    undisturbed = detour["p=0.0"]["legal_rate"]
    for mode in det["modes"]:
        for p in det["probabilities"]:
            if p > 0:
                detour[f"{mode}/p={p}"] = detour_drive(model, starts, dests, p, mode, det["seed"], device, batch)
                drops[f"detour/{mode}/p={p}"] = undisturbed - detour[f"{mode}/p={p}"]["legal_rate"]

    lv = SUITE["least_visited"]
    data = nav.routes(train_distribution, "test", n_rare_routes or lv["n_test_routes"], seed=lv["seed"])
    least = least_visited_matched(nav.correct_at(model, data, batch), data, maps_tasks.load_rare(train_distribution))
    if least["n_positions"]:
        rates["least_visited"], counts["least_visited"] = least["rate"], least["n_positions"]
        drops["least_visited"] = least["drop"]

    used = {k: v for k, v in drops.items() if k not in (excluded or [])}
    fragility = float(np.mean(list(used.values()))) if used else float("nan")
    return {
        "suite_version": SUITE_VERSION, "suite_hash": suite_hash(),
        "train_distribution": train_distribution,
        "in_distribution_legal_rate": ref,
        "legal_rate": rates, "positions_scored": counts,
        "detour": detour, "least_visited": least,
        "drops": drops,
        "pilot_rule_applied": applied,
        "excluded_by_pilot_rule": list(excluded or []),
        "drops_used": sorted(used),
        "fragility": fragility,
        "relative_fragility": fragility / ref if ref > 0 else None,
    }


def result_record(model, res: dict, runtime_sec: float) -> dict[str, Any]:
    """The JSON written per model: its id and fields, the suite output and where it was produced."""
    return {"model_id": model.model_id, **nav.model_fields(model), **res,
            "provenance": results.provenance(round(runtime_sec, 1))}


def result_path(model_id: str) -> Path:
    return nav.results_dir(model_id) / RESULTS_FOLDER / f"{model_id}.json"


# ==========================================================================
# Pilot rule (G8)
# ==========================================================================

def pilot_rule(folder: Path | None = None) -> dict[str, Any]:
    """The standard deviation (ddof 1) of every drop across the 9 pilots' suite results, and the drops below 0.001.
    A cross distribution drop is computed over the pilots that have it (6 of the 9)."""
    folder = folder or nav.PILOT_RESULTS_DIR / RESULTS_FOLDER
    missing = [m for m in nav.pilot_ids() if not (folder / f"{m}.json").exists()]
    if missing:
        raise SystemExit(f"the pilot rule needs the suite results of all 9 pilots; missing in {folder}: {missing}")
    res = {m: json.loads((folder / f"{m}.json").read_text()) for m in nav.pilot_ids()}
    rule = SUITE["pilot_rule"]
    sd, values = {}, {}
    for name in all_drop_names():
        v = [r["drops"][name] for r in res.values() if name in r["drops"]]
        values[name] = [round(float(x), 6) for x in v]
        sd[name] = float(np.std(v, ddof=rule["ddof"])) if len(v) > 1 else 0.0
    excluded = [name for name in all_drop_names() if sd[name] < rule["sd_below"]]
    return {"rule": rule, "sd": {k: round(v, 6) for k, v in sd.items()}, "values": values,
            "excluded": excluded, "suite_hash_before": suite_hash(),
            "pilot_suite_hashes": sorted({r["suite_hash"] for r in res.values()}),
            "git_commit": results._git_commit()}


# ==========================================================================
# Command line
# ==========================================================================

def score_one(model_id: str, save: bool, batch: int, device: str | None = None) -> dict[str, Any]:
    model = nav.load(model_id, device)
    t0 = time.time()
    res = score_model(model, model.device, nav.training_distribution(model), batch)
    rec = result_record(model, res, time.time() - t0)
    if save:
        path = result_path(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2) + "\n")
        log.info("saved %s", path)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Navigation fragility suite maps-fragility-v1")
    ap.add_argument("--models", nargs="+", help="navigation ids: zoo, pilot (seed 100) or maps_vafa_*")
    ap.add_argument("--save", action="store_true", help="write results/fragility_maps/<id>.json (pilots: results/pilot_scoring/)")
    ap.add_argument("--batch", type=int, default=nav.BATCH)
    ap.add_argument("--pilot_rule", action="store_true", help="compute the G8 pilot rule from the 9 pilot results")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if a.pilot_rule:
        out = pilot_rule()
        PILOT_RULE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PILOT_RULE_FILE.write_text(json.dumps(out, indent=2) + "\n")
        for name, s in out["sd"].items():
            print(f"  {name:40s} sd {s:.6f}  {'LEFT OUT' if name in out['excluded'] else 'kept'}")
        print(f"excluded = {out['excluded']}; written to {PILOT_RULE_FILE}. Now set EXCLUDED_BY_PILOT_RULE = "
              f"{out['excluded']} in src/fragility/maps.py and in src/docs/maps_fragility_suite.md, then tag {SUITE_VERSION}")
        return
    if not a.models:
        ap.error("--models is required (or --pilot_rule)")
    nav.check_can_score(a.models, a.save)
    zoo = [m for m in a.models if nav.kind(m) == "zoo"]
    if zoo and EXCLUDED_BY_PILOT_RULE is None:
        sys.exit("refusing to score zoo models: the pilot rule has not been applied (EXCLUDED_BY_PILOT_RULE is None)")
    log.info("fragility suite %s (%s)", SUITE_VERSION, suite_hash())
    failed = []
    for m in a.models:
        try:
            rec = score_one(m, a.save, a.batch)
        except Exception as e:  # noqa: BLE001  one model must not stop the others
            log.warning("%s failed: %s", m, e)
            failed.append((m, f"{type(e).__name__}: {e}"))
            continue
        print(f"{m:45s} in distribution {rec['in_distribution_legal_rate']:.4f}  fragility {rec['fragility']:+.4f}  "
              f"relative {rec['relative_fragility'] if rec['relative_fragility'] is None else round(rec['relative_fragility'], 4)}")
        for name, d in rec["drops"].items():
            print(f"{'':45s}   {name:38s} drop {d:+.4f}{'  (left out)' if name in rec['excluded_by_pilot_rule'] else ''}")
    nav.report_not_loaded(failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
