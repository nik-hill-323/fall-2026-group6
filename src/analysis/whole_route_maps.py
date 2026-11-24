"""Whole route measure for navigation (preregistration Amendment 5, G9; preregistered secondary; ticket N12).

D1 and the fragility suite score one next token at a time. This measure looks at whole routes, in two ways, as
the Othello whole game measure does (src/analysis/sequence_othello.py, v3), but preregistered here before any
navigation zoo model was scored.

1. Teacher forced. D1's own data: 1,000 test routes of the model's training distribution, data seed 0, the same
   batch (256) and the same scored positions. At every position the top token is checked against the true legal
   next set. Reported: the share of routes in which every top token is legal, the legal rate by step and overall.
   The overall rate must equal D1 (maps.legal_rate on the same routes, computed here as a check); a difference
   stops the run. When this model's D1 result file exists, its score is compared too and the result recorded.
2. Free play. From the start and destination of the same 1,000 test routes (seed 0) the model drives itself.
   At every step its top token is scored against the true legal next set. The move played is drawn from the
   model's own probabilities over the legal next set only (softmax of its logits restricted to the legal tokens,
   temperature 1; play seed 0, one uniform draw per live drive per step, in drive order), so the route it reads
   is always legal and one wrong top token never ends the measurement. A drive ends
       at `end`                  only legal at the destination: "reached the destination"
       at a dead end             an intersection with no turn out that is not the destination: "stuck". The drive
                                 stops on arrival, before a prediction there (every token would be illegal)
       at 96 turns               the length limit (routes are at most 99 tokens): its last top token, at 96 turns,
                                 is scored; if the move drawn there is a turn, the drive ends without arriving
   Reported: share of drives with no illegal top token, share that reached the destination, share stuck at a
   dead end, share stopped by the length limit, legal rate by step and overall, errors per drive, and the step
   of the first error.

Steps: step s is the prediction made after reading s + 1 tokens (start, destination and s - 1 moves), the same
position p = s as in maps.build and D1. Step 1 predicts the first turn; at most step 97 (96 turns read).

The secondary analysis of G9 (Spearman of each diagnostic with the share of free play drives with no illegal top
token, configuration bootstrap of Amendment 4, F1) is part of the navigation analysis (N14), not this file.

Results (one JSON per model, `status` ok or failed with the reason)
    results/whole_route_maps/{model_id}.json; pilots: results/pilot_scoring/whole_route_maps/{model_id}.json

Usage (EC2, repo root, conda env wmconcord):
    python src/analysis/whole_route_maps.py --models maps_transformer_small_shortest_paths_s100 --save
    JOBS=G9 bash src/shellscripts/maps_score_slice.sh <slice>         # every model of a slice
Doc: src/docs/navigation_scoring.md, "Whole route measure (G9)". Tests: src/tests/test_whole_route_maps.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("whole_route_maps")

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
results = nav.results

MEASURE = "G9"
VERSION = "whole-route-maps-v1"
RESULTS_FOLDER = "whole_route_maps"
D1_FOLDER = "D1_next_token"
N_ROUTES = 1000          # D1's routes (G4) and the free play pairs (G9)
DATA_SEED = 0
PLAY_SEED = 0
SPLIT = "test"
BATCH = nav.BATCH        # 256, as D1, so the teacher forced rate equals D1 exactly
MAX_TURNS = maps.MAX_TURNS
D1_TOLERANCE = 1e-12

REACHED, STUCK, LENGTH_LIMIT = 1, 2, 3
OUTCOMES = {REACHED: "reached_destination", STUCK: "stuck_at_dead_end", LENGTH_LIMIT: "length_limit"}

METHOD = ("whole route measure (Amendment 5, G9): teacher forced on D1's 1,000 test routes (share of routes with every "
          "top token legal, legal rate by step); free play from the same 1,000 start and destination pairs, top token "
          "scored at every step, move drawn from the model's probabilities over the legal next set (seed 0), ends at "
          "end, at a dead end that is not the destination, or at 96 turns")


def _step_stats(ok: np.ndarray, n: np.ndarray) -> list[list[int]]:
    """[[step, n_legal, n_predictions], ...] for every step with at least one prediction."""
    return [[int(s), int(ok[s]), int(n[s])] for s in range(len(n)) if n[s] > 0]


def _first_error_summary(first: np.ndarray) -> dict[str, Any]:
    """first: (G,) step of each route's or drive's first illegal top token, 0 where there is none."""
    hit = first[first > 0]
    steps, counts = np.unique(hit, return_counts=True)
    return {"n_with_error": int(len(hit)),
            "mean": float(hit.mean()) if len(hit) else None,
            "median": float(np.median(hit)) if len(hit) else None,
            "histogram": {str(int(s)): int(c) for s, c in zip(steps, counts)}}


# ==========================================================================
# 1. Teacher forced
# ==========================================================================

@torch.no_grad()
def teacher_forced(model, data: dict, batch: int = BATCH) -> dict[str, Any]:
    """Top token legality along the real routes (D1's positions). Raises if the overall rate is not maps.legal_rate."""
    correct = nav.correct_at(model, data, batch)                 # (G, N_CTX): scored and legal
    sc = data["score"][:, :maps.N_CTX]
    n_pred, n_ok = int(sc.sum()), int(correct.sum())
    rate = n_ok / max(n_pred, 1)
    d1_rate = maps.legal_rate(model, data, model.device, batch)
    if abs(rate - d1_rate) > D1_TOLERANCE:
        raise ValueError(f"teacher forced legal rate {rate!r} differs from the D1 computation {d1_rate!r}")
    wrong = sc & ~correct
    all_legal = ~wrong.any(1)
    first = np.where(wrong.any(1), wrong.argmax(1), 0)          # position p = step s
    return {"n_routes": int(len(sc)), "n_predictions": n_pred, "legal_rate": rate, "d1_rate_recomputed": d1_rate,
            "share_routes_all_legal": float(all_legal.mean()),
            "errors_per_route": float(wrong.sum(1).mean()),
            "first_error_step": _first_error_summary(first),
            "legal_rate_by_step": _step_stats(correct.sum(0), sc.sum(0))}


# ==========================================================================
# 2. Free play
# ==========================================================================

@torch.no_grad()
def _last_logits(model, tokens: np.ndarray, batch: int) -> np.ndarray:
    """Logits after the last token of each prefix, float64, as the fragility suite's detour drives read them."""
    out = []
    for i in range(0, len(tokens), batch):
        x = torch.as_tensor(tokens[i:i + batch], dtype=torch.long)
        out.append(model(x)[:, -1].double().cpu().numpy())
    return np.concatenate(out)


def legal_slots(cur: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """(n, 9) bool: the legal next moves (TURNS order, then end) at intersection tokens cur with destinations dest."""
    out = np.zeros((len(cur), len(maps.SLOT_TOKEN)), dtype=bool)
    out[:, :len(maps.TURNS)] = maps.NEXT[cur] >= 0
    out[:, maps.END_SLOT] = cur == dest
    return out


def sample_legal(logits_slots: np.ndarray, legal: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One draw per row from softmax(logits) restricted to the legal slots, by inversion with uniforms u in [0, 1).
    Every row must have a legal slot; the draw is always a legal slot (rounding can only fall on the last one)."""
    z = np.where(legal, logits_slots, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    cum = np.cumsum(p, 1)
    choice = (cum <= u[:, None]).sum(1)
    last_legal = legal.shape[1] - 1 - np.argmax(legal[:, ::-1], axis=1)
    choice = np.minimum(choice, last_legal)
    # a draw can only land on a slot with probability above 0, i.e. a legal one
    bad = ~legal[np.arange(len(choice)), choice]
    if bad.any():
        raise AssertionError("a free play draw fell on an illegal move")
    return choice


def free_play(model, starts: np.ndarray, dests: np.ndarray, seed: int = PLAY_SEED, batch: int = BATCH,
              max_turns: int = MAX_TURNS) -> dict[str, Any]:
    """The model drives itself from each start towards its destination (module docstring, 2). Drives run side by
    side; the uniform draws come from one generator seeded `seed`, one per live drive per step, in drive order."""
    starts = np.asarray(starts, dtype=np.int64)
    dests = np.asarray(dests, dtype=np.int64)
    n = len(starts)
    rng = np.random.default_rng(seed)
    tokens = np.zeros((n, max_turns + 3), dtype=np.int64)
    tokens[:, 0], tokens[:, 1] = starts, dests
    cur = starts.copy()
    turns = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    outcome = np.zeros(n, dtype=np.int64)
    errors = np.zeros(n, dtype=np.int64)
    first = np.zeros(n, dtype=np.int64)
    n_steps = max_turns + 2                                     # step index 1 .. max_turns + 1
    ok_by_step = np.zeros(n_steps, dtype=np.int64)
    n_by_step = np.zeros(n_steps, dtype=np.int64)
    for k in range(max_turns + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        legal = legal_slots(cur[idx], dests[idx])
        dead = ~legal.any(1)                                    # a dead end that is not the destination
        if dead.any():
            outcome[idx[dead]] = STUCK
            alive[idx[dead]] = False
            idx, legal = idx[~dead], legal[~dead]
            if len(idx) == 0:
                break
        step = k + 1
        t = 2 + k                                               # tokens read: start, destination, k moves
        logits = _last_logits(model, tokens[idx, :t], batch)    # (m, VOCAB)
        top_slot = maps.TOKEN_SLOT[logits.argmax(-1)]
        ok = (top_slot >= 0) & legal[np.arange(len(idx)), np.maximum(top_slot, 0)]
        ok_by_step[step], n_by_step[step] = int(ok.sum()), len(idx)
        bad = idx[~ok]
        errors[bad] += 1
        first[bad] = np.where(first[bad] == 0, step, first[bad])
        choice = sample_legal(logits[:, maps.SLOT_TOKEN], legal, rng.random(len(idx)))
        tokens[idx, t] = maps.SLOT_TOKEN[choice]
        end = choice == maps.END_SLOT
        outcome[idx[end]] = REACHED
        alive[idx[end]] = False
        go = idx[~end]
        if k == max_turns:                                      # 96 turns read: no 97th turn, the drive stops
            outcome[go] = LENGTH_LIMIT
            alive[go] = False
            break
        cur[go] = maps.NEXT[cur[go], choice[~end]]
        turns[go] += 1
    outcome[alive] = LENGTH_LIMIT                               # (none: every drive ends inside the loop)
    n_pred = int(n_by_step.sum())
    return {"n_drives": int(n), "n_predictions": n_pred, "legal_rate": int(ok_by_step.sum()) / max(n_pred, 1),
            "share_drives_no_illegal": float((errors == 0).mean()),
            "share_reached_destination": float((outcome == REACHED).mean()),
            "share_stuck_at_dead_end": float((outcome == STUCK).mean()),
            "share_length_limit": float((outcome == LENGTH_LIMIT).mean()),
            "outcome_counts": {name: int((outcome == code).sum()) for code, name in OUTCOMES.items()},
            "mean_turns": float(turns.mean()), "longest_drive_turns": int(turns.max()) if n else 0,
            "errors_per_drive": float(errors.mean()),
            "first_error_step": _first_error_summary(first),
            "legal_rate_by_step": _step_stats(ok_by_step, n_by_step)}


# ==========================================================================
# One model
# ==========================================================================

def d1_file_check(model_id: str, rate: float) -> dict[str, Any]:
    """The model's saved D1 score, if this machine has it, and whether it equals the teacher forced rate."""
    path = nav.results_dir(model_id) / D1_FOLDER / f"{model_id}.json"
    if not path.exists():
        return {"d1_file": None}
    name = f"{path.parent.parent.name}/{D1_FOLDER}/{path.name}"
    d = json.loads(path.read_text())
    if d.get("status") != "ok":
        return {"d1_file": name, "d1_file_status": d.get("status")}
    return {"d1_file": name, "d1_file_score": d["score"],
            "matches_d1_file": abs(d["score"] - rate) <= D1_TOLERANCE}


def score_model(model, batch: int = BATCH, n_routes: int = N_ROUTES) -> dict[str, Any]:
    """Both parts of G9 on one model (n_routes below 1,000 only in tests and smoke runs)."""
    dist = nav.training_distribution(model)
    data = nav.routes(dist, SPLIT, n_routes, seed=DATA_SEED)
    tf = teacher_forced(model, data, batch)
    tf.update(d1_file_check(model.model_id, tf["legal_rate"]))
    fp = free_play(model, data["tokens"][:, 0], data["dest"], PLAY_SEED, batch)
    return {"measure": MEASURE, "version": VERSION, "method": METHOD, "status": "ok", "train_distribution": dist,
            "settings": {"split": SPLIT, "n_routes": int(n_routes), "data_seed": DATA_SEED, "play_seed": PLAY_SEED,
                         "batch": int(batch), "max_turns": MAX_TURNS, "temperature": 1.0},
            "teacher_forced": tf, "free_play": fp}


def result_record(model_id: str, fields: dict, res: dict, runtime_sec: float) -> dict[str, Any]:
    return {"model_id": model_id, **fields, **res, "provenance": results.provenance(round(runtime_sec, 1))}


def result_path(model_id: str) -> Path:
    return nav.results_dir(model_id) / RESULTS_FOLDER / f"{model_id}.json"


def _write(model_id: str, rec: dict) -> None:
    path = result_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    log.info("saved %s", path)


def score_one(model_id: str, save: bool, batch: int, device: str | None = None, load_fn=None) -> dict[str, Any]:
    """Load, score and (with save) write one model. A failure after loading is written as status failed with the
    reason (Section 5) and raised again, so the runner logs it and a rerun tries again."""
    model = (load_fn or (lambda m: nav.load(m, device)))(model_id)
    fields = nav.model_fields(model)
    t0 = time.time()
    try:
        res = score_model(model, batch)
    except Exception as e:  # noqa: BLE001  recorded, then raised
        if save:
            _write(model_id, result_record(model_id, fields, {
                "measure": MEASURE, "version": VERSION, "status": "failed",
                "reason": f"{type(e).__name__}: {e}"}, time.time() - t0))
        raise
    rec = result_record(model_id, fields, res, time.time() - t0)
    if save:
        _write(model_id, rec)
    return rec


def headline(rec: dict) -> str:
    tf, fp = rec["teacher_forced"], rec["free_play"]
    return (f"teacher forced: all legal {tf['share_routes_all_legal']:.4f}  rate {tf['legal_rate']:.4f}  |  "
            f"free play: no illegal {fp['share_drives_no_illegal']:.4f}  rate {fp['legal_rate']:.4f}  "
            f"reached {fp['share_reached_destination']:.4f}  stuck {fp['share_stuck_at_dead_end']:.4f}  "
            f"limit {fp['share_length_limit']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Whole route measure for navigation (Amendment 5, G9)")
    ap.add_argument("--models", nargs="+", required=True, help="navigation ids: zoo, pilot (seed 100) or maps_vafa_*")
    ap.add_argument("--save", action="store_true", help="write results/whole_route_maps/<id>.json (pilots: results/pilot_scoring/)")
    ap.add_argument("--batch", type=int, default=BATCH, help="routes per forward pass (256 as D1; smaller only for memory)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    nav.check_can_score(a.models, a.save)
    failed = []
    for m in a.models:
        t0 = time.time()
        try:
            rec = score_one(m, a.save, a.batch)
        except Exception as e:  # noqa: BLE001  one model must not stop the others
            log.warning("%s failed: %s", m, e)
            failed.append((m, f"{type(e).__name__}: {e}"))
            continue
        print(f"{m:45s} {headline(rec)}  ({time.time() - t0:.0f} s)")
        if rec["teacher_forced"].get("matches_d1_file") is False:
            print(f"{'':45s} note: teacher forced rate differs from the saved D1 score "
                  f"{rec['teacher_forced']['d1_file_score']}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nav.report_not_loaded(failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
