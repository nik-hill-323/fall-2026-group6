"""Othello fragility suite, version 2: the H3 outcome after preregistration Amendment 3.

Version 1 (src/fragility/othello.py, tag fragility-v1) is not changed and is still run and
reported. This file imports it and replaces the two parts that Amendment 3 found not to measure
fragility on the zoo; everything else is v1, called through v1's own functions.

Why (reports/preregistration.md, Amendment 3)
    detour      v1 scores the share of self play games with no illegal proposal in about 58 moves.
                A model that is legal with probability q per move completes about q^58 of them:
                0.84 -> 0.00004, 0.97 -> 0.16. On the first zoo model every completion rate was
                0.000, so all four detour drops were 0 and carried no information.
                v2: every game is played to its end. The model's top move is scored legal or not,
                and the move played is the model's best LEGAL move, so an illegal proposal does
                not end the game. With probability p that move is replaced by another legal move
                (random, or the one the model rates least likely), exactly as in v1.
                Score: legal rate over all predictions. Drop: rate at p = 0 minus rate at p.
                Every setting plays full games, so every p sees the same stages of the game.
    after pass  v1 compares the legal rate at positions right after a pass with the legal rate
                over all positions. A pass means one player has no move at all, and on those boards
                most empty squares are legal for the player who keeps moving (first zoo model: 0.934
                of the empty squares legal after a pass, 0.660 at the same move numbers otherwise),
                so these positions are much easier, and the comparison measures that, not the pass
                (0.972 after a pass vs 0.842 overall, drop -0.130; vs 0.854 at the same move numbers).
                v2: each pass position is compared with the non pass positions at the same move
                number AND with the same number of legal moves, in the same test games (the move
                number also fixes the number of empty squares). Baseline = the mean, over pass
                positions, of the non pass legal rate in that cell. Drop: baseline minus after pass
                rate (first zoo model: 0.963 vs 0.972, drop -0.009).
    unchanged   in distribution, cross distribution and the three symmetries (v1 functions,
                same games, same numbers).

Score written per model (results/fragility_v2/{model_id}.json)
    one rate per variant, the drop of each, `fragility` = mean of the drops (higher = more
    fragile), and `in_distribution_legal_rate` for the relative fragility of Amendment 1, A2.

Usage (EC2, repo root):
    python src/fragility/othello_v2.py --model outputs/zoo/<model_id>/model.pt \
        --arch transformer --scale small --train_distribution synthetic --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("fragility_v2")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v1 = _load("fragility_othello", "src/fragility/othello.py")
rules = v1.rules

# ==========================================================================
# The frozen definition of v2. Changing anything here changes suite_hash() and fails the
# freeze test. Do not edit after the tag fragility-v2; add a v3 instead.
# ==========================================================================

SUITE_VERSION = "othello-fragility-v2"
SUITE: dict = {
    "base": "othello-fragility-v1",
    "n_test_games": v1.SUITE["n_test_games"],
    "data_seed": v1.SUITE["data_seed"],
    "cross_distribution": v1.SUITE["cross_distribution"],
    "symmetries": v1.SUITE["symmetries"],
    "after_pass": "drop = mean over pass positions of the non pass legal rate at the same move "
                  "number and the same number of legal moves, minus the legal rate at the pass positions",
    "detour": {
        **v1.SUITE["detour"],
        "score": "legal rate over all predictions; full games; the move played is the model's "
                 "best legal move, replaced with probability p",
    },
}


def suite_hash() -> str:
    blob = json.dumps({"version": SUITE_VERSION, "suite": SUITE}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ==========================================================================
# After pass, compared at the same move number
# ==========================================================================

@torch.no_grad()
def correct_at(model, data: dict, device: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """(G, 59) bool `correct` (the model's top next move is legal) and (G, 59) bool `scored`
    (a next move exists). The same positions and the same rule as v1.legal_rate_at."""
    G = len(data["tokens"])
    correct = np.zeros((G, rules.N_CTX), dtype=bool)
    scored = np.zeros((G, rules.N_CTX), dtype=bool)
    scored[:, :-1] = data["mask"][:, 1:]
    for i in range(0, G, batch):
        tok = torch.tensor(data["tokens"][i:i + batch], device=device)
        pred = model(tok).argmax(-1).cpu().numpy()
        sl = slice(i, i + len(tok))
        correct[sl] = np.take_along_axis(data["legal_next"][sl], pred[..., None], axis=2)[..., 0]
    return correct & scored, scored


def after_pass_matched(correct: np.ndarray, scored: np.ndarray, where: np.ndarray,
                       n_legal: np.ndarray) -> dict:
    """Legal rate at pass positions and its baseline. For each pass position (g, t), the cell is
    every non pass position at the same move number t with the same number of legal next moves;
    the baseline is the mean, over pass positions, of the cell's legal rate. A pass position whose
    cell is empty is left out (and counted)."""
    pas = where & scored
    non = (~where) & scored
    rates, base, moves = [], [], []
    left_out = 0
    for g, t in zip(*np.nonzero(pas)):
        cell = non[:, t] & (n_legal[:, t] == n_legal[g, t])
        if not cell.any():
            left_out += 1
            continue
        rates.append(float(correct[g, t]))
        base.append(float(correct[cell, t].mean()))
        moves.append(int(t))
    if not rates:
        return {"rate": None, "baseline": None, "n_positions": 0, "n_left_out": left_out}
    return {"rate": float(np.mean(rates)), "baseline": float(np.mean(base)), "n_positions": len(rates),
            "n_left_out": left_out, "mean_move_number": float(np.mean(moves)),
            "mean_n_legal": float(n_legal[pas].mean())}


# ==========================================================================
# Detour, played to the end of every game
# ==========================================================================

@torch.no_grad()
def detour_rollout(model, n_games: int, p: float, mode: str, seed: int, device: str) -> dict:
    """Self play with forced detours, every game to its end (or 59 moves). The first move is a
    random legal move. At each later move the model's top move is scored legal or illegal. The
    move played is the model's best legal move (its top move when that is legal); with
    probability p it is replaced by another legal move, a uniformly random one ("random") or
    the one the model rates least likely ("adversarial"), as in v1."""
    if mode not in ("random", "adversarial"):
        raise ValueError(f"mode must be random or adversarial, not {mode!r}")
    model.eval()
    rng = np.random.default_rng(seed)
    boards = [rules.start_board() for _ in range(n_games)]
    to_move = [rules.BLACK] * n_games
    tokens = np.zeros((n_games, rules.N_CTX), dtype=np.int64)
    alive = np.ones(n_games, dtype=bool)
    n_pred = n_legal = n_detours = 0
    length = np.ones(n_games, dtype=np.int64)

    def play(g: int, sq: int, t: int) -> None:
        player = to_move[g]
        boards[g][rules.flips_for(boards[g], sq, player)] = player
        boards[g][sq] = player
        tokens[g, t] = rules.SQ_TO_TOKEN[sq]
        to_move[g] = rules.other(player)

    for g in range(n_games):
        play(g, int(rng.choice(rules.legal_moves(boards[g], rules.BLACK))), 0)

    for t in range(1, rules.N_CTX):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        logits = model(torch.tensor(tokens[idx, :t], device=device))[:, -1]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()      # (A, 61)
        for row, g in enumerate(idx):
            legal = rules.legal_moves(boards[g], to_move[g])
            if not legal:                                # pass
                to_move[g] = rules.other(to_move[g])
                legal = rules.legal_moves(boards[g], to_move[g])
                if not legal:                            # nobody can move: game over
                    alive[g] = False
                    continue
            n_pred += 1
            top_sq = rules.TOKEN_TO_SQ.get(int(probs[row].argmax()))
            if top_sq in legal:
                n_legal += 1
                best = top_sq
            else:
                best = max(legal, key=lambda s: probs[row, rules.SQ_TO_TOKEN[s]])
            sq = best
            if len(legal) > 1 and rng.random() < p:
                alts = [s for s in legal if s != best]
                if mode == "random":
                    sq = int(rng.choice(alts))
                else:
                    sq = min(alts, key=lambda s: probs[row, rules.SQ_TO_TOKEN[s]])
                n_detours += 1
            play(g, sq, t)
            length[g] = t + 1
    return {"legal_rate": n_legal / max(n_pred, 1), "n_games": n_games, "n_predictions": n_pred,
            "n_detours": n_detours, "mean_game_length": float(length.mean())}


# ==========================================================================
# The whole suite for one model
# ==========================================================================

def score_model(model, device: str, train_distribution: str, batch: int = 256,
                games_by_distribution: dict[str, np.ndarray] | None = None,
                n_detour_games: int | None = None) -> dict:
    """Run every v2 variant. `games_by_distribution` and `n_detour_games` are for tests and
    smoke runs only, as in v1; real scoring uses neither."""
    model.eval()
    n = SUITE["n_test_games"]

    def test_games(dist: str) -> np.ndarray | None:
        if games_by_distribution is not None:
            return games_by_distribution.get(dist)
        try:
            return rules.load_distribution(dist, "test", n, SUITE["data_seed"])
        except FileNotFoundError as e:
            log.warning("no test data for %s: %s", dist, e)
            return None

    own = test_games(train_distribution)
    if own is None:
        raise FileNotFoundError(f"no test games for the training distribution {train_distribution!r}")
    base = rules.build(own, legal=True)
    correct, scored = correct_at(model, base, device, batch)
    ref = float(correct.sum() / max(scored.sum(), 1))
    rates: dict[str, float] = {}
    counts: dict[str, int] = {"in_distribution": int(scored.sum())}

    for dist in SUITE["cross_distribution"]:
        if dist == train_distribution:
            continue
        games = test_games(dist)
        if games is None:
            continue
        r, k = v1.legal_rate_at(model, rules.build(games, legal=True), device, batch)
        rates[f"cross_distribution/{dist}"], counts[f"cross_distribution/{dist}"] = r, k

    for sym in SUITE["symmetries"]:
        r, k = v1.legal_rate_at(model, rules.build(v1.transform_games(own, sym), legal=True), device, batch)
        rates[f"symmetry/{sym}"], counts[f"symmetry/{sym}"] = r, k

    drops = {name: ref - r for name, r in rates.items()}

    ap = after_pass_matched(correct, scored, v1.after_pass_positions(base), base["legal_next"].sum(-1))
    if ap["n_positions"]:     # no pass in a tiny sample: leave the variant out rather than score 0 of 0
        rates["after_pass"], counts["after_pass"] = ap["rate"], ap["n_positions"]
        drops["after_pass"] = ap["baseline"] - ap["rate"]

    d = SUITE["detour"]
    n_det = n_detour_games or d["n_games"]
    detour: dict[str, dict] = {}
    for mode in d["modes"]:
        for p in d["probabilities"]:
            detour[f"{mode}/p={p}"] = detour_rollout(model, n_det, p, mode, d["seed"], device)
    for mode in d["modes"]:
        undisturbed = detour[f"{mode}/p=0.0"]["legal_rate"]
        for p in d["probabilities"]:
            if p > 0:
                drops[f"detour/{mode}/p={p}"] = undisturbed - detour[f"{mode}/p={p}"]["legal_rate"]

    return {
        "suite_version": SUITE_VERSION, "suite_hash": suite_hash(),
        "base_suite_hash": v1.suite_hash(),
        "train_distribution": train_distribution,
        "in_distribution_legal_rate": ref,
        "legal_rate": rates, "positions_scored": counts,
        "after_pass": ap,
        "detour": detour, "drops": drops,
        "fragility": float(np.mean(list(drops.values()))),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to model.pt written by src/zoo/train.py")
    p.add_argument("--arch", required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--train_distribution", default="synthetic", choices=rules.DISTRIBUTIONS)
    p.add_argument("--seed", type=int, default=0, help="the model's training seed, for its id")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="results/fragility_v2")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    zoo = _load("zoo_train", "src/zoo/train.py")
    model = zoo.build_model(a.arch, a.scale)
    model.load_state_dict(torch.load(a.model, map_location=a.device))
    model.to(a.device)

    model_id = f"othello_{a.arch}_{a.scale}_{a.train_distribution}_s{a.seed}"
    log.info("fragility suite %s (%s, on %s) on %s", SUITE_VERSION, suite_hash(), v1.suite_hash(), model_id)
    res = {"model_id": model_id, **score_model(model, a.device, a.train_distribution, a.batch)}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_id}.json").write_text(json.dumps(res, indent=2))
    log.info("in distribution legal rate %.4f", res["in_distribution_legal_rate"])
    ap = res["after_pass"]
    if ap["n_positions"]:
        log.info("after pass %.4f vs baseline %.4f at the same move number and number of legal moves "
                 "(%d positions, %d left out, mean move %.1f)", ap["rate"], ap["baseline"], ap["n_positions"],
                 ap["n_left_out"], ap["mean_move_number"])
    for name, drop in res["drops"].items():
        log.info("  %-32s drop %+.4f", name, drop)
    log.info("fragility v2 (mean drop) %.4f   ->  %s", res["fragility"], out / f"{model_id}.json")


if __name__ == "__main__":
    main()
