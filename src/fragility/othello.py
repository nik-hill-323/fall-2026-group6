"""Othello fragility suite: the external criterion for H3 (predictive validity).

H3 asks which diagnostic best predicts whether a model breaks on tasks that are
related to its training task but shifted. This file defines "breaks" for Othello.
It must be frozen BEFORE any zoo model is scored by any diagnostic, otherwise H3
is circular. Freeze = this file merged, tagged `fragility-v1`, and the hash in
src/tests/test_othello_fragility.py unchanged from then on.

Every variant keeps the Othello rules exactly as they are and changes only WHICH
states the model is asked about, so the correct answer is always given by the
rules in src/domains/othello.py. Nothing here retrains or adapts the model
(adaptation is what D3 measures; keeping it out keeps H3 independent of D3).

Variants
    in_distribution   legal-move rate on the test split of the model's own training
                      distribution. The reference every drop is measured against.
    cross_distribution  the same score on the test split of each distribution the
                      model was NOT trained on (synthetic <-> championship).
    after_pass        the same score only at positions where the opponent has no move
                      and the same player moves again. Rare, and it breaks the
                      "players alternate" shortcut.
    symmetry          the same score on test games mapped through the three board
                      symmetries that keep the start position (180 degree rotation
                      and the two diagonal reflections). These are relabelings of the
                      same DFA; human games are played in one orientation by
                      convention, so this is a shift for championship-trained models.
    detour            the model plays a game against itself; at each move, with
                      probability p, its chosen move is replaced by another legal move
                      (a random one, or the one the model thinks least likely). Score:
                      share of games it finishes without ever proposing an illegal
                      move. This is the detour test of Vafa et al. (2024) for a board game.

Score written per model (results/fragility/{model_id}.json)
    one rate per variant, the drop of each variant from its reference, and
    `fragility` = the mean of those drops. Higher = more fragile.

Usage (EC2, repo root):
    python src/fragility/othello.py --model outputs/zoo/<model_id>/model.pt \
        --arch transformer --scale small --train_distribution synthetic
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

log = logging.getLogger("fragility")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load("othello", "src/domains/othello.py")

# ==========================================================================
# The frozen definition. Changing anything here changes suite_hash() and fails
# the freeze test. Do not edit after the tag; add a v2 instead.
# ==========================================================================

SUITE_VERSION = "othello-fragility-v1"
SUITE: dict = {
    "n_test_games": 1000,          # games per static variant, from the test split
    "data_seed": 0,
    "cross_distribution": ["synthetic", "championship"],
    "symmetries": ["rot180", "transpose", "anti_transpose"],
    "detour": {
        "n_games": 500,
        "probabilities": [0.0, 0.1, 0.5],   # 0.0 = undisturbed self play, the detour reference
        "modes": ["random", "adversarial"],
        "seed": 0,
    },
}


def suite_hash() -> str:
    blob = json.dumps({"version": SUITE_VERSION, "suite": SUITE}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ==========================================================================
# Symmetry: relabelings of the board that keep the start position
# ==========================================================================

def symmetry_map(name: str) -> np.ndarray:
    """(64,) array: where each square goes. Only maps that keep the start board
    (and therefore keep every game legal with the same colours) are allowed. A 90
    degree rotation or a left-right mirror swaps the colours of the start squares,
    which turns black's first moves into white's, so those are not included."""
    r, c = np.divmod(np.arange(rules.N_SQ), rules.SIZE)
    last = rules.SIZE - 1
    if name == "rot180":
        return (last - r) * rules.SIZE + (last - c)
    if name == "transpose":
        return c * rules.SIZE + r
    if name == "anti_transpose":
        return (last - c) * rules.SIZE + (last - r)
    raise ValueError(f"unknown symmetry {name!r}")


def transform_games(games: np.ndarray, name: str) -> np.ndarray:
    """Apply a symmetry to (G, 60) int8 games. Padding (-1) stays padding."""
    m = symmetry_map(name).astype(np.int8)
    out = games.copy()
    real = games >= 0
    out[real] = m[games[real]]
    return out


# ==========================================================================
# Static variants: legal-move rate over chosen positions
# ==========================================================================

def after_pass_positions(data: dict) -> np.ndarray:
    """(G, 59) bool. True at position t when the next move is made by the same player
    who made move t, i.e. the opponent had to pass. The prediction at t is the one scored."""
    where = np.zeros_like(data["mask"])
    same = data["movers"][:, 1:] == data["movers"][:, :-1]
    where[:, :-1] = same & data["mask"][:, 1:]
    return where


@torch.no_grad()
def legal_rate_at(model, data: dict, device: str, batch: int,
                  where: np.ndarray | None = None) -> tuple[float, int]:
    """Share of the model's top next moves that are legal, over positions that have a
    recorded next move and (if given) are True in `where`. Returns (rate, positions scored).
    `data` comes from rules.build(games, legal=True). With where=None this is the same
    number as rules.legal_rate."""
    ok = total = 0
    for i in range(0, len(data["tokens"]), batch):
        tok = torch.tensor(data["tokens"][i:i + batch], device=device)
        pred = model(tok).argmax(-1).cpu().numpy()                     # (B, 59)
        sl = slice(i, i + len(tok))
        sel = np.zeros_like(data["mask"][sl])
        sel[:, :-1] = data["mask"][sl][:, 1:]                          # a next move exists
        if where is not None:
            sel &= where[sl]
        legal = np.take_along_axis(data["legal_next"][sl], pred[..., None], axis=2)[..., 0]
        ok += int((legal & sel).sum())
        total += int(sel.sum())
    return ok / max(total, 1), total


# ==========================================================================
# Detour: the model plays itself and is pushed off its preferred line
# ==========================================================================

@torch.no_grad()
def detour_rollout(model, n_games: int, p: float, mode: str, seed: int, device: str) -> dict:
    """Self play with forced detours. The first move is a random legal move (the model has
    no input yet). After that the model's top move is checked against the rules: illegal
    ends the game as a failure. If legal, with probability p it is replaced by another
    legal move: a uniformly random one (mode "random") or the one the model gives the
    lowest probability (mode "adversarial"). A game that reaches its end, or 59 moves,
    without an illegal proposal counts as completed."""
    if mode not in ("random", "adversarial"):
        raise ValueError(f"mode must be random or adversarial, not {mode!r}")
    model.eval()
    rng = np.random.default_rng(seed)
    boards = [rules.start_board() for _ in range(n_games)]
    to_move = [rules.BLACK] * n_games
    tokens = np.zeros((n_games, rules.N_CTX), dtype=np.int64)
    alive = np.ones(n_games, dtype=bool)
    valid = np.ones(n_games, dtype=bool)
    n_pred = n_legal = n_detours = 0

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
                if not legal:                            # nobody can move: game over, completed
                    alive[g] = False
                    continue
            n_pred += 1
            top_sq = rules.TOKEN_TO_SQ.get(int(probs[row].argmax()))
            if top_sq not in legal:
                valid[g] = False
                alive[g] = False
                continue
            n_legal += 1
            sq = top_sq
            if len(legal) > 1 and rng.random() < p:
                alts = [s for s in legal if s != top_sq]
                if mode == "random":
                    sq = int(rng.choice(alts))
                else:
                    sq = min(alts, key=lambda s: probs[row, rules.SQ_TO_TOKEN[s]])
                n_detours += 1
            play(g, sq, t)
    return {"completion_rate": float(valid.mean()), "legal_rate": n_legal / max(n_pred, 1),
            "n_games": n_games, "n_predictions": n_pred, "n_detours": n_detours}


# ==========================================================================
# The whole suite for one model
# ==========================================================================

def score_model(model, device: str, train_distribution: str, batch: int = 256,
                games_by_distribution: dict[str, np.ndarray] | None = None,
                n_detour_games: int | None = None) -> dict:
    """Run every variant. `games_by_distribution` replaces the test files (used by the tests
    and smoke runs); `n_detour_games` shrinks the rollouts for the same reason. Real scoring
    uses neither, so it always runs the frozen SUITE."""
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
    ref, n_ref = legal_rate_at(model, base, device, batch)
    rates: dict[str, float] = {}
    counts: dict[str, int] = {"in_distribution": n_ref}

    for dist in SUITE["cross_distribution"]:
        if dist == train_distribution:
            continue
        games = test_games(dist)
        if games is None:
            continue
        r, k = legal_rate_at(model, rules.build(games, legal=True), device, batch)
        rates[f"cross_distribution/{dist}"], counts[f"cross_distribution/{dist}"] = r, k

    r, k = legal_rate_at(model, base, device, batch, where=after_pass_positions(base))
    if k:   # no pass positions in a tiny sample: leave the variant out rather than score 0 of 0
        rates["after_pass"], counts["after_pass"] = r, k

    for sym in SUITE["symmetries"]:
        r, k = legal_rate_at(model, rules.build(transform_games(own, sym), legal=True), device, batch)
        rates[f"symmetry/{sym}"], counts[f"symmetry/{sym}"] = r, k

    drops = {name: ref - r for name, r in rates.items()}

    d = SUITE["detour"]
    n_det = n_detour_games or d["n_games"]
    detour: dict[str, dict] = {}
    for mode in d["modes"]:
        for p in d["probabilities"]:
            detour[f"{mode}/p={p}"] = detour_rollout(model, n_det, p, mode, d["seed"], device)
    for mode in d["modes"]:
        undisturbed = detour[f"{mode}/p=0.0"]["completion_rate"]
        for p in d["probabilities"]:
            if p > 0:
                drops[f"detour/{mode}/p={p}"] = undisturbed - detour[f"{mode}/p={p}"]["completion_rate"]

    return {
        "suite_version": SUITE_VERSION, "suite_hash": suite_hash(),
        "train_distribution": train_distribution,
        "in_distribution_legal_rate": ref,
        "legal_rate": rates, "positions_scored": counts,
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
    p.add_argument("--out_dir", default="results/fragility")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    zoo = _load("zoo_train", "src/zoo/train.py")
    model = zoo.build_model(a.arch, a.scale)
    model.load_state_dict(torch.load(a.model, map_location=a.device))
    model.to(a.device)

    model_id = f"othello_{a.arch}_{a.scale}_{a.train_distribution}_s{a.seed}"
    log.info("fragility suite %s (%s) on %s", SUITE_VERSION, suite_hash(), model_id)
    res = {"model_id": model_id, **score_model(model, a.device, a.train_distribution, a.batch)}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_id}.json").write_text(json.dumps(res, indent=2))
    log.info("in distribution legal rate %.4f", res["in_distribution_legal_rate"])
    for name, drop in res["drops"].items():
        log.info("  %-32s drop %+.4f", name, drop)
    log.info("fragility (mean drop) %.4f   ->  %s", res["fragility"], out / f"{model_id}.json")


if __name__ == "__main__":
    main()
