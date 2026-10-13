"""Diagnostic 2: state equivalence (Myhill Nerode), after Vafa et al. 2024.

The idea in plain words
    Two different move histories can reach the same Othello board. If a model really tracks
    the board, it should treat those two histories the same way (compression). And when two
    histories reach boards with different legal moves, the model should tell them apart on
    exactly the moves that differ (distinction). The true board from src/domains/othello.py is
    the answer key. Explainer: src/docs/d2_state_equivalence.md.

Protocol (fixed in reports/preregistration.md, Amendment 1, A4)
    data             test games of the model's own training distribution, 1,000 games, data seed 0
    history          every prefix of length 1 to 59; its true state is (board, player to move);
                     histories with no legal move are left out
    accepted moves   A(h) = the model's k highest scoring squares after h (token 0 excluded),
                     k = number of truly legal moves L(h); no threshold to tune
    same state       two histories with the same state but different move sequences,
                     up to 2,000 pairs sampled uniformly with seed 0; if fewer than 500 exist,
                     more test games are used in steps of 1,000, up to 5,000
    different state  two histories of the same length whose legal move sets differ,
                     up to 2,000 pairs sampled uniformly with seed 0
    compression err  |A(h) xor A(g)| / |A(h) or A(g)|, mean over same state pairs (0 is best)
    distinction err  share of boundary moves (legal after exactly one of h, g) that the model
                     does not separate correctly, mean over different state pairs (0 is best)
    score            1 minus the mean of the two errors (higher = more world model like)

The pairs depend only on the data, so they are built once per (distribution, split, games,
seed) and reused for every model of that distribution.

Run on the EC2 (repo root, conda env wmconcord):
    python src/diagnostics/d2/state_equiv.py --models othello_gpt_synthetic othello_gpt_championship
    python src/diagnostics/d2/state_equiv.py --models othello_transformer_small_synthetic_s0 --save
Zoo models are refused until the git tag prereg-v1.1 exists (the protocol must be frozen first).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("d2")

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    """Import a file from src/ by path (src/ is not a package), shared through sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


itf = _load("interface", "src/diagnostics/interface.py")
oth = itf.oth
results = itf.results

DIAGNOSTIC = "D2"
METHOD = ("state equivalence (Vafa et al. 2024), next move: 1 minus mean of compression error and "
          "distinction error, accepted moves = top k with k = number of legal moves, higher is better")

# Protocol constants: fixed by the preregistration (Amendment 1, A4). Change only by amendment.
MAX_PAIRS = 2000
MIN_SAME_PAIRS = 500
GAME_STEP = 1000
MAX_GAMES = 5000
PAIR_SEED = 0
PREREG_TAG = "prereg-v1.1"


# ==========================================================================
# Histories: every prefix of every game, with its true state and legal moves
# ==========================================================================

@dataclass
class Histories:
    """One row per distinct history (prefix). Arrays are aligned by row."""
    game: np.ndarray        # (H,) index of the game the prefix comes from
    pos: np.ndarray         # (H,) position t: the prefix is moves 0..t, so its length is t + 1
    state: list[bytes]      # (H,) board bytes + player to move: equal bytes = same true state
    legal: np.ndarray       # (H, 61) bool, truly legal next tokens L(h)
    tokens: np.ndarray      # (G, 59) the games as tokens, input to the model

    def __len__(self) -> int:
        return len(self.game)


def player_to_move(board: np.ndarray, mover: int, legal_sq: list[int]) -> int:
    """Who moves next after `mover` moved: the other player, unless they must pass.
    `legal_sq` is legal_next(board, mover), so its first square tells us who it belongs to."""
    nxt = oth.other(mover)
    return nxt if oth.flips_for(board, legal_sq[0], nxt) else mover


def build_histories(games: np.ndarray) -> Histories:
    """Replay every game and keep one row per distinct prefix that has a legal next move.
    Two games that share an opening give the same prefix; it is kept once, because a pair of
    identical move sequences says nothing about state equivalence."""
    data = oth.build(games, legal=True)
    seen: set[bytes] = set()
    rows_g, rows_t, states, legal = [], [], [], []
    for g in range(len(data["tokens"])):
        T = int(data["mask"][g].sum())
        for t in range(T):
            nxt = data["legal_next"][g, t]
            if not nxt.any():          # game over: no legal move, left out
                continue
            prefix = data["tokens"][g, : t + 1].tobytes()
            if prefix in seen:
                continue
            seen.add(prefix)
            board = data["black_white"][g, t]
            legal_sq = [oth.TOKEN_TO_SQ[int(k)] for k in np.flatnonzero(nxt)]
            p = player_to_move(board, int(data["movers"][g, t]), legal_sq)
            rows_g.append(g)
            rows_t.append(t)
            states.append(board.tobytes() + bytes([p]))
            legal.append(nxt)
    return Histories(game=np.array(rows_g, dtype=np.int64), pos=np.array(rows_t, dtype=np.int64),
                     state=states, legal=np.array(legal, dtype=bool).reshape(-1, oth.VOCAB),
                     tokens=data["tokens"])


# ==========================================================================
# Pairs
# ==========================================================================

def same_state_pairs(h: Histories, max_pairs: int = MAX_PAIRS, seed: int = PAIR_SEED) -> np.ndarray:
    """(n, 2) rows of histories with the same true state and different move sequences.
    All such pairs are listed, then up to max_pairs are sampled uniformly with `seed`."""
    groups: dict[bytes, list[int]] = {}
    for i, s in enumerate(h.state):
        groups.setdefault(s, []).append(i)
    pairs = [(a, b) for idx in groups.values() if len(idx) > 1
             for x, a in enumerate(idx) for b in idx[x + 1:]]
    pairs_arr = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs_arr) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs_arr = pairs_arr[np.sort(rng.choice(len(pairs_arr), max_pairs, replace=False))]
    return pairs_arr


def different_state_pairs(h: Histories, max_pairs: int = MAX_PAIRS, seed: int = PAIR_SEED,
                          max_tries: int = 50) -> np.ndarray:
    """(n, 2) rows of histories with the same length and different legal move sets, sampled
    uniformly with `seed`: a length is drawn in proportion to its number of history pairs,
    then two distinct histories of that length; pairs with equal legal sets or already drawn
    are drawn again."""
    rng = np.random.default_rng(seed)
    by_len = [np.flatnonzero(h.pos == t) for t in range(oth.N_CTX)]
    weights = np.array([len(ix) * (len(ix) - 1) / 2 for ix in by_len], dtype=float)
    if weights.sum() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    weights /= weights.sum()
    chosen: set[tuple[int, int]] = set()
    tries = 0
    while len(chosen) < max_pairs and tries < max_tries * max_pairs:
        tries += 1
        ix = by_len[rng.choice(oth.N_CTX, p=weights)]
        a, b = (int(v) for v in rng.choice(ix, 2, replace=False))
        key = (min(a, b), max(a, b))
        if key in chosen or np.array_equal(h.legal[a], h.legal[b]):
            continue
        chosen.add(key)
    return np.array(sorted(chosen), dtype=np.int64).reshape(-1, 2)


# ==========================================================================
# The model's accepted moves and the two errors
# ==========================================================================

@torch.no_grad()
def accepted_sets(model, h: Histories, batch: int = 256) -> np.ndarray:
    """(H, 61) bool: A(h) = the model's k highest scoring squares after each history, with
    k = |L(h)|. Token 0 (padding) is never accepted. One forward pass per game gives every
    prefix, because the models are causal: position t only sees moves 0..t."""
    need = np.unique(h.game)
    scores = np.zeros((len(h.tokens), oth.N_CTX, oth.VOCAB - 1), dtype=np.float32)
    for i in range(0, len(need), batch):
        g = need[i:i + batch]
        logits = model.logits(torch.as_tensor(h.tokens[g]))
        scores[g] = logits[..., 1:].float().cpu().numpy()
    s = scores[h.game, h.pos]                                   # (H, 60)
    k = h.legal.sum(1)                                          # (H,)
    rank = np.argsort(np.argsort(-s, axis=1, kind="stable"), axis=1, kind="stable")
    acc = np.zeros((len(h), oth.VOCAB), dtype=bool)
    acc[:, 1:] = rank < k[:, None]
    return acc


def compression_errors(acc: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per same state pair: share of accepted moves accepted after one history only."""
    a, b = acc[pairs[:, 0]], acc[pairs[:, 1]]
    return (a ^ b).sum(1) / np.maximum((a | b).sum(1), 1)


def distinction_errors(acc: np.ndarray, legal: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per different state pair: share of boundary moves (legal after exactly one history) that
    the model does not separate correctly. A move legal only after h is separated correctly
    when the model accepts it after h and not after g, and the mirror case for g."""
    a, b = acc[pairs[:, 0]], acc[pairs[:, 1]]
    la, lb = legal[pairs[:, 0]], legal[pairs[:, 1]]
    only_a, only_b = la & ~lb, lb & ~la
    correct = (only_a & a & ~b).sum(1) + (only_b & b & ~a).sum(1)
    boundary = (only_a | only_b).sum(1)
    return 1.0 - correct / np.maximum(boundary, 1)


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return float(x.mean()), se


def evaluate(model, h: Histories, same: np.ndarray, diff: np.ndarray, batch: int = 256) -> dict[str, Any]:
    """Both errors and the score for one model on prepared histories and pairs."""
    if len(same) == 0 or len(diff) == 0:
        raise ValueError(f"not enough pairs: {len(same)} same state, {len(diff)} different state")
    acc = accepted_sets(model, h, batch)
    ce, ce_se = _mean_se(compression_errors(acc, same))
    de, de_se = _mean_se(distinction_errors(acc, h.legal, diff))
    lengths_same = h.pos[same[:, 0]] + 1
    return {
        "score": 1.0 - (ce + de) / 2.0,
        "score_se": 0.5 * float(np.hypot(ce_se, de_se)),
        "compression_error": ce, "compression_error_se": ce_se,
        "distinction_error": de, "distinction_error_se": de_se,
        "n_same_state_pairs": int(len(same)), "n_different_state_pairs": int(len(diff)),
        "n_histories": int(len(h)),
        "same_state_pair_lengths": {str(int(L)): int(c) for L, c in zip(*np.unique(lengths_same, return_counts=True), strict=True)},
        "mean_legal_moves": float(h.legal.sum(1).mean()),
    }


# ==========================================================================
# Data preparation (cached per distribution) and the diagnostic contract
# ==========================================================================

_CACHE: dict[tuple, tuple[Histories, np.ndarray, np.ndarray, int]] = {}


def prepare(distribution: str, split: str, n_games: int, data_seed: int, max_pairs: int = MAX_PAIRS,
            min_same_pairs: int = MIN_SAME_PAIRS, game_step: int = GAME_STEP,
            max_games: int = MAX_GAMES) -> tuple[Histories, np.ndarray, np.ndarray, int]:
    """Histories and both pair sets for one distribution. Starts at n_games test games and adds
    game_step games while there are fewer than min_same_pairs same state pairs, up to max_games.
    Returns (histories, same pairs, different pairs, games used)."""
    key = (distribution, split, n_games, data_seed, max_pairs, min_same_pairs, game_step, max_games)
    if key in _CACHE:
        return _CACHE[key]
    n = n_games
    while True:
        games = oth.load_distribution(distribution, split, n, seed=data_seed)
        h = build_histories(games)
        same = same_state_pairs(h, max_pairs, PAIR_SEED)
        used = len(games)
        log.info("D2 %s %s: %d games, %d histories, %d same state pairs", distribution, split, used, len(h), len(same))
        if len(same) >= min_same_pairs or n >= max_games or used < n:
            break
        n = min(n + game_step, max_games)
    diff = different_state_pairs(h, max_pairs, PAIR_SEED)
    _CACHE[key] = (h, same, diff, used)
    return _CACHE[key]


def training_distribution(model) -> str:
    """The distribution the model was trained on: models.csv for zoo models, meta for anchors."""
    row = results.load_models().get(model.model_id)
    return row["distribution"] if row else model.meta["distribution"]


def score(model, split: str = "test", n_games: int = 1000, data_seed: int = 0, batch: int = 256,
          **config: Any):
    """The diagnostic contract (src/docs/diagnostic_interface.md). config may override the
    protocol constants for tests only (max_pairs, min_same_pairs, game_step, max_games)."""
    t0 = time.time()
    dist = training_distribution(model)
    h, same, diff, used = prepare(dist, split, n_games, data_seed, **config)
    out = evaluate(model, h, same, diff, batch)
    details = {k: v for k, v in out.items() if k not in ("score", "score_se")}
    details.update({"distribution": dist, "games_requested": n_games, "games_used": used})
    return itf.result_for(model, DIAGNOSTIC, method=METHOD, split=split, n_games=used,
                          data_seed=data_seed, score=out["score"], score_std=out["score_se"],
                          details=details, runtime_sec=round(time.time() - t0, 1))


# ==========================================================================
# Command line
# ==========================================================================

def prereg_tag_exists() -> bool:
    out = subprocess.run(["git", "tag", "-l", PREREG_TAG], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() == PREREG_TAG


def load_any(model_id: str):
    """othello_gpt_synthetic / othello_gpt_championship are the anchors; anything else is a zoo id."""
    if model_id.startswith("othello_gpt_"):
        return itf.othello_gpt(model_id.removeprefix("othello_gpt_"))
    return itf.from_checkpoint(model_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic 2, state equivalence")
    ap.add_argument("--models", nargs="+", required=True, help="othello_gpt_synthetic, othello_gpt_championship or zoo model ids")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_games", type=int, default=1000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write results/D2_state_equiv/{model_id}.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    zoo = [m for m in args.models if not m.startswith("othello_gpt_")]
    if zoo and not prereg_tag_exists():
        sys.exit(f"refusing to score zoo models {zoo}: the git tag {PREREG_TAG} does not exist yet. "
                 "Anchors (othello_gpt_synthetic, othello_gpt_championship) can be scored now.")
    models = [load_any(m) for m in args.models]
    out = itf.run(sys.modules[__name__], models, split=args.split, n_games=args.n_games,
                  data_seed=args.data_seed, save=args.save)
    for r in out:
        d = r.details
        if r.status == "ok":
            print(f"{r.model_id:45s} score {r.score:.4f}  compression err {d['compression_error']:.4f}  "
                  f"distinction err {d['distinction_error']:.4f}  pairs {d['n_same_state_pairs']}/"
                  f"{d['n_different_state_pairs']}  games {r.n_games}")
        else:
            print(f"{r.model_id:45s} {r.status}: {r.reason}")


if __name__ == "__main__":
    main()
