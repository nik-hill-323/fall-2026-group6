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

Navigation (domain maps, preregistration Amendment 5, G5; ticket N5): A4 unchanged except
    history          the route prefix up to every scored position of 1,000 test routes (tokens 0..p,
                     1 <= p <= length - 2); identical prefixes are kept once
    true state       (current intersection, destination), from src/domains/maps.py
    accepted set     the model's k highest scoring tokens over the whole vocabulary (padding excluded),
                     k = size of the legal next set (turns at the intersection, plus end at the destination)
    same state       two different prefixes that reach the same (intersection, destination)
    different state  two prefixes with the same number of turns and different legal next sets
    pair counts, the 500 pair minimum, the 2,000 pair cap, the steps of 1,000 routes up to 5,000, seeds and
    the score are A4's. Limitation as A4: next token only, so two intersections with the same turns are
    different states D2 cannot tell apart.
    python src/diagnostics/d2/state_equiv.py --models maps_transformer_small_shortest_paths_s100 --save
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
nav = _load("navigation", "src/diagnostics/navigation.py")
maps = nav.maps

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
MAPS_METHOD = ("state equivalence (Vafa et al. 2024), navigation (Amendment 5 G5), next token: 1 minus mean of "
               "compression error and distinction error, state = (current intersection, destination), accepted "
               "tokens = top k with k = size of the legal next set, higher is better")


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
    protocol constants for tests only (max_pairs, min_same_pairs, game_step, max_games).
    Navigation models go to score_maps (Amendment 5, G5)."""
    if nav.is_maps(model):
        return score_maps(model, split, n_games, data_seed, batch, **config)
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
# Navigation (Amendment 5, G5)
# ==========================================================================

@dataclass
class MapsHistories:
    """One row per distinct route prefix ending at a scored position. Arrays are aligned by row."""
    route: np.ndarray       # (H,) index of the route the prefix comes from
    pos: np.ndarray         # (H,) scored position p: the prefix is tokens 0..p
    n_turns: np.ndarray     # (H,) turns in the prefix
    state: np.ndarray       # (H,) int64 key of (current intersection, destination): equal key = same state
    legal: np.ndarray       # (H, 9) bool, legal next slots (maps.SLOT_TOKEN order: 8 turns, then end)
    data: dict              # the routes, from maps.build

    def __len__(self) -> int:
        return len(self.route)


def maps_build_histories(data: dict) -> MapsHistories:
    """Every scored position of every route is a history; a prefix shared by two routes is kept once (a pair
    of identical token sequences says nothing about state equivalence). Positions with an empty legal next
    set are left out (never on a data route)."""
    seen: set[bytes] = set()
    rows_g, rows_p = [], []
    tok = data["tokens"]
    for g in range(len(tok)):
        for p in range(1, int(data["length"][g]) - 1):
            if not data["legal"][g, p].any():
                continue
            prefix = tok[g, :p + 1].tobytes()
            if prefix in seen:
                continue
            seen.add(prefix)
            rows_g.append(g)
            rows_p.append(p)
    g = np.array(rows_g, dtype=np.int64)
    p = np.array(rows_p, dtype=np.int64)
    state = data["current"][g, p] * (maps.N_NODES + 1) + data["dest"][g]
    return MapsHistories(route=g, pos=p, n_turns=data["n_turns"][g, p], state=state,
                         legal=data["legal"][g, p].reshape(-1, len(maps.SLOT_TOKEN)), data=data)


def _pair_in_group(r: int, n: int) -> tuple[int, int]:
    """Pair number r (counting from 0) of the pairs (x, y), x < y, of n items in the order x = 0, y = 1..n-1; x = 1, y = 2..n-1; ..."""
    x = 0
    while r >= n - 1 - x:
        r -= n - 1 - x
        x += 1
    return x, x + 1 + r


def maps_same_state_pairs(h: MapsHistories, max_pairs: int = MAX_PAIRS, seed: int = PAIR_SEED) -> np.ndarray:
    """(n, 2) rows of histories with the same state (they are different prefixes by construction). All such
    pairs are counted, then up to max_pairs are drawn uniformly with `seed`, as A4; the list of all pairs is
    never built, since a busy intersection can hold many prefixes."""
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(h.state.tolist()):
        groups.setdefault(s, []).append(i)
    members = [idx for idx in groups.values() if len(idx) > 1]
    counts = np.array([len(m) * (len(m) - 1) // 2 for m in members], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if total <= max_pairs:
        chosen = np.arange(total)
    else:
        chosen = np.sort(np.random.default_rng(seed).choice(total, max_pairs, replace=False))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    grp = np.searchsorted(starts, chosen, side="right") - 1
    out = np.zeros((len(chosen), 2), dtype=np.int64)
    for j, (r, k) in enumerate(zip(chosen.tolist(), grp.tolist())):
        x, y = _pair_in_group(r - int(starts[k]), len(members[k]))
        out[j] = (members[k][x], members[k][y])
    return out


def maps_different_state_pairs(h: MapsHistories, max_pairs: int = MAX_PAIRS, seed: int = PAIR_SEED,
                               max_tries: int = 50) -> np.ndarray:
    """(n, 2) rows of histories with the same number of turns and different legal next sets, drawn as A4 draws
    them on Othello: a turn count in proportion to its number of history pairs, then two distinct histories with
    it; pairs with equal legal sets or already drawn are drawn again."""
    rng = np.random.default_rng(seed)
    n_len = maps.MAX_TURNS + 1
    by_len = [np.flatnonzero(h.n_turns == t) for t in range(n_len)]
    weights = np.array([len(ix) * (len(ix) - 1) / 2 for ix in by_len], dtype=float)
    if weights.sum() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    weights /= weights.sum()
    chosen: set[tuple[int, int]] = set()
    tries = 0
    while len(chosen) < max_pairs and tries < max_tries * max_pairs:
        tries += 1
        ix = by_len[rng.choice(n_len, p=weights)]
        a, b = (int(v) for v in rng.choice(ix, 2, replace=False))
        key = (min(a, b), max(a, b))
        if key in chosen or np.array_equal(h.legal[a], h.legal[b]):
            continue
        chosen.add(key)
    return np.array(sorted(chosen), dtype=np.int64).reshape(-1, 2)


def maps_accepted_sets(model, h: MapsHistories, batch: int = 256) -> list[frozenset[int]]:
    """A(h) per history: the model's k highest scoring tokens at the prefix's last position (padding excluded),
    k = size of the legal next set. One forward pass per route gives every prefix (the models are causal)."""
    k_max = len(maps.SLOT_TOKEN)
    need = np.unique(h.route)
    sub = {"tokens": h.data["tokens"][need]}
    top = nav.top_tokens(model, sub, k_max, batch)                    # (len(need), N_CTX, 9)
    row_of = np.full(len(h.data["tokens"]), -1, dtype=np.int64)
    row_of[need] = np.arange(len(need))
    k = h.legal.sum(1)
    t = top[row_of[h.route], h.pos]                                     # (H, 9)
    return [frozenset(t[i, :k[i]].tolist()) for i in range(len(h))]


def maps_compression_errors(acc: list[frozenset[int]], pairs: np.ndarray) -> np.ndarray:
    """Per same state pair: |A(h) xor A(g)| / |A(h) or A(g)| (A4)."""
    return np.array([len(acc[a] ^ acc[b]) / max(len(acc[a] | acc[b]), 1) for a, b in pairs.tolist()], dtype=float)


def maps_distinction_errors(acc: list[frozenset[int]], legal: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per different state pair: share of boundary tokens (legal after exactly one history) that the model does
    not separate correctly. A token legal only after h is right when it is accepted after h and not after g,
    and the mirror case for g (A4)."""
    out = np.zeros(len(pairs))
    for j, (a, b) in enumerate(pairs.tolist()):
        la = set(maps.SLOT_TOKEN[legal[a]].tolist())
        lb = set(maps.SLOT_TOKEN[legal[b]].tolist())
        only_a, only_b = la - lb, lb - la
        correct = sum(1 for t in only_a if t in acc[a] and t not in acc[b])
        correct += sum(1 for t in only_b if t in acc[b] and t not in acc[a])
        out[j] = 1.0 - correct / max(len(only_a) + len(only_b), 1)
    return out


def maps_evaluate(model, h: MapsHistories, same: np.ndarray, diff: np.ndarray, batch: int = 256) -> dict[str, Any]:
    """Both errors and the score for one navigation model on prepared histories and pairs."""
    if len(same) == 0 or len(diff) == 0:
        raise ValueError(f"not enough pairs: {len(same)} same state, {len(diff)} different state")
    acc = maps_accepted_sets(model, h, batch)
    ce, ce_se = _mean_se(maps_compression_errors(acc, same))
    de, de_se = _mean_se(maps_distinction_errors(acc, h.legal, diff))
    turns_same = h.n_turns[same[:, 0]]
    same_route = h.route[same[:, 0]] == h.route[same[:, 1]]
    return {
        "score": 1.0 - (ce + de) / 2.0,
        "score_se": 0.5 * float(np.hypot(ce_se, de_se)),
        "compression_error": ce, "compression_error_se": ce_se,
        "distinction_error": de, "distinction_error_se": de_se,
        "n_same_state_pairs": int(len(same)), "n_different_state_pairs": int(len(diff)),
        "n_histories": int(len(h)),
        "same_state_pair_turns": {str(int(L)): int(c) for L, c in zip(*np.unique(turns_same, return_counts=True), strict=True)},
        "same_state_pairs_within_one_route": int(same_route.sum()),
        "same_state_pairs_at_the_destination": int((h.state[same[:, 0]] // (maps.N_NODES + 1)
                                                    == h.state[same[:, 0]] % (maps.N_NODES + 1)).sum()),
        "mean_legal_moves": float(h.legal.sum(1).mean()),
    }


_MAPS_CACHE: dict[tuple, tuple[MapsHistories, np.ndarray, np.ndarray, int]] = {}


def maps_prepare(distribution: str, split: str, n_routes: int, data_seed: int, max_pairs: int = MAX_PAIRS,
                 min_same_pairs: int = MIN_SAME_PAIRS, game_step: int = GAME_STEP,
                 max_games: int = MAX_GAMES) -> tuple[MapsHistories, np.ndarray, np.ndarray, int]:
    """Histories and both pair sets for one navigation distribution. Starts at n_routes test routes and adds
    game_step routes while there are fewer than min_same_pairs same state pairs, up to max_games (A4).
    Returns (histories, same pairs, different pairs, routes used)."""
    key = (distribution, split, n_routes, data_seed, max_pairs, min_same_pairs, game_step, max_games)
    if key in _MAPS_CACHE:
        return _MAPS_CACHE[key]
    n = n_routes
    while True:
        data = nav.routes(distribution, split, n, seed=data_seed)
        h = maps_build_histories(data)
        same = maps_same_state_pairs(h, max_pairs, PAIR_SEED)
        used = len(data["tokens"])
        log.info("D2 %s %s: %d routes, %d histories, %d same state pairs", distribution, split, used, len(h), len(same))
        if len(same) >= min_same_pairs or n >= max_games or used < n:
            break
        n = min(n + game_step, max_games)
    diff = maps_different_state_pairs(h, max_pairs, PAIR_SEED)
    _MAPS_CACHE[key] = (h, same, diff, used)
    return _MAPS_CACHE[key]


def score_maps(model, split: str = "test", n_games: int = 1000, data_seed: int = 0, batch: int = 256,
               **config: Any):
    """D2 on a navigation model (Amendment 5, G5). config may override the A4 constants for tests only."""
    t0 = time.time()
    dist = nav.training_distribution(model)
    h, same, diff, used = maps_prepare(dist, split, n_games, data_seed, **config)
    out = maps_evaluate(model, h, same, diff, batch)
    details = {k: v for k, v in out.items() if k not in ("score", "score_se")}
    details.update({"distribution": dist, "routes_requested": n_games, "routes_used": used})
    return itf.result_for(model, DIAGNOSTIC, method=MAPS_METHOD, split=split, n_games=used,
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
    ap.add_argument("--models", nargs="+", required=True, help="othello_gpt_synthetic, othello_gpt_championship, zoo model ids, or navigation ids (zoo, pilot, maps_vafa_*)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_games", type=int, default=1000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write results/D2_state_equiv/{model_id}.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if any(nav.is_maps_id(m) for m in args.models):      # navigation: gates, loading, pilot folder (N0)
        out, not_loaded = nav.main_for(sys.modules[__name__], args.models, args.save, split=args.split,
                                       n_games=args.n_games, data_seed=args.data_seed)
        for r in out:
            d = r.details
            if r.status == "ok":
                print(f"{r.model_id:45s} score {r.score:.4f}  compression err {d['compression_error']:.4f}  "
                      f"distinction err {d['distinction_error']:.4f}  pairs {d['n_same_state_pairs']}/"
                      f"{d['n_different_state_pairs']}  routes {r.n_games}")
            else:
                print(f"{r.model_id:45s} {r.status}: {r.reason}")
        nav.report_not_loaded(not_loaded)
        sys.exit(1 if not_loaded else 0)

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
