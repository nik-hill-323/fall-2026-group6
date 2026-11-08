"""Navigation domain (domain id `maps`): the one shared source of truth for the Manhattan street map,
the route tokens, the true state at every position, and the navigation data splits.

Every part of the project imports from here for navigation: the zoo trainer, all four diagnostics, the
fragility suite and the tests. Nothing else keeps its own copy of the map. Design and every decision
behind the numbers below: src/docs/navigation_design.md.

The domain follows Vafa et al. (2024). A route is a token sequence

    start destination turn turn ... turn end

A turn is one of 8 compass directions and moves the car along one street segment to the next
intersection. For each route the domain gives the common triple of the proposal:
    sequence          the route as tokens
    true state        (current intersection, destination) at every position
    legal next set    the turns that exist at the current intersection, plus `end` when the current
                      intersection is the destination (Vafa's next token rule)

The map (committed with the repo, written once by src/domains/maps_build.py)
    src/domains/maps_edges.csv        node, turn, next_node: the 9,705 street segments of the canonical
                                      map (the map of Vafa's noisy shortest paths and random walks data)
    src/domains/maps_coordinates.csv  node, lat, lon: every intersection, OpenStreetMap as of 2024-05-05
    4,579 intersections: 4,562 with at least one turn and 17 dead ends (a turn leads in, none leads out).

Tokens (one vocabulary for all three distributions)
    0                       padding
    1 .. N_NODES            intersections, in increasing OpenStreetMap id order
    N_NODES + 1 .. + 8      the turns N NE E SE S SW W NW
    N_NODES + 9             end
    Every route is at most MAX_TOKENS = 99 tokens. A model reads at most N_CTX = 98 tokens (the route
    minus its last token) and is trained to predict tokens 2 onward (never the destination, which is a
    random intersection).

Positions (the same for training, every diagnostic, the fragility suite and the whole route measure)
    Position p means "the model has read tokens 0..p and predicts token p + 1".
    The state at position p >= 1 is (current intersection after the turns among tokens 2..p, destination).
    Scored positions are 1 <= p <= length - 2: every prediction of a turn or of `end`. train_targets()
    gives exactly these positions a training target, and a test checks that they are the same.

Data (built once by src/domains/maps_build.py into DATA_DIR, recorded in src/domains/maps_data_manifest.json)
    {distribution}_{split}.npy   uint16 (routes, MAX_TOKENS), 0 padded. train = the 2,000,000 route
                                 training cache; val and test = held out routes split by start and
                                 destination pair, shared by all three distributions
    pairs_val.npy, pairs_test.npy  int64 (pairs, 2): the OpenStreetMap ids of each split's pairs
    Splits: train = model training only; val = checks during training and diagnostic fitting;
    test = final diagnostic scores only. No val or test pair is in any training cache.

Import it like the rest of src (no package install needed):
    spec = importlib.util.spec_from_file_location("maps", ROOT / "src/domains/maps.py")
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("maps")

ROOT = Path(__file__).resolve().parents[2]

# ==========================================================================
# Constants
# ==========================================================================

DOMAIN = "maps"
DISTRIBUTIONS = ["shortest_paths", "noisy_shortest_paths", "random_walks"]
VAFA_NAMES = {"shortest_paths": "shortest-paths", "noisy_shortest_paths": "noisy-shortest-paths",
              "random_walks": "random-walks"}
SPLITS = ["train", "val", "test"]

EDGES_CSV = ROOT / "src/domains/maps_edges.csv"
COORDS_CSV = ROOT / "src/domains/maps_coordinates.csv"
MANIFEST = ROOT / "src/domains/maps_data_manifest.json"
VAFA_DATA_DIR = os.environ.get("WMC_VAFA_DATA", os.path.expanduser("~/artifacts/world-model-evaluation/data"))
DATA_DIR = os.environ.get("WMC_MAPS_DATA", os.path.expanduser("~/artifacts/maps"))

TURNS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# The shortest paths map calls one intersection 42438066; the canonical map calls it 7732571914
# (same neighbours in and out, about 2 to 3 m apart in OpenStreetMap). Routes are renamed on input.
RENAME_SHORTEST_PATHS = {42438066: 7732571914}

MAX_TOKENS = 99           # longest route in every split, all distributions (design 3.6)
N_CTX = MAX_TOKENS - 1    # tokens a model reads: 98
MAX_TURNS = MAX_TOKENS - 3  # start, destination, turns, end: at most 96 turns
PAD = 0

SPLIT_SEED = 0            # val / test split of the held out pairs
CACHE_SEED = 0            # draw of the training cache
N_CACHE = 2_000_000       # training routes per distribution


# ==========================================================================
# The map
# ==========================================================================

def _read_edges(path: Path = EDGES_CSV) -> list[tuple[int, str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python src/domains/maps_build.py export-map` on the EC2 "
                                "(design 3.2), or check out a commit that has it")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [(int(r["node"]), r["turn"], int(r["next_node"])) for r in rows]


_EDGES = _read_edges()
NODE_IDS = np.array(sorted({u for u, _, _ in _EDGES} | {v for _, _, v in _EDGES}), dtype=np.int64)
N_NODES = len(NODE_IDS)
NODE_TO_TOKEN: dict[int, int] = {int(n): i + 1 for i, n in enumerate(NODE_IDS)}
TURN0 = N_NODES + 1                                   # token of N; TURN0 + i is TURNS[i]
TURN_TOKEN: dict[str, int] = {t: TURN0 + i for i, t in enumerate(TURNS)}
END = N_NODES + 9
VOCAB = N_NODES + 10                                  # pad, intersections, 8 turns, end

# NEXT[node token, i] = token of the intersection reached by TURNS[i], or -1 if that turn does not exist.
# Row 0 (padding) has no turns.
NEXT = np.full((N_NODES + 1, len(TURNS)), -1, dtype=np.int32)
for _u, _t, _v in _EDGES:
    _i = TURNS.index(_t)
    if NEXT[NODE_TO_TOKEN[_u], _i] != -1:
        raise ValueError(f"{EDGES_CSV}: turn {_t} out of {_u} is listed twice")
    NEXT[NODE_TO_TOKEN[_u], _i] = NODE_TO_TOKEN[_v]
N_EDGES = len(_EDGES)
OUT_DEGREE = (NEXT[1:] >= 0).sum(1)                   # per intersection, index = token - 1
IS_DEAD_END = np.concatenate([[False], OUT_DEGREE == 0])  # index = token

# The 9 possible next moves ("slots"): the 8 turns, then end. SLOT_TOKEN[s] is the token of slot s;
# TOKEN_SLOT[token] is its slot, or -1 for padding and intersections (never a legal next token).
SLOT_TOKEN = np.array([TURN_TOKEN[t] for t in TURNS] + [END], dtype=np.int64)
END_SLOT = len(TURNS)
TOKEN_SLOT = np.full(VOCAB, -1, dtype=np.int64)
TOKEN_SLOT[SLOT_TOKEN] = np.arange(len(SLOT_TOKEN))


def is_node(tok: int) -> bool:
    return 1 <= tok <= N_NODES


def token_of_node(node_id: int) -> int:
    return NODE_TO_TOKEN[int(node_id)]


def node_of_token(tok: int) -> int:
    if not is_node(tok):
        raise ValueError(f"token {tok} is not an intersection")
    return int(NODE_IDS[tok - 1])


def token_name(tok: int) -> str:
    """Readable form of one token: the OpenStreetMap id, a turn, end or <pad>."""
    tok = int(tok)
    if tok == PAD:
        return "<pad>"
    if is_node(tok):
        return str(node_of_token(tok))
    if tok == END:
        return "end"
    if TURN0 <= tok < TURN0 + len(TURNS):
        return TURNS[tok - TURN0]
    raise ValueError(f"token {tok} is outside the vocabulary of {VOCAB}")


def encode(route: str) -> np.ndarray:
    """'start destination turn ... end' (OpenStreetMap ids of the canonical map) -> int64 tokens.
    Raises KeyError on a word that is not an intersection of the canonical map, a turn or end."""
    words = route.split()
    if len(words) < 3:
        raise ValueError(f"a route needs start, destination and end: {route!r}")
    out = [NODE_TO_TOKEN[int(words[0])], NODE_TO_TOKEN[int(words[1])]]
    for w in words[2:]:
        out.append(END if w == "end" else TURN_TOKEN[w])
    return np.array(out, dtype=np.int64)


def decode(tokens) -> str:
    return " ".join(token_name(t) for t in np.asarray(tokens).ravel() if int(t) != PAD)


def legal_next(cur: int, dest: int) -> list[int]:
    """Tokens that may legally follow when the car is at intersection token `cur` with destination
    token `dest`: the existing turns (in TURNS order), then end if cur is the destination. Empty at a
    dead end that is not the destination."""
    out = [TURN0 + i for i in range(len(TURNS)) if NEXT[cur, i] >= 0]
    if cur == dest:
        out.append(END)
    return out


def replay(tokens) -> list[int] | None:
    """Replay one route. Returns the intersection token at every position 1..length-1 (index 0 of the
    list is position 1, the start), or None if the route is not legal: unknown start or destination,
    a turn that does not exist, end anywhere but at the destination, end missing, or anything after end."""
    tok = [int(t) for t in np.asarray(tokens).ravel()]
    while tok and tok[-1] == PAD:
        tok.pop()
    if len(tok) < 3 or len(tok) > MAX_TOKENS or not is_node(tok[0]) or not is_node(tok[1]) or tok[-1] != END:
        return None
    cur, dest, states = tok[0], tok[1], [tok[0]]
    for p in range(2, len(tok)):
        t = tok[p]
        if t not in legal_next(cur, dest):
            return None
        if t == END:
            if p != len(tok) - 1:
                return None
            states.append(cur)
        else:
            cur = int(NEXT[cur, t - TURN0])
            states.append(cur)
    return states


# ==========================================================================
# Routes -> tokens, true state and legal next moves (vectorized)
# ==========================================================================

def pad_routes(routes: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """List of token arrays (or an array) -> (G, MAX_TOKENS) int64, 0 padded."""
    if isinstance(routes, np.ndarray) and routes.ndim == 2:
        if routes.shape[1] > MAX_TOKENS and (routes[:, MAX_TOKENS:] != PAD).any():
            raise ValueError(f"a route is longer than {MAX_TOKENS} tokens")
        out = np.zeros((len(routes), MAX_TOKENS), dtype=np.int64)
        w = min(routes.shape[1], MAX_TOKENS)
        out[:, :w] = routes[:, :w]
        return out
    out = np.zeros((len(routes), MAX_TOKENS), dtype=np.int64)
    for g, r in enumerate(routes):
        r = np.asarray(r)
        if len(r) > MAX_TOKENS:
            raise ValueError(f"route {g} has {len(r)} tokens, more than {MAX_TOKENS}")
        out[g, :len(r)] = r
    return out


def build(routes) -> dict[str, np.ndarray | int]:
    """Replay every route and return, per route and position p (0 .. MAX_TOKENS - 1):
        tokens       (G, MAX_TOKENS) int64, 0 = padding
        length       (G,) int64, number of real tokens
        score        (G, MAX_TOKENS) bool, True at the scored positions 1 <= p <= length - 2
        current      (G, MAX_TOKENS) int64, intersection token at position p (1 <= p <= length - 1), else 0
        dest         (G,) int64, destination token
        n_turns      (G, MAX_TOKENS) int64, turns among tokens 2..p (1 <= p <= length - 1), else 0
        legal        (G, MAX_TOKENS, 9) bool, legal[g, p, s]: slot s (TURNS, then end) is a legal next
                     move after position p; all False where score is False
        n_illegal    routes dropped because they are not legal (expected 0 on the built data)
        kept         (G_in,) bool, which input routes were kept
    """
    tok = pad_routes(routes)
    G = len(tok)
    real = tok != PAD
    length = real.sum(1)
    # padding only at the end, and the routes the vectorized replay below can read
    ok = (length >= 3) & (real == (np.arange(MAX_TOKENS)[None, :] < length[:, None])).all(1)
    ok &= (tok[:, 0] >= 1) & (tok[:, 0] <= N_NODES) & (tok[:, 1] >= 1) & (tok[:, 1] <= N_NODES)
    last = tok[np.arange(G), np.maximum(length - 1, 0)]
    ok &= last == END
    tok = np.where(ok[:, None], tok, 0)
    length = np.where(ok, length, 0)

    current = np.zeros((G, MAX_TOKENS), dtype=np.int64)
    legal = np.zeros((G, MAX_TOKENS, len(SLOT_TOKEN)), dtype=bool)
    dest = tok[:, 1].copy()
    cur = tok[:, 0].copy()
    for p in range(1, MAX_TOKENS):
        if p >= 2:  # apply token p
            t = tok[:, p]
            is_turn = (t >= TURN0) & (t < TURN0 + len(TURNS))
            idx = np.clip(t - TURN0, 0, len(TURNS) - 1)
            nxt = NEXT[cur, idx]
            # move only on a turn that exists; a turn that does not exist is caught by the legality
            # check one position earlier, and cur must never become -1 (it indexes NEXT)
            cur = np.where(is_turn & (p < length) & (nxt >= 0), nxt, cur)
        alive = p <= length - 1
        current[:, p] = np.where(alive, cur, 0)
        scored = p <= length - 2
        if not scored.any():
            continue
        turns_ok = NEXT[cur] >= 0                                   # (G, 8)
        legal[:, p, :len(TURNS)] = turns_ok & scored[:, None]
        legal[:, p, END_SLOT] = (cur == dest) & scored
        # the token actually read next must be legal; end only as the last token
        nt = tok[:, p + 1]
        slot = TOKEN_SLOT[nt]
        good = (slot >= 0) & legal[np.arange(G), p, np.maximum(slot, 0)]
        good &= (nt != END) | (p + 1 == length - 1)
        ok &= ~scored | good
    keep = ok
    n_illegal = int((~keep).sum())
    score = (np.arange(MAX_TOKENS)[None, :] >= 1) & (np.arange(MAX_TOKENS)[None, :] <= length[:, None] - 2)
    pos = np.arange(MAX_TOKENS)[None, :]
    # turns among tokens 2..p: p - 1 up to the last turn, then constant (the end token is not a turn)
    n_turns = np.minimum(pos - 1, length[:, None] - 3)
    n_turns = np.where((pos >= 1) & (pos <= length[:, None] - 1), n_turns, 0)
    out = {"tokens": tok[keep], "length": length[keep], "score": score[keep], "current": current[keep],
           "dest": dest[keep], "n_turns": n_turns[keep], "legal": legal[keep],
           "n_illegal": n_illegal, "kept": keep}
    if n_illegal:
        log.warning("build: %d of %d routes are not legal on the canonical map and were dropped", n_illegal, G)
    return out


def is_legal(data: dict, pred: np.ndarray) -> np.ndarray:
    """(G, P) predicted tokens at positions 0..P-1 -> (G, P) bool, True where the token is legal.
    Only meaningful where data['score'] is True."""
    pred = np.asarray(pred, dtype=np.int64)
    G, P = pred.shape
    slot = TOKEN_SLOT[np.clip(pred, 0, VOCAB - 1)]
    slot = np.where((pred >= 0) & (pred < VOCAB), slot, -1)
    lg = data["legal"][:, :P]
    hit = np.take_along_axis(lg, np.maximum(slot, 0)[..., None], axis=2)[..., 0]
    return hit & (slot >= 0)


def train_targets(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, MAX_TOKENS) route tokens -> (inputs, targets) for next token training.
    inputs = x[:, :N_CTX]; targets = x[:, 1:] with the destination (target of position 0) set to padding,
    so the loss (ignore_index = PAD) covers exactly the scored positions 1 .. length - 2."""
    if x.shape[1] != MAX_TOKENS:
        raise ValueError(f"expected (B, {MAX_TOKENS}) tokens, got {tuple(x.shape)}")
    inputs = x[:, :N_CTX]
    targets = x[:, 1:].clone()
    targets[:, 0] = PAD
    return inputs, targets


@torch.no_grad()
def legal_rate(model, data: dict, device: str, batch: int) -> float:
    """How often the model's top predicted next token is legal, over every scored position (D1 rule).
    `model` takes (B, N_CTX) tokens and returns (B, N_CTX, VOCAB) logits. Needs build(...)."""
    ok = total = 0
    for i in range(0, len(data["tokens"]), batch):
        x = torch.tensor(data["tokens"][i:i + batch, :N_CTX], device=device)
        pred = model(x).argmax(-1).cpu().numpy()
        part = {"legal": data["legal"][i:i + batch]}
        sc = data["score"][i:i + batch, :N_CTX]
        ok += int((is_legal(part, pred) & sc).sum())
        total += int(sc.sum())
    return ok / max(total, 1)


# ==========================================================================
# Distances, coordinates and safe moves
# ==========================================================================

@functools.lru_cache(maxsize=1)
def turn_distance() -> np.ndarray:
    """(N_NODES + 1, N_NODES + 1) int16: fewest turns from intersection token a to token b, -1 if b cannot
    be reached from a. Row and column 0 (padding) are -1. About 42 MB, computed once per process."""
    import scipy.sparse as sps
    import scipy.sparse.csgraph as cg
    src, dst = np.nonzero(NEXT[1:] >= 0)
    rows = src
    cols = NEXT[1:][src, dst] - 1
    A = sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N_NODES, N_NODES))
    D = cg.shortest_path(A, unweighted=True, directed=True)
    out = np.full((N_NODES + 1, N_NODES + 1), -1, dtype=np.int16)
    fin = np.isfinite(D)
    out[1:, 1:][fin] = D[fin].astype(np.int16)
    return out


@functools.lru_cache(maxsize=1)
def coordinates() -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) float64 arrays indexed by intersection token (index 0 is nan)."""
    lat = np.full(N_NODES + 1, np.nan)
    lon = np.full(N_NODES + 1, np.nan)
    with open(COORDS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            t = NODE_TO_TOKEN[int(r["node"])]
            lat[t], lon[t] = float(r["lat"]), float(r["lon"])
    if np.isnan(lat[1:]).any():
        raise ValueError(f"{COORDS_CSV} misses {int(np.isnan(lat[1:]).sum())} intersections")
    return lat, lon


def safe_moves(cur: int, dest: int, turns_left: int) -> list[int]:
    """Legal next tokens after which the route can still end at the destination with at most
    `turns_left` turns in total from here (design 5.1): a turn is safe if the destination is at most
    turns_left - 1 turns from the next intersection; end is safe at the destination."""
    D = turn_distance()
    out = []
    for i in range(len(TURNS)):
        nxt = int(NEXT[cur, i])
        if nxt >= 0 and 0 <= D[nxt, dest] <= turns_left - 1:
            out.append(TURN0 + i)
    if cur == dest:
        out.append(END)
    return out


# ==========================================================================
# Data splits (built by src/domains/maps_build.py)
# ==========================================================================

def split_path(distribution: str, split: str, data_dir: str = DATA_DIR) -> Path:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {distribution!r}, expected one of {DISTRIBUTIONS}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return Path(data_dir) / f"{distribution}_{split}.npy"


def have_data(data_dir: str = DATA_DIR) -> bool:
    return all(split_path(d, s, data_dir).exists() for d in DISTRIBUTIONS for s in SPLITS)


def load_distribution(name: str, split: str, n_routes: int | None = None, seed: int = 0,
                      data_dir: str = DATA_DIR) -> np.ndarray:
    """Routes of one distribution and split as (n, MAX_TOKENS) tokens, 0 padded.
    train  the training cache as a read only uint16 memory map (already in random order); the first
           n_routes rows if n_routes is given. Cast batches to int64 before use.
    val, test  int64; n_routes drawn without replacement with `seed` (all routes if n_routes is None
           or larger than the split, with a warning in the second case)."""
    path = split_path(name, split, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: build the data with src/domains/maps_build.py")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != MAX_TOKENS:
        raise ValueError(f"{path} has shape {arr.shape}, expected (routes, {MAX_TOKENS})")
    if split == "train":
        return arr if n_routes is None else arr[:n_routes]
    if n_routes is None or n_routes >= len(arr):
        if n_routes is not None and n_routes > len(arr):
            log.warning("%s %s has %d routes, asked for %d: using all", name, split, len(arr), n_routes)
        return np.asarray(arr, dtype=np.int64)
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), n_routes, replace=False))
    return np.asarray(arr[idx], dtype=np.int64)


def load_pairs(split: str, data_dir: str = DATA_DIR) -> np.ndarray:
    """(pairs, 2) int64 OpenStreetMap ids (start, destination) of the val or test split."""
    if split not in ("val", "test"):
        raise ValueError("pairs exist for val and test only")
    return np.load(Path(data_dir) / f"pairs_{split}.npy")


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data(data_dir: str = DATA_DIR) -> dict[str, bool]:
    """Compare every data file with the sha256 recorded in maps_data_manifest.json."""
    man = json.loads(MANIFEST.read_text())
    return {name: sha256(Path(data_dir) / name) == digest for name, digest in man["files"].items()}
