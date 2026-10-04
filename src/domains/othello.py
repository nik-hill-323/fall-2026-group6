"""Othello domain: the one shared source of truth for the 8x8 rules, the game data, the
tokens, and the true state at every move.

Every part of the project imports from here: the zoo trainer (src/zoo/train.py), all four
diagnostics, and the tests. Nothing else should keep its own copy of the rules.

For each game the domain gives the common triple from the proposal:
    sequence          the moves as tokens
    true state        the board after every move (and who made the move)
    legal next set    the moves that are legal after every move

Board and tokens
    squares   0 to 63, row by row (0 = A1 top left, 63 = H8 bottom right)
    board     int8 array of 64: 0 empty, 1 black, 2 white
    tokens    1 to 60 = the 60 squares that are not the 4 start squares, in order; 0 = padding
              (same vocabulary as Othello GPT, Li et al. 2023)

Data: three training distributions (load_distribution)
    synthetic     othello_world synthetic corpus (Li et al. 2023), random legal moves,
                  repackaged as .bin by alexandretl/othello: one byte per move, square 0 to 63,
                  60 bytes per game, 255 pads games shorter than 60 moves (read as int8: -1).
                  Folder ~/artifacts/othello_data/data with train/ and val/ (WMC_OTHELLO_DATA).
                  Split test = the last SYNTHETIC_TEST_FILES files of val/, split val = the rest.
    championship  WTHOR tournament archive (French Othello Federation), human games 1977 to 2025,
                  one WTH_YYYY.wtb file per year in ~/artifacts/othello_championship
                  (WMC_CHAMPIONSHIP_DATA). Duplicates removed, seeded 80 / 10 / 10 train / val / test split.
    mixed         50% synthetic, 50% championship, shuffled together (same ratio in every split).
    Splits: train = model training only; val = checks during training and diagnostic fitting
    (probe training, D3 adaptation); test = final diagnostic scores only.
    Download everything with src/shellscripts/fetch_othello_data.sh.

Import it like the rest of src (no package install needed):
    spec = importlib.util.spec_from_file_location("othello", ROOT / "src/domains/othello.py")
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import torch

log = logging.getLogger("othello")

# ==========================================================================
# Constants
# ==========================================================================

SIZE = 8
N_SQ = 64
EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CENTER = [27, 28, 35, 36]
PLAYABLE = [s for s in range(N_SQ) if s not in CENTER]  # the 60 squares a move can go to
SQ_TO_TOKEN = {s: i + 1 for i, s in enumerate(PLAYABLE)}
TOKEN_TO_SQ = {t: s for s, t in SQ_TO_TOKEN.items()}
VOCAB = 61   # 0 = pad, 1 to 60 = squares
MAX_MOVES = 60
N_CTX = 59   # Othello GPT reads the first 59 moves and predicts the next one at each position

DATA_DIR = os.environ.get("WMC_OTHELLO_DATA", os.path.expanduser("~/artifacts/othello_data/data"))
CHAMPIONSHIP_DIR = os.environ.get("WMC_CHAMPIONSHIP_DATA", os.path.expanduser("~/artifacts/othello_championship"))
DISTRIBUTIONS = ["synthetic", "championship", "mixed"]
SPLITS = ["train", "val", "test"]
CHAMPIONSHIP_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 0       # fixed: the championship split never changes
SYNTHETIC_TEST_FILES = 10  # last 10 of the 48 synthetic val files are the synthetic test split
MIXED_SYNTHETIC_SHARE = 0.5


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


# ==========================================================================
# Rules
# ==========================================================================

def start_board() -> np.ndarray:
    b = np.zeros(N_SQ, dtype=np.int8)
    b[27], b[36] = WHITE, WHITE
    b[28], b[35] = BLACK, BLACK
    return b


def flips_for(board: np.ndarray, sq: int, player: int) -> list[int]:
    """Squares flipped if `player` plays `sq`. Empty list means the move is illegal."""
    if board[sq] != EMPTY:
        return []
    opp = other(player)
    r0, c0 = divmod(sq, SIZE)
    flips: list[int] = []
    for dr, dc in DIRECTIONS:
        r, c, line = r0 + dr, c0 + dc, []
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == opp:
            line.append(r * SIZE + c)
            r, c = r + dr, c + dc
        if line and 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == player:
            flips.extend(line)
    return flips


def legal_moves(board: np.ndarray, player: int) -> list[int]:
    """Squares `player` can legally play, in square order."""
    return [s for s in PLAYABLE if flips_for(board, s, player)]


def legal_next(board: np.ndarray, mover: int) -> list[int]:
    """Legal next moves after `mover` just moved. The other player moves next; if they
    have no legal move they pass and `mover` moves again. Empty list means the game is over."""
    return legal_moves(board, other(mover)) or legal_moves(board, mover)


def replay(moves: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Replay one game. Returns boards (T, 64) after each move and the mover of each move,
    or None if any move is illegal. A player with no legal move passes, so the mover is
    whichever player can legally make the recorded move, checking the player to move first."""
    board, player = start_board(), BLACK
    boards, movers = [], []
    for sq in moves:
        sq = int(sq)
        flips = flips_for(board, sq, player)
        if not flips:  # current player must have passed
            player = other(player)
            flips = flips_for(board, sq, player)
            if not flips:
                return None
        board[flips] = player
        board[sq] = player
        boards.append(board.copy())
        movers.append(player)
        player = other(player)
    return np.stack(boards), np.array(movers, dtype=np.int8)


def play_random_game(rng: np.random.Generator) -> np.ndarray:
    """One random legal game as 60 int8 squares, -1 padded (same format as the .bin data).
    Each move is picked uniformly from the legal moves, like the othello_world synthetic set."""
    board, player = start_board(), BLACK
    moves: list[int] = []
    while len(moves) < MAX_MOVES:
        legal = legal_moves(board, player)
        if not legal:
            player = other(player)
            legal = legal_moves(board, player)
            if not legal:
                break
        sq = int(rng.choice(legal))
        board[flips_for(board, sq, player)] = player
        board[sq] = player
        moves.append(sq)
        player = other(player)
    out = np.full(MAX_MOVES, -1, dtype=np.int8)
    out[: len(moves)] = moves
    return out


def random_games(n: int, seed: int) -> np.ndarray:
    """n random legal games, (n, 60) int8, -1 padded."""
    rng = np.random.default_rng(seed)
    return np.stack([play_random_game(rng) for _ in range(n)])


# ==========================================================================
# Data files
# ==========================================================================

def list_files(data_dir: str, split: str) -> list[str]:
    """The synthetic .bin files of one split, in number order. train = train/ folder;
    val and test share the val/ folder: test is its last SYNTHETIC_TEST_FILES files."""
    folder = "train" if split == "train" else "val"
    files = sorted(glob.glob(os.path.join(data_dir, folder, "*.bin")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    if split == "val":
        return files[:-SYNTHETIC_TEST_FILES]
    if split == "test":
        return files[-SYNTHETIC_TEST_FILES:]
    return files


def load_games(files: list[str], n_games: int) -> np.ndarray:
    """First n_games games (each 60 int8 moves, -1 padded) from the given .bin files."""
    chunks, total = [], 0
    for f in files:
        a = np.fromfile(f, dtype=np.int8).reshape(-1, MAX_MOVES)
        chunks.append(a)
        total += len(a)
        if total >= n_games:
            break
    games = np.concatenate(chunks)[:n_games]
    if len(games) < n_games:
        log.warning("only %d games available, asked for %d", len(games), n_games)
    return games


def load_split(data_dir: str, split: str, n_games: int) -> np.ndarray:
    files = list_files(data_dir, split)
    if not files:
        raise FileNotFoundError(f"no .bin files under {data_dir}/{split}")
    return load_games(files, n_games)


# ==========================================================================
# Championship games (WTHOR) and the three training distributions
# ==========================================================================

WTHOR_HEADER = 16   # bytes: date, number of games (int32 at offset 4), year, board size, ...
WTHOR_RECORD = 68   # bytes per game: tournament, black id, white id, 2 scores (8 bytes), then 60 moves


def read_wtb(path: str) -> np.ndarray:
    """One WTHOR .wtb file as (G, 60) int8 squares, -1 padded. WTHOR writes a move as
    10 * row + column with row and column 1 to 8 (11 = A1, 88 = H8) and 0 for no move."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(raw[4:8].view("<i4")[0])
    body = raw[WTHOR_HEADER:]
    if len(body) != n * WTHOR_RECORD:
        raise ValueError(f"{path}: header says {n} games but the file holds {len(body) / WTHOR_RECORD}")
    mv = body.reshape(n, WTHOR_RECORD)[:, 8:].astype(np.int16)
    sq = np.where(mv > 0, (mv // 10 - 1) * SIZE + (mv % 10 - 1), -1)
    if ((sq < -1) | (sq > 63)).any():
        raise ValueError(f"{path}: move byte outside 11 to 88")
    return sq.astype(np.int8)


def _keep_legal(games: np.ndarray) -> np.ndarray:
    ok = np.array([replay(g[g >= 0]) is not None for g in games], dtype=bool)
    return games[ok]


def load_championship(data_dir: str = CHAMPIONSHIP_DIR) -> dict[str, np.ndarray | dict]:
    """All WTHOR games, cleaned and split once, then cached in data_dir/championship_split.npz.
    Cleaning: exact duplicate games removed, games with an illegal move removed.
    Split: seeded shuffle, then CHAMPIONSHIP_SPLIT (80 / 10 / 10) into train, val, test."""
    cache = os.path.join(data_dir, "championship_split.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {"train": z["train"], "val": z["val"], "test": z["test"],
                "counts": {k: int(z[k]) for k in ("n_raw", "n_duplicates", "n_illegal")}}
    files = sorted(glob.glob(os.path.join(data_dir, "WTH_*.wtb")))
    if not files:
        raise FileNotFoundError(f"no WTH_*.wtb files under {data_dir}")
    raw = np.concatenate([read_wtb(f) for f in files])
    uniq = np.unique(raw, axis=0)  # also sorts, so the shuffle below is what sets the order
    legal = _keep_legal(uniq)
    rng = np.random.default_rng(SPLIT_SEED)
    legal = legal[rng.permutation(len(legal))]
    n_val = int(round(CHAMPIONSHIP_SPLIT["val"] * len(legal)))
    n_test = int(round(CHAMPIONSHIP_SPLIT["test"] * len(legal)))
    out = {"val": legal[:n_val], "test": legal[n_val:n_val + n_test], "train": legal[n_val + n_test:],
           "counts": {"n_raw": len(raw), "n_duplicates": len(raw) - len(uniq), "n_illegal": len(uniq) - len(legal)}}
    np.savez(cache, train=out["train"], val=out["val"], test=out["test"], **out["counts"])
    log.info("championship: %d raw games, %d duplicates, %d illegal, %d train, %d val, %d test (cached in %s)",
             len(raw), out["counts"]["n_duplicates"], out["counts"]["n_illegal"],
             len(out["train"]), len(out["val"]), len(out["test"]), cache)
    return out


def load_distribution(name: str, split: str, n_games: int, seed: int = 0,
                      synthetic_dir: str = DATA_DIR, championship_dir: str = CHAMPIONSHIP_DIR) -> np.ndarray:
    """(n_games, 60) int8 games from one training distribution and split (train, val or test).
    synthetic     first n_games of the othello_world files (as before)
    championship  n_games drawn from the WTHOR split; if fewer exist, all of them (with a warning)
    mixed         MIXED_SYNTHETIC_SHARE synthetic + the rest championship, shuffled with `seed`"""
    if name not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {name!r}, expected one of {DISTRIBUTIONS}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    rng = np.random.default_rng(seed)
    if name == "synthetic":
        return load_split(synthetic_dir, split, n_games)
    champ = load_championship(championship_dir)[split]
    if name == "championship":
        if n_games >= len(champ):
            if n_games > len(champ):
                log.warning("championship %s has %d games, asked for %d: using all", split, len(champ), n_games)
            return champ
        return champ[rng.choice(len(champ), n_games, replace=False)]
    n_syn = int(round(MIXED_SYNTHETIC_SHARE * n_games))
    n_ch = min(n_games - n_syn, len(champ))
    games = np.concatenate([load_split(synthetic_dir, split, n_syn),
                            champ[rng.choice(len(champ), n_ch, replace=False)]])
    return games[rng.permutation(len(games))]


# ==========================================================================
# Tokens and the true state per move
# ==========================================================================

def to_tokens(games: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G, 59) int64 tokens with 0 padding, and a bool mask of real positions.
    Fast path for training: no replay, so moves are not checked for legality."""
    G = len(games)
    tok = np.zeros((G, N_CTX), dtype=np.int64)
    for g in range(G):
        mv = games[g][games[g] >= 0][:N_CTX]
        tok[g, : len(mv)] = [SQ_TO_TOKEN[int(s)] for s in mv]
    return tok, tok > 0


def build(games: np.ndarray, legal: bool = False) -> dict[str, np.ndarray]:
    """Replay every game and return, per game and position (first 59 moves):
        tokens       (G, 59) int64, 0 = padding
        mask         (G, 59) bool, True at real moves
        black_white  (G, 59, 64) int8 board after the move: 0 empty, 1 black, 2 white
        mine_theirs  (G, 59, 64) int8 same board relative to the mover: 0 empty, 1 mine, 2 theirs
        movers       (G, 59) int8 who made the move
        legal_next   (G, 59, 61) bool, token is a legal next move (only when legal=True)
        n_illegal_games  games dropped because a move was illegal (expected 0)
    """
    G = len(games)
    tokens = np.zeros((G, N_CTX), dtype=np.int64)
    mine = np.zeros((G, N_CTX, N_SQ), dtype=np.int8)
    color = np.zeros((G, N_CTX, N_SQ), dtype=np.int8)
    mask = np.zeros((G, N_CTX), dtype=bool)
    movers_all = np.zeros((G, N_CTX), dtype=np.int8)
    nxt = np.zeros((G, N_CTX, VOCAB), dtype=bool) if legal else None
    keep = np.ones(G, dtype=bool)
    for g in range(G):
        moves = games[g][games[g] >= 0][:N_CTX]
        out = replay(moves)
        if out is None:
            keep[g] = False
            continue
        boards, movers = out
        T = len(moves)
        tokens[g, :T] = [SQ_TO_TOKEN[int(s)] for s in moves]
        color[g, :T] = boards
        rel = np.zeros_like(boards)
        rel[boards == movers[:, None]] = 1  # mine
        rel[(boards != movers[:, None]) & (boards != EMPTY)] = 2  # theirs
        mine[g, :T] = rel
        movers_all[g, :T] = movers
        mask[g, :T] = True
        if legal:
            for t in range(T):
                for s in legal_next(boards[t], int(movers[t])):
                    nxt[g, t, SQ_TO_TOKEN[s]] = True
    out = {"tokens": tokens[keep], "mine_theirs": mine[keep], "black_white": color[keep],
           "mask": mask[keep], "movers": movers_all[keep], "n_illegal_games": int((~keep).sum())}
    if legal:
        out["legal_next"] = nxt[keep]
    return out


# ==========================================================================
# Legal move rate of a model (used by the zoo trainer and the D4 sanity check)
# ==========================================================================

@torch.no_grad()
def legal_rate(model, data: dict, device: str, batch: int) -> float:
    """How often the model's top predicted next move is legal. Needs build(..., legal=True).
    Scores every position that has a recorded next move (the last move of a game is skipped).
    `model` takes (B, 59) tokens and returns (B, 59, 61) logits."""
    if "legal_next" not in data:
        raise ValueError("build the data with legal=True to score legal moves")
    ok = total = 0
    for i in range(0, len(data["tokens"]), batch):
        tok = torch.tensor(data["tokens"][i:i + batch], device=device)
        pred = model(tok).argmax(-1).cpu().numpy()  # (B, T)
        for b in range(len(tok)):
            g = i + b
            T = int(data["mask"][g].sum())
            t = np.arange(T - 1)
            ok += int(data["legal_next"][g, t, pred[b, t]].sum())
            total += T - 1
    return ok / max(total, 1)
