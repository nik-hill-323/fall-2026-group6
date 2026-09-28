"""
D4 toy: representation probing on a tiny Othello model.

What this script shows, in five steps:
  1. Play random legal games of 4x4 Othello. The rules give us the true board after every move.
  2. Train a tiny transformer to predict the next move. It only ever sees move tokens, never a board.
  3. Record the model's internal activations (residual stream) at every layer, after every move.
  4. Train a linear probe per board square to read the board from those activations.
     Labels are relative to the player who just moved: empty, mine, theirs.
  5. Control task (Hewitt and Liang, 2019): every input type gets a fixed random label per square.
     A probe that scores well here is memorizing, not reading the board. The input type is the move
     just played plus its position in the game (--control_type move_pos, default), or the move alone
     (--control_type move).
     Selectivity = probe accuracy minus control accuracy.
  6. Layer 0 baseline: the embedding layer already knows which square was just played, so part of
     the board is readable with no computation. Gain over layer 0 = probe accuracy minus layer 0 accuracy.

Results do not matter here. The point is what the test computes.
Runs on a laptop CPU in a few minutes.

Usage:
    python toy_probe.py
    python toy_probe.py --n_games 4000 --train_steps 3000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# 4x4 board, squares numbered 0..15, row by row
SIZE = 4
N_SQ = SIZE * SIZE
EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
PAD = 0  # token 0 is padding, move on square s is token s + 1

log = logging.getLogger("d4_toy")

Board = np.ndarray  # shape (N_SQ,), values EMPTY / BLACK / WHITE


# ==========================================================================
# Step 1: Othello rules and random games
# ==========================================================================

def start_board() -> Board:
    b = np.zeros(N_SQ, dtype=np.int64)
    b[1 * SIZE + 1] = WHITE
    b[1 * SIZE + 2] = BLACK
    b[2 * SIZE + 1] = BLACK
    b[2 * SIZE + 2] = WHITE
    return b


def flips_for(board: Board, sq: int, player: int) -> list[int]:
    """Squares that would flip if `player` plays on `sq`. Empty list means illegal."""
    if board[sq] != EMPTY:
        return []
    opp = WHITE if player == BLACK else BLACK
    r0, c0 = divmod(sq, SIZE)
    flips = []
    for dr, dc in DIRECTIONS:
        r, c, line = r0 + dr, c0 + dc, []
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == opp:
            line.append(r * SIZE + c)
            r, c = r + dr, c + dc
        if line and 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == player:
            flips.extend(line)
    return flips


def legal_moves(board: Board, player: int) -> list[int]:
    return [s for s in range(N_SQ) if flips_for(board, s, player)]


def play_random_game(rng: np.random.Generator) -> tuple[list[int], list[Board], list[int], list[set[int]]]:
    """Returns moves, the board after each move, the mover of each move, and the legal set before each move."""
    board, player = start_board(), BLACK
    moves, boards, movers, legal_sets = [], [], [], []
    while True:
        legal = legal_moves(board, player)
        if not legal:
            player = WHITE if player == BLACK else BLACK  # pass
            legal = legal_moves(board, player)
            if not legal:
                break  # nobody can move, game over
        sq = int(rng.choice(legal))
        legal_sets.append(set(legal))
        for f in flips_for(board, sq, player):
            board[f] = player
        board[sq] = player
        moves.append(sq)
        boards.append(board.copy())
        movers.append(player)
        player = WHITE if player == BLACK else BLACK
    return moves, boards, movers, legal_sets


def relative_labels(board: Board, mover: int) -> np.ndarray:
    """Per square: 0 empty, 1 mine (player who just moved), 2 theirs."""
    lab = np.zeros(N_SQ, dtype=np.int64)
    lab[board == mover] = 1
    lab[(board != mover) & (board != EMPTY)] = 2
    return lab


# ==========================================================================
# Step 2: tiny transformer that predicts the next move
# ==========================================================================

class TinyGPT(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(N_SQ + 1, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_SQ + 1)

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        acts = [h]  # layer 0 = embeddings only
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
            acts.append(h)  # Step 3: residual stream after each layer
        logits = self.head(self.norm(h))
        return (logits, acts) if return_acts else logits


def to_tensor(games: list[dict], max_len: int) -> torch.Tensor:
    X = np.full((len(games), max_len), PAD, dtype=np.int64)
    for i, g in enumerate(games):
        X[i, :len(g["moves"])] = np.array(g["moves"]) + 1
    return torch.tensor(X)


def train_model(model: TinyGPT, X: torch.Tensor, steps: int, batch: int, lr: float, seed: int) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, X.shape[0], (batch,))
        xb = X[idx]
        logits = model(xb[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0:
            log.info("  step %5d  loss %.3f", step, loss.item())
    model.eval()


@torch.no_grad()
def legal_move_rate(model: TinyGPT, games: list[dict], X: torch.Tensor) -> float:
    """Sanity check: how often the model's top next move is legal."""
    logits = model(X[:, :-1])
    pred = logits.argmax(-1) - 1  # back to square index
    ok = total = 0
    for i, g in enumerate(games):
        for t in range(len(g["moves"]) - 1):
            ok += int(pred[i, t].item() in g["legal"][t + 1])
            total += 1
    return ok / total


# ==========================================================================
# Steps 3 to 6: activations, probes, control task, selectivity, layer 0 baseline
# ==========================================================================

@torch.no_grad()
def collect(model: TinyGPT, games: list[dict], X: torch.Tensor) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """For every layer: activation vectors, true relative board labels, and the input type
    (square just played), one row per move."""
    _, acts = model(X, return_acts=True)
    feats: list[list[np.ndarray]] = [[] for _ in acts]
    labels, types = [], []
    for i, g in enumerate(games):
        for t in range(len(g["moves"])):
            for L, a in enumerate(acts):
                feats[L].append(a[i, t].numpy())
            labels.append(relative_labels(g["boards"][t], g["movers"][t]))
            types.append((g["moves"][t], t))
    return [np.stack(f) for f in feats], np.stack(labels), np.array(types)


def type_ids(types: np.ndarray, control_type: str, max_len: int) -> np.ndarray:
    """Map (square just played, position) pairs to integer input types."""
    if control_type == "move":
        return types[:, 0]
    return types[:, 0] * max_len + types[:, 1]


def make_control_task(Y: np.ndarray, n_types: int, rng: np.random.Generator) -> np.ndarray:
    """Hewitt and Liang (2019) control task. For each square, every input type is given one fixed
    random label, drawn from that square's real label distribution.
    Returns a (n_types, N_SQ) lookup table: control label = table[type, square]."""
    table = np.zeros((n_types, N_SQ), dtype=np.int64)
    for s in range(N_SQ):
        freq = np.bincount(Y[:, s], minlength=3) / len(Y)
        table[:, s] = rng.choice(3, size=n_types, p=freq)
    return table


def fit_probes(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray) -> tuple[float, list[LogisticRegression]]:
    """One logistic regression per square. Returns mean test accuracy and the fitted probes."""
    probes, accs = [], []
    for s in range(N_SQ):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, Ytr[:, s])
        accs.append(clf.score(Xte, Yte[:, s]))
        probes.append(clf)
    return float(np.mean(accs)), probes


def show_board(lab: np.ndarray) -> list[str]:
    sym = {0: ".", 1: "X", 2: "O"}
    return ["".join(sym[int(v)] for v in lab[r * SIZE:(r + 1) * SIZE]) for r in range(SIZE)]


# ==========================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_games", type=int, default=3000)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control_type", choices=["move_pos", "move"], default="move_pos")
    p.add_argument("--out_dir", default="outputs/d4_toy")
    cfg = vars(p.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t0 = time.time()

    log.info("Step 1: playing random 4x4 Othello games")
    games = []
    for _ in range(cfg["n_games"]):
        m, b, mv, lg = play_random_game(rng)
        games.append({"moves": m, "boards": b, "movers": mv, "legal": lg})
    max_len = max(len(g["moves"]) for g in games)
    X = to_tensor(games, max_len)
    n_train = int(0.8 * len(games))  # split by game, so test games are never seen
    log.info(f"  {len(games)} games, longest {max_len} moves, {n_train} train / {len(games) - n_train} test")

    log.info("Step 2: training tiny transformer on next move prediction")
    model = TinyGPT(cfg["d_model"], cfg["n_layers"], cfg["n_heads"], max_len)
    train_model(model, X[:n_train], cfg["train_steps"], cfg["batch"], cfg["lr"], cfg["seed"])
    legal_rate = legal_move_rate(model, games[n_train:], X[n_train:])
    log.info(f"  top predicted move is legal {legal_rate:.1%} of the time (test games)")

    log.info("Step 3: recording activations at every layer")
    feats_tr, Y_tr, T_tr = collect(model, games[:n_train], X[:n_train])
    feats_te, Y_te, T_te = collect(model, games[n_train:], X[n_train:])
    log.info(f"  {len(Y_tr)} train rows, {len(Y_te)} test rows, vector size {feats_tr[0].shape[1]}")

    log.info("Steps 4 to 6: probe per layer, Hewitt and Liang control task, selectivity, layer 0 baseline")
    n_types = N_SQ * max_len if cfg["control_type"] == "move_pos" else N_SQ
    table = make_control_task(Y_tr, n_types, rng)
    C_tr = table[type_ids(T_tr, cfg["control_type"], max_len)]  # fixed random label per (type, square)
    C_te = table[type_ids(T_te, cfg["control_type"], max_len)]
    majority = float(np.mean([(Y_te[:, s] == np.bincount(Y_tr[:, s]).argmax()).mean() for s in range(N_SQ)]))
    results, best, acc0 = [], None, None
    log.info(f"  {'layer':<14}{'probe acc':>10}{'control acc':>13}{'selectivity':>13}{'gain over L0':>14}")
    for L in range(len(feats_tr)):
        acc, probes = fit_probes(feats_tr[L], Y_tr, feats_te[L], Y_te)
        ctrl, _ = fit_probes(feats_tr[L], C_tr, feats_te[L], C_te)  # scored on control labels
        if L == 0:
            acc0 = acc
        name = "0 (embedding)" if L == 0 else str(L)
        log.info(f"  {name:<14}{acc:>10.3f}{ctrl:>13.3f}{acc - ctrl:>13.3f}{acc - acc0:>14.3f}")
        results.append({"layer": L, "probe_acc": acc, "control_acc": ctrl,
                        "selectivity": acc - ctrl, "gain_over_layer0": acc - acc0})
        if best is None or acc > best[1]:
            best = (L, acc, probes)
    log.info(f"  (always guessing the most common label per square scores {majority:.3f})")

    L, _, probes = best
    g = games[n_train]
    t = len(g["moves"]) // 2
    true_lab = relative_labels(g["boards"][t], g["movers"][t])
    pred_lab = np.array([probes[s].predict(feats_te[L][t:t + 1])[0] for s in range(N_SQ)])
    log.info(f"\nOne test game, after move {t + 1}, decoded from layer {L}  (X = mine, O = theirs, . = empty)")
    log.info("  true    decoded")
    for a, b in zip(show_board(true_lab), show_board(pred_lab)):
        log.info(f"  {a}    {b}")
    log.info(f"  {int((true_lab == pred_lab).sum())} of {N_SQ} squares correct")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = {"config": cfg, "legal_move_rate": legal_rate, "majority_baseline": majority,
           "layer0_probe_acc": acc0, "control_task": f"hewitt_liang_2019, type = {cfg['control_type']}",
           "layers": results, "runtime_sec": round(time.time() - t0, 1)}
    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"\nSaved config, seed and results to {cfg['out_dir']}/results.json  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
