"""D2 toy check: show, on two concrete pairs of move histories, exactly what Diagnostic 2 computes.

It finds in random legal games
    1. a same state pair: two different move orders that reach the identical board
       (the model should accept the same moves after both: compression)
    2. a different state pair: two histories of the same length whose boards allow different
       moves (the model should accept each board's own moves: distinction)
and prints both boards, the truly legal moves, the moves the model accepts (its top k, k = number
of legal moves) and the error of each pair. Results do not matter here, only what the test computes.

Run on the EC2 (repo root, conda env wmconcord):
    python src/diagnostics/d2/toy_check.py                         # reference Othello GPT (synthetic)
    python src/diagnostics/d2/toy_check.py --model untrained       # tiny untrained model, any laptop
Zoo models are not allowed here: the protocol may only be tried on the reference models
(reports/preregistration.md, Section 3).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("d2_state_equiv", ROOT / "src/diagnostics/d2/state_equiv.py")
d2 = sys.modules.get("d2_state_equiv") or importlib.util.module_from_spec(_spec)
if "d2_state_equiv" not in sys.modules:
    sys.modules["d2_state_equiv"] = d2
    _spec.loader.exec_module(d2)
oth, itf = d2.oth, d2.itf

COLS = "ABCDEFGH"


def name(sq: int) -> str:
    """Square 0..63 to Othello notation: column letter, row number (19 = D3)."""
    return f"{COLS[sq % 8]}{sq // 8 + 1}"


def tok_names(tokens: np.ndarray) -> list[str]:
    return [name(oth.TOKEN_TO_SQ[int(t)]) for t in tokens if t > 0]


def history_text(h: "d2.Histories", i: int) -> str:
    return " ".join(tok_names(h.tokens[h.game[i], : h.pos[i] + 1]))


def board_lines(state: bytes, legal: np.ndarray) -> list[str]:
    """8 rows: B black, W white, * legal next move, . empty."""
    board = np.frombuffer(state[:64], dtype=np.int8)
    legal_sq = {oth.TOKEN_TO_SQ[int(t)] for t in np.flatnonzero(legal)}
    rows = ["   " + " ".join(COLS)]
    for r in range(8):
        cells = []
        for c in range(8):
            s = r * 8 + c
            cells.append("B" if board[s] == oth.BLACK else "W" if board[s] == oth.WHITE else "*" if s in legal_sq else ".")
        rows.append(f"{r + 1}  " + " ".join(cells))
    return rows


def side_by_side(left: list[str], right: list[str], gap: int = 8) -> str:
    w = max(len(x) for x in left)
    return "\n".join(f"{a:<{w}}{' ' * gap}{b}" for a, b in zip(left, right, strict=True))


def set_text(row: np.ndarray) -> str:
    return "{" + ", ".join(sorted(tok_names(np.flatnonzero(row)))) + "}"


def load_model(which: str):
    if which == "untrained":
        train = itf.load_module("zoo_train", "src/zoo/train.py")
        torch.manual_seed(0)
        module = train.ARCHS["transformer"](n_layers=2, d_model=32, n_heads=2)
        m = itf.TrainPyModel(module, "untrained_tiny_transformer", "transformer", 2, 32, "cpu")
        return m
    if which in ("othello_gpt_synthetic", "othello_gpt_championship"):
        path = os.path.join(itf.GPT_DIR, f"{which.removeprefix('othello_gpt_')}_model.pth")
        if not os.path.exists(path):
            sys.exit(f"{path} not found: run on the EC2, or use --model untrained")
        return itf.othello_gpt(which.removeprefix("othello_gpt_"))
    sys.exit("--model must be othello_gpt_synthetic, othello_gpt_championship or untrained (no zoo models here)")


def show_pair(title: str, h, acc: np.ndarray, a: int, b: int, err: float, err_name: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"history 1 ({h.pos[a] + 1} moves): {history_text(h, a)}")
    print(f"history 2 ({h.pos[b] + 1} moves): {history_text(h, b)}")
    print()
    print(side_by_side(["board after history 1"] + board_lines(h.state[a], h.legal[a]),
                       ["board after history 2"] + board_lines(h.state[b], h.legal[b])))
    print()
    print(f"truly legal after 1:   {set_text(h.legal[a])}")
    print(f"truly legal after 2:   {set_text(h.legal[b])}")
    print(f"model accepts after 1: {set_text(acc[a])}")
    print(f"model accepts after 2: {set_text(acc[b])}")
    print(f"{err_name} for this pair: {err:.3f}   (0 is best)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="D2 toy check on two pairs of histories")
    ap.add_argument("--model", default="othello_gpt_synthetic")
    ap.add_argument("--n_games", type=int, default=1000, help="random legal games to search for pairs")
    args = ap.parse_args()

    model = load_model(args.model)
    games = oth.random_games(args.n_games, seed=0)
    h = d2.build_histories(games)
    same = d2.same_state_pairs(h, max_pairs=10 ** 9)
    if len(same) == 0:
        sys.exit("no same state pair found: raise --n_games")
    diff = d2.different_state_pairs(h, max_pairs=200)
    acc = d2.accepted_sets(model, h, batch=128)

    # the shortest same state pair, and a different state pair of the same length with the
    # smallest boundary, so both examples are easy to follow by hand
    s = same[np.argmin(h.pos[same[:, 0]])]
    ce = d2.compression_errors(acc, s[None])[0]
    same_len = np.flatnonzero(h.pos == h.pos[s[0]])
    cand = [(int(i), int(j)) for x, i in enumerate(same_len) for j in same_len[x + 1:]
            if not np.array_equal(h.legal[i], h.legal[j])] or [tuple(p) for p in diff]
    boundary = [int((h.legal[i] ^ h.legal[j]).sum()) for i, j in cand]
    dpair = np.array(cand[int(np.argmin(boundary))])
    de = d2.distinction_errors(acc, h.legal, dpair[None])[0]

    print(f"\nModel: {model.model_id}   (searched {len(h)} distinct histories in {args.n_games} random games)\n")
    show_pair("1. SAME STATE: different move order, identical board (compression)",
              h, acc, int(s[0]), int(s[1]), ce, "compression error")
    print("   A model that tracks the board accepts the same moves after both histories.\n")
    show_pair("2. DIFFERENT STATE: same length, different legal moves (distinction)",
              h, acc, int(dpair[0]), int(dpair[1]), de, "distinction error")
    print("   Boundary moves = legal after exactly one history. A model that tracks the board")
    print("   accepts each of them only after the history where it is legal.\n")

    out = d2.evaluate(model, h, d2.same_state_pairs(h), diff, batch=128)
    print("=" * 78)
    print(f"Over all pairs found here: {out['n_same_state_pairs']} same state, {out['n_different_state_pairs']} different state")
    print(f"compression error {out['compression_error']:.3f}   distinction error {out['distinction_error']:.3f}   "
          f"D2 score {out['score']:.3f}   (1 minus the mean of the two errors)")
    print("This toy run uses random games, not the test split: the real score comes from state_equiv.py.")


if __name__ == "__main__":
    main()
