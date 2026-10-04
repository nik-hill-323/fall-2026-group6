"""Toy version of Diagnostic 3, the inductive-bias probe.

World: a walker on a line with positions 0..W-1 and walls at both ends.
Tokens: S<k> (start at k), L, R. Walking into a wall keeps you in place.

Pretraining task: predict the next token on sequences where the walker never
bumps a wall. To know when L or R is impossible the model has to track where
the walker is, so a model that solves this well has some notion of position.

Inductive-bias probe: give the pretrained model a NEW task it has never seen,
"what is the final position?", with only a handful of labelled examples, and
see whether it picks up the true rule (position with walls) or a shortcut
(count R minus count L, which ignores walls). Compare against an untrained
model adapted on the same handful. The gap is the diagnostic.

Run from the repo root:
    python src/diagnostics/d3/toy_probe.py                    # one seed, full walk-through
    python src/diagnostics/d3/toy_probe.py --seeds 0 1 2 3 4  # table over seeds, mean and spread

Writes outputs/d3_toy/results.json with the config and one row per seed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn

W = 6          # line has positions 0..5
T = 12         # moves per sequence
N_ADAPT = 16   # tiny labelled set for the new task
N_TEST = 300   # fresh sequences the adapted models are scored on

VOCAB = [f"S{k}" for k in range(W)] + ["L", "R"]
TOK = {t: i for i, t in enumerate(VOCAB)}


# --------------------------------------------------------------------------- #
# The true world model
# --------------------------------------------------------------------------- #

def step(pos: int, move: str) -> int:
    """Apply one move with walls. This is the ground-truth rule."""
    if move == "L":
        return max(0, pos - 1)
    return min(W - 1, pos + 1)


def walk(bump_walls: bool) -> tuple[list[str], int]:
    """Return (token sequence, final position).

    bump_walls=False: only legal moves, used for pretraining.
    bump_walls=True:  random moves that may hit walls, used for the new task.
    """
    pos = random.randrange(W)
    seq = [f"S{pos}"]
    for _ in range(T):
        if bump_walls:
            move = random.choice("LR")
        else:
            options = [m for m in "LR" if step(pos, m) != pos]
            move = random.choice(options)
        seq.append(move)
        pos = step(pos, move)
    return seq, pos


def shortcut_rule(seq: list[str]) -> int:
    """The tempting wrong rule: start + count(R) - count(L), no walls."""
    start = int(seq[0][1:])
    return start + seq.count("R") - seq.count("L")


def encode(seqs: list[list[str]]) -> torch.Tensor:
    return torch.tensor([[TOK[t] for t in s] for s in seqs])


# --------------------------------------------------------------------------- #
# A tiny sequence model
# --------------------------------------------------------------------------- #

class TinyModel(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.next_token = nn.Linear(hidden, len(VOCAB))   # pretraining head
        self.position = nn.Linear(hidden, W)              # new-task head

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.emb(x))
        return h


def pretrain(model: TinyModel, n: int = 4000, epochs: int = 40, verbose: bool = True) -> None:
    seqs = [walk(bump_walls=False)[0] for _ in range(n)]
    x = encode(seqs)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        total = 0.0
        for i in range(0, n, 64):
            xb = x[i:i + 64]
            logits = model.next_token(model.hidden_states(xb[:, :-1]))
            loss = loss_fn(logits.reshape(-1, len(VOCAB)), xb[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if verbose and ep % 10 == 9:
            print(f"  pretrain epoch {ep + 1:2d}  next-token loss {total / (n / 64):.3f}")


def adapt(model: TinyModel, seqs: list[list[str]], labels: list[int], steps: int = 200) -> None:
    """Fine-tune the WHOLE model on the new task with very little data.

    The new head learns at a normal rate. The body (embedding + GRU) learns
    at a much smaller rate, so adaptation can reshape what pretraining built
    but cannot simply overwrite it. That is the point of the probe: we want to
    see what the pretrained model brings, not what 200 steps can teach from
    scratch.
    """
    x, y = encode(seqs), torch.tensor(labels)
    body = [p for name, p in model.named_parameters() if not name.startswith("position")]
    opt = torch.optim.Adam([
        {"params": model.position.parameters(), "lr": 3e-3},
        {"params": body, "lr": 1e-4},
    ])
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        logits = model.position(model.hidden_states(x)[:, -1])
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def predict_position(model: TinyModel, seqs: list[list[str]]) -> list[int]:
    logits = model.position(model.hidden_states(encode(seqs))[:, -1])
    return logits.argmax(-1).tolist()


def accuracy(pred: list[int], true: list[int]) -> float:
    return sum(p == t for p, t in zip(pred, true, strict=True)) / len(true)


# --------------------------------------------------------------------------- #
# The diagnostic
# --------------------------------------------------------------------------- #

def run(seed: int, verbose: bool = True) -> dict[str, float]:
    """One full run of the probe for one seed. Returns the three accuracies and the gap."""
    random.seed(seed)
    torch.manual_seed(seed)

    def say(msg: str = "") -> None:
        if verbose:
            print(msg)

    say("1. Pretrain a tiny model on next-token prediction (legal walks only)")
    pretrained = TinyModel()
    pretrain(pretrained, verbose=verbose)

    say(f"\n2. New task: predict final position. Only {N_ADAPT} labelled examples.")
    adapt_data = [walk(bump_walls=True) for _ in range(N_ADAPT)]
    adapt_seqs = [s for s, _ in adapt_data]
    adapt_labels = [p for _, p in adapt_data]

    say("   Adapting the pretrained model ...")
    adapt(pretrained, adapt_seqs, adapt_labels)

    say("   Adapting an UNTRAINED model on the same examples (the baseline) ...")
    scratch = TinyModel()
    adapt(scratch, adapt_seqs, adapt_labels)

    say(f"\n3. Test on {N_TEST} fresh sequences that bump into walls")
    test = [walk(bump_walls=True) for _ in range(N_TEST)]
    test_seqs = [s for s, _ in test]
    truth = [p for _, p in test]

    pre_pred = predict_position(pretrained, test_seqs)
    scr_pred = predict_position(scratch, test_seqs)
    cut_pred = [min(W - 1, max(0, shortcut_rule(s))) for s in test_seqs]
    pre_acc, scr_acc, cut_acc = accuracy(pre_pred, truth), accuracy(scr_pred, truth), accuracy(cut_pred, truth)

    say(f"   pretrained then adapted : {pre_acc:.2f}")
    say(f"   untrained then adapted  : {scr_acc:.2f}")
    say(f"   shortcut rule, no walls : {cut_acc:.2f}")

    # On the test cases where the shortcut is wrong (a wall was hit), how often does the
    # adapted model give the shortcut's answer instead of the true one?
    wall = [i for i in range(N_TEST) if cut_pred[i] != truth[i]]
    agrees_shortcut = sum(pre_pred[i] == cut_pred[i] for i in wall) / max(len(wall), 1)
    agrees_truth = sum(pre_pred[i] == truth[i] for i in wall) / max(len(wall), 1)

    say("\n4. What rule did the adapted model pick up? Look at wall-bump cases:")
    say(f"   {'sequence':<40} true  pretrained  scratch  shortcut")
    shown = 0
    for s, t, a, b, c in zip(test_seqs, truth, pre_pred, scr_pred, cut_pred, strict=True):
        if c != t and shown < 8:      # only cases where the shortcut is wrong
            say(f"   {' '.join(s):<40} {t:>4}  {a:>10}  {b:>7}  {c:>8}")
            shown += 1

    gap = pre_acc - scr_acc
    say(f"\nInductive-bias score (pretrained minus scratch): {gap:+.2f}")
    say("Big positive gap  -> pretraining gave the model the world (position with walls).")
    say("Gap near zero     -> pretraining gave it nothing useful for this task.")
    say("Matches shortcut  -> it learned counting, not the world.")

    return {"seed": seed, "pretrained": pre_acc, "scratch": scr_acc, "shortcut": cut_acc, "gap": gap,
            "wall_cases": len(wall), "wall_agrees_truth": agrees_truth, "wall_agrees_shortcut": agrees_shortcut}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out_dir", default="outputs/d3_toy")
    a = p.parse_args()

    rows = [run(seed, verbose=len(a.seeds) == 1) for seed in a.seeds]

    if len(rows) > 1:
        print("| seed | pretrained | scratch | shortcut | gap | on wall cases: true answer | on wall cases: shortcut answer |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['seed']} | {r['pretrained']:.2f} | {r['scratch']:.2f} | {r['shortcut']:.2f} | "
                  f"{r['gap']:+.2f} | {r['wall_agrees_truth']:.2f} | {r['wall_agrees_shortcut']:.2f} |")

        def ms(key: str) -> str:
            vals = [r[key] for r in rows]
            return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"

        print(f"| mean ± sd | {ms('pretrained')} | {ms('scratch')} | {ms('shortcut')} | {ms('gap')} | "
              f"{ms('wall_agrees_truth')} | {ms('wall_agrees_shortcut')} |")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {"W": W, "T": T, "N_ADAPT": N_ADAPT, "N_TEST": N_TEST, "seeds": a.seeds}
    (out / "results.json").write_text(json.dumps({"config": config, "runs": rows}, indent=2))
    print(f"\nsaved {out / 'results.json'}")


if __name__ == "__main__":
    main()
