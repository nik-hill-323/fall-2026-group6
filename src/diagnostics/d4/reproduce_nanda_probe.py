"""
D4 reproduction gate: linear probe on Othello GPT (synthetic model).

Reproduces Table 1 of Nanda, Lee and Wattenberg (2023), "Emergent Linear Representations in
World Models of Self-Supervised Sequence Models" (arXiv 2309.00941):

    linear probe, mine / theirs / empty    layer 0: 90.9   layer 4: 99.0   layer 7: 99.5
    linear probe, black / white / empty    layer 0: 62.2   layer 4: 75.0   layer 7: 74.4

What the script does:
  1. Reads games from the othello_world dataset (.bin files, one int8 per move, square 0..63, -1 = padding).
  2. Replays every game with our own 8x8 Othello rules (src/domains/othello.py) to get the true board after each move.
     Any game with an illegal move is counted and dropped (expected: 0).
  3. Runs the synthetic Othello GPT and reads the residual stream after every layer.
  4. Trains one linear probe per layer (all 64 squares x 3 classes at once) on the GPU,
     for two labelings: mine/theirs/empty and black/white/empty.
  5. Tests on held out games and writes a table with our number next to the published one.

"Mine" is the player who just moved at that position. Swapping which class is called mine or
theirs does not change accuracy.

Usage (on the EC2, from the repo root):
    python src/diagnostics/d4/reproduce_nanda_probe.py
    python src/diagnostics/d4/reproduce_nanda_probe.py --n_train_games 20000 --epochs 1   # quick run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("d4_repro")

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("othello", ROOT / "src/domains/othello.py")
oth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oth)

N_SQ = oth.N_SQ
N_CTX = oth.N_CTX

PUBLISHED = {  # Nanda et al. 2023, Table 1, linear probes, accuracy in percent
    "mine_theirs": {0: 90.9, 4: 99.0, 7: 99.5},
    "black_white": {0: 62.2, 4: 75.0, 7: 74.4},
}


# ==========================================================================
# Step 3: model
# ==========================================================================

def load_model(path: str, device: str):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = HookedTransformerConfig(n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
                                  d_vocab=61, n_ctx=N_CTX, act_fn="gelu", normalization_type="LNPre",
                                  device=device)
    model = HookedTransformer(cfg)
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def resid(model, tokens: torch.Tensor) -> torch.Tensor:
    """Residual stream after each of the 8 layers: shape (8, B, T, 512)."""
    names = [f"blocks.{L}.hook_resid_post" for L in range(8)]
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
    return torch.stack([cache[n] for n in names])


# ==========================================================================
# Step 4 and 5: probes
# ==========================================================================

def train_and_test(model, train: dict, test: dict, scheme: str, cfg: dict) -> list[float]:
    """Train 8 linear probes (one per layer) at once, return test accuracy per layer in percent."""
    dev = cfg["device"]
    torch.manual_seed(cfg["seed"])
    probes = nn.ModuleList([nn.Linear(512, N_SQ * 3) for _ in range(8)]).to(dev)
    opt = torch.optim.AdamW(probes.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss()
    G = len(train["tokens"])
    rng = np.random.default_rng(cfg["seed"])
    step = 0
    for ep in range(cfg["epochs"]):
        order = rng.permutation(G)
        for i in range(0, G, cfg["batch"]):
            idx = order[i:i + cfg["batch"]]
            tok = torch.tensor(train["tokens"][idx], device=dev)
            y = torch.tensor(train[scheme][idx], device=dev, dtype=torch.long)  # (B, T, 64)
            m = torch.tensor(train["mask"][idx], device=dev)
            acts = resid(model, tok)  # (8, B, T, 512)
            loss = 0.0
            for L in range(8):
                logits = probes[L](acts[L][m]).view(-1, N_SQ, 3)  # (N, 64, 3)
                loss = loss + loss_fn(logits.reshape(-1, 3), y[m].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                log.info("  %s  epoch %d  step %d  mean loss %.4f", scheme, ep + 1, step, loss.item() / 8)
    correct = torch.zeros(8, device=dev)
    count = 0
    with torch.no_grad():
        for i in range(0, len(test["tokens"]), cfg["batch"]):
            tok = torch.tensor(test["tokens"][i:i + cfg["batch"]], device=dev)
            y = torch.tensor(test[scheme][i:i + cfg["batch"]], device=dev, dtype=torch.long)
            m = torch.tensor(test["mask"][i:i + cfg["batch"]], device=dev)
            acts = resid(model, tok)
            for L in range(8):
                pred = probes[L](acts[L][m]).view(-1, N_SQ, 3).argmax(-1)
                correct[L] += (pred == y[m]).sum()
            count += int(m.sum()) * N_SQ
    return [round(100 * float(c) / count, 2) for c in correct]


# ==========================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/artifacts/othello_gpt_tl/synthetic_model.pth"))
    p.add_argument("--data", default=oth.DATA_DIR)
    p.add_argument("--n_train_games", type=int, default=100000)
    p.add_argument("--n_test_games", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/d4_repro")
    cfg = vars(p.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()

    log.info("Step 1 and 2: loading games and replaying them with the Othello rules")
    train = oth.build(oth.load_split(cfg["data"], "train", cfg["n_train_games"]))
    test = oth.build(oth.load_split(cfg["data"], "val", cfg["n_test_games"]), legal=True)
    log.info("  train games %d, test games %d, illegal games dropped: train %d, test %d",
             len(train["tokens"]), len(test["tokens"]), train["n_illegal_games"], test["n_illegal_games"])

    log.info("Step 3: loading Othello GPT (synthetic)")
    model = load_model(cfg["model"], cfg["device"])
    lr = oth.legal_rate(model, test, cfg["device"], cfg["batch"])
    log.info("  top predicted move is legal %.2f%% of the time on test games", 100 * lr)

    results = {}
    for scheme in ["mine_theirs", "black_white"]:
        log.info("Step 4 and 5: linear probes, %s", scheme)
        results[scheme] = train_and_test(model, train, test, scheme, cfg)

    lines = ["| Labels | Layer | Ours (%) | Nanda et al. 2023 (%) | Difference |", "|---|---|---|---|---|"]
    for scheme in ["mine_theirs", "black_white"]:
        for L in range(8):
            pub = PUBLISHED[scheme].get(L)
            ours = results[scheme][L]
            diff = f"{ours - pub:+.1f}" if pub is not None else ""
            lines.append(f"| {scheme.replace('_', '/')} | {L} | {ours:.1f} | {pub if pub is not None else ''} | {diff} |")
    table = "\n".join(lines)
    log.info("\n%s", table)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = {"config": cfg, "legal_move_rate": lr, "probe_acc_by_layer": results, "published": PUBLISHED,
           "n_train_games": len(train["tokens"]), "n_test_games": len(test["tokens"]),
           "illegal_games_dropped": {"train": train["n_illegal_games"], "test": test["n_illegal_games"]},
           "runtime_sec": round(time.time() - t0, 1)}
    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(cfg["out_dir"], "results.md"), "w") as f:
        f.write(table + "\n")
    log.info("\nSaved results.json and results.md to %s  (%.0f s)", cfg["out_dir"], out["runtime_sec"])


if __name__ == "__main__":
    main()
