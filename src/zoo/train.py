"""Train one zoo model on Othello move sequences and measure throughput.

This is the baseline the working session asked for (next-move prediction,
scored against the simulator) and the throughput measurement Review #2 asks
for before PR #19 merges.

What it does:
  1. Loads games from the othello_world .bin files (same loader and rules as
     src/diagnostics/d4/reproduce_nanda_probe.py, imported from there).
  2. Builds a model at the requested scale (src/component/run_matrix.py
     SCALE_CONFIG) and trains it to predict the next move.
  3. Times every optimizer step and extrapolates to the fixed zoo budget
     (run_matrix.TRAINING_BUDGET), so a short run gives the minutes per model.
  4. Evaluates on held-out games: next-token accuracy and legal-move rate
     (fraction of top-1 predictions that the rules say are legal). That is D1.
  5. Writes model.pt, config.json and results.json under outputs/zoo/{model_id}/
     and prints one markdown table row for the PR.

Usage (on the EC2, from the repo root):
    python src/zoo/train.py --arch transformer --scale small --steps 300     # throughput
    python src/zoo/train.py --arch transformer --scale small                 # full budget
    python src/zoo/train.py --arch lstm --scale medium --steps 300

Smoke test anywhere, no data needed:
    python src/zoo/train.py --synthetic 200 --steps 20 --batch 16 --device cpu
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("zoo")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load("othello_rules", "src/diagnostics/d4/reproduce_nanda_probe.py")
matrix = _load("run_matrix", "src/component/run_matrix.py")

VOCAB = 61          # 0 = pad, 1..60 = the playable squares
N_CTX = rules.N_CTX  # 59 moves per game at most


# ==========================================================================
# Data
# ==========================================================================

def play_random_game(rng: np.random.Generator) -> np.ndarray:
    """One random legal 8x8 game as 60 int8 squares, -1 padded. For smoke tests."""
    board, player = rules.start_board(), rules.BLACK
    moves: list[int] = []
    while len(moves) < 60:
        legal = rules.legal_moves(board, player)
        if not legal:
            player = rules.WHITE if player == rules.BLACK else rules.BLACK
            legal = rules.legal_moves(board, player)
            if not legal:
                break
        sq = int(rng.choice(legal))
        board[rules.flips_for(board, sq, player)] = player
        board[sq] = player
        moves.append(sq)
        player = rules.WHITE if player == rules.BLACK else rules.BLACK
    out = np.full(60, -1, dtype=np.int8)
    out[: len(moves)] = moves
    return out


def load_split(data_dir: str, split: str, n_games: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.bin")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no .bin files under {data_dir}/{split}")
    return rules.load_games(files, n_games)


def to_tokens(games: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G, 59) int64 tokens with 0 padding, and a bool mask of real positions."""
    G = len(games)
    tok = np.zeros((G, N_CTX), dtype=np.int64)
    for g in range(G):
        mv = games[g][games[g] >= 0][:N_CTX]
        tok[g, : len(mv)] = [rules.SQ_TO_TOKEN[int(s)] for s in mv]
    return tok, tok > 0


# ==========================================================================
# Models. All take (B, T) tokens and return (B, T, VOCAB) logits.
# ==========================================================================

class GPT(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(N_CTX, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                       batch_first=True, norm_first=True, activation="gelu")
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(self.norm(h))


class LSTM(nn.Module):
    def __init__(self, n_layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.tok(x))
        return self.head(h)


class Mamba(nn.Module):
    def __init__(self, n_layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba as MambaBlock  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install mamba-ssm causal-conv1d (needs the CUDA instance)") from e
        self.tok = nn.Embedding(VOCAB, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
                                     for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))
        return self.head(self.norm(h))


ARCHS = {"transformer": GPT, "lstm": LSTM, "mamba": Mamba}


def build_model(arch: str, scale: str) -> nn.Module:
    cfg = matrix.SCALE_CONFIG[scale]
    return ARCHS[arch](n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"])


# ==========================================================================
# Train and evaluate
# ==========================================================================

def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


def train(model: nn.Module, tok: np.ndarray, cfg: dict) -> dict:
    dev = cfg["device"]
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad targets are 0
    rng = np.random.default_rng(cfg["seed"])
    G = len(tok)
    step_times: list[float] = []
    losses: list[float] = []
    t_start = time.time()
    for step in range(cfg["steps"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg["lr"], cfg["warmup_steps"], cfg["budget_steps"])
        idx = rng.integers(0, G, size=cfg["batch"])
        x = torch.tensor(tok[idx], device=dev)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        logits = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        losses.append(loss.item())
        if (step + 1) % cfg["log_every"] == 0 or step == 0:
            log.info("  step %5d  loss %.4f  %.3f s/step", step + 1, loss.item(), step_times[-1])
    timed = step_times[10:] if len(step_times) > 20 else step_times  # drop warm-up jitter
    return {
        "steps_run": cfg["steps"],
        "final_loss": float(np.mean(losses[-20:])),
        "sec_per_step": float(np.mean(timed)),
        "wall_sec": time.time() - t_start,
    }


@torch.no_grad()
def evaluate(model: nn.Module, games: np.ndarray, cfg: dict) -> dict:
    """Next-token accuracy and legal-move rate (D1) on held-out games."""
    dev = cfg["device"]
    model.eval()
    data = rules.build(games)  # tokens, mask, black_white boards, movers
    tok = torch.tensor(data["tokens"], device=dev)
    mask = torch.tensor(data["mask"], device=dev)
    correct = total = 0
    for i in range(0, len(tok), cfg["batch"]):
        x, m = tok[i:i + cfg["batch"]], mask[i:i + cfg["batch"]]
        pred = model(x[:, :-1]).argmax(-1)
        tgt, mm = x[:, 1:], m[:, 1:]
        correct += int(((pred == tgt) & mm).sum())
        total += int(mm.sum())
    legal = rules.legal_rate(model, data, dev, cfg["batch"])
    return {"next_token_acc": correct / max(total, 1), "legal_move_rate": legal,
            "n_eval_games": len(data["tokens"]), "illegal_games_dropped": data["n_illegal_games"]}


# ==========================================================================

def main() -> None:
    B = matrix.TRAINING_BUDGET
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="transformer", choices=sorted(ARCHS))
    p.add_argument("--scale", default="small", choices=matrix.SCALES)
    p.add_argument("--distribution", default="synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=os.path.expanduser("~/artifacts/othello_data/data"))
    p.add_argument("--n_train_games", type=int, default=int(B["n_train_games"]))
    p.add_argument("--n_val_games", type=int, default=1000)
    p.add_argument("--steps", type=int, default=int(B["steps"]),
                   help="steps to actually run; throughput is extrapolated to the full budget")
    p.add_argument("--batch", type=int, default=int(B["batch_games"]))
    p.add_argument("--lr", type=float, default=B["lr"])
    p.add_argument("--warmup_steps", type=int, default=int(B["warmup_steps"]))
    p.add_argument("--weight_decay", type=float, default=B["weight_decay"])
    p.add_argument("--synthetic", type=int, default=0,
                   help="generate this many random legal games instead of reading .bin files")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--out_dir", default="outputs/zoo")
    cfg = vars(p.parse_args())
    cfg["budget_steps"] = int(B["steps"])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model_id = f"othello_{cfg['arch']}_{cfg['scale']}_{cfg['distribution']}_s{cfg['seed']}"
    out = Path(cfg["out_dir"]) / model_id
    out.mkdir(parents=True, exist_ok=True)

    log.info("1. data")
    if cfg["synthetic"]:
        rng = np.random.default_rng(cfg["seed"])
        games = np.stack([play_random_game(rng) for _ in range(cfg["synthetic"])])
        n_val = max(10, cfg["synthetic"] // 10)
        train_games, val_games = games[n_val:], games[:n_val]
        log.info("  synthetic: %d train games, %d val games", len(train_games), len(val_games))
    else:
        train_games = load_split(cfg["data"], "train", cfg["n_train_games"])
        val_games = load_split(cfg["data"], "val", cfg["n_val_games"])
        log.info("  %d train games, %d val games from %s", len(train_games), len(val_games), cfg["data"])
    tok, _ = to_tokens(train_games)

    log.info("2. model %s / %s", cfg["arch"], cfg["scale"])
    model = build_model(cfg["arch"], cfg["scale"])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  %s  %.2fM params", matrix.SCALE_CONFIG[cfg["scale"]], n_params / 1e6)

    log.info("3. train %d steps, batch %d games, device %s", cfg["steps"], cfg["batch"], cfg["device"])
    tr = train(model, tok, cfg)

    log.info("4. evaluate on %d held-out games", len(val_games))
    ev = evaluate(model, val_games, cfg)

    full_min = tr["sec_per_step"] * cfg["budget_steps"] / 60
    per_scale_h = full_min * 81 / 60          # 81 models per scale in the matrix
    res = {
        "model_id": model_id, "arch": cfg["arch"], "scale": cfg["scale"],
        "scale_config": matrix.SCALE_CONFIG[cfg["scale"]], "n_params": n_params,
        "train": tr, "eval": ev,
        "throughput": {"sec_per_step": tr["sec_per_step"],
                       "min_per_model_full_budget": full_min,
                       "hours_per_81_models": per_scale_h,
                       "budget": {k: B[k] for k in ("steps", "batch_games")}},
        "device": cfg["device"], "gpu": torch.cuda.get_device_name(0) if cfg["device"].startswith("cuda") else "cpu",
        "config": cfg,
    }
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "results.json").write_text(json.dumps(res, indent=2))

    log.info("\nnext-token acc %.3f   legal-move rate %.4f   final loss %.3f",
             ev["next_token_acc"], ev["legal_move_rate"], tr["final_loss"])
    log.info("throughput: %.3f s/step  ->  %.1f min per model at %d steps  ->  %.1f h for 81 models",
             tr["sec_per_step"], full_min, cfg["budget_steps"], per_scale_h)
    row = (f"| {cfg['arch']} | {cfg['scale']} | {n_params/1e6:.2f}M | {cfg['steps']} | "
           f"{tr['sec_per_step']:.3f} | {full_min:.1f} | {per_scale_h:.1f} | "
           f"{ev['legal_move_rate']:.4f} | {ev['next_token_acc']:.3f} |")
    log.info("\n| arch | scale | params | steps timed | s/step | min/model (full) | h / 81 models | legal rate | next-tok acc |")
    log.info("|---|---|---|---|---|---|---|---|---|")
    log.info("%s", row)
    (out / "row.md").write_text(row + "\n")
    log.info("\nsaved to %s", out)


if __name__ == "__main__":
    main()
