"""Train one zoo model on Othello move sequences and measure throughput.

This is the baseline the working session asked for (next-move prediction,
scored against the simulator) and the throughput measurement Review #2 asks
for before PR #19 merges.

What it does:
  1. Loads games from the othello_world .bin files (loader and rules from
     src/domains/othello.py, shared with every diagnostic).
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
    python src/zoo/train.py --arch mamba --scale large --micro_batch 64      # big model: split each batch of 256 into 4 chunks

Smoke test anywhere, no data needed:
    python src/zoo/train.py --synthetic 200 --steps 20 --batch 16 --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
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


rules = _load("othello", "src/domains/othello.py")
matrix = _load("run_matrix", "src/component/run_matrix.py")

VOCAB = rules.VOCAB  # 0 = pad, 1..60 = the playable squares
N_CTX = rules.N_CTX  # 59 moves per game at most


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
    """Mamba in plain PyTorch (mambapy). mamba_ssm does not build on this EC2:
    the Ubuntu 26.04 math headers clash with CUDA 13.1 (rsqrt declaration)."""

    def __init__(self, n_layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        from mambapy.mamba import Mamba as MambaStack, MambaConfig
        self.tok = nn.Embedding(VOCAB, d_model)
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=16, d_conv=4, expand_factor=2)
        self.backbone = MambaStack(cfg)  # backbone.layers: one residual block per layer
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.backbone(self.tok(x))))


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
    # Split each batch into chunks of this size when the model is too big for GPU memory.
    # Gradients add up across chunks, so one optimizer step still uses the full batch.
    mb = cfg["micro_batch"] or cfg["batch"]
    n_chunks = math.ceil(cfg["batch"] / mb)
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
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for c in range(0, cfg["batch"], mb):
            xc = x[c:c + mb]
            logits = model(xc[:, :-1])
            loss = loss_fn(logits.reshape(-1, VOCAB), xc[:, 1:].reshape(-1)) / n_chunks
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        losses.append(step_loss)
        if (step + 1) % cfg["log_every"] == 0 or step == 0:
            log.info("  step %5d  loss %.4f  %.3f s/step", step + 1, step_loss, step_times[-1])
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
    bs = cfg["micro_batch"] or cfg["batch"]  # same chunk size as training, so big models fit
    data = rules.build(games, legal=True)  # tokens, mask, boards, movers, legal next moves
    tok = torch.tensor(data["tokens"], device=dev)
    mask = torch.tensor(data["mask"], device=dev)
    correct = total = 0
    for i in range(0, len(tok), bs):
        x, m = tok[i:i + bs], mask[i:i + bs]
        pred = model(x[:, :-1]).argmax(-1)
        tgt, mm = x[:, 1:], m[:, 1:]
        correct += int(((pred == tgt) & mm).sum())
        total += int(mm.sum())
    legal = rules.legal_rate(model, data, dev, bs)
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
    p.add_argument("--data", default=rules.DATA_DIR)
    p.add_argument("--n_train_games", type=int, default=int(B["n_train_games"]))
    p.add_argument("--n_val_games", type=int, default=1000)
    p.add_argument("--steps", type=int, default=int(B["steps"]),
                   help="steps to actually run; throughput is extrapolated to the full budget")
    p.add_argument("--batch", type=int, default=int(B["batch_games"]))
    p.add_argument("--micro_batch", type=int, default=0,
                   help="split each batch into chunks of this size to fit GPU memory (0 = no split)")
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
        games = np.stack([rules.play_random_game(rng) for _ in range(cfg["synthetic"])])
        n_val = max(10, cfg["synthetic"] // 10)
        train_games, val_games = games[n_val:], games[:n_val]
        log.info("  synthetic: %d train games, %d val games", len(train_games), len(val_games))
    else:
        train_games = rules.load_split(cfg["data"], "train", cfg["n_train_games"])
        val_games = rules.load_split(cfg["data"], "val", cfg["n_val_games"])
        log.info("  %d train games, %d val games from %s", len(train_games), len(val_games), cfg["data"])
    tok, _ = rules.to_tokens(train_games)

    log.info("2. model %s / %s", cfg["arch"], cfg["scale"])
    model = build_model(cfg["arch"], cfg["scale"])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  %s  %.2fM params", matrix.SCALE_CONFIG[cfg["scale"]], n_params / 1e6)

    log.info("3. train %d steps, batch %d games (chunks of %d), device %s",
             cfg["steps"], cfg["batch"], cfg["micro_batch"] or cfg["batch"], cfg["device"])
    tr = train(model, tok, cfg)

    log.info("4. evaluate on %d held-out games", len(val_games))
    ev = evaluate(model, val_games, cfg)

    full_min = tr["sec_per_step"] * cfg["budget_steps"] / 60
    per_scale_h = full_min * 27 / 60          # 27 models per arch and scale (3 domains x 3 distributions x 3 seeds)
    res = {
        "model_id": model_id, "arch": cfg["arch"], "scale": cfg["scale"],
        "scale_config": matrix.SCALE_CONFIG[cfg["scale"]], "n_params": n_params,
        "train": tr, "eval": ev,
        "throughput": {"sec_per_step": tr["sec_per_step"],
                       "min_per_model_full_budget": full_min,
                       "hours_per_27_models": per_scale_h,
                       "budget": {k: B[k] for k in ("steps", "batch_games")},
                       "micro_batch": cfg["micro_batch"]},
        "device": cfg["device"], "gpu": torch.cuda.get_device_name(0) if cfg["device"].startswith("cuda") else "cpu",
        "config": cfg,
    }
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "results.json").write_text(json.dumps(res, indent=2))

    log.info("\nnext-token acc %.3f   legal-move rate %.4f   final loss %.3f",
             ev["next_token_acc"], ev["legal_move_rate"], tr["final_loss"])
    log.info("throughput: %.3f s/step  ->  %.1f min per model at %d steps  ->  %.1f h for 27 models",
             tr["sec_per_step"], full_min, cfg["budget_steps"], per_scale_h)
    row = (f"| {cfg['arch']} | {cfg['scale']} | {n_params/1e6:.2f}M | {cfg['steps']} | "
           f"{tr['sec_per_step']:.3f} | {full_min:.1f} | {per_scale_h:.1f} | "
           f"{ev['legal_move_rate']:.4f} | {ev['next_token_acc']:.3f} |")
    log.info("\n| arch | scale | params | steps timed | s/step | min/model (full) | h / 27 models | legal rate | next-tok acc |")
    log.info("|---|---|---|---|---|---|---|---|---|")
    log.info("%s", row)
    (out / "row.md").write_text(row + "\n")
    log.info("\nsaved to %s", out)


if __name__ == "__main__":
    main()