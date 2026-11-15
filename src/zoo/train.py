"""Train one zoo model (Othello or navigation) and measure throughput.

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
    python src/zoo/train.py --distribution championship --steps 300           # synthetic, championship or mixed
    python src/zoo/train.py --arch mamba --scale large --micro_batch 64      # big model: split each batch of 256 into 4 chunks

Smoke test anywhere, no data needed:
    python src/zoo/train.py --synthetic 200 --steps 20 --batch 16 --device cpu

Navigation (domain maps, reports/preregistration.md Amendment 5, src/docs/navigation_design.md):
    python src/zoo/train.py --domain maps --arch transformer --scale small --distribution shortest_paths --seed 0
    python src/zoo/train.py --domain maps --distribution random_walks --synthetic 200 --steps 20 --batch 16 --device cpu
  * data: the distribution's 2,000,000 route training cache and the same 1,000 validation routes for every
    model (data seed 0), from src/domains/maps.py; the files must match src/domains/maps_data_manifest.json
  * loss on turns and end only (maps.train_targets), averaged over the scored tokens of the whole batch, so
    micro batching gives exactly the full batch gradient; the model reads 98 tokens
  * a full budget run on real data must be a zoo model of models.csv or one of the 9 pilots (small, seed 100),
    and trains from committed code under src/
  * pilots: record in results/pilot_training/, checkpoint in outputs/pilot/, never in the zoo registry
  * a zoo model also needs the git tag prereg-v1.5 and all 9 pilot records (Amendment 5, gates and G2)
  * the whole zoo and the pilots run through src/shellscripts/maps_train_slice.sh (src/docs/maps_zoo_slices.csv)
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
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

VOCAB = rules.VOCAB  # 0 = pad, 1..60 = the playable squares (Othello; navigation sizes come from maps.py)
N_CTX = rules.N_CTX  # 59 moves per game at most
DOMAINS = ["othello", "maps"]


def domain_module(domain: str):
    """The domain module that owns the tokens, data and legal rate: othello.py or maps.py."""
    if domain == "othello":
        return rules
    if domain == "maps":
        import sys
        if "maps" not in sys.modules:
            spec = importlib.util.spec_from_file_location("maps", ROOT / "src/domains/maps.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["maps"] = mod
            spec.loader.exec_module(mod)
        return sys.modules["maps"]
    raise ValueError(f"unknown domain {domain!r}, expected one of {DOMAINS}")


def domain_sizes(domain: str) -> tuple[int, int]:
    """(vocabulary size, position table size) of a domain's models. Othello: 61 and 59 (the model reads 58
    moves and its position table has one spare row, as always); navigation: 4,589 and 98."""
    if domain == "othello":
        return VOCAB, N_CTX
    m = domain_module(domain)
    return m.VOCAB, m.N_CTX


# ==========================================================================
# Models. All take (B, T) tokens and return (B, T, VOCAB) logits.
# Every model also has layer_outputs(x): the vector after each layer, (B, T, d_model) per
# layer, which is what Diagnostic 4 probes, and readout(h): last layer vector -> logits.
# forward(x) is readout(layer_outputs(x)[-1]), so the probed vectors are exactly the ones
# the model uses to predict.
# ==========================================================================

class GPT(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, vocab: int = VOCAB, n_ctx: int = N_CTX) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(n_ctx, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                       batch_first=True, norm_first=True, activation="gelu")
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after each transformer block."""
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        out = []
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
            out.append(h)
        return out

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.layer_outputs(x)[-1])


class LSTM(nn.Module):
    """A stack of single layer LSTMs. Same parameters and computation as one nn.LSTM with
    num_layers=n_layers, but the hidden state after every layer is reachable, so Diagnostic 4
    can probe each layer (one multi layer nn.LSTM only returns the last layer)."""

    def __init__(self, n_layers: int, d_model: int, vocab: int = VOCAB, **_: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab)

    def layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Hidden state after each LSTM layer."""
        h = self.tok(x)
        out = []
        for layer in self.layers:
            h, _ = layer(h)
            out.append(h)
        return out

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.layer_outputs(x)[-1])


class Mamba(nn.Module):
    """Mamba in plain PyTorch (mambapy). mamba_ssm does not build on this EC2:
    the Ubuntu 26.04 math headers clash with CUDA 13.1 (rsqrt declaration)."""

    def __init__(self, n_layers: int, d_model: int, vocab: int = VOCAB, **_: int) -> None:
        super().__init__()
        from mambapy.mamba import Mamba as MambaStack, MambaConfig
        self.tok = nn.Embedding(vocab, d_model)
        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=16, d_conv=4, expand_factor=2)
        self.backbone = MambaStack(cfg)  # backbone.layers: one residual block per layer
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def layer_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after each Mamba block (backbone.layers, as in mambapy's forward)."""
        h = self.tok(x)
        out = []
        for layer in self.backbone.layers:
            h = layer(h)
            out.append(h)
        return out

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.layer_outputs(x)[-1])


ARCHS = {"transformer": GPT, "lstm": LSTM, "mamba": Mamba}


def build_model(arch: str, scale: str, domain: str = "othello") -> nn.Module:
    cfg = matrix.SCALE_CONFIG[scale]
    if domain == "othello":   # exactly as before navigation existed, so every Othello model is unchanged
        return ARCHS[arch](n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"])
    vocab, n_ctx = domain_sizes(domain)
    return ARCHS[arch](n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                       vocab=vocab, n_ctx=n_ctx)


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
    # Navigation routes differ in length, so the loss is summed over the scored tokens of each chunk and divided
    # once by the scored tokens of the whole batch: the mean over the batch's scored tokens, exactly the same with
    # or without micro batching. (Othello positions are all scored, so its chunk means already add up exactly.)
    loss_sum = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    vocab = model.head.out_features
    maps_domain = cfg.get("domain", "othello") == "maps"
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
        x = torch.tensor(np.asarray(tok[idx], dtype=np.int64), device=dev)   # the maps cache is uint16
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        if maps_domain:
            n_scored = int((domain_module("maps").train_targets(x)[1] != 0).sum())
        for c in range(0, cfg["batch"], mb):
            xc = x[c:c + mb]
            if maps_domain:   # loss on turns and end only: the destination target is set to padding
                inputs, targets = domain_module("maps").train_targets(xc)
            else:
                inputs, targets = xc[:, :-1], xc[:, 1:]
            logits = model(inputs)
            if maps_domain:
                loss = loss_sum(logits.reshape(-1, vocab), targets.reshape(-1)) / n_scored
            else:
                loss = loss_fn(logits.reshape(-1, vocab), targets.reshape(-1)) / n_chunks
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
def evaluate_maps(model: nn.Module, routes: np.ndarray, cfg: dict) -> dict:
    """Navigation: next token accuracy and legal rate (the D1 rule, maps.legal_rate) on the validation
    routes, over the scored positions (every prediction of a turn or of end)."""
    m = domain_module("maps")
    dev = cfg["device"]
    model.eval()
    bs = cfg["micro_batch"] or cfg["batch"]
    data = m.build(routes)
    correct = total = 0
    for i in range(0, len(data["tokens"]), bs):
        x = torch.tensor(data["tokens"][i:i + bs], device=dev)
        pred = model(x[:, :m.N_CTX]).argmax(-1)
        sc = torch.tensor(data["score"][i:i + bs, :m.N_CTX], device=dev)
        correct += int(((pred == x[:, 1:]) & sc).sum())
        total += int(sc.sum())
    legal = m.legal_rate(model, data, dev, bs)
    return {"next_token_acc": correct / max(total, 1), "legal_move_rate": legal,
            "n_eval_games": len(data["tokens"]), "illegal_games_dropped": int(data["n_illegal"])}


@torch.no_grad()
def evaluate(model: nn.Module, games: np.ndarray, cfg: dict) -> dict:
    """Next-token accuracy and legal-move rate (D1) on held-out games."""
    if cfg.get("domain", "othello") == "maps":
        return evaluate_maps(model, games, cfg)
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
# Navigation: data, gates and records (Amendment 5)
# ==========================================================================

PREREG_TAG = "prereg-v1.5"
PILOT_SEED = 100
PILOT_RECORDS = ROOT / "results" / "pilot_training"


def pilot_ids() -> list[str]:
    """The 9 navigation pilots of Amendment 5, G2: one small model per architecture and distribution, seed 100."""
    dists = domain_module("maps").DISTRIBUTIONS
    return [f"maps_{a}_small_{d}_s{PILOT_SEED}" for a in ("transformer", "lstm", "mamba") for d in dists]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def in_zoo(model_id: str) -> bool:
    """True if the model is one of the planned zoo models of src/docs/models.csv (pilots are not)."""
    import csv
    with open(ROOT / "src/docs/models.csv", newline="") as f:
        return any(row["model_id"] == model_id for row in csv.DictReader(f))


def maps_gates(model_id: str, full_budget: bool, synthetic: bool) -> None:
    """Amendment 5 for navigation runs on real data at the full budget: only zoo models (models.csv) and the
    9 pilots; always from committed code under src/; a zoo model only with the tag prereg-v1.5 present and
    only after all 9 pilot training records exist in results/pilot_training/ (G2: pilots train first).
    Smoke tests (--synthetic) and throughput runs (fewer steps) are not gated and never leave a record."""
    if synthetic or not full_budget:
        return
    zoo = in_zoo(model_id)
    if not zoo and model_id not in pilot_ids():
        raise SystemExit(f"refusing to train {model_id}: a full budget navigation run must be a zoo model of "
                         f"src/docs/models.csv or one of the 9 pilots (seed {PILOT_SEED}, small)")
    dirty = [line[3:] for line in _git("status", "--porcelain", "--", "src").splitlines() if line.strip()]
    if dirty:
        raise SystemExit(f"refusing to train {model_id}: these files under src/ are not committed: {dirty}")
    if in_zoo(model_id) and not _git("tag", "-l", PREREG_TAG).strip():
        raise SystemExit(f"refusing to train {model_id}: the tag {PREREG_TAG} does not exist here "
                         f"(Amendment 5: no navigation zoo model is trained before it). Run: git fetch --tags")
    missing = [p for p in pilot_ids() if not (PILOT_RECORDS / f"{p}.json").exists()]
    if zoo and missing:
        raise SystemExit(f"refusing to train {model_id}: Amendment 5 (G2) trains all 9 pilots before any zoo model, "
                         f"and these pilot records are not in results/pilot_training/ here: {missing}. "
                         f"Pull main after the pilot records are merged.")


def record_dir(domain: str, model_id: str) -> Path:
    """Where a full budget training record goes: the zoo registry holds exactly the models of models.csv;
    navigation pilots go to results/pilot_training/ (Amendment 5, G2). Othello is unchanged."""
    if domain == "othello" or in_zoo(model_id):
        return ROOT / "results" / "zoo_training"
    return PILOT_RECORDS


def maps_synthetic_routes(n: int, seed: int) -> np.ndarray:
    """n random legal routes on the canonical map (random walks of 1 to 40 turns, the destination is where
    the walk ends), for smoke tests without the data."""
    m = domain_module("maps")
    rng = np.random.default_rng(seed)
    starts = np.flatnonzero(m.OUT_DEGREE > 0) + 1
    out = []
    while len(out) < n:
        cur = int(rng.choice(starts))
        route = [cur, 0]
        for _ in range(int(rng.integers(1, 41))):
            opts = np.flatnonzero(m.NEXT[cur] >= 0)
            if len(opts) == 0:
                break
            i = int(rng.choice(opts))
            route.append(m.TURN0 + i)
            cur = int(m.NEXT[cur, i])
        route[1] = cur
        route.append(m.END)
        out.append(np.array(route, dtype=np.int64))
    return m.pad_routes(out)


def load_maps_data(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(training routes, validation routes) of a navigation distribution, checked against the manifest."""
    m = domain_module("maps")
    if cfg["synthetic"]:
        routes = maps_synthetic_routes(cfg["synthetic"], cfg["seed"])
        n_val = max(10, cfg["synthetic"] // 10)
        return routes[n_val:], routes[:n_val]
    wanted = {f"{cfg['distribution']}_train.npy", f"{cfg['distribution']}_val.npy"}
    bad = [f for f, ok in m.verify_data(cfg["maps_data"]).items() if f in wanted and not ok]
    if bad:
        raise SystemExit(f"these data files do not match src/domains/maps_data_manifest.json: {bad}")
    train_routes = m.load_distribution(cfg["distribution"], "train", data_dir=cfg["maps_data"])
    val_routes = m.load_distribution(cfg["distribution"], "val", cfg["n_val_games"], seed=0, data_dir=cfg["maps_data"])
    return train_routes, val_routes


# ==========================================================================

def main() -> None:
    B = matrix.TRAINING_BUDGET
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="othello", choices=DOMAINS)
    p.add_argument("--arch", default="transformer", choices=sorted(ARCHS))
    p.add_argument("--scale", default="small", choices=matrix.SCALES)
    p.add_argument("--distribution", default="synthetic",
                   help="othello: synthetic, championship or mixed; maps: shortest_paths, noisy_shortest_paths or random_walks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=rules.DATA_DIR, help="synthetic .bin folder (train/ and val/)")
    p.add_argument("--championship_data", default=rules.CHAMPIONSHIP_DIR, help="WTHOR .wtb folder")
    p.add_argument("--maps_data", default=None, help="navigation data folder (default: maps.DATA_DIR, ~/artifacts/maps)")
    p.add_argument("--n_train_games", type=int, default=int(B["n_train_games"]),
                   help="Othello only; navigation always uses the whole 2,000,000 route training cache")
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
    p.add_argument("--out_dir", default=None,
                   help="checkpoint folder (default outputs/zoo; outputs/pilot for the navigation pilots)")
    cfg = vars(p.parse_args())
    cfg["budget_steps"] = int(B["steps"])
    dom = domain_module(cfg["domain"])
    if cfg["distribution"] not in dom.DISTRIBUTIONS:
        p.error(f"--distribution for {cfg['domain']} must be one of {dom.DISTRIBUTIONS}")
    if cfg["domain"] == "maps" and cfg["maps_data"] is None:
        cfg["maps_data"] = dom.DATA_DIR
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model_id = f"{cfg['domain']}_{cfg['arch']}_{cfg['scale']}_{cfg['distribution']}_s{cfg['seed']}"
    if cfg["domain"] == "maps":
        maps_gates(model_id, cfg["steps"] == cfg["budget_steps"], bool(cfg["synthetic"]))
    if cfg["out_dir"] is None:
        cfg["out_dir"] = ("outputs/pilot" if cfg["domain"] == "maps" and not cfg["synthetic"]
                          and model_id in pilot_ids() else "outputs/zoo")
    out = Path(cfg["out_dir"]) / model_id
    out.mkdir(parents=True, exist_ok=True)

    log.info("1. data")
    if cfg["domain"] == "maps":
        tok, val_games = load_maps_data(cfg)
        log.info("  maps %s: %d training routes, %d validation routes%s", cfg["distribution"], len(tok),
                 len(val_games), " (synthetic)" if cfg["synthetic"] else "")
    elif cfg["synthetic"]:
        rng = np.random.default_rng(cfg["seed"])
        games = np.stack([rules.play_random_game(rng) for _ in range(cfg["synthetic"])])
        n_val = max(10, cfg["synthetic"] // 10)
        train_games, val_games = games[n_val:], games[:n_val]
        log.info("  synthetic: %d train games, %d val games", len(train_games), len(val_games))
    else:
        dirs = {"synthetic_dir": cfg["data"], "championship_dir": cfg["championship_data"]}
        train_games = rules.load_distribution(cfg["distribution"], "train", cfg["n_train_games"], cfg["seed"], **dirs)
        val_games = rules.load_distribution(cfg["distribution"], "val", cfg["n_val_games"], cfg["seed"], **dirs)
        log.info("  %s: %d train games, %d val games", cfg["distribution"], len(train_games), len(val_games))
    if cfg["domain"] == "othello":
        tok, _ = rules.to_tokens(train_games)

    log.info("2. model %s / %s", cfg["arch"], cfg["scale"])
    model = build_model(cfg["arch"], cfg["scale"], cfg["domain"])
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

    # Training record for the zoo registry: only for full budget runs on real data, so smoke
    # tests and throughput runs never enter the registry. Small JSON, committed to git;
    # the checkpoint itself stays out of git (outputs/, later S3).
    if not cfg["synthetic"] and cfg["steps"] == cfg["budget_steps"]:
        ckpt = (out / "model.pt").resolve()
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
        record = {**res, "domain": cfg["domain"], "distribution": cfg["distribution"], "seed": cfg["seed"],
                  "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
                  "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  "git_commit": commit, "host": platform.node(),
                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec_dir = record_dir(cfg["domain"], model_id)
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / f"{model_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        log.info("training record: %s", rec_dir / f"{model_id}.json")


if __name__ == "__main__":
    main()