"""Diagnostic 4: representation probing with a control task (Hewitt and Liang 2019).

The idea in plain words
    The model only ever sees move sequences. If it keeps a picture of the board inside, a simple
    linear classifier (a probe) should be able to read "mine, theirs or empty" for every square
    from the model's activations. A control probe, trained the same way on labels that have no
    link to the board, shows how much a probe can score by memorizing alone. Selectivity = probe
    accuracy minus control accuracy. Explainer: src/docs/d4_representation_probe.md.

Protocol (fixed in reports/preregistration.md, Section 3 and Amendment 1, A6)
    data         val games of the model's own training distribution, data seed 0: the first
                 10,000 of 11,000 fit the probes, the last 1,000 choose the layer; 1,000 test
                 games give the score
    layers       ZooModel.activations: residual stream after each block (transformer, Mamba),
                 hidden state after each layer (LSTM); layer 0 = output of the first block
    labels       mine / theirs / empty for each of the 64 squares, relative to the player who
                 just moved
    probe        one linear map per layer on activations standardized per feature (mean and
                 standard deviation of the fitting games), -> 64 squares x 3 classes, cross entropy,
                 AdamW lr 1e-3, weight decay 0.01, 2 epochs, activations 256 games at
                 a time, one optimizer step per 1,024 positions, seed 0
    control      input type = (square just played, position); per type and square a fixed random
                 label drawn from that square's class frequencies in the fitting games (seed 0);
                 same probe and training
    selectivity  probe accuracy minus control accuracy (all 64 squares)
    score        selectivity, on test games, of the layer with the highest selectivity on the
                 layer choice games
    details      per layer on test: probe acc, control acc, selectivity, gain over layer 0, probe
                 acc on occupied squares only; best layer and its depth; occupancy only baseline
                 and each layer's gain over it

Run on the EC2 (repo root, conda env wmconcord):
    python src/diagnostics/d4/representation_probe.py --models othello_gpt_synthetic othello_gpt_championship
    python src/diagnostics/d4/representation_probe.py --models othello_transformer_small_synthetic_s0 --save
Zoo models are refused until the git tag prereg-v1.1 exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("d4")

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

DIAGNOSTIC = "D4"
METHOD = ("linear probe mine/theirs/empty per layer, selectivity over a Hewitt and Liang control task "
          "(type = square played + position); layer chosen on val, scored on test; higher is better")

# Protocol constants: fixed by the preregistration (Amendment 1, A6). Change only by amendment.
N_FIT = 10000
N_CHOOSE = 1000
EPOCHS = 2
BATCH = 256                          # games per forward pass
POS_BATCH = 1024                     # positions per optimizer step
LR = 1e-3
WEIGHT_DECAY = 0.01
SEED = 0
N_CLASSES = 3                       # 0 empty, 1 mine, 2 theirs (src/domains/othello.py)
N_TYPES = (oth.VOCAB - 1) * oth.N_CTX   # (square just played, position) pairs
PREREG_TAG = "prereg-v1.1"


# ==========================================================================
# Labels, control task, occupancy baseline
# ==========================================================================

def type_ids(tokens: np.ndarray) -> np.ndarray:
    """(G, 59) input type of every position: (token of the square just played, position)."""
    pos = np.arange(tokens.shape[1])[None, :]
    return np.clip(tokens - 1, 0, None) * oth.N_CTX + pos


def class_freq(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(64, 3) share of each class per square over the real positions."""
    y = labels[mask]                                            # (N, 64)
    return np.stack([(y == c).mean(0) for c in range(N_CLASSES)], axis=1)


def control_table(freq: np.ndarray, seed: int = SEED) -> np.ndarray:
    """(N_TYPES, 64) Hewitt and Liang control labels: one fixed random label per input type and
    square, drawn from that square's class frequencies."""
    rng = np.random.default_rng(seed)
    u = rng.random((N_TYPES, oth.N_SQ))
    cdf = np.cumsum(freq, axis=1)                               # (64, 3)
    return (u[:, :, None] > cdf[None, :, :]).sum(-1).clip(0, N_CLASSES - 1).astype(np.int64)


def occupancy_rule(freq: np.ndarray) -> np.ndarray:
    """(64,) the class an occupied square gets under the occupancy only rule: the square's more
    common disc class (1 mine or 2 theirs) in the fitting games."""
    return np.where(freq[:, 1] >= freq[:, 2], 1, 2)


def occupancy_accuracy(labels: np.ndarray, mask: np.ndarray, rule: np.ndarray) -> float:
    """Accuracy of knowing only which squares are occupied: empty stays empty, occupied gets rule."""
    y = labels[mask]
    pred = np.where(y == 0, 0, rule[None, :])
    return float((pred == y).mean())


# ==========================================================================
# Probes
# ==========================================================================

def _batches(n: int, batch: int, rng: np.random.Generator | None = None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch):
        yield order[i:i + batch]


@torch.no_grad()
def activation_stats(model, data: dict, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """(L, d) mean and standard deviation of every activation feature over the real positions of
    the fitting games. Probes read standardized activations, so the same learning rate works for
    every architecture (LSTM hidden states lie in [-1, 1], residual streams are much larger)."""
    L, d, dev = model.n_layers, model.d_model, model.device
    s1 = torch.zeros(L, d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(L, d, dtype=torch.float64, device=dev)
    n = 0
    for idx in _batches(len(data["tokens"]), batch):
        m = torch.as_tensor(data["mask"][idx], device=dev)
        x = model.activations(torch.as_tensor(data["tokens"][idx])).double()[:, m]   # (L, N, d)
        s1 += x.sum(1)
        s2 += (x * x).sum(1)
        n += x.shape[1]
    mu = s1 / max(n, 1)
    sd = (s2 / max(n, 1) - mu * mu).clamp_min(0).sqrt().clamp_min(1e-6)
    return mu.float(), sd.float()


def fit_probes(model, data: dict, ctrl: np.ndarray, mu: torch.Tensor, sd: torch.Tensor, epochs: int = EPOCHS, batch: int = BATCH,
               pos_batch: int = POS_BATCH, lr: float = LR, weight_decay: float = WEIGHT_DECAY, seed: int = SEED) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Train one real and one control linear probe per layer, all at once from the same activations.
    Activations are computed `batch` games at a time; the probes take one optimizer step per
    `pos_batch` positions, so 10,000 games give about 1,100 steps over 2 epochs."""
    torch.manual_seed(seed)
    L, d, dev = model.n_layers, model.d_model, model.device
    real = nn.ModuleList([nn.Linear(d, oth.N_SQ * N_CLASSES) for _ in range(L)]).to(dev)
    control = nn.ModuleList([nn.Linear(d, oth.N_SQ * N_CLASSES) for _ in range(L)]).to(dev)
    opt = torch.optim.AdamW(list(real.parameters()) + list(control.parameters()), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    types = type_ids(data["tokens"])
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for idx in _batches(len(data["tokens"]), batch, rng):
            m = torch.as_tensor(data["mask"][idx], device=dev)
            acts = model.activations(torch.as_tensor(data["tokens"][idx])).float()   # (L, B, T, d), no grad
            y = torch.as_tensor(data["mine_theirs"][idx], device=dev, dtype=torch.long)[m]   # (N, 64)
            c = torch.as_tensor(ctrl[types[idx]], device=dev)[m]                              # (N, 64)
            x_all = (acts[:, m] - mu[:, None]) / sd[:, None]                               # (L, N, d), standardized
            perm = torch.as_tensor(rng.permutation(len(y)), device=dev)
            for j in range(0, len(y), pos_batch):
                k = perm[j:j + pos_batch]
                loss = 0.0
                for layer in range(L):
                    x = x_all[layer][k]
                    loss = loss + loss_fn(real[layer](x).view(-1, N_CLASSES), y[k].reshape(-1))
                    loss = loss + loss_fn(control[layer](x).view(-1, N_CLASSES), c[k].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
    return real, control


@torch.no_grad()
def evaluate_probes(model, real: nn.ModuleList, control: nn.ModuleList, data: dict, ctrl: np.ndarray,
                    mu: torch.Tensor, sd: torch.Tensor, batch: int = BATCH) -> dict[str, np.ndarray]:
    """Per layer: probe accuracy, control accuracy (on control labels), probe accuracy on occupied squares."""
    L, dev = model.n_layers, model.device
    hit_p, hit_c, hit_o = np.zeros(L), np.zeros(L), np.zeros(L)
    n_all = n_occ = 0
    types = type_ids(data["tokens"])
    for idx in _batches(len(data["tokens"]), batch):
        m = torch.as_tensor(data["mask"][idx], device=dev)
        acts = model.activations(torch.as_tensor(data["tokens"][idx])).float()
        y = torch.as_tensor(data["mine_theirs"][idx], device=dev, dtype=torch.long)[m]
        c = torch.as_tensor(ctrl[types[idx]], device=dev)[m]
        occ = y != 0
        x_all = (acts[:, m] - mu[:, None]) / sd[:, None]
        for layer in range(L):
            x = x_all[layer]
            pred = real[layer](x).view(-1, oth.N_SQ, N_CLASSES).argmax(-1)
            pc = control[layer](x).view(-1, oth.N_SQ, N_CLASSES).argmax(-1)
            hit_p[layer] += float((pred == y).sum())
            hit_c[layer] += float((pc == c).sum())
            hit_o[layer] += float((pred == y)[occ].sum())
        n_all += y.numel()
        n_occ += int(occ.sum())
    return {"probe_acc": hit_p / max(n_all, 1), "control_acc": hit_c / max(n_all, 1),
            "probe_acc_occupied": hit_o / max(n_occ, 1)}


# ==========================================================================
# Data and the diagnostic contract
# ==========================================================================

def training_distribution(model) -> str:
    """The distribution the model was trained on: models.csv for zoo models, meta for anchors."""
    row = results.load_models().get(model.model_id)
    return row["distribution"] if row else model.meta["distribution"]


def load_data(distribution: str, split: str, n_games: int, data_seed: int, n_fit: int = N_FIT,
              n_choose: int = N_CHOOSE) -> tuple[dict, dict, dict]:
    """(fit, choose, test): the first n_fit and last n_choose of n_fit + n_choose val games, and
    n_games games of `split` for the score."""
    val = oth.load_distribution(distribution, "val", n_fit + n_choose, seed=data_seed)
    if len(val) < n_choose + 1:
        raise ValueError(f"only {len(val)} val games for {distribution}")
    fit = oth.build(val[: len(val) - n_choose])
    choose = oth.build(val[len(val) - n_choose:])
    test = oth.build(oth.load_distribution(distribution, split, n_games, seed=data_seed))
    return fit, choose, test


def score(model, split: str = "test", n_games: int = 1000, data_seed: int = 0, **config: Any):
    """The diagnostic contract (src/docs/diagnostic_interface.md). config may override n_fit,
    n_choose, epochs, batch for tests only."""
    t0 = time.time()
    n_fit, n_choose = config.pop("n_fit", N_FIT), config.pop("n_choose", N_CHOOSE)
    dist = training_distribution(model)
    fit, choose, test = load_data(dist, split, n_games, data_seed, n_fit, n_choose)

    freq = class_freq(fit["mine_theirs"], fit["mask"])
    ctrl = control_table(freq, SEED)
    batch = config.get("batch", BATCH)
    mu, sd = activation_stats(model, fit, batch)
    real, control = fit_probes(model, fit, ctrl, mu, sd, **config)
    ch = evaluate_probes(model, real, control, choose, ctrl, mu, sd, batch)
    te = evaluate_probes(model, real, control, test, ctrl, mu, sd, batch)

    sel_choose = ch["probe_acc"] - ch["control_acc"]
    sel_test = te["probe_acc"] - te["control_acc"]
    best = int(np.argmax(sel_choose))
    occ_base = occupancy_accuracy(test["mine_theirs"], test["mask"], occupancy_rule(freq))
    r = lambda a: [round(float(v), 5) for v in a]   # noqa: E731
    details = {
        "best_layer": best, "best_layer_depth": round((best + 1) / model.n_layers, 4), "n_layers": model.n_layers,
        "probe_acc_by_layer": r(te["probe_acc"]), "control_acc_by_layer": r(te["control_acc"]),
        "selectivity_by_layer": r(sel_test), "selectivity_by_layer_choose": r(sel_choose),
        "gain_over_layer0_by_layer": r(te["probe_acc"] - te["probe_acc"][0]),
        "probe_acc_occupied_by_layer": r(te["probe_acc_occupied"]),
        "occupancy_baseline_acc": round(occ_base, 5),
        "gain_over_occupancy_by_layer": r(te["probe_acc"] - occ_base),
        "distribution": dist, "n_fit_games": len(fit["tokens"]), "n_choose_games": len(choose["tokens"]),
        "n_test_games": len(test["tokens"]),
        "illegal_games_dropped": {"fit": fit["n_illegal_games"], "choose": choose["n_illegal_games"],
                                  "test": test["n_illegal_games"]},
    }
    return itf.result_for(model, DIAGNOSTIC, method=METHOD, split=split, n_games=len(test["tokens"]),
                          data_seed=data_seed, score=float(sel_test[best]), details=details,
                          runtime_sec=round(time.time() - t0, 1))


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
    ap = argparse.ArgumentParser(description="Diagnostic 4, representation probing")
    ap.add_argument("--models", nargs="+", required=True, help="othello_gpt_synthetic, othello_gpt_championship or zoo model ids")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_games", type=int, default=1000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write results/D4_probing/{model_id}.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    zoo = [m for m in args.models if not m.startswith("othello_gpt_")]
    if zoo and not prereg_tag_exists():
        sys.exit(f"refusing to score zoo models {zoo}: the git tag {PREREG_TAG} does not exist yet. "
                 "Anchors (othello_gpt_synthetic, othello_gpt_championship) can be scored now.")
    models = [load_any(m) for m in args.models]
    out = itf.run(sys.modules[__name__], models, split=args.split, n_games=args.n_games,
                  data_seed=args.data_seed, save=args.save)
    for res in out:
        d = res.details
        if res.status == "ok":
            b = d["best_layer"]
            print(f"{res.model_id:45s} score {res.score:.4f}  best layer {b}  probe {d['probe_acc_by_layer'][b]:.4f}  "
                  f"control {d['control_acc_by_layer'][b]:.4f}  occupied {d['probe_acc_occupied_by_layer'][b]:.4f}  "
                  f"occupancy baseline {d['occupancy_baseline_acc']:.4f}")
        else:
            print(f"{res.model_id:45s} {res.status}: {res.reason}")


if __name__ == "__main__":
    main()
