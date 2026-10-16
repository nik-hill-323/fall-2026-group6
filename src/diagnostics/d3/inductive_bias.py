"""Diagnostic 3: inductive bias probe, after Vafa et al. 2025.

The idea in plain words
    Give the model a job it was never trained on, whose answer depends on the true board, and
    almost no labelled examples to learn it from. A model that already tracks the board learns
    the job quickly; one that only learned move statistics cannot. The same job is given to a
    model of the same architecture and size with random weights. The gap between the two is how
    much the pretraining helps on a board question: the model's inductive bias towards the board.
    Explainer and toy version: src/docs/d3_inductive_bias_probe.md, src/diagnostics/d3/toy_probe.py.

Protocol (fixed in reports/preregistration.md, Amendment 1, A7)
    tasks            4 binary questions about one square after a move: "is it mine?" (mine = the
                     player who just moved, the mine_theirs board of src/domains/othello.py) for
                     the squares in TASK_SQUARES. Each needs the colour of a disc, which only the
                     flips give away, so it cannot be read off the move list.
    labelled data    N_ADAPT validation games of the model's own training distribution, data seed
                     0, one uniformly random position per game; the same examples for every task
    adapted          a copy of the whole model plus a new linear head on the last layer vector at
                     that position. Head lr HEAD_LR, rest of the model lr BODY_LR, AdamW, ADAPT_STEPS
                     steps of ADAPT_BATCH examples drawn with replacement
    baseline         same architecture and size with random weights, adapted the same way
    seeds            ADAPT_SEEDS: head init, batch draws and the random baseline's weights
    scored on        n_games test games (1,000 by default), one random position per game, data seed 0
    task score       balanced accuracy (mean of the hit rate on mine and on not mine)
    score            mean over tasks and seeds of (pretrained minus random balanced accuracy);
                     higher = more world model like

Symbolic regression arm (Amendment 2, C; descriptive, the score above is unchanged)
    after each adaptation (pretrained and random, every task and seed), PySR searches for a short
    formula (+, -, *, at most 15 nodes, 10 squares kept) that reproduces the adapted model's
    margin, logit(mine) - logit(not mine), on the test positions from the 64 true squares (+1
    mine, -1 theirs, 0 empty). Law recovered = the task's own square has the largest mine vs theirs
    contrast in the formula, mine raises the margin, and the formula's sign beats chance against
    the true rule. A formula that only sees whether the square is occupied (x squared) is counted
    as the occupancy shortcut, not as recovered. Reported: recovery and occupancy shortcut rates
    (pretrained vs random), per task, stability across the 3 seeds (mean Jaccard of the squares
    used), fit R squared next to a least squares reference on 128 mine / theirs indicators, and
    every formula (details["symbolic_regression"])

The original model is never trained: adaptation always runs on a deep copy, so other diagnostics
see the original weights.

Run on the EC2 (repo root):
    python src/diagnostics/d3/inductive_bias.py --models othello_gpt_synthetic othello_gpt_championship
    python src/diagnostics/d3/inductive_bias.py --models othello_gpt_synthetic --save
Zoo models are refused until the git tag prereg-v1.2 exists (the protocol must be frozen first).
"""

from __future__ import annotations

try:  # PySR (Julia) must be imported before torch, or it can crash (see requirements notes)
    import pysr
except ImportError:  # CI and laptops: the symbolic regression arm then needs a regressor passed in
    pysr = None

import argparse
import copy
import importlib.util
import logging
import inspect
import itertools
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("d3")

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

DIAGNOSTIC = "D3"
METHOD = ("inductive bias probe (Vafa et al. 2025): balanced accuracy of the adapted model minus that of a "
          "random weight model of the same architecture, 4 'is square X mine' tasks, higher is better")

# Protocol constants: fixed by the preregistration (Amendment 1, A7). Change only by amendment.
TASK_SQUARES = ("C4", "F5", "C3", "F6")
N_ADAPT = 64            # labelled validation games per task (one position each)
ADAPT_STEPS = 300
ADAPT_BATCH = 16
HEAD_LR = 1e-3
BODY_LR = 1e-4
WEIGHT_DECAY = 0.01
ADAPT_SEEDS = (0, 1, 2)
POSITION_SEED = 0       # picks the one position per game, for adaptation and scoring data alike
PREREG_TAG = "prereg-v1.2"   # Amendment 2 fixes the symbolic regression arm

# Symbolic regression arm (Amendment 2, C). Descriptive: reported in details, the score is unchanged.
# Inputs: the 64 squares of the true board right after the move, +1 mine, -1 theirs, 0 empty.
# Target: the adapted model's answer as a margin, logit("mine") - logit("not mine"), on the test positions.
SR_BINARY = ["+", "-", "*"]
SR_UNARY: list[str] = []
SR_MAXSIZE = 15
SR_NITERATIONS = 40
SR_POPULATIONS = 15
SR_SELECT_K = 10        # PySR keeps the 10 most relevant squares (random forest) before the search
SR_SEEDS = ADAPT_SEEDS  # one search per adaptation seed, random_state = that seed

_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def square(name: str) -> int:
    """'C4' -> square index 0..63 (row by row from A1 at the top left, as in src/domains/othello.py)."""
    return (int(name[1]) - 1) * oth.SIZE + "ABCDEFGH".index(name[0])


# ==========================================================================
# Data: one random position per game, with "is square X mine" labels
# ==========================================================================

def examples(games: np.ndarray, seed: int = POSITION_SEED) -> dict[str, np.ndarray]:
    """tokens (G, 59), position (G,) and labels (G, n_tasks) in {0, 1} at one random real position
    per game. Games with an illegal move are dropped by oth.build."""
    data = oth.build(games)
    T = data["mask"].sum(1)
    rng = np.random.default_rng(seed)
    pos = np.array([rng.integers(0, t) for t in T], dtype=np.int64)
    g = np.arange(len(T))
    board = data["mine_theirs"][g, pos]                                   # (G, 64) 0 empty, 1 mine, 2 theirs
    labels = np.stack([(board[:, square(s)] == 1) for s in TASK_SQUARES], axis=1).astype(np.int64)
    return {"tokens": data["tokens"], "pos": pos, "labels": labels, "board": board}


def prepare(distribution: str, split: str, n_games: int, data_seed: int) -> dict[str, np.ndarray]:
    key = (distribution, split, n_games, data_seed)
    if key not in _CACHE:
        _CACHE[key] = examples(oth.load_distribution(distribution, split, n_games, seed=data_seed))
    return _CACHE[key]


# ==========================================================================
# Models: the last layer vector at a position, with gradients, for any architecture
# ==========================================================================

def last_layer(module: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """(B, T) tokens -> (B, T, d_model) vector after the last layer. Differentiable."""
    if hasattr(module, "layer_outputs"):                       # src/zoo/train.py models
        return module.layer_outputs(tokens)[-1]
    if hasattr(module, "cfg") and hasattr(module, "blocks"):   # TransformerLens (reference Othello GPT)
        return module(tokens, stop_at_layer=module.cfg.n_layers)
    raise TypeError(f"do not know how to read the last layer of {type(module).__name__}")


def random_like(model, seed: int) -> nn.Module:
    """A model of the same architecture and size with random weights."""
    torch.manual_seed(seed)
    module = model.module
    if hasattr(module, "random_copy"):                         # test models define their own
        return module.random_copy(seed)
    if hasattr(module, "cfg") and hasattr(module, "blocks"):   # TransformerLens
        from transformer_lens import HookedTransformer
        return HookedTransformer(module.cfg)
    row = results.load_models().get(model.model_id)
    if row is not None and hasattr(module, "layer_outputs"):   # zoo model
        train = _load("zoo_train", "src/zoo/train.py")
        return train.build_model(row["arch"], row["scale"])
    fresh = copy.deepcopy(module)
    for m in fresh.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return fresh


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean of the hit rate on each class present in y (0.5 for a constant prediction)."""
    rates = [float((pred[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return float(np.mean(rates))


def adapt_and_score(body: nn.Module, d_model: int, train: dict[str, np.ndarray], test: dict[str, np.ndarray],
                    task: int, seed: int, device: str, steps: int = ADAPT_STEPS, batch: int = ADAPT_BATCH,
                    head_lr: float = HEAD_LR, body_lr: float = BODY_LR) -> dict[str, float]:
    """Adapt body (in place: pass a copy) plus a new head on one task, then score on test."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    body = body.to(device).train()          # train mode: cuDNN LSTM backward needs it; there is no dropout
    head = nn.Linear(d_model, 2).to(device)
    groups = [{"params": head.parameters(), "lr": head_lr}]
    params = [p for p in body.parameters() if p.requires_grad]
    if params:
        groups.append({"params": params, "lr": body_lr})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    tok = torch.as_tensor(train["tokens"], device=device)
    pos = torch.as_tensor(train["pos"], device=device)
    y = torch.as_tensor(train["labels"][:, task], device=device)
    n = len(tok)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=min(batch, n)), device=device)
        h = last_layer(body, tok[idx])[torch.arange(len(idx), device=device), pos[idx]]
        loss = loss_fn(head(h), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    body.eval()
    preds, margins = [], []
    with torch.no_grad():
        for i in range(0, len(test["tokens"]), 256):
            t = torch.as_tensor(test["tokens"][i:i + 256], device=device)
            p = torch.as_tensor(test["pos"][i:i + 256], device=device)
            h = last_layer(body, t)[torch.arange(len(t), device=device), p]
            out = head(h)
            preds.append(out.argmax(-1).cpu().numpy())
            margins.append((out[:, 1] - out[:, 0]).float().cpu().numpy())
    pred = np.concatenate(preds)
    yt = test["labels"][:, task]
    return {"balanced_accuracy": balanced_accuracy(pred, yt), "accuracy": float((pred == yt).mean()),
            "margin": np.concatenate(margins)}


# ==========================================================================
# Symbolic regression arm (Amendment 2, C; Vafa et al. 2025)
#
# Plain words: after adaptation, the model answers "is square X mine?" for 1,000 test positions.
# Symbolic regression searches for a short formula that reproduces those answers from the true
# board. If the formula mainly uses square X itself, the adapted model has learned the true rule
# (the law is recovered); if it leans on other squares, the model answers through a shortcut.
# The random weight model, adapted the same way, is the null. Searches are stochastic, so each
# (task, seed) has its own search and stability across seeds is reported.
# ==========================================================================

def square_name(sq: int) -> str:
    """Square index 0..63 -> 'A1'..'H8' (inverse of square())."""
    return "ABCDEFGH"[sq % oth.SIZE] + str(sq // oth.SIZE + 1)


SQUARE_NAMES = [square_name(s) for s in range(oth.N_SQ)]
# Names PySR sees. A bare square name can clash with a sympy function (E1 is the exponential
# integral), which PySR refuses, so every variable gets a prefix: sq_C4.
SR_VARIABLE_NAMES = [f"sq_{n}" for n in SQUARE_NAMES]


def board_features(board: np.ndarray) -> np.ndarray:
    """(N, 64) mine / theirs / empty codes (1, 2, 0) -> +1 mine, -1 theirs, 0 empty."""
    return np.where(board == 1, 1.0, np.where(board == 2, -1.0, 0.0))


def pysr_fit(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], str, int]:
    """Fit one PySR search with the Amendment 2 settings. Returns (predict, equation, complexity)."""
    if pysr is None:
        raise ImportError("pysr is not installed: install requirements.txt (EC2) or pass regressor=")
    kwargs = dict(binary_operators=SR_BINARY, unary_operators=SR_UNARY, maxsize=SR_MAXSIZE,
                  niterations=SR_NITERATIONS, populations=SR_POPULATIONS, select_k_features=SR_SELECT_K,
                  deterministic=True, parallelism="serial", random_state=seed, progress=False,
                  verbosity=0, model_selection="best")
    params = inspect.signature(pysr.PySRRegressor.__init__).parameters
    with tempfile.TemporaryDirectory(prefix="d3_sr_") as tmp:   # removed after the fit
        if "output_directory" in params:      # PySR 1.x+ writes its hall of fame files here, not in the repo
            kwargs["output_directory"] = tmp
        elif "tempdir" in params:
            kwargs.update(tempdir=tmp, temp_equation_file=True)
        reg = pysr.PySRRegressor(**kwargs)
        reg.fit(X, y, variable_names=SR_VARIABLE_NAMES)
        best = reg.get_best()
    return reg.predict, str(best["equation"]), int(best["complexity"])


def r_squared(pred: np.ndarray, y: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else 0.0


def linear_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R squared of an ordinary least squares fit of y on 128 indicator columns (mine_j and
    theirs_j for every square) plus an intercept: the reference for how much of the model's margin
    any additive board formula can explain. Indicators, not the signed squares, so the reference can
    also express occupancy (x squared), which a formula with * can build."""
    A = np.c_[(X == 1).astype(np.float64), (X == -1).astype(np.float64), np.ones(len(X))]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return r_squared(A @ coef, y)


def _set(X: np.ndarray, j: int, value: float) -> np.ndarray:
    out = X.copy()
    out[:, j] = value
    return out


def describe_law(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y_model: np.ndarray,
                 y_true: np.ndarray, task_sq: int) -> dict[str, Any]:
    """How a fitted formula behaves, square by square, on the test boards.
        contrast importance of square j = mean |f(X, j mine) - f(X, j theirs)|: how much the formula
                             tells mine from theirs on that square (what "is X mine" needs);
        signed contrast    = mean of f(X, j mine) - f(X, j theirs): positive when mine raises the margin;
        occupancy effect   = mean |(f(X, j mine) + f(X, j theirs)) / 2 - f(X, j empty)|: how much the
                             formula reacts to the square being occupied at all.
    Law recovered (Amendment 2, C): the task's own square has the largest contrast importance, its
    signed contrast is positive, and the formula's sign beats chance against the true rule
    (law_agreement above 0.5). A formula in x squared (occupancy) has zero contrast and is counted
    as the occupancy shortcut, never as recovered. linear_r2 is the reference fit."""
    base = np.asarray(predict(X), dtype=np.float64)
    contrast, signed, occupancy = np.zeros(oth.N_SQ), np.zeros(oth.N_SQ), np.zeros(oth.N_SQ)
    for j in range(oth.N_SQ):
        f_mine = np.asarray(predict(_set(X, j, 1.0)), dtype=np.float64)
        f_theirs = np.asarray(predict(_set(X, j, -1.0)), dtype=np.float64)
        f_empty = np.asarray(predict(_set(X, j, 0.0)), dtype=np.float64)
        contrast[j] = float(np.abs(f_mine - f_theirs).mean())
        signed[j] = float((f_mine - f_theirs).mean())
        occupancy[j] = float(np.abs((f_mine + f_theirs) / 2 - f_empty).mean())
    eps = 1e-9
    used = [SQUARE_NAMES[j] for j in np.flatnonzero((contrast > eps) | (occupancy > eps))]
    top = int(np.argmax(contrast))
    agreement = balanced_accuracy((base > 0).astype(np.int64), y_true)
    recovered = bool(contrast[top] > eps and top == task_sq and signed[task_sq] > 0 and agreement > 0.5)
    return {"top_square": SQUARE_NAMES[top] if contrast[top] > eps else None,
            "recovered": recovered,
            "own_square_contrast": round(float(contrast[task_sq]), 4),
            "own_square_signed_contrast": round(float(signed[task_sq]), 4),
            "own_square_contrast_share": round(float(contrast[task_sq] / contrast.sum()) if contrast.sum() > eps else 0.0, 4),
            "own_square_occupancy_effect": round(float(occupancy[task_sq]), 4),
            "occupancy_shortcut": bool(occupancy[task_sq] > eps and contrast[task_sq] <= eps),
            "squares_used": used,
            "fit_r2": round(r_squared(base, y_model), 4),
            "linear_r2": round(linear_r2(X, y_model), 4),
            "law_agreement": round(agreement, 4)}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def symbolic_arm(test: dict[str, np.ndarray], margins: dict[tuple[str, int, int], np.ndarray],
                 seeds, regressor: Callable = pysr_fit) -> dict[str, Any]:
    """One search per (model kind, task, seed) on the adapted model's test margins."""
    X = board_features(test["board"])
    cells = []
    for kind in ("pretrained", "random"):
        for k, sq in enumerate(TASK_SQUARES):
            for seed in seeds:
                predict, eq, complexity = regressor(X, margins[(kind, k, seed)], seed)
                d = describe_law(predict, X, margins[(kind, k, seed)], test["labels"][:, k], square(sq))
                cells.append({"model": kind, "task": f"is {sq} mine", "seed": seed, "equation": eq,
                              "complexity": complexity, **d})

    def rate(kind, task=None, key="recovered"):
        c = [x for x in cells if x["model"] == kind and (task is None or x["task"] == task)]
        return round(float(np.mean([x[key] for x in c])), 4)

    tasks = [f"is {sq} mine" for sq in TASK_SQUARES]
    stability = {}
    for t in tasks:
        used = [x["squares_used"] for x in cells if x["model"] == "pretrained" and x["task"] == t]
        pairs = list(itertools.combinations(used, 2))
        stability[t] = round(float(np.mean([jaccard(a, b) for a, b in pairs])), 4) if pairs else None
    return {
        "recovery_rate_pretrained": rate("pretrained"), "recovery_rate_random": rate("random"),
        "occupancy_shortcut_rate_pretrained": rate("pretrained", key="occupancy_shortcut"),
        "occupancy_shortcut_rate_random": rate("random", key="occupancy_shortcut"),
        "recovery_by_task_pretrained": {t: rate("pretrained", t) for t in tasks},
        "recovery_by_task_random": {t: rate("random", t) for t in tasks},
        "stability_by_task_pretrained": stability,
        "mean_fit_r2_pretrained": round(float(np.mean([x["fit_r2"] for x in cells if x["model"] == "pretrained"])), 4),
        "mean_linear_r2_pretrained": round(float(np.mean([x["linear_r2"] for x in cells if x["model"] == "pretrained"])), 4),
        "mean_fit_r2_random": round(float(np.mean([x["fit_r2"] for x in cells if x["model"] == "random"])), 4),
        "mean_linear_r2_random": round(float(np.mean([x["linear_r2"] for x in cells if x["model"] == "random"])), 4),
        "settings": {"binary_operators": SR_BINARY, "unary_operators": SR_UNARY, "maxsize": SR_MAXSIZE,
                     "niterations": SR_NITERATIONS, "populations": SR_POPULATIONS,
                     "select_k_features": SR_SELECT_K, "seeds": list(seeds), "deterministic": True,
                     "inputs": "64 squares, +1 mine, -1 theirs, 0 empty",
                     "target": "adapted model margin logit(mine) - logit(not mine)"},
        "cells": cells,
    }


# ==========================================================================
# The diagnostic contract
# ==========================================================================

def training_distribution(model) -> str:
    """The distribution the model was trained on: models.csv for zoo models, meta for anchors."""
    row = results.load_models().get(model.model_id)
    return row["distribution"] if row else model.meta["distribution"]


def evaluate(model, train: dict[str, np.ndarray], test: dict[str, np.ndarray], seeds=ADAPT_SEEDS,
             make_random: Callable[[Any, int], nn.Module] = random_like, symbolic: bool = True,
             regressor: Callable | None = None, **adapt: Any) -> dict[str, Any]:
    """Every (task, seed): adapt a copy of the model and a random weight model, score both. Then
    (symbolic=True) the symbolic regression arm on the adapted models' test margins."""
    cells = []
    margins: dict[tuple[str, int, int], np.ndarray] = {}
    for seed in seeds:
        random_body = make_random(model, 1000 + seed)          # one random model per seed, reused across tasks
        for k, sq in enumerate(TASK_SQUARES):
            pre = adapt_and_score(copy.deepcopy(model.module), model.d_model, train, test, k, seed,
                                  model.device, **adapt)
            rnd = adapt_and_score(copy.deepcopy(random_body), model.d_model, train, test, k, seed,
                                  model.device, **adapt)
            margins[("pretrained", k, seed)], margins[("random", k, seed)] = pre["margin"], rnd["margin"]
            cells.append({"task": f"is {sq} mine", "seed": seed,
                          "pretrained_balanced_accuracy": pre["balanced_accuracy"],
                          "random_balanced_accuracy": rnd["balanced_accuracy"],
                          "pretrained_accuracy": pre["accuracy"], "random_accuracy": rnd["accuracy"],
                          "gap": pre["balanced_accuracy"] - rnd["balanced_accuracy"]})
    gaps = np.array([c["gap"] for c in cells])
    per_seed = [float(np.mean([c["gap"] for c in cells if c["seed"] == s])) for s in seeds]
    per_task = {f"is {sq} mine": float(np.mean([c["gap"] for c in cells if c["task"] == f"is {sq} mine"]))
                for sq in TASK_SQUARES}
    out = {
        "score": float(gaps.mean()),
        "score_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
        "gap_by_seed": per_seed,
        "gap_by_task": per_task,
        "pretrained_balanced_accuracy": float(np.mean([c["pretrained_balanced_accuracy"] for c in cells])),
        "random_balanced_accuracy": float(np.mean([c["random_balanced_accuracy"] for c in cells])),
        "positive_rate_adapt": train["labels"].mean(0).round(4).tolist(),
        "positive_rate_test": test["labels"].mean(0).round(4).tolist(),
        "cells": cells,
    }
    if symbolic:
        out["symbolic_regression"] = symbolic_arm(test, margins, seeds, regressor or pysr_fit)
    return out


def score(model, split: str = "test", n_games: int = 1000, data_seed: int = 0, n_adapt: int = N_ADAPT,
          seeds=ADAPT_SEEDS, **adapt: Any):
    """The diagnostic contract (src/docs/diagnostic_interface.md). n_adapt, seeds and adapt
    (steps, batch, head_lr, body_lr, make_random, symbolic, regressor) may be overridden for
    tests only."""
    t0 = time.time()
    dist = training_distribution(model)
    train = prepare(dist, "val", n_adapt, data_seed)
    test = prepare(dist, split, n_games, data_seed)
    out = evaluate(model, train, test, seeds=seeds, **adapt)
    details = {k: v for k, v in out.items() if k not in ("score", "score_std")}
    details.update({"distribution": dist, "tasks": [f"is {s} mine" for s in TASK_SQUARES],
                    "n_adapt_games": len(train["tokens"]), "seeds": list(seeds),
                    "protocol": {"steps": adapt.get("steps", ADAPT_STEPS), "batch": adapt.get("batch", ADAPT_BATCH),
                                 "head_lr": adapt.get("head_lr", HEAD_LR), "body_lr": adapt.get("body_lr", BODY_LR),
                                 "weight_decay": WEIGHT_DECAY, "position_seed": POSITION_SEED}})
    return itf.result_for(model, DIAGNOSTIC, method=METHOD, split=split, n_games=len(test["tokens"]),
                          data_seed=data_seed, score=out["score"], score_std=out["score_std"],
                          details=details, runtime_sec=round(time.time() - t0, 1))


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
    ap = argparse.ArgumentParser(description="Diagnostic 3, inductive bias probe")
    ap.add_argument("--models", nargs="+", required=True, help="othello_gpt_synthetic, othello_gpt_championship or zoo model ids")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_games", type=int, default=1000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write results/D3_inductive_bias/{model_id}.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    zoo = [m for m in args.models if not m.startswith("othello_gpt_")]
    if zoo and not prereg_tag_exists():
        sys.exit(f"refusing to score zoo models {zoo}: the git tag {PREREG_TAG} does not exist yet. "
                 "Anchors (othello_gpt_synthetic, othello_gpt_championship) can be scored now.")
    models = [load_any(m) for m in args.models]
    out = itf.run(sys.modules[__name__], models, split=args.split, n_games=args.n_games,
                  data_seed=args.data_seed, save=args.save)
    for r in out:
        d = r.details
        if r.status == "ok":
            print(f"{r.model_id:45s} score {r.score:.4f}  pretrained {d['pretrained_balanced_accuracy']:.4f}  "
                  f"random {d['random_balanced_accuracy']:.4f}  by seed {[round(x, 3) for x in d['gap_by_seed']]}  "
                  f"games {r.n_games}  {r.provenance.get('runtime_sec', '?')} s")
            if "symbolic_regression" in d:
                sr = d["symbolic_regression"]
                print(f"{'':45s} symbolic regression: recovered pretrained {sr['recovery_rate_pretrained']:.3f}  "
                      f"random {sr['recovery_rate_random']:.3f}  occupancy shortcut pretrained "
                      f"{sr['occupancy_shortcut_rate_pretrained']:.3f}  random {sr['occupancy_shortcut_rate_random']:.3f}  "
                      f"by task {sr['recovery_by_task_pretrained']}  "
                      f"stability {sr['stability_by_task_pretrained']}  fit r2 {sr['mean_fit_r2_pretrained']:.3f}  "
                      f"linear reference r2 {sr['mean_linear_r2_pretrained']:.3f}  "
                      f"(random: fit {sr['mean_fit_r2_random']:.3f}, linear {sr['mean_linear_r2_random']:.3f})")
        else:
            print(f"{r.model_id:45s} {r.status}: {r.reason}")


if __name__ == "__main__":
    main()
