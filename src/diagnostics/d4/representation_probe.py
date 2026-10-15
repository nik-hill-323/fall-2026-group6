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

Added by Amendment 2 (prereg-v1.2); the score above is unchanged
    mlp          nonlinear probe: one hidden layer of 256 with ReLU, its own control probe, same
                 data, seed and training; per layer accuracy, control, selectivity, its own best
                 layer (chosen on val) and its gain over the linear probe (details["mlp"])
    causal       on one case per test game: flip one disc (mine <-> theirs) so the legal next
                 moves change; at every layer from the middle one (n_layers // 2, or the linear
                 best layer if earlier) to the last, edit the vector at that position until that
                 layer's linear probe reads the edited board (0.99 on the flipped square), and
                 let the model run on. Boundary accuracy = share of
                 moves that became legal (or stopped being legal) that the model now accepts (or
                 drops). causal score = boundary accuracy of the probe edit minus that of a random
                 edit of the same size; 95% bootstrap interval over cases. Label: "not decodable"
                 (selectivity below 0.10), "decodable and causal" (causal score at least 0.20 and
                 interval above 0) or "decodable but not causal" (details["causal"])

Run on the EC2 (repo root, conda env wmconcord):
    python src/diagnostics/d4/representation_probe.py --models othello_gpt_synthetic othello_gpt_championship
    python src/diagnostics/d4/representation_probe.py --models othello_transformer_small_synthetic_s0 --save
Zoo models are refused until the git tag prereg-v1.2 exists (Amendment 2).
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
          "(type = square played + position); layer chosen on val, scored on test; higher is better. "
          "details add an MLP probe with its own control and a causal check (probe guided edit vs random edit)")

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
PREREG_TAG = "prereg-v1.2"          # Amendment 2 fixes the MLP probe and the causal check

# Nonlinear probe (Amendment 2, B1). Reported in details; the score stays the linear selectivity (A6).
MLP_HIDDEN = 256

# Causal check (Amendment 2, B2). Reported in details with a label; the score is unchanged.
CAUSAL_SEED = 0                     # picks the position and the flipped square of each case
EDIT_STEPS = 100                    # most steps of the probe guided edit
EDIT_CONFIDENCE = 0.99              # the edit stops once the probe gives the flipped square's new class this probability
EDIT_MODE = "from_middle"           # edited layers: from the middle layer (or the best, if earlier) to the last
EDIT_STEP_SIZE = 0.05               # step length per edit step, in units of sqrt(width) (standardized space)
RANDOM_EDIT_SEED = 0
N_BOOT = 2000
BOOT_SEED = 0
DECODABLE_MIN = 0.10                # selectivity at or above this = decodable (Section 3)
CAUSAL_PASS = 0.20                  # causal score at or above this, with the bootstrap interval above 0, = causal


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


def mlp_probe(d: int, hidden: int) -> nn.Module:
    """The nonlinear probe (Amendment 2, B1): one hidden layer with ReLU, -> 64 squares x 3 classes."""
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, oth.N_SQ * N_CLASSES))


def fit_probes(model, data: dict, ctrl: np.ndarray, mu: torch.Tensor, sd: torch.Tensor, epochs: int = EPOCHS, batch: int = BATCH,
               pos_batch: int = POS_BATCH, lr: float = LR, weight_decay: float = WEIGHT_DECAY, seed: int = SEED,
               mlp_hidden: int = MLP_HIDDEN) -> dict[str, nn.ModuleList | None]:
    """Train, per layer, a real and a control linear probe (A6) and a real and a control MLP probe
    (Amendment 2, B1), all from the same activations. Activations are computed `batch` games at a
    time; the probes take one optimizer step per `pos_batch` positions, so 10,000 games give about
    1,100 steps over 2 epochs. The linear probes are created first and have their own optimizer, so
    they train exactly as before the MLP probes were added (same initialisation, same steps)."""
    torch.manual_seed(seed)
    L, d, dev = model.n_layers, model.d_model, model.device
    real = nn.ModuleList([nn.Linear(d, oth.N_SQ * N_CLASSES) for _ in range(L)]).to(dev)
    control = nn.ModuleList([nn.Linear(d, oth.N_SQ * N_CLASSES) for _ in range(L)]).to(dev)
    opt = torch.optim.AdamW(list(real.parameters()) + list(control.parameters()), lr=lr, weight_decay=weight_decay)
    mlp_real = mlp_control = opt_mlp = None
    if mlp_hidden:
        mlp_real = nn.ModuleList([mlp_probe(d, mlp_hidden) for _ in range(L)]).to(dev)
        mlp_control = nn.ModuleList([mlp_probe(d, mlp_hidden) for _ in range(L)]).to(dev)
        opt_mlp = torch.optim.AdamW(list(mlp_real.parameters()) + list(mlp_control.parameters()), lr=lr,
                                    weight_decay=weight_decay)
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
                for probes, ctrls, optim in ((real, control, opt), (mlp_real, mlp_control, opt_mlp)):
                    if probes is None:
                        continue
                    loss = 0.0
                    for layer in range(L):
                        x = x_all[layer][k]
                        loss = loss + loss_fn(probes[layer](x).view(-1, N_CLASSES), y[k].reshape(-1))
                        loss = loss + loss_fn(ctrls[layer](x).view(-1, N_CLASSES), c[k].reshape(-1))
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
    return {"real": real, "control": control, "mlp_real": mlp_real, "mlp_control": mlp_control}


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
# Causal check (Amendment 2, B2)
#
# Plain words: flip one disc in the true board (mine <-> theirs) so that the legal next moves
# change. Edit the model's vector at the best layer until the linear probe reads that edited
# board, let the rest of the model run, and ask whether its predicted moves now follow the edited
# board. The same edit size in a random direction is the control. A model that only stores the
# board (decodable) but does not use it for its predictions shows no difference between the two.
# ==========================================================================

def flip_relative(rel: np.ndarray, sq: int) -> np.ndarray:
    """Mine <-> theirs on one occupied square of a mine / theirs / empty board (0 empty, 1, 2)."""
    out = rel.copy()
    out[sq] = 3 - out[sq]
    return out


def causal_cases(data: dict, seed: int = CAUSAL_SEED, max_cases: int | None = None) -> dict[str, np.ndarray]:
    """One case per game: a random position that has a recorded next move, and a random occupied
    square whose flip changes the set of legal next moves (and leaves at least one). Positions and
    squares are tried in a random order (seed), so the first that qualifies is a uniform draw.
    Returns per case: game, position, square, the edited mine / theirs board (64,), and the legal
    next move sets before and after the flip as (61,) token masks."""
    rng = np.random.default_rng(seed)
    games, positions, squares, targets, before, after = [], [], [], [], [], []
    for g in range(len(data["tokens"])):
        T = int(data["mask"][g].sum())
        for t in rng.permutation(max(T - 1, 0)):
            bw = data["black_white"][g, t]
            mover = int(data["movers"][g, t])
            orig = oth.legal_next(bw, mover)
            if not orig:
                continue
            hit = None
            for sq in rng.permutation(np.flatnonzero(bw != oth.EMPTY)):
                edited = bw.copy()
                edited[sq] = oth.other(int(edited[sq]))
                new = oth.legal_next(edited, mover)
                if new and set(new) != set(orig):
                    hit = (int(sq), new)
                    break
            if hit is None:
                continue
            sq, new = hit
            games.append(g)
            positions.append(int(t))
            squares.append(sq)
            targets.append(flip_relative(data["mine_theirs"][g, t], sq))
            mb, ma = np.zeros(oth.VOCAB, dtype=bool), np.zeros(oth.VOCAB, dtype=bool)
            mb[[oth.SQ_TO_TOKEN[s] for s in orig]] = True
            ma[[oth.SQ_TO_TOKEN[s] for s in new]] = True
            before.append(mb)
            after.append(ma)
            break
        if max_cases is not None and len(games) >= max_cases:
            break
    return {"game": np.array(games, dtype=np.int64), "pos": np.array(positions, dtype=np.int64),
            "square": np.array(squares, dtype=np.int64),
            "target": np.array(targets, dtype=np.int64).reshape(-1, oth.N_SQ),
            "legal_before": np.array(before, dtype=bool).reshape(-1, oth.VOCAB),
            "legal_after": np.array(after, dtype=bool).reshape(-1, oth.VOCAB)}


def edit_reached(logits: torch.Tensor, target: torch.Tensor, square: torch.Tensor,
                 confidence: float = EDIT_CONFIDENCE) -> torch.Tensor:
    """(N,) the probe reads the whole target board and gives the flipped square's new class at
    least `confidence` probability (a clear reading, not one on the edge of the probe's boundary)."""
    rows = torch.arange(len(target), device=target.device)
    p_new = logits[rows, square].softmax(-1)[rows, target[rows, square]]
    return (logits.argmax(-1) == target).all(1) & (p_new >= confidence)


def probe_edit(probe: nn.Module, z: torch.Tensor, target: torch.Tensor, square: torch.Tensor,
               steps: int = EDIT_STEPS, step_size: float = EDIT_STEP_SIZE,
               confidence: float = EDIT_CONFIDENCE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edit standardized vectors z (N, d) until the linear probe reads `target` (N, 64) on all 64
    squares, with at least `confidence` on the flipped `square` (N,). Each step moves a case by
    step_size * sqrt(d) along its normalized loss gradient; a case stops as soon as it is reached,
    so edits stay as small as that allows. Returns the edit (N, d), whether it was reached, and the
    steps used."""
    d = z.shape[1]
    step = step_size * d ** 0.5
    delta = torch.zeros_like(z)
    done = torch.zeros(len(z), dtype=torch.bool, device=z.device)
    used = torch.zeros(len(z), dtype=torch.long, device=z.device)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for _ in range(steps):
        x = (z + delta).detach().requires_grad_(True)
        logits = probe(x).view(len(z), oth.N_SQ, N_CLASSES)
        done = done | edit_reached(logits.detach(), target, square, confidence)
        if bool(done.all()):
            break
        loss = loss_fn(logits.reshape(-1, N_CLASSES), target.reshape(-1)).view(len(z), -1).sum(1)
        (grad,) = torch.autograd.grad(loss.sum(), x)
        unit = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        move = (~done).float()[:, None]
        delta = delta - step * unit * move
        used = used + (~done).long()
    with torch.no_grad():
        done = done | edit_reached(probe(z + delta).view(len(z), oth.N_SQ, N_CLASSES), target, square, confidence)
    return delta.detach(), done, used


def boundary_accuracy(logits: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per case: the model accepts its k best moves, k = number of legal moves after the flip
    (token 0, padding, never counts). Boundary moves are those legal before or after the flip but
    not both. A move that becomes legal is right if accepted; a move that stops being legal is
    right if not accepted. Returns the share of boundary moves that are right, per case."""
    scores = logits.copy()
    scores[:, 0] = -np.inf
    k = after.sum(1)
    order = np.argsort(-scores, axis=1)
    accepted = np.zeros_like(after)
    for i in range(len(scores)):
        accepted[i, order[i, : k[i]]] = True
    gained, lost = after & ~before, before & ~after
    right = (gained & accepted).sum(1) + (lost & ~accepted).sum(1)
    return right / np.maximum((gained | lost).sum(1), 1)


def bootstrap_mean(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def causal_label(selectivity: float, causal_score: float, ci_low: float) -> str:
    """The label H3 uses (Section 3 and Amendment 2, B2)."""
    if selectivity < DECODABLE_MIN:
        return "not decodable"
    if causal_score >= CAUSAL_PASS and ci_low > 0:
        return "decodable and causal"
    return "decodable but not causal"


def edit_layers(best: int, n_layers: int, mode: str = None) -> list[int]:
    """Which layers the causal check edits. "from_best": the best layer and every layer after it
    (Li et al. 2023 intervene from one layer to the last); "best": the best layer only;
    "from_middle": from layer n_layers // 2 (or the best layer, if earlier) to the last; "all"."""
    mode = mode or EDIT_MODE
    if mode == "best":
        return [best]
    if mode == "from_best":
        return list(range(best, n_layers))
    if mode == "from_middle":
        return list(range(min(best, n_layers // 2), n_layers))
    if mode == "all":
        return list(range(n_layers))
    raise ValueError(f"unknown edit mode {mode}")


def causal_check(model, probes: nn.ModuleList, best: int, data: dict, mu: torch.Tensor, sd: torch.Tensor,
                 selectivity: float, batch: int = BATCH, seed: int = CAUSAL_SEED, max_cases: int | None = None,
                 mode: str | None = None, confidence: float | None = None) -> dict[str, Any]:
    """Probe guided edit vs random edit of the same size, on one case per game. The edit runs in
    the forward pass at every layer of edit_layers(best): each layer's own linear probe guides its
    edit, and later layers see the earlier edits. The random edit has, at every layer, the same
    length as the probe edit there."""
    mode = mode or EDIT_MODE
    confidence = EDIT_CONFIDENCE if confidence is None else confidence
    layers = edit_layers(best, model.n_layers, mode)
    cases = causal_cases(data, seed, max_cases)
    n = len(cases["game"])
    if n == 0:
        raise ValueError("no causal check cases: no position where one flip changes the legal moves")
    dev = model.device
    acc = {"edit": [], "random": [], "none": []}
    reached = {layer: [] for layer in layers}
    flip_read = {layer: [] for layer in layers}
    steps_used = {layer: [] for layer in layers}
    sizes = {layer: [] for layer in layers}
    for b0 in range(0, n, batch):
        sl = slice(b0, b0 + batch)
        g, p = cases["game"][sl], cases["pos"][sl]
        tok = torch.as_tensor(data["tokens"][g], device=dev)
        pos = torch.as_tensor(p, device=dev)
        target = torch.as_tensor(cases["target"][sl], device=dev)
        sq = torch.as_tensor(cases["square"][sl], device=dev)
        rows = torch.arange(len(g), device=dev)
        norm_at: dict[int, torch.Tensor] = {}

        def probe_step(layer):
            def fn(v: torch.Tensor) -> torch.Tensor:
                z = (v - mu[layer]) / sd[layer]
                with torch.enable_grad():
                    delta, ok, used = probe_edit(probes[layer], z, target, sq, confidence=confidence)
                with torch.no_grad():
                    read = probes[layer](z + delta).view(len(z), oth.N_SQ, N_CLASSES).argmax(-1)
                norm_at[layer] = delta.norm(dim=1)
                reached[layer].append(ok.cpu().numpy())
                flip_read[layer].append((read[rows, sq] == target[rows, sq]).cpu().numpy())
                steps_used[layer].append(used.cpu().numpy())
                sizes[layer].append(norm_at[layer].cpu().numpy())
                return (z + delta) * sd[layer] + mu[layer]
            return fn

        def random_step(layer):
            def fn(v: torch.Tensor) -> torch.Tensor:
                z = (v - mu[layer]) / sd[layer]
                gen = torch.Generator(device="cpu").manual_seed(RANDOM_EDIT_SEED + 1000 * layer + b0)
                r = torch.randn(z.shape, generator=gen).to(dev)
                r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12) * norm_at[layer][:, None]
                return (z + r) * sd[layer] + mu[layer]
            return fn

        runs = {"edit": {layer: probe_step(layer) for layer in layers},
                "random": {layer: random_step(layer) for layer in layers},   # needs the edit sizes: runs second
                "none": {}}
        for name, edits in runs.items():
            logits = model.logits_edited(tok, pos, edits)[rows, pos].float().cpu().numpy()
            acc[name].append(boundary_accuracy(logits, cases["legal_before"][sl], cases["legal_after"][sl]))
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    diff = acc["edit"] - acc["random"]
    lo, hi = bootstrap_mean(diff)
    cs = float(diff.mean())
    cat = lambda d: {str(k): round(float(np.concatenate(v).mean()), 5) for k, v in d.items()}   # noqa: E731
    reached_all = np.all(np.stack([np.concatenate(reached[layer]) for layer in layers]), axis=0)
    return {
        "causal_score": round(cs, 5), "causal_score_ci": [round(lo, 5), round(hi, 5)],
        "causal_label": causal_label(selectivity, cs, lo),
        "boundary_acc_edit": round(float(acc["edit"].mean()), 5),
        "boundary_acc_random": round(float(acc["random"].mean()), 5),
        "boundary_acc_none": round(float(acc["none"].mean()), 5),
        "edit_success_rate": round(float(reached_all.mean()), 5),
        "edit_success_by_layer": cat(reached), "flip_read_by_layer": cat(flip_read),
        "edit_steps_by_layer": cat(steps_used), "edit_size_by_layer": cat(sizes),
        "edit_size_unit": "standardized activation units (L2 norm)",
        "mode": mode, "confidence": confidence, "layers": layers, "best_layer": best, "n_cases": n,
        "mean_boundary_moves": round(float(((cases["legal_before"] ^ cases["legal_after"]).sum(1)).mean()), 3),
    }


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
    n_choose, epochs, batch, pos_batch, mlp_hidden (0 = no MLP probe), causal (False = no causal
    check) and causal_cases_max for tests only."""
    t0 = time.time()
    n_fit, n_choose = config.pop("n_fit", N_FIT), config.pop("n_choose", N_CHOOSE)
    causal, causal_cases_max = config.pop("causal", True), config.pop("causal_cases_max", None)
    causal_mode, causal_confidence = config.pop("causal_mode", None), config.pop("causal_confidence", None)
    dist = training_distribution(model)
    fit, choose, test = load_data(dist, split, n_games, data_seed, n_fit, n_choose)

    freq = class_freq(fit["mine_theirs"], fit["mask"])
    ctrl = control_table(freq, SEED)
    batch = config.get("batch", BATCH)
    mu, sd = activation_stats(model, fit, batch)
    probes = fit_probes(model, fit, ctrl, mu, sd, **config)
    real, control = probes["real"], probes["control"]
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
    if probes["mlp_real"] is not None:       # nonlinear probe (Amendment 2, B1)
        mch = evaluate_probes(model, probes["mlp_real"], probes["mlp_control"], choose, ctrl, mu, sd, batch)
        mte = evaluate_probes(model, probes["mlp_real"], probes["mlp_control"], test, ctrl, mu, sd, batch)
        msel_choose = mch["probe_acc"] - mch["control_acc"]
        msel_test = mte["probe_acc"] - mte["control_acc"]
        mbest = int(np.argmax(msel_choose))
        details["mlp"] = {
            "hidden": config.get("mlp_hidden", MLP_HIDDEN), "best_layer": mbest,
            "selectivity_at_best": round(float(msel_test[mbest]), 5),
            "probe_acc_by_layer": r(mte["probe_acc"]), "control_acc_by_layer": r(mte["control_acc"]),
            "selectivity_by_layer": r(msel_test), "selectivity_by_layer_choose": r(msel_choose),
            "probe_acc_occupied_by_layer": r(mte["probe_acc_occupied"]),
            "gain_over_linear_by_layer": r(mte["probe_acc"] - te["probe_acc"]),
        }
    if causal:                               # causal check at the linear best layer (Amendment 2, B2)
        details["causal"] = causal_check(model, real, best, test, mu, sd, float(sel_test[best]), batch,
                                         max_cases=causal_cases_max, mode=causal_mode, confidence=causal_confidence)
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
            if "mlp" in d:
                mb = d["mlp"]["best_layer"]
                print(f"{'':45s} mlp: selectivity {d['mlp']['selectivity_at_best']:.4f}  best layer {mb}  "
                      f"probe {d['mlp']['probe_acc_by_layer'][mb]:.4f}  control {d['mlp']['control_acc_by_layer'][mb]:.4f}")
            if "causal" in d:
                c = d["causal"]
                print(f"{'':45s} causal ({c['mode']}, layers {c['layers']}, confidence {c['confidence']}): "
                      f"score {c['causal_score']:.4f} {c['causal_score_ci']}  {c['causal_label']}  "
                      f"edit {c['boundary_acc_edit']:.4f}  random {c['boundary_acc_random']:.4f}  "
                      f"none {c['boundary_acc_none']:.4f}  edit success {c['edit_success_rate']:.4f}  "
                      f"cases {c['n_cases']}  {res.provenance.get('runtime_sec', '?')} s")
        else:
            print(f"{res.model_id:45s} {res.status}: {res.reason}")


if __name__ == "__main__":
    main()
