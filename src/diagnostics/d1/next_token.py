"""Diagnostic 1: next token validity.

Does the model only propose legal moves? For every position with a recorded next move, the true
board gives the set of legal moves, and we check the model's proposal against it.
Protocol: src/docs/d1_next_token_validity.md (PR #62), frozen by reports/preregistration.md A6.

    score      share of positions where the model's top predicted move is legal (argmax),
               computed by othello.legal_rate
    secondary  at each position, 10 moves sampled at temperature 1.0 (sampling seed 0);
               share of sampled moves that are legal (stored in details)
    data       1,000 test games of the model's own training distribution, data seed 0

Run on the EC2 (repo root, conda env wmconcord):
    python src/diagnostics/d1/next_token.py --models othello_gpt_synthetic othello_gpt_championship
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

log = logging.getLogger("d1")

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

DIAGNOSTIC = "D1"
METHOD = "legal move rate: share of positions where the top predicted move is legal (argmax), higher is better"

# Protocol constants: fixed by src/docs/d1_next_token_validity.md and preregistration A6.
N_SAMPLES = 10
TEMPERATURE = 1.0
SAMPLE_SEED = 0
BATCH = 256
PREREG_TAG = "prereg-v1.1"


@torch.no_grad()
def sampled_legal_rate(model, data: dict, n_samples: int = N_SAMPLES, temperature: float = TEMPERATURE,
                       seed: int = SAMPLE_SEED, batch: int = BATCH) -> float:
    """Share of sampled moves that are legal, over the same positions as othello.legal_rate
    (every position with a recorded next move; the last move of a game is skipped)."""
    gen = torch.Generator(device=model.device).manual_seed(seed)
    ok = total = 0
    for i in range(0, len(data["tokens"]), batch):
        logits = model.logits(torch.as_tensor(data["tokens"][i:i + batch])).float()   # (B, T, 61)
        probs = torch.softmax(logits / temperature, dim=-1)
        B, T, V = probs.shape
        draws = torch.multinomial(probs.reshape(-1, V), n_samples, replacement=True, generator=gen)
        draws = draws.reshape(B, T, n_samples).cpu().numpy()
        for b in range(B):
            g = i + b
            t = np.arange(int(data["mask"][g].sum()) - 1)
            legal = data["legal_next"][g, t]                       # (len(t), 61)
            picks = draws[b, t]                                    # (len(t), n_samples)
            ok += int(np.take_along_axis(legal, picks, axis=1).sum())
            total += picks.size
    return ok / max(total, 1)


def training_distribution(model) -> str:
    """The distribution the model was trained on: models.csv for zoo models, meta for anchors."""
    row = results.load_models().get(model.model_id)
    return row["distribution"] if row else model.meta["distribution"]


def score(model, split: str = "test", n_games: int = 1000, data_seed: int = 0, batch: int = BATCH,
          **config: Any):
    """The diagnostic contract (src/docs/diagnostic_interface.md)."""
    t0 = time.time()
    dist = training_distribution(model)
    data = oth.build(oth.load_distribution(dist, split, n_games, seed=data_seed), legal=True)
    argmax_rate = oth.legal_rate(model, data, model.device, batch)
    sampled = sampled_legal_rate(model, data, batch=batch)
    n_positions = int((data["mask"].sum(1) - 1).clip(min=0).sum())
    details = {"argmax_legal_rate": argmax_rate, "sampled_legal_rate": sampled,
               "n_samples": N_SAMPLES, "temperature": TEMPERATURE, "sample_seed": SAMPLE_SEED,
               "n_positions": n_positions, "distribution": dist,
               "illegal_games_dropped": data["n_illegal_games"]}
    return itf.result_for(model, DIAGNOSTIC, method=METHOD, split=split, n_games=len(data["tokens"]),
                          data_seed=data_seed, score=float(argmax_rate), details=details,
                          runtime_sec=round(time.time() - t0, 1))


def prereg_tag_exists() -> bool:
    out = subprocess.run(["git", "tag", "-l", PREREG_TAG], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() == PREREG_TAG


def load_any(model_id: str):
    """othello_gpt_synthetic / othello_gpt_championship are the anchors; anything else is a zoo id."""
    if model_id.startswith("othello_gpt_"):
        return itf.othello_gpt(model_id.removeprefix("othello_gpt_"))
    return itf.from_checkpoint(model_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic 1, next token validity")
    ap.add_argument("--models", nargs="+", required=True, help="othello_gpt_synthetic, othello_gpt_championship or zoo model ids")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_games", type=int, default=1000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write results/D1_next_token/{model_id}.json")
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
        if res.status == "ok":
            print(f"{res.model_id:45s} score {res.score:.4f}  sampled {res.details['sampled_legal_rate']:.4f}  "
                  f"positions {res.details['n_positions']}")
        else:
            print(f"{res.model_id:45s} {res.status}: {res.reason}")


if __name__ == "__main__":
    main()
