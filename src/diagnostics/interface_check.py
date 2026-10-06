"""End to end check of the common interface on real models (EC2 only, needs the data and weights).

For each model: load it through the interface, check logits and activations shapes, and compute
the legal move rate on 1,000 test games of its own distribution. Nothing is saved: this only
proves the plumbing works. Expected: Othello GPT about 0.999, our large transformer about 0.97.

    python src/diagnostics/interface_check.py
    python src/diagnostics/interface_check.py --zoo othello_transformer_large_synthetic_s0 othello_transformer_small_synthetic_s0
"""

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("interface", ROOT / "src/diagnostics/interface.py")
itf = importlib.util.module_from_spec(_spec)
sys.modules["interface"] = itf
_spec.loader.exec_module(itf)
oth = itf.oth

log = logging.getLogger("interface_check")


def check(model: "itf.ZooModel", distribution: str, n_games: int, batch: int) -> dict:
    t0 = time.time()
    games = oth.load_distribution(distribution, "test", n_games, seed=0)
    data = oth.build(games, legal=True)
    tok = torch.as_tensor(data["tokens"][:4])
    lg = model.logits(tok)
    acts = model.activations(tok)
    assert lg.shape == (4, oth.N_CTX, oth.VOCAB), lg.shape
    assert acts.shape == (model.n_layers, 4, oth.N_CTX, model.d_model), acts.shape
    rate = oth.legal_rate(model, data, model.device, batch)
    return {"model": model.model_id, "logits": tuple(lg.shape), "activations": tuple(acts.shape),
            "legal_rate": round(rate, 4), "games": len(data["tokens"]), "sec": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zoo", nargs="*", default=["othello_transformer_large_synthetic_s0"])
    p.add_argument("--n_games", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--no_gpt", action="store_true", help="skip the reference Othello GPT")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = []
    if not a.no_gpt:
        rows.append(check(itf.othello_gpt("synthetic"), "synthetic", a.n_games, a.batch))
    for model_id in a.zoo:
        model = itf.from_checkpoint(model_id)
        dist = itf.results.load_models()[model_id]["distribution"]
        rows.append(check(model, dist, a.n_games, a.batch))

    log.info("\n| model | logits | activations | legal move rate (test) | games | seconds |")
    log.info("|---|---|---|---|---|---|")
    for r in rows:
        log.info("| %s | %s | %s | %.4f | %d | %.1f |", r["model"], r["logits"], r["activations"],
                 r["legal_rate"], r["games"], r["sec"])


if __name__ == "__main__":
    main()
