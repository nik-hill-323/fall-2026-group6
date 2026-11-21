"""Shared test helpers for the navigation scorers (N0): a small fake data folder and models whose answers are known.

Not a test file itself (no test_ prefix). The tests of D1 to D4, the fragility suite and the scoring foundation
import it, so every navigation scorer is checked on the same fakes, with no data and no checkpoints:

    fake_data_dir(tmp)    {distribution}_{split}.npy for all three distributions and splits, built from random
                          legal routes on the real canonical map (half random walks, half fewest turn routes)
    use_fake_data(...)    points maps.DATA_DIR at that folder and turns off the manifest check
    MapsOracle            a ZooModel whose logits put all weight on the legal next tokens (or on one fixed
                          token), and whose layers hold (0) the token just read and (1) the current
                          intersection, as fixed random vectors: a model with a world model in its last layer
    tiny(arch)            an untrained src/zoo/train.py model (small scale) wrapped as a zoo model

Every fake uses a real navigation zoo id from src/docs/models.csv (ZOO_ID), so results fill their fields from it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nav = _load("navigation", "src/diagnostics/navigation.py")
itf, maps = nav.itf, nav.maps
dom = _load("test_maps_domain", "src/tests/test_maps_domain.py")
train = _load("zoo_train", "src/zoo/train.py")

ZOO_ID = "maps_transformer_small_shortest_paths_s0"      # a real models.csv row (shortest paths)
PILOT_ID = "maps_transformer_small_shortest_paths_s100"   # a real pilot id
SPLIT_SEEDS = {"train": 1, "val": 2, "test": 3}


def fake_data_dir(tmp: Path, n: int = 400) -> Path:
    """A data folder with n routes per distribution and split (different routes in each file)."""
    tmp.mkdir(parents=True, exist_ok=True)
    for k, dist in enumerate(maps.DISTRIBUTIONS):
        for split, s in SPLIT_SEEDS.items():
            arr = maps.pad_routes(dom.random_routes(n, seed=100 * k + s)).astype(np.uint16)
            np.save(tmp / f"{dist}_{split}.npy", arr)
    return tmp


def use_fake_data(monkeypatch, tmp: Path, n: int = 400) -> Path:
    path = fake_data_dir(tmp, n)
    monkeypatch.setattr(maps, "DATA_DIR", str(path))
    monkeypatch.setattr(nav, "VERIFY_DATA", False)
    return path


def legal_logits(tokens: torch.Tensor, high: float = 10.0, low: float = -10.0) -> torch.Tensor:
    """(B, T, VOCAB): `high` on every legal next token after each prefix, `low` elsewhere. Position 0 (only the
    start read) and padding put their weight on padding, which is never scored."""
    tok = tokens.cpu().numpy()
    cur = nav.current_intersections(tok)
    dest = tok[:, 1:2] if tok.shape[1] > 1 else np.zeros((len(tok), 1), dtype=np.int64)
    out = np.full((*tok.shape, maps.VOCAB), low, dtype=np.float32)
    live = cur > 0
    turns = (maps.NEXT[cur] >= 0) & live[..., None]                       # (B, T, 8)
    out[..., maps.TURN0:maps.TURN0 + len(maps.TURNS)] = np.where(turns, high, low)
    out[..., maps.END] = np.where(live & (cur == dest), high, low)
    out[..., maps.PAD] = np.where(live, low, high)
    return torch.as_tensor(out)


class MapsOracle(itf.ZooModel):
    """kind 'legal': all weight on the legal next tokens (D1 = 1, every top k set is the legal set).
    kind 'fixed': all weight on one token `fixed`. kind 'noise': random logits (seeded per call by the input).
    Layer 0 holds a fixed random vector of the token just read, layer 1 one of the current intersection."""

    def __init__(self, kind: str = "legal", fixed: int = maps.END, model_id: str = ZOO_ID, d_model: int = 32,
                 seed: int = 0, meta: dict | None = None) -> None:
        super().__init__(nn.Identity(), model_id, "transformer", 2, d_model, "cpu", meta)
        g = torch.Generator().manual_seed(seed)
        self.tok_emb = torch.randn(maps.VOCAB, d_model, generator=g)
        self.cur_emb = torch.randn(maps.N_NODES + 1, d_model, generator=g)
        self.kind, self.fixed, self.seed = kind, fixed, seed

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.kind == "legal":
            return legal_logits(tokens)
        if self.kind == "fixed":
            out = torch.full((*tokens.shape, maps.VOCAB), -10.0)
            out[..., self.fixed] = 10.0
            return out
        g = torch.Generator().manual_seed(int(tokens.sum()) + self.seed)
        return torch.randn((*tokens.shape, maps.VOCAB), generator=g)

    def _activations(self, tokens: torch.Tensor) -> torch.Tensor:
        cur = torch.as_tensor(nav.current_intersections(tokens.cpu().numpy()))
        return torch.stack([self.tok_emb[tokens.cpu()], self.cur_emb[cur]])


def tiny(arch: str = "transformer", model_id: str = ZOO_ID, seed: int = 0) -> "itf.ZooModel":
    """An untrained small zoo model of `arch` (src/zoo/train.py), on the CPU."""
    torch.manual_seed(seed)
    module = train.build_model(arch, "small", "maps")
    cfg = train.matrix.SCALE_CONFIG["small"]
    return itf.TrainPyModel(module, model_id, arch, cfg["n_layers"], cfg["d_model"], "cpu")
