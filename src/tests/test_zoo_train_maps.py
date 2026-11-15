"""Tests for the navigation side of the zoo trainer (N3): src/zoo/train.py with --domain maps.

The trainer is shared with Othello, so these also check that every Othello model is built exactly as before
(the same parameter shapes and counts; the PR records that 30 steps of the old and new trainer give identical
weights for all three architectures). Navigation rules: reports/preregistration.md, Amendment 5, G2 and G3.

Run from the repo root:
    python -m pytest src/tests/test_zoo_train_maps.py -q
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
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


train = _load("zoo_train", "src/zoo/train.py")
maps = train.domain_module("maps")
ARCHS = ["transformer", "lstm", "mamba"]


# ==========================================================================
# Othello is unchanged
# ==========================================================================

def test_othello_models_are_built_as_before() -> None:
    counts = {"transformer": 816573, "lstm": 544061, "mamba": 482365}   # small scale, before N3
    for arch in ARCHS:
        m = train.build_model(arch, "small")
        assert sum(p.numel() for p in m.parameters()) == counts[arch]
        assert m.tok.num_embeddings == 61 and m.head.out_features == 61
        assert train.build_model(arch, "small", "othello").state_dict().keys() == m.state_dict().keys()
    assert train.build_model("transformer", "small").pos.num_embeddings == 59
    assert train.domain_sizes("othello") == (61, 59)


def test_othello_build_is_seed_identical_with_or_without_the_domain_argument() -> None:
    for arch in ARCHS:
        torch.manual_seed(0)
        a = train.build_model(arch, "small")
        torch.manual_seed(0)
        b = train.build_model(arch, "small", "othello")
        assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


# ==========================================================================
# Navigation models
# ==========================================================================

@pytest.mark.parametrize("arch", ARCHS)
def test_maps_models_have_the_navigation_sizes(arch: str) -> None:
    m = train.build_model(arch, "small", "maps")
    assert train.domain_sizes("maps") == (maps.VOCAB, maps.N_CTX) == (4589, 98)
    assert m.tok.num_embeddings == maps.VOCAB and m.head.out_features == maps.VOCAB
    if arch == "transformer":
        assert m.pos.num_embeddings == maps.N_CTX
    x = torch.randint(1, maps.VOCAB, (2, maps.N_CTX))
    with torch.no_grad():
        assert m(x).shape == (2, maps.N_CTX, maps.VOCAB)
        assert len(m.layer_outputs(x)) == 4


def test_synthetic_routes_are_legal() -> None:
    routes = train.maps_synthetic_routes(200, seed=0)
    assert routes.shape == (200, maps.MAX_TOKENS)
    assert maps.build(routes)["n_illegal"] == 0


def _cfg(**kw) -> dict:
    cfg = {"device": "cpu", "lr": 3e-4, "weight_decay": 0.01, "seed": 0, "micro_batch": 0, "batch": 8,
           "steps": 1, "warmup_steps": 1, "budget_steps": 1, "log_every": 1000, "domain": "maps"}
    cfg.update(kw)
    return cfg


def test_training_loss_covers_only_turns_and_end() -> None:
    """With learning rate 0 the reported loss is the loss of the first batch; it must equal cross entropy over
    the scored positions only (no loss on the destination, none on padding)."""
    routes = train.maps_synthetic_routes(50, seed=1)
    torch.manual_seed(0)
    model = train.build_model("transformer", "small", "maps")
    ref = train.build_model("transformer", "small", "maps")
    ref.load_state_dict(model.state_dict())
    cfg = _cfg(lr=0.0, weight_decay=0.0)
    out = train.train(model, routes, cfg)
    idx = np.random.default_rng(cfg["seed"]).integers(0, len(routes), size=cfg["batch"])
    x = torch.tensor(routes[idx])
    d = maps.build(routes[idx])
    with torch.no_grad():
        logits = ref(x[:, :maps.N_CTX])
    sc = torch.tensor(d["score"][:, :maps.N_CTX])
    manual = nn.functional.cross_entropy(logits[sc], x[:, 1:][sc])
    assert out["final_loss"] == pytest.approx(float(manual), rel=1e-5)
    # the destination (target of position 0) is never a target
    assert not d["score"][:, 0].any()


def test_micro_batching_gives_the_full_batch_gradient(monkeypatch) -> None:
    """Routes differ in length, so chunks hold different numbers of scored tokens; one step with chunks of 3
    must give the same weights as one step on the whole batch of 8 (gradient accumulation is exact). Plain SGD
    stands in for AdamW here: Adam's first step divides each gradient by its own size, which turns float noise
    in near zero gradients into full steps and would hide what is being checked."""
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)
    routes = train.maps_synthetic_routes(40, seed=5)
    states = []
    for mb in (0, 3):
        torch.manual_seed(0)
        model = train.build_model("lstm", "small", "maps")
        train.train(model, routes, _cfg(steps=1, batch=8, micro_batch=mb, lr=1e-2, warmup_steps=0))
        states.append(model.state_dict())
    for k in states[0]:
        assert torch.allclose(states[0][k], states[1][k], atol=1e-6), k


def test_training_reads_the_uint16_cache_and_learns() -> None:
    routes = train.maps_synthetic_routes(400, seed=2).astype(np.uint16)
    torch.manual_seed(0)
    model = train.build_model("lstm", "small", "maps")
    cfg = _cfg(lr=3e-3, steps=60, batch=16, warmup_steps=5, budget_steps=60)
    out = train.train(model, routes, cfg)
    assert np.isfinite(out["final_loss"]) and out["final_loss"] < np.log(maps.VOCAB)


def test_evaluate_maps_is_the_d1_legal_rate() -> None:
    routes = train.maps_synthetic_routes(60, seed=3)
    torch.manual_seed(0)
    model = train.build_model("transformer", "small", "maps")
    ev = train.evaluate(model, routes, _cfg(batch=16))
    data = maps.build(routes)
    assert ev["legal_move_rate"] == pytest.approx(maps.legal_rate(model, data, "cpu", 16))
    assert ev["n_eval_games"] == 60 and ev["illegal_games_dropped"] == 0
    assert 0.0 <= ev["next_token_acc"] <= 1.0


# ==========================================================================
# Gates and records (Amendment 5)
# ==========================================================================

def test_zoo_and_pilot_models() -> None:
    assert train.in_zoo("maps_transformer_small_shortest_paths_s0")
    assert not train.in_zoo("maps_transformer_small_shortest_paths_s100")
    assert train.record_dir("maps", "maps_transformer_small_shortest_paths_s0").name == "zoo_training"
    assert train.record_dir("maps", "maps_mamba_small_random_walks_s100").name == "pilot_training"
    assert train.record_dir("othello", "othello_lstm_small_mixed_s0").name == "zoo_training"


def test_real_data_loading_follows_amendment5(monkeypatch) -> None:
    """Full training cache, 1,000 validation routes drawn with data seed 0, and a refusal when a file
    does not match the manifest (Amendment 5: admission uses 1,000 val routes, data seed 0)."""
    calls = []
    def fake_load(name, split, n_routes=None, seed=0, data_dir=None):
        calls.append((name, split, n_routes, seed))
        return np.zeros((n_routes or 5, maps.MAX_TOKENS), dtype=np.uint16)
    ok = {"noisy_shortest_paths_train.npy": True, "noisy_shortest_paths_val.npy": True, "random_walks_val.npy": False}
    monkeypatch.setattr(maps, "load_distribution", fake_load)
    monkeypatch.setattr(maps, "verify_data", lambda d=None: dict(ok))
    cfg = {"synthetic": 0, "distribution": "noisy_shortest_paths", "maps_data": "x", "n_val_games": 1000, "seed": 3}
    train.load_maps_data(cfg)
    assert calls == [("noisy_shortest_paths", "train", None, 0), ("noisy_shortest_paths", "val", 1000, 0)]
    ok["noisy_shortest_paths_val.npy"] = False
    with pytest.raises(SystemExit, match="do not match"):
        train.load_maps_data(cfg)


def test_gates(monkeypatch, tmp_path: Path) -> None:
    calls = {"status": "", "tag": ""}
    monkeypatch.setattr(train, "_git", lambda *a: calls["status"] if a[0] == "status" else calls["tag"])
    monkeypatch.setattr(train, "PILOT_RECORDS", tmp_path)
    zoo, pilot = "maps_transformer_small_shortest_paths_s0", "maps_transformer_small_shortest_paths_s100"
    assert len(set(train.pilot_ids())) == 9 and pilot in train.pilot_ids()
    assert not any(train.in_zoo(p) for p in train.pilot_ids())
    with pytest.raises(SystemExit, match="prereg-v1.5"):
        train.maps_gates(zoo, full_budget=True, synthetic=False)        # no tag yet
    train.maps_gates(pilot, full_budget=True, synthetic=False)          # pilots need no tag
    train.maps_gates(zoo, full_budget=False, synthetic=False)           # throughput runs are free
    train.maps_gates(zoo, full_budget=True, synthetic=True)             # smoke tests are free
    with pytest.raises(SystemExit, match="zoo model .* or one of the 9 pilots"):
        train.maps_gates("maps_transformer_small_shortest_paths_s7", full_budget=True, synthetic=False)
    with pytest.raises(SystemExit, match="zoo model .* or one of the 9 pilots"):
        train.maps_gates("maps_transformer_medium_shortest_paths_s100", full_budget=True, synthetic=False)
    calls["tag"] = "prereg-v1.5\n"
    for p in train.pilot_ids()[:-1]:
        (tmp_path / f"{p}.json").write_text("{}")
    with pytest.raises(SystemExit, match="pilots before any zoo model"):
        train.maps_gates(zoo, full_budget=True, synthetic=False)        # 8 of 9 pilots: still refused
    (tmp_path / f"{train.pilot_ids()[-1]}.json").write_text("{}")
    train.maps_gates(zoo, full_budget=True, synthetic=False)            # tag and all 9 pilots: allowed
    calls["status"] = " M src/zoo/train.py\n"
    with pytest.raises(SystemExit, match="not committed"):
        train.maps_gates(pilot, full_budget=True, synthetic=False)      # never from uncommitted code


# ==========================================================================
# Loading a navigation checkpoint for the diagnostics
# ==========================================================================

def test_checkpoint_loading_and_d3_random_baseline(tmp_path: Path) -> None:
    itf = _load("interface", "src/diagnostics/interface.py")
    mid = "maps_lstm_small_noisy_shortest_paths_s1"
    torch.manual_seed(0)
    m = train.build_model("lstm", "small", "maps")
    (tmp_path / mid).mkdir()
    torch.save(m.state_dict(), tmp_path / mid / "model.pt")
    zm = itf.from_checkpoint(mid, "cpu", tmp_path)
    x = torch.tensor(train.maps_synthetic_routes(3, seed=4)[:, :maps.N_CTX])
    assert zm.logits(x).shape == (3, maps.N_CTX, maps.VOCAB)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(zm.logits(x), m(x), atol=1e-6)   # same weights; only float rounding differs
    d3 = _load("d3_inductive_bias", "src/diagnostics/d3/inductive_bias.py")
    fresh = d3.random_like(zm, seed=0)
    assert fresh.head.out_features == maps.VOCAB and fresh.tok.num_embeddings == maps.VOCAB


# ==========================================================================
# The command line
# ==========================================================================

def test_command_line_smoke(tmp_path: Path) -> None:
    cmd = [sys.executable, str(ROOT / "src/zoo/train.py"), "--domain", "maps", "--distribution", "shortest_paths",
           "--synthetic", "60", "--steps", "3", "--batch", "4", "--device", "cpu", "--out_dir", str(tmp_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "maps_transformer_small_shortest_paths_s0" / "results.json").read_text())
    assert res["config"]["domain"] == "maps" and "legal_move_rate" in res["eval"]
    bad = subprocess.run([sys.executable, str(ROOT / "src/zoo/train.py"), "--domain", "maps", "--synthetic", "10",
                          "--steps", "1", "--device", "cpu", "--out_dir", str(tmp_path)],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert bad.returncode != 0 and "must be one of" in bad.stderr   # the Othello default is not a maps distribution
