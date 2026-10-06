"""Tests for the zoo registry (src/zoo/registry.py).

They train nothing: a tiny real checkpoint is saved with random weights and a training record
is written by hand, so they run anywhere.

Run from the repo root:
    python -m pytest src/tests/test_registry.py -q
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


reg = _load("registry", "src/zoo/registry.py")
train = _load("zoo_train", "src/zoo/train.py")
itf = _load("interface", "src/diagnostics/interface.py")

MID = "othello_lstm_small_synthetic_s2"


def fake_record(tmp: Path, model_id: str = MID, arch: str = "lstm", legal: float = 0.81) -> Path:
    """A small scale checkpoint with random weights and its training record, as train.py writes them."""
    torch.manual_seed(0)
    module = train.build_model(arch, "small")
    ckpt = tmp / "outputs" / "zoo" / model_id / "model.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save(module.state_dict(), ckpt)
    rec = {"model_id": model_id, "arch": arch, "scale": "small", "n_params": 1,
           "train": {"steps_run": 5000, "final_loss": 2.7, "sec_per_step": 0.01, "wall_sec": 60.0},
           "eval": {"legal_move_rate": legal, "next_token_acc": 0.1},
           "config": {"distribution": "synthetic", "seed": 2, "synthetic": 0},
           "domain": "othello", "distribution": "synthetic", "seed": 2,
           "checkpoint": str(ckpt), "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
           "gpu": "cpu", "host": "test", "git_commit": "abc", "time_utc": "2026-10-06T00:00:00Z"}
    rdir = tmp / "records"
    rdir.mkdir(exist_ok=True)
    (rdir / f"{model_id}.json").write_text(json.dumps(rec))
    return rdir


def test_registry_rows(tmp_path: Path) -> None:
    rdir = fake_record(tmp_path)
    rows = reg.build_registry(rdir)
    assert len(rows) == 1 and list(rows[0]) == reg.COLUMNS
    r = rows[0]
    assert (r["model_id"], r["arch"], r["val_legal_rate"], r["train_minutes"]) == (MID, "lstm", 0.81, 1.0)
    assert r["admitted"] == "tbd"


def test_admission_threshold() -> None:
    assert reg.admitted(0.81, None) == "tbd"
    assert reg.admitted(0.81, 0.5) == "yes" and reg.admitted(0.3, 0.5) == "no"


def test_unknown_model_rejected(tmp_path: Path) -> None:
    rdir = fake_record(tmp_path, model_id="othello_lstm_huge_synthetic_s0")
    with pytest.raises(ValueError, match="not in models.csv"):
        reg.build_registry(rdir)


def test_write_registry_csv(tmp_path: Path) -> None:
    rdir = fake_record(tmp_path)
    reg.write_registry(tmp_path / "zoo_registry.csv", records_dir=rdir)
    lines = (tmp_path / "zoo_registry.csv").read_text().splitlines()
    assert lines[0] == ",".join(reg.COLUMNS) and lines[1].startswith(MID)


def test_load_model_round_trip(tmp_path: Path) -> None:
    rdir = fake_record(tmp_path)
    m = reg.load_model(MID, device="cpu", records_dir=rdir)
    assert isinstance(m, itf.ZooModel) and m.arch == "lstm" and m.n_layers == 4
    tok = torch.ones(2, 59, dtype=torch.long)
    assert m.activations(tok).shape == (4, 2, 59, 128)


def test_load_model_detects_changed_checkpoint(tmp_path: Path) -> None:
    rdir = fake_record(tmp_path)
    ckpt = tmp_path / "outputs" / "zoo" / MID / "model.pt"
    torch.save({"changed": torch.zeros(1)}, ckpt)
    with pytest.raises(ValueError, match="sha256"):
        reg.load_model(MID, device="cpu", records_dir=rdir)


def test_load_model_not_trained(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not trained"):
        reg.load_model(MID, records_dir=tmp_path)
