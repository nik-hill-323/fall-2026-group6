"""Tests for the three Othello training distributions (synthetic, championship, mixed) in
src/domains/othello.py.

The WTHOR reader and the distribution logic are tested on small hand made files, so they
run everywhere (including GitHub Actions). The test on the real WTHOR archive is skipped
when the files are not there (they only exist on the EC2).

Run from the repo root:
    python -m pytest src/tests/test_othello_distributions.py -q
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("othello", ROOT / "src/domains/othello.py")
oth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oth)

HAVE_WTHOR = bool(glob.glob(os.path.join(oth.CHAMPIONSHIP_DIR, "WTH_*.wtb")))


# ==========================================================================
# Helpers: write tiny data files in the real formats
# ==========================================================================

def write_wtb(path: Path, games: np.ndarray) -> None:
    """Write (G, 60) int8 squares as a WTHOR file: 16 byte header, 68 bytes per game."""
    header = np.zeros(16, dtype=np.uint8)
    header[4:8] = np.frombuffer(np.int32(len(games)).tobytes(), dtype=np.uint8)
    rec = np.zeros((len(games), 68), dtype=np.uint8)
    sq = games.astype(np.int16)
    rec[:, 8:] = np.where(sq >= 0, (sq // 8 + 1) * 10 + (sq % 8 + 1), 0)
    path.write_bytes(header.tobytes() + rec.tobytes())


def write_bin(folder: Path, split: str, games: np.ndarray, n_files: int = 1) -> None:
    """Write games into n_files .bin files named games_0.bin, games_1.bin, ..."""
    (folder / split).mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(np.array_split(games, n_files)):
        part.astype(np.int8).tofile(folder / split / f"games_{i}.bin")


@pytest.fixture()
def dirs(tmp_path: Path) -> dict[str, str]:
    """A small synthetic folder (300 train; 120 games in 12 val files, so 2 val files and
    10 test files) and a small championship folder."""
    syn = tmp_path / "synthetic"
    write_bin(syn, "train", oth.random_games(300, seed=10))
    write_bin(syn, "val", oth.random_games(120, seed=11), n_files=12)
    ch = tmp_path / "championship"
    ch.mkdir()
    games = oth.random_games(200, seed=12)
    write_wtb(ch / "WTH_2001.wtb", games[:120])
    write_wtb(ch / "WTH_2002.wtb", np.concatenate([games[120:], games[:5]]))  # 5 duplicates
    return {"synthetic_dir": str(syn), "championship_dir": str(ch)}


# ==========================================================================
# WTHOR reader
# ==========================================================================

def test_read_wtb_round_trip(tmp_path: Path) -> None:
    games = oth.random_games(25, seed=1)
    write_wtb(tmp_path / "WTH_2000.wtb", games)
    assert np.array_equal(oth.read_wtb(str(tmp_path / "WTH_2000.wtb")), games)


def test_read_wtb_corner_codes(tmp_path: Path) -> None:
    # 11 = A1 = square 0, 18 = H1 = square 7, 81 = A8 = square 56, 88 = H8 = square 63
    rec = np.zeros(68, dtype=np.uint8)
    rec[8:12] = [11, 18, 81, 88]
    header = np.zeros(16, dtype=np.uint8)
    header[4] = 1
    (tmp_path / "WTH_1.wtb").write_bytes(header.tobytes() + rec.tobytes())
    assert oth.read_wtb(str(tmp_path / "WTH_1.wtb"))[0, :5].tolist() == [0, 7, 56, 63, -1]


def test_read_wtb_bad_count(tmp_path: Path) -> None:
    write_wtb(tmp_path / "WTH_2000.wtb", oth.random_games(3, seed=1))
    b = bytearray((tmp_path / "WTH_2000.wtb").read_bytes())
    b[4] = 4  # header claims 4 games, file holds 3
    (tmp_path / "WTH_2000.wtb").write_bytes(bytes(b))
    with pytest.raises(ValueError):
        oth.read_wtb(str(tmp_path / "WTH_2000.wtb"))


# ==========================================================================
# Championship cleaning, split and cache
# ==========================================================================

def test_championship_clean_split_cache(dirs: dict[str, str]) -> None:
    ch = oth.load_championship(dirs["championship_dir"])
    c = ch["counts"]
    assert c["n_raw"] == 205 and c["n_duplicates"] == 5 and c["n_illegal"] == 0
    assert len(ch["train"]) + len(ch["val"]) + len(ch["test"]) == 200
    assert len(ch["val"]) == len(ch["test"]) == 20 and len(ch["train"]) == 160
    # no game in two splits
    seen = [{g.tobytes() for g in ch[k]} for k in oth.SPLITS]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    # second load reads the cache and gives the same split
    assert os.path.exists(os.path.join(dirs["championship_dir"], "championship_split.npz"))
    again = oth.load_championship(dirs["championship_dir"])
    assert all(np.array_equal(again[k], ch[k]) for k in oth.SPLITS)


def test_championship_drops_illegal(tmp_path: Path) -> None:
    games = oth.random_games(10, seed=3)
    bad = games[0].copy()
    bad[0] = 0  # corner on move 1: illegal
    write_wtb(tmp_path / "WTH_2000.wtb", np.concatenate([games, bad[None]]))
    ch = oth.load_championship(str(tmp_path))
    assert ch["counts"]["n_illegal"] == 1 and sum(len(ch[k]) for k in oth.SPLITS) == 10


# ==========================================================================
# load_distribution
# ==========================================================================

def test_distribution_synthetic_unchanged(dirs: dict[str, str]) -> None:
    g = oth.load_distribution("synthetic", "train", 50, **dirs)
    assert np.array_equal(g, oth.load_split(dirs["synthetic_dir"], "train", 50))


def test_synthetic_val_and_test_files(dirs: dict[str, str]) -> None:
    val = oth.list_files(dirs["synthetic_dir"], "val")
    test = oth.list_files(dirs["synthetic_dir"], "test")
    assert len(val) == 2 and len(test) == oth.SYNTHETIC_TEST_FILES and not set(val) & set(test)
    assert [Path(f).name for f in val] == ["games_0.bin", "games_1.bin"]  # first val files stay val
    assert len(oth.load_distribution("synthetic", "test", 1000, **dirs)) == 100


def test_distribution_championship(dirs: dict[str, str]) -> None:
    train = oth.load_championship(dirs["championship_dir"])["train"]
    g = oth.load_distribution("championship", "train", 40, seed=0, **dirs)
    assert g.shape == (40, 60)
    pool = {x.tobytes() for x in train}
    assert all(x.tobytes() in pool for x in g)
    assert len(oth.load_distribution("championship", "train", 10_000, **dirs)) == len(train)  # all of them


def test_distribution_mixed_ratio(dirs: dict[str, str]) -> None:
    g = oth.load_distribution("mixed", "train", 100, seed=0, **dirs)
    assert g.shape == (100, 60)
    champ = {x.tobytes() for x in oth.load_championship(dirs["championship_dir"])["train"]}
    n_champ = sum(x.tobytes() in champ for x in g)
    assert n_champ == 100 - round(oth.MIXED_SYNTHETIC_SHARE * 100)
    assert np.array_equal(g, oth.load_distribution("mixed", "train", 100, seed=0, **dirs))  # seeded


def test_distribution_bad_name(dirs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        oth.load_distribution("random", "train", 10, **dirs)
    with pytest.raises(ValueError):
        oth.load_distribution("synthetic", "holdout", 10, **dirs)


# ==========================================================================
# Real WTHOR archive (EC2 only)
# ==========================================================================

@pytest.mark.skipif(not HAVE_WTHOR, reason=f"no WTH_*.wtb files under {oth.CHAMPIONSHIP_DIR}")
def test_real_championship() -> None:
    ch = oth.load_championship()
    c = ch["counts"]
    n = sum(len(ch[k]) for k in oth.SPLITS)
    assert n + c["n_duplicates"] + c["n_illegal"] == c["n_raw"]
    assert n > 100_000
    for k in oth.SPLITS:
        assert abs(len(ch[k]) / n - oth.CHAMPIONSHIP_SPLIT[k]) < 0.001
    d = oth.build(ch["test"][:2000])
    assert d["n_illegal_games"] == 0
