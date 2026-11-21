"""Navigation scoring foundation (N0): what every navigation scorer shares.

D1 to D4, the fragility suite and the whole route measure all score the same models on the same data
with the same rules. This file holds those rules once, so no scorer keeps its own copy:

    which models    the 81 zoo models of src/docs/models.csv (domain maps), the 9 pilots of Amendment 5,
                    G2 (small, seed 100, never in any analysis) and the 3 released models of Vafa et al.
                    (2024) (anchors, description only, G11)
    loading         load(model_id) gives a ZooModel (src/diagnostics/interface.py) for any of them. A zoo
                    model or pilot is loaded only if its checkpoint matches checkpoint_sha256 in its
                    training record, so every score belongs to the model that was trained
    gates           check_can_score(): navigation zoo models are refused until the git tags prereg-v1.5
                    and maps-fragility-v1 both exist (Amendment 5, gates). Pilots and anchors may be scored
                    before. Results are saved only from committed code under src/
    where results   zoo models and anchors: results/<diagnostic folder>/<model_id>.json as for Othello.
                    Pilots: results/pilot_scoring/<diagnostic folder>/<model_id>.json, a folder the
                    analysis never reads (G2: pilots are never in any analysis)
    data            routes(): routes of one distribution and split, checked against the data manifest,
                    replayed by maps.build (tokens, scored positions, current intersection, legal next
                    moves). Every scorer reads its data through it
    positions       a model reads tokens[:, :N_CTX]; scored positions are 1 <= p <= length - 2
                    (maps.build "score"), exactly the positions the training loss covers
    helpers         correct_at() (top token legal, per position), top_tokens() (the k best tokens per
                    position), legal_token_mask(), current_intersections()

Design: src/docs/navigation_design.md. Protocol: reports/preregistration.md, Amendment 5.
How to run the scorers: src/docs/navigation_scoring.md.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch

log = logging.getLogger("navigation")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str) -> ModuleType:
    """Import a file from src/ by path (src/ is not a package), shared through sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


itf = _load("interface", "src/diagnostics/interface.py")
results = itf.results
maps = _load("maps", "src/domains/maps.py")

# ==========================================================================
# Constants (Amendment 5)
# ==========================================================================

DOMAIN = "maps"
PREREG_TAG = "prereg-v1.5"
FRAGILITY_TAG = "maps-fragility-v1"
GATE_TAGS = (PREREG_TAG, FRAGILITY_TAG)
PILOT_SEED = 100
PILOT_SCALE = "small"
ARCHS = ("transformer", "lstm", "mamba")

ZOO_DIR = ROOT / "outputs" / "zoo"
PILOT_DIR = ROOT / "outputs" / "pilot"
ZOO_RECORDS = ROOT / "results" / "zoo_training"
PILOT_RECORDS = ROOT / "results" / "pilot_training"
PILOT_RESULTS_DIR = ROOT / "results" / "pilot_scoring"

ANCHORS = tuple(f"maps_vafa_{a}" for a in itf.VAFA_ANCHORS)
BATCH = 256                 # routes per forward pass, as D1 on Othello

# Data files are compared with src/domains/maps_data_manifest.json before they are read. Tests that build a
# small fake data folder turn this off.
VERIFY_DATA = True
_VERIFIED: set[tuple[str, str]] = set()


# ==========================================================================
# Which model is which
# ==========================================================================

def pilot_ids() -> list[str]:
    """The 9 navigation pilots of Amendment 5, G2 (the same list as src/zoo/train.py pilot_ids)."""
    return [f"maps_{a}_{PILOT_SCALE}_{d}_s{PILOT_SEED}" for a in ARCHS for d in maps.DISTRIBUTIONS]


def zoo_ids() -> list[str]:
    """The 81 navigation zoo models of src/docs/models.csv, in file order."""
    return [m for m, row in results.load_models().items() if row["domain"] == DOMAIN]


def kind(model_id: str) -> str:
    """'zoo', 'pilot' or 'anchor'. Anything else is not a navigation model and raises ValueError."""
    if model_id in ANCHORS:
        return "anchor"
    if model_id in pilot_ids():
        return "pilot"
    row = results.load_models().get(model_id)
    if row is not None and row["domain"] == DOMAIN:
        return "zoo"
    raise ValueError(f"{model_id} is not a navigation zoo model (src/docs/models.csv), pilot "
                     f"(seed {PILOT_SEED}, small) or anchor {ANCHORS}")


def is_maps_id(model_id: str) -> bool:
    return model_id.startswith("maps_")


def parse_id(model_id: str) -> dict[str, Any]:
    """maps_{arch}_{scale}_{distribution}_s{seed} -> its fields (for pilots, which are not in models.csv)."""
    if not is_maps_id(model_id) or "_s" not in model_id:
        raise ValueError(f"not a navigation model id: {model_id}")
    body, seed = model_id[len("maps_"):].rsplit("_s", 1)
    arch, scale, dist = body.split("_", 2)
    if arch not in ARCHS or dist not in maps.DISTRIBUTIONS:
        raise ValueError(f"not a navigation model id: {model_id}")
    return {"domain": DOMAIN, "arch": arch, "scale": scale, "distribution": dist, "seed": int(seed)}


def model_fields(model) -> dict[str, Any]:
    """domain, arch, scale, distribution and seed of a loaded model: models.csv for zoo models, the model's
    meta for pilots and anchors."""
    row = results.load_models().get(model.model_id)
    if row is not None:
        return {"domain": row["domain"], "arch": row["arch"], "scale": row["scale"],
                "distribution": row["distribution"], "seed": int(row["seed"])}
    missing = [k for k in itf.MODEL_FIELDS if k not in model.meta]
    if missing:
        raise ValueError(f"{model.model_id}: not in models.csv and its meta lacks {missing}")
    return {k: model.meta[k] for k in itf.MODEL_FIELDS}


def is_maps(model) -> bool:
    """True if the model belongs to the navigation domain."""
    row = results.load_models().get(model.model_id)
    if row is not None:
        return row["domain"] == DOMAIN
    return model.meta.get("domain") == DOMAIN


def training_distribution(model) -> str:
    return model_fields(model)["distribution"]


# ==========================================================================
# Gates (Amendment 5): tags and committed code
# ==========================================================================

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def missing_tags(tags: Iterable[str] = GATE_TAGS) -> list[str]:
    """The gate tags that do not exist in this clone."""
    return [t for t in tags if _git("tag", "-l", t).strip() != t]


def uncommitted_src() -> list[str]:
    """Files under src/ with uncommitted changes (a saved result records the commit, so it must hold the code)."""
    return [line[3:] for line in _git("status", "--porcelain", "--", "src").splitlines() if line.strip()]


def check_can_score(model_ids: list[str], save: bool) -> None:
    """Refuse (SystemExit with the reason) when Amendment 5 does not allow these models to be scored now:
        * every id must be a navigation zoo model, pilot or anchor (and not mixed with Othello ids);
        * a zoo model needs both tags prereg-v1.5 and maps-fragility-v1 (no navigation zoo model is scored by a
          diagnostic or by the fragility suite before maps-fragility-v1 exists);
        * saving a zoo model or pilot result needs committed code under src/."""
    if not model_ids:
        raise SystemExit("no models given")
    other = [m for m in model_ids if not is_maps_id(m)]
    if other:
        raise SystemExit(f"score Othello and navigation models in separate commands; not navigation: {other}")
    unknown = []
    kinds = {}
    for m in model_ids:
        try:
            kinds[m] = kind(m)
        except ValueError as e:
            unknown.append(str(e))
    if unknown:
        raise SystemExit("\n".join(unknown))
    zoo = [m for m, k in kinds.items() if k == "zoo"]
    if zoo:
        miss = missing_tags()
        if miss:
            raise SystemExit(f"refusing to score navigation zoo models {zoo[:3]}{' ...' if len(zoo) > 3 else ''}: "
                             f"the git tag(s) {miss} do not exist here (Amendment 5: no navigation zoo model is scored "
                             f"before {PREREG_TAG} and {FRAGILITY_TAG}). Run: git fetch --tags. "
                             f"Pilots and anchors can be scored now.")
    if save and any(k in ("zoo", "pilot") for k in kinds.values()):
        dirty = uncommitted_src()
        if dirty:
            raise SystemExit(f"refusing to save results: these files under src/ are not committed: {dirty}")


# ==========================================================================
# Loading models
# ==========================================================================

def record_path(model_id: str) -> Path:
    """The training record of a zoo model or pilot (written by src/zoo/train.py)."""
    k = kind(model_id)
    if k == "anchor":
        raise ValueError("anchors have no training record")
    return (ZOO_RECORDS if k == "zoo" else PILOT_RECORDS) / f"{model_id}.json"


def checkpoint_path(model_id: str, zoo_dir: Path | str | None = None, pilot_dir: Path | str | None = None) -> Path:
    k = kind(model_id)
    if k == "zoo":
        return Path(zoo_dir or ZOO_DIR) / model_id / "model.pt"
    if k == "pilot":
        return Path(pilot_dir or PILOT_DIR) / model_id / "model.pt"
    raise ValueError("anchors are loaded with interface.vafa_navigation")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(model_id: str, path: Path) -> str:
    """The checkpoint's sha256, after checking it equals checkpoint_sha256 in the model's training record."""
    rec = record_path(model_id)
    if not rec.exists():
        raise FileNotFoundError(f"no training record {rec}: pull main")
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {path}: copy it from S3 or from the machine that trained it "
                                f"(src/docs/navigation_scoring.md)")
    want = json.loads(rec.read_text())["checkpoint_sha256"]
    got = sha256_file(path)
    if got != want:
        raise ValueError(f"{path} has sha256 {got[:12]}..., its training record says {want[:12]}...: "
                         f"this is not the trained model")
    return got


def load(model_id: str, device: str | None = None, verify: bool = True, zoo_dir: Path | str | None = None,
         pilot_dir: Path | str | None = None):
    """Any navigation model as a ZooModel. Zoo models and pilots are checked against their training record
    (verify=False only in tests). Pilots carry their fields in meta, since they are not in models.csv."""
    k = kind(model_id)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if k == "anchor":
        return itf.vafa_navigation(model_id.removeprefix("maps_vafa_"), dev)
    path = checkpoint_path(model_id, zoo_dir, pilot_dir)
    if verify:
        verify_checkpoint(model_id, path)
    if k == "zoo":
        return itf.from_checkpoint(model_id, dev, path.parent.parent)
    fields = parse_id(model_id)
    train = _load("zoo_train", "src/zoo/train.py")
    module = train.build_model(fields["arch"], fields["scale"], DOMAIN)
    module.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    cfg = train.matrix.SCALE_CONFIG[fields["scale"]]
    return itf.TrainPyModel(module, model_id, fields["arch"], cfg["n_layers"], cfg["d_model"], dev, fields)


def results_dir(model_id: str) -> Path:
    """Where a result of this model is saved: pilots apart from everything the analysis reads (G2)."""
    return PILOT_RESULTS_DIR if kind(model_id) == "pilot" else results.RESULTS_DIR


# ==========================================================================
# Data
# ==========================================================================

def verify_split(distribution: str, split: str, data_dir: str | Path) -> None:
    """Compare one data file with its sha256 in src/domains/maps_data_manifest.json (once per process)."""
    key = (str(data_dir), f"{distribution}_{split}")
    if not VERIFY_DATA or key in _VERIFIED:
        return
    name = f"{distribution}_{split}.npy"
    want = json.loads(maps.MANIFEST.read_text())["files"].get(name)
    if want is None:
        raise ValueError(f"{name} is not in {maps.MANIFEST.name}")
    got = maps.sha256(Path(data_dir) / name)
    if got != want:
        raise ValueError(f"{Path(data_dir) / name} does not match the data manifest (sha256 {got[:12]}...)")
    _VERIFIED.add(key)


def routes(distribution: str, split: str, n_routes: int, seed: int = 0, data_dir: str | Path | None = None) -> dict:
    """n_routes routes of one distribution and split (drawn with `seed` as maps.load_distribution does), checked
    against the manifest and replayed by maps.build. Every route of the built data is legal, so a dropped route
    means broken data and raises."""
    data_dir = data_dir or maps.DATA_DIR
    verify_split(distribution, split, data_dir)
    d = maps.build(maps.load_distribution(distribution, split, n_routes, seed=seed, data_dir=str(data_dir)))
    if d["n_illegal"]:
        raise ValueError(f"{distribution} {split}: {d['n_illegal']} routes are not legal on the canonical map")
    return d


def inputs(data: dict, rows=slice(None)) -> torch.Tensor:
    """The tokens a model reads for these routes: (B, N_CTX)."""
    return torch.as_tensor(np.asarray(data["tokens"][rows, :maps.N_CTX], dtype=np.int64))


def legal_token_mask(cur: int, dest: int) -> np.ndarray:
    """(VOCAB,) bool: the legal next tokens at intersection token `cur` with destination token `dest`."""
    out = np.zeros(maps.VOCAB, dtype=bool)
    out[maps.legal_next(int(cur), int(dest))] = True
    return out


def slots_to_tokens(legal_slots: np.ndarray) -> np.ndarray:
    """(..., 9) legal slots (maps.build "legal") -> (..., VOCAB) bool legal tokens."""
    out = np.zeros(legal_slots.shape[:-1] + (maps.VOCAB,), dtype=bool)
    out[..., maps.SLOT_TOKEN] = legal_slots
    return out


def current_intersections(tokens: np.ndarray) -> np.ndarray:
    """(G, T) tokens of route prefixes (no end needed) -> (G, T) intersection token at each position p >= 1, as
    maps.build "current" gives for full routes; 0 at position 0 and at padding. A turn that does not exist
    leaves the car where it is (only used on legal routes)."""
    tok = np.asarray(tokens, dtype=np.int64)
    G, T = tok.shape
    out = np.zeros((G, T), dtype=np.int64)
    if T < 2:
        return out
    cur = tok[:, 0].copy()
    for p in range(1, T):
        if p >= 2:
            t = tok[:, p]
            is_turn = (t >= maps.TURN0) & (t < maps.TURN0 + len(maps.TURNS))
            nxt = maps.NEXT[cur, np.clip(t - maps.TURN0, 0, len(maps.TURNS) - 1)]
            cur = np.where(is_turn & (nxt >= 0), nxt, cur)
        out[:, p] = np.where(tok[:, p] != maps.PAD, cur, 0)
    return out


# ==========================================================================
# What a model predicts at the scored positions
# ==========================================================================

@torch.no_grad()
def correct_at(model, data: dict, batch: int = BATCH) -> np.ndarray:
    """(G, N_CTX) bool: True at a scored position where the model's top token is legal (the D1 rule, position
    by position). correct_at(...).sum() / data['score'][:, :N_CTX].sum() equals maps.legal_rate."""
    G = len(data["tokens"])
    out = np.zeros((G, maps.N_CTX), dtype=bool)
    for i in range(0, G, batch):
        pred = model(inputs(data, slice(i, i + batch))).argmax(-1).cpu().numpy()
        part = {"legal": data["legal"][i:i + batch]}
        out[i:i + batch] = maps.is_legal(part, pred) & data["score"][i:i + batch, :maps.N_CTX]
    return out


@torch.no_grad()
def top_tokens(model, data: dict, k: int, batch: int = BATCH, exclude_pad: bool = True) -> np.ndarray:
    """(G, N_CTX, k) int64: the model's k highest scoring tokens at every position, best first. Padding is never
    among them when exclude_pad (it is never a legal next token, as token 0 on Othello)."""
    G = len(data["tokens"])
    out = np.zeros((G, maps.N_CTX, k), dtype=np.int64)
    for i in range(0, G, batch):
        logits = model.logits(inputs(data, slice(i, i + batch))).float()
        if exclude_pad:
            logits[..., maps.PAD] = -float("inf")
        out[i:i + batch] = logits.topk(k, dim=-1).indices.cpu().numpy()
    return out


# ==========================================================================
# Running a scorer over navigation models
# ==========================================================================

def run(diagnostic: ModuleType, model_ids: list[str], save: bool = True, device: str | None = None,
        load_fn=None, **kw: Any) -> tuple[list, list[tuple[str, str]]]:
    """Score models one at a time with one diagnostic (interface.run per model, so a crash becomes a failed
    result). Pilot results go to results/pilot_scoring/. A model that cannot be loaded (no checkpoint here, a
    checksum that does not match) gets no result file; it is returned in the second list with the reason."""
    load_fn = load_fn or (lambda m: load(m, device))
    out, not_loaded = [], []
    for m in model_ids:
        t0 = time.time()
        try:
            model = load_fn(m)
        except Exception as e:  # noqa: BLE001  one missing checkpoint must not stop a sweep
            log.warning("cannot load %s: %s", m, e)
            not_loaded.append((m, f"{type(e).__name__}: {e}"))
            continue
        out.extend(itf.run(diagnostic, [model], results_dir=results_dir(m), save=save, **kw))
        log.info("%s %s done in %.0f s", diagnostic.DIAGNOSTIC, m, time.time() - t0)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, not_loaded


def report_not_loaded(not_loaded: list[tuple[str, str]]) -> None:
    for m, why in not_loaded:
        print(f"{m:45s} NOT SCORED: {why}")


def main_for(diagnostic: ModuleType, model_ids: list[str], save: bool, **kw: Any) -> tuple[list, list]:
    """The command line path of every navigation scorer: gates, then run(). Exits with status 1 when a model
    could not be loaded, so a launcher sees it."""
    check_can_score(model_ids, save)
    return run(diagnostic, model_ids, save=save, **kw)
