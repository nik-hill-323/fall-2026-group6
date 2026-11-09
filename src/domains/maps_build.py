"""Build the navigation data once, on the EC2 (design: src/docs/navigation_design.md, sections 3.1 to 3.6).

Steps, in this order (each step checks its own output and stops on any surprise):

    export-map  Vafa's random walks tokenizer (the canonical map, identical to the noisy one) and the
                OpenStreetMap coordinate file -> src/domains/maps_edges.csv and src/domains/maps_coordinates.csv.
                These two files are committed; everything after this reads the map from them.
    splits      held out routes of the three distributions -> the val and test pairs (one split of the
                union of all held out pairs, seed 0) and the val and test routes of every distribution
    cache       training routes -> 2,000,000 per distribution, none with a val or test pair
    manifest    sha256 of every data file, counts and the code commit -> src/domains/maps_data_manifest.json

What is dropped, and why (counts go into the build report and the manifest)
    shortest paths routes that use one of the 39 streets the canonical map does not have (design 3.1);
    the one renamed intersection 42438066 -> 7732571914 is rewritten, not dropped
    routes longer than MAX_TOKENS = 99 tokens (random walks only; design 3.6)
    training routes whose start and destination pair is a val or test pair (design 3.5)

Run on the EC2 from the repo root (conda env wmconcord), inside tmux:
    python src/domains/maps_build.py export-map --coords ~/nav_checks/canonical_coordinates_2024-05-05.csv
    python -m pytest src/tests/test_maps_domain.py -q        # then commit maps.py, this file and the two csv
    python src/domains/maps_build.py splits
    python src/domains/maps_build.py cache
    python src/domains/maps_build.py manifest                 # then commit the manifest
splits, cache and manifest refuse to run while files under src/ are not committed, so the commit they
record is the code that built the data.

Inputs: Vafa's data folder (WMC_VAFA_DATA, default ~/artifacts/world-model-evaluation/data).
Outputs: WMC_MAPS_DATA (default ~/artifacts/maps), plus the files under src/domains named above.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import importlib.util
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

log = logging.getLogger("maps_build")

ROOT = Path(__file__).resolve().parents[2]
EDGES_CSV = ROOT / "src/domains/maps_edges.csv"
COORDS_CSV = ROOT / "src/domains/maps_coordinates.csv"
MANIFEST = ROOT / "src/domains/maps_data_manifest.json"
VAFA_DATA_DIR = os.environ.get("WMC_VAFA_DATA", os.path.expanduser("~/artifacts/world-model-evaluation/data"))
DATA_DIR = os.environ.get("WMC_MAPS_DATA", os.path.expanduser("~/artifacts/maps"))
TURNS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# What the data checks of Nov 7 found (design section 2). The build stops if the files say otherwise.
EXPECTED = {"nodes": 4579, "edges": 9705, "dead_ends": 17, "heldout_pairs": 107331}


def maps_module():
    """src/domains/maps.py, loaded on demand (it reads maps_edges.csv when imported, so export-map must
    not import it)."""
    if "maps" in sys.modules:
        return sys.modules["maps"]
    spec = importlib.util.spec_from_file_location("maps", ROOT / "src/domains/maps.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maps"] = mod
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# Vafa's maps (read from his pickled tokenizers)
# ==========================================================================

class _VafaTokenizer:
    """Stand in for Vafa's SimpleTokenizer: only its attributes are read (valid_turns,
    node_and_direction_to_neighbor, word_to_id)."""


class _VafaUnpickler(pickle.Unpickler):
    """Vafa's tokenizer.pt files pickle a class named SimpleTokenizer; depending on how they were made it is
    recorded as model.SimpleTokenizer or __main__.SimpleTokenizer. Any module's SimpleTokenizer is read
    into the stand in class, so his environment (pytorch_lightning, old transformers) is not needed."""

    def find_class(self, module: str, name: str):
        if name == "SimpleTokenizer":
            return _VafaTokenizer
        return super().find_class(module, name)


_vafa_pickle = types.ModuleType("vafa_pickle")
_vafa_pickle.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
_vafa_pickle.Unpickler = _VafaUnpickler


def load_vafa_tokenizer(path: str):
    import torch
    return torch.load(path, weights_only=False, pickle_module=_vafa_pickle, map_location="cpu")


def vafa_map(name: str, vafa_dir: str = VAFA_DATA_DIR) -> tuple[dict[int, set[str]], dict[tuple[int, str], int]]:
    """(turns per intersection, (intersection, turn) -> next intersection) of one Vafa data folder."""
    tok = load_vafa_tokenizer(os.path.join(vafa_dir, name, "tokenizer.pt"))
    vt = {int(n): set(t) for n, t in tok.valid_turns.items()}
    nb = {(int(u), d): int(v) for (u, d), v in tok.node_and_direction_to_neighbor.items()}
    return vt, nb


def compass(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Nearest of the 8 compass directions from point 1 to point 2."""
    b = (math.degrees(math.atan2((lon2 - lon1) * math.cos(math.radians(lat1)), lat2 - lat1)) + 360) % 360
    return TURNS[int((b + 22.5) // 45) % 8]


# ==========================================================================
# Step 1: export the canonical map
# ==========================================================================

def export_map(coords_path: str, vafa_dir: str = VAFA_DATA_DIR) -> dict:
    vt, nb = vafa_map("random-walks", vafa_dir)
    vt2, nb2 = vafa_map("noisy-shortest-paths", vafa_dir)
    if vt != vt2 or nb != nb2:
        raise SystemExit("the random walks and noisy shortest paths maps differ: the canonical map is not defined")
    for (u, d), v in nb.items():
        if d not in TURNS:
            raise SystemExit(f"unknown turn {d!r} out of {u}")
    by_node = collections.defaultdict(set)
    for (u, d) in nb:
        by_node[u].add(d)
    for u in set(vt) | set(by_node):
        if by_node.get(u, set()) != vt.get(u, set()):
            raise SystemExit(f"turn table and neighbour table disagree at {u}")
    nodes = sorted(set(vt) | set(nb.values()))
    dead_ends = [n for n in nodes if not vt.get(n)]
    found = {"nodes": len(nodes), "edges": len(nb), "dead_ends": len(dead_ends)}
    for k, v in found.items():
        if v != EXPECTED[k]:
            raise SystemExit(f"expected {EXPECTED[k]} {k}, found {v}: the data is not what the design was written for")

    coords: dict[int, tuple[float, float]] = {}
    with open(os.path.expanduser(coords_path), newline="") as f:
        for r in csv.DictReader(f):
            coords[int(r["node"])] = (float(r["lat"]), float(r["lon"]))
    missing = [n for n in nodes if n not in coords]
    if missing:
        raise SystemExit(f"{len(missing)} intersections have no coordinates, e.g. {missing[:5]}")
    wrong = [(u, d, v) for (u, d), v in nb.items() if compass(*coords[u], *coords[v]) != d]
    if wrong:
        raise SystemExit(f"{len(wrong)} turn labels differ from the compass bearing, e.g. {wrong[:3]}: "
                         "the coordinates do not belong to this map")

    rows = sorted(((u, d, v) for (u, d), v in nb.items()), key=lambda r: (r[0], TURNS.index(r[1])))
    with open(EDGES_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["node", "turn", "next_node"])
        w.writerows(rows)
    with open(COORDS_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["node", "lat", "lon"])
        for n in nodes:
            w.writerow([n, f"{coords[n][0]:.7f}", f"{coords[n][1]:.7f}"])
    log.info("wrote %s (%d segments) and %s (%d intersections, %d dead ends)", EDGES_CSV.name, len(rows),
             COORDS_CSV.name, len(nodes), len(dead_ends))
    return found


# ==========================================================================
# Routes onto the canonical map
# ==========================================================================

class Translator:
    """Rewrites a route of one of Vafa's distributions as canonical map text, or says why it cannot.

    shortest paths: replay the route on its own map to get its intersections, rename 42438066, and write
    each step with the canonical turn label; a step whose street the canonical map does not have drops the
    route. noisy shortest paths and random walks: already on the canonical map, kept as they are.
    Every kept route is replayed on the canonical map again by maps.build before it is saved."""

    def __init__(self, distribution: str, vafa_dir: str = VAFA_DATA_DIR) -> None:
        m = maps_module()
        self.m = m
        self.distribution = distribution
        self.rename = {str(a): str(b) for a, b in m.RENAME_SHORTEST_PATHS.items()} if distribution == "shortest_paths" else {}
        self.label: dict[tuple[int, int], str] = {}
        for u, d, v in m._EDGES:
            self.label[(u, v)] = d
        if distribution == "shortest_paths":
            self.vt, self.nb = vafa_map(m.VAFA_NAMES[distribution], vafa_dir)

    def pair(self, words: list[str]) -> tuple[str, str]:
        """(start, destination) as canonical id strings."""
        return self.rename.get(words[0], words[0]), self.rename.get(words[1], words[1])

    def __call__(self, line: str) -> tuple[str | None, str]:
        words = line.split()
        if len(words) < 3:
            return None, "blank_or_short"
        if len(words) > self.m.MAX_TOKENS:
            return None, "longer_than_max_tokens"
        if self.distribution != "shortest_paths":
            return " ".join(words), "ok"
        s, dest = int(words[0]), int(words[1])
        path = [s]
        for t in words[2:]:
            if t == "end":
                break
            if t not in self.vt.get(s, ()):
                return None, "illegal_on_own_map"
            s = self.nb[(s, t)]
            path.append(s)
        else:
            return None, "no_end"
        if s != dest:
            return None, "end_not_at_destination"
        ren = self.m.RENAME_SHORTEST_PATHS
        path = [ren.get(x, x) for x in path]
        dest = ren.get(dest, dest)
        turns = []
        for u, v in zip(path, path[1:]):
            lab = self.label.get((u, v))
            if lab is None:
                return None, "street_missing_from_canonical_map"
            turns.append(lab)
        if path[0] not in self.m.NODE_TO_TOKEN or dest not in self.m.NODE_TO_TOKEN:
            return None, "intersection_not_on_canonical_map"
        return " ".join([str(path[0]), str(dest)] + turns + ["end"]), "ok"


def _lines(path: str):
    with open(path) as f:
        for line in f:
            yield line.rstrip("\n")


def _check_legal(m, tokens: np.ndarray, what: str) -> None:
    """Every route must replay legally on the canonical map (design 6.2)."""
    for i in range(0, len(tokens), 100_000):
        d = m.build(tokens[i:i + 100_000])
        if d["n_illegal"]:
            raise SystemExit(f"{what}: {d['n_illegal']} routes are not legal on the canonical map")


# ==========================================================================
# Step 2: val and test splits
# ==========================================================================

def build_splits(vafa_dir: str = VAFA_DATA_DIR, out_dir: str = DATA_DIR) -> dict:
    m = maps_module()
    os.makedirs(out_dir, exist_ok=True)
    held: dict[str, list[tuple[tuple[str, str], str | None, str]]] = {}
    union: set[tuple[int, int]] = set()
    report: dict = {"drops": {}, "routes": {}, "pairs": {}}
    for d in m.DISTRIBUTIONS:
        tr = Translator(d, vafa_dir)
        rows, reasons = [], collections.Counter()
        for line in _lines(os.path.join(vafa_dir, m.VAFA_NAMES[d], "heldout_sequences.txt")):
            words = line.split()
            if not words:
                continue
            p = tr.pair(words)
            union.add((int(p[0]), int(p[1])))
            text, why = tr(line)
            reasons[why] += 1
            rows.append((p, text, why))
        held[d] = rows
        report["drops"][f"{d}_heldout"] = dict(reasons)
        log.info("%s held out: %s", d, dict(reasons))
    if len(union) != EXPECTED["heldout_pairs"]:
        raise SystemExit(f"expected {EXPECTED['heldout_pairs']} held out pairs, found {len(union)}")
    pairs = np.array(sorted(union), dtype=np.int64)
    perm = np.random.default_rng(m.SPLIT_SEED).permutation(len(pairs))
    n_val = len(pairs) // 2
    split_pairs = {"val": pairs[np.sort(perm[:n_val])], "test": pairs[np.sort(perm[n_val:])]}
    which = {}
    for s, arr in split_pairs.items():
        np.save(os.path.join(out_dir, f"pairs_{s}.npy"), arr)
        report["pairs"][s] = len(arr)
        for a, b in arr:
            which[(str(a), str(b))] = s

    # pairs with no route on the canonical map must all come from dropped shortest path routes
    D = m.turn_distance()
    kept_pairs = {p for d in m.DISTRIBUTIONS for p, text, why in held[d] if why == "ok"}
    unreachable = [(a, b) for a, b in pairs
                   if a not in m.NODE_TO_TOKEN or b not in m.NODE_TO_TOKEN
                   or D[m.NODE_TO_TOKEN[a], m.NODE_TO_TOKEN[b]] < 0]
    bad = [p for p in unreachable if (str(p[0]), str(p[1])) in kept_pairs]
    if bad:
        raise SystemExit(f"kept routes whose pair has no route on the canonical map: {bad[:5]}")
    report["unreachable_pairs"] = len(unreachable)
    log.info("val pairs %d, test pairs %d, pairs with no route on the canonical map %d (all from dropped routes)",
             report["pairs"]["val"], report["pairs"]["test"], len(unreachable))

    for d in m.DISTRIBUTIONS:
        for s in ("val", "test"):
            toks = [m.encode(text) for p, text, why in held[d] if why == "ok" and which[p] == s]
            arr = m.pad_routes(toks)
            _check_legal(m, arr, f"{d} {s}")
            np.save(m.split_path(d, s, out_dir), arr.astype(np.uint16))
            report["routes"][f"{d}_{s}"] = len(arr)
            log.info("%s %s: %d routes", d, s, len(arr))
    _write_report(out_dir, "splits", report)
    return report


# ==========================================================================
# Step 3: training caches
# ==========================================================================

def build_cache(distribution: str, vafa_dir: str = VAFA_DATA_DIR, out_dir: str = DATA_DIR,
                n_cache: int | None = None) -> dict:
    """Two passes over the training file: count the eligible routes, draw n_cache of them with
    CACHE_SEED, then collect, translate, encode and check them. The draw depends only on the file and
    the seed, not on memory or reading order."""
    m = maps_module()
    n_cache = n_cache or m.N_CACHE
    held_pairs = set()
    for s in ("val", "test"):
        for a, b in m.load_pairs(s, out_dir):
            held_pairs.add((str(a), str(b)))
    tr = Translator(distribution, vafa_dir)
    path = os.path.join(vafa_dir, m.VAFA_NAMES[distribution], "train_sequences.txt")

    t0 = time.time()
    eligible = bytearray()
    reasons = collections.Counter()
    for line in _lines(path):
        words = line.split(" ", 2)
        if len(words) < 3:
            eligible.append(0)
            reasons["blank_or_short"] += 1
            continue
        if tr.pair(words) in held_pairs:
            eligible.append(0)
            reasons["val_or_test_pair"] += 1
            continue
        if distribution == "shortest_paths":
            text, why = tr(line)
        else:
            n_tok = line.count(" ") + 1            # fast; can overcount (trailing or double spaces), so
            if n_tok > m.MAX_TOKENS:               # a long count is confirmed with split()
                n_tok = len(line.split())
            why = "longer_than_max_tokens" if n_tok > m.MAX_TOKENS else "ok"
        reasons[why] += 1
        eligible.append(1 if why == "ok" else 0)
    mask = np.frombuffer(bytes(eligible), dtype=np.uint8).astype(bool)
    elig_lines = np.flatnonzero(mask)
    log.info("%s train: %d lines, %s, eligible %d (%.0fs)", distribution, len(mask), dict(reasons),
             len(elig_lines), time.time() - t0)
    if len(elig_lines) < n_cache:
        raise SystemExit(f"only {len(elig_lines)} eligible routes, need {n_cache}")
    draw = np.random.default_rng(m.CACHE_SEED).choice(len(elig_lines), n_cache, replace=False)
    lines = elig_lines[draw]                       # row r of the cache is file line lines[r]
    order = np.argsort(lines)
    sorted_lines = lines[order]

    t0 = time.time()
    out = np.zeros((n_cache, m.MAX_TOKENS), dtype=np.uint16)
    k = 0
    for i, line in enumerate(_lines(path)):
        if k == n_cache:
            break
        if i != sorted_lines[k]:
            continue
        text, why = tr(line)
        if why != "ok":
            raise SystemExit(f"line {i} was eligible in pass 1 but not in pass 2 ({why})")
        tok = m.encode(text)
        out[order[k], :len(tok)] = tok
        k += 1
    if k != n_cache:
        raise SystemExit(f"collected {k} of {n_cache} routes: the file changed between the passes")
    _check_legal(m, out, f"{distribution} train")
    # no val or test pair in the cache (design 3.5), checked again on the encoded routes
    tok_pairs = {(m.NODE_TO_TOKEN[int(a)], m.NODE_TO_TOKEN[int(b)]) for a, b in held_pairs
                 if int(a) in m.NODE_TO_TOKEN and int(b) in m.NODE_TO_TOKEN}
    cache_pairs = set(map(tuple, out[:, :2].astype(np.int64).tolist()))
    if cache_pairs & tok_pairs:
        raise SystemExit("a val or test pair is in the training cache")
    np.save(m.split_path(distribution, "train", out_dir), out)
    log.info("%s train cache: %d routes, %d distinct pairs, longest %d tokens (%.0fs)", distribution, n_cache,
             len(cache_pairs), int((out != 0).sum(1).max()), time.time() - t0)
    report = {"lines": int(len(mask)), "reasons": dict(reasons), "eligible": int(len(elig_lines)),
              "cache_routes": n_cache, "cache_pairs": len(cache_pairs)}
    _write_report(out_dir, f"cache_{distribution}", report)
    return report


# ==========================================================================
# Step 4: manifest
# ==========================================================================

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def git_commit() -> str:
    return _git("rev-parse", "--short", "HEAD").strip() or "unknown"


def uncommitted_code() -> list[str]:
    """Files under src/ that differ from the commit (the manifest itself does not count)."""
    out = _git("status", "--porcelain", "--", "src")
    files = [line[3:] for line in out.splitlines() if line.strip()]
    return [f for f in files if not f.endswith("maps_data_manifest.json")]


def _write_report(out_dir: str, step: str, report: dict) -> None:
    path = Path(out_dir) / "build_report.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[step] = {**report, "git_commit": git_commit(), "date": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    path.write_text(json.dumps(allr, indent=2, sort_keys=True) + "\n")


def write_manifest(out_dir: str = DATA_DIR) -> dict:
    m = maps_module()
    files = ["pairs_val.npy", "pairs_test.npy"] + [f"{d}_{s}.npy" for d in m.DISTRIBUTIONS for s in m.SPLITS]
    missing = [f for f in files if not (Path(out_dir) / f).exists()]
    if missing:
        raise SystemExit(f"missing data files: {missing}")
    report = json.loads((Path(out_dir) / "build_report.json").read_text())
    man = {
        "domain": m.DOMAIN,
        "design": "src/docs/navigation_design.md",
        "files": {f: m.sha256(Path(out_dir) / f) for f in files},
        "map_files": {"src/domains/maps_edges.csv": m.sha256(EDGES_CSV),
                      "src/domains/maps_coordinates.csv": m.sha256(COORDS_CSV)},
        "shapes": {f: list(np.load(Path(out_dir) / f, mmap_mode="r").shape) for f in files},
        "constants": {"MAX_TOKENS": m.MAX_TOKENS, "N_CTX": m.N_CTX, "VOCAB": m.VOCAB, "N_NODES": m.N_NODES,
                      "N_EDGES": m.N_EDGES, "N_CACHE": m.N_CACHE, "SPLIT_SEED": m.SPLIT_SEED,
                      "CACHE_SEED": m.CACHE_SEED},
        "build_report": report,
        "manifest_git_commit": git_commit(),
    }
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s", MANIFEST)
    return man


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the navigation map files, splits and training caches")
    ap.add_argument("step", choices=["export-map", "splits", "cache", "manifest"])
    ap.add_argument("--coords", default="~/nav_checks/canonical_coordinates_2024-05-05.csv",
                    help="OpenStreetMap coordinates (node,lat,lon), for export-map")
    ap.add_argument("--distribution", default="all", help="for cache: one distribution or all")
    ap.add_argument("--vafa_dir", default=VAFA_DATA_DIR)
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--n_cache", type=int, default=None, help="routes per training cache (tests only; default N_CACHE)")
    ap.add_argument("--allow_uncommitted", action="store_true",
                    help="run splits, cache or manifest with uncommitted code under src/ (tests only; never for real data)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.step == "export-map":
        export_map(args.coords, args.vafa_dir)
        return
    dirty = uncommitted_code()
    if dirty and not args.allow_uncommitted:
        sys.exit(f"refusing to run: these files under src/ are not committed: {dirty}. Commit them first, so "
                 "that the commit recorded with the data is the code that built it.")
    if args.step == "splits":
        build_splits(args.vafa_dir, args.out_dir)
    elif args.step == "cache":
        m = maps_module()
        names = m.DISTRIBUTIONS if args.distribution == "all" else [args.distribution]
        for d in names:
            build_cache(d, args.vafa_dir, args.out_dir, args.n_cache)
    else:
        write_manifest(args.out_dir)


if __name__ == "__main__":
    main()
