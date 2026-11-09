"""Tests for the navigation domain (src/domains/maps.py) and its data build (src/domains/maps_build.py).

Every navigation diagnostic, the trainer and the fragility suite use this module, so these check the
map, the tokens, the true state, the legal next moves, the positions that are trained and scored, and
the splits. Design: src/docs/navigation_design.md (the section numbers below refer to it).

Tests on the committed map files and on generated routes run everywhere (including GitHub Actions).
Tests on the built data (~/artifacts/maps) are skipped when it is not there (it only exists on the EC2).

Run from the repo root:
    python -m pytest src/tests/test_maps_domain.py -q
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
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


maps = _load("maps", "src/domains/maps.py")
build_mod = _load("maps_build", "src/domains/maps_build.py")

needs_data = pytest.mark.skipif(not (maps.have_data() and maps.MANIFEST.exists()),
                                reason=f"no built navigation data under {maps.DATA_DIR}")


# ==========================================================================
# Route generators on the real map (for tests only)
# ==========================================================================

def random_walk(rng: np.random.Generator, n_turns: int, start: int | None = None) -> tuple[int, list[str], list[int]]:
    """A walk of up to n_turns legal turns from a start with turns. Returns (start token, turns, states)."""
    with_turns = np.flatnonzero(maps.OUT_DEGREE > 0) + 1
    cur = int(start if start is not None else rng.choice(with_turns))
    s0, turns, states = cur, [], [cur]
    for _ in range(n_turns):
        opts = [i for i in range(8) if maps.NEXT[cur, i] >= 0]
        if not opts:
            break
        i = int(rng.choice(opts))
        turns.append(maps.TURNS[i])
        cur = int(maps.NEXT[cur, i])
        states.append(cur)
    return s0, turns, states


def shortest_route(a: int, b: int) -> list[str]:
    """Turns of one fewest turn route from token a to token b (b reachable from a)."""
    D = maps.turn_distance()
    assert D[a, b] >= 0
    turns, cur = [], a
    while cur != b:
        for i in range(8):
            nxt = int(maps.NEXT[cur, i])
            if nxt >= 0 and D[nxt, b] == D[cur, b] - 1:
                turns.append(maps.TURNS[i])
                cur = nxt
                break
    return turns


def text(start: int, dest: int, turns: list[str]) -> str:
    return " ".join([str(maps.node_of_token(start)), str(maps.node_of_token(dest))] + turns + ["end"])


def random_routes(n: int, seed: int) -> list[np.ndarray]:
    """n legal routes: half random walks (destination = where the walk ends), half fewest turn routes."""
    rng = np.random.default_rng(seed)
    D = maps.turn_distance()
    out = []
    while len(out) < n:
        if len(out) % 2 == 0:
            s, turns, states = random_walk(rng, int(rng.integers(1, maps.MAX_TURNS + 1)))
            out.append(maps.encode(text(s, states[-1], turns)))
        else:
            a, b = (int(x) for x in rng.integers(1, maps.N_NODES + 1, size=2))
            if a == b or not 0 < D[a, b] <= maps.MAX_TURNS or maps.OUT_DEGREE[a - 1] == 0:
                continue
            out.append(maps.encode(text(a, b, shortest_route(a, b))))
    return out


# ==========================================================================
# The map (design 2.2, 2.3, 3.2)
# ==========================================================================

def test_map_matches_the_data_checks() -> None:
    assert maps.N_NODES == 4579 and maps.N_EDGES == 9705
    assert int(maps.IS_DEAD_END.sum()) == 17
    deg = collections.Counter(int(d) for d in maps.OUT_DEGREE if d > 0)
    assert dict(deg) == {1: 717, 2: 2698, 3: 996, 4: 151}
    assert 7732571914 in maps.NODE_TO_TOKEN and 42438066 not in maps.NODE_TO_TOKEN
    assert maps.RENAME_SHORTEST_PATHS == {42438066: 7732571914}
    # every dead end is reachable (otherwise it would not be on the map)
    reached = set(maps.NEXT[maps.NEXT >= 0].tolist())
    assert all(t in reached for t in np.flatnonzero(maps.IS_DEAD_END))


def test_coordinates_belong_to_this_map() -> None:
    lat, lon = maps.coordinates()
    assert np.isfinite(lat[1:]).all() and np.isfinite(lon[1:]).all()
    assert 40.68 < lat[1:].min() and lat[1:].max() < 40.89 and -74.03 < lon[1:].min() and lon[1:].max() < -73.90
    wrong = [(u, i) for u in range(1, maps.N_NODES + 1) for i in range(8) if maps.NEXT[u, i] >= 0
             and build_mod.compass(lat[u], lon[u], lat[maps.NEXT[u, i]], lon[maps.NEXT[u, i]]) != maps.TURNS[i]]
    assert wrong == []


def test_vocabulary() -> None:
    assert maps.VOCAB == 4589 and maps.END == 4588 and maps.TURN0 == 4580 and maps.N_CTX == 98
    assert [maps.token_name(maps.TURN0 + i) for i in range(8)] == maps.TURNS
    assert maps.token_name(maps.END) == "end" and maps.token_name(0) == "<pad>"
    assert all(maps.token_of_node(maps.node_of_token(t)) == t for t in range(1, maps.N_NODES + 1))
    assert list(maps.NODE_IDS) == sorted(maps.NODE_IDS)
    # only turns and end have a slot; padding and intersections never count as legal
    assert maps.TOKEN_SLOT[0] == -1 and (maps.TOKEN_SLOT[1:maps.N_NODES + 1] == -1).all()
    assert list(maps.TOKEN_SLOT[maps.TURN0:]) == list(range(9))
    with pytest.raises(ValueError):
        maps.token_name(maps.VOCAB)


def test_encode_decode_round_trip() -> None:
    for r in random_routes(50, seed=1):
        assert np.array_equal(maps.encode(maps.decode(r)), r)
    with pytest.raises(KeyError):
        maps.encode("42438066 7732571914 N end")  # the shortest paths id is not on the canonical map
    with pytest.raises(ValueError):
        maps.encode("7732571914 7732571914")


# ==========================================================================
# State and legal next set (design 3.4)
# ==========================================================================

def test_build_matches_replay() -> None:
    routes = random_routes(400, seed=2)
    d = maps.build(routes)
    assert d["n_illegal"] == 0 and len(d["tokens"]) == 400
    for g, r in enumerate(routes):
        L = len(r)
        states = maps.replay(r)
        assert states is not None and len(states) == L - 1
        assert d["length"][g] == L and d["dest"][g] == r[1]
        assert list(d["current"][g, 1:L]) == states and (d["current"][g, L:] == 0).all() and d["current"][g, 0] == 0
        assert list(np.flatnonzero(d["score"][g])) == list(range(1, L - 1))
        for p in range(1, L - 1):
            slots = np.flatnonzero(d["legal"][g, p])
            assert [int(maps.SLOT_TOKEN[s]) for s in slots] == maps.legal_next(states[p - 1], int(r[1]))
            assert int(r[p + 1]) in maps.legal_next(states[p - 1], int(r[1]))  # the route's own next token
            assert d["n_turns"][g, p] == p - 1
        assert d["n_turns"][g, L - 1] == L - 3                     # end is not a turn
        assert not d["legal"][g, L - 1:].any() and not d["legal"][g, 0].any()


def test_route_may_pass_through_its_destination() -> None:
    rng = np.random.default_rng(3)
    for _ in range(10_000):
        s, turns, states = random_walk(rng, 40)
        later = [i for i in range(1, len(states) - 1) if states[i] == states[-1]]
        if later:
            break
    else:
        pytest.fail("no walk revisits its end point")
    dest = states[-1]
    full = maps.encode(text(s, dest, turns))
    early = maps.encode(text(s, dest, turns[:later[0]]))         # stops at the first visit
    d = maps.build([full, early])
    assert d["n_illegal"] == 0
    p = later[0] + 1                                             # position where the car first reaches dest
    assert d["current"][0, p] == dest and d["legal"][0, p, maps.END_SLOT]
    assert d["legal"][0, p, :8].any()                           # and it may continue
    # but `end` is final: a route that says end at its first visit and then drives on is not legal
    k = later[0]
    end_then_more = maps.encode(text(s, dest, turns[:k] + ["end"] + turns[k:]))
    assert maps.replay(end_then_more) is None
    assert maps.build([end_then_more])["n_illegal"] == 1


def test_dead_end_destination() -> None:
    D = maps.turn_distance()
    dead = int(np.flatnonzero(maps.IS_DEAD_END)[0])
    starts = [a for a in range(1, maps.N_NODES + 1) if maps.OUT_DEGREE[a - 1] > 0 and 0 < D[a, dead] <= 30]
    a = starts[0]
    r = maps.encode(text(a, dead, shortest_route(a, dead)))
    d = maps.build([r])
    assert d["n_illegal"] == 0
    last = len(r) - 2
    assert list(np.flatnonzero(d["legal"][0, last])) == [maps.END_SLOT]
    other = 1 if dead != 1 else 2
    assert maps.legal_next(dead, other) == []                    # stuck: a dead end that is not the destination


def test_illegal_routes_are_dropped() -> None:
    rng = np.random.default_rng(4)
    s, turns, states = random_walk(rng, 10)
    good = maps.encode(text(s, states[-1], turns))
    missing = [t for t in maps.TURNS if t not in turns[:1] and maps.NEXT[s, maps.TURNS.index(t)] < 0][0]
    bad_turn = maps.encode(text(s, states[-1], [missing] + turns[1:]))
    wrong_dest = maps.encode(text(s, states[-2], turns))         # end not at the destination
    no_end = good[:-1]
    after_end = np.concatenate([good, [maps.TURN0]])
    node_as_turn = good.copy()
    node_as_turn[2] = 5
    gap = np.concatenate([good[:3], [0], good[3:]])              # padding inside the route
    d = maps.build([good, bad_turn, wrong_dest, no_end, after_end, node_as_turn, gap])
    assert d["n_illegal"] == 6 and list(d["kept"]) == [True] + [False] * 6
    for r in (bad_turn, wrong_dest, no_end, after_end, node_as_turn):
        assert maps.replay(r) is None
    with pytest.raises(ValueError):
        maps.pad_routes([np.ones(maps.MAX_TOKENS + 1, dtype=np.int64)])


# ==========================================================================
# Trained and scored positions are the same (design 3.6; the Othello move 60 lesson)
# ==========================================================================

def test_training_targets_cover_exactly_the_scored_positions() -> None:
    d = maps.build(random_routes(300, seed=5))
    x = torch.tensor(d["tokens"])
    inputs, targets = maps.train_targets(x)
    assert inputs.shape[1] == maps.N_CTX == targets.shape[1]
    trained = (targets != maps.PAD).numpy()
    assert np.array_equal(trained, d["score"][:, :maps.N_CTX])
    assert not d["score"][:, maps.N_CTX:].any()                  # nothing is scored beyond what a model reads
    # the target at a trained position is the route's own next token
    g, p = np.nonzero(trained)
    assert np.array_equal(targets.numpy()[g, p], d["tokens"][g, p + 1])


def test_longest_route_fits() -> None:
    rng = np.random.default_rng(6)
    for _ in range(1000):
        s, turns, states = random_walk(rng, maps.MAX_TURNS)
        if len(turns) == maps.MAX_TURNS:
            break
    r = maps.encode(text(s, states[-1], turns))
    assert len(r) == maps.MAX_TOKENS
    d = maps.build([r])
    assert d["n_illegal"] == 0 and d["score"][0].sum() == maps.MAX_TOKENS - 2
    assert np.flatnonzero(d["score"][0]).max() == maps.N_CTX - 1


# ==========================================================================
# legal_rate (the D1 rule)
# ==========================================================================

class _Oracle(torch.nn.Module):
    """Replays the input and puts all weight on a legal next token (the first one) or on a fixed token."""

    def __init__(self, fixed: int | None = None) -> None:
        super().__init__()
        self.fixed = fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        out = torch.full((B, T, maps.VOCAB), -1e9)
        for b in range(B):
            row = [int(t) for t in x[b]]
            cur, dest = row[0], row[1]
            for p in range(T):
                if p >= 2 and maps.TURN0 <= row[p] < maps.TURN0 + 8:
                    cur = int(maps.NEXT[cur, row[p] - maps.TURN0])
                if self.fixed is not None:
                    out[b, p, self.fixed] = 0.0
                    continue
                legal = maps.legal_next(cur, dest) if p >= 1 and row[p] != maps.PAD else []
                out[b, p, legal[0] if legal else maps.PAD] = 0.0
        return out


def test_legal_rate_of_reference_models() -> None:
    routes = random_routes(60, seed=7)
    d = maps.build(routes)
    assert maps.legal_rate(_Oracle(), d, "cpu", batch=16) == 1.0
    assert maps.legal_rate(_Oracle(fixed=maps.PAD), d, "cpu", batch=16) == 0.0
    assert maps.legal_rate(_Oracle(fixed=1), d, "cpu", batch=16) == 0.0      # an intersection is never legal
    at_dest = (d["current"] == d["dest"][:, None]) & d["score"]
    expected = at_dest.sum() / d["score"].sum()
    assert maps.legal_rate(_Oracle(fixed=maps.END), d, "cpu", batch=16) == pytest.approx(expected)
    # batch size does not change the result
    assert maps.legal_rate(_Oracle(fixed=maps.END), d, "cpu", batch=7) == pytest.approx(expected)


# ==========================================================================
# Distances and safe moves (design 2.2, 4 D3, 5.1)
# ==========================================================================

def test_turn_distance_matches_the_data_checks() -> None:
    D = maps.turn_distance()
    inner = D[1:, 1:]
    assert (np.diag(inner) == 0).all()
    off = ~np.eye(maps.N_NODES, dtype=bool)
    assert int(((inner < 0) & off).sum()) == 505_523             # unreachable ordered pairs (check 5)
    fin = inner[(inner >= 0) & off]
    assert int(np.median(fin)) == 42 and int(fin.max()) == 124
    for r in random_routes(100, seed=8):                         # a route is never shorter than the distance
        assert D[r[0], r[1]] <= len(r) - 3


def test_safe_moves() -> None:
    D = maps.turn_distance()
    rng = np.random.default_rng(9)
    n_checked = 0
    while n_checked < 300:
        cur, dest = (int(x) for x in rng.integers(1, maps.N_NODES + 1, size=2))
        if D[cur, dest] < 0:
            continue
        left = int(D[cur, dest]) + int(rng.integers(0, 4))
        safe = maps.safe_moves(cur, dest, left)
        assert safe, "a reachable destination always leaves a safe move"
        legal = maps.legal_next(cur, dest)
        for t in safe:
            assert t in legal
            if t != maps.END:
                nxt = int(maps.NEXT[cur, t - maps.TURN0])
                assert 0 <= D[nxt, dest] <= left - 1
        assert (maps.END in safe) == (cur == dest)
        n_checked += 1
    assert maps.safe_moves(cur, dest, 0) == ([maps.END] if cur == dest else [])


# ==========================================================================
# The build: translation of shortest paths routes, splits and caches (design 3.1, 3.5, 3.6)
# ==========================================================================

def _sp_map():
    """The canonical map as the shortest paths map would differ from it: 7732571914 called 42438066,
    and one street replaced by a street the canonical map does not have."""
    vt, nb = collections.defaultdict(set), {}
    ren = {7732571914: 42438066}
    for u, t, v in maps._EDGES:
        u2, v2 = ren.get(u, u), ren.get(v, v)
        vt[u2].add(t)
        nb[(u2, t)] = v2
    # pick a segment a -> b, and redirect it to an intersection c that a has no street to
    a, t, b = next((u, t, v) for u, t, v in maps._EDGES if u not in ren and v not in ren)
    c = next(n for n in maps.NODE_IDS if int(n) not in (a, b) and int(n) not in ren
             and maps.token_of_node(int(n)) not in set(maps.NEXT[maps.token_of_node(a)].tolist()))
    nb[(a, t)] = int(c)
    return dict(vt), nb, (a, t, b, int(c))


def test_translator_on_a_modified_map() -> None:
    vt, nb, (a, t, b, c) = _sp_map()
    tr = object.__new__(build_mod.Translator)
    tr.m, tr.distribution = maps, "shortest_paths"
    tr.rename = {"42438066": "7732571914"}
    tr.label = {(u, v): tt for u, tt, v in maps._EDGES}
    tr.vt, tr.nb = vt, nb
    # a route through the renamed intersection is rewritten, with the canonical id
    tok_new = maps.token_of_node(7732571914)
    D = maps.turn_distance()
    tried = 0
    for s in (x for x in range(1, maps.N_NODES + 1) if maps.OUT_DEGREE[x - 1] > 0 and 2 <= D[x, tok_new] <= 6):
        turns = shortest_route(s, tok_new)
        line = " ".join([str(maps.node_of_token(s)), "42438066"] + turns + ["end"])
        out, why = tr(line)
        tried += 1
        if why == "ok":
            break
    assert why == "ok", f"no route to the renamed intersection translated in {tried} tries"
    assert out.split()[1] == "7732571914" and maps.replay(maps.encode(out)) is not None
    assert out.split()[2:] == turns + ["end"]                     # same street, same canonical labels
    assert tr.pair(line.split()) == (str(maps.node_of_token(s)), "7732571914")
    # a route over the street the canonical map does not have is dropped
    out, why = tr(f"{a} {c} {t} end")
    assert out is None and why == "street_missing_from_canonical_map"
    # illegal on its own map, end not at the destination, too long
    assert tr(f"{a} {c} XX end")[1] == "illegal_on_own_map"
    assert tr(f"{a} {a} {t} end")[1] == "end_not_at_destination"
    assert tr(" ".join(["1"] * (maps.MAX_TOKENS + 1)))[1] == "longer_than_max_tokens"


def _fake_vafa(tmp: Path, seed: int = 10) -> Path:
    """A small Vafa style data folder on the real map: tokenizers, held out and train files."""
    # Vafa's real files pickle __main__.SimpleTokenizer; other copies may say model.SimpleTokenizer. The fake
    # folder uses both (random walks, the canonical map, uses __main__), and both classes are removed
    # again after saving, so loading must work without either of them.
    def saver(module: str):
        mod = sys.modules.setdefault(module, types.ModuleType(module))
        cls = type("SimpleTokenizer", (), {"__module__": module, "__qualname__": "SimpleTokenizer"})
        return mod, cls
    vt_c = collections.defaultdict(list)
    nb_c = {}
    for u, t, v in maps._EDGES:
        vt_c[u].append(t)
        nb_c[(u, t)] = v
    vt_s, nb_s, _ = _sp_map()
    rng = np.random.default_rng(seed)
    D = maps.turn_distance()

    def walks(n, long_share=0.0):
        out = []
        while len(out) < n:
            k = maps.MAX_TURNS + 2 if rng.random() < long_share else int(rng.integers(1, 40))
            s, turns, states = random_walk(rng, k)
            if len(turns) == k:
                out.append(text(s, states[-1], turns))
        return out

    def shortest(n):
        out = []
        while len(out) < n:
            a, b = (int(x) for x in rng.integers(1, maps.N_NODES + 1, size=2))
            if a != b and 0 < D[a, b] <= 40 and maps.OUT_DEGREE[a - 1] > 0:
                out.append(text(a, b, shortest_route(a, b)))
        return out

    tok_new = maps.token_of_node(7732571914)
    a_new = next(x for x in range(1, maps.N_NODES + 1) if maps.OUT_DEGREE[x - 1] > 0 and 3 <= D[x, tok_new] <= 8)
    to_renamed = text(a_new, tok_new, shortest_route(a_new, tok_new))
    files = {"shortest-paths": (vt_s, nb_s, shortest(300) + [to_renamed], shortest(4000)),
             "noisy-shortest-paths": (vt_c, nb_c, shortest(300), shortest(4000)),
             "random-walks": (vt_c, nb_c, walks(300, 0.05), walks(4000, 0.05))}
    # train files also contain some held out pairs (these must be excluded from the caches)
    for name, (vt, nb, held, train) in files.items():
        d = tmp / name
        d.mkdir(parents=True)
        mod, cls = saver("__main__" if name == "random-walks" else "model")
        mod.SimpleTokenizer = cls
        tok = cls()
        tok.valid_turns = {u: list(t) for u, t in vt.items()}
        tok.node_and_direction_to_neighbor = nb
        torch.save(tok, d / "tokenizer.pt")
        del mod.SimpleTokenizer
        other = files["noisy-shortest-paths"][2][:50]
        (d / "heldout_sequences.txt").write_text("\n".join(held) + "\n")
        (d / "train_sequences.txt").write_text("\n".join(train + other) + "\n")
    # the shortest paths files use 42438066 for 7732571914
    for f in ("heldout_sequences.txt", "train_sequences.txt"):
        p = tmp / "shortest-paths" / f
        p.write_text(p.read_text().replace("7732571914", "42438066"))
    return tmp


def test_vafa_tokenizers_load_without_their_class(tmp_path: Path) -> None:
    vafa = _fake_vafa(tmp_path / "vafa")
    assert not hasattr(sys.modules["__main__"], "SimpleTokenizer") and not hasattr(sys.modules["model"], "SimpleTokenizer")
    for name in ("shortest-paths", "noisy-shortest-paths", "random-walks"):
        vt, nb = build_mod.vafa_map(name, str(vafa))
        assert len(nb) == maps.N_EDGES
    assert open(vafa / "random-walks" / "tokenizer.pt", "rb").read().find(b"__main__") >= 0


def test_build_splits_and_caches_end_to_end(tmp_path: Path, monkeypatch) -> None:
    vafa = _fake_vafa(tmp_path / "vafa")
    out = tmp_path / "maps"
    held = set()
    for name in ("shortest-paths", "noisy-shortest-paths", "random-walks"):
        for line in (vafa / name / "heldout_sequences.txt").read_text().split("\n"):
            if line:
                w = line.split()
                held.add((w[0].replace("42438066", "7732571914"), w[1].replace("42438066", "7732571914")))
    monkeypatch.setitem(build_mod.EXPECTED, "heldout_pairs", len(held))
    rep = build_mod.build_splits(str(vafa), str(out))
    pv, pt = maps.load_pairs("val", str(out)), maps.load_pairs("test", str(out))
    sv, st = set(map(tuple, pv.tolist())), set(map(tuple, pt.tolist()))
    assert not sv & st and len(sv) + len(st) == len(held) and abs(len(sv) - len(st)) <= 1
    assert rep["drops"]["random_walks_heldout"]["longer_than_max_tokens"] > 0
    for dist in maps.DISTRIBUTIONS:
        for split, pairs in (("val", sv), ("test", st)):
            arr = maps.load_distribution(dist, split, data_dir=str(out))
            d = maps.build(arr)
            assert d["n_illegal"] == 0 and d["length"].max() <= maps.MAX_TOKENS
            got = {(maps.node_of_token(int(a)), maps.node_of_token(int(b))) for a, b in arr[:, :2]}
            assert got <= pairs
    for dist in maps.DISTRIBUTIONS:
        r = build_mod.build_cache(dist, str(vafa), str(out), n_cache=1000)
        assert r["reasons"].get("val_or_test_pair", 0) >= 50     # the planted held out pairs were excluded
        cache = np.array(maps.load_distribution(dist, "train", data_dir=str(out)))   # a copy, not a memory map
        assert cache.shape == (1000, maps.MAX_TOKENS) and cache.dtype == np.uint16
        assert maps.build(np.asarray(cache, dtype=np.int64))["n_illegal"] == 0
        got = {(maps.node_of_token(int(a)), maps.node_of_token(int(b))) for a, b in cache[:, :2]}
        assert not got & (sv | st)
        again = build_mod.build_cache(dist, str(vafa), str(out), n_cache=1000)   # same seed, same cache
        assert np.array_equal(np.array(maps.load_distribution(dist, "train", data_dir=str(out))), cache)
        assert again["eligible"] == r["eligible"]
    # the renamed intersection is written with its canonical id in the shortest paths splits
    sp = np.concatenate([maps.load_distribution("shortest_paths", s, data_dir=str(out)) for s in ("val", "test")])
    assert maps.token_of_node(7732571914) in set(sp[:, 1].tolist())
    assert rep["routes"]["shortest_paths_val"] + rep["routes"]["shortest_paths_test"] == len(sp)


# ==========================================================================
# The built data (EC2 only)
# ==========================================================================

@needs_data
def test_data_files_match_the_manifest() -> None:
    bad = [f for f, ok in maps.verify_data().items() if not ok]
    assert bad == []
    man = json.loads(maps.MANIFEST.read_text())
    assert man["map_files"]["src/domains/maps_edges.csv"] == maps.sha256(maps.EDGES_CSV)
    assert man["map_files"]["src/domains/maps_coordinates.csv"] == maps.sha256(maps.COORDS_CSV)
    assert man["constants"]["MAX_TOKENS"] == maps.MAX_TOKENS and man["constants"]["VOCAB"] == maps.VOCAB


@needs_data
def test_splits_are_disjoint_and_legal() -> None:
    sv = set(map(tuple, maps.load_pairs("val").tolist()))
    st = set(map(tuple, maps.load_pairs("test").tolist()))
    assert not sv & st and len(sv) + len(st) == 107_331
    for dist in maps.DISTRIBUTIONS:
        for split, pairs in (("val", sv), ("test", st)):
            arr = maps.load_distribution(dist, split)
            got = {(maps.node_of_token(int(a)), maps.node_of_token(int(b))) for a, b in np.unique(arr[:, :2], axis=0)}
            assert got <= pairs, (dist, split)
            for i in range(0, len(arr), 100_000):
                assert maps.build(arr[i:i + 100_000])["n_illegal"] == 0, (dist, split)


@needs_data
def test_caches_hold_no_val_or_test_pair() -> None:
    held = set(map(tuple, maps.load_pairs("val").tolist())) | set(map(tuple, maps.load_pairs("test").tolist()))
    held_tok = {(maps.NODE_TO_TOKEN[a], maps.NODE_TO_TOKEN[b]) for a, b in held
                if a in maps.NODE_TO_TOKEN and b in maps.NODE_TO_TOKEN}
    for dist in maps.DISTRIBUTIONS:
        cache = maps.load_distribution(dist, "train")
        assert cache.shape == (maps.N_CACHE, maps.MAX_TOKENS)
        pairs = set(map(tuple, np.unique(np.asarray(cache[:, :2], dtype=np.int64), axis=0).tolist()))
        assert not pairs & held_tok, dist
        sample = np.asarray(cache[np.random.default_rng(0).choice(len(cache), 50_000, replace=False)], dtype=np.int64)
        assert maps.build(sample)["n_illegal"] == 0, dist
