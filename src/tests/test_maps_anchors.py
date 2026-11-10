"""Tests for the navigation anchors (Vafa et al. 2024 models, N2): the vocabulary translation, the
HFGPT2Model wrapper, weight extraction from a Lightning style checkpoint, and the reproduction code.

Everything here runs on small GPT 2 models built in the test, on the real canonical map. The released
checkpoints themselves only exist on the EC2 (src/diagnostics/maps_reproduce_vafa.py).

Run from the repo root:
    python -m pytest src/tests/test_maps_anchors.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

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
itf = _load("interface", "src/diagnostics/interface.py")
repro = _load("maps_repro", "src/diagnostics/maps_reproduce_vafa.py")


# ==========================================================================
# A small Vafa style world on the real map
# ==========================================================================

def _routes(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    with_turns = np.flatnonzero(maps.OUT_DEGREE > 0) + 1
    out = []
    while len(out) < n:
        cur = int(rng.choice(with_turns))
        start, turns = cur, []
        for _ in range(int(rng.integers(1, 30))):
            opts = [i for i in range(8) if maps.NEXT[cur, i] >= 0]
            if not opts:
                break
            i = int(rng.choice(opts))
            turns.append(maps.TURNS[i])
            cur = int(maps.NEXT[cur, i])
        out.append(" ".join([str(maps.node_of_token(start)), str(maps.node_of_token(cur))] + turns + ["end"]))
    return out


def _vocab(sentences: list[str]) -> dict[str, int]:
    """Vafa's SimpleTokenizer: words reverse sorted, ids from 1, <pad> = 0."""
    words = set()
    for s in sentences:
        words.update(s.split())
    w2i = {w: i + 1 for i, w in enumerate(sorted(words, reverse=True))}
    w2i["<pad>"] = 0
    return w2i


@pytest.fixture(scope="module")
def world():
    routes = _routes(300, seed=0)
    w2i = _vocab(routes)
    config = {"vocab_size": len(w2i), "n_embd": 32, "n_layer": 2, "n_head": 4}
    torch.manual_seed(0)
    module = itf.build_gpt2(config).eval()
    return routes, w2i, config, module


# ==========================================================================
# Vocabulary translation
# ==========================================================================

def test_vocab_maps_round_trip(world) -> None:
    routes, w2i, _, _ = world
    in_map, out_map = itf.vafa_vocab_maps(w2i, "random_walks")
    assert in_map[0] == 0 and out_map[0] == 0
    for t in range(1, maps.VOCAB):
        word = str(maps.node_of_token(t)) if maps.is_node(t) else maps.token_name(t)
        if word in w2i:
            assert in_map[t] == out_map[t] == w2i[word]
        else:
            assert in_map[t] == 0 and out_map[t] == -1          # unknown: its padding in, never predicted
    # every word of its vocabulary except <pad> is reachable from exactly one of our tokens
    ids = [out_map[t] for t in range(1, maps.VOCAB) if out_map[t] > 0]
    assert sorted(ids) == sorted(i for w, i in w2i.items() if w != "<pad>")


def test_shortest_paths_rename_in_vocab_maps() -> None:
    w2i = {"42438066": 3, "end": 2, "N": 1, "<pad>": 0}
    in_map, out_map = itf.vafa_vocab_maps(w2i, "shortest_paths")
    t = maps.token_of_node(7732571914)
    assert in_map[t] == out_map[t] == 3
    in_map2, out_map2 = itf.vafa_vocab_maps(w2i, "random_walks")
    assert out_map2[t] == -1                                       # other tokenizers use the canonical id


# ==========================================================================
# The wrapper
# ==========================================================================

def _wrapped(world):
    routes, w2i, config, module = world
    in_map, out_map = itf.vafa_vocab_maps(w2i, "random_walks")
    return itf.HFGPT2Model(module, "maps_vafa_test", config["n_layer"], config["n_embd"], in_map, out_map, "cpu")


def test_wrapper_logits_equal_the_plain_model(world) -> None:
    routes, w2i, config, module = world
    m = _wrapped(world)
    data = maps.build([maps.encode(r) for r in routes[:40]])
    ours = torch.tensor(data["tokens"][:, :maps.N_CTX])
    lg = m.logits(ours)
    assert lg.shape == (40, maps.N_CTX, maps.VOCAB)
    ids = torch.tensor([[w2i[w] for w in r.split()] + [0] * (maps.N_CTX - len(r.split())) for r in routes[:40]])
    ids = ids[:, :maps.N_CTX]
    with torch.no_grad():
        plain = module(input_ids=ids, attention_mask=(ids != 0).long()).logits
    real = torch.tensor(data["score"][:, :maps.N_CTX])
    _, out_map = itf.vafa_vocab_maps(w2i, "random_walks")
    ours_idx = [t for t in range(1, maps.VOCAB) if out_map[t] > 0]
    theirs_idx = [out_map[t] for t in ours_idx]
    assert torch.equal(lg[..., ours_idx][real], plain[..., theirs_idx][real])   # same numbers, reordered
    # the top token agrees wherever it is scored (no unknown token can win: their logits are -1e9)
    i2w = {i: w for w, i in w2i.items()}
    top_ours, top_plain = lg.argmax(-1), plain.argmax(-1)
    for g, p in zip(*np.nonzero(real.numpy())):
        assert maps.token_name(int(top_ours[g, p])) == i2w[int(top_plain[g, p])]


def test_wrapper_activations_and_edits(world) -> None:
    routes, w2i, config, module = world
    m = _wrapped(world)
    data = maps.build([maps.encode(r) for r in routes[:16]])
    tok = torch.tensor(data["tokens"][:, :maps.N_CTX])
    acts = m.activations(tok)
    assert acts.shape == (config["n_layer"], 16, maps.N_CTX, config["n_embd"])
    ids, mask = m._inputs(tok)
    with torch.no_grad():
        last = module.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(module.transformer.ln_f(acts[-1]), last, atol=1e-5)   # residual before the final norm
    pos = torch.full((16,), 3)
    same = m.logits_edited(tok, pos, {0: lambda v: v})
    assert torch.allclose(same, m.logits(tok), atol=1e-5)
    changed = m.logits_patched(tok, 1, pos, torch.zeros(16, config["n_embd"]))
    assert not torch.allclose(changed[:, 3], m.logits(tok)[:, 3])
    assert torch.allclose(changed[:, :3], m.logits(tok)[:, :3], atol=1e-5)        # causal: earlier positions unchanged


def test_unknown_words_are_flagged_and_never_predicted(world) -> None:
    routes, w2i, config, module = world
    m = _wrapped(world)
    missing = next(t for t in range(1, maps.N_NODES + 1) if str(maps.node_of_token(t)) not in w2i)
    tok = torch.tensor(maps.build([maps.encode(r) for r in routes[:4]])["tokens"][:, :maps.N_CTX])
    assert m.known(tok).all()
    bad = tok.clone()
    bad[0, 1] = missing
    assert m.known(bad).tolist() == [False, True, True, True]
    assert (m.logits(tok)[..., missing] <= itf.HFGPT2Model.NEG).all()


# ==========================================================================
# Weight extraction from a Lightning style checkpoint
# ==========================================================================

def test_prepare_reads_a_lightning_checkpoint(world, tmp_path: Path, monkeypatch) -> None:
    routes, w2i, config, module = world
    # classes the real checkpoint pickles: model.SimpleTokenizer and a Lightning class not installed here
    model_mod = sys.modules.setdefault("model", types.ModuleType("model"))
    tok_cls = type("SimpleTokenizer", (), {"__module__": "model", "__qualname__": "SimpleTokenizer"})
    model_mod.SimpleTokenizer = tok_cls
    pl = types.ModuleType("pytorch_lightning_fake_for_test")
    ad = type("AttributeDict", (dict,), {"__module__": pl.__name__, "__qualname__": "AttributeDict"})
    pl.AttributeDict = ad
    sys.modules[pl.__name__] = pl
    tok = tok_cls()
    tok.word_to_id = dict(w2i)
    tok.valid_turns = {}
    tok.node_and_direction_to_neighbor = {}
    ckpt = {"state_dict": {f"model.{k}": v for k, v in module.state_dict().items()},
            "optimizer_states": [{"state": {0: {"exp_avg": torch.zeros(3)}}}],
            "loops": {"fit_loop": ad(state="finished")},
            "hyper_parameters": {"tokenizer": tok, **config}}
    vafa = tmp_path / "vafa"
    (vafa / "ckpts" / "random-walks").mkdir(parents=True)
    (vafa / "data" / "random-walks").mkdir(parents=True)
    torch.save(ckpt, vafa / "ckpts" / "random-walks" / "model.ckpt")
    torch.save(tok, vafa / "data" / "random-walks" / "tokenizer.pt")
    del model_mod.SimpleTokenizer, sys.modules[pl.__name__]       # neither class exists when reading back
    info = repro.prepare("random_walks", str(vafa), str(tmp_path / "weights"))
    assert info["config"]["n_layer"] == 2 and info["config"]["n_embd"] == 32 and info["config"]["n_head"] == 4
    assert info["tokenizer_in_checkpoint_matches_file"] is True and info["dropped_buffers"] == []
    m = itf.vafa_navigation("random_walks", "cpu", str(tmp_path / "weights"))
    data = maps.build([maps.encode(r) for r in routes[:20]])
    tok_t = torch.tensor(data["tokens"][:, :maps.N_CTX])
    assert torch.allclose(m.logits(tok_t), _wrapped(world).logits(tok_t))
    assert m.model_id == "maps_vafa_random_walks" and m.meta["domain"] == "maps"


# ==========================================================================
# Reproduction code
# ==========================================================================

def test_vafa_next_token_equals_our_d1_path(world, tmp_path: Path, monkeypatch) -> None:
    routes, w2i, config, module = world
    tok = types.SimpleNamespace(
        word_to_id=w2i, id_to_word={i: w for w, i in w2i.items()}, pad=0,
        valid_turns={maps.node_of_token(u): [maps.TURNS[i] for i in range(8) if maps.NEXT[u, i] >= 0]
                     for u in range(1, maps.N_NODES + 1)},
        node_and_direction_to_neighbor={(maps.node_of_token(u), maps.TURNS[i]): maps.node_of_token(int(maps.NEXT[u, i]))
                                        for u in range(1, maps.N_NODES + 1) for i in range(8) if maps.NEXT[u, i] >= 0})
    tok.encode = lambda s: [w2i.get(w, 0) for w in s.split()]
    tok.decode = lambda ids: " ".join(tok.id_to_word[int(i)] for i in np.asarray(ids).ravel() if int(i) != 0)
    seqs = routes[:200] + [""]                                     # plus the empty entry after a final newline
    v = repro.next_token_vafa(module, tok, seqs, "cpu")
    assert v["n_empty_skipped"] == 1 and v["n_routes"] == 200
    m = _wrapped(world)
    data = maps.build([maps.encode(r) for r in routes[:200]])
    ours = maps.legal_rate(m, data, "cpu", 64)
    assert ours == pytest.approx(v["rate"], abs=1e-12)
    assert v["n_positions"] == int(data["score"].sum())


def test_probe_trainer_and_feature_alignment() -> None:
    rng = np.random.default_rng(0)
    K, d, n = 300, 16, 6000
    means = rng.normal(size=(K, d)).astype(np.float32)
    y = 42_000_000 + 7 * rng.integers(0, K, n)
    X = means[(y - 42_000_000) // 7] + 0.05 * rng.normal(size=(n, d)).astype(np.float32)
    out = repro.train_probe(X, y, "cpu")                          # the real settings: 100 epochs, lr 0.01
    assert out["accuracy"] > 0.98 and out["n_classes"] == K and out["n_train"] == int(0.8 * n)
    # states: the start, then one intersection per turn; end adds none
    nb = {(1, "N"): 2, (2, "E"): 3}
    assert repro.state_sequence(["1", "3", "N", "E", "end"], nb) == [1, 2, 3]


def _vafa_tok(w2i):
    tok = types.SimpleNamespace(
        word_to_id=w2i, id_to_word={i: w for w, i in w2i.items()}, pad=0,
        valid_turns={maps.node_of_token(u): [maps.TURNS[i] for i in range(8) if maps.NEXT[u, i] >= 0]
                     for u in range(1, maps.N_NODES + 1)},
        node_and_direction_to_neighbor={(maps.node_of_token(u), maps.TURNS[i]): maps.node_of_token(int(maps.NEXT[u, i]))
                                        for u in range(1, maps.N_NODES + 1) for i in range(8) if maps.NEXT[u, i] >= 0})
    tok.encode = lambda s: [w2i.get(w, 0) for w in s.split()]
    tok.decode = lambda ids: " ".join(tok.id_to_word[int(i)] for i in np.asarray(ids).ravel() if int(i) != 0)
    return tok


class _AlwaysEnd(torch.nn.Module):
    """Predicts end everywhere: legal exactly where the car is at its destination."""

    def __init__(self, vocab_size: int, end_id: int) -> None:
        super().__init__()
        self.v, self.e = vocab_size, end_id

    def forward(self, input_ids, attention_mask=None):
        lg = torch.zeros(*input_ids.shape, self.v)
        lg[..., self.e] = 1.0
        return types.SimpleNamespace(logits=lg)


def test_end_counts_only_at_the_destination(world) -> None:
    routes, w2i, config, _ = world
    tok = _vafa_tok(w2i)
    model = _AlwaysEnd(len(w2i), w2i["end"])
    v = repro.next_token_vafa(model, tok, routes[:150], "cpu")
    data = maps.build([maps.encode(r) for r in routes[:150]])
    at_dest = ((data["current"] == data["dest"][:, None]) & data["score"]).sum() / data["score"].sum()
    assert v["rate"] == pytest.approx(at_dest, abs=1e-12) and 0 < at_dest < 0.5
    in_map, out_map = itf.vafa_vocab_maps(w2i, "random_walks")
    m = itf.HFGPT2Model(model, "end", 1, 1, in_map, out_map, "cpu")
    assert maps.legal_rate(m, data, "cpu", 64) == pytest.approx(at_dest, abs=1e-12)


class _StateVectors:
    """last_hidden_state at position p = a fixed vector for the intersection after reading tokens 0..p."""

    def __init__(self, tok, d: int = 8) -> None:
        self.tok, self.config, self.rng, self.vec = tok, types.SimpleNamespace(n_embd=d), np.random.default_rng(0), {}

    def v(self, node: int) -> np.ndarray:
        if node not in self.vec:
            self.vec[node] = self.rng.normal(size=self.config.n_embd).astype(np.float32)
        return self.vec[node]

    def transformer(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        out = np.zeros((B, L, self.config.n_embd), np.float32)
        for b in range(B):
            words = [self.tok.id_to_word[int(i)] for i in input_ids[b] if int(i) != 0]
            cur = int(words[0])
            for p, w in enumerate(words):
                if p >= 2 and w != "end":
                    cur = self.tok.node_and_direction_to_neighbor[(cur, w)]
                out[b, p] = self.v(cur)
        return types.SimpleNamespace(last_hidden_state=torch.tensor(out))


def test_probe_features_pair_each_vector_with_its_intersection(world) -> None:
    routes, w2i, _, _ = world
    tok = _vafa_tok(w2i)
    model = _StateVectors(tok)
    X, y, skipped = repro.probe_features(model, tok, routes[:60] + [""], "cpu")
    assert skipped == 1
    assert len(X) == sum(1 + sum(w != "end" for w in r.split()[2:]) for r in routes[:60])
    assert all(np.array_equal(X[i], model.v(int(y[i]))) for i in range(len(X)))
