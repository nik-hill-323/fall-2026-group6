"""Navigation reproduction gate (N2, design section 6.3): Vafa et al. (2024), Table 1, on their released models.

Before any navigation zoo model is scored, our code must reproduce two published numbers on the three
released checkpoints (shortest paths, noisy shortest paths, random walks):

    next token test        1.00, 1.00, 1.00    share of held out positions where the top token is legal
    current state probe    0.91, 0.92, 0.99    linear probe from the last layer to the current intersection
    (untrained transformer 0.03 and 0.10, optional context row)

Tolerances, fixed before any comparison (as for the Othello gate, src/docs/d4_reproduction.md):
    next token test        at least 0.99 for every model (the paper rounds to 1.00; footnote 3: no model is perfect)
    our pipeline           on the same routes, our D1 path (maps.build, the HFGPT2Model translation, maps.legal_rate)
                           within 0.001 of Vafa's computation, for the two models on the canonical map
    current state probe    within 0.02 of the published value

What "Vafa's computation" means here: his code, copied step by step (next_token_test.py, probe_test.py,
utils.py, model.py of github.com/keyonvafa/world-model-evaluation), run on his data and his tokenizers:
    next token   heldout_sequences.txt read with f.read().split("\\n") (a final newline adds one empty entry,
                 which counts in the length), 1,000 routes drawn with np.random.RandomState(42).choice(len,
                 1000, replace=False), batches of 128 padded on the right, top token at every position that
                 predicts a turn or end, legal if it is a turn of the current intersection or end at the
                 destination
    probe        25,000 routes drawn the same way from train_sequences.txt (streamed here, same indices), last
                 layer (after GPT 2's final layer norm), the start at position 0 and the intersection after
                 turn j at position j + 1, LabelEncoder over all labels, a random 80/20 split of positions
                 (unseeded in his code; seed 0 here, reported), one linear layer, Adam lr 0.01, cross entropy,
                 batch 2,048 in order, 100 epochs, accuracy on the 20% after the last epoch

Steps, on the EC2 from the repo root (conda env wmconcord), inside tmux:
    python src/diagnostics/maps_reproduce_vafa.py prepare        # weights out of the 17 GB checkpoints, once
    python src/diagnostics/maps_reproduce_vafa.py next-token
    python src/diagnostics/maps_reproduce_vafa.py probe
    python src/diagnostics/maps_reproduce_vafa.py report         # table against the paper, pass or fail
Options: --anchors (default all three), --untrained (also an untrained copy, the paper's context row).
next-token, probe and report refuse to run while files under src/ are not committed.

Outputs: ~/artifacts/vafa_weights/{anchor}/ (model.pt, config.json, vocab.json, checksums.json);
results/reproduction/maps/{anchor}_{test}.json and results/reproduction/maps/summary.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("maps_repro")

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
mb = _load("maps_build", "src/domains/maps_build.py")
interface = _load("interface", "src/diagnostics/interface.py")

VAFA_ROOT = os.environ.get("WMC_VAFA_ROOT", os.path.expanduser("~/artifacts/world-model-evaluation"))
WEIGHTS_DIR = interface.VAFA_WEIGHTS_DIR
OUT_DIR = ROOT / "results" / "reproduction" / "maps"
ANCHORS = list(interface.VAFA_ANCHORS)
PUBLISHED = {  # Vafa et al. 2024, Table 1
    "next_token": {"shortest_paths": 1.00, "noisy_shortest_paths": 1.00, "random_walks": 1.00, "untrained": 0.03},
    "probe": {"shortest_paths": 0.91, "noisy_shortest_paths": 0.92, "random_walks": 0.99, "untrained": 0.10},
}
TOL = {"next_token_min": 0.99, "pipeline": 0.001, "probe": 0.02}
N_NEXT_TOKEN, NEXT_TOKEN_SEED, BATCH = 1000, 42, 128
N_PROBE, PROBE_SEED, PROBE_SPLIT_SEED, PROBE_INIT_SEED = 25_000, 42, 0, 0
PROBE_LR, PROBE_EPOCHS, PROBE_BATCH = 0.01, 100, 2048
UNTRAINED_SEED = 42  # load_model(use_untrained_model=True) seeds torch with 42 before building the model


# ==========================================================================
# Step 1: weights out of the released Lightning checkpoints
# ==========================================================================

class _Anything(dict):
    """Stand in for any class the checkpoint pickles that is not installed here (Lightning bookkeeping)."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        super().__init__()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)

    def __call__(self, *args, **kwargs):
        return _Anything()


class _CheckpointUnpickler(mb._VafaUnpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            log.info("  checkpoint class %s.%s is not installed here: read as a plain dict", module, name)
            return _Anything


_ckpt_pickle = type(sys)("vafa_ckpt_pickle")
_ckpt_pickle.__dict__.update(mb._vafa_pickle.__dict__)
_ckpt_pickle.Unpickler = _CheckpointUnpickler


def vafa_name(anchor: str) -> str:
    return maps.VAFA_NAMES[anchor]


def prepare(anchor: str, vafa_root: str = VAFA_ROOT, out_dir: str = WEIGHTS_DIR) -> dict:
    """Read the checkpoint with mmap (the optimizer state, about two thirds of the 17 GB, is never loaded),
    keep the GPT 2 weights, check they load into a freshly built GPT2LMHeadModel with nothing missing or
    left over, and save them with the tokenizer's vocabulary."""
    name = vafa_name(anchor)
    path = Path(vafa_root) / "ckpts" / name / "model.ckpt"
    t0 = time.time()
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False, pickle_module=_ckpt_pickle)
    sd = ckpt["state_dict"]
    weights = {k[len("model."):]: v.detach().clone().float() for k, v in sd.items() if k.startswith("model.")}
    other = [k for k in sd if not k.startswith("model.")]
    dropped = [k for k in weights if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias")]
    for k in dropped:  # causal mask buffers of older transformers versions, rebuilt by the model itself
        del weights[k]
    wte = weights["transformer.wte.weight"]
    n_layer = 1 + max(int(k.split(".")[2]) for k in weights if k.startswith("transformer.h."))
    hp = dict(ckpt.get("hyper_parameters", {}) or {})
    config = {"vocab_size": int(wte.shape[0]), "n_embd": int(wte.shape[1]), "n_layer": n_layer,
              "n_head": int(hp.get("n_head", {768: 12, 1600: 25}.get(int(wte.shape[1]), 0))),
              "n_positions": int(weights["transformer.wpe.weight"].shape[0])}
    for k in ("vocab_size", "n_embd", "n_layer"):
        if k in hp and int(hp[k]) != config[k]:
            raise SystemExit(f"{anchor}: hyper_parameters say {k} = {hp[k]}, the weights say {config[k]}")
    if config["n_head"] <= 0 or config["n_embd"] % config["n_head"]:
        raise SystemExit(f"{anchor}: cannot tell the number of heads ({config})")
    model = interface.build_gpt2(config)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if k != "lm_head.weight" or "lm_head.weight" in weights]
    if missing or unexpected:
        raise SystemExit(f"{anchor}: weights do not fit GPT2LMHeadModel: missing {missing}, unexpected {unexpected}")
    tok = mb.load_vafa_tokenizer(str(Path(vafa_root) / "data" / name / "tokenizer.pt"))
    word_to_id = {str(w): int(i) for w, i in tok.word_to_id.items()}
    if max(word_to_id.values()) + 1 != config["vocab_size"]:
        raise SystemExit(f"{anchor}: tokenizer has {len(word_to_id)} words, the model {config['vocab_size']} rows")
    ck_tok = hp.get("tokenizer")
    same_tok = getattr(ck_tok, "word_to_id", None) == tok.word_to_id if ck_tok is not None else None
    folder = Path(out_dir) / anchor
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (folder / "vocab.json").write_text(json.dumps(word_to_id, sort_keys=True) + "\n")
    info = {"anchor": anchor, "checkpoint": str(path), "config": config, "n_tensors": len(weights),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "dropped_buffers": dropped, "other_state_dict_keys": other,
            "tokenizer_in_checkpoint_matches_file": same_tok,
            "files": {f: maps.sha256(folder / f) for f in ("model.pt", "config.json", "vocab.json")},
            "seconds": round(time.time() - t0, 1)}
    (folder / "checksums.json").write_text(json.dumps(info, indent=2) + "\n")
    log.info("%s: %d layers x %d, %d parameters, saved to %s (%.0fs)", anchor, n_layer, config["n_embd"],
             info["n_parameters"], folder, info["seconds"])
    return info


# ==========================================================================
# Vafa's data, read exactly as his code reads it
# ==========================================================================

def heldout_sample(anchor: str, n: int = N_NEXT_TOKEN, seed: int = NEXT_TOKEN_SEED,
                   vafa_root: str = VAFA_ROOT) -> list[str]:
    """load_heldout_data(): f.read().split("\\n"), then RandomState(seed).choice(len, n, replace=False)."""
    with open(Path(vafa_root) / "data" / vafa_name(anchor) / "heldout_sequences.txt") as f:
        seqs = f.read().split("\n")
    idx = np.random.RandomState(seed).choice(len(seqs), n, replace=False)
    return [seqs[i] for i in idx]


def train_sample(anchor: str, n: int = N_PROBE, seed: int = PROBE_SEED, vafa_root: str = VAFA_ROOT) -> list[str]:
    """load_train_data(num_samples=n): the same draw over f.read().split("\\n"), without reading the file into
    memory (the random walks file is 14.6 GB): the entry count is the number of newlines plus one."""
    path = Path(vafa_root) / "data" / vafa_name(anchor) / "train_sequences.txt"
    n_newlines = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 26), b""):
            n_newlines += chunk.count(b"\n")
    total = n_newlines + 1
    idx = np.random.RandomState(seed).choice(total, n, replace=False)
    want = {int(i): k for k, i in enumerate(idx)}
    out: list[str | None] = [None] * n
    with open(path) as f:
        for i, line in enumerate(f):
            k = want.get(i)
            if k is not None:
                out[k] = line[:-1] if line.endswith("\n") else line
    for i, k in want.items():  # only the entry after a final newline is never read as a line: it is ""
        if out[k] is None:
            out[k] = ""
    log.info("%s train: %d entries, drew %d (seed %d)", anchor, total, n, seed)
    return out  # type: ignore[return-value]


class VafaTokenizer:
    """The parts of Vafa's SimpleTokenizer the tests use, read from his tokenizer.pt."""

    def __init__(self, anchor: str, vafa_root: str = VAFA_ROOT) -> None:
        tok = mb.load_vafa_tokenizer(str(Path(vafa_root) / "data" / vafa_name(anchor) / "tokenizer.pt"))
        self.word_to_id = dict(tok.word_to_id)
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.valid_turns = tok.valid_turns
        self.node_and_direction_to_neighbor = tok.node_and_direction_to_neighbor
        self.pad = 0

    def encode(self, sentence: str) -> list[int]:
        return [self.word_to_id.get(w, self.pad) for w in sentence.split()]

    def decode(self, ids) -> str:
        ids = np.asarray(ids).ravel()
        return " ".join(self.id_to_word[int(i)] for i in ids if int(i) != self.pad)


def collate(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """model.collate_fn: pad on the right with 0; attention mask 1 on every real token."""
    L = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[b, :len(s)] = 1
    return ids, mask


def load_anchor_module(anchor: str, untrained: bool, device: str) -> nn.Module:
    folder = Path(WEIGHTS_DIR) / anchor
    config = json.loads((folder / "config.json").read_text())
    if untrained:
        torch.manual_seed(UNTRAINED_SEED)
        model = interface.build_gpt2(config)
    else:
        model = interface.build_gpt2(config)
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


# ==========================================================================
# Next token test
# ==========================================================================

@torch.no_grad()
def next_token_vafa(model: nn.Module, tok: VafaTokenizer, seqs: list[str], device: str) -> dict:
    """next_token_test.py, step by step. An empty entry (the one after a final newline) would stop his code
    with an error; it is skipped here and counted, and contributes (0, 0) to per_route."""
    keep = [k for k, s in enumerate(seqs) if s.strip()]
    encoded = [tok.encode(seqs[k]) for k in keep]
    ok = total = 0
    per_route = [(0, 0)] * len(seqs)
    for i in range(0, len(encoded), BATCH):
        ids, mask = collate(encoded[i:i + BATCH])
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        top = logits.argmax(-1).cpu()
        for b in range(len(ids)):
            words = tok.decode(ids[b]).split(" ")
            start, end_node = int(words[0]), int(words[1])
            cur = start
            r_ok = r_total = 0
            for L in range(2, len(words)):
                pred = tok.decode(top[b, L - 1])
                r_total += 1
                if pred in tok.valid_turns[cur]:
                    r_ok += 1
                elif pred == "end" and cur == end_node:
                    r_ok += 1
                if words[L] != "end":
                    cur = tok.node_and_direction_to_neighbor[(cur, words[L])]
            ok += r_ok
            total += r_total
            per_route[keep[i + b]] = (r_ok, r_total)
    p = ok / total
    return {"rate": p, "se": math.sqrt(p * (1 - p) / total), "n_ok": ok, "n_positions": total,
            "n_routes": len(keep), "n_empty_skipped": len(seqs) - len(keep), "per_route": per_route}


def next_token_ours(anchor: str, model: nn.Module, seqs: list[str], vafa_rate_of, device: str) -> dict:
    """The same routes through our D1 path: canonical map text (shortest paths routes rewritten by
    maps_build.Translator), maps.encode, maps.build, HFGPT2Model, maps.legal_rate. Routes our domain does
    not keep (a street missing from the canonical map, longer than 99 tokens) are left out of both sides."""
    folder = Path(WEIGHTS_DIR) / anchor
    word_to_id = json.loads((folder / "vocab.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    in_map, out_map = interface.vafa_vocab_maps(word_to_id, anchor)
    wrapped = interface.HFGPT2Model(model, f"maps_vafa_{anchor}", config["n_layer"], config["n_embd"],
                                    in_map, out_map, device)
    tr = mb.Translator(anchor)
    keep, routes, reasons = [], [], {}
    for k, s in enumerate(seqs):
        text, why = tr(s)
        reasons[why] = reasons.get(why, 0) + 1
        if why == "ok":
            keep.append(k)
            routes.append(maps.encode(text))
    data = maps.build(routes)
    if data["n_illegal"]:
        raise SystemExit(f"{anchor}: {data['n_illegal']} routes are not legal on the canonical map")
    known = wrapped.known(torch.tensor(data["tokens"])).cpu().numpy()
    ours = maps.legal_rate(wrapped, data, device, BATCH)
    theirs = vafa_rate_of(keep)
    return {"rate": ours, "vafa_rate_same_routes": theirs, "difference": ours - theirs,
            "n_routes": len(keep), "translator": reasons, "routes_with_words_unknown_to_the_model": int((~known).sum())}


def run_next_token(anchor: str, untrained: bool, device: str) -> dict:
    t0 = time.time()
    tok = VafaTokenizer(anchor)
    seqs = heldout_sample(anchor)
    model = load_anchor_module(anchor, untrained, device)
    v = next_token_vafa(model, tok, seqs, device)
    out = {"anchor": anchor, "untrained": untrained, "test": "next_token",
           "published": PUBLISHED["next_token"]["untrained" if untrained else anchor],
           "vafa": {k: v[k] for k in ("rate", "se", "n_ok", "n_positions", "n_routes", "n_empty_skipped")}}

    def vafa_rate_of(keep: list[int]) -> float:
        ok = sum(v["per_route"][k][0] for k in keep)
        tot = sum(v["per_route"][k][1] for k in keep)
        return ok / max(tot, 1)

    if not untrained:
        out["ours"] = next_token_ours(anchor, model, seqs, vafa_rate_of, device)
    out["seconds"] = round(time.time() - t0, 1)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    log.info("%s%s next token: %.4f (published %.2f)%s", anchor, " untrained" if untrained else "",
             v["rate"], out["published"],
             "" if untrained else f"; ours {out['ours']['rate']:.4f} vs Vafa's {out['ours']['vafa_rate_same_routes']:.4f} on the same {out['ours']['n_routes']} routes")
    return out


# ==========================================================================
# Current state probe
# ==========================================================================

def state_sequence(words: list[str], nb: dict) -> list[int]:
    """utils.get_state_sequence."""
    cur = int(words[0])
    out = [cur]
    for m in words[2:]:
        if m != "end":
            cur = nb[(cur, m)]
            out.append(cur)
    return out


@torch.no_grad()
def probe_features(model: nn.Module, tok: VafaTokenizer, seqs: list[str], device: str) -> tuple[np.ndarray, np.ndarray, int]:
    """probe_test.py: last layer vectors and current intersection labels, in his order."""
    encoded, skipped = [], 0
    for s in seqs:
        if not s.strip():
            skipped += 1  # his code would stop with an error on an empty entry
            continue
        encoded.append(tok.encode(s))
    decoded = [tok.decode(e).split(" ") for e in encoded]
    states = [state_sequence(w, tok.node_and_direction_to_neighbor) for w in decoded]
    n = sum(len(s) for s in states)
    d = model.config.n_embd
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    r = 0
    for i in range(0, len(encoded), BATCH):
        ids, mask = collate(encoded[i:i + BATCH])
        h = model.transformer(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        h = h.float().cpu().numpy()
        for b in range(len(ids)):
            st = states[i + b]
            pos = [0] + list(range(2, len(st) + 1))   # state 0 at position 0, state j at position j + 1
            X[r:r + len(st)] = h[b, pos]
            y[r:r + len(st)] = st
            r += len(st)
    assert r == n
    return X, y, skipped


def train_probe(X: np.ndarray, y: np.ndarray, device: str) -> dict:
    """probe_test.py: LabelEncoder, 80/20 split of positions, multinomial logistic regression, Adam lr 0.01,
    batch 2,048 in order, 100 epochs; accuracy on the 20% after every epoch."""
    classes, enc = np.unique(y, return_inverse=True)     # LabelEncoder: sorted classes, same codes
    rs = np.random.RandomState(PROBE_SPLIT_SEED)
    train_inds = rs.choice(len(X), size=int(0.8 * len(X)), replace=False)
    test_inds = np.setdiff1d(np.arange(len(X)), train_inds)
    # one copy of the features on the device (8.5 GB for the 1.5B models); batches are gathered from it
    Xg = torch.from_numpy(X).to(device)
    yg = torch.from_numpy(enc).to(device)
    tr = torch.from_numpy(train_inds).to(device)
    te = torch.from_numpy(test_inds).to(device)
    torch.manual_seed(PROBE_INIT_SEED)
    net = nn.Linear(X.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()
    curve = []
    for epoch in range(PROBE_EPOCHS):
        for i in range(0, len(tr), PROBE_BATCH):
            b = tr[i:i + PROBE_BATCH]
            loss = loss_fn(net(Xg[b]), yg[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = sum(int((net(Xg[te[i:i + 65536]]).argmax(-1) == yg[te[i:i + 65536]]).sum())
                      for i in range(0, len(te), 65536)) / len(te)
        curve.append(acc)
    acc = curve[-1]
    return {"accuracy": acc, "se": math.sqrt(acc * (1 - acc) / len(te)), "best_epoch_accuracy": max(curve),
            "curve_every_10_epochs": curve[9::10], "n_positions": int(len(X)), "n_train": int(len(tr)),
            "n_test": int(len(te)), "n_classes": int(len(classes)), "split_seed": PROBE_SPLIT_SEED,
            "init_seed": PROBE_INIT_SEED}


def run_probe(anchor: str, untrained: bool, device: str) -> dict:
    t0 = time.time()
    tok = VafaTokenizer(anchor)
    seqs = train_sample(anchor)
    model = load_anchor_module(anchor, untrained, device)
    X, y, skipped = probe_features(model, tok, seqs, device)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    log.info("%s%s probe: %d positions x %d, %d classes", anchor, " untrained" if untrained else "",
             X.shape[0], X.shape[1], len(np.unique(y)))
    res = train_probe(X, y, device)
    out = {"anchor": anchor, "untrained": untrained, "test": "probe",
           "published": PUBLISHED["probe"]["untrained" if untrained else anchor],
           "vafa_recipe": res, "empty_routes_skipped": skipped, "seconds": round(time.time() - t0, 1)}
    log.info("%s%s probe: %.4f (published %.2f), %.0fs", anchor, " untrained" if untrained else "",
             res["accuracy"], out["published"], out["seconds"])
    return out


# ==========================================================================
# Report
# ==========================================================================

def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _save(res: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res["provenance"] = {"git_commit": _commit(), "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                         "torch": torch.__version__, "python": platform.python_version(),
                         "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import transformers
        res["provenance"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    tag = "_untrained" if res["untrained"] else ""
    path = OUT_DIR / f"{res['anchor']}{tag}_{res['test']}.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    return path


def report() -> dict:
    rows, passed = [], True
    for a in ANCHORS:
        nt = OUT_DIR / f"{a}_next_token.json"
        pr = OUT_DIR / f"{a}_probe.json"
        row = {"anchor": a}
        if nt.exists():
            r = json.loads(nt.read_text())
            row["next_token"] = r["vafa"]["rate"]
            row["next_token_pass"] = r["vafa"]["rate"] >= TOL["next_token_min"]
            if a != "shortest_paths":
                row["pipeline_difference"] = r["ours"]["difference"]
                row["pipeline_pass"] = abs(r["ours"]["difference"]) <= TOL["pipeline"]
            else:
                row["pipeline_difference"] = r["ours"]["difference"]
                row["pipeline_pass"] = None  # different map: described, not a gate
        if pr.exists():
            r = json.loads(pr.read_text())
            row["probe"] = r["vafa_recipe"]["accuracy"]
            row["probe_pass"] = abs(r["vafa_recipe"]["accuracy"] - PUBLISHED["probe"][a]) <= TOL["probe"]
        for k in ("next_token_pass", "pipeline_pass", "probe_pass"):
            if row.get(k) is False or (k != "pipeline_pass" and k not in row):
                passed = False
        rows.append(row)
    extra = {}
    for a in ANCHORS:
        for t in ("next_token", "probe"):
            p = OUT_DIR / f"{a}_untrained_{t}.json"
            if p.exists():
                r = json.loads(p.read_text())
                extra[f"{a}_untrained_{t}"] = r["vafa"]["rate"] if t == "next_token" else r["vafa_recipe"]["accuracy"]
    summary = {"passed": passed, "tolerances": TOL, "published": PUBLISHED, "rows": rows, "untrained": extra,
               "git_commit": _commit()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Model | Next token: ours (published) | Our pipeline minus Vafa's | Probe: ours (published) |",
             "|---|---|---|---|"]
    for row in rows:
        a = row["anchor"]
        nt = f"{row['next_token']:.4f} ({PUBLISHED['next_token'][a]:.2f})" if "next_token" in row else "not run"
        pd_ = f"{row['pipeline_difference']:+.4f}" if "pipeline_difference" in row else "not run"
        pb = f"{row['probe']:.4f} ({PUBLISHED['probe'][a]:.2f})" if "probe" in row else "not run"
        lines.append(f"| {a} | {nt} | {pd_} | {pb} |")
    lines.append("")
    lines.append(f"Gate: **{'passed' if passed else 'not passed'}** (next token at least {TOL['next_token_min']}, "
                 f"pipeline within {TOL['pipeline']} on the canonical map models, probe within {TOL['probe']}).")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce Vafa et al. 2024 Table 1 on the released navigation models")
    ap.add_argument("step", choices=["prepare", "next-token", "probe", "report"])
    ap.add_argument("--anchors", nargs="*", default=ANCHORS, choices=ANCHORS)
    ap.add_argument("--untrained", action="store_true", help="also run an untrained copy (the paper's context row)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_uncommitted", action="store_true", help="tests only; never for results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.step == "prepare":
        for a in args.anchors:
            prepare(a)
        return
    dirty = mb.uncommitted_code()
    if dirty and not args.allow_uncommitted:
        sys.exit(f"refusing to run: these files under src/ are not committed: {dirty}. Commit them first, so "
                 "that the git_commit each result records is the code that produced it.")
    if args.step == "report":
        report()
        return
    fn = run_next_token if args.step == "next-token" else run_probe
    for a in args.anchors:
        for untrained in ([False, True] if args.untrained else [False]):
            path = _save(fn(a, untrained, args.device))
            log.info("wrote %s", path)


if __name__ == "__main__":
    main()
