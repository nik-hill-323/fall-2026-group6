"""WM-Concord demo: an Othello world-model lab for a general audience.

Run from the repo root:

    streamlit run demo/app.py

Seven tabs, in the order of the presentation's demo slide. Each shows one question, one big number with an
(i) tooltip that says what it means, a gauge from the zoo's lowest to its best model (the published
Othello-GPT marked), and one animated picture. The full pipeline of each test is in a closed
"How it's computed" box.
    Model zoo     the 81 models as discs on a 9 x 9 board (click one to select it)
    D1            a real game replayed (play button); the model's pick when its weights are here
    D2            two real move orders that reach the same board, played side by side
    D3            the four "is this square mine?" questions on a real board; trained model vs random copy
    D4            an X-ray of the board layer by layer; one disc flipped inside the model
    Fragility     a rotating board; the drop in legal moves per kind of shift
    Concordance   the three answers (H1 to H3) and the agreement grid, raw or with skill removed

Everything is read from the repo (src/docs/zoo_registry.csv, results/, results/analysis/othello/) and the
Othello data; checkpoints under outputs/zoo/ are used only if present. Nothing is written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load("othello", "src/domains/othello.py")
registry = _load("registry", "src/zoo/registry.py")
results_mod = _load("results", "src/component/results.py")

DIAGS = ["D1", "D2", "D3", "D4"]
DIAG_NAME = {"D1": "Next-move validity", "D2": "State equivalence", "D3": "Inductive bias",
             "D4": "Board probe", "FRAG": "Fragility v2"}
ANCHORS = {"othello_gpt_synthetic": "synthetic", "othello_gpt_championship": "championship"}
ARCHS = ["transformer", "lstm", "mamba"]
SCALES = ["small", "medium", "large"]
DISTS = ["synthetic", "championship", "mixed"]
SEEDS = [0, 1, 2]
ARCH_LABEL = {"transformer": "Transformer", "lstm": "LSTM", "mamba": "Mamba", "anchor": "Othello-GPT"}
ARCH_COLOR = {"transformer": "#e3b341", "lstm": "#58a6ff", "mamba": "#f778ba", "anchor": "#ffffff"}
FRAGILITY_DIRS = {"v1": "results/fragility", "v2": "results/fragility_v2"}
GOLD = "#e3b341"
DIAG_COLOR = {"D1": "#cfd8e3", "D2": "#5b8def", "D3": "#f08a4b", "D4": "#2fbf8f", "FRAG": "#e3b341"}


# ==========================================================================
# Look and feel
# ==========================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;600&display=swap');
:root { --felt:#1b6b45; --felt-dark:#0b3221; --line:#0a2a1b; --gold:#e3b341; --ink:#eef5f0; --muted:#a9c2b4; }
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #23825a 0%, #11492f 45%, #082418 100%); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display: flex !important; visibility: visible !important; background: rgba(227,179,65,.18); border-radius: 8px; }
.chip { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align: middle; }
.zoo a { display:flex; width:100%; height:100%; align-items:center; justify-content:center; }
.zoo a:hover .zd { transform: scale(1.12); box-shadow: 0 0 0 2px rgba(227,179,65,.7), 0 6px 12px rgba(0,0,0,.6); }
.zoo .zd { transition: transform .12s ease; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #07170f 0%, #0b2419 100%); border-right: 1px solid rgba(227,179,65,.25); }
h1, h2, h3, .wm-title { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.01em; }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; }
[data-testid="stMetric"] { background: rgba(0,0,0,.28); border: 1px solid rgba(255,255,255,.08); border-radius: 14px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
[data-testid="stMetricLabel"] p { color: var(--muted); }
.wm-card { background: rgba(0,0,0,.28); border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 16px 18px; margin: 6px 0 14px; }
.wm-note { color: var(--muted); font-size: .88rem; }
.wm-pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: 6px; }
.wm-pill.ok { background: rgba(63,185,80,.18); color: #7ee787; border: 1px solid rgba(126,231,135,.4); }
.wm-pill.warn { background: rgba(227,179,65,.15); color: var(--gold); border: 1px solid rgba(227,179,65,.45); }
.wm-pill.bad { background: rgba(248,81,73,.15); color: #ff7b72; border: 1px solid rgba(255,123,114,.45); }

/* hero */
.wm-hero { display:flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
           background: rgba(0,0,0,.25); border: 1px solid rgba(227,179,65,.25); border-radius: 20px; padding: 16px 22px; margin-bottom: 18px; }
.wm-hero .who { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 700; }
.wm-hero .sub { color: var(--muted); font-size: .9rem; }
.rank-row { display:flex; gap: 16px; flex-wrap: wrap; }
.rank { display:flex; flex-direction: column; align-items:center; width: 92px; }
.rank .rd { width: 54px; height: 54px; border-radius: 50%; box-shadow: 0 4px 10px rgba(0,0,0,.55), inset 0 0 0 2px rgba(0,0,0,.35); position: relative; }
.rank .rd::after { content:""; position:absolute; inset: 0; border-radius:50%; background: radial-gradient(circle at 32% 28%, rgba(255,255,255,.35), rgba(0,0,0,0) 45%, rgba(0,0,0,.35) 100%); }
.rank .lab { font-size: .72rem; color: var(--muted); margin-top: 6px; text-align:center; line-height: 1.1; }
.rank .val { font-family: 'Space Grotesk', sans-serif; font-size: .95rem; font-weight: 700; }

/* board */
.board { display:inline-grid; grid-template-columns: 22px repeat(8, var(--sq)); grid-auto-rows: var(--sq);
         background: #0d3a25; padding: 10px 14px 14px 6px; border-radius: 14px;
         box-shadow: 0 18px 40px rgba(0,0,0,.55), inset 0 0 0 6px #3b2a14, inset 0 0 0 8px #1f150a; }
.board .lbl { color: #cfe3d6; font-size: .72rem; display:flex; align-items:center; justify-content:center; opacity: .8; }
.board .top { align-items: flex-end; padding-bottom: 3px; }
.sq { position: relative; background: var(--felt); border: 1px solid var(--line); display:flex; align-items:center; justify-content:center; }
.sq.dot::before { content:""; position:absolute; width:7px; height:7px; border-radius:50%; background: var(--line); top:-4px; left:-4px; z-index: 2; }
.disc { width: 80%; height: 80%; border-radius: 50%; position: relative; z-index: 1; }
.disc.b { background: radial-gradient(circle at 32% 28%, #6f6f6f 0%, #1c1c1c 42%, #000 100%); box-shadow: 0 3px 6px rgba(0,0,0,.65); }
.disc.w { background: radial-gradient(circle at 32% 28%, #ffffff 0%, #eef1ef 48%, #b7c0ba 100%); box-shadow: 0 3px 6px rgba(0,0,0,.5); }
.disc.new { animation: dropin .45s cubic-bezier(.2,.9,.3,1.3); }
.disc.flip { animation: flip .55s ease; }
@keyframes dropin { from { transform: scale(.2); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@keyframes flip { 0% { transform: rotateY(0); } 50% { transform: rotateY(90deg); } 100% { transform: rotateY(0); } }
.last-mark { position:absolute; width: 18%; height: 18%; border-radius: 50%; background: var(--gold); z-index: 3; box-shadow: 0 0 6px var(--gold); }
.hint { width: 24%; height: 24%; border-radius: 50%; background: rgba(227,179,65,.75); box-shadow: 0 0 8px rgba(227,179,65,.6); }
.ring-ok { box-shadow: inset 0 0 0 3px #58d68d, 0 0 14px rgba(88,214,141,.7); z-index: 4; }
.ring-bad { box-shadow: inset 0 0 0 3px #ff5f56, 0 0 14px rgba(255,95,86,.8); z-index: 4; }
.sqtxt { position:absolute; bottom: 2px; right: 3px; font-size: .62rem; font-weight: 700; color: #fff; text-shadow: 0 1px 2px #000; z-index: 5; }
.sqname { position:absolute; top: 2px; left: 4px; font-size: .6rem; font-weight: 700; color: rgba(255,255,255,.85); z-index: 5; }

/* zoo board */
.zoo-wrap { overflow-x: auto; padding-bottom: 4px; }
.zoo { --zs: clamp(30px, 3.1vw, 50px); display:grid; grid-template-columns: 136px repeat(9, var(--zs)); gap: 0; background:#0d3a25; padding: 12px 16px 16px 12px; border-radius: 16px;
       box-shadow: 0 18px 40px rgba(0,0,0,.55), inset 0 0 0 6px #3b2a14, inset 0 0 0 8px #1f150a; width: max-content; }
.zoo .h { color:#cfe3d6; font-size:.7rem; text-align:center; padding: 2px 0 6px; }
.zoo .g { color: var(--gold); font-size:.72rem; font-weight:700; text-align:center; grid-column: span 3; border-bottom: 1px solid rgba(227,179,65,.35); margin: 0 4px 4px; }
.zoo .r { color:#cfe3d6; font-size:.78rem; display:flex; align-items:center; padding-right: 8px; justify-content: flex-end; }
.zoo .r b { color: var(--ink); margin-right: 5px; }
.zoo .c { width: var(--zs); height: var(--zs); background: var(--felt); border: 1px solid var(--line); display:flex; align-items:center; justify-content:center; position: relative; }
.zoo .c.gap { border-left: 3px solid var(--line); }
.zoo .zd { width: 78%; height: 78%; border-radius: 50%; box-shadow: 0 3px 6px rgba(0,0,0,.6); }
.zoo .sel { box-shadow: 0 0 0 3px var(--gold), 0 0 16px var(--gold); }
.zoo .empty { width: 30%; height: 30%; border-radius: 50%; background: rgba(0,0,0,.35); }
.legend { display:flex; align-items:center; gap: 10px; color: var(--muted); font-size: .8rem; margin: 8px 0 0; }
.legend .bar { width: 160px; height: 12px; border-radius: 6px; background: linear-gradient(90deg, #111, #eee); box-shadow: inset 0 0 0 1px rgba(255,255,255,.2); }

/* agreement grid */
.agree { display:grid; gap: 6px; width: max-content; }
.agree .a { width: 74px; height: 58px; border-radius: 12px; display:flex; flex-direction: column; align-items:center; justify-content:center;
            font-family:'Space Grotesk', sans-serif; font-weight:700; color:#081a10; }
.agree .a small { font-family: 'Inter'; font-weight: 600; font-size: .62rem; opacity: .75; }
.agree .hd { color: var(--muted); font-size:.78rem; display:flex; align-items:center; justify-content:center; text-align:center; }
.logo { display:grid; grid-template-columns: 18px 18px; gap: 3px; background: var(--felt); padding: 5px; border-radius: 8px; border: 2px solid #3b2a14; width: max-content; }
.logo span { width:18px; height:18px; border-radius:50%; }
.logo .b { background: radial-gradient(circle at 32% 28%, #6f6f6f, #000 70%); }
.logo .w { background: radial-gradient(circle at 32% 28%, #fff, #b7c0ba 80%); }
</style>
"""


def disc_shade(p: float | None) -> str:
    """Percentile 0..1 -> a disc colour from black (low) to white (high)."""
    if p is None or np.isnan(p):
        return "#3a3a3a"
    v = int(18 + p * 222)
    return f"rgb({v},{v},{v})"


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


# ==========================================================================
# Data access (cached)
# ==========================================================================

@st.cache_data
def load_registry() -> pd.DataFrame:
    path = ROOT / "src/docs/zoo_registry.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_record(model_id: str) -> dict | None:
    path = ROOT / "results/zoo_training" / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_result(model_id: str, diagnostic: str) -> dict | None:
    path = ROOT / "results" / results_mod.diag_folder(diagnostic) / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_fragility(model_id: str, version: str = "v2") -> dict | None:
    path = ROOT / FRAGILITY_DIRS[version] / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_scores(model_ids: tuple[str, ...]) -> pd.DataFrame:
    """One row per model: its val legal rate, D1 to D4 score, fragility v2, and arch / scale / dist."""
    rows = []
    for m in model_ids:
        rec = load_record(m)
        row = {"model_id": m, "legal": rec["eval"]["legal_move_rate"] if rec else np.nan}
        for d in DIAGS:
            r = load_result(m, d)
            row[d] = r["score"] if r and r.get("status") == "ok" and r.get("score") is not None else np.nan
        fr = load_fragility(m, "v2")
        row["FRAG"] = fr["fragility"] if fr else np.nan
        row.update(parse_id(m))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_id(model_id: str) -> dict:
    if model_id in ANCHORS:
        return {"arch": "anchor", "scale": "8 layer", "distribution": ANCHORS[model_id], "seed": -1}
    _, arch, scale, dist, seed = model_id.split("_")
    return {"arch": arch, "scale": scale, "distribution": dist, "seed": int(seed[1:])}


def pretty(model_id: str) -> str:
    if model_id in ANCHORS:
        return f"★ Othello-GPT {ANCHORS[model_id]} (reference)"
    p = parse_id(model_id)
    return f"{ARCH_LABEL[p['arch']]} · {p['scale']} · {p['distribution']} · seed {p['seed']}"


@st.cache_data
def load_games(distribution: str, split: str, n: int) -> np.ndarray | None:
    try:
        return rules.load_distribution(distribution, split, n, seed=0)
    except (FileNotFoundError, ValueError, OSError):
        return None


@st.cache_data
def has_weights(model_id: str) -> bool:
    """True if this zoo model's checkpoint is here and matches checkpoint_sha256 in its training record."""
    path, rec = ROOT / "outputs/zoo" / model_id / "model.pt", load_record(model_id)
    if not path.exists() or not rec or not rec.get("checkpoint_sha256"):
        return False
    if parse_id(model_id)["arch"] == "mamba" and importlib.util.find_spec("mambapy") is None:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == rec["checkpoint_sha256"]


@st.cache_data(show_spinner=False)
def game_logits(model_id: str, dist: str, split: str, g: int) -> np.ndarray | None:
    model = load_model(model_id)
    if isinstance(model, str):
        return None
    data = rules.build(load_games(dist, split, 50)[g][None], legal=True)
    return model.logits(data["tokens"])[0].cpu().numpy()


@st.cache_resource
def load_model(model_id: str):
    """A ZooModel, or an error string if the checkpoint or its packages are not on this machine."""
    try:
        if model_id in ANCHORS:
            itf = _load("interface", "src/diagnostics/interface.py")
            return itf.othello_gpt(ANCHORS[model_id], device="cpu")
        return registry.load_model(model_id, device="cpu")
    except Exception as e:  # noqa: BLE001  (shown to the user, not hidden)
        return f"{type(e).__name__}: {e}"


def percentile(scores: pd.DataFrame, col: str, value: float, higher_is_better: bool = True) -> float | None:
    """Share of zoo models (anchors excluded) this value beats on col, 0..1."""
    if value is None or np.isnan(value):
        return None
    pool = scores.loc[scores["arch"] != "anchor", col].dropna().to_numpy()
    if len(pool) == 0:
        return None
    p = float((pool < value).mean() + 0.5 * (pool == value).mean())
    return p if higher_is_better else 1 - p


# ==========================================================================
# Board drawing
# ==========================================================================

DOT_SQUARES = {18, 22, 50, 54}   # C3, G3, C7, G7: the four marks at the corners of the centre 4 x 4


def sq_name(s: int) -> str:
    return "ABCDEFGH"[s % 8] + str(s // 8 + 1)


def board_html(board: np.ndarray | None = None, *, last: int | None = None, legal: set[int] | None = None,
               pick: int | None = None, flipped: set[int] | None = None, heat: dict[int, str] | None = None,
               text: dict[int, str] | None = None, names: set[int] | None = None, sq: int = 50,
               pulse: set[int] | None = None, loop_flip: dict[int, str] | None = None, cls_extra: str = "") -> str:
    """An 8 x 8 Othello board. board: 64 codes (0 empty, 1 black, 2 white) or None for an empty board.
    legal: gold hints. pick: the model's choice, green ring if legal, red if not. flipped: discs that
    just turned (animated). heat: square -> background colour. text: square -> small label."""
    legal, flipped, heat, text, names = legal or set(), flipped or set(), heat or {}, text or {}, names or set()
    pulse, loop_flip = pulse or set(), loop_flip or {}
    out = [f"<div class='board {cls_extra}' style='--sq:{sq}px'><div></div>"]
    out += [f"<div class='lbl top'>{c}</div>" for c in "ABCDEFGH"]
    for r in range(8):
        out.append(f"<div class='lbl'>{r + 1}</div>")
        for c in range(8):
            s = r * 8 + c
            cls = "sq" + (" dot" if s in DOT_SQUARES else "") + (" pulse" if s in pulse else "")
            if s == pick:
                cls += " ring-ok" if s in legal else " ring-bad"
            style = f"background:{heat[s]};" if s in heat else ""
            inner = ""
            v = int(board[s]) if board is not None else 0
            if v in (rules.BLACK, rules.WHITE):
                anim = " new" if s == last else (" flip" if s in flipped else "")
                if s in loop_flip:
                    anim = f" loop-{loop_flip[s]}"
                inner = f"<div class='disc {'b' if v == rules.BLACK else 'w'}{anim}'></div>"
                if s == last:
                    inner += "<div class='last-mark'></div>"
            elif s in legal:
                inner = "<div class='hint'></div>"
            if s in names:
                inner += f"<span class='sqname'>{sq_name(s)}</span>"
            if s in text:
                inner += f"<span class='sqtxt'>{text[s]}</span>"
            out.append(f"<div class='{cls}' style='{style}' title='{sq_name(s)}'>{inner}</div>")
    out.append("</div>")
    return "".join(out)


# ==========================================================================
# Shared pieces
# ==========================================================================


# ==========================================================================
# Pages
# ==========================================================================


# ==========================================================================
# Pipelines: every step of every test, explained with real data
# ==========================================================================

STEP_CSS = """
<style>
.flow { display:flex; align-items:stretch; gap:0; margin: 6px 0 14px; flex-wrap: wrap; row-gap: 10px; }
.flow .st { flex: 1 1 120px; min-width: 110px; background: rgba(0,0,0,.28); border: 1px solid rgba(255,255,255,.10);
            padding: 10px 12px; border-radius: 12px; position: relative; margin-right: 24px; }
.flow .st:last-child { margin-right: 0; }
.flow .st:not(:last-child)::after { content: "➜"; position: absolute; right: -20px; top: 50%; transform: translateY(-50%);
            color: var(--gold); font-size: 1rem; }
.flow .st.on { background: rgba(227,179,65,.16); box-shadow: 0 0 18px rgba(227,179,65,.25); }
.flow .n { font-family: 'Space Grotesk'; font-weight: 700; color: var(--gold); font-size: .78rem; letter-spacing: .04em; }
.flow .t { font-weight: 700; font-size: .92rem; margin: 2px 0; }
.flow .d { color: var(--muted); font-size: .74rem; line-height: 1.25; }
.tok { display:inline-block; padding: 3px 7px; margin: 2px; border-radius: 7px; background: rgba(0,0,0,.35);
       border: 1px solid rgba(255,255,255,.12); font-family: 'Space Grotesk'; font-size: .85rem; }
.tok b { color: var(--gold); margin-left: 4px; font-size: .75rem; }
.tok.hl { border-color: var(--gold); background: rgba(227,179,65,.18); }
.big { font-family: 'Space Grotesk'; font-size: 2.1rem; font-weight: 700; line-height: 1.05; }
.says { font-family: 'Space Grotesk'; font-size: 1.05rem; margin: 6px 0; }
</style>
"""


@st.cache_data(show_spinner=False)
def built_games(dist: str, split: str, n: int) -> dict | None:
    games = load_games(dist, split, n)
    return None if games is None else rules.build(games, legal=True)


def moves_of(data: dict, g: int, t: int) -> list[int]:
    """Squares of the first t + 1 moves of game g."""
    return [rules.TOKEN_TO_SQ[int(x)] for x in data["tokens"][g, : t + 1]]


def move_chips(moves: list[int], with_tokens: bool = False, limit: int = 14) -> str:
    shown = moves[-limit:]
    pre = "<span class='tok'>…</span>" if len(moves) > limit else ""
    return "<div style='display:flex;flex-wrap:wrap;gap:2px'>" + pre + "".join(
        f"<span class='tok'>{sq_name(s)}{f'<b>{rules.SQ_TO_TOKEN[s]}</b>' if with_tokens else ''}</span>" for s in shown) + "</div>"


@st.cache_data(show_spinner=False)
def same_board_pair(dist: str, pick: int) -> dict | None:
    """Two different move orders in real test games that reach the same board with the same player to move."""
    data = built_games(dist, "test", 400)
    if data is None:
        return None
    seen: dict[bytes, tuple[int, int]] = {}
    found = []
    for t in range(3, 14):
        for g in range(len(data["tokens"])):
            if not data["mask"][g, t]:
                continue
            k = data["black_white"][g, t].tobytes() + bytes([int(data["movers"][g, t])])
            if k in seen:
                g0, t0 = seen[k]
                if list(data["tokens"][g0, : t0 + 1]) != list(data["tokens"][g, : t + 1]):
                    found.append((g0, g, t))
            else:
                seen[k] = (g, t)
    if not found:
        return None
    g0, g1, t = found[pick % len(found)]
    board = data["black_white"][g1, t]
    return {"h": moves_of(data, g0, t), "g": moves_of(data, g1, t), "board": board,
            "legal": set(rules.legal_next(board, int(data["movers"][g1, t]))), "n_pairs": len(found)}


@st.cache_data(show_spinner=False)
def task_position(dist: str, pick: int) -> dict | None:
    """One real test position and the four D3 answers, read off the true board."""
    data = built_games(dist, "test", 200)
    if data is None:
        return None
    rng = np.random.default_rng(pick)
    g = int(rng.integers(len(data["tokens"])))
    t = int(rng.integers(10, int(data["mask"][g].sum()) - 2))
    board, mover = data["black_white"][g, t], int(data["movers"][g, t])
    answers = {sq: bool(board[rules_square(sq)] == mover) for sq in ("C4", "F5", "C3", "F6")}
    return {"board": board, "mover": mover, "answers": answers, "t": t, "moves": moves_of(data, g, t)}


@st.cache_data(show_spinner=False)
def flip_case(dist: str, pick: int) -> dict | None:
    """A real position and one disc whose flip (mine <-> theirs) changes the legal next moves: the D4 causal edit."""
    data = built_games(dist, "test", 200)
    if data is None:
        return None
    rng = np.random.default_rng(pick)
    for _ in range(500):
        g = int(rng.integers(len(data["tokens"])))
        T = int(data["mask"][g].sum())
        t = int(rng.integers(8, max(9, T - 10)))
        board, mover = data["black_white"][g, t], int(data["movers"][g, t])
        before = set(rules.legal_next(board, mover))
        for sq in rng.permutation(np.flatnonzero(board != rules.EMPTY)):
            edited = board.copy()
            edited[sq] = rules.other(int(edited[sq]))
            after = set(rules.legal_next(edited, mover))
            if before and after and before != after and len(before ^ after) <= 3:
                return {"board": board, "edited": edited, "sq": int(sq), "before": before, "after": after, "t": t}
    return None


# ==========================================================================
# Flowcharts: the full pipeline of each diagnostic, drawn as boxes and arrows, with the model's numbers
# ==========================================================================

FLOW_KIND = {   # fill, border, title colour
    "data": ("#16402b", "#3f7a5a", "#eef5f0"),
    "rules": ("#3b2a14", "#a0782f", "#f3d58b"),
    "model": ("#1d2a3a", "#5b8def", "#d6e4ff"),
    "op": ("#10261b", "#2f5a43", "#eef5f0"),
    "twin": ("#262626", "#7d8b83", "#d9dedb"),
    "result": ("#3a2e10", "#e3b341", "#ffe6a6"),
}


def _esc(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_flow(nodes: list[dict], edges: list[tuple], width: int, height: int, accent: str) -> str:
    """nodes: {id, x, y, w, h, title, lines, kind, step}; edges: (src, dst, label, mode) with mode 'h' (right to
    left side, elbow if needed) or 'v' (bottom or top centre). Boxes appear one step after another."""
    by = {n["id"]: n for n in nodes}
    parts = [f"<svg viewBox='0 0 {width} {height}' width='100%' xmlns='http://www.w3.org/2000/svg' "
             f"style='max-width:{width}px;font-family:Inter,sans-serif'>",
             "<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
             f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{accent}'/></marker></defs>",
             "<style>.fg{opacity:0;animation:fin .5s ease forwards}@keyframes fin{from{opacity:0;transform:translateY(6px)}"
             "to{opacity:1;transform:none}}</style>"]
    for src, dst, label, mode in edges:
        a, b = by[src], by[dst]
        delay = 0.35 * max(a.get("step", 0), b.get("step", 0) - 1) + 0.2
        if mode == "v":
            x1, x2 = a["x"] + a["w"] / 2, b["x"] + b["w"] / 2
            y1, y2 = (a["y"], b["y"] + b["h"]) if a["y"] > b["y"] else (a["y"] + a["h"], b["y"])
            d = f"M{x1},{y1} L{x2},{y2}"
            lx, ly = x1 + 8, (y1 + y2) / 2
        else:
            x1, y1 = a["x"] + a["w"], a["y"] + a["h"] / 2
            x2, y2 = b["x"], b["y"] + b["h"] / 2
            if abs(y1 - y2) < 2:
                d = f"M{x1},{y1} L{x2 - 2},{y2}"
            else:
                xm = (x1 + x2) / 2
                d = f"M{x1},{y1} L{xm},{y1} L{xm},{y2} L{x2 - 2},{y2}"
            lx, ly = (x1 + x2) / 2 + 4, min(y1, y2) - 6 if abs(y1 - y2) < 2 else (y1 + y2) / 2
        parts.append(f"<g class='fg' style='animation-delay:{delay:.2f}s'><path d='{d}' fill='none' stroke='{accent}' "
                     f"stroke-width='2' marker-end='url(#ah)' opacity='.85'/>"
                     + (f"<text x='{lx}' y='{ly}' font-size='11' fill='#a9c2b4'>{_esc(label)}</text>" if label else "")
                     + "</g>")
    for n in nodes:
        fill, border, tcol = FLOW_KIND[n.get("kind", "op")]
        if n.get("kind") == "result":
            border = accent
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        delay = 0.35 * n.get("step", 0)
        g = [f"<g class='fg' style='animation-delay:{delay:.2f}s'><title>{_esc(n.get('tip', n['title']))}</title>",
             f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='12' fill='{fill}' stroke='{border}' stroke-width='"
             f"{3 if n.get('kind') == 'result' else 1.6}'/>"]
        if "num" in n:
            g.append(f"<circle cx='{x + 16}' cy='{y + 16}' r='10' fill='{accent}'/><text x='{x + 16}' y='{y + 20}' "
                     f"font-size='11.5' font-weight='700' text-anchor='middle' fill='#0b2016'>{n['num']}</text>")
        tx = x + (32 if "num" in n else 12)
        g.append(f"<text x='{tx}' y='{y + 21}' font-size='14' font-weight='700' fill='{tcol}'>{_esc(n['title'])}</text>")
        ty = y + 40
        for line in n.get("lines", []):
            big = isinstance(line, tuple)
            txt, size, colour, weight = (line[0], 24, accent, 700) if big else (line, 12, "#c9dccf", 400)
            ty += 10 if big else 0
            g.append(f"<text x='{x + 12}' y='{ty}' font-size='{size}' font-weight='{weight}' fill='{colour}'>{_esc(txt)}</text>")
            ty += 22 if big else 17
        g.append("</g>")
        parts.append("".join(g))
    parts.append("</svg>")
    return "".join(parts)


def show_flow(svg: str, caption: str) -> None:
    html(f"<div class='wm-card' style='padding:14px 14px 6px'>{svg}<div class='wm-note' style='margin:4px 4px 6px'>"
         f"{caption}</div></div>")


def f3(x, fmt="{:.3f}") -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def flowchart_d1(model_id: str) -> None:
    r = load_result(model_id, "D1")
    d = r["details"] if r else {}
    W, H, w, h = 1240, 300, 150, 108
    xs = [10, 185, 360, 535, 710, 885, 1070]
    nodes = [
        dict(id="g", x=xs[0], y=40, w=w, h=h, num=1, step=0, kind="data", title="Test games",
             lines=["1,000 games of the", "model's own dataset", f"{d.get('distribution', '')}"]),
        dict(id="t", x=xs[1], y=40, w=w, h=h, num=2, step=1, kind="data", title="Moves → tokens",
             lines=["D3 C3 C4 → 20 19 27", "the model never", "sees the board"]),
        dict(id="m", x=xs[2], y=40, w=w, h=h, num=3, step=2, kind="model", title="The model",
             lines=["scores all 60", "squares for the", "next move"]),
        dict(id="p", x=xs[3], y=40, w=w, h=h, num=4, step=3, kind="op", title="Top-1 move",
             lines=["take its single", "best guess,", "e.g. E3"]),
        dict(id="c", x=xs[4], y=40, w=w, h=h, num=5, step=4, kind="op", title="Legal?",
             lines=["is the guess in the", "true legal set?", "yes = 1, no = 0"]),
        dict(id="a", x=xs[5], y=40, w=w, h=h, num=6, step=5, kind="op", title="Average",
             lines=["over every position", f"{d.get('n_positions', 0):,} positions" if d else "of 1,000 games",
                    "(59 per game)"]),
        dict(id="s", x=xs[6], y=34, w=160, h=120, step=6, kind="result", title="D1 score",
             lines=[(f3(r['score']) if r else "–",), "share of legal", "top moves"]),
        dict(id="r", x=xs[3] - 10, y=200, w=330, h=82, step=4, kind="rules", title="Rules engine (othello.py)",
             lines=["replays the same moves: true board + legal", "next moves, e.g. {C5, E3}: the answer key"]),
    ]
    edges = [("g", "t", "", "h"), ("t", "m", "", "h"), ("m", "p", "", "h"), ("p", "c", "", "h"), ("c", "a", "", "h"),
             ("a", "s", "", "h"), ("r", "c", "legal set", "v")]
    show_flow(svg_flow(nodes, edges, W, H, DIAG_COLOR["D1"]), "Nothing is fitted: the rules engine is the answer key.")


def flowchart_d2(model_id: str) -> None:
    r = load_result(model_id, "D2")
    d = r["details"] if r else {}
    W, H, w, h = 1240, 330, 172, 100
    nodes = [
        dict(id="g", x=10, y=110, w=170, h=110, num=1, step=0, kind="data", title="Test games",
             lines=["every prefix is a", "history", f"{d.get('n_histories', 0):,} histories" if d else ""]),
        dict(id="a1", x=225, y=20, w=w, h=h, num=2, step=1, kind="rules", title="Same board",
             lines=["two different move", "orders, one board", f"{d.get('n_same_state_pairs', '–')} pairs"]),
        dict(id="a2", x=430, y=20, w=w, h=h, num=3, step=2, kind="model", title="Accepted moves",
             lines=["model's top k after", "each history", "(k = # legal moves)"]),
        dict(id="a3", x=635, y=20, w=w, h=h, num=4, step=3, kind="op", title="Compare",
             lines=["|A(h) △ A(g)|", "÷ |A(h) ∪ A(g)|", "0 = identical"]),
        dict(id="a4", x=840, y=20, w=w, h=h, step=4, kind="op", title="Compression err.",
             lines=[(f3(d.get("compression_error")),), "same board, treated", "differently"]),
        dict(id="b1", x=225, y=210, w=w, h=h, num=2, step=1, kind="rules", title="Different boards",
             lines=["two histories, same", "length, other legal", f"moves · {d.get('n_different_state_pairs', '–')} pairs"]),
        dict(id="b2", x=430, y=210, w=w, h=h, num=3, step=2, kind="rules", title="Boundary moves",
             lines=["legal after exactly", "one of the two", "histories"]),
        dict(id="b3", x=635, y=210, w=w, h=h, num=4, step=3, kind="model", title="Right side?",
             lines=["accept it where it", "is legal, reject it", "where it is not"]),
        dict(id="b4", x=840, y=210, w=w, h=h, step=4, kind="op", title="Distinction err.",
             lines=[(f3(d.get("distinction_error")),), "different boards,", "not told apart"]),
        dict(id="s", x=1060, y=105, w=170, h=120, step=5, kind="result", title="D2 score",
             lines=[(f3(r["score"]) if r else "–",), "1 − mean of the", "two errors"]),
    ]
    edges = [("g", "a1", "", "h"), ("a1", "a2", "", "h"), ("a2", "a3", "", "h"), ("a3", "a4", "", "h"), ("a4", "s", "", "h"),
             ("g", "b1", "", "h"), ("b1", "b2", "", "h"), ("b2", "b3", "", "h"), ("b3", "b4", "", "h"), ("b4", "s", "", "h")]
    show_flow(svg_flow(nodes, edges, W, H, DIAG_COLOR["D2"]), "Top lane: merge what should be merged. Bottom lane: separate what should be separated.")


def flowchart_d3(model_id: str) -> None:
    r = load_result(model_id, "D3")
    d = r["details"] if r else {}
    sr = d.get("symbolic_regression") or {}
    W, H, w, h = 1240, 360, 172, 100
    nodes = [
        dict(id="x", x=10, y=120, w=180, h=120, num=1, step=0, kind="data", title="64 positions",
             lines=["real validation", "positions, labels from", "the true board:", "“is C4/F5/C3/F6 mine?”"]),
        dict(id="a1", x=240, y=20, w=w, h=h, num=2, step=1, kind="model", title="Trained model",
             lines=["a deep copy (the", "original is never", "changed)"]),
        dict(id="a2", x=445, y=20, w=w, h=h, num=3, step=2, kind="op", title="+ new head",
             lines=["linear, 2 outputs:", "mine / not mine"]),
        dict(id="a3", x=650, y=20, w=w, h=h, num=4, step=3, kind="op", title="Adapt",
             lines=["300 steps of 16", "head lr 1e-3", "body lr 1e-4"]),
        dict(id="a4", x=855, y=20, w=w, h=h, num=5, step=4, kind="op", title="Test",
             lines=["1,000 new positions", "balanced accuracy",
                    f"= {f3(d.get('pretrained_balanced_accuracy'))}" if d else ""]),
        dict(id="b1", x=240, y=230, w=w, h=h, num=2, step=1, kind="twin", title="Random twin",
             lines=["same architecture", "and size, random", "weights"]),
        dict(id="b2", x=445, y=230, w=w, h=h, num=3, step=2, kind="op", title="+ new head",
             lines=["the same 2-way", "head"]),
        dict(id="b3", x=650, y=230, w=w, h=h, num=4, step=3, kind="op", title="Adapt",
             lines=["exactly the same", "64 examples and", "settings"]),
        dict(id="b4", x=855, y=230, w=w, h=h, num=5, step=4, kind="op", title="Test",
             lines=["the same 1,000", "positions",
                    f"= {f3(d.get('random_balanced_accuracy'))}" if d else "balanced accuracy"]),
        dict(id="s", x=1065, y=120, w=165, h=120, step=5, kind="result", title="D3 score",
             lines=[(f3(r["score"], "{:+.3f}") if r else "–",), "trained − random,", "4 questions × 3 seeds"]),
    ]
    edges = [("x", "a1", "", "h"), ("a1", "a2", "", "h"), ("a2", "a3", "", "h"), ("a3", "a4", "", "h"), ("a4", "s", "", "h"),
             ("x", "b1", "", "h"), ("b1", "b2", "", "h"), ("b2", "b3", "", "h"), ("b3", "b4", "", "h"), ("b4", "s", "", "h")]
    show_flow(svg_flow(nodes, edges, W, H, DIAG_COLOR["D3"]), "If the board is already inside, 64 examples are enough to learn the question.")


def flowchart_d4(model_id: str) -> None:
    r = load_result(model_id, "D4")
    d = r["details"] if r else {}
    cz = d.get("causal") or {}
    b = d.get("best_layer")
    W, H, w, h = 1240, 400, 172, 100
    nodes = [
        dict(id="g", x=10, y=40, w=170, h=h, num=1, step=0, kind="data", title="10,000 val games",
             lines=["labels from the true", "board: mine / theirs /", "empty per square"]),
        dict(id="m", x=225, y=40, w=w, h=h, num=2, step=1, kind="model", title="Model activations",
             lines=["the vector after", f"every layer ({d.get('n_layers', '–')})", "at every move"]),
        dict(id="p", x=430, y=8, w=w, h=86, num=3, step=2, kind="op", title="Linear probe",
             lines=["reads 64 squares", f"acc {f3(d['probe_acc_by_layer'][b]) if d else '–'} (best layer)"]),
        dict(id="c", x=430, y=104, w=w, h=86, num=3, step=2, kind="twin", title="Control probe",
             lines=["same probe, fake labels", f"acc {f3(d['control_acc_by_layer'][b]) if d else '–'}"]),
        dict(id="sel", x=635, y=40, w=w, h=h, num=4, step=3, kind="op", title="Selectivity",
             lines=["probe − control,", "per layer"]),
        dict(id="bl", x=840, y=40, w=w, h=h, num=5, step=4, kind="op", title="Best layer",
             lines=["chosen on 1,000 more", "val games", f"→ layer {b} of {d.get('n_layers')}" if d else ""]),
        dict(id="s", x=1060, y=30, w=170, h=120, step=5, kind="result", title="D4 score",
             lines=[(f3(r["score"]) if r else "–",), "selectivity on", "1,000 test games"]),
        dict(id="e", x=225, y=262, w=w, h=h, num=6, step=6, kind="rules", title="Flip one disc",
             lines=["a real position; one", "flip that changes", "the legal moves"]),
        dict(id="ed", x=430, y=262, w=w, h=h, num=7, step=7, kind="model", title="Edit inside",
             lines=["move the vectors", "(middle → last layer)", "until probes read it"]),
        dict(id="rn", x=635, y=262, w=w, h=h, num=8, step=8, kind="op", title="Moves follow?",
             lines=[f"probe edit: {f3(cz.get('boundary_acc_edit'), '{:.0%}')} right",
                    f"random edit: {f3(cz.get('boundary_acc_random'), '{:.0%}')} right", "(same size)"]),
        dict(id="cs", x=840, y=262, w=w, h=h, num=9, step=9, kind="op", title="Causal score",
             lines=[(f3(cz.get("causal_score")),), "probe − random edit"]),
        dict(id="lb", x=1060, y=262, w=170, h=h, step=10, kind="result", title="Label",
             lines=[cz.get("causal_label", "–"), "pass: ≥ 0.20 with the", "interval above 0"]),
    ]
    edges = [("g", "m", "", "h"), ("m", "p", "", "h"), ("m", "c", "", "h"), ("p", "sel", "", "h"), ("c", "sel", "", "h"),
             ("sel", "bl", "", "h"), ("bl", "s", "", "h"), ("e", "ed", "", "h"), ("ed", "rn", "", "h"),
             ("rn", "cs", "", "h"), ("cs", "lb", "", "h"), ("m", "ed", "edit here", "h")]
    show_flow(svg_flow(nodes, edges, W, H, DIAG_COLOR["D4"]), "Top lane: is the board readable? Bottom lane: is it used?")


def rules_square(name: str) -> int:
    return (int(name[1]) - 1) * 8 + "ABCDEFGH".index(name[0])


@st.cache_data
def load_analysis(domain: str = "othello") -> dict | None:
    path = ROOT / "results/analysis" / domain / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


# ==========================================================================
# Simple pages for the presentation: one question, one number, one picture per tab
# ==========================================================================

SIMPLE_CSS = """
<style>
.kick { font-family:'Space Grotesk'; font-size:.95rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-top:4px; }
.q { font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; line-height:1.15; margin: 4px 0 2px; }
.q small { display:block; font-family:'Inter'; font-size:.95rem; font-weight:400; color: var(--muted); margin-top:6px; }
.kpi { display:flex; align-items:flex-end; gap:14px; margin: 14px 0 0; }
.kpi .num { font-family:'Space Grotesk'; font-size:4.6rem; font-weight:700; line-height:.9; color: var(--c);
            text-shadow: 0 0 28px color-mix(in srgb, var(--c) 45%, transparent); animation: pop .7s cubic-bezier(.2,.9,.3,1.4); }
.kpi .lab { font-size:1.05rem; color: var(--ink); padding-bottom: 8px; }
.kpi .lab .sub { display:block; color: var(--muted); font-size:.85rem; }
@keyframes pop { from { transform: scale(.6); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.tip { display:inline-flex; align-items:center; justify-content:center; position:relative; cursor:help; width:22px; height:22px;
       border-radius:50%; background: rgba(255,255,255,.14); font-size:.8rem; font-weight:700; margin-left:8px; vertical-align: middle; }
.tip .tt { visibility:hidden; opacity:0; position:absolute; left:30px; top:-8px; width:320px; background:#0b2016; color: var(--ink);
           border:1px solid var(--gold); padding:10px 12px; border-radius:10px; font-size:.86rem; font-weight:400; line-height:1.35;
           z-index:60; transition: opacity .15s; box-shadow: 0 10px 24px rgba(0,0,0,.5); text-align:left; }
.tip:hover .tt { visibility:visible; opacity:1; }
.gauge { position:relative; height:16px; border-radius:8px; background: rgba(0,0,0,.38); margin: 30px 0 62px; max-width: 760px; }
.gauge .fill { height:100%; border-radius:8px; width:0; background: var(--c); animation: grow 1.3s ease forwards .2s; }
@keyframes grow { to { width: var(--w); } }
.gauge .mk { position:absolute; top:-7px; width:3px; height:30px; background: rgba(255,255,255,.8); border-radius:2px; }
.gauge .mk span { position:absolute; top:32px; left:-55px; width:113px; text-align:center; font-size:.72rem; color: var(--muted); white-space:nowrap; }
.gauge .mk.r2 span { top:48px; }
.gauge .mk.me { background: var(--c); width:5px; box-shadow: 0 0 10px var(--c); }
.gauge .mk.me span { color: var(--ink); font-weight:700; top:-24px; }
.race { max-width: 620px; margin: 10px 0; }
.race .row { display:flex; align-items:center; gap:12px; margin: 14px 0; }
.race .who { width:150px; font-weight:700; }
.race .bar { flex:1; height:30px; background: rgba(0,0,0,.35); border-radius:8px; position:relative; overflow:hidden; }
.race .bar div { height:100%; width:0; border-radius:8px; animation: grow 1.6s ease forwards; }
.race .bar em { position:absolute; left:50%; top:0; bottom:0; border-left:2px dashed rgba(255,255,255,.55); }
.race .val { width:70px; font-family:'Space Grotesk'; font-size:1.3rem; font-weight:700; }
.xray { display:flex; align-items:flex-end; gap:10px; height:200px; padding: 10px 0; max-width: 560px; }
.xray .col { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; }
.xray .b { width:100%; border-radius:8px 8px 3px 3px; height:0; animation: rise .8s ease forwards; }
@keyframes rise { to { height: var(--h); } }
.xray .l { font-size:.75rem; color: var(--muted); margin-top:6px; }
.xray .col.best .l { color: var(--gold); font-weight:700; }
.sq.pulse { animation: sqpulse 1.4s ease-in-out infinite; }
@keyframes sqpulse { 0%,100% { box-shadow: inset 0 0 0 0 rgba(227,179,65,0); } 50% { box-shadow: inset 0 0 0 4px rgba(227,179,65,.95); } }
.disc.loop-wb { animation: lwb 2.6s ease-in-out infinite; }
.disc.loop-bw { animation: lbw 2.6s ease-in-out infinite; }
@keyframes lwb { 0%,35% { transform:rotateY(0); background: radial-gradient(circle at 32% 28%, #fff, #b7c0ba 90%); }
                 50% { transform: rotateY(90deg); } 65%,100% { transform:rotateY(0); background: radial-gradient(circle at 32% 28%, #6f6f6f, #000 70%); } }
@keyframes lbw { 0%,35% { transform:rotateY(0); background: radial-gradient(circle at 32% 28%, #6f6f6f, #000 70%); }
                 50% { transform: rotateY(90deg); } 65%,100% { transform:rotateY(0); background: radial-gradient(circle at 32% 28%, #fff, #b7c0ba 90%); } }
.board.spin { animation: spin 5s ease-in-out infinite; }
@keyframes spin { 0%,25% { transform: rotate(0); } 50%,75% { transform: rotate(180deg); } 100% { transform: rotate(360deg); } }
.badge { display:inline-block; padding: 8px 16px; border-radius: 999px; font-family:'Space Grotesk'; font-weight:700; font-size:1.1rem;
         background: rgba(63,185,80,.2); border: 2px solid #7ee787; color:#7ee787; animation: pop .6s ease; }
.badge.no { background: rgba(255,95,86,.15); border-color:#ff7b72; color:#ff7b72; }
.who-line { color: var(--muted); font-size:.9rem; }
</style>
"""


def question(text: str, sub: str, colour: str, kicker: str = "") -> None:
    top = f"<div class='kick' style='color:{colour}'>{kicker}</div>" if kicker else ""
    html(SIMPLE_CSS + top + f"<div class='q'><span class='chip' style='width:18px;height:18px;background:{colour}'></span>"
         f"{text}<small>{sub}</small></div>")


def kpi(value: str, label: str, tip: str, colour: str, sub: str = "", size: str = "4.6rem") -> None:
    html(f"<div class='kpi' style='--c:{colour}'><div class='num' style='font-size:{size}'>{value}</div><div class='lab'>{label}"
         f"<span class='tip'>i<span class='tt'>{tip}</span></span>{f'<span class=\"sub\">{sub}</span>' if sub else ''}</div></div>")


def gauge(value: float | None, lo: float, hi: float, colour: str, marks: list[tuple[float, str]]) -> None:
    """A bar from lo to hi that fills to this model's value, with labelled markers (zoo range, reference)."""
    if value is None or np.isnan(value):
        return
    pos = lambda v: max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100   # noqa: E731
    taken, mk = [pos(value)], []
    for v, t in sorted(((v, t) for v, t in marks if v is not None), key=lambda x: x[0]):
        x = pos(v)
        row2 = any(abs(x - y) < 14 for y in taken)          # too close to another label: second row
        taken.append(x)
        mk.append(f"<div class='mk{' r2' if row2 else ''}' style='left:{x:.1f}%'><span>{t}</span></div>")
    mk = "".join(mk)
    html(f"<div class='gauge' style='--c:{colour};--w:{pos(value):.1f}%'><div class='fill'></div>{mk}"
         f"<div class='mk me' style='left:{pos(value):.1f}%'><span>this model</span></div></div>")


def zoo_marks(scores: pd.DataFrame, col: str, ref: float | None, fmt: str = "{:.2f}") -> list[tuple[float, str]]:
    z = scores.loc[scores["arch"] != "anchor", col].dropna()
    out = [(float(z.min()), f"zoo lowest {fmt.format(z.min())}"), (float(z.max()), f"zoo best {fmt.format(z.max())}")]
    if ref is not None:
        out.append((ref, f"Othello-GPT {fmt.format(ref)}"))
    return out


def ref_score(diagnostic: str) -> float | None:
    r = load_result("othello_gpt_synthetic", diagnostic)
    return r["score"] if r and r.get("status") == "ok" else None


def who(model_id: str) -> None:
    html(f"<div class='who-line'>Model: <b>{pretty(model_id)}</b> · change it in the sidebar</div>")


def how_it_works(fn, model_id: str) -> None:
    with st.expander("How it's computed (pipeline)"):
        fn(model_id)


def player(key: str, n: int, label: str = "Step") -> int:
    """A slider with ▶ play that advances one step every 0.8 s. Returns the 0-based step."""
    if key not in st.session_state:
        st.session_state[key] = 1
    if st.session_state.pop(f"{key}_adv", False):
        st.session_state[key] = min(st.session_state[key] + 1, n)
    c = st.columns([1.1, 6])
    playing = c[0].toggle("▶ play", key=f"{key}_play")
    k = c[1].slider(label, 1, n, key=key)
    st.session_state[f"{key}_playing"] = playing and k < n
    return k - 1


def keep_playing(key: str) -> None:
    """Call at the end of the page: if ▶ play is on, wait and show the next step."""
    if st.session_state.get(f"{key}_playing"):
        time.sleep(0.8)
        st.session_state[f"{key}_adv"] = True
        st.rerun()


def simple_zoo(model_id: str, reg: pd.DataFrame, scores: pd.DataFrame) -> None:
    question("81 small models learned Othello", "Each disc is one model: rows = brain type and size, columns = training "
             "games and seed. Click a disc to pick it.", GOLD)
    shade = st.segmented_control("Colour the discs by", ["Legal moves", "D1", "D2", "D3", "D4"], default="Legal moves",
                                 key="zoo_metric") or "Legal moves"
    col = "legal" if shade == "Legal moves" else shade
    by_id = scores.set_index("model_id")
    cells = ["<div class='zoo'><div></div>"] + [f"<div class='g'>{d}</div>" for d in DISTS] + ["<div></div>"]
    cells += [f"<div class='h'>s{s_}</div>" for _ in DISTS for s_ in SEEDS]
    for arch in ARCHS:
        for scale in SCALES:
            cells.append(f"<div class='r'><b>{ARCH_LABEL[arch]}</b>{scale}</div>")
            for di, dist in enumerate(DISTS):
                for seed in SEEDS:
                    mid = f"othello_{arch}_{scale}_{dist}_s{seed}"
                    gap = " gap" if seed == 0 and di > 0 else ""
                    if mid not in by_id.index or np.isnan(by_id.loc[mid, col]):
                        cells.append(f"<div class='c{gap}'><div class='empty'></div></div>")
                        continue
                    v = by_id.loc[mid, col]
                    p = percentile(scores, col, v)
                    sel = " sel" if mid == model_id else ""
                    cells.append(f"<div class='c{gap}' title='{pretty(mid)}: {v:.3f}'><a href='?model={mid}' target='_self'>"
                                 f"<div class='zd{sel}' style='background:radial-gradient(circle at 32% 28%, rgba(255,255,255,.35), "
                                 f"rgba(0,0,0,0) 45%, rgba(0,0,0,.35) 100%), {disc_shade(p)}'></div></a></div>")
    cells.append("</div>")
    left, right = st.columns([1.5, 1])
    with left:
        html("<div class='zoo-wrap'>" + "".join(cells) + "</div>")
        html("<div class='legend'>low<div class='bar'></div>high · white = best in the zoo</div>")
    with right:
        rec = load_record(model_id)
        if rec:
            kpi(f"{rec['eval']['legal_move_rate']:.0%}", "legal moves",
                "Out of every 100 moves this model suggests on new games, this many are allowed by the rules. All 81 "
                "models are above 50%, so all are admitted.", GOLD, sub=pretty(model_id))
            st.caption(f"{rec['n_params'] / 1e6:.1f}M parameters · trained in {rec['train']['wall_sec'] / 60:.0f} min")
        html("<div class='wm-note' style='margin-top:18px'>Switch the colour between D1 and D4: if all tests agreed, "
             "the board would look the same.</div>")


def simple_d1(model_id: str, scores: pd.DataFrame) -> None:
    question("D1 · Does it only play legal moves?", "We show the model a game and check its top move against the rules.",
             DIAG_COLOR["D1"], kicker="Diagnostic 1 · Next-token validity")
    who(model_id)
    r = load_result(model_id, "D1")
    if r:
        kpi(f"{r['score']:.1%}", "of its moves are legal",
            "On 1,000 new games, at every position we take the model's single best guess and ask the rules engine: "
            "is it allowed? This is the share that was. 100% = never an illegal move.", DIAG_COLOR["D1"])
        gauge(r["score"], 0.5, 1.0, DIAG_COLOR["D1"], zoo_marks(scores, "D1", ref_score("D1"), "{:.0%}"))
    dist = ANCHORS.get(model_id) or parse_id(model_id)["distribution"]
    data = built_games(dist, "test", 50)
    if data is None:
        return
    T = int(data["mask"][0].sum())
    t = player("s1_t", T, "Move")
    board = data["black_white"][0, t]
    prev = data["black_white"][0, t - 1] if t > 0 else rules.start_board()
    mover = int(data["movers"][0, t])
    legal = set(rules.legal_next(board, mover))
    flipped = {s_ for s_ in range(64) if prev[s_] != 0 and board[s_] != prev[s_]}
    pick = None
    if model_id not in ANCHORS and has_weights(model_id):
        lg = game_logits(model_id, dist, "test", 0)
        if lg is not None:
            pick = rules.TOKEN_TO_SQ.get(int(lg[t].argmax()))
    left, right = st.columns([1, 1])
    with left:
        html(board_html(board, last=rules.TOKEN_TO_SQ[int(data["tokens"][0, t])], legal=legal, pick=pick, flipped=flipped, sq=46))
    with right:
        html("<div class='wm-card'><b>Gold dots</b> = moves the rules allow next.<br>"
             + ("<b style='color:#58d68d'>Green ring</b> = the model's pick is legal · <b style='color:#ff5f56'>red</b> = illegal"
                if pick is not None else "<span class='wm-note'>The model's own pick appears here when its weights are on "
                "this computer.</span>") + "</div>")
        if pick is not None:
            html(f"<span class='badge{'' if pick in legal else ' no'}'>{sq_name(pick)} · {'legal' if pick in legal else 'illegal'}</span>")
    how_it_works(flowchart_d1, model_id)
    keep_playing("s1_t")


def simple_d2(model_id: str, scores: pd.DataFrame) -> None:
    question("D2 · Same board, same answer?", "Two different move orders can reach the same board. A model that "
             "understands the board should treat them the same.", DIAG_COLOR["D2"], kicker="Diagnostic 2 · State equivalence")
    who(model_id)
    r = load_result(model_id, "D2")
    if r:
        d = r["details"]
        kpi(f"{r['score']:.2f}", "out of 1",
            f"1 = perfect. It averages two checks: same board reached two ways → does the model suggest the same moves? "
            f"(it got {1 - d['compression_error']:.0%} right) and different boards → does it tell them apart? "
            f"(it got {1 - d['distinction_error']:.0%} right).", DIAG_COLOR["D2"])
        gauge(r["score"], 0.5, 1.0, DIAG_COLOR["D2"], zoo_marks(scores, "D2", ref_score("D2")))
    dist = ANCHORS.get(model_id) or parse_id(model_id)["distribution"]
    pick = st.session_state.get("s2_pick", 0)
    ex = same_board_pair(dist, pick)
    if ex is None:
        return
    L = len(ex["h"])
    k = player("s2_k", L, "Move")
    bh, _ = rules.replay(np.array(ex["h"][: k + 1]))
    bg, _ = rules.replay(np.array(ex["g"][: k + 1]))
    c = st.columns([1, 1, 0.8])
    with c[0]:
        html(f"<b>Route 1</b><br>{move_chips(ex['h'][: k + 1], limit=8)}")
        html(board_html(bh[-1], last=ex["h"][k], sq=34))
    with c[1]:
        html(f"<b>Route 2</b><br>{move_chips(ex['g'][: k + 1], limit=8)}")
        html(board_html(bg[-1], last=ex["g"][k], sq=34))
    with c[2]:
        same = bool((bh[-1] == bg[-1]).all())
        html("<div style='height:70px'></div>" + (f"<span class='badge'>same board ✓</span><div class='wm-note' "
             f"style='margin-top:10px'>Different order, identical board: the model should suggest the same moves.</div>"
             if same else "<span class='wm-note'>Press ▶ play: two different routes…</span>"))

        def another() -> None:
            st.session_state["s2_pick"] = st.session_state.get("s2_pick", 0) + 1
            st.session_state["s2_k"] = 1

        st.button("🎲 another example", on_click=another)
    how_it_works(flowchart_d2, model_id)
    keep_playing("s2_k")


def simple_d3(model_id: str, scores: pd.DataFrame) -> None:
    question("D3 · Does its training help it learn something new?", "We teach it a new question, “is this square mine?”, "
             "with only 64 examples, and a random copy of it too.", DIAG_COLOR["D3"], kicker="Diagnostic 3 · Inductive-bias probe")
    who(model_id)
    r = load_result(model_id, "D3")
    if not r:
        return
    d = r["details"]
    kpi(f"{r['score']:+.3f}", "head start from training",
        "Accuracy of the trained model minus accuracy of a random copy, after both learn the same 64 examples. Above 0 = "
        "its training helped it learn the board question. 0 = no help.", DIAG_COLOR["D3"])
    gauge(r["score"], -0.02, 0.16, DIAG_COLOR["D3"], zoo_marks(scores, "D3", ref_score("D3"), "{:+.2f}"))
    left, right = st.columns([1, 1.2])
    with left:
        dist = ANCHORS.get(model_id) or parse_id(model_id)["distribution"]
        ex = task_position(dist, st.session_state.get("s3_pick", 0))
        if ex:
            sq4 = {rules_square(s_) for s_ in ex["answers"]}
            html(board_html(ex["board"], pulse=sq4, names=sq4,
                            text={rules_square(s_): ("yes" if a else "no") for s_, a in ex["answers"].items()}, sq=40))
            html(f"<div class='legend'>“Is this square mine?” on 4 squares · {'Black' if ex['mover'] == rules.BLACK else 'White'} "
                 "just moved · answers from the real board</div>")
    with right:
        pre, rnd = d["pretrained_balanced_accuracy"], d["random_balanced_accuracy"]
        w = lambda a: max(0.0, min(1.0, a)) * 100   # noqa: E731
        html(f"<div class='race'><div class='wm-note'>After learning from 64 examples, accuracy on 1,000 new positions "
             f"(dashed line = coin flip)</div>"
             f"<div class='row'><div class='who'>Trained model</div><div class='bar'><div style='--w:{w(pre):.1f}%;"
             f"background:{DIAG_COLOR['D3']}'></div><em></em></div><div class='val'>{pre:.0%}</div></div>"
             f"<div class='row'><div class='who'>Random copy</div><div class='bar'><div style='--w:{w(rnd):.1f}%;"
             f"background:#7d8b83;animation-delay:.3s'></div><em></em></div><div class='val'>{rnd:.0%}</div></div></div>")

        def another() -> None:
            st.session_state["s3_pick"] = st.session_state.get("s3_pick", 0) + 1

        st.button("🎲 another position", on_click=another)
    how_it_works(flowchart_d3, model_id)


def simple_d4(model_id: str, scores: pd.DataFrame) -> None:
    question("D4 · Is the board inside the model, and does it use it?", "We read the board straight out of the model's "
             "insides, then secretly flip one disc inside it and see if its moves change.", DIAG_COLOR["D4"], kicker="Diagnostic 4 · Representation probe")
    who(model_id)
    r = load_result(model_id, "D4")
    if not r:
        return
    d = r["details"]
    cz = d.get("causal") or {}
    c = st.columns([1.2, 1])
    with c[0]:
        kpi(f"{r['score']:.2f}", "board readable inside",
            "A simple reader tries to read all 64 squares from the model's insides. This is how much better it does than "
            "the same reader guessing from fake labels. Higher = the board is clearer inside the model.", DIAG_COLOR["D4"])
    with c[1]:
        used = cz.get("causal_label") == "decodable and causal"
        kpi("yes" if used else "no", "does it use it?",
            "We edit the board inside the model (flip one disc) and check whether its next moves follow the edit, compared "
            f"with a random edit of the same size. Here: {cz.get('boundary_acc_edit', 0):.0%} vs "
            f"{cz.get('boundary_acc_random', 0):.0%}. Only 10 of 81 zoo models pass.", "#7ee787" if used else "#ff7b72")
    gauge(r["score"], 0.0, 0.5, DIAG_COLOR["D4"], zoo_marks(scores, "D4", ref_score("D4")))
    left, right = st.columns([1, 1])
    with left:
        n, best = d["n_layers"], d["best_layer"]
        cols = "".join(
            f"<div class='col{' best' if k == best else ''}'><div class='b' style='--h:{max(4, 100 * (d['probe_acc_by_layer'][k] - 0.5) / 0.5):.0f}%;"
            f"background:{DIAG_COLOR['D4'] if k == best else 'rgba(47,191,143,.4)'};animation-delay:{0.12 * k:.2f}s'></div>"
            f"<div class='l'>layer {k}</div></div>" for k in range(n))
        html(f"<b>X-ray, layer by layer</b><div class='wm-note'>how much of the board a reader sees at each layer</div>"
             f"<div class='xray'>{cols}</div>")
    with right:
        dist = ANCHORS.get(model_id) or parse_id(model_id)["distribution"]
        ex = flip_case(dist, st.session_state.get("s4_pick", 0))
        if ex:
            bnd = ex["before"] ^ ex["after"]
            v = int(ex["board"][ex["sq"]])
            html("<b>The secret edit</b><div class='wm-note'>flip one disc: the glowing squares change from legal to "
                 "illegal or back</div>")
            html(board_html(ex["board"], legal=ex["after"], pulse=bnd, sq=34,
                            loop_flip={ex["sq"]: "wb" if v == rules.WHITE else "bw"}))
    how_it_works(flowchart_d4, model_id)


def simple_fragility(model_id: str, scores: pd.DataFrame) -> None:
    question("Does it break when the game looks different?", "Same rules, new situations: a rotated board, "
             "games from other players, a pass, forced detours.", GOLD, kicker="Fragility suite")
    who(model_id)
    fr = load_fragility(model_id, "v2")
    if not fr:
        return
    drop = fr["fragility"]
    kpi(f"{drop:+.3f}", "drop under new situations",
        "How much its share of legal moves drops, on average, when the game looks different (rotated board, other "
        "players' games, after a pass, forced detours). Around 0 or below = robust. Higher = it breaks. Most models "
        "are near 0; the ones trained on human games break on rotated boards.", "#ff7b72" if drop > 0.05 else GOLD)
    zr = scores.loc[scores["arch"] != "anchor", "FRAG"].dropna()
    gauge(fr["fragility"], -0.1, 0.2, "#ff7b72", [(float(zr.min()), "zoo most robust"), (float(zr.max()), "zoo most fragile")])
    left, right = st.columns([0.9, 1.1])
    with left:
        data = built_games(ANCHORS.get(model_id) or parse_id(model_id)["distribution"], "test", 50)
        if data is not None:
            html(board_html(data["black_white"][0, 20], sq=36, cls_extra="spin"))
            html("<div class='legend'>a rotated board: same game, same rules, looks new</div>")
    with right:
        groups = {"other dataset": "cross_distribution", "rotated / mirrored": "symmetry", "after a pass": "after_pass",
                  "forced detours": "detour"}
        rows = [{"shift": k, "drop": float(np.mean([v for kk, v in fr["drops"].items() if kk.startswith(p_)]))}
                for k, p_ in groups.items() if any(kk.startswith(p_) for kk in fr["drops"])]
        df = pd.DataFrame(rows)
        df["kind"] = np.where(df["drop"] > 0, "breaks", "fine")
        st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=6).encode(
            x=alt.X("drop:Q", title="drop in legal moves (positive = breaks)"), y=alt.Y("shift:N", sort=None, title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["breaks", "fine"], range=["#ff7b72", "#7ee787"]), legend=None),
            tooltip=["shift", alt.Tooltip("drop:Q", format="+.3f")]).properties(height=200), width="stretch")


def simple_concordance(model_id: str, scores: pd.DataFrame) -> None:
    question("Do the four tests agree?", "The project's main question, answered on all 81 models.", GOLD, kicker="Concordance · H1, H2, H3")
    s = load_analysis()
    if s is None:
        st.info("The analysis is not on this machine yet (git pull).")
        return
    h1 = s["H1"]
    c = st.columns(3)
    with c[0]:
        kpi(f"{h1['kendall_w']:.2f} → {h1['kendall_w_residual']:.2f}", "agreement",
            "Kendall's W: 1 = the four tests rank the 81 models identically, 0 = no agreement. Raw it is high, but once we "
            "remove 'how good the model is' (its legal move rate), it falls: the tests mostly agree because better trained "
            "models win every test.", GOLD, sub="raw → skill removed", size="2.6rem")
    with c[1]:
        kpi("noise", "why they disagree",
            "Where the tests disagree, it is not because of the brain type, the size or the training games: the "
            "disagreement is about as big as the random variation between repeat runs.", "#7ee787", sub="H2", size="2.6rem")
    with c[2]:
        kpi("none", "predicts breaking",
            "No test reliably predicts which models break on new situations. D4 looks like it does, but only because the "
            "models trained on human games are both the most fragile and the least readable.", "#ff7b72", sub="H3", size="2.6rem")
    st.markdown("&nbsp;")
    mode = st.segmented_control("Show agreement", ["raw", "with skill removed"], default="raw", key="conc_mode") or "raw"
    pairs = {p["pair"]: p for p in h1["pairs"]}
    grid = [f"<div class='agree' style='grid-template-columns: 70px repeat(4, 88px)'><div></div>"]
    grid += [f"<div class='hd'><span class='chip' style='background:{DIAG_COLOR[d_]}'></span>{d_}</div>" for d_ in DIAGS]
    for a in DIAGS:
        grid.append(f"<div class='hd'><span class='chip' style='background:{DIAG_COLOR[a]}'></span>{a}</div>")
        for b in DIAGS:
            if a == b:
                grid.append("<div class='a' style='background:rgba(255,255,255,.08);color:#a9c2b4'>–</div>")
                continue
            p = pairs.get(f"{min(a, b)}-{max(a, b)}")
            v = p["tau"] if mode == "raw" else p["partial_tau"]
            cat = p["category"] if mode == "raw" else p["partial_category"]
            ok = cat == "agree"
            colr = GOLD if ok else ("#f0c987" if cat == "partial" else "#ff7b72" if "disagree" in cat else "#7d8b83")
            word = "agree" if ok else ("partly" if cat == "partial" else "no" if "disagree" in cat else "n/a")
            grid.append(f"<div class='a' style='background:{colr};transition: background .4s'>{v:.2f}<small>{word}</small></div>")
    grid.append("</div>")
    html("".join(grid))
    st.caption("Kendall tau between two tests' rankings of the 81 models · gold = agree, red = do not · "
               "n/a: D1 is almost the same as skill, so it cannot be separated from it")


# ==========================================================================

PAGES = ["Model zoo", "D1 · Next-token validity", "D2 · State equivalence", "D3 · Inductive-bias probe",
         "D4 · Representation probe", "Fragility", "Concordance"]


def main() -> None:
    st.set_page_config(page_title="WM-Concord · Othello world-model lab", page_icon="⚫", layout="wide",
                       initial_sidebar_state="expanded")
    html(CSS + STEP_CSS)
    reg = load_registry()
    options = list(reg["model_id"]) if not reg.empty else []
    gpt_dir = Path(rules.DATA_DIR).parent.parent / "othello_gpt_tl"
    if (gpt_dir / "synthetic_model.pth").exists():
        options += list(ANCHORS)
    with st.sidebar:
        html("<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px'>"
             "<div class='logo'><span class='w'></span><span class='b'></span><span class='b'></span><span class='w'></span></div>"
             "<div><div class='wm-title' style='font-size:1.35rem;font-weight:700'>WM-Concord</div>"
             "<div class='wm-note'>Othello world-model lab</div></div></div>")
        if not options:
            st.error("No models found. Train one with src/zoo/train.py.")
            return
        def pick_random() -> None:
            st.session_state["model"] = str(np.random.default_rng().choice([o for o in options if o not in ANCHORS]))

        wanted = st.query_params.get("model")
        if wanted in options and st.session_state.get("_url_model") != wanted:
            st.session_state["model"] = wanted               # a click on the zoo board
            st.session_state["_url_model"] = wanted
        if "model" not in st.session_state:
            here = [m for m in options if m not in ANCHORS and has_weights(m)]
            best = "othello_transformer_large_synthetic_s0"
            st.session_state["model"] = (best if best in here else here[0]) if here else (best if best in options else options[0])
        model_id = st.selectbox("Model", options, format_func=pretty, key="model")
        st.query_params["model"] = model_id
        st.session_state["_url_model"] = model_id
        st.button("🎲 random model", on_click=pick_random, width="stretch")
        wanted_page = st.query_params.get("tab")
        if wanted_page and st.session_state.get("_url_tab") != wanted_page:
            match = [pg for pg in PAGES if pg.split(" ·")[0].lower() == wanted_page.lower()]
            if match:
                st.session_state["page"] = match[0]
            st.session_state["_url_tab"] = wanted_page
        page = st.radio("Page", PAGES, key="page")
        st.query_params["tab"] = page.split(" ·")[0]
        st.session_state["_url_tab"] = st.query_params["tab"]
    scores = load_scores(tuple(options))
    route = {"Model zoo": lambda: simple_zoo(model_id, reg, scores), "D1": lambda: simple_d1(model_id, scores),
             "D2": lambda: simple_d2(model_id, scores), "D3": lambda: simple_d3(model_id, scores),
             "D4": lambda: simple_d4(model_id, scores), "Fragility": lambda: simple_fragility(model_id, scores),
             "Concordance": lambda: simple_concordance(model_id, scores)}
    route[page.split(" ·")[0]]()

main()
