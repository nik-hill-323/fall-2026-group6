"""WM-Concord demo: pick a zoo model, watch each diagnostic run on it, see the metrics.

Run from the repo root (on a machine that has the checkpoints and the Othello data):

    streamlit run demo/app.py

Pages
    Model zoo     the registry, the selected model's training record, which results exist for it
    D1 live       replay a real game move by move: true board, legal moves, the model's pick
    D2 / D3 / D4  the saved result for the selected model, or what the test will show once scored
    Fragility     the suite's drops for the selected model
    Concordance   master table and rank agreement between diagnostics, once models are scored

Everything is read from the repo: src/docs/zoo_registry.csv, results/zoo_training/, results/D*/,
results/fragility/, and checkpoints under outputs/zoo/. Nothing is written.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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

DIAG_TITLES = {
    "D1": "D1 next token validity",
    "D2": "D2 state equivalence (Myhill Nerode)",
    "D3": "D3 inductive bias probe",
    "D4": "D4 representation probing",
}
DIAG_BLURB = {
    "D1": "Does the model only ever propose legal moves? Score: share of positions where its top "
          "predicted move is legal according to the rules.",
    "D2": "Does the model treat two histories that reach the same board the same way, and different "
          "boards differently? Score: 1 minus the mean of compression and distinction error (Vafa et al. 2024).",
    "D3": "When adapted to a new task from a handful of examples, does the model learn it as if it knew "
          "the board? Score: adaptation gain over a randomly initialised model of the same size (Vafa et al. 2025).",
    "D4": "Can a linear probe read the board from the model's activations, and does it beat a control "
          "probe? Score: selectivity at the best layer, chosen on validation, measured on test.",
}
ANCHORS = {"othello_gpt_synthetic": "synthetic", "othello_gpt_championship": "championship"}


# ==========================================================================
# Data access (cached)
# ==========================================================================

@st.cache_data
def load_registry() -> pd.DataFrame:
    path = ROOT / "src/docs/zoo_registry.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_record(model_id: str) -> dict | None:
    path = ROOT / "results/zoo_training" / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_result(model_id: str, diagnostic: str) -> dict | None:
    path = ROOT / "results" / results_mod.diag_folder(diagnostic) / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_fragility(model_id: str) -> dict | None:
    path = ROOT / "results/fragility" / f"{model_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_games(distribution: str, split: str, n: int) -> np.ndarray | None:
    try:
        return rules.load_distribution(distribution, split, n, seed=0)
    except (FileNotFoundError, ValueError):
        return None


@st.cache_resource
def load_model(model_id: str):
    """A ZooModel, or an error string if the checkpoint or weights are not on this machine."""
    try:
        if model_id in ANCHORS:
            itf = _load("interface", "src/diagnostics/interface.py")
            return itf.othello_gpt(ANCHORS[model_id], device="cpu")
        return registry.load_model(model_id, device="cpu")
    except Exception as e:  # noqa: BLE001  (shown to the user, not hidden)
        return f"{type(e).__name__}: {e}"


# ==========================================================================
# Board drawing
# ==========================================================================

def board_html(board: np.ndarray, last: int | None, legal: set[int], pick: int | None) -> str:
    """8x8 board as an HTML table. Black and white discs, green dot = legal next move,
    blue frame = model's pick if legal, red frame if illegal, yellow frame = last move."""
    cells = []
    for s in range(rules.N_SQ):
        v = int(board[s])
        disc = "&#9679;" if v == rules.BLACK else ("&#9675;" if v == rules.WHITE else "")
        style = "width:34px;height:34px;text-align:center;font-size:22px;border:1px solid #888;background:#2e7d32;"
        if s == last:
            style += "box-shadow: inset 0 0 0 3px #ffd54f;"
        if s == pick:
            style += "box-shadow: inset 0 0 0 3px %s;" % ("#2196f3" if s in legal else "#e53935")
        if v == rules.EMPTY and s in legal:
            disc = "<span style='font-size:12px;color:#b9f6ca'>&#9679;</span>"
        if v == rules.WHITE:
            disc = "<span style='color:white'>&#9679;</span>"
        if v == rules.BLACK:
            disc = "<span style='color:black'>&#9679;</span>"
        cells.append(f"<td style='{style}'>{disc}</td>")
    rows = ["<tr>" + "".join(cells[r * 8:(r + 1) * 8]) + "</tr>" for r in range(8)]
    header = "<tr>" + "".join(f"<td style='text-align:center;color:#888'>{c}</td>" for c in "ABCDEFGH") + "</tr>"
    return "<table style='border-collapse:collapse'>" + header + "".join(rows) + "</table>"


def sq_name(s: int) -> str:
    return "ABCDEFGH"[s % 8] + str(s // 8 + 1)


# ==========================================================================
# Pages
# ==========================================================================

def page_zoo(model_id: str, reg: pd.DataFrame) -> None:
    st.header("Model zoo")
    st.caption("One row per trained model, from src/docs/zoo_registry.csv. Admission: validation "
               "legal move rate at least 0.50 (reports/preregistration.md).")
    if reg.empty:
        st.warning("No registry yet. Train a model with src/zoo/train.py.")
        return
    show = reg[["model_id", "arch", "scale", "distribution", "seed", "n_params", "val_legal_rate",
                "train_minutes", "admitted"]].copy()
    st.dataframe(show.style.apply(lambda r: ["background:#1e3a5f" if r["model_id"] == model_id else ""
                                             for _ in r], axis=1), width="stretch", hide_index=True)
    planned = len(pd.read_csv(ROOT / "src/docs/models.csv")) if (ROOT / "src/docs/models.csv").exists() else 0
    othello = reg[reg["domain"] == "othello"] if "domain" in reg else reg
    c1, c2, c3 = st.columns(3)
    c1.metric("Trained", len(reg), help=f"of {planned} planned")
    c2.metric("Admitted", int((reg["admitted"] == "yes").sum()))
    c3.metric("Othello trained", len(othello), help="of 81")

    st.subheader(f"Selected: {model_id}")
    rec = load_record(model_id)
    if rec is None:
        st.info("Reference Othello GPT (anchor): scored for comparison, never part of the statistics."
                if model_id in ANCHORS else "No training record for this model.")
    else:
        c = st.columns(5)
        c[0].metric("Legal move rate", f"{rec['eval']['legal_move_rate']:.3f}")
        c[1].metric("Next token acc", f"{rec['eval']['next_token_acc']:.3f}")
        c[2].metric("Final loss", f"{rec['train']['final_loss']:.3f}")
        c[3].metric("Params", f"{rec['n_params'] / 1e6:.2f}M")
        c[4].metric("Train minutes", f"{rec['train']['wall_sec'] / 60:.1f}")
        st.caption(f"{rec['scale_config']}  ·  {rec['train']['steps_run']} steps  ·  {rec.get('gpu', '')}  ·  "
                   f"commit {rec.get('git_commit', '?')}  ·  {rec.get('time_utc', '')}")

    st.subheader("Results on file for this model")
    rows = []
    for d in results_mod.DIAGNOSTICS:
        r = load_result(model_id, d)
        rows.append({"diagnostic": DIAG_TITLES[d], "status": r["status"] if r else "not scored",
                     "score": r.get("score") if r else None, "method": r.get("method", "") if r else ""})
    fr = load_fragility(model_id)
    rows.append({"diagnostic": "Fragility suite v1", "status": "ok" if fr else "not scored",
                 "score": fr["fragility"] if fr else None, "method": "mean drop, higher = more fragile"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_d1(model_id: str) -> None:
    st.header(DIAG_TITLES["D1"])
    st.write(DIAG_BLURB["D1"])
    model = load_model(model_id)
    if isinstance(model, str):
        st.error(f"Model not available on this machine: {model}")
        return

    col = st.columns(3)
    dist = col[0].selectbox("Games from", rules.DISTRIBUTIONS, index=0)
    split = col[1].selectbox("Split", ["test", "val"], index=0)
    games = load_games(dist, split, 50)
    if games is None:
        st.error(f"No {dist} {split} games on this machine. Run src/shellscripts/fetch_othello_data.sh.")
        return
    g = col[2].number_input("Game", 0, len(games) - 1, 0)
    game = games[g]
    data = rules.build(game[None], legal=True)
    if data["n_illegal_games"]:
        st.error("This recorded game has an illegal move; pick another.")
        return
    tokens, mask = data["tokens"], data["mask"]
    T = int(mask[0].sum())
    logits = model.logits(tokens)[0].cpu().numpy()                 # (59, 61)
    preds = logits.argmax(-1)                                        # predicted next move at each position
    legal_next = data["legal_next"][0]                               # (59, 61) bool
    scored = np.arange(T - 1)
    legal_hits = legal_next[scored, preds[scored]]
    rate = float(legal_hits.mean()) if len(scored) else float("nan")

    t = st.slider("After move", 1, T, 1) - 1
    board = data["black_white"][0, t]
    mover = int(data["movers"][0, t])
    legal = {s for s in rules.legal_next(board, mover)}
    pick_tok = int(preds[t])
    pick = rules.TOKEN_TO_SQ.get(pick_tok)
    last = rules.TOKEN_TO_SQ[int(tokens[0, t])]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(board_html(board, last, legal, pick), unsafe_allow_html=True)
        st.caption("Yellow: move just played. Small green: legal next moves. Blue frame: the model's pick "
                   "(legal). Red frame: the model's pick (illegal).")
    with right:
        st.markdown(f"**Moves so far:** {' '.join(sq_name(rules.TOKEN_TO_SQ[int(x)]) for x in tokens[0, :t + 1])}")
        st.markdown(f"**Just moved:** {'black' if mover == rules.BLACK else 'white'}  ·  "
                    f"**legal next moves:** {', '.join(sq_name(s) for s in sorted(legal)) or 'none (game over)'}")
        ok = pick in legal
        st.markdown(f"**Model's pick:** {sq_name(pick) if pick is not None else 'pad'}  ·  "
                    f"{'legal' if ok else 'ILLEGAL'}")
        probs = np.exp(logits[t] - logits[t].max())
        probs /= probs.sum()
        top = np.argsort(-probs)[:5]
        st.dataframe(pd.DataFrame({
            "move": [sq_name(rules.TOKEN_TO_SQ[int(k)]) if int(k) in rules.TOKEN_TO_SQ else "pad" for k in top],
            "probability": [round(float(probs[k]), 3) for k in top],
            "legal": [bool(legal_next[t, k]) for k in top],
        }), hide_index=True, width="stretch")
        if t < T - 1:
            actual = rules.TOKEN_TO_SQ[int(tokens[0, t + 1])]
            st.caption(f"The recorded next move was {sq_name(actual)}. The model does not need to match it, "
                       "only to pick a legal move.")

    st.subheader("Over this game")
    st.metric("Legal move rate, this game", f"{rate:.3f}", help=f"{int(legal_hits.sum())} legal of {len(scored)} predictions")
    st.line_chart(pd.DataFrame({"legal (1) / illegal (0)": legal_hits.astype(int)}, index=scored + 1))
    r = load_result(model_id, "D1")
    if r:
        st.success(f"Saved D1 score on {r['n_games']} {r['split']} games: {r['score']:.4f}  ({r['method']})")
    else:
        rec = load_record(model_id)
        if rec:
            st.info(f"Validation legal move rate from training, 1,000 games: {rec['eval']['legal_move_rate']:.4f}. "
                    "The D1 module has not scored this model on the test split yet.")


def page_result(model_id: str, diagnostic: str) -> None:
    st.header(DIAG_TITLES[diagnostic])
    st.write(DIAG_BLURB[diagnostic])
    r = load_result(model_id, diagnostic)
    if r is None:
        st.info("Not scored yet for this model. The result will appear here once the diagnostic module "
                f"writes results/{results_mod.diag_folder(diagnostic)}/{model_id}.json.")
        if diagnostic == "D3":
            st.caption("Toy version: python src/diagnostics/d3/toy_probe.py --seeds 0 1 2 3 4. "
                       "Explainer: src/docs/d3_inductive_bias_probe.md.")
        if diagnostic == "D4":
            st.caption("Toy version: python src/diagnostics/d4/toy_probe.py. Reproduction of Nanda et al. "
                       "2023: src/docs/d4_reproduction.md.")
        return
    c = st.columns(4)
    c[0].metric("Score", f"{r['score']:.4f}" if r.get("score") is not None else "–")
    c[1].metric("Status", r["status"])
    c[2].metric("Split", r["split"])
    c[3].metric("Games", r["n_games"])
    st.caption(r.get("method", ""))
    if r.get("score_ci"):
        st.caption(f"95% interval: {r['score_ci']}")
    details = r.get("details") or {}
    # Per layer numbers (D4) or per task numbers (D3) as a chart when they are lists.
    series = {k: v for k, v in details.items() if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v)}
    if series:
        n = max(len(v) for v in series.values())
        df = pd.DataFrame({k: v + [None] * (n - len(v)) for k, v in series.items()})
        df.index.name = "layer" if diagnostic == "D4" else "index"
        st.line_chart(df)
    with st.expander("Full result"):
        st.json(r)


def page_fragility(model_id: str) -> None:
    st.header("Fragility suite v1")
    st.write("Same Othello rules, shifted situations. For each variant: the model's legal move rate, and "
             "its drop from the in distribution rate. Fragility = mean drop. Higher = more fragile. "
             "This is the outcome H3 predicts. Frozen with the tag fragility-v1.")
    fr = load_fragility(model_id)
    if fr is None:
        st.info("Not scored yet. Run: python src/fragility/othello.py --model outputs/zoo/<id>/model.pt "
                "--arch <arch> --scale <scale> --train_distribution <dist>")
        st.caption("Definition: src/docs/fragility_suite.md")
        return
    c = st.columns(3)
    c[0].metric("In distribution legal rate", f"{fr['in_distribution_legal_rate']:.4f}")
    c[1].metric("Fragility (mean drop)", f"{fr['fragility']:.4f}")
    c[2].metric("Suite", f"{fr['suite_version']} ({fr['suite_hash']})")
    drops = pd.Series(fr["drops"], name="drop in legal move rate").sort_values()
    st.bar_chart(drops)
    st.subheader("Detours: self play with forced alternatives")
    det = pd.DataFrame({k: {"completion rate": v["completion_rate"], "legal rate": v["legal_rate"],
                            "detours": v["n_detours"]} for k, v in fr["detour"].items()}).T
    st.dataframe(det, width="stretch")
    with st.expander("Full result"):
        st.json(fr)


def page_concordance(reg: pd.DataFrame) -> None:
    st.header("Concordance across diagnostics (H1)")
    st.write("One row per admitted model, one score per diagnostic. Rank correlations between diagnostics "
             "say whether the four tests agree on which models have a world model.")
    try:
        rows = results_mod.master_table()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the master table: {e}")
        return
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No results yet.")
        return
    score_cols = [c for c in df.columns if c.endswith("_score")]
    scored = df.dropna(subset=score_cols, how="all") if score_cols else df.iloc[0:0]
    st.caption(f"{len(scored)} of {len(df)} planned models have at least one score.")
    if "admitted" in reg and not reg.empty:
        adm = set(reg.loc[reg["admitted"] == "yes", "model_id"])
        scored = scored[scored["model_id"].isin(adm)]
        st.caption(f"{len(scored)} of those are admitted (legal move rate at least 0.50).")
    if scored.empty:
        st.info("Nothing to correlate yet. Scores appear here as the diagnostic modules run on the zoo.")
        return
    st.dataframe(scored[["model_id"] + score_cols], width="stretch", hide_index=True)
    have = [c for c in score_cols if scored[c].notna().sum() >= 3]
    if len(have) >= 2:
        from scipy.stats import kendalltau, spearmanr
        names = [c.replace("_score", "") for c in have]
        tau = pd.DataFrame(index=names, columns=names, dtype=float)
        rho = tau.copy()
        for i, a in enumerate(have):
            for j, b in enumerate(have):
                both = scored[[a, b]].dropna()
                tau.iloc[i, j] = kendalltau(both[a], both[b]).statistic if len(both) >= 3 else np.nan
                rho.iloc[i, j] = spearmanr(both[a], both[b]).statistic if len(both) >= 3 else np.nan
        c1, c2 = st.columns(2)
        c1.subheader("Kendall tau")
        c1.dataframe(tau.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"), width="stretch")
        c2.subheader("Spearman rho")
        c2.dataframe(rho.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"), width="stretch")
        st.caption("Preregistered rule: a pair agrees if tau is at least 0.50 and its bootstrap interval excludes 0; "
                   "disagrees if the interval includes 0 or tau is below 0.30; partial otherwise. Intervals come "
                   "with the analysis code, not this page.")
    else:
        st.info("Correlations need at least two diagnostics scored on at least three admitted models.")


# ==========================================================================

def main() -> None:
    st.set_page_config(page_title="WM-Concord demo", layout="wide")
    st.sidebar.title("WM-Concord")
    st.sidebar.caption("Do world model diagnostics agree? Pick a model, watch each test run on it.")
    reg = load_registry()
    options = list(reg["model_id"]) if not reg.empty else []
    gpt_dir = Path(rules.DATA_DIR).parent.parent / "othello_gpt_tl"
    if (gpt_dir / "synthetic_model.pth").exists():
        options += list(ANCHORS)
    if not options:
        st.sidebar.error("No models found. Train one with src/zoo/train.py.")
        return
    model_id = st.sidebar.selectbox("Model", options, index=0)
    rec = load_record(model_id)
    if rec:
        st.sidebar.caption(f"{rec['arch']} · {rec['scale']} · {rec['config']['distribution']} · seed {rec['config']['seed']}  "
                           f"· legal move rate {rec['eval']['legal_move_rate']:.3f}")
    page = st.sidebar.radio("Page", ["Model zoo", "D1 live", "D2", "D3", "D4", "Fragility", "Concordance"])

    if page == "Model zoo":
        page_zoo(model_id, reg)
    elif page == "D1 live":
        page_d1(model_id)
    elif page in ("D2", "D3", "D4"):
        page_result(model_id, page)
    elif page == "Fragility":
        page_fragility(model_id)
    else:
        page_concordance(reg)


main()
