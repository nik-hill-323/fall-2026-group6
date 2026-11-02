"""Exploratory, not preregistered: whole game measures for the Othello zoo (Instructor Review #6).

The professor asked at the preliminary presentation whether our results only cover the next move.
D1 and the fragility suite are per move rates computed with teacher forcing: the model always sees
the real history, so one wrong guess never enters what it reads next. This script adds two whole
game views. It is decided after the preregistered Othello results were read, so everything it
writes is labelled exploratory and never replaces a preregistered result.

How a model predicts (all three architectures, src/zoo/train.py): it reads the moves so far, up to
59 (othello.N_CTX), and scores the 60 squares for the next move. There is no sliding window: a
whole game fits, so the history only grows and nothing is ever dropped.

1. Teacher forced, by move number. The same 1,000 test games of the model's own training
   distribution (data seed 0) and the same positions as D1. At every position the model's top
   move is checked against the true legal set. Reported per move number, plus the share of test
   games in which every top move along the real game is legal. The overall rate equals the D1
   score, which the script checks.
2. Free play. The model plays 1,000 whole games against itself from its own moves (play seed 0).
   The first move is a uniformly random legal move (the model is never asked to open), as in
   fragility v2. At each later move the model's top move is checked against the true rules (an
   illegal top move is counted as an error). The move actually played is drawn from the model's
   own probabilities over the legal moves only (play seed 0), so every game is the model's own
   game and continues to its natural end. Version 1 always played the top legal move instead;
   because that choice is deterministic, its 1,000 games collapsed into at most 4 distinct games
   (one per opening move), so v1 results are not used. Passes are not tokens (as in Othello GPT):
   when the player to move has no legal move the other player moves again. Reported: legal rate
   per move number, the share of games with no illegal top move at all (full game rate), errors per
   game, and the move number of the first error.

Run on the EC2 from the repo root (conda env wmconcord), inside tmux:
    python src/analysis/sequence_othello.py --all --anchors            # every zoo model found here
    python src/analysis/sequence_othello.py --models othello_lstm_small_synthetic_s0
    python src/analysis/sequence_othello.py --summarize                # table, summary, figures
Writes results/exploratory/sequence/{model_id}.json (one per model; existing files are skipped
unless --overwrite), then with --summarize results/analysis/othello/sequence.csv,
sequence_summary.json, reports/figures/othello/sequence_by_move.svg and
reports/Latex_report/fig/othello_sequence.pdf.

Smoke test anywhere (tiny untrained transformer, no data needed):
    python src/analysis/sequence_othello.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger("sequence")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "exploratory" / "sequence"
ANALYSIS_DIR = ROOT / "results" / "analysis" / "othello"
FIG_DIR = ROOT / "reports" / "figures" / "othello"
LATEX_FIG_DIR = ROOT / "reports" / "Latex_report" / "fig"
VERSION = "sequence-v2"   # v1 played the top legal move: only 4 distinct games (see docstring)
N_GAMES = 1000
DATA_SEED = 0
PLAY_SEED = 0
BATCH = 256   # the same as D1 (src/diagnostics/d1/next_token.py), so the teacher forced rate matches D1 exactly
ANCHORS = ["othello_gpt_synthetic", "othello_gpt_championship"]
MOVES = 61   # index = move number 1 to 60; index 0 unused


def _load(name: str, rel: str):
    """Import a file from src/ by path (src/ is not a package), shared through sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oth = _load("othello", "src/domains/othello.py")


# ==========================================================================
# 1. Teacher forced, by move number
# ==========================================================================

@torch.no_grad()
def teacher_forced(model, data: dict, batch: int = BATCH) -> dict[str, Any]:
    """Top move legality at every position of the real games, by the move number predicted.
    Position t (0 based) holds moves 1 to t+1, so its prediction is move t+2. Same positions as
    othello.legal_rate (the last move of each game has no next move and is skipped)."""
    ok = np.zeros(MOVES, dtype=np.int64)
    n = np.zeros(MOVES, dtype=np.int64)
    clean_games = 0
    G = len(data["tokens"])
    for i in range(0, G, batch):
        pred = model(torch.as_tensor(data["tokens"][i:i + batch])).argmax(-1).cpu().numpy()
        for b in range(len(pred)):
            g = i + b
            T = int(data["mask"][g].sum())
            t = np.arange(T - 1)
            legal = data["legal_next"][g, t, pred[b, t]]
            np.add.at(ok, t + 2, legal.astype(np.int64))
            np.add.at(n, t + 2, 1)
            clean_games += int(legal.all())
    return {"overall": float(ok.sum() / max(n.sum(), 1)),
            "full_game_rate": clean_games / max(G, 1),
            "n_games": G, "n_positions": int(n.sum()),
            "by_move": {"legal": ok.tolist(), "n": n.tolist()}}


# ==========================================================================
# 2. Free play
# ==========================================================================

@torch.no_grad()
def free_play(model, n_games: int = N_GAMES, seed: int = PLAY_SEED, device: str | None = None,
              batch: int = BATCH) -> dict[str, Any]:
    """The model plays whole games from its own moves (module docstring, part 2). Games are run
    through the model `batch` at a time; the batch size changes memory use, not the result."""
    rng = np.random.default_rng(seed)
    boards = [oth.start_board() for _ in range(n_games)]
    to_move = [oth.BLACK] * n_games
    tokens = np.zeros((n_games, oth.N_CTX), dtype=np.int64)
    alive = np.ones(n_games, dtype=bool)
    length = np.ones(n_games, dtype=np.int64)
    errors = np.zeros(n_games, dtype=np.int64)
    first_error = np.zeros(n_games, dtype=np.int64)        # 0 = no error in the game
    ok = np.zeros(MOVES, dtype=np.int64)
    n = np.zeros(MOVES, dtype=np.int64)

    def play(g: int, sq: int, t: int) -> None:
        player = to_move[g]
        boards[g][oth.flips_for(boards[g], sq, player)] = player
        boards[g][sq] = player
        tokens[g, t] = oth.SQ_TO_TOKEN[sq]
        to_move[g] = oth.other(player)

    for g in range(n_games):
        play(g, int(rng.choice(oth.legal_moves(boards[g], oth.BLACK))), 0)

    # t moves played so far; the model reads all t of them (at most 59) and predicts move t+1
    for t in range(1, oth.N_CTX + 1):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        # Who moves next, and is the game over? (a pass gives the turn back; no pass token)
        live = []
        for g in idx:
            legal = oth.legal_moves(boards[g], to_move[g])
            if not legal:
                to_move[g] = oth.other(to_move[g])
                legal = oth.legal_moves(boards[g], to_move[g])
                if not legal:
                    alive[g] = False
                    continue
            live.append((g, legal))
        if not live:
            break
        rows = np.array([g for g, _ in live])
        probs = np.concatenate([
            torch.softmax(model(torch.as_tensor(tokens[rows[i:i + batch], :t]))[:, -1].float(), dim=-1).cpu().numpy()
            for i in range(0, len(rows), batch)])
        for r, (g, legal) in enumerate(live):
            top_sq = oth.TOKEN_TO_SQ.get(int(probs[r].argmax()))
            mv = t + 1
            n[mv] += 1
            if top_sq in legal:
                ok[mv] += 1
            else:
                errors[g] += 1
                if first_error[g] == 0:
                    first_error[g] = mv
            # The move played: drawn from the model's probabilities over the legal moves only
            w = np.array([probs[r, oth.SQ_TO_TOKEN[s]] for s in legal], dtype=np.float64)
            w = w / w.sum() if w.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
            sq = int(legal[int(rng.choice(len(legal), p=w))])
            if t < oth.N_CTX:
                play(g, sq, t)
                length[g] = t + 1
            else:                                           # move 60 checked: the board is full
                alive[g] = False
                length[g] = oth.MAX_MOVES
    had_error = first_error > 0
    distinct = len({tokens[g].tobytes() for g in range(n_games)})
    return {"n_distinct_games": distinct,"overall": float(ok.sum() / max(n.sum(), 1)),
            "full_game_rate": float((~had_error).mean()),
            "mean_errors_per_game": float(errors.mean()),
            "mean_first_error_move": float(first_error[had_error].mean()) if had_error.any() else None,
            "median_first_error_move": float(np.median(first_error[had_error])) if had_error.any() else None,
            "mean_game_length": float(length.mean()),
            "n_games": n_games, "n_predictions": int(n.sum()),
            "by_move": {"legal": ok.tolist(), "n": n.tolist()}}


# ==========================================================================
# One model
# ==========================================================================

def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def uncommitted_code() -> list[str]:
    """Files under src/ that differ from the commit: results must come from committed code, so that the
    git_commit they record is the code that produced them (PR #131 review: v2 results recorded the v1 commit)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ["(git status failed)"]
    return [line[3:] for line in out.splitlines() if line.strip()]


def distribution_of(model) -> str:
    results = _load("results", "src/component/results.py")
    row = results.load_models().get(model.model_id)
    return row["distribution"] if row else model.meta["distribution"]


def measure(model, n_games: int = N_GAMES, data_seed: int = DATA_SEED, play_seed: int = PLAY_SEED,
            data: dict | None = None, batch: int = BATCH) -> dict[str, Any]:
    t0 = time.time()
    dist = distribution_of(model)
    if data is None:
        data = oth.build(oth.load_distribution(dist, "test", n_games, seed=data_seed), legal=True)
    tf = teacher_forced(model, data, batch=batch)
    fp = free_play(model, n_games=n_games, seed=play_seed, batch=batch)
    d1_path = ROOT / "results" / "D1_next_token" / f"{model.model_id}.json"
    d1 = json.loads(d1_path.read_text()).get("score") if d1_path.exists() else None
    return {"model_id": model.model_id, "exploratory": True, "version": VERSION,
            "note": "exploratory, not preregistered (Instructor Review #6); see the module docstring",
            "distribution": dist, "split": "test", "data_seed": data_seed, "play_seed": play_seed,
            "teacher_forced": tf, "free_play": fp,
            "d1_score": d1,
            "matches_d1": None if d1 is None else bool(abs(d1 - tf["overall"]) < 1e-6),
            "provenance": {"git_commit": _commit(), "uncommitted_src": uncommitted_code(), "device": str(model.device),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                           "torch": torch.__version__, "host": platform.node(),
                           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "runtime_sec": round(time.time() - t0, 1)}


def load_any(model_id: str):
    itf = _load("interface", "src/diagnostics/interface.py")
    if model_id.startswith("othello_gpt_"):
        return itf.othello_gpt(model_id.removeprefix("othello_gpt_"))
    reg = _load("registry", "src/zoo/registry.py")
    return reg.load_model(model_id)          # checks the checkpoint against its training record


def zoo_ids_here() -> list[str]:
    """Zoo models with a training record and a checkpoint on this machine, in models.csv order."""
    results = _load("results", "src/component/results.py")
    ids = []
    for mid, m in results.load_models().items():
        rec = ROOT / "results" / "zoo_training" / f"{mid}.json"
        if m["domain"] != "othello" or not rec.exists():
            continue
        ckpt = ROOT / json.loads(rec.read_text()).get("checkpoint", f"outputs/zoo/{mid}/model.pt")
        if ckpt.exists():
            ids.append(mid)
        else:
            log.warning("[skip] %s: checkpoint %s is not on this machine", mid, ckpt)
    return ids


# ==========================================================================
# Summary: table, correlations, figures
# ==========================================================================

def _rate(by_move: dict) -> np.ndarray:
    legal, n = np.array(by_move["legal"], float), np.array(by_move["n"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, legal / n, np.nan)


def summarize() -> dict[str, Any]:
    hyp = _load("hypotheses", "src/analysis/hypotheses.py")
    table = {r["model_id"]: r for r in csv.DictReader(open(ANALYSIS_DIR / "model_table.csv"))}
    seq = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))}
    rows = []
    for mid, t in table.items():
        s = seq.get(mid)
        if s is None:
            continue
        r = {"model_id": mid, "arch": t["arch"], "scale": t["scale"], "distribution": t["distribution"],
             "seed": int(t["seed"]), "competence": float(t["competence"]),
             **{d: float(t[d]) for d in hyp.DIAGS}, "rel_frag_v2": float(t["rel_frag_v2"]),
             "tf_overall": s["teacher_forced"]["overall"], "tf_full_game": s["teacher_forced"]["full_game_rate"],
             "fp_overall": s["free_play"]["overall"], "fp_full_game": s["free_play"]["full_game_rate"],
             "fp_errors_per_game": s["free_play"]["mean_errors_per_game"],
             "fp_first_error": s["free_play"]["mean_first_error_move"],
             "matches_d1": s["matches_d1"]}
        rows.append(r)
    missing = [m for m in table if m not in seq]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(ANALYSIS_DIR / "sequence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out: dict[str, Any] = {"exploratory": True, "version": VERSION, "n_models": len(rows),
                           "missing": missing, "matches_d1_all": all(r["matches_d1"] is not False for r in rows)}
    # Means by architecture and scale, and by training data
    groups: dict[str, dict[str, Any]] = {}
    for key in (("arch", "scale"), ("distribution",)):
        for r in rows:
            k = " ".join(r[c] for c in key)
            groups.setdefault(k, []).append(r)
    out["means"] = {k: {c: float(np.mean([r[c] for r in v if r[c] is not None]))
                        for c in ("competence", "tf_overall", "tf_full_game", "fp_overall", "fp_full_game",
                                  "fp_errors_per_game")} | {"n": len(v)}
                    for k, v in groups.items()}
    # Does the whole game view rank models like the next move view? (configuration bootstrap)
    if len(rows) >= 9:
        idx = hyp.boot_indices(rows)
        corr = {}
        for x in ("competence", "D1", "D2", "D3", "D4", "rel_frag_v2"):
            for y in ("fp_full_game", "tf_full_game", "fp_overall"):
                a, b = hyp.column(rows, x), hyp.column(rows, y)
                est = hyp.spearman(a, b)
                ci = hyp.interval(hyp.boot(lambda i: hyp.spearman(a[i], b[i]), idx))
                corr[f"{x} vs {y}"] = {"spearman": est, "ci": ci}
        # Partial: does full game rate carry anything beyond the per move rate (D1)?
        c, d1 = hyp.column(rows, "fp_full_game"), hyp.column(rows, "D1")
        for x in ("D2", "D3", "D4"):
            a = hyp.column(rows, x)
            est = hyp.partial_spearman(a, c, d1[:, None])
            ci = hyp.interval(hyp.boot(lambda i: hyp.partial_spearman(a[i], c[i], d1[i][:, None]), idx))
            corr[f"{x} vs fp_full_game given D1"] = {"partial_spearman": est, "ci": ci}
        out["correlations"] = corr
    (ANALYSIS_DIR / "sequence_summary.json").write_text(json.dumps(hyp._clean(out), indent=2) + "\n")
    if rows:
        _figure(seq, rows)
    return out


def _figure(seq: dict[str, dict], rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hyp = _load("hypotheses", "src/analysis/hypotheses.py")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "svg.fonttype": "none", "svg.hashsalt": "wmconcord"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    styles = {"small": ":", "medium": "--", "large": "-"}
    moves = np.arange(MOVES)
    for ax, part, title in ((axes[0], "teacher_forced", "Teacher forced (real history)"),
                            (axes[1], "free_play", "Free play (own moves)")):
        for arch, color in hyp.ARCH_COLOR.items():
            for scale, ls in styles.items():
                ids = [r["model_id"] for r in rows if r["arch"] == arch and r["scale"] == scale]
                if not ids:
                    continue
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # move numbers 0 and 1 are never predicted
                    curve = np.nanmean([_rate(seq[i][part]["by_move"]) for i in ids], axis=0)
                ax.plot(moves[2:], curve[2:], ls=ls, color=color, lw=1.3,
                        label=f"{arch} {scale}" if part == "free_play" else None)
        ax.set_title(title)
        ax.set_xlabel("move number")
        ax.set_xlim(2, 60)
    axes[0].set_ylabel("share of top moves that are legal")
    axes[1].legend(fontsize=7, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "sequence_by_move.svg", metadata={"Date": None})
    fig.savefig(LATEX_FIG_DIR / "othello_sequence.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# ==========================================================================

def smoke() -> None:
    """Tiny untrained transformer on 50 random games, CPU: checks the code path, not the numbers."""
    train = _load("zoo_train", "src/zoo/train.py")
    itf = _load("interface", "src/diagnostics/interface.py")
    module = train.GPT(n_layers=1, d_model=16, n_heads=2)
    model = itf.TrainPyModel(module, "smoke", "transformer", 1, 16, "cpu",
                             {"domain": "othello", "arch": "transformer", "scale": "smoke",
                              "distribution": "synthetic", "seed": 0})
    data = oth.build(oth.random_games(50, seed=0), legal=True)
    res = measure(model, n_games=50, data=data)
    print(json.dumps({k: res[k] for k in ("model_id", "matches_d1")} |
                     {"tf": {k: v for k, v in res["teacher_forced"].items() if k != "by_move"},
                      "fp": {k: v for k, v in res["free_play"].items() if k != "by_move"}}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory whole game measures for the Othello zoo")
    ap.add_argument("--models", nargs="*", default=[], help="zoo model ids or othello_gpt_synthetic/championship")
    ap.add_argument("--all", action="store_true", help="every zoo model whose checkpoint is on this machine")
    ap.add_argument("--anchors", action="store_true", help="also the two reference Othello GPTs")
    ap.add_argument("--n_games", type=int, default=N_GAMES)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="games per forward pass; lower it if the GPU runs out of memory (result unchanged)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_uncommitted", action="store_true",
                    help="run even if files under src/ are not committed (tests only; never for results)")
    ap.add_argument("--summarize", action="store_true", help="only build the table, summary and figures")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.smoke:
        smoke()
        return
    if args.summarize:
        out = summarize()
        print(json.dumps({k: out[k] for k in ("n_models", "missing", "matches_d1_all")}, indent=2))
        return
    ids = list(args.models) + (zoo_ids_here() if args.all else []) + (ANCHORS if args.anchors else [])
    if not ids:
        sys.exit("nothing to do: pass --models, --all and/or --anchors (or --summarize)")
    dirty = uncommitted_code()
    if dirty and not args.allow_uncommitted:
        sys.exit(f"refusing to run: these files under src/ are not committed: {dirty}. Commit them first, so "
                 "that the git_commit each result records is the code that produced it.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = 0
    for mid in ids:
        path = OUT_DIR / f"{mid}.json"
        if path.exists() and not args.overwrite:
            log.info("[skip] %s (exists)", mid)
            skipped += 1
            continue
        try:
            model = load_any(mid)
            res = measure(model, n_games=args.n_games, batch=args.batch)
        except Exception as e:   # one bad model must not stop the sweep
            log.warning("[FAIL] %s: %s: %s", mid, type(e).__name__, e)
            failed += 1
            continue
        finally:
            model = None                       # free this model's GPU memory before the next one
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        path.write_text(json.dumps(res, indent=2) + "\n")
        log.info("[done] %s  tf %.4f (D1 match %s)  free play %.4f  full games %.3f  %.0f s", mid,
                 res["teacher_forced"]["overall"], res["matches_d1"], res["free_play"]["overall"],
                 res["free_play"]["full_game_rate"], res["runtime_sec"])
        done += 1
    log.info("%d done, %d skipped, %d failed", done, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
