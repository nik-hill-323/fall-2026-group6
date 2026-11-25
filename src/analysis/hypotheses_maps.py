"""Navigation analysis: H1, H2 and H3 on the navigation zoo, as reports/preregistration.md fixes them for navigation
(Amendment 5, G10 to G12; Amendment 6, H6; Amendment 7). Ticket N14.

Everything H1 to H3 computes is the Othello code (src/analysis/hypotheses.py, Section 4, Amendment 1, Amendment 3
E4, Amendment 4 F1 and F3), called on the navigation rows. This file only says which navigation files are read and
adds what Amendments 5 and 6 add for navigation. Nothing in src/analysis/hypotheses.py or in the Othello outputs
changes.

Inputs (all from the repo)
    src/docs/models.csv                   the 81 planned navigation models (domain maps)
    results/zoo_training/maps_*.json      competence (validation legal rate) and admission (G3: at least 0.50), read
                                          with src/zoo/registry.py
    results/D1..D4/<id>.json              diagnostic scores (status ok only)
    results/fragility_maps/<id>.json      maps-fragility-v1 (G8); only results of the frozen suite (hash below) count
    results/whole_route_maps/<id>.json    the whole route measure (G9)
    the same folders for the three anchors maps_vafa_* (G11, description only)

Which models (G12, Amendment 6 H1)
    A model is "scored" when D1, D2, D3, D4, the fragility suite and G9 all have a result file (status ok, or failed
    or skipped with a reason; for fragility, a result of the frozen suite). An architecture counts when all 27 of
    its models are scored; the analysis runs on the models of those architectures. If fewer than three architectures
    count, architecture is left out of H2 and of every regression (G10). Within them, as for Othello (E4): H1 and H2
    use the admitted models with four scores of status ok; H3 also needs a fragility result.

What is computed
    Section 5 counts       planned, trained, admitted per architecture and scale; every failed or skipped run with
                           its reason; which architectures are complete (G12)
    H1, A3, H2             hypotheses.h1, a3, h2 (Section 4, Amendment 1, E4, F1)
    F3                     the distribution of fragility and relative fragility, before H3
    H3 primary             hypotheses.h3_primary on relative fragility (fragility / in distribution legal rate), the
                           A2 rule with D1 as the reference. Amendment 4 F2 (the Othello test model) does not apply
    H3 secondary           Section 4: Spearman of all four diagnostics with fragility and with relative fragility, the
                           Section 4 best predictor rule, D1 on its own (hypotheses.h3_secondary); the D4 causal flag
                           beyond D4 selectivity (hypotheses.causal_and_occupancy)
    G10 control            preregistered secondary: (a) the A2 regression with training distribution as a categorical
                           covariate, configuration bootstrap interval of each diagnostic's coefficient; (b) the A2
                           partial Spearman (given D1) within each training distribution, configuration bootstrap over
                           its configurations. The H3 decision stays the A2 rule. The same two on Othello, labelled
                           exploratory (results/analysis/othello/training_distribution_control.json)
    H6 sensitivity         H1 to H3 again with the secondary D4 number (probe accuracy at best_layer minus the larger of
                           the start and destination baselines) in place of the primary D4 score; the primary decides
    G9 secondary           Spearman of each diagnostic with the share of free play drives with no illegal top token,
                           configuration bootstrap; reported, in no H decision
    D3 shortcut            descriptive: Spearman of the D3 score with its mean shortcut agreement (G6)
    G11 anchors            every number of the three anchors, description only, never in an H statistic

Run (repo root):
    python src/analysis/hypotheses_maps.py
Writes results/analysis/maps/summary.json, model_table.csv, h1_pairs.csv, h3.csv, anchors.csv, master_table.csv,
reports/figures/maps/fragility_distribution.svg, and results/analysis/othello/training_distribution_control.json.
Tests: src/tests/test_hypotheses_maps.py (fake navigation zoos in the real file formats).
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hy = _load("hypotheses", "src/analysis/hypotheses.py")

DOMAIN = "maps"
DIAGS = hy.DIAGS
ARCHS = ("transformer", "lstm", "mamba")
N_PER_ARCH = 27
ADMISSION = 0.50                                  # G3
FRAGILITY_FOLDER = "fragility_maps"
FRAGILITY_SUITE = "maps-fragility-v1"
FRAGILITY_HASH = "47068aee3f4b"                   # frozen, Amendment 6 H2
G9_FOLDER = "whole_route_maps"
G9_OUTCOME = "share_drives_no_illegal"
ANCHORS = ("maps_vafa_shortest_paths", "maps_vafa_noisy_shortest_paths", "maps_vafa_random_walks")
DIAG_FOLDERS = {"D1": "D1_next_token", "D2": "D2_state_equiv", "D3": "D3_inductive_bias", "D4": "D4_probing"}
OUTCOME = "rel_frag"
# hypotheses.h3_secondary reads these two keys; on navigation they hold maps-fragility-v1 (renamed in the output)
ALIAS = {"frag_v2": "frag", "rel_frag_v2": "rel_frag"}


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ==========================================================================
# Data
# ==========================================================================

def d4_secondary(res: dict) -> float | None:
    """Amendment 6 H6: probe accuracy at best_layer minus the larger of the start and destination baselines."""
    det = res.get("details") or {}
    try:
        acc = float(det["probe_acc_by_layer"][int(det["best_layer"])])
        return acc - max(float(det["start_baseline_acc"]), float(det["destination_baseline_acc"]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def read_model(model_id: str, root: Path) -> dict[str, Any]:
    """Every number of one model (zoo or anchor) from its result files, with each file's status."""
    r: dict[str, Any] = {"model_id": model_id}
    for d, folder in DIAG_FOLDERS.items():
        res = _json(root / "results" / folder / f"{model_id}.json")
        ok = res is not None and res.get("status") == "ok"
        r[d] = float(res["score"]) if ok else None
        r[f"{d}_status"] = res.get("status", "no status") if res else "missing"
        r[f"{d}_reason"] = (res.get("reason") or "") if res and not ok else ""
        if d == "D4" and ok:
            r["D4_secondary"] = d4_secondary(res)
            c = res["details"].get("causal") or {}
            r["causal_label"], r["causal_score"] = c.get("causal_label"), c.get("causal_score")
        if d == "D3" and ok:
            sa = res["details"].get("shortcut_agreement_by_task_pretrained") or {}
            r["d3_shortcut_agreement"] = float(np.mean(list(sa.values()))) if sa else None
    for k in ("D4_secondary", "causal_label", "causal_score", "d3_shortcut_agreement"):
        r.setdefault(k, None)
    r["occupancy_shortcut"] = None                       # Othello only (Amendment 2, C)
    fr = _json(root / "results" / FRAGILITY_FOLDER / f"{model_id}.json")
    if fr is None:
        r["fragility_status"] = "missing"
    elif fr.get("suite_hash") != FRAGILITY_HASH or "fragility" not in fr:
        r["fragility_status"] = f"not the frozen suite (hash {fr.get('suite_hash')})"
    else:
        r["fragility_status"] = "ok"
    frozen = r["fragility_status"] == "ok"
    r["frag"] = float(fr["fragility"]) if frozen and fr["fragility"] is not None else None
    r["in_dist"] = float(fr["in_distribution_legal_rate"]) if frozen else None
    r["rel_frag"] = (r["frag"] / r["in_dist"]) if frozen and r["frag"] is not None and r["in_dist"] else None
    g9 = _json(root / "results" / G9_FOLDER / f"{model_id}.json")
    r["G9_status"] = g9.get("status", "no status") if g9 else "missing"
    r["G9_reason"] = (g9.get("reason") or "") if g9 and g9.get("status") != "ok" else ""
    ok9 = g9 is not None and g9.get("status") == "ok"
    r["g9_no_illegal"] = float(g9["free_play"][G9_OUTCOME]) if ok9 else None
    r["g9_reached"] = float(g9["free_play"]["share_reached_destination"]) if ok9 else None
    r["g9_teacher_forced_all_legal"] = float(g9["teacher_forced"]["share_routes_all_legal"]) if ok9 else None
    return r


def scored(r: dict[str, Any]) -> bool:
    """G12: every diagnostic, the fragility suite and G9 have a result file (ok, or failed or skipped with a
    reason); the fragility file must come from the frozen suite."""
    files = all(r[f"{d}_status"] in ("ok", "failed", "skipped") for d in DIAGS)
    return files and r["fragility_status"] == "ok" and r["G9_status"] in ("ok", "failed", "skipped")


def load_table(root: Path = ROOT) -> list[dict[str, Any]]:
    """One row per planned navigation model: fields, competence and admission (training records), every score."""
    results = _load("results", "src/component/results.py")
    registry = _load("registry", "src/zoo/registry.py")
    models = results.load_models(root / "src/docs/models.csv")
    reg = {g["model_id"]: g for g in registry.build_registry(root / "results" / "zoo_training", ADMISSION,
                                                              root / "src/docs/models.csv")}
    rows = []
    for mid, m in models.items():
        if m["domain"] != DOMAIN:
            continue
        r = read_model(mid, root)
        r.update({"arch": m["arch"], "scale": m["scale"], "distribution": m["distribution"], "seed": int(m["seed"])})
        g = reg.get(mid)
        r["trained"] = g is not None
        r["competence"] = float(g["val_legal_rate"]) if g else None
        r["admitted"] = bool(g) and g["admitted"] == "yes"
        r["scored"] = scored(r)
        rows.append(r)
    return rows


# ==========================================================================
# Section 5 and G12
# ==========================================================================

def completeness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {a: [r for r in rows if r["arch"] == a] for a in ARCHS}
    out = {a: {"planned": len(v), "scored": sum(r["scored"] for r in v),
               "not_scored": [r["model_id"] for r in v if not r["scored"]]} for a, v in by.items()}
    complete = [a for a in ARCHS if out[a]["planned"] == N_PER_ARCH and out[a]["scored"] == N_PER_ARCH]
    return {"by_arch": out, "complete_architectures": complete}


def section5(rows: list[dict[str, Any]], root: Path = ROOT) -> dict[str, Any]:
    registry = _load("registry", "src/zoo/registry.py")
    reg = registry.build_registry(root / "results" / "zoo_training", ADMISSION, root / "src/docs/models.csv")
    table = registry.admission_table(reg, root / "src/docs/models.csv", domain=DOMAIN)
    not_ok = []
    for r in rows:
        for d in DIAGS:
            if r[f"{d}_status"] != "ok":
                not_ok.append({"model_id": r["model_id"], "run": d, "status": r[f"{d}_status"], "reason": r[f"{d}_reason"]})
        if r["fragility_status"] != "ok":
            not_ok.append({"model_id": r["model_id"], "run": "fragility", "status": r["fragility_status"], "reason": ""})
        if r["G9_status"] != "ok":
            not_ok.append({"model_id": r["model_id"], "run": "G9", "status": r["G9_status"], "reason": r["G9_reason"]})
    not_admitted = [{"model_id": r["model_id"], "competence": r["competence"]} for r in rows
                    if r["trained"] and not r["admitted"]]
    return {"admission_by_arch_scale": table, "not_admitted": not_admitted, "runs_not_ok": not_ok}


# ==========================================================================
# H2 with a factor that does not vary (G10)
# ==========================================================================

def h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """hypotheses.h2; a factor with a single level (architecture when fewer than three count) is left out and the
    Holm correction runs over the factors that remain."""
    out = hy.h2(rows)
    if "factors" not in out:
        return out
    keep = [t for t in out["factors"] if len(t["level_means"]) > 1]
    dropped = [t["factor"] for t in out["factors"] if len(t["level_means"]) <= 1]
    if dropped:
        adj = hy.holm([t["p"] if np.isfinite(t["p"]) else 1.0 for t in keep])
        floor = out["noise_floor_disagreement_sd"]
        for t, a in zip(keep, adj):
            t["p_holm"] = a
            t["systematic"] = bool(a < hy.ALPHA and floor is not None and t["effect"] > floor)
        out["factors"] = keep
        out["factors_left_out"] = dropped
        out["verdict"] = "systematic" if any(t["systematic"] for t in keep) else "consistent with noise"
    return out


# ==========================================================================
# G10: training distribution as a control
# ==========================================================================

def distribution_control(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """(a) the A2 regression of the outcome on D1, the diagnostic, architecture, scale and training distribution,
    with the configuration bootstrap interval of the diagnostic's coefficient; (b) the A2 partial Spearman given D1
    within each training distribution, with the configuration bootstrap over that distribution's configurations."""
    n = len(rows)
    if n < 6:
        return {"n_models": n, "note": "fewer than 6 models: not computed"}
    y = hy.column(rows, outcome)
    S = {d: hy.column(rows, d) for d in DIAGS}
    idx = hy.boot_indices(rows)
    F = [np.array([r[k] for r in rows]) for k in ("arch", "scale", "distribution")]
    cov = hy.design(F)
    reg = {}
    for d in ("D2", "D3", "D4"):
        X = np.column_stack([S["D1"], S[d], cov])
        fit = hy.ols(y, X)
        reg[d] = {"coef": float(fit["beta"][2]), "ci": hy.coef_interval(y, X, 2, idx),
                  "adds_to_D1_within_distributions": hy.excludes_zero(hy.coef_interval(y, X, 2, idx))}
    within = {}
    for dist in sorted({r["distribution"] for r in rows}):
        sub = [r for r in rows if r["distribution"] == dist]
        ys, d1 = hy.column(sub, outcome), hy.column(sub, "D1")
        sidx = hy.boot_indices(sub)
        within[dist] = {"n_models": len(sub), "n_configurations": len(hy.clusters(sub))}
        for d in ("D2", "D3", "D4"):
            x = hy.column(sub, d)
            ci = hy.interval(hy.boot(lambda i: hy.partial_spearman(x[i], ys[i], d1[i]), sidx))
            within[dist][d] = {"partial_spearman_given_D1": hy.partial_spearman(x, ys, d1), "ci": ci,
                               "adds_to_D1": hy.excludes_zero(ci)}
    return {"n_models": n, "outcome": outcome, "a_regression_with_distribution": reg,
            "b_within_distribution": within}


# ==========================================================================
# G9 secondary, D3 shortcut, anchors
# ==========================================================================

def g9_secondary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    use = [r for r in rows if r["g9_no_illegal"] is not None]
    if len(use) < 6:
        return {"n_models": len(use), "note": "fewer than 6 models: not computed"}
    y = hy.column(use, "g9_no_illegal")
    idx = hy.boot_indices(use)
    out: dict[str, Any] = {"n_models": len(use), "outcome": f"free play {G9_OUTCOME}",
                           "note": "preregistered secondary (G9): reported, not used in any H decision"}
    for d in DIAGS:
        x = hy.column(use, d)
        out[d] = {"spearman": hy.spearman(x, y), "ci": hy.interval(hy.boot(lambda i: hy.spearman(x[i], y[i]), idx))}
    return out


def d3_shortcut(rows: list[dict[str, Any]]) -> dict[str, Any]:
    use = [r for r in rows if r["d3_shortcut_agreement"] is not None]
    if len(use) < 6:
        return {"n_models": len(use), "note": "fewer than 6 models: not computed"}
    x, s = hy.column(use, "D3"), hy.column(use, "d3_shortcut_agreement")
    idx = hy.boot_indices(use)
    return {"n_models": len(use), "note": "descriptive (G6): D3 score against its mean shortcut agreement over T1, T2, T4",
            "spearman": hy.spearman(x, s), "ci": hy.interval(hy.boot(lambda i: hy.spearman(x[i], s[i]), idx)),
            "mean_shortcut_agreement": float(s.mean())}


ANCHOR_KEYS = ["model_id", *DIAGS, *[f"{d}_status" for d in DIAGS], "D4_secondary", "causal_label",
               "fragility_status", "frag", "in_dist", "rel_frag", "G9_status", "g9_teacher_forced_all_legal",
               "g9_no_illegal", "g9_reached"]


def anchors(root: Path = ROOT) -> list[dict[str, Any]]:
    """G11: every number of the three anchors, description only."""
    return [{k: read_model(a, root).get(k) for k in ANCHOR_KEYS} for a in ANCHORS]


# ==========================================================================
# The whole analysis
# ==========================================================================

def _aliased(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    for r in out:
        for k, v in ALIAS.items():
            r[k] = r[v]
        r["frag_v1"] = r["rel_frag_v1"] = None           # one suite on navigation
    return out


def _rename(o: Any) -> Any:
    """hypotheses.h3_secondary keys its outcomes frag_v2 and rel_frag_v2; on navigation they are maps-fragility-v1."""
    if isinstance(o, dict):
        return {{"frag_v2": "fragility", "rel_frag_v2": "relative_fragility"}.get(k, k): _rename(v) for k, v in o.items()}
    return o


def fragility_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment 4 F3 on maps-fragility-v1: summarised before any H3 statistic."""
    out = hy.fragility_distribution(_aliased(rows))
    return {"fragility" if k == "frag_v2" else "relative_fragility": v for k, v in out.items() if k in ALIAS}


def run_h(rows_complete: list[dict[str, Any]], rows_admitted: list[dict[str, Any]], with_frag: list[dict[str, Any]],
          ) -> dict[str, Any]:
    """H1, A3, H2, F3 and H3 (primary and secondary) on one set of rows."""
    out: dict[str, Any] = {}
    out["H1"] = hy.h1(rows_complete)
    out["A3"] = hy.a3(rows_admitted, rows_complete)
    out["H2"] = h2(rows_complete)
    out["fragility_distribution"] = fragility_distribution(with_frag)
    out["H3"] = hy.h3_primary(with_frag, OUTCOME)
    out["H3_secondary"] = _rename(hy.h3_secondary(_aliased(with_frag)))
    out["H3_causal_flag"] = hy.causal_and_occupancy(with_frag, OUTCOME)
    return out


def analyse(root: Path = ROOT) -> dict[str, Any]:
    rows = load_table(root)
    comp = completeness(rows)
    arch_ok = set(comp["complete_architectures"])
    in_scope = [r for r in rows if r["arch"] in arch_ok]
    admitted = [r for r in in_scope if r["trained"] and r["admitted"]]
    complete = [r for r in admitted if all(r[d] is not None for d in DIAGS)]
    with_frag = [r for r in complete if r[OUTCOME] is not None]
    summary: dict[str, Any] = {
        "domain": DOMAIN,
        "protocol": "reports/preregistration.md: Section 4, Amendments 1, 3 (E4), 4 (F1, F3), 5 (G3, G8 to G12), "
                    "6 (H1, H6), 7 (H4 not tested)",
        "G12_completeness": comp,
        "architectures_analysed": [a for a in ARCHS if a in arch_ok],
        "architecture_left_out_of_H2_and_regressions": len(arch_ok) < 3,
        "counts": {"planned": len(rows), "trained": sum(r["trained"] for r in rows),
                   "admitted": sum(r["trained"] and r["admitted"] for r in rows),
                   "in_complete_architectures": len(in_scope), "admitted_in_scope": len(admitted),
                   "all_four_scores": len(complete), "with_fragility": len(with_frag)},
        "missing_scores": {d: [r["model_id"] for r in admitted if r[d] is None] for d in DIAGS},
        "bootstrap": {"resamples": hy.N_BOOT, "seed": hy.BOOT_SEED, "level": 0.95,
                      "unit": "configuration (arch, scale, distribution), seeds kept together (Amendment 4)",
                      "n_configurations": len(hy.clusters(complete)) if complete else 0},
        "section5": section5(rows, root),
    }
    if not arch_ok:
        summary["note"] = ("no architecture has all 27 models scored by the cutoff (G12): navigation is reported "
                           "without H1 to H3")
        summary["_rows"] = rows
        summary["anchors"] = anchors(root)
        return summary
    summary.update(run_h(complete, admitted, with_frag))
    summary["G10_training_distribution_control"] = distribution_control(with_frag, OUTCOME)
    sec = [dict(r, D4=r["D4_secondary"]) for r in complete if r["D4_secondary"] is not None]
    sec_adm = [dict(r, D4=r["D4_secondary"]) for r in admitted]
    summary["H6_sensitivity_secondary_D4"] = {
        "note": "Amendment 6 H6: H1 to H3 with the secondary D4 number in place of the primary; the primary decides",
        **{k: v for k, v in run_h(sec, sec_adm, [r for r in sec if r[OUTCOME] is not None]).items() if k != "A3"}}
    summary["G9_secondary"] = g9_secondary(complete)
    summary["D3_shortcut"] = d3_shortcut(complete)
    summary["anchors"] = anchors(root)
    summary["_rows"] = rows
    return summary


def othello_distribution_control(root: Path = ROOT) -> dict[str, Any]:
    """G10: the same training distribution control on Othello, exploratory (its results were seen first)."""
    rows = hy.load_table("othello", root)
    use = [r for r in rows if r["trained"] and r["admitted"] and all(r[d] is not None for d in DIAGS)
           and r["rel_frag_v2"] is not None]
    return {"label": "EXPLORATORY: Amendment 5 G10 asks for this on Othello too, labelled exploratory, because the "
                     "Othello results were seen first", "domain": "othello", "outcome": "relative fragility v2",
            **distribution_control(use, "rel_frag_v2")}


# ==========================================================================
# Output
# ==========================================================================

TABLE_KEYS = ["model_id", "arch", "scale", "distribution", "seed", "trained", "competence", "admitted", "scored",
              *DIAGS, *[f"{d}_status" for d in DIAGS], "D4_secondary", "causal_label", "causal_score",
              "d3_shortcut_agreement", "disagreement", "fragility_status", "frag", "in_dist", "rel_frag",
              "G9_status", "g9_teacher_forced_all_legal", "g9_no_illegal", "g9_reached"]


def _csv(path: Path, keys: list[str], rows: list[dict[str, Any]]) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    return path


def write_outputs(summary: dict[str, Any], out_dir: Path, fig_dir: Path | None, othello: dict | None,
                  othello_path: Path | None, root: Path = ROOT) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary.pop("_rows")
    written = []
    p = out_dir / "summary.json"
    p.write_text(json.dumps(hy._clean(summary), indent=2) + "\n")
    written.append(p)
    written.append(_csv(out_dir / "model_table.csv", TABLE_KEYS, rows))
    written.append(_csv(out_dir / "anchors.csv", ANCHOR_KEYS, summary["anchors"]))
    if isinstance(summary.get("H1", {}).get("pairs"), list) and summary["H1"]["pairs"]:
        p = out_dir / "h1_pairs.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "tau", "tau_low", "tau_high", "category", "spearman", "partial_tau",
                        "partial_low", "partial_high", "partial_category", "agreement_through_competence"])
            for q in summary["H1"]["pairs"]:
                lo, hi = q["tau_ci"] or [None, None]
                plo, phi = q["partial_tau_ci"] or [None, None]
                w.writerow([q["pair"], q["tau"], lo, hi, q["category"], q["spearman"], q["partial_tau"],
                            plo, phi, q["partial_category"], q["agreement_through_competence"]])
        written.append(p)
    if "D2" in summary.get("H3", {}):
        p = out_dir / "h3.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["diagnostic", "partial_spearman_given_D1", "low", "high", "adds_to_D1",
                        "regression_coef", "regression_low", "regression_high"])
            ref = summary["H3"]["D1_reference"]
            lo, hi = ref["ci"] or [None, None]
            w.writerow(["D1 (reference, Spearman)", ref["spearman"], lo, hi, "", "", "", ""])
            for d in ("D2", "D3", "D4"):
                q = summary["H3"][d]
                lo, hi = q["ci"] or [None, None]
                rlo, rhi = q["regression_ci"] or [None, None]
                w.writerow([d, q["partial_spearman_given_D1"], lo, hi, q["adds_to_D1"], q["regression_coef"], rlo, rhi])
        written.append(p)
    results = _load("results", "src/component/results.py")
    written.append(results.write_master_table(out_dir / "master_table.csv", root / "results",
                                              root / "src/docs/models.csv", domain=DOMAIN))
    if othello is not None and othello_path is not None:
        othello_path.parent.mkdir(parents=True, exist_ok=True)
        othello_path.write_text(json.dumps(hy._clean(othello), indent=2) + "\n")
        written.append(othello_path)
    if fig_dir is not None:
        written += _figures(rows, set(summary["architectures_analysed"]), fig_dir)
    summary["_rows"] = rows
    return written


def _figures(rows: list[dict[str, Any]], archs: set[str], fig_dir: Path) -> list[Path]:
    """F3: the relative fragility distribution of the admitted models with a fragility result, by architecture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use = [r for r in rows if r["arch"] in archs and r["admitted"] and r["rel_frag"] is not None]
    if not use:
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "wmconcord",
                         "figure.facecolor": hy.SURFACE, "savefig.facecolor": hy.SURFACE})
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    hy._style(ax)
    v = np.array([r["rel_frag"] for r in use])
    bins = np.linspace(min(0.0, v.min()), v.max() + 1e-9, 21)
    bottom = np.zeros(len(bins) - 1)
    for arch, col in hy.ARCH_COLOR.items():
        h, _ = np.histogram([r["rel_frag"] for r in use if r["arch"] == arch], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins) * 0.95, bottom=bottom, align="edge", color=col, label=arch,
               edgecolor=hy.SURFACE, linewidth=1)
        bottom += h
    ax.set_xlabel("relative fragility (maps-fragility-v1)")
    ax.set_ylabel("models")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Navigation: relative fragility of {len(use)} admitted models", fontsize=10)
    p = fig_dir / "fragility_distribution.svg"
    fig.savefig(p, metadata={"Date": None})
    plt.close(fig)
    return [p]


def _f(x: Any, fmt: str = "+.3f") -> str:
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def _print(summary: dict[str, Any]) -> None:
    comp = summary["G12_completeness"]
    print(f"navigation: architectures complete (G12): {comp['complete_architectures']}")
    for a, v in comp["by_arch"].items():
        print(f"  {a:12s} {v['scored']}/{v['planned']} scored" + (f"  not scored: {v['not_scored']}" if v["not_scored"] else ""))
    c = summary["counts"]
    print(f"  {c['planned']} planned, {c['trained']} trained, {c['admitted']} admitted, {c['all_four_scores']} with "
          f"all four scores, {c['with_fragility']} with fragility")
    print(f"  runs not ok: {len(summary['section5']['runs_not_ok'])}")
    if "H1" not in summary:
        print(summary.get("note"))
        return
    s = copy.deepcopy(summary)
    s["counts"]["with_fragility_v2"] = s["counts"]["with_fragility"]
    hy._print(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Navigation H1, H2, H3 (Amendments 5 to 7)")
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--no_othello_control", action="store_true", help="skip the exploratory Othello G10 control")
    a = p.parse_args()
    summ = analyse(ROOT)
    oth = None if a.no_othello_control else othello_distribution_control(ROOT)
    paths = write_outputs(summ, ROOT / "results" / "analysis" / DOMAIN,
                          None if a.no_figures else ROOT / "reports" / "figures" / DOMAIN, oth,
                          ROOT / "results" / "analysis" / "othello" / "training_distribution_control.json")
    _print(summ)
    for q in paths:
        print("wrote", q.relative_to(ROOT))


if __name__ == "__main__":
    main()
