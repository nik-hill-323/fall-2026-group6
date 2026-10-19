"""Analysis of H1, H2 and H3 for one domain, exactly as reports/preregistration.md fixes them.

Inputs (all from the repo, nothing is fitted here except the preregistered statistics)
    src/docs/models.csv                 the planned zoo (domain, arch, scale, distribution, seed)
    src/docs/zoo_registry.csv           competence (val_legal_rate) and admission per trained model
    results/D1..D4/<model_id>.json      diagnostic scores (status ok only)
    results/fragility/<model_id>.json   fragility v1 (fragility-v1, secondary outcome)
    results/fragility_v2/<model_id>.json  fragility v2 (fragility-v2, the H3 outcome, Amendment 3)
The two reference Othello GPTs are not in models.csv, so they never enter these statistics
(Section 1).

What is computed (section numbers refer to the preregistration)
    H1 (Section 4, A1, A5)  for the 6 pairs of diagnostics: Kendall tau b and Spearman rho with 95%
        bootstrap intervals, the category (agree: tau >= 0.50 and interval excludes 0; disagree:
        interval includes 0 or tau < 0.30; otherwise partial); the partial Kendall tau controlling
        for competence, its interval and category, "agreement through competence" (raw agree,
        partial disagree), and the A5 rule (if tau(D1, competence) >= 0.90 the partial tau of the
        3 pairs with D1 is marked not interpretable). Kendall W on the raw scores and on the
        residual ranks (each diagnostic's ranks with the competence ranks regressed out by least
        squares), each with an interval.
    A3   competence of admitted models: count, mean and range per architecture and scale, a
        histogram; the H1 sensitivity run on models with competence >= 0.90, only if at least 10
        qualify.
    H2 (Section 4)  per model disagreement = standard deviation (population, ddof 0) of its four
        rank percentiles (average rank / n). Noise floor: for each configuration (arch, scale,
        distribution) the standard deviation (ddof 1) across its seeds of each diagnostic's score,
        and of the disagreement itself; the mean over configurations is reported. Least squares
        regression of disagreement on architecture, scale and distribution (categorical), one F
        test per factor (the model without that factor against the full model), Holm correction
        over the 3 factors. Effect of a factor = the range of its level means of disagreement.
        Systematic if at least one factor has Holm p < 0.05 and an effect larger than the noise
        floor of the disagreement (both in the same units, rank percentile standard deviation).
    H3 (A2, Amendment 3)  outcome = relative fragility v2 = fragility v2 / in distribution legal
        rate. D1 is the reference: Spearman(D1, outcome) with interval. For D2, D3, D4: the partial
        Spearman correlation with the outcome controlling for D1 (Pearson correlation of the rank
        residuals after regressing out the D1 ranks), with interval; least squares regression of
        the outcome on D1, the diagnostic, architecture and scale (categorical), with the
        diagnostic's coefficient and 95% t interval. Best predictor: the most negative partial
        correlation, and the bootstrap interval of each difference to the other two lies below 0.
        Adds to D1: the partial correlation's interval excludes 0.
        Secondary (Section 4, labelled): Spearman of all four diagnostics with fragility v2 and with
        relative fragility v2, the Section 4 best predictor rule over all four, a regression of the
        outcome on each score with architecture and scale; D1 tested on its own; the whole primary
        analysis repeated on relative fragility v1. D4 causal label: counts, and whether the flag
        "decodable and causal" adds to D4 selectivity (regression of the outcome on D4 and the
        flag; partial Spearman of the flag controlling for D4). D3: Spearman of the D3 score with
        its occupancy shortcut rate (Amendment 2, C).
    Every interval: 95% percentile bootstrap over models, 2,000 resamples, seed 0 (Section 4). All
    statistics of one analysis use the same resamples, so differences have proper intervals.

Run (repo root):
    python src/analysis/hypotheses.py --domain othello
Writes results/analysis/<domain>/summary.json, model_table.csv, h1_pairs.csv, h3.csv and figures in
reports/figures/<domain>/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DIAGS = ["D1", "D2", "D3", "D4"]
N_BOOT = 2000
BOOT_SEED = 0
AGREE_TAU, DISAGREE_TAU = 0.50, 0.30
A5_LIMIT = 0.90
A3_COMPETENCE, A3_MIN_MODELS = 0.90, 10
ALPHA = 0.05
CAUSAL = "decodable and causal"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# Statistics
# ==========================================================================

def ranks(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.kendalltau(x, y).statistic)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def partial_tau(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    """A1: tau(x, y | c) = (txy - txc tyc) / sqrt((1 - txc^2)(1 - tyc^2))."""
    txy, txc, tyc = kendall_tau(x, y), kendall_tau(x, c), kendall_tau(y, c)
    den = (1 - txc ** 2) * (1 - tyc ** 2)
    if not np.isfinite([txy, txc, tyc]).all() or den <= 0:
        return float("nan")
    return float((txy - txc * tyc) / math.sqrt(den))


def residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of a least squares fit of y on X plus an intercept."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of the ranks of x and y after regressing out the ranks of z."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    ex, ey = residual(rx, rz), residual(ry, rz)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def kendall_w(M: np.ndarray) -> float:
    """Kendall's W for n objects (rows) ranked by m raters (columns), with the tie correction."""
    n, m = M.shape
    if n < 2:
        return float("nan")
    R = np.column_stack([ranks(M[:, j]) for j in range(m)])
    S = float(((R.sum(1) - R.sum(1).mean()) ** 2).sum())
    T = 0.0
    for j in range(m):
        _, counts = np.unique(R[:, j], return_counts=True)
        T += float((counts ** 3 - counts).sum())
    den = m ** 2 * (n ** 3 - n) - m * T
    return float(12 * S / den) if den > 0 else float("nan")


def residual_rank_w(M: np.ndarray, c: np.ndarray) -> float:
    """A1: rank each diagnostic, regress the competence ranks out of it, W on the residuals."""
    rc = ranks(c)
    E = np.column_stack([residual(ranks(M[:, j]), rc) for j in range(M.shape[1])])
    return kendall_w(E)


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def interval(values: np.ndarray) -> list[float] | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 0.5 * N_BOOT:   # most resamples degenerate: no interval
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot(fn: Callable[[np.ndarray], float], idx: np.ndarray) -> np.ndarray:
    return np.array([fn(i) for i in idx], dtype=float)


def excludes_zero(ci: list[float] | None) -> bool:
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def category(tau: float, ci: list[float] | None) -> str:
    """Section 4: agree if tau >= 0.50 and the interval excludes 0; disagree if the interval
    includes 0 or tau < 0.30; anything else partial."""
    if not np.isfinite(tau) or ci is None:
        return "not computable"
    if not excludes_zero(ci) or tau < DISAGREE_TAU:
        return "disagree"
    if tau >= AGREE_TAU:
        return "agree"
    return "partial"


def design(factors: list[np.ndarray]) -> np.ndarray:
    """Treatment coded dummies (first level sorted is the reference) for categorical factors."""
    cols = []
    for f in factors:
        levels = sorted(set(f.tolist()))
        for lev in levels[1:]:
            cols.append((f == lev).astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(factors[0]), 0))


def ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Least squares with an intercept: coefficients, standard errors, residual sum of squares."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(((y - A @ beta) ** 2).sum())
    n, p = A.shape
    rank = np.linalg.matrix_rank(A)
    dof = n - rank
    se = np.full(p, np.nan)
    if dof > 0:
        cov = rss / dof * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"beta": beta, "se": se, "rss": rss, "dof": dof, "rank": rank}


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj.tolist()


# ==========================================================================
# Data
# ==========================================================================

def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_table(domain: str, root: Path = ROOT) -> list[dict[str, Any]]:
    """One row per planned model of the domain: model fields, competence and admission (registry),
    the four scores (None unless status ok), fragility v1 and v2, the D4 causal label and score,
    and the D3 occupancy shortcut rate."""
    results = _load("results", "src/component/results.py")
    models = results.load_models(root / "src/docs/models.csv")
    reg = {}
    reg_path = root / "src/docs/zoo_registry.csv"
    if reg_path.exists():
        with open(reg_path, newline="") as f:
            reg = {r["model_id"]: r for r in csv.DictReader(f)}
    rows = []
    for mid, m in models.items():
        if m["domain"] != domain:
            continue
        r: dict[str, Any] = {"model_id": mid, "arch": m["arch"], "scale": m["scale"],
                             "distribution": m["distribution"], "seed": int(m["seed"])}
        g = reg.get(mid)
        r["trained"] = g is not None
        r["competence"] = float(g["val_legal_rate"]) if g else None
        r["admitted"] = bool(g) and g["admitted"] == "yes"
        for d in DIAGS:
            res = _json(root / "results" / results.diag_folder(d) / f"{mid}.json")
            ok = res is not None and res.get("status") == "ok"
            r[d] = float(res["score"]) if ok else None
            r[f"{d}_status"] = res.get("status") if res else "missing"
            if d == "D4" and ok:
                c = res["details"].get("causal", {})
                r["causal_label"] = c.get("causal_label")
                r["causal_score"] = c.get("causal_score")
            if d == "D3" and ok:
                sr = res["details"].get("symbolic_regression", {})
                r["occupancy_shortcut"] = sr.get("occupancy_shortcut_rate_pretrained")
        for v, folder in (("v1", "fragility"), ("v2", "fragility_v2")):
            fr = _json(root / "results" / folder / f"{mid}.json")
            r[f"frag_{v}"] = float(fr["fragility"]) if fr else None
            r[f"in_dist_{v}"] = float(fr["in_distribution_legal_rate"]) if fr else None
            r[f"rel_frag_{v}"] = (float(fr["fragility"]) / float(fr["in_distribution_legal_rate"])
                                  if fr and fr["in_distribution_legal_rate"] else None)
        r.setdefault("causal_label", None)
        r.setdefault("causal_score", None)
        r.setdefault("occupancy_shortcut", None)
        rows.append(r)
    return rows


def column(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


# ==========================================================================
# H1 and A1, A5
# ==========================================================================

def h1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n_models": n, "pairs": [], "kendall_w": None}
    if n < 4:
        out["note"] = "fewer than 4 models: not computed"
        return out
    S = {d: column(rows, d) for d in DIAGS}
    c = column(rows, "competence")
    idx = boot_indices(n)
    tau_c = {d: kendall_tau(S[d], c) for d in DIAGS}
    a5 = tau_c["D1"] >= A5_LIMIT if np.isfinite(tau_c["D1"]) else False
    out["tau_with_competence"] = tau_c
    out["a5_d1_not_interpretable"] = bool(a5)
    for a, b in itertools.combinations(DIAGS, 2):
        x, y = S[a], S[b]
        t = kendall_tau(x, y)
        t_ci = interval(boot(lambda i: kendall_tau(x[i], y[i]), idx))
        rho = spearman(x, y)
        rho_ci = interval(boot(lambda i: spearman(x[i], y[i]), idx))
        pt = partial_tau(x, y, c)
        pt_ci = interval(boot(lambda i: partial_tau(x[i], y[i], c[i]), idx))
        raw_cat, part_cat = category(t, t_ci), category(pt, pt_ci)
        not_interp = a5 and "D1" in (a, b)
        out["pairs"].append({
            "pair": f"{a}-{b}", "tau": t, "tau_ci": t_ci, "category": raw_cat,
            "spearman": rho, "spearman_ci": rho_ci,
            "partial_tau": pt, "partial_tau_ci": pt_ci,
            "partial_category": "not interpretable (A5)" if not_interp else part_cat,
            "agreement_through_competence": (raw_cat == "agree" and part_cat == "disagree" and not not_interp),
        })
    M = np.column_stack([S[d] for d in DIAGS])
    out["kendall_w"] = kendall_w(M)
    out["kendall_w_ci"] = interval(boot(lambda i: kendall_w(M[i]), idx))
    out["kendall_w_residual"] = residual_rank_w(M, c)
    out["kendall_w_residual_ci"] = interval(boot(lambda i: residual_rank_w(M[i], c[i]), idx))
    cats = [p["category"] for p in out["pairs"]]
    out["all_pairs_agree"] = all(x == "agree" for x in cats)
    return out


def a3(rows_admitted: list[dict[str, Any]], rows_complete: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[float]] = {}
    for r in rows_admitted:
        by.setdefault(f"{r['arch']}/{r['scale']}", []).append(r["competence"])
    table = {k: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
             for k, v in sorted(by.items())}
    hi = [r for r in rows_complete if r["competence"] >= A3_COMPETENCE]
    out: dict[str, Any] = {"competence_by_arch_scale": table, "n_at_or_above_0.90": len(hi)}
    if len(hi) >= A3_MIN_MODELS:
        out["sensitivity_h1"] = h1(hi)
    else:
        out["sensitivity_h1"] = f"not run: {len(hi)} models at or above {A3_COMPETENCE}, fewer than {A3_MIN_MODELS}"
    return out


# ==========================================================================
# H2
# ==========================================================================

def disagreement(rows: list[dict[str, Any]]) -> np.ndarray:
    n = len(rows)
    P = np.column_stack([ranks(column(rows, d)) / n for d in DIAGS])
    return P.std(axis=1, ddof=0)


def h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n < 6:
        return {"n_models": n, "note": "fewer than 6 models: not computed"}
    dis = disagreement(rows)
    for r, v in zip(rows, dis):
        r["disagreement"] = float(v)
    configs: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        configs.setdefault((r["arch"], r["scale"], r["distribution"]), []).append(i)
    multi = [ix for ix in configs.values() if len(ix) >= 2]
    floor_diag = {d: (float(np.mean([np.std(column(rows, d)[ix], ddof=1) for ix in multi])) if multi else None)
                  for d in DIAGS}
    floor_dis = float(np.mean([np.std(dis[ix], ddof=1) for ix in multi])) if multi else None

    names = ["arch", "scale", "distribution"]
    F = {k: np.array([r[k] for r in rows]) for k in names}
    full = ols(dis, design([F[k] for k in names]))
    tests = []
    for k in names:
        reduced = ols(dis, design([F[j] for j in names if j != k]))
        q = full["rank"] - reduced["rank"]
        if q <= 0 or full["dof"] <= 0 or full["rss"] <= 0:
            p = float("nan"); Fstat = float("nan")
        else:
            Fstat = ((reduced["rss"] - full["rss"]) / q) / (full["rss"] / full["dof"])
            p = float(stats.f.sf(Fstat, q, full["dof"]))
        levels = sorted(set(F[k].tolist()))
        means = {lev: float(dis[F[k] == lev].mean()) for lev in levels}
        tests.append({"factor": k, "F": float(Fstat), "df": [int(q), int(full["dof"])], "p": p,
                      "level_means": means, "effect": float(max(means.values()) - min(means.values()))})
    ps = [t["p"] for t in tests]
    adj = holm([p if np.isfinite(p) else 1.0 for p in ps])
    for t, a in zip(tests, adj):
        t["p_holm"] = a
        t["systematic"] = bool(a < ALPHA and floor_dis is not None and t["effect"] > floor_dis)
    return {"n_models": n, "mean_disagreement": float(dis.mean()),
            "noise_floor_score_sd_by_diagnostic": floor_diag, "noise_floor_disagreement_sd": floor_dis,
            "n_configs_with_seeds": len(multi), "factors": tests,
            "verdict": "systematic" if any(t["systematic"] for t in tests) else "consistent with noise"}


# ==========================================================================
# H3
# ==========================================================================

def h3_primary(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """A2 on one outcome column: D1 the reference, the contest among D2, D3 and D4."""
    n = len(rows)
    if n < 6:
        return {"n_models": n, "outcome": outcome, "note": "fewer than 6 models: not computed"}
    y = column(rows, outcome)
    S = {d: column(rows, d) for d in DIAGS}
    idx = boot_indices(n)
    out: dict[str, Any] = {"n_models": n, "outcome": outcome}
    d1 = S["D1"]
    out["D1_reference"] = {"spearman": spearman(d1, y),
                           "ci": interval(boot(lambda i: spearman(d1[i], y[i]), idx))}
    archs = np.array([r["arch"] for r in rows])
    scales = np.array([r["scale"] for r in rows])
    cov = design([archs, scales])
    part, part_boot = {}, {}
    for d in ("D2", "D3", "D4"):
        x = S[d]
        part[d] = partial_spearman(x, y, d1)
        part_boot[d] = boot(lambda i: partial_spearman(x[i], y[i], d1[i]), idx)
        fit = ols(y, np.column_stack([d1, x, cov]))
        b, se, dof = float(fit["beta"][2]), float(fit["se"][2]), fit["dof"]
        tq = float(stats.t.ppf(0.975, dof)) if dof > 0 else float("nan")
        ci = interval(part_boot[d])
        out[d] = {"partial_spearman_given_D1": part[d], "ci": ci, "adds_to_D1": excludes_zero(ci),
                  "regression_coef": b, "regression_se": se, "regression_ci": [b - tq * se, b + tq * se],
                  "regression_dof": int(dof)}
    best = min(part, key=lambda d: part[d] if np.isfinite(part[d]) else np.inf)
    diffs = {}
    wins = np.isfinite(part[best]) and part[best] < 0
    for d in part:
        if d == best:
            continue
        ci = interval(part_boot[best] - part_boot[d])
        diffs[f"{best}-{d}"] = ci
        wins = wins and ci is not None and ci[1] < 0
    out["strongest"] = best
    out["strongest_minus_others_ci"] = diffs
    out["best_predictor"] = best if wins else "no single diagnostic wins"
    out["any_adds_to_D1"] = any(out[d]["adds_to_D1"] for d in ("D2", "D3", "D4"))
    return out


def h3_secondary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n < 6:
        return {"n_models": n, "note": "fewer than 6 models: not computed"}
    idx = boot_indices(n)
    out: dict[str, Any] = {"n_models": n}
    archs = np.array([r["arch"] for r in rows])
    scales = np.array([r["scale"] for r in rows])
    cov = design([archs, scales])
    for outcome in ("frag_v2", "rel_frag_v2"):
        y = column(rows, outcome)
        res, bootv = {}, {}
        for d in DIAGS:
            x = column(rows, d)
            bootv[d] = boot(lambda i: spearman(x[i], y[i]), idx)
            fit = ols(y, np.column_stack([x, cov]))
            res[d] = {"spearman": spearman(x, y), "ci": interval(bootv[d]),
                      "regression_coef": float(fit["beta"][1]), "regression_se": float(fit["se"][1])}
        best = min(res, key=lambda d: res[d]["spearman"] if np.isfinite(res[d]["spearman"]) else np.inf)
        wins = np.isfinite(res[best]["spearman"])
        diffs = {}
        for d in res:
            if d != best:
                ci = interval(bootv[best] - bootv[d])
                diffs[f"{best}-{d}"] = ci
                wins = wins and ci is not None and ci[1] < 0
        out[outcome] = {"by_diagnostic": res, "strongest": best, "strongest_minus_others_ci": diffs,
                        "best_predictor_section4": best if wins else "no single diagnostic wins",
                        "D1_differs_from_0": excludes_zero(res["D1"]["ci"])}
    return out


def causal_and_occupancy(rows: list[dict[str, Any]], outcome: str = "rel_frag_v2") -> dict[str, Any]:
    labels: dict[str, int] = {}
    for r in rows:
        labels[str(r["causal_label"])] = labels.get(str(r["causal_label"]), 0) + 1
    out: dict[str, Any] = {"causal_labels": labels}
    use = [r for r in rows if r["causal_label"] is not None]
    if len(use) >= 6:
        y = column(use, outcome)
        d4 = column(use, "D4")
        flag = np.array([1.0 if r["causal_label"] == CAUSAL else 0.0 for r in use])
        if 0 < flag.sum() < len(flag):
            fit = ols(y, np.column_stack([d4, flag]))
            dof = fit["dof"]
            tq = float(stats.t.ppf(0.975, dof)) if dof > 0 else float("nan")
            b, se = float(fit["beta"][2]), float(fit["se"][2])
            idx = boot_indices(len(use))
            ci = interval(boot(lambda i: partial_spearman(flag[i], y[i], d4[i]), idx))
            out["flag_beyond_D4"] = {"n_models": len(use), "n_causal": int(flag.sum()),
                                     "regression_coef": b, "regression_ci": [b - tq * se, b + tq * se],
                                     "partial_spearman_given_D4": partial_spearman(flag, y, d4),
                                     "partial_ci": ci, "adds_to_D4": excludes_zero(ci)}
        else:
            out["flag_beyond_D4"] = "not computed: every model has the same label"
    occ = [r for r in rows if r["occupancy_shortcut"] is not None]
    if len(occ) >= 6:
        x, s = column(occ, "D3"), column(occ, "occupancy_shortcut")
        idx = boot_indices(len(occ))
        out["D3_vs_occupancy_shortcut"] = {"n_models": len(occ), "spearman": spearman(x, s),
                                           "ci": interval(boot(lambda i: spearman(x[i], s[i]), idx)),
                                           "mean_shortcut_rate": float(s.mean())}
    return out


# ==========================================================================
# The whole analysis
# ==========================================================================

def analyse(domain: str, root: Path = ROOT) -> dict[str, Any]:
    rows = load_table(domain, root)
    trained = [r for r in rows if r["trained"]]
    admitted = [r for r in trained if r["admitted"]]
    complete = [r for r in admitted if all(r[d] is not None for d in DIAGS)]
    with_v2 = [r for r in complete if r["rel_frag_v2"] is not None]
    with_v1 = [r for r in complete if r["rel_frag_v1"] is not None]
    summary: dict[str, Any] = {
        "domain": domain,
        "counts": {"planned": len(rows), "trained": len(trained), "admitted": len(admitted),
                   "not_admitted": len(trained) - len(admitted), "all_four_scores": len(complete),
                   "with_fragility_v2": len(with_v2), "with_fragility_v1": len(with_v1)},
        "missing_scores": {d: [r["model_id"] for r in admitted if r[d] is None] for d in DIAGS},
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "level": 0.95},
    }
    summary["H1"] = h1(complete)
    summary["A3"] = a3(admitted, complete)
    summary["H2"] = h2(complete)
    summary["H3"] = h3_primary(with_v2, "rel_frag_v2")
    summary["H3_secondary"] = h3_secondary(with_v2)
    summary["H3_secondary_v1"] = h3_primary(with_v1, "rel_frag_v1")
    summary["H3_causal_and_occupancy"] = causal_and_occupancy(with_v2)
    summary["_rows"] = rows
    return summary


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 6)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_outputs(summary: dict[str, Any], out_dir: Path, fig_dir: Path | None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summary.pop("_rows")
    written = []
    p = out_dir / "summary.json"
    p.write_text(json.dumps(_clean(summary), indent=2) + "\n")
    written.append(p)
    keys = ["model_id", "arch", "scale", "distribution", "seed", "trained", "competence", "admitted",
            *DIAGS, "causal_label", "causal_score", "occupancy_shortcut", "disagreement",
            "frag_v1", "in_dist_v1", "rel_frag_v1", "frag_v2", "in_dist_v2", "rel_frag_v2"]
    p = out_dir / "model_table.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    written.append(p)
    if isinstance(summary["H1"].get("pairs"), list) and summary["H1"]["pairs"]:
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
    if "D2" in summary["H3"]:
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
                w.writerow([d, q["partial_spearman_given_D1"], lo, hi, q["adds_to_D1"],
                            q["regression_coef"], *q["regression_ci"]])
        written.append(p)
    if fig_dir is not None:
        written += _figures(summary, rows, fig_dir)
    summary["_rows"] = rows
    return written


# ==========================================================================
# Figures (static, for the report and slides)
# ==========================================================================

ARCH_COLOR = {"transformer": "#2a78d6", "lstm": "#eb6834", "mamba": "#1baf7a"}  # categorical slots 1 to 3
ARCH_MARK = {"transformer": "o", "lstm": "s", "mamba": "^"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figures(summary: dict[str, Any], rows: list[dict[str, Any]], fig_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

    pairs = summary["H1"].get("pairs") or []
    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, key, title in ((axes[0], "tau", "Kendall tau"),
                               (axes[1], "partial_tau", "Partial tau, competence controlled")):
            M = np.eye(4)
            for q in pairs:
                a, b = q["pair"].split("-")
                i, j = DIAGS.index(a), DIAGS.index(b)
                v = q[key] if q[key] is not None and np.isfinite(q[key]) else np.nan
                M[i, j] = M[j, i] = v
            im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
            for i in range(4):
                for j in range(4):
                    if i != j and np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
            ax.set_xticks(range(4), DIAGS)
            ax.set_yticks(range(4), DIAGS)
            ax.set_title(title, fontsize=10, color=INK)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.colorbar(im, ax=axes, shrink=0.8, label="rank correlation")
        fig.suptitle(f"H1: agreement between diagnostics ({summary['H1']['n_models']} admitted models)",
                     fontsize=11, color=INK)
        p = fig_dir / "h1_concordance.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    adm = [r for r in rows if r["admitted"]]
    if adm:
        fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
        _style(ax)
        bins = np.linspace(0.5, 1.0, 21)
        bottom = np.zeros(len(bins) - 1)
        for arch, col in ARCH_COLOR.items():
            v = [r["competence"] for r in adm if r["arch"] == arch]
            h, _ = np.histogram(v, bins=bins)
            ax.bar(bins[:-1], h, width=np.diff(bins) - 0.002, bottom=bottom, align="edge",
                   color=col, label=arch, edgecolor=SURFACE, linewidth=1)
            bottom += h
        ax.axvline(A3_COMPETENCE, color=INK2, linewidth=1, linestyle="--")
        ax.text(A3_COMPETENCE + 0.004, bottom.max() * 0.95, "0.90 sensitivity", fontsize=8, color=INK2)
        ax.set_xlabel("competence (validation legal move rate)")
        ax.set_ylabel("models")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"A3: competence of the {len(adm)} admitted models", fontsize=10)
        p = fig_dir / "a3_competence.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    use = [r for r in rows if r.get("rel_frag_v2") is not None and r["admitted"]
           and all(r[d] is not None for d in DIAGS)]
    if use:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, constrained_layout=True)
        for ax, d in zip(axes, DIAGS):
            _style(ax)
            for arch, col in ARCH_COLOR.items():
                xs = [r[d] for r in use if r["arch"] == arch]
                ys = [r["rel_frag_v2"] for r in use if r["arch"] == arch]
                ax.scatter(xs, ys, s=28, color=col, marker=ARCH_MARK[arch], label=arch,
                           edgecolors=SURFACE, linewidths=1)
            ax.set_xlabel(f"{d} score")
            ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        axes[0].set_ylabel("relative fragility v2")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("H3: diagnostic score against relative fragility (higher = breaks more)", fontsize=11)
        p = fig_dir / "h3_scatter.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    dis = [r for r in rows if r.get("disagreement") is not None]
    if dis:
        fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
        _style(ax)
        scales = ["small", "medium", "large"]
        width = 0.26
        for k, (arch, col) in enumerate(ARCH_COLOR.items()):
            means = [np.mean([r["disagreement"] for r in dis if r["arch"] == arch and r["scale"] == s])
                     if any(r["arch"] == arch and r["scale"] == s for r in dis) else np.nan for s in scales]
            ax.bar(np.arange(3) + (k - 1) * width, means, width - 0.03, color=col, label=arch)
        floor = summary["H2"].get("noise_floor_disagreement_sd")
        if floor:
            ax.axhline(floor, color=INK2, linewidth=1, linestyle="--")
            ax.text(2.45, floor, " seed noise floor", fontsize=8, color=INK2, va="bottom", ha="right")
        ax.set_xticks(range(3), scales)
        ax.set_ylabel("mean disagreement (SD of rank percentiles)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("H2: disagreement by architecture and scale", fontsize=10)
        p = fig_dir / "h2_disagreement.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)
    return written


def _print(summary: dict[str, Any]) -> None:
    c = summary["counts"]
    print(f"{summary['domain']}: {c['planned']} planned, {c['trained']} trained, {c['admitted']} admitted, "
          f"{c['all_four_scores']} with all four scores, {c['with_fragility_v2']} with fragility v2")
    h = summary["H1"]
    if h.get("pairs"):
        print(f"H1 ({h['n_models']} models), tau with competence {({k: round(v, 3) for k, v in h['tau_with_competence'].items()})}"
              f"{'  A5: D1 pairs not interpretable' if h['a5_d1_not_interpretable'] else ''}")
        for q in h["pairs"]:
            ci = q["tau_ci"] or [float("nan")] * 2
            pci = q["partial_tau_ci"] or [float("nan")] * 2
            print(f"  {q['pair']}  tau {q['tau']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {q['category']:9s}  "
                  f"partial {q['partial_tau']:+.3f} [{pci[0]:+.3f}, {pci[1]:+.3f}] {q['partial_category']}"
                  f"{'  AGREEMENT THROUGH COMPETENCE' if q['agreement_through_competence'] else ''}")
        print(f"  Kendall W {h['kendall_w']:.3f} {h['kendall_w_ci']}, on residual ranks {h['kendall_w_residual']:.3f} "
              f"{h['kendall_w_residual_ci']}")
    else:
        print("H1:", h.get("note"))
    s = summary["A3"]["sensitivity_h1"]
    print("A3 sensitivity:", s if isinstance(s, str) else f"run on {s['n_models']} models")
    h2r = summary["H2"]
    if "factors" in h2r:
        print(f"H2: {h2r['verdict']} (noise floor of disagreement {h2r['noise_floor_disagreement_sd']})")
        for t in h2r["factors"]:
            print(f"  {t['factor']:12s} F {t['F']:.2f}  p {t['p']:.4f}  Holm {t['p_holm']:.4f}  effect {t['effect']:.3f}")
    else:
        print("H2:", h2r.get("note"))
    h3r = summary["H3"]
    if "D2" in h3r:
        print(f"H3 on {h3r['outcome']} ({h3r['n_models']} models): D1 reference Spearman "
              f"{h3r['D1_reference']['spearman']:+.3f} {h3r['D1_reference']['ci']}")
        for d in ("D2", "D3", "D4"):
            q = h3r[d]
            print(f"  {d} partial given D1 {q['partial_spearman_given_D1']:+.3f} {q['ci']}  adds to D1: {q['adds_to_D1']}")
        print(f"  best predictor: {h3r['best_predictor']}")
    else:
        print("H3:", h3r.get("note"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="H1, H2, H3 analysis for one domain (preregistration v1.3).")
    p.add_argument("--domain", default="othello")
    p.add_argument("--no_figures", action="store_true")
    a = p.parse_args()
    summ = analyse(a.domain)
    out = ROOT / "results" / "analysis" / a.domain
    figs = None if a.no_figures else ROOT / "reports" / "figures" / a.domain
    paths = write_outputs(summ, out, figs)
    _print(summ)
    for q in paths:
        print("wrote", q.relative_to(ROOT))
