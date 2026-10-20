# WM-Concord: Othello results (81 models)

Source files: `summary.json`, `h1_pairs.csv`, `h3.csv`, `model_table.csv` (preregistered analysis,
`src/analysis/hypotheses.py`, run after tag `prereg-v1.4`), and `exploratory.json`, `descriptives.csv`
(exploratory, `src/analysis/exploratory_othello.py`, decided after the preregistered results were read).
Figures: `reports/figures/othello/`. Every interval is a 95% configuration bootstrap (27 configurations,
3 seeds kept together, 2,000 resamples, seed 0; Amendment 4).

## In one paragraph

All 81 zoo models (transformer, LSTM, Mamba; small, medium, large; synthetic, championship, mixed
training data; 3 seeds) passed admission and were scored by all four diagnostics and both fragility
suites. The four diagnostics agree strongly when ranking models (Kendall W 0.87), but most of that
agreement is competence: once the validation legal move rate is controlled for, D2 and D3 disagree and the
pairs with D4 fall to "agreement through competence" (W on residual ranks 0.27). D1 is almost the same
quantity as competence (tau 0.97), so under the preregistered A5 rule its controlled pairs are not
interpretable. Disagreement between the diagnostics is consistent with seed noise (H2). For predicting
fragility (H3), D1 carries no signal; D4 is the only diagnostic that adds to D1 under the preregistered
analysis, but no single diagnostic formally wins, and an exploratory check shows the D4 signal comes from
the training distribution: championship trained models are both the most fragile and the lowest on D4, and
within each training distribution, or controlling for it, D4 no longer predicts fragility.

## Zoo and admission

| | value |
|---|---|
| Models planned, trained, admitted | 81, 81, 81 (0 below the 0.50 rule) |
| Competence (validation legal move rate) | 0.559 to 0.969, mean 0.821 |
| Models at competence 0.90 or above (A3) | 23 |
| Results on main | 486 files (81 models x D1, D2, D3, D4, fragility v1, fragility v2), all status ok |
| Missing | 0 |

## Descriptives (means; `descriptives.csv`)

| Architecture / scale | competence | D1 | D2 | D3 | D4 | relative fragility v2 | decodable and causal |
|---|---|---|---|---|---|---|---|
| transformer small | 0.841 | 0.843 | 0.751 | 0.014 | 0.194 | 0.022 | 0 of 9 |
| transformer medium | 0.915 | 0.917 | 0.817 | 0.038 | 0.247 | 0.021 | 3 of 9 |
| transformer large | 0.932 | 0.933 | 0.840 | 0.056 | 0.268 | 0.013 | 4 of 9 |
| LSTM small | 0.583 | 0.584 | 0.607 | 0.013 | 0.120 | 0.008 | 0 of 9 |
| LSTM medium | 0.688 | 0.690 | 0.632 | 0.014 | 0.155 | 0.020 | 0 of 9 |
| LSTM large | 0.778 | 0.780 | 0.666 | 0.017 | 0.174 | 0.014 | 0 of 9 |
| Mamba small | 0.835 | 0.837 | 0.722 | 0.025 | 0.203 | 0.019 | 0 of 9 |
| Mamba medium | 0.897 | 0.899 | 0.789 | 0.044 | 0.240 | 0.007 | 1 of 9 |
| Mamba large | 0.924 | 0.925 | 0.826 | 0.070 | 0.269 | 0.008 | 2 of 9 |

| Training data | competence | D1 | D2 | D3 | D4 | relative fragility v2 | decodable and causal |
|---|---|---|---|---|---|---|---|
| synthetic | 0.831 | 0.832 | 0.744 | 0.040 | 0.236 | -0.024 | 9 of 27 |
| championship | 0.812 | 0.814 | 0.719 | 0.031 | 0.175 | +0.090 | 0 of 27 |
| mixed | 0.821 | 0.823 | 0.753 | 0.025 | 0.212 | -0.022 | 1 of 27 |

* Every diagnostic rises with scale for every architecture. LSTMs are the weakest on all four; transformers
  and Mamba are close, Mamba large is the highest on D3 (0.070).
* All 10 "decodable and causal" models are transformer or Mamba, medium or large, and 9 of 10 were trained
  on synthetic games.
* Anchors for comparison (test split): Othello GPT synthetic D1 0.9996, D2 0.999, D3 0.140, D4 0.435
  (decodable and causal); championship D1 0.896, D2 0.766, D3 0.059, D4 0.272 (decodable but not causal).
  The best zoo models (D4 up to 0.314) sit between the two anchors on D4.

## Diagnostic details (means over 81 models; `exploratory.json`)

* **D2:** compression error 0.146 (0.021 to 0.289), distinction error 0.376 (0.160 to 0.648). Models treat
  histories that reach the same board alike fairly well, but often fail to separate boards whose legal moves
  differ.
* **D3:** adapted pretrained balanced accuracy 0.579 against 0.547 for a random weight model of the same
  size; mean gap 0.032 (anchors 0.140 and 0.059). Symbolic regression recovers the "is X mine" law in 3.8%
  of searches (random weight models 2.7%), while the occupancy shortcut appears in 22.7% (random 14.8%).
  Across models the D3 score correlates with the occupancy shortcut rate (Spearman 0.36 [0.04, 0.61]): part
  of what D3 rewards is the occupancy shortcut, as Amendment 2 anticipated.
* **D4:** linear selectivity 0.208 (0.111 to 0.314); MLP selectivity 0.183, lower than linear, as Hewitt
  and Liang predict. The best layer sits at 83% of depth on average. The best probe beats the occupancy only
  baseline by 0.041 on average, and for some models not at all (minimum -0.083). Causal score 0.126 (0.009
  to 0.256); 10 of 81 models pass the 0.20 rule.

## H1: do the diagnostics agree? (preregistered; `h1_pairs.csv`, `h1_concordance.png`)

| Pair | Kendall tau [95% CI] | category | partial tau given competence [95% CI] | category |
|---|---|---|---|---|
| D1 and D2 | 0.842 [0.711, 0.920] | agree | 0.303 [0.154, 0.539] | not interpretable (A5) |
| D1 and D3 | 0.574 [0.407, 0.688] | agree | 0.076 [-0.003, 0.211] | not interpretable (A5) |
| D1 and D4 | 0.801 [0.688, 0.867] | agree | 0.238 [0.106, 0.440] | not interpretable (A5) |
| D2 and D3 | 0.480 [0.264, 0.631] | partial | 0.006 [-0.107, 0.171] | disagree |
| D2 and D4 | 0.683 [0.504, 0.804] | agree | 0.090 [-0.029, 0.341] | disagree: agreement through competence |
| D3 and D4 | 0.553 [0.376, 0.680] | agree | 0.194 [0.003, 0.393] | disagree: agreement through competence |

* Kendall W, raw: 0.872 [0.764, 0.921]. On residual ranks (competence removed): 0.267 [0.189, 0.427].
* tau with competence: D1 0.968, D2 0.826, D4 0.790, D3 0.577. D1 is above the 0.90 line of A5, so the
  competence controlled conclusion rests on the three pairs among D2, D3 and D4, all of which disagree.
* **Verdict:** the diagnostics agree, and they agree because they all follow competence. With competence
  controlled, none of the D2, D3, D4 pairs agrees.
* **A3 sensitivity (23 models at competence 0.90 or above):** raw agree for D1 and D2, D1 and D4, D2 and D4;
  the rest disagree; with competence controlled only D1 and D2 still agree (0.532 [0.088, 0.838]; tau(D1,
  competence) is 0.855 in this subset, so A5 does not apply). W 0.830, on residual ranks 0.466.

## H2: is the disagreement systematic? (preregistered; `h2_disagreement.png`)

* Mean disagreement (SD of the four rank percentiles) 0.089; seed noise floor of the disagreement 0.028.
* F tests: architecture p 0.783, scale p 0.027 (Holm 0.081), distribution p 0.171 (Holm 0.341).
* **Verdict:** consistent with noise; no factor is significant after Holm. Scale comes closest
  (disagreement 0.110 small, 0.083 medium, 0.073 large).

## H3: which diagnostic predicts fragility? (preregistered; `h3.csv`, `h3_scatter.png`)

Fragility first (Amendment 4, F3; `fragility_distribution.png`): relative fragility v2 has mean 0.015,
sd 0.070, range -0.124 to 0.215, and 52% of models are below 0 (better under the shifted tests than on
their own test games).

| | statistic [95% CI] | adds to D1 |
|---|---|---|
| D1 (reference): Spearman with relative fragility v2 | 0.150 [-0.315, 0.588] | |
| D2: partial Spearman given D1 | 0.031 [-0.297, 0.438] | no |
| D3: partial Spearman given D1 | 0.026 [-0.373, 0.386] | no |
| D4: partial Spearman given D1 | **-0.554 [-0.734, -0.124]** | **yes** |

* D4 regression coefficient (with D1, architecture, scale): -1.48 [-2.47, -0.60].
* Best predictor rule: D4 is strongest, but the difference to D2 includes 0 ([-1.114, 0.098]), so **no single
  diagnostic wins**; D4 is the only diagnostic that adds to D1.
* Robustness (preregistered): without othello_transformer_small_synthetic_s0 (Amendment 4, F2) D4 is
  -0.561 [-0.738, -0.117]; on relative fragility v1 D4 is -0.540 and still adds to D1; same verdict.
* Section 4 (secondary): raw Spearmans with fragility are all near 0 (D1 0.15, D2 0.15, D3 0.13, D4 -0.04);
  D1 alone does not predict fragility. D4's signal appears only after controlling for D1.
* D4 causal flag: adds nothing beyond D4 selectivity (partial -0.049 [-0.241, 0.122]).

## Exploratory, not preregistered: fragility is mostly a training data effect (`exploratory.json`)

Mean drop per fragility v2 variant, by training data:

| Shift | synthetic | championship | mixed |
|---|---|---|---|
| other training distribution | -0.007 (tested on championship) | **+0.190** (tested on synthetic) | +0.009 on championship, -0.009 on synthetic |
| symmetries (rot180, transpose, anti transpose) | about 0.000 | **+0.18, +0.14, +0.17** | about -0.013 |
| after pass (matched) | -0.020 | +0.034 | -0.007 |
| detour, random p 0.1 / 0.5 | -0.007 / -0.048 | -0.005 / -0.032 | -0.010 / -0.046 |
| detour, adversarial p 0.1 / 0.5 | -0.011 / -0.050 | -0.030 / -0.031 | -0.010 / -0.041 |

* Championship trained models break on random games and on rotated or mirrored boards (human games are
  recorded in one orientation); synthetic and mixed models do not. With detours every group is slightly more often
  legal than on its own line (negative drops), so detours do not make these models fragile.
* Championship models also have the lowest D4 (0.175 against 0.236 synthetic). The preregistered H3 controls
  for D1 in the partial correlation and for architecture and scale in the regression, not for training data.

| D4 partial Spearman with relative fragility v2 given D1 | value [95% CI] |
|---|---|
| all 81 models (preregistered) | -0.554 [-0.734, -0.124] |
| within synthetic (27) | +0.449 [-0.226, 0.726] |
| within championship (27) | -0.530 [-0.761, 0.191] |
| within mixed (27) | +0.341 [-0.221, 0.666] |
| all 81, controlling for D1 and training data | +0.231 [-0.336, 0.589] |

* The sign flips between training distributions and every within distribution interval includes 0; with
  training data controlled the D4 effect disappears (D2 +0.153 [-0.346, 0.555], D3 -0.259 [-0.502, 0.078]).
* **Reading:** the preregistered D4 result is real as computed but is a between distribution effect (a
  Simpson's paradox): it says championship trained models are both less probe readable and more fragile,
  not that probing predicts fragility among comparable models. With 9 configurations per distribution the
  within distribution estimates are imprecise; the honest summary is that no diagnostic shows a reliable
  within distribution link to fragility on Othello.

## What to say

1. The four diagnostics agree on which Othello models are better (W 0.87), but that agreement is competence;
   with competence controlled they disagree (W 0.27).
2. D1 is competence (tau 0.97), so it cannot be separated from it.
3. Disagreement is not systematic across architecture, scale or training data (H2).
4. Only D4 adds predictive value beyond D1 for fragility under the preregistered test, but this comes from
   the training data, not from within comparable models; no diagnostic is a reliable fragility predictor on
   Othello.
5. D3 partly rewards an occupancy shortcut; only 10 of 81 models use their board representation causally,
   all of them medium or large transformers or Mamba, 9 of them trained on synthetic games.
