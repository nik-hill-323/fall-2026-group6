# WM-Concord preregistration (version 1)

Written 2026-10-06, before any zoo model is scored by any diagnostic or by the fragility suite.
Frozen with the git tag `prereg-v1` once the team approves this file. After the tag, a change is
only allowed as a dated amendment at the end of this file, made before the results it affects
are seen. The paper reports every amendment.

Why this exists: H1 to H3 are judged against numbers we could, without meaning to, tune after
seeing results (thresholds, which models count, how a score is summarised). Writing every such
choice down first makes the results confirmatory rather than exploratory (Instructor Review 3).

## 1. What is fixed about the models

| Item | Value | Where |
|---|---|---|
| Domains | Othello, navigation, interpreter | `src/component/run_matrix.py` |
| Zoo | 3 domains x 3 architectures (transformer, LSTM, Mamba) x 3 scales x 3 distributions x 3 seeds = 243 models | `src/docs/models.csv` |
| Scales | small 4 layers x 128, medium 6 x 256, large 8 x 512 | `run_matrix.SCALE_CONFIG` |
| Training budget | 5,000 steps, batch 256 games, 1,280,000 training games, lr 3e-4, 200 warmup steps, weight decay 0.01, the same for every model | `run_matrix.TRAINING_BUDGET` |
| Othello distributions | synthetic (othello_world), championship (WTHOR), mixed (50 / 50) | `src/domains/othello.py` |
| Splits | train 80%, val 10%, test 10% (championship); synthetic test = last 10 val files | `src/domains/othello.py` |
| Split use | train: model training only. val: checks during training and fitting a diagnostic (probes, adaptation, any layer choice). test: final scores only | |
| H4 held out domain | the interpreter | `run_matrix.HELD_OUT_DOMAIN` |
| H4 held out architecture | none | `run_matrix.HELD_OUT_ARCH` |
| Fragility suite | Othello version 1, frozen with the tag `fragility-v1` | `src/fragility/othello.py` |

**Training budget decision.** The large transformer at this budget reaches a legal move rate of
0.969 on validation games (Othello GPT, trained much longer, reaches 0.999). We keep the budget:
a fourfold budget would cost about 150 GPU hours for Othello alone, and the study needs models
of different quality, since rank correlations between diagnostics only mean something when the
models differ. Measured on 2026-10-06, `src/docs/zoo_registry.csv`.

**Reference models.** The two published Othello GPTs (synthetic and championship, Li et al. 2023)
are scored by every diagnostic as anchors. They are reported, but never enter the H1 to H4
statistics, because they were trained with a different budget and code.

## 2. Which models count: the admission rule

A zoo model enters H1 to H3 only if its **legal move rate on 1,000 validation games is at least
0.50**, as written by `src/zoo/train.py` in its training record.

* Why 0.50: an untrained model scores about 0.15 (about 9 of 60 squares are legal on average), and
  the first trained models score 0.57 to 0.97. A model below 0.50 has not learned the task, so a
  world model diagnostic has nothing to measure in it.
* The rule uses validation games, never test games, and was chosen after seeing only training
  records, before any diagnostic or fragility score.
* Models that fail are still scored and reported, with `admitted = no` in `zoo_registry.csv`.
  The paper states how many failed per architecture and scale.

## 3. The four diagnostics: the one number each writes

Every diagnostic writes one result per model in the format of `src/docs/results_schema.md`.
`score` is the master table number, and **higher always means a more world model like model**.
Final scores use 1,000 test games of the model's own training distribution, data seed 0.

| Diagnostic | `score` | Fitted on | Owner |
|---|---|---|---|
| D1 next token validity | share of positions where the model's top predicted move is legal | nothing fitted | Viharika |
| D2 state equivalence | 1 minus the mean of compression error and distinction error (Vafa et al. 2024); both errors in `details` | nothing fitted | Venkatesh |
| D3 inductive bias | adaptation gain over a randomly initialised model of the same architecture and scale, on small tasks whose labels come from the board (Vafa et al. 2025) | adaptation on val games | Nikhil |
| D4 probing | selectivity (probe accuracy minus Hewitt and Liang control task accuracy) of a linear probe for the mine / theirs / empty board, at the layer with the highest selectivity on val games, measured on test games | probes on val games | Siddu |

D4 also writes in `details`, for every layer: probe accuracy, control accuracy, selectivity, gain
over layer 0, and the occupancy only baseline. The layer used for `score` is chosen on val games
so that the choice never sees test games.

**D2 and D3 details.** Before the first D2 or D3 score is computed, its owner adds an amendment
here that fixes the remaining protocol choices (number of sampled pairs or tasks, adaptation steps,
learning rate, seeds). Those choices may be tested on the reference Othello GPT only, never on
zoo models.

**Causal check (D4).** After the main D4 run, the probe direction is used to intervene on the
model. A model whose board is decodable (selectivity at least 0.10) but whose next move
predictions do not follow the intervention is flagged "decodable but not causal". The exact
pass rule is added as an amendment before the causal check is run on any zoo model.

## 4. Analysis plan

All analyses run per domain on admitted models. Seeds are separate models (3 per configuration).
Every interval is a 95% bootstrap interval over models, 2,000 resamples, seed 0.

### H1: do the four diagnostics agree?

* For each of the 6 pairs of diagnostics: Kendall tau (main) and Spearman rho between their scores
  across models, with bootstrap intervals.
* All four at once: Kendall W.
* Decision rule: a pair **agrees** if tau is at least 0.50 and its interval excludes 0; a pair
  **disagrees** if its interval includes 0 or tau is below 0.30; anything else is **partial**.
* If every pair agrees, we report the null result for H1 and the paper's weight moves to H3 and H4
  (as the proposal's stage gate says).

### H2: is the disagreement systematic or noise?

* Per model, disagreement = standard deviation of its four rank percentiles (one per diagnostic).
* Noise floor: for each configuration (architecture, scale, distribution), the standard deviation
  of each diagnostic's score across its 3 seeds.
* Regress disagreement on architecture, scale and distribution (categorical, least squares).
* Decision rule: disagreement is **systematic** if at least one factor is significant at 0.05 after
  Holm correction and its effect is larger than the noise floor; otherwise it is reported as
  consistent with noise.

### H3: which diagnostic predicts fragility?

* Outcome: `fragility` from the frozen suite (`fragility-v1`): the mean drop in legal move rate under
  shifted situations. Higher is more fragile.
* For each diagnostic: Spearman correlation between its score and fragility (expected negative:
  a better world model breaks less), and a regression of fragility on the score with architecture
  and scale as covariates.
* Decision rule: a diagnostic is the **best predictor** if its correlation with fragility is
  stronger than each other diagnostic's and the bootstrap interval of each difference excludes 0.
  Otherwise we report that no single diagnostic wins and give the ranking with intervals.
* D1 is tested on its own: we report whether its correlation with fragility is different from 0
  at all (the literature expects it not to be).
* D4 "decodable but not causal" models are reported separately, and we test whether the flag adds
  predictive power beyond D4 selectivity.

### H4: do the findings hold on the held out domain?

* H1 to H3 are computed on Othello and navigation and written down before any interpreter model is
  scored. Then the same code runs on the interpreter.
* Findings **generalize** if, on the interpreter, each pair keeps its H1 category (agree, partial,
  disagree) and the H3 best predictor (or "no winner") is the same.

## 5. What is reported no matter what

* Every planned run has a score or a written reason (`failed` or `skipped`), per
  `src/component/results.py` `missing_report`.
* The number of admitted and not admitted models per architecture and scale.
* All four diagnostic scores for the two reference Othello GPTs.

## 6. Who signs

| Name | Role | Approved |
|---|---|---|
| Siddardha Reddy | D4, coordination | |
| Nikhil Obuleni | D3, zoo | |
| Viharika Vemparala | D1 | |
| Venkatesh Nagarjuna | D2 | |

Approval is given on the pull request. After merge: `git tag prereg-v1 && git push origin prereg-v1`.

## Amendments

### Amendment 1 (prereg-v1.1), 2026-10-12

Written after Instructor Review #4 (#70, critiques 1 and 2) and before any zoo model is scored by
D1, D2, D3, D4 or the fragility suite. At this point only training records exist (legal move rate on
validation games). Frozen with the git tag `prereg-v1.1`. Sections 1 to 5 above stay in force except
where this amendment says otherwise.

**Why.** Every diagnostic is likely to score a better trained model higher. Across a zoo whose
validation legal move rate runs from 0.50 to about 0.97, the four diagnostics could agree only
because they all follow competence, not because they measure the same world model. And D1 is the
legal move rate while fragility is a drop in legal move rate, so D1 and the H3 outcome partly
measure the same thing. A1 to A5 handle both before any data exists.

**Competence** below always means a model's validation legal move rate on 1,000 validation games,
the `val_legal_rate` column of `src/docs/zoo_registry.csv`. It is fixed before any diagnostic runs,
so no diagnostic score enters it.

#### A1. H1 is also reported with competence controlled

* For each of the 6 pairs of diagnostics, next to Kendall tau (Section 4), we report the **partial
  Kendall tau** controlling for competence:
  tau(x, y | c) = (tau(x, y) minus tau(x, c) times tau(y, c)) divided by the square root of
  ((1 minus tau(x, c) squared) times (1 minus tau(y, c) squared)).
  Intervals: the same bootstrap as Section 4 (95%, 2,000 resamples over models, seed 0), with the
  partial tau recomputed in every resample.
* The partial tau gets the same categories as the raw tau (agree, partial, disagree, same
  thresholds).
* A pair whose raw tau is **agree** and whose partial tau is **disagree** is reported as
  **agreement through competence**.
* Kendall W is reported on the raw scores (as in Section 4) and also on the residual ranks: each
  diagnostic's scores are ranked, the competence ranks are regressed out by least squares, and W is
  computed on the residuals.
* The H1 conclusion in the paper states both results. If they differ, the partial result is the one
  used to say whether the diagnostics measure the same thing.

#### A2. H3: fragility relative to each model's own baseline, D1 as the reference

* **Outcome.** Relative fragility = `fragility` divided by `in_distribution_legal_rate`, both taken
  from the output of `score_model` in the frozen suite (`src/fragility/othello.py`, `fragility-v1`).
  The frozen code is not changed; the division is done in the analysis. Raw `fragility` is still
  reported as a secondary result.
* **D1 is the reference, not a competitor.** We report the Spearman correlation between D1 and
  relative fragility, with its bootstrap interval, as the baseline any other diagnostic has to beat.
* **The contest is among D2, D3 and D4.** For each of them:
  1. the partial Spearman correlation with relative fragility, controlling for D1;
  2. a least squares regression of relative fragility on D1, the diagnostic, architecture and scale
     (categorical), reporting the diagnostic's coefficient and its 95% interval.
* **Decision rule (replaces the H3 best predictor rule in Section 4).** A diagnostic among D2, D3
  and D4 is the **best predictor** if its partial correlation (step 1) is the strongest in the
  expected negative direction and the bootstrap interval of each difference to the other two
  excludes 0. A diagnostic **adds to D1** if its partial correlation interval excludes 0. Otherwise
  we report that no diagnostic adds predictive value beyond the legal move rate, which is itself a
  finding.
* The Section 4 analysis (raw correlations, all four diagnostics) is still reported, labelled
  secondary.

#### A3. Sensitivity: the competence range of the admitted zoo

* We report the distribution of competence for admitted models (a histogram, plus the count, mean
  and range for each architecture and scale).
* **Sensitivity run.** H1 (raw and partial tau, Kendall W) is repeated on admitted models with
  competence of at least **0.90**. It is run only if at least 10 models qualify; otherwise we report
  how many qualified. This is a sensitivity analysis and does not replace the main H1 result.

#### A4. D2 protocol (state equivalence)

Fixed here as Section 3 requires, before the first D2 score. It follows the compression and
distinction errors of Vafa et al. (2024), using the next move as the continuation.

* **Data.** Test games of the model's own training distribution, 1,000 games, data seed 0
  (Section 3). Every prefix of length 1 to 59 is a history. Its true state is the board and the
  player to move, from `src/domains/othello.py` (`replay`). Histories with no legal move for the
  player to move are left out.
* **Accepted moves.** For a history h, the model's accepted set A(h) is its k highest scoring
  squares at the last position of h (token 0, padding, excluded), where k is the number of truly
  legal moves L(h). No threshold is tuned.
* **Same state pairs.** Two histories with the same true state but different move sequences. Up to
  2,000 pairs, sampled uniformly at random from all such pairs with seed 0. If fewer than 500 exist,
  the number of test games is raised in steps of 1,000, up to 5,000, and the number used is
  reported.
* **Different state pairs.** Two histories of the same length whose true legal move sets differ.
  Up to 2,000 pairs, sampled uniformly at random with seed 0.
* **Compression error** of a same state pair (h, g): the share of accepted moves that are accepted
  after one history but not after the other, |A(h) symmetric difference A(g)| divided by
  |A(h) union A(g)|. The model should treat the two histories the same, so 0 is best.
* **Distinction error** of a different state pair (h, g): the boundary is every move legal after
  exactly one of them (L(h) symmetric difference L(g)). A boundary move m that is legal after h only
  is separated correctly if m is in A(h) and not in A(g) (and the mirror case for g). The error is
  the share of boundary moves not separated correctly. 0 is best.
* **Score.** `score` = 1 minus the mean of the mean compression error and the mean distinction
  error. `details` holds both mean errors, their standard errors over pairs, the number of pairs of
  each kind and the number of games.
* **Check before the zoo.** D2 is run first on the two reference Othello GPTs only (Section 3).
  Nothing in this protocol is changed after any zoo model is scored.
* **Limitation, stated in the paper.** Vafa et al. (2024) also use longer continuations; we use the
  next move only, which keeps D2 cheap enough for 81 models per domain and uses the same ground
  truth as the rest of the study.

#### A5. Pairs that include D1

D1 is the legal move rate on test games, so it is nearly the same quantity as competence. For the 3
H1 pairs that include D1, the partial tau is still reported, but if tau(D1, competence) is at least
0.90 it is marked **not interpretable** and the competence controlled conclusion rests on the 3 pairs
among D2, D3 and D4.

#### A6. D4 protocol (representation probing) and D1

Fixed here before the first D4 zoo score. Section 3 (score = selectivity at the layer with the
highest selectivity on val games, measured on test games) is unchanged; this fixes the rest.

* **Data.** Games of the model's own training distribution, data seed 0:
  * probe fitting: the first 10,000 of 11,000 val games;
  * layer choice: the last 1,000 of those 11,000 val games (never used for fitting);
  * score: 1,000 test games.
* **Layers.** `ZooModel.activations`: the residual stream after each block (transformer, Mamba) or
  the hidden state after each layer (LSTM). Layer 0 is the output of the first block. Every real
  position of every game is used.
* **Labels.** Mine, theirs or empty for each of the 64 squares, relative to the player who just
  moved (`mine_theirs` in `src/domains/othello.py`).
* **Probe.** One linear map per layer from the activation to 64 squares x 3 classes, cross entropy,
  AdamW (learning rate 1e-3, weight decay 0.01), 2 epochs, batches of 256 games, seed 0.
* **Control task (Hewitt and Liang 2019).** The input type of a position is (square just played,
  position in the game). For each type and each square, a fixed random label is drawn once from
  that square's class frequencies in the probe fitting games (seed 0). The control probe has the
  same form and training as the real probe, on these labels.
* **Selectivity** of a layer = probe accuracy minus control accuracy (all 64 squares). The best
  layer is the one with the highest selectivity on the layer choice games. `score` = selectivity of
  that layer on the test games.
* **Details, per layer, on test games:** probe accuracy, control accuracy, selectivity, gain over
  layer 0 (probe accuracy minus layer 0 probe accuracy), and probe accuracy on occupied squares
  only (whose disc is on each square, which the move list alone does not give away). Also the
  best layer, its depth as a share of the layers, and the **occupancy only baseline**: the accuracy
  of a rule that knows only which squares are occupied (empty squares are called empty, occupied
  squares get that square's most common disc class in the probe fitting games), with each layer's
  gain over it.
* **D1** follows `src/docs/d1_next_token_validity.md` (PR #62) unchanged: nothing is fitted, the score
  is the argmax legal move rate on 1,000 test games of the model's own training distribution.

#### Still to come, each before its first zoo score

* A7, D3 protocol (owner: Nikhil): number of tasks, adaptation steps, learning rate, seeds.
* D4 causal check pass rule (owner: Siddu), before the causal check in Weeks 9 to 10.
