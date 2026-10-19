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
  AdamW (learning rate 1e-3, weight decay 0.01), 2 epochs, seed 0. Activations are computed 256
  games at a time and the probe takes one step per 1,024 positions (about 1,100 steps for 10,000
  games). Each activation feature is first standardized with its mean and standard deviation over
  the probe fitting games, so the same learning rate is fair to every architecture (LSTM hidden
  states lie between minus 1 and 1, residual streams are much larger). Checked on a model whose
  layer holds the true board: the probe reads it at 0.9999 with these settings.
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

#### A7. D3 protocol (inductive bias)

Fixed here before the first D3 zoo score, as Section 3 requires. It follows the inductive bias
probe of Vafa et al. (2025): adapt the model to a small task whose labels come from the true
board, and compare with a model of the same architecture and size that has random weights.
Code: `src/diagnostics/d3/inductive_bias.py`.

* **Tasks.** Four binary questions about one square right after a move: "is square X mine?", for
  X = C4, F5, C3 and F6, where mine means the disc belongs to the player who just moved (the
  `mine_theirs` board of `src/domains/othello.py`). Occupancy tasks ("is X occupied?") are not
  used: on 2,000 synthetic validation games, whether a square is occupied equals whether its token
  appears in the move list in 100% of cases, so it can be answered without any board. Whose disc it
  is depends on the flips and cannot be read off the move list. The positive rate of the four tasks
  is 0.41 to 0.45 on the same games.
* **Labelled data for adaptation.** 64 validation games of the model's own training distribution,
  data seed 0, one uniformly random position per game (position seed 0). The same 64 examples are
  used for all four tasks.
* **What is adapted.** A deep copy of the whole model, plus a new linear head (d_model to 2) on the
  vector after the last layer at that position. Head learning rate 1e-3, rest of the model 1e-4,
  AdamW, weight decay 0.01, 300 steps of 16 examples drawn with replacement, cross entropy. The
  original model is never changed.
* **How the learning rates were chosen.** On the synthetic reference Othello GPT only, never on a
  zoo model, as Section 3 allows, with everything else as in this amendment (2026-10-13):

  | head lr | body lr | score | sd over seeds | pretrained | random |
  |---|---|---|---|---|---|
  | 1e-3 | 1e-5 | 0.103 | 0.003 | 0.666 | 0.563 |
  | **1e-3** | **1e-4** | **0.140** | 0.010 | 0.699 | 0.559 |
  | 1e-2 | 1e-5 | 0.094 | 0.006 | 0.653 | 0.559 |
  | 1e-2 | 1e-4 | 0.132 | 0.003 | 0.682 | 0.550 |

  A body learning rate of 1e-5 was not chosen for a second reason: it leaves the model almost
  frozen, so D3 would become a linear probe on the last layer, close to D4 by construction, and
  their agreement under H1 would be partly built in.
* **Baseline.** A model of the same architecture and scale with random weights
  (`src/zoo/train.py` `build_model` for zoo models, a freshly initialised 8 layer TransformerLens
  model for the reference Othello GPTs), adapted in exactly the same way. One random model per seed.
* **Seeds.** 3 adaptation seeds (0, 1, 2). The seed sets the head initialisation, the batch draws
  and the random baseline's weights. The toy version showed a gap between 0.12 and 0.33 across seeds,
  so one seed is not enough.
* **Scored on.** 1,000 test games of the model's own training distribution, data seed 0, one
  uniformly random position per game (position seed 0).
* **Task score.** Balanced accuracy: the mean of the hit rate on "mine" and on "not mine", so a
  model that always answers the more common class scores 0.5, not 0.55 to 0.59.
* **Score.** `score` = the mean over the 4 tasks and 3 seeds of (adapted pretrained balanced
  accuracy minus adapted random balanced accuracy). Higher is more world model like. `score_std`
  is the standard deviation of the per seed mean gap across the 3 seeds.
* **Details, per task and seed:** both balanced accuracies, both plain accuracies and the gap. Also
  the mean gap per task and per seed, the mean balanced accuracy of each model, and the positive
  rate of every task in the adaptation and test data.
* **Check before the zoo.** D3 is run first on the two reference Othello GPTs only. Nothing in this
  protocol is changed after any zoo model is scored.
* **Limitation, stated in the paper.** A model whose last layer already holds the board reaches
  only about 0.8 balanced accuracy from 64 examples (tested with a model whose last layer is the
  true board), because many features fit 64 points. D3 therefore measures how much the pretrained
  model helps, not how close it gets to a perfect answer.

#### Still to come, each before its first zoo score

* D4 causal check pass rule (owner: Siddu): done in Amendment 2, B2, before any zoo model was scored.

### Amendment 2 (prereg-v1.2), 2026-10-15

Written and tagged before any zoo model is scored by any diagnostic: when it was written, `results/`
held only the two reference Othello GPTs for D1 to D4. It completes D2, D3 and D4 as the proposal
describes them, so that no part of a diagnostic is added after zoo scores exist. Nothing in Sections 1
to 5 or Amendment 1 changes: every master table `score` is computed exactly as before (the D3 and D4
anchor scores were rerun with this code and reproduce to every digit; each anchor file records the
commit that holds the code). All choices below were made on the two reference Othello GPTs only, never
on a zoo model, as Section 3 allows. D3 and D4 refuse zoo models until the git tag `prereg-v1.2`
exists. D1 and D2 keep their prereg-v1.1 gate: their code is unchanged by this amendment, and no zoo
model is to be scored by any diagnostic before `prereg-v1.2` exists.
After merge: `git tag prereg-v1.2 && git push origin prereg-v1.2`.

#### B1. D4 nonlinear probe (proposal, Week 9: "linear and nonlinear probes")

* **Probe.** Per layer, one hidden layer of 256 units with ReLU, then 64 squares x 3 classes.
* **Control.** Its own Hewitt and Liang control probe of the same form, on the A6 control labels.
* **Data and training.** Exactly as A6: same fitting, layer choice and test games, same standardized
  activations, AdamW (learning rate 1e-3, weight decay 0.01), 2 epochs, one step per 1,024
  positions, seed 0. The MLP probes are created after the linear probes and have their own
  optimizer, so the linear probes, and therefore `score`, are unchanged.
* **Reported in `details["mlp"]`:** per layer probe accuracy, control accuracy, selectivity and
  probe accuracy on occupied squares; its own best layer (highest selectivity on the layer choice
  games) and selectivity there; per layer gain over the linear probe.
* **Not the score.** The D4 score stays the linear selectivity (A6). Hewitt and Liang (2019) show that
  a more expressive probe also raises control accuracy, so the linear probe is the cleaner measure of
  what the model itself makes readable. The anchors show exactly this:

  | Anchor (test) | linear: layer, probe, control, selectivity | MLP: layer, probe, control, selectivity |
  |---|---|---|
  | Othello GPT synthetic | 6, 0.9903, 0.5557, **0.4345** | 7, 0.9898, 0.5740, 0.4157 |
  | Othello GPT championship | 3, 0.8989, 0.6269, **0.2720** | 4, 0.9018, 0.6722, 0.2297 |

#### B2. D4 causal check and its pass rule (Section 3, "Causal check (D4)"; proposal, Week 10)

* **Cases.** One per test game (the same 1,000 test games as the score). In each game, positions that
  have a recorded next move are tried in a random order (seed 0); at each, occupied squares are tried
  in a random order. The first (position, square) where flipping that disc (mine to theirs or theirs
  to mine) changes the set of legal next moves, and leaves at least one, is the case.
* **Edited layers.** Every layer from the middle one (n_layers // 2, or the linear best layer of A6 if
  that is earlier) to the last, each with its own linear probe from A6. The edits run during one
  forward pass in layer order, so each later layer sees the earlier edits
  (`ZooModel.logits_edited`).
* **Edit at a layer.** In the standardized activation space, the vector at the case's position moves
  along the normalized gradient of that layer's probe cross entropy for the edited board (all 64
  squares), step length 0.05 x sqrt(width), until the probe reads the whole edited board and gives the
  flipped square's new class a probability of at least **0.99**, at most 100 steps.
* **Control edit.** At every edited layer, a random direction (seed 0) with exactly the same length as
  the probe edit there.
* **Boundary accuracy** of a case: the model accepts its k best moves, k = the number of legal moves
  after the flip. Boundary moves are those legal before or after the flip but not both. A move that
  becomes legal is right if accepted, a move that stops being legal is right if not accepted. Boundary
  accuracy is the share of boundary moves that are right (the same idea as the D2 distinction error).
* **Causal score** = mean over cases of (boundary accuracy with the probe edit minus boundary accuracy
  with the random edit), with a 95% bootstrap interval over cases (2,000 resamples, seed 0). Also
  reported: boundary accuracy with no edit, the share of cases where every layer's probe reached the
  edited board, and per layer the edit success, flipped square reading, steps and edit size.
* **Pass rule and label.**
  * selectivity below 0.10: **not decodable** (the causal check is still reported);
  * selectivity at least 0.10, causal score at least **0.20** and the interval above 0:
    **decodable and causal**;
  * otherwise: **decodable but not causal**.
* **Why 0.20.** It asks the probe edit to get at least one boundary move in five more right than a
  random edit of the same size. It was written down before the comparison below was run and was not
  changed after it.
* **How the edited layers and the confidence were chosen.** On the reference Othello GPTs only. The
  synthetic Othello GPT is the positive control: Li et al. (2023) and Nanda et al. (2023) show that
  editing its board changes its moves. The setting with the largest causal score on it was taken.

  | Edited layers | confidence | synthetic: causal score [95% CI] | championship: causal score [95% CI] |
  |---|---|---|---|
  | best only | 0.9 | 0.047 [0.030, 0.063] | 0.079 [0.056, 0.101] |
  | best only | 0.99 | 0.076 [0.057, 0.097] | 0.130 [0.105, 0.156] |
  | best to last | 0.9 | 0.070 [0.051, 0.087] | 0.122 [0.098, 0.146] |
  | best to last | 0.99 | 0.115 [0.093, 0.137] | 0.151 [0.122, 0.180] |
  | middle to last | 0.9 | 0.652 [0.622, 0.679] | 0.122 [0.098, 0.146] |
  | **middle to last** | **0.99** | **0.693 [0.665, 0.719]** | **0.151 [0.122, 0.180]** |
  | all layers | 0.9 | 0.578 [0.546, 0.607] | 0.152 [0.128, 0.178] |
  | all layers | 0.99 | 0.529 [0.498, 0.559] | 0.180 [0.152, 0.211] |

  Editing only from the best layer (6 of 8 on synthetic) comes too late: the later layers still attend
  to earlier positions that hold the unedited board, and the synthetic model builds its board around
  layer 4 (Nanda et al. 2023). Editing from layer 0 adds noise where no board is held yet (the random
  edit's boundary accuracy rises from about 0.18 to 0.24 to 0.29). The probe edit reached its target
  in 100% of cases in every setting.
* **Anchors (test split), with the chosen setting:**

  | Anchor | edited layers | causal score [95% CI] | boundary acc: probe edit, random edit, no edit | label |
  |---|---|---|---|---|
  | Othello GPT synthetic | 4 to 7 | 0.6927 [0.6647, 0.7192] | 0.8719, 0.1792, 0.1975 | decodable and causal |
  | Othello GPT championship | 3 to 7 | 0.1510 [0.1220, 0.1803] | 0.5440, 0.3930, 0.3868 | decodable but not causal |

* **Use.** The label and the causal score go in `details["causal"]`. H3 uses the label as Section 4
  says (reported separately, and whether it adds predictive power beyond D4 selectivity).

#### C. D3 symbolic regression arm (proposal, Week 8)

* **What is fitted.** After every adaptation in A7 (pretrained and random weight model, 4 tasks,
  3 seeds), the adapted model's margin, logit("mine") minus logit("not mine"), on the 1,000 test
  positions.
* **Inputs.** The 64 squares of the true board at that position: +1 mine, -1 theirs, 0 empty (named
  `sq_A1` to `sq_H8` for PySR, since some bare square names are sympy functions).
* **Search.** PySR 2.4.0: operators +, - and x, no unary operators, at most 15 nodes, the 10 most
  relevant squares kept first (`select_k_features`), 40 iterations, 15 populations, deterministic,
  serial, `random_state` = the adaptation seed, best equation by PySR's `model_selection="best"`.
* **How a formula is read.** For every square j, on the 1,000 test boards: the **contrast** = mean
  |f(board with j mine) - f(board with j theirs)| (how much the formula tells mine from theirs on j),
  the **signed contrast** = the mean of f(j mine) - f(j theirs) (positive when mine raises the margin),
  and the **occupancy effect** = mean |(f(j mine) + f(j theirs)) / 2 - f(j empty)| (how much the
  formula reacts to j being occupied at all).
* **Law recovered** for a (model, task, seed): the task's own square has the largest contrast, its
  signed contrast is positive, and the formula's sign beats chance against the true rule
  (`law_agreement`, balanced accuracy, above 0.5). A formula such as a x sq_X x sq_X + b is 1 for mine
  and for theirs alike, so it has zero contrast and is never counted as recovered.
* **Occupancy shortcut** for a (model, task, seed): the formula reacts to the task's own square being
  occupied but has zero contrast on it. Occupancy can be read off the move list, which is why A7 does
  not use occupancy tasks; this flag shows when an adapted model answers "is X mine" through it.
* **Reported in `details["symbolic_regression"]`:** every formula with its complexity, top contrast
  square, squares used, own square contrast, signed contrast, contrast share and occupancy effect, the
  occupancy shortcut flag, R squared against the model's margin, and `law_agreement`; the recovery and
  occupancy shortcut rates of the pretrained and of the random weight models, overall and per task;
  stability per task = the mean Jaccard overlap of the squares used across the 3 seeds; mean R squared.
* **Reference fit.** Next to each formula, the R squared of a least squares fit of the same margin on
  128 indicator columns (mine and theirs for each square) plus an intercept. Indicators, not the
  signed squares, so that the reference can also express occupancy. It has 129 free parameters while a
  formula has at most 15 nodes over 10 squares, so the reference is an upper guide for an additive
  board formula, not a target the short formulas are expected to reach.
* **Not the score.** The D3 score stays the adaptation gap of A7. As the proposal's risk 4 says, "the
  probe's primary score is the adaptation based measure, with symbolic recovery reported as a
  qualitative, stability annotated companion result rather than a load bearing number".
* **Anchors (test split).** The D3 scores reproduce A7 to every digit (0.13969 and 0.05930).

  | Anchor | law recovered: pretrained, random | occupancy shortcut: pretrained, random | stability per task (C4, F5, C3, F6) | R squared: formula, 128 indicator reference |
  |---|---|---|---|---|
  | Othello GPT synthetic | 2/12, 1/12 | 7/12, 1/12 | 0.67, 0.33, 1.00, 0.33 | 0.139, 0.374 |
  | Othello GPT championship | 0/12, 0/12 | 6/12, 0/12 | 0.33, 1.00, 1.00, 0.33 | 0.345, 0.690 |

  On both reference models, the short formulas almost never recover the "is X mine" law (the 2
  synthetic recoveries are both on C4). What they find instead, in about half the searches and almost
  never for the random weight models, is the occupancy shortcut: once adapted to "is X mine", the
  pretrained models' answers largely follow whether X is occupied. The formulas explain less of the
  margin than the 128 indicator reference, as expected from their size, so these results describe the
  dominant short structure of the answers, not all of it.
* **What this means for D3 (stated in the paper).** Part of the pretrained advantage that the D3 score
  measures on these tasks may come from occupancy, which the move list gives away, rather than from
  whose disc is on the square. The D3 score is unchanged (A7); the occupancy shortcut rate is reported
  next to it for every zoo model, so the analysis can show how much of D3 it accounts for.

#### D. D2 sample size (proposal, Week 7: "explicit sample size justification")

Each D2 error is a mean over pairs of a number between 0 and 1, so its standard error is at most
0.5 / sqrt(n): at most 0.022 at the 500 pair minimum of A4 and at most 0.011 at 2,000 pairs. On the
reference Othello GPTs the largest standard error was 0.005 (championship compression error, 768
pairs) and every other was below 0.004, so a D2 score is known to about plus or minus 0.01, well
below the spread between models that H1 compares. The protocol of A4 is unchanged.

### Amendment 3 (prereg-v1.3), 2026-10-18

Written before the zoo is scored. At this point one zoo model, `othello_transformer_small_synthetic_s0`,
has been scored by D1 to D4 and by fragility v1, as the test run of the scoring launcher (#102). No
relation between any diagnostic and fragility has been computed, and none can be computed from one model.
The reason for this amendment comes from that model's fragility output and from arithmetic about the
suite itself, not from any diagnostic score. Frozen with the git tags `fragility-v2` and `prereg-v1.3`.
Sections 1 to 5 and Amendments 1 and 2 stay in force except where this amendment says otherwise.

#### E1. Two parts of fragility v1 do not measure fragility on this zoo

* **Detour floor.** v1 scores the share of self play games in which the model never proposes an illegal
  move in about 58 moves. A model that is legal with probability q per move completes about q^58 of them:
  0.84 gives 0.00004, and the best zoo model (0.969 on validation games) about 0.16. On the test model all
  six completion rates were 0.000 (one was 0.002), so the four detour drops were 0 and carried no
  information. This holds for almost every model in the zoo, so 4 of the 9 terms of `fragility` would be
  near 0 for all of them.
* **After pass compares easy positions with harder ones.** v1 compares the legal rate right after a pass
  with the legal rate over all positions. A pass means one player has no move at all, and on those boards
  most empty squares are legal for the player who keeps moving. On the test model's 1,000 test games
  (180 pass positions, mean move number 55): 0.934 of the empty squares were legal after a pass against
  0.660 at the same move numbers otherwise, and the model's legal rate rises with that share (0.78 to 0.80
  between 0.2 and 0.5, 0.94 above 0.7). Its legal rate was 0.972 after a pass against 0.842 overall, a
  drop of minus 0.130, which alone is minus 0.0144 of its fragility of minus 0.0149. Matching on the move
  number only (the first draft of v2) does not remove this: the baseline becomes 0.854, a drop of minus
  0.119. Matching on the move number and the number of legal moves does: the baseline becomes 0.963, a
  drop of minus 0.009. The v1 variant therefore measures how easy pass positions are, not whether the
  model handles "the same player moves again".

| Test model, fragility v1 | rate | drop |
|---|---|---|
| in distribution | 0.8423 | |
| cross distribution, championship | 0.8515 | minus 0.0092 |
| after pass | 0.9722 | minus 0.1299 |
| symmetry rot180, transpose, anti transpose | 0.8412, 0.8395, 0.8391 | 0.0010, 0.0028, 0.0032 |
| detour completion, random p = 0, 0.1, 0.5 | 0.000, 0.000, 0.000 | 0, 0 |
| detour completion, adversarial p = 0, 0.1, 0.5 | 0.000, 0.002, 0.000 | minus 0.002, 0 |
| **fragility** | | **minus 0.0149** |

#### E2. Fragility v2 (`src/fragility/othello_v2.py`, tag `fragility-v2`)

v1 is not changed. v2 imports it and replaces only the two parts above; the in distribution, cross
distribution and symmetry variants are v1's functions on v1's games.

* **Detour.** Every game is played to its end (or 59 moves). At each move the model's top move is scored
  legal or illegal, and the move played is the model's best legal move (its top move when that is legal),
  so an illegal proposal no longer ends the game. With probability p that move is replaced by another
  legal move, a uniformly random one or the one the model rates least likely, as in v1 (same p, modes,
  500 games, seed 0). Score: the legal rate over all predictions. Drop: the rate at p = 0 minus the rate
  at p. Because every setting plays full games, every p is scored on the same stages of the game.
* **After pass.** Each pass position is compared with the non pass positions at the same move number and
  with the same number of legal moves, in the same test games (the move number also fixes the number of
  empty squares, so the share of empty squares that are legal is matched too). Baseline = the mean, over
  pass positions, of the non pass legal rate in that cell. Drop: baseline minus the legal rate at the pass
  positions. Pass positions whose cell is empty are left out and counted.
* **Fragility v2** = the mean of the drops, as in v1. Output: `results/fragility_v2/{model_id}.json`.
* **Checks** (`src/tests/test_othello_fragility_v2.py`): a perfect model scores 0 on every drop; when the
  legal rate depends on the move number and the number of legal moves and pass positions are no worse
  than non pass positions in the same cell, the v2 after pass drop is near 0 while matching on the move
  number alone, or the v1 style comparison, is strongly negative; a
  model that is illegal only right after a pass gets a drop of 1; a broken model is still scored on full
  games; v1 and v2 give identical numbers on the unchanged variants; v1's hash is unchanged.

#### E3. Use in H3

* The H3 outcome of Amendment 1, A2, becomes **relative fragility v2** = fragility v2 divided by
  `in_distribution_legal_rate`. Every A2 rule (D1 as the reference, the contest among D2, D3 and D4, the
  partial correlations, the regression, the decision rule) is unchanged and is applied to this outcome.
* Fragility v1 and relative fragility v1 are run on every model and reported as the preregistered
  secondary outcome, labelled as such, with the reason of E1. The per variant drops of both versions are
  reported, so a reader can see what each version is made of.
* Both suites are run on every admitted model, including the test model above, with no other change.
* On the test model, fragility v2 = minus 0.0134 before the after pass matching was extended (first
  draft); its v2 numbers with this definition are those written by the scoring run.
