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

#### E4. Analysis details, fixed before any H1 to H3 statistic is computed

The analysis code is `src/analysis/hypotheses.py` (tests: `src/tests/test_hypotheses.py`). It was written
and tested on fake zoos in the real file formats before any zoo model had all four scores. These choices
were left open by Section 4 and Amendment 1 and are fixed here:

* **Models.** H1 and H2 use admitted models whose four scores all have status ok; H3 also needs fragility
  v2. Every admitted model without a score is listed by diagnostic in the output.
* **Statistics.** Kendall tau b and Spearman rho (both handle ties). Intervals: 95% percentile bootstrap,
  2,000 resamples over models, seed 0; every statistic of one analysis uses the same resamples, so the
  difference of two statistics has its own interval. An interval is not reported if fewer than 1,000
  resamples give a finite value.
* **Partial Kendall tau (A1)** is the A1 formula. It does not fall all the way to 0 when two diagnostics
  share only competence (on simulated data with raw tau above 0.5 it stays near 0.2), so it is a
  conservative test of "agreement through competence"; the categories and thresholds are unchanged.
* **H2.** Rank percentile = average rank divided by the number of models; disagreement = their population
  standard deviation over the four diagnostics. Effect of a factor = the range of its level means of
  disagreement. The decision compares that effect with the noise floor **of the disagreement** (the mean,
  over configurations, of the standard deviation of disagreement across seeds), so both are in the same
  units; the Section 4 noise floor of each diagnostic's score is reported next to it. F test per factor:
  the least squares model without that factor against the full model; Holm over the 3 factors.
* **H3 (A2).** Partial Spearman of a diagnostic given D1 = Pearson correlation of the rank residuals after
  a least squares fit on the D1 ranks. Regression covariates are treatment coded; the interval of a
  coefficient is the 95% t interval. The Section 4 secondary rule takes the most negative Spearman as the
  strongest. The D4 causal flag (decodable and causal) is tested beyond D4 selectivity by a regression of
  the outcome on D4 and the flag and by the partial Spearman of the flag given D4.

### Amendment 4 (prereg-v1.4), 2026-10-19

Written after Instructor Review #5 (2026-10-19, Critiques 1, 2 and 4) and before any H1 to H3 statistic is
computed on zoo data. Frozen with the git tag `prereg-v1.4`. Sections 1 to 5 and Amendments 1 to 3 stay in
force except where this amendment says otherwise.

**What existed when this was written.** The zoo was scored with `src/shellscripts/score_slice.sh` on
2026-10-18 and 2026-10-19, before Review #5 arrived: the D1 to D4 and fragility v1 and v2 result files of
61 of the 81 models were on main, and the remaining 20 were being committed. Per model scores were printed
by the scoring launcher and checked for completeness and plausibility (status, fragility suite hashes,
values next to the reference Othello GPTs). No statistic of H1, A1, A3, H2 or H3 had been computed on zoo
data: no correlation between diagnostics or with fragility, no Kendall W, no regression and no interval.
`src/analysis/hypotheses.py` had been run only on simulated zoos in its tests. The change below follows from
the design of the zoo, not from any result.

#### F1. Intervals resample configurations, not models (Critique 1)

* The zoo is 27 configurations (architecture x scale x distribution) with 3 seeds each. Seeds of one
  configuration share everything but the random seed, so their scores are close; resampling the 81 models
  as independent units treats near copies as independent observations and makes intervals too narrow.
* **Every interval of Section 4, Amendment 1, Amendment 3 E4 and this amendment** is now a 95% percentile
  **cluster bootstrap**: each resample draws 27 configurations with replacement (as many as there are in
  the analysed set) and keeps every model (seed) of each drawn configuration; 2,000 resamples, seed 0. The
  statistic is computed on the pooled models of the resample, exactly as on the full set. This applies to
  the H1 taus, Spearman rhos, partial taus and both Kendall W, the A3 sensitivity run, every H3 correlation,
  partial correlation and difference of correlations, the D4 causal flag analysis and D3 against its
  occupancy shortcut. All statistics of one analysis still share the same resamples.
* **Regression coefficients** (H3 under A2 and Section 4, the D4 causal flag) get the same cluster
  bootstrap interval, which replaces the 95% t interval of E4 in every decision; the model based t interval
  is still reported next to it, labelled.
* **H2 is unchanged.** Its F tests compare configurations against the spread between seeds of the same
  configuration, which is the replication the design provides, and its decision already compares a
  factor's effect with the seed noise floor of the disagreement.
* Categories, thresholds and decision rules do not change; only the intervals they read do.
* Reference for the method: Cameron, Gelbach and Miller (2008) on bootstrap inference with clustered data;
  Bouthillier et al. (2021) on seed variance in machine learning benchmarks.

#### F2. H3 with and without the model that motivated Amendment 3 (Critique 2)

The primary H3 analysis (A2 on relative fragility v2) is reported twice: on all admitted models, and without
`othello_transformer_small_synthetic_s0`, the model whose fragility v1 output motivated Amendment 3 (its
fragility v1 file is `results/fragility/othello_transformer_small_synthetic_s0.json`). The decision of H3 is
taken on all models; if the two versions differ, the paper reports both and says so.

#### F3. The fragility distribution comes before H3 (Critique 4)

Before any H3 statistic, the analysis reports the distribution of fragility v1, fragility v2 and both
relative fragilities over the analysed models: count, mean, standard deviation, range, the 5th, 25th, 50th,
75th and 95th percentiles, the share below 0, the share within 0.01 of 0, and the mean per architecture,
with a histogram per architecture (`reports/figures/<domain>/fragility_distribution.png`). This shows how
much of H3 rests on a few models when most values sit near 0. It does not change the H3 rules.

#### F4. Code

`src/analysis/hypotheses.py` implements F1 to F3 (`clusters`, `boot_indices`, `coef_interval`,
`fragility_distribution`, `H3_without_test_model`). Tests in `src/tests/test_hypotheses.py` check that every
resample is made of whole configurations, that the configuration bootstrap gives a wider interval than the
model bootstrap when seeds are near copies, and that the new outputs are written.

### Amendment 5 (prereg-v1.5), 2026-11-10: the navigation domain

Written before any navigation zoo model is trained. At this point the navigation data are built and checked
(#141: map, splits, training caches, manifest), the three released models of Vafa et al. (2024) have passed the
reproduction gate (#145, `src/docs/navigation_reproduction.md`), and the numbers in G6 and G8 below were computed
from the data alone (`src/domains/maps_tasks.py`, commit 7ed6331, `results/prereg/maps_amendment5_numbers.json`).
No navigation zoo model exists, so no navigation diagnostic, fragility or H statistic has been seen. The Othello
results are complete and known; G10 says which navigation choice they motivated. Frozen with the git tag
`prereg-v1.5`. Sections 1 to 5 and Amendments 1 to 4 stay in force for navigation except where this amendment
says otherwise. Design and every reason behind the numbers: `src/docs/navigation_design.md`.

**Gates.** No navigation zoo model is trained before `prereg-v1.5` exists, and none is scored by a diagnostic or
by the fragility suite before `maps-fragility-v1` (G8) exists. The scoring code refuses navigation zoo models
until both tags exist, as Amendment 2 did for Othello. The pilots of G2 are not zoo models and may be trained
before either tag.

#### G1. Data (`src/domains/maps.py`, `src/domains/maps_build.py`)

* Routes of Vafa et al. (2024) on the Manhattan street map: `start destination turn ... end`. All three
  distributions use one **canonical map** (the map of the noisy shortest paths and random walks data): 4,579
  intersections (17 of them dead ends), 9,705 street segments, OpenStreetMap coordinates as of 2024-05-05 for every
  intersection, every turn label equal to the compass bearing. Shortest path routes are rewritten on it by their
  intersections; the 6.1% that use one of the 39 streets it does not have are dropped (limitation, stated in the paper).
* **Vocabulary** 4,589 tokens: padding, the intersections, 8 turns, `end`. **State** = (current intersection,
  destination). **Legal next set** = the turns at the current intersection, plus `end` at the destination.
* **One maximum length, 99 tokens, for all three distributions** (random walk routes of 100 to 102 tokens, 2.8%,
  are dropped), so every model is trained on every position any scorer reads.
* **Splits by start and destination pair.** The 107,331 pairs of the three held out files are split 50/50 (seed 0)
  into 53,665 validation and 53,666 test pairs, shared by all distributions; no route with one of them is in any
  training cache. Routes (val / test): shortest paths 13,991 / 13,890, noisy shortest paths 154,432 / 153,485,
  random walks 443,448 / 438,441. Training caches: 2,000,000 routes per distribution (seed 0).
* Every data file matches the sha256 in `src/domains/maps_data_manifest.json` (its own sha256:
  0e3109126789878adeae7ccd7ce1ca63fa485894c53376eb23ccc8bc221d7295); a test checks this on every machine.
* Split use is Section 1's: train for training only, val for checks during training and for fitting a diagnostic,
  test for final scores only.

#### G2. Models and training

* The 81 models of `src/docs/models.csv` with domain `maps`: transformer, LSTM, Mamba x small, medium, large
  (`SCALE_CONFIG`) x shortest paths, noisy shortest paths, random walks x seeds 0, 1, 2. Model id
  `maps_{arch}_{scale}_{distribution}_s{seed}`.
* Training exactly as Section 1 (5,000 steps of 256 routes, lr 3e-4, 200 warmup steps, weight decay 0.01), with
  batches drawn from the distribution's 2,000,000 route training cache as `src/zoo/train.py` draws Othello games
  (seeded by the model's seed). The loss covers only turns and `end` (the targets at token index 2 onward); the
  destination token is a random intersection and is not a training target. A model reads at most 98 tokens.
* **Pilots.** One small model per architecture and distribution (9 models, seed 100), under `outputs/pilot/`, never
  in any analysis. They are trained before any zoo model is trained (throughput, and a check that training and the
  admission number work) and scored by every diagnostic and the fragility suite before any zoo model is scored (a
  complete run of every scorer, and the fragility rule in G8).

#### G3. Admission (Section 2, for navigation)

A navigation model is admitted if its legal next token rate on 1,000 validation routes is at least **0.50**, as
for Othello. The validation routes are the same for every model (data seed 0), written by the trainer in the
training record. An untrained model scores about 0.0005 (about 2 legal tokens of 4,589); the released
models of Vafa et al. score 0.9996 to 1.0000 (G11).

#### G4. D1 (`score`)

Share of scored positions (every prediction of a turn or of `end`) where the top token is legal, on 1,000 test
routes of the model's own distribution, data seed 0. The same computation (`maps.legal_rate`) gives Vafa et al.'s
next token rate exactly on their models (G11). Secondary, as for Othello (`src/docs/d1_next_token_validity.md`): at
each scored position, 10 tokens sampled at temperature 1.0 (sampling seed 0), the share that are legal.

#### G5. D2 (Amendment 1, A4, with the navigation state)

A4 unchanged except: histories are route prefixes up to every scored position of 1,000 test routes; the true
state is (current intersection, destination); same state pairs are different prefixes that reach the same state;
different state pairs have the same number of turns and different legal next sets; the accepted set has k = the
size of the legal next set. Pair counts, the 500 pair minimum and the 2,000 pair cap, seeds and the score are A4's.
Limitation as in A4: next token only (two intersections with the same turns are different states D2 cannot tell
apart).

#### G6. D3 (Amendment 1, A7, and Amendment 2, C, with navigation tasks)

* **Tasks**, about the state at one uniformly random scored position (`maps_tasks.random_positions`, seed 0),
  labels from `src/domains/maps_tasks.py`:
  * **T1** the destination is at most **10** turns away on the canonical map;
  * **T2** the destination is at most **823.9 m** away in a straight line;
  * **T4** the current intersection is east of the destination (higher longitude).
  10 and 823.9 are the medians over 30,000 positions (10,000 validation routes per distribution, seed 0).
* **Drop rule, written before the numbers were computed:** a task is dropped if, in any distribution, its positive
  rate is outside 0.10 to 0.90, or its shortcut rule (a threshold on the number of turns so far for T1 and T2; the
  start in place of the current intersection for T3 and T4) reaches a balanced accuracy of 0.90. Results:

  | Task | positive rate (sp, noisy, rw) | shortcut balanced accuracy (sp, noisy, rw) | |
  |---|---|---|---|
  | T1 | 0.428, 0.297, 0.815 | 0.536, 0.607, 0.551 | kept |
  | T2 | 0.421, 0.300, 0.779 | 0.544, 0.616, 0.544 | kept |
  | T3 north of the destination | 0.483, 0.426, 0.471 | **0.922**, **0.908**, 0.747 | **dropped** |
  | T4 | 0.463, 0.409, 0.481 | 0.877, 0.878, 0.735 | kept |

  T3 is dropped because shortest and noisy shortest path routes head almost straight for the destination, so
  where the start lies says where the car lies. T4's shortcut stays high (0.88), so for every adapted model the
  **shortcut agreement** is reported next to its score, as for Othello's occupancy shortcut: the balanced accuracy of
  the adapted model's answers against the shortcut rule's answers on the same test positions (for T1 and T2 the
  rule is the threshold on the number of turns so far recorded per distribution in the numbers file).
* **Protocol** exactly A7 (64 validation routes of the model's own distribution, data seed 0, one random scored
  position per route; a copy of the model plus a new linear head on the last layer vector at that position; head lr
  1e-3, body lr 1e-4, AdamW, weight decay 0.01, 300 steps of 16; random weight baseline of the same architecture and
  scale; seeds 0, 1, 2; 1,000 test routes, one random scored position each; balanced accuracy). `score` = mean over
  the 3 tasks and 3 seeds of pretrained minus random balanced accuracy.
* **Symbolic regression arm** (C, descriptive): the adapted model's margin on the 1,000 test positions is fitted
  from 7 inputs, all used (no feature selection): the latitude and longitude of the current, start and destination
  intersections, and the number of turns so far, each standardized over those positions. PySR settings are C's.
  For an input j, the **contrast** is the mean over the test positions of |f(x with x_j at its 90th percentile)
  minus f(x with x_j at its 10th percentile)|, and the **signed contrast** is the mean of that difference without
  the absolute value. For T4 the law is **recovered** if the current longitude has the largest contrast and a
  positive signed contrast, the destination longitude has a negative signed contrast, and the formula's sign beats
  chance against the true rule (balanced accuracy above 0.5). The **start shortcut** is flagged when the start
  longitude has a larger contrast than the current longitude. For T1 and T2 the formulas, their contrasts and R
  squared are reported without a recovery rule.

#### G7. D4 (Amendment 1, A6, and Amendment 2, B1 and B2, with the current intersection)

* **Probe** per layer: a linear map from the standardized activation to the current intersection (4,579 classes)
  at every scored position. Data: the first 10,000 of 11,000 validation routes (seed 0) fit, the last 1,000 choose
  the layer, 1,000 test routes give the score. Optimizer, standardization, step size and seed are A6's.
* **Epochs, by a rule fixed now:** a synthetic layer of width 128 (the smallest zoo width) holds, at every scored
  position, a fixed random Gaussian vector for the true current intersection (seed 0; no model involved). For each
  distribution, the probe is fitted on its D4 fitting routes and scored on its D4 test routes with everything else
  as above. The epoch count is the smallest of 2, 5, 10 and 20 at which all three distributions reach an accuracy of
  at least 0.99 (20 if none does, with the accuracies reported). Othello's 2 epochs read a board of 64 x 3 classes; a
  4,579 way probe may need more, and an under trained probe would compress the differences between models. The
  count found is written in the D4 documentation before any navigation model is scored and applies to every
  navigation model and anchor.
* **Control task:** the type of a position is (token just read, number of turns so far); each type gets a fixed
  random intersection drawn from the intersection frequencies of the fitting routes (seed 0). Selectivity and
  `score` as A6. MLP probe as B1.
* **Details** add two baselines: the accuracy of answering the start, and of answering the destination.
* **Causal check** as B2 (edited layers middle to last, step 0.05 x sqrt(width), confidence 0.99, at most 100
  steps, random edit of the same length, 2,000 bootstrap resamples, the same pass rule and labels): one case per
  test route, at the first scored position (random order, seed 0) with a neighbouring intersection, one legal turn
  away, whose legal next set differs from the current one and is not empty (the first such neighbour in turn order);
  the edit moves the probe's reading to that neighbour; boundary accuracy uses k = the size of the neighbour's legal
  next set. The B2 settings were chosen on the Othello GPTs and are not retuned; their behaviour on the three
  anchors is reported.

#### G8. Fragility suite `maps-fragility-v1` (`src/fragility/maps.py`, frozen with that tag before any navigation zoo model is scored)

Built on fragility v2 (Amendment 3, E2): every comparison matched, no score that sits at a floor, the route played
always stays legal.

| Variant | Score and drop |
|---|---|
| in distribution | legal rate on 1,000 test routes of the model's own distribution (the reference) |
| cross distribution | legal rate on 1,000 test routes of each other distribution (unseen pairs); drop = in distribution minus it (2 drops) |
| detour, random and adversarial | the model drives from the start to the destination of 500 test routes of its own distribution (seed 0); its top token is scored at every step; the move played is its highest rated **safe** move (legal, and the destination still reachable within 96 turns in all); with probability p (detour draws seed 0; never at the destination) it is replaced by another safe turn, a uniformly random one or the one the model rates lowest, and kept when there is no other safe turn; the drive ends at `end` or at 96 turns; drop = legal rate at p = 0 minus legal rate at p, p = 0.1 and 0.5 (4 drops) |
| least visited intersections | positions at the intersections of the model's distribution in `src/domains/maps_rare_intersections.json`, on 5,000 test routes (seed 0); drop = the mean, over those positions, of the legal rate of the other positions in the same cell (number of legal next tokens, at the destination or not, block of 10 turns so far) minus their own legal rate; positions with an empty cell are left out and counted (1 drop) |

* **Least visited intersections, defined from the data:** per distribution, the intersections with the fewest
  visits in its training cache, in order of visits, until they hold 10% of the scored positions of its test split.
  Sets of 2,749 (shortest paths), 2,816 (noisy) and 1,130 (random walks) intersections; on 5,000 test routes they
  give 18,418, 26,891 and 24,785 positions, of which over 99.9% have a matched cell. A first definition, the 10% of
  intersections with the fewest visits, was replaced before any model existed: those intersections are almost never
  on a route (6 positions in 5,000 shortest path test routes, 38 for noisy shortest paths).
* `fragility` = the mean of the 7 drops. **Relative fragility** = `fragility` divided by the in distribution legal
  rate. "Reached the destination with `end`" is saved for every drive as a descriptive number. There is no symmetry
  variant (no symmetry maps the street map onto itself).
* **Pilot rule, fixed now:** a variant whose drop has a standard deviation below 0.001 across the 9 pilots is
  reported for every model but left out of `fragility`. Order: the suite's code is run on the pilots, the rule is
  applied, its outcome is written into the suite's code and documentation, and only then is the tag
  `maps-fragility-v1` made. No zoo model is scored by the suite before that.

#### G9. Whole route measure (preregistered secondary)

* Teacher forced: on D1's 1,000 test routes, the share of routes whose every top token is legal, and the legal rate
  by step (the overall rate equals D1, a check).
* Free play: from 1,000 test start and destination pairs (seed 0) the model drives itself; its top token is scored at
  every step; the move played is drawn from its own probabilities over the legal next set (seed 0); the drive ends at
  `end`, at 96 turns, or at a dead end that is not the destination. Reported: share of drives with no illegal top
  token, share ending at the destination, legal rate by step, step of the first error, share stuck at a dead end.
* Secondary analysis: the Spearman correlation of each diagnostic with the share of drives with no illegal top token,
  with the configuration bootstrap of F1. Reported, not used in any H decision.

#### G10. Analysis

* H1 to H3 run on navigation exactly as on Othello (Section 4, Amendment 1, Amendment 3 E4, Amendment 4 F1 and F3), per
  domain. The H3 outcome is relative fragility from `maps-fragility-v1`. Amendment 4 F2 (the Othello test model) does
  not apply.
* **Training distribution as a control (preregistered secondary for navigation).** On Othello, the only link from D4
  to fragility came from models trained on championship games (an exploratory finding, #131). For navigation, next to
  the unchanged A2 analysis: (a) the A2 regression with training distribution added as a categorical covariate, with
  the configuration bootstrap interval of each diagnostic's coefficient; (b) the A2 partial Spearman (given D1)
  computed within each training distribution, each with the configuration bootstrap over its 9 configurations. The H3
  decision stays the A2 rule; (a) and (b) say whether a link survives within distributions. The same two analyses are
  reported for Othello, labelled exploratory, since its result was seen first.
* If the navigation analysis runs on fewer than three architectures (G12), a factor that does not vary
  (architecture) is left out of H2 and of every regression; every other rule is unchanged.
* H4 is unchanged: H1 to H3 on Othello and navigation are written down before any interpreter model is scored.

#### G11. Anchors

The three released models of Vafa et al. (2024) (`interface.vafa_navigation`, ids `maps_vafa_shortest_paths`,
`maps_vafa_noisy_shortest_paths`, `maps_vafa_random_walks`) are scored by D1 to D4, the fragility suite and the whole
route measure, as description only, never in an H statistic. D3 adapts the whole model with AdamW (A7); for the two
1.5B anchors the weights, gradients and optimizer state alone need about 24 GB, more than the A10G's 23 GB, so their D3
is reported as `skipped` with that reason (Section 5); the shortest paths anchor (89M) runs D3 as A7, with a freshly
initialised GPT 2 of its configuration as the random weight baseline. The reproduction
gate (#145): next token 0.9996, 0.9998,
1.0000 (published 1.00), our D1 path equal to their computation on the same routes, probe 0.9089, 0.9142, 0.9915
(published 0.91, 0.92, 0.99).

#### G12. Order and cutoff (Instructor Review #8)

* **Order.** Navigation models are trained and scored by architecture: all 27 transformers (3 scales x 3 distributions
  x 3 seeds), then the 27 LSTMs, then the 27 Mamba models. A model may be scored as soon as it is trained; nothing in
  the protocol changes during the zoo run.
* **Scored** means: every diagnostic, the fragility suite and the whole route measure have a result file for the
  model (status ok, or `failed` or `skipped` with a reason, Section 5).
* **Primary analysis:** all 81 models.
* **Cutoff, 2026-11-22.** If not every navigation model is scored by the end of 2026-11-22 (US Eastern time), the navigation analysis is
  run on the architectures whose 27 models are all scored by then, and the paper says which. If no architecture is
  complete, the paper reports navigation as the domain, the reproduction and this preregistration, without H1 to H3
  results for it.

#### G13. Files

`src/domains/maps.py`, `maps_build.py`, `maps_tasks.py`, `maps_edges.csv`, `maps_coordinates.csv`,
`maps_rare_intersections.json`, `maps_data_manifest.json`; `results/prereg/maps_amendment5_numbers.json`;
`src/docs/navigation_design.md`, `src/docs/navigation_reproduction.md`. After merge:
`git tag prereg-v1.5 && git push origin prereg-v1.5`.
