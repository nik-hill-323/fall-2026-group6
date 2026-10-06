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

None yet.
