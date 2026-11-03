# Code guide for src/

This guide explains how the code works, in plain language, for everyone on the team. Read Sections 1 and 2 first:
they give the whole picture. Use the other sections to look things up.

| Section | What it answers |
|---|---|
| [1](#1-the-path-of-one-model-from-start-to-finish) | What happens to one model, from data to report figure? |
| [2](#2-how-the-next-move-is-predicted) | How does a model predict the next move? |
| [3](#3-the-three-models-line-by-line) | How do the transformer, LSTM and Mamba work, line by line? |
| [4](#4-the-common-interface-zoomodel) | How does one diagnostic work on all three models? |
| [5](#5-every-file-in-src) | What does each file in `src/` do? |
| [6](#6-how-to-run-each-step) | What command runs each step? |
| [7](#7-things-that-surprise-people) | What is easy to get wrong? |

Words used throughout:

* **Zoo:** the 81 Othello models we trained. 3 architectures x 3 scales x 3 training distributions x 3 seeds.
* **Anchors:** the two published Othello GPT models (`othello_gpt_synthetic`, `othello_gpt_championship`). Every
  diagnostic scores them as a reference, but they never enter the H1 to H3 statistics.
* **Model id:** `{domain}_{arch}_{scale}_{distribution}_s{seed}`, for example `othello_lstm_small_synthetic_s0`.
* **Competence:** the legal move rate on validation games, measured at the end of training.
* **Square and token:** a square is a board position, 0 to 63. A token is the number the model reads, 0 to 60.
  They are different numbers. Section 2 explains the mapping.


## 1. The path of one model from start to finish

We follow `othello_lstm_small_synthetic_s0`. Every other model takes the same path.

```text
 games on disk            src/domains/othello.py
      |
      v
 training                 src/zoo/train.py, started by src/shellscripts/train_slice.sh
      |
      +--> checkpoint     outputs/zoo/<model_id>/model.pt   (not in git; copied to S3)
      +--> record         results/zoo_training/<model_id>.json   (in git)
      |
      v
 registry                 src/zoo/registry.py -> src/docs/zoo_registry.csv, zoo_admission.csv
      |
      v
 scoring                  src/shellscripts/score_slice.sh runs D1, D2, D4, fragility v1, fragility v2, D3
      |
      v
 one JSON per job         results/D1_next_token/, D2_state_equiv/, D3_inductive_bias/, D4_probing/,
      |                   fragility/, fragility_v2/
      v
 analysis                 src/analysis/hypotheses.py, exploratory_othello.py, sequence_othello.py
      |
      v
 tables and figures       results/analysis/othello/, reports/figures/othello/, reports/Latex_report/fig/
```

### Step 1: data (`src/domains/othello.py`)

This file is the only copy of the Othello rules and the only place that loads games. Everything else imports it.

A game on disk is 60 numbers: the squares played, in order, with -1 as padding for games shorter than 60 moves.
`load_distribution` returns games from one training distribution and one split:

| Distribution | Where the games come from | How the split is made |
|---|---|---|
| `synthetic` | Random legal games from `othello_world`, in `.bin` files | `train/` folder; `val/` folder, whose last 10 files (`SYNTHETIC_TEST_FILES`) are the test split |
| `championship` | Human games from WTHOR, in `.wtb` files | `load_championship` removes duplicates and illegal games, then makes a fixed 80 / 10 / 10 split (`SPLIT_SEED`) |
| `mixed` | Half of each (`MIXED_SYNTHETIC_SHARE`) | The same ratio in every split |

What each split may be used for:

* **train:** model training only.
* **val:** competence, and fitting the diagnostics (D3 adaptation, D4 probes).
* **test:** final scores only.

Two functions turn games into arrays:

* `to_tokens` is the fast path for training. It only converts squares to tokens and never replays the game.
* `build` replays every game with the rules. It returns the tokens, the mask of real moves, the board after every
  move (`black_white` and `mine_theirs`), who moved (`movers`) and, with `legal=True`, the legal next moves
  (`legal_next`). Diagnostics use this one.

### Step 2: training (`src/zoo/train.py`, `src/shellscripts/train_slice.sh`)

`train_slice.sh` reads `src/docs/zoo_slices.csv`, which says who trains which model, and calls `train.py` once per
model. It skips any model that already has a training record, so after a crash you just run the same command again.
(Our example model is in slice 0, the four first models, which `train_slice.sh` does not cover. They went through the
same `train.py`.)

`main` in `train.py` does this:

1. Loads games with `rules.load_distribution` (`rules` is `othello.py`).
2. Builds the model with `build_model(arch, scale)`. Sizes come from `SCALE_CONFIG` in `src/component/run_matrix.py`.
3. Trains it with `train`.
4. Checks it with `evaluate` on 1,000 validation games, which gives next token accuracy and the legal move rate.
5. Writes `model.pt`, `config.json`, `results.json` and `row.md` to `outputs/zoo/<model_id>/`.
6. Writes the training record, but only for a full budget run on real data.

The budget is the same for every model (`TRAINING_BUDGET` in `run_matrix.py`):

| Setting | Value |
|---|---|
| Optimizer steps | 5,000 |
| Games per step | 256, so 1,280,000 games are drawn in all |
| Optimizer | AdamW, learning rate 3e-4, weight decay 0.01 |
| Learning rate schedule | Rises over the first 200 steps, then follows a cosine curve down to 0 (`lr_at`) |
| Gradient clipping | 1.0 |
| Big models | `--micro_batch` splits each batch into chunks to fit GPU memory. It is still one optimizer step on the full batch. Mamba large uses 64 |

The training loop in `train`, line by line:

```python
idx = rng.integers(0, G, size=cfg["batch"])
```

Draw 256 random games. The draw is with replacement, so a game can come up more than once.

```python
logits = model(xc[:, :-1])
loss = loss_fn(logits.reshape(-1, VOCAB), xc[:, 1:].reshape(-1)) / n_chunks
```

Input is every token except the last. Target is every token except the first, so position `t` learns to predict the
token at position `t+1`. This is teacher forcing: the model always reads the real history, never its own guesses.

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad targets are 0
```

Token 0 is padding, so positions after the end of a game do not count in the loss.

### Step 3: record and registry (`results/zoo_training/`, `src/zoo/registry.py`)

The training record is a small JSON file in git. It holds the config, the final loss, the validation legal rate, the
git commit, the machine and `checkpoint_sha256`, the SHA 256 hash of the checkpoint. The hash proves later that a
score belongs to the model that was trained.

`registry.py` builds two tables from all the records:

* `src/docs/zoo_registry.csv`: one row per trained model (`build_registry`, `write_registry`).
* `src/docs/zoo_admission.csv`: per architecture and scale, how many models were planned, trained and admitted
  (`admission_table`, `write_admission`).

A model is admitted if its validation legal rate is at least 0.50 (`ADMISSION_MIN_LEGAL_RATE`). `load_model` loads a
model as a `ZooModel` and checks its hash first.

### Step 4: scoring (`src/shellscripts/score_slice.sh`)

For each model in the slice, the script checks the checkpoint hash against the record, then runs these jobs in order:

| Job | Script | Writes |
|---|---|---|
| D1 | `src/diagnostics/d1/next_token.py` | `results/D1_next_token/<id>.json` |
| D2 | `src/diagnostics/d2/state_equiv.py` | `results/D2_state_equiv/<id>.json` |
| D4 | `src/diagnostics/d4/representation_probe.py` | `results/D4_probing/<id>.json` |
| Fragility v1 | `src/fragility/othello.py` | `results/fragility/<id>.json` |
| Fragility v2 | `src/fragility/othello_v2.py` | `results/fragility_v2/<id>.json` |
| D3 | `src/diagnostics/d3/inductive_bias.py` | `results/D3_inductive_bias/<id>.json` |

D3 runs last because PySR makes it the slowest. A job that already has a finished result is skipped, and a failed job
is logged while the slice goes on.

What each one asks, using 1,000 test games of the model's own training distribution:

* **D1, next token validity:** how often is the model's top move legal?
* **D2, state equivalence:** does the model treat two histories that reach the same board alike, and separate
  histories whose legal moves differ? Score = 1 minus the mean of the two errors.
* **D3, inductive bias probe:** given only 64 labelled positions, how much better does the trained model learn "is
  this square mine?" than a random model of the same size? Score = trained minus random balanced accuracy.
* **D4, representation probe:** can a linear probe read the board from inside the model? Score = probe accuracy minus
  the accuracy of a control probe trained on meaningless labels, at the best layer. A causal check also edits the
  model's internal vector and tests whether its moves change to match the edit.
* **Fragility:** how far does the legal rate fall on shifted tasks (other training distributions, board symmetries,
  positions after a pass, forced detours)? Version 2 is the H3 outcome. Version 1 is still run and reported.

### Step 5: results JSON (`src/component/results.py`)

D1 to D4 all write the same format, the `Result` class. The most important field is `score`, and **higher always means
a more world model like model**. That is why D2 stores 1 minus its error. Without this rule, rank correlations between
diagnostics would have the wrong sign. Raw numbers go in `details`, and `provenance` records the commit and machine.

The fragility suites write their own simpler JSON, with the fields `fragility`, `drops` and
`in_distribution_legal_rate`.

### Step 6: analysis (`src/analysis/`)

* **`hypotheses.py`** is the preregistered analysis of H1, H2 and H3. It reads `src/docs/models.csv`,
  `src/docs/zoo_registry.csv` and the result JSONs, and writes `results/analysis/othello/summary.json`,
  `model_table.csv`, `h1_pairs.csv` and `h3.csv`.
* **`exploratory_othello.py`** holds checks decided after the preregistered results were read: descriptives, H3 within
  each training distribution, and H3 sensitivity. It writes `exploratory.json` and `descriptives.csv`.
* **`sequence_othello.py`** holds the exploratory whole game measures. It runs the models again, so it needs the
  checkpoints and a GPU, and it writes `results/exploratory/sequence/<id>.json`. With `--summarize` it builds
  `sequence.csv` and `sequence_summary.json`.

### Step 7: report figures

| Figure | Written by |
|---|---|
| `reports/figures/othello/h1_concordance.svg`, `a3_competence.svg`, `h3_scatter.svg`, `fragility_distribution.svg`, `h2_disagreement.svg` | `_figures` in `hypotheses.py` |
| `reports/figures/othello/sequence_by_move.svg` and `reports/Latex_report/fig/othello_sequence.pdf` | `_figure` in `sequence_othello.py` |
| `reports/Latex_report/fig/zoo_pipeline.pdf` | Not a script; the source diagram is `reports/diagrams/zoo_training_pipeline.drawio` |
| `reports/Latex_report/fig/othello_h1.pdf`, `othello_shifts.pdf`, `othello_h3_d4.pdf` | `reports/Latex_report/make_figures.py` |


## 2. How the next move is predicted

**Tokens.** The 4 centre squares (27, 28, 35, 36) already hold discs at the start, so only 60 squares can ever be
played (`PLAYABLE`). They get tokens 1 to 60 in square order (`SQ_TO_TOKEN`, `TOKEN_TO_SQ`), and token 0 is padding, so
`VOCAB` is 61. This is the same vocabulary as Othello GPT. Because the centre squares are skipped, a token is not the
same number as its square. Always convert with the two maps.

**Input and output.** A model takes tokens of shape `(B, T)` and returns scores of shape `(B, T, 61)`. The scores at
position `t` are its guess for the move after position `t`. The prediction is the token with the highest score.

**No window.** A game has at most 60 moves and the models read at most 59 tokens (`N_CTX`), so the history always fits.
It only grows, and nothing is ever dropped. There is no sliding window.

**Passes are not tokens.** After a pass, the same player simply moves again, and the move list shows nothing. The model
has to work this out from the board it has tracked. `legal_next` handles this in the rules: the other player moves
next, unless they have no legal move, in which case the same player moves again.

**The three ways a model is used:**

| Where | What the model reads | What is checked |
|---|---|---|
| Training (`train`) | The real history (teacher forcing), at most 58 tokens | Cross entropy against the real next move |
| D1 (`legal_rate`, `next_token.py`) | The real history (teacher forcing), all 59 token positions | Is the top move legal? The last recorded move of each game has no next move, so it is skipped |
| Free play (`free_play`, `sequence_othello.py`) | Its own game so far, at most 59 tokens | Is the top move legal? |

**Teacher forcing** means one wrong guess never enters what the model reads next. This is the unit the preregistration
fixes. It also means a model can look good at each single step yet still lose track of the board over a long game.

**Free play** tests that case. Each of the 1,000 games goes like this:

1. The first move is a random legal move, so the model is never asked to open.
2. At each later move, the model's top move is checked against the rules, and an illegal one counts as an error.
3. The move actually played is drawn from the model's own probabilities over the legal moves only. So the game always
   continues to its natural end, and the 1,000 games differ from one another.

`teacher_forced` in the same file is D1 split by move number. Its overall rate must equal the D1 score, and the script
checks this in `matches_d1`.

**Why training stops one token short:** `to_tokens` keeps the first 59 moves, then `train` uses `xc[:, :-1]`, so a model
reads at most 58 tokens and predicts moves 2 to 59. **The zoo models are never trained to predict move 60.** See
[Section 7](#7-things-that-surprise-people) for what this affects.


## 3. The three models, line by line

All three classes are in `src/zoo/train.py`, and all three have the same three methods:

* `layer_outputs(x)`: a list with the vector after each layer, each of shape `(B, T, d_model)`.
* `readout(h)`: turns the last layer's vector into scores.
* `forward(x)`: `readout(layer_outputs(x)[-1])`. So the vectors D4 probes are exactly the ones the model uses to
  predict.

Sizes come from `SCALE_CONFIG`, and the LSTM and Mamba use the same layers and width as the transformer:

| Scale | Layers | Width (`d_model`) | Transformer parameters | LSTM parameters | Mamba parameters |
|---|---|---|---|---|---|
| small | 4 | 128 | 0.82M | 0.54M | 0.48M |
| medium | 6 | 256 | 4.79M | 3.19M | 2.66M |
| large | 8 | 512 | 25.31M | 16.87M | 13.63M |

### GPT (the transformer)

```python
self.tok = nn.Embedding(VOCAB, d_model)
self.pos = nn.Embedding(N_CTX, d_model)
```

One learned vector per token, and one per position. A transformer has no built in sense of order, so it needs the
position vectors.

```python
nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                           batch_first=True, norm_first=True, activation="gelu")
```

One block: attention, then a feed forward network 4 times the width. There is no dropout. `norm_first=True` means
LayerNorm runs before each part, which matches Othello GPT's `LNPre`.

```python
mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
```

The **causal mask** stops position `t` from attending to later positions. Without it, the model could see the answer.
Each input vector is its token vector plus its position vector.

```python
h = layer(h, src_mask=mask, is_causal=True)
out.append(h)
```

Run each block and keep its output. This list is `layer_outputs`, the residual stream after each block.

```python
return self.head(self.norm(h))
```

`readout`: a final LayerNorm, then a linear layer to the 61 scores.

### LSTM (a stack of single layer LSTMs)

```python
self.layers = nn.ModuleList([nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
                             for _ in range(n_layers)])
```

**Why single layer LSTMs in a list?** A stack of them has the same parameters and does the same computation as one
`nn.LSTM(num_layers=n_layers)`. But a multi layer `nn.LSTM` only returns its last layer's output, while D4 has to probe
every layer. The list makes each layer's hidden state reachable.

```python
h, _ = layer(h)
out.append(h)
```

An LSTM returns `(output, (h, c))`. We keep the output, feed it to the next layer, and record it.

```python
return self.head(h)
```

`readout` is only a linear layer, with no LayerNorm.

The LSTM has no position embedding, because it reads moves one at a time and carries a memory forward, so order is
built in. It is one directional (`bidirectional` keeps its default, False), so position `t` never sees later moves. It
takes `n_heads` only to match the same call as the others, and ignores it through `**_`.

### Mamba (`mambapy`)

```python
from mambapy.mamba import Mamba as MambaStack, MambaConfig
```

**Why `mambapy` and not `mamba_ssm`?** `mamba_ssm` does not build on our EC2: the Ubuntu 26.04 math headers clash with
CUDA 13.1 (an `rsqrt` declaration). `mambapy` is Mamba written in plain PyTorch, so it has nothing to compile. It is
pinned in `requirements.txt` as `mambapy==1.2.0`.

```python
cfg = MambaConfig(d_model=d_model, n_layers=n_layers, d_state=16, d_conv=4, expand_factor=2)
self.backbone = MambaStack(cfg)  # backbone.layers: one residual block per layer
```

`d_state`, `d_conv` and `expand_factor` are Mamba's own settings: the size of its state, the size of its convolution
and how much it widens inside a block.

```python
for layer in self.backbone.layers:
    h = layer(h)
    out.append(h)
```

We loop over the blocks ourselves instead of calling the backbone once, the same way `mambapy`'s own forward does, so we
can record the residual stream after each block.

```python
return self.head(self.norm(h))
```

`readout`: LayerNorm, then the linear layer, like the transformer. Mamba also has no position embedding, because it
reads the sequence in order, like the LSTM.


## 4. The common interface: `ZooModel`

**The problem it solves:** a transformer, an LSTM, a Mamba and the TransformerLens anchors are all built differently,
but a diagnostic should not have to care. `ZooModel` in `src/diagnostics/interface.py` puts them behind the same calls.

| Call | Input | Output | Used by |
|---|---|---|---|
| `logits(tokens)` | `(batch, positions)` | `(batch, positions, 61)` scores for the next move | D1, D2 |
| `activations(tokens)` | `(batch, positions)` | `(layers, batch, positions, width)`, the vector after each layer | D4 |
| `logits_edited(tokens, positions, edits)` | Also, per game, one position, and `{layer: fn}` | Scores after `fn` changed the vector at those layers and positions | D4 causal check |
| `logits_patched(tokens, layer, positions, vectors)` | Also one layer and fixed vectors | The same, with the vector simply replaced | Tests; no diagnostic calls it |

`logits` and `activations` run without gradients, so a diagnostic cannot accidentally train a model. Calling the object
directly, as `model(tokens)`, is the same as `model.logits(tokens)`, which is why a `ZooModel` works in `legal_rate`.

**How `logits_edited` works.** It applies the edits in layer order, so every later layer sees the changed vector, and
the model then runs on from there. This is what lets D4 ask "if we change what the probe reads, does the model's
behaviour change?"

* `TrainPyModel` does it with PyTorch forward hooks on the layer modules (`_layer_modules`).
* `TransformerLensModel` does it with TransformerLens hooks at `hook_resid_post`.
* Both use `_edit_at` to change the one position per game.

**Loading models:**

* `othello_gpt(which)` loads an anchor as a `TransformerLensModel`.
* `from_checkpoint(model_id)` loads a zoo model as a `TrainPyModel`. It takes the architecture and scale from
  `src/docs/models.csv`. It does not check the hash.
* `load_model(model_id)` in `registry.py` checks the record and the hash, then calls `from_checkpoint`.

**The contract for a diagnostic.** Every diagnostic module defines two things:

```python
DIAGNOSTIC = "D1"
def score(model: ZooModel, split: str, n_games: int, data_seed: int, **config) -> Result
```

It builds its `Result` with `result_for`, which fills in the model fields. `check_diagnostic` enforces the contract.
`run` scores a list of models with one diagnostic and saves each result. If a model crashes, `run` saves a failed result
with the error as the reason, and the sweep carries on.

**Two parts that do not go through `ZooModel`:**

* **D3** has to train the model, and `activations` runs without gradients. So `last_layer` calls
  `module.layer_outputs(tokens)` directly, on a deep copy, so the original weights never change.
* **The fragility suites** are frozen files that take a plain model. Their `main` uses `build_model` and
  `load_state_dict` directly.


## 5. Every file in src/

### `src/domains/`

**`othello.py`** holds the rules, the data and the true state, as Section 1 describes. Rules: `start_board`, `flips_for`,
`legal_moves`, `legal_next`, `replay`, `play_random_game`, `random_games`. Data: `list_files`, `load_games`,
`load_split`, `read_wtb`, `load_championship`, `load_distribution`. Arrays: `to_tokens`, `build`. Score: `legal_rate`.
It reads the game files in `~/artifacts`, and writes only `championship_split.npz`, a cache of the championship split.

### `src/zoo/`

**`train.py`** holds the three model classes and the trainer: `GPT`, `LSTM`, `Mamba`, `build_model`, `lr_at`, `train`,
`evaluate`, `main`. It reads games and `run_matrix.py`. It writes `outputs/zoo/<model_id>/` and, for full budget runs,
`results/zoo_training/<model_id>.json`.

**`registry.py`** tracks which models are trained, how well, and where their checkpoints are. Main functions:
`load_records`, `build_registry`, `write_registry`, `admission_table`, `write_admission`, `load_model`,
`import_outputs`. It reads `results/zoo_training/` and writes `src/docs/zoo_registry.csv` and `zoo_admission.csv`.

### `src/component/`

**`run_matrix.py`** defines the experiment: `SCALE_CONFIG`, `TRAINING_BUDGET`, `DISTRIBUTIONS`, and the model and run
lists (`model_rows`, `eval_rows`). It writes `src/docs/models.csv` (243 models across all domains) and
`src/docs/run_matrix.csv` (972 runs).

**`results.py`** defines the result format and the master table: `Result`, `make_result`, `validate`, `save`, `load`,
`load_results`, `load_models`, `provenance`, `master_table`, `missing_report`, `write_master_table`. It reads and
writes `results/D1_next_token/` to `results/D4_probing/`, and writes `results/master_table.csv`.

### `src/diagnostics/`

**`interface.py`** is the common interface in Section 4. Main parts: `ZooModel`, `TrainPyModel`,
`TransformerLensModel`, `othello_gpt`, `from_checkpoint`, `result_for`, `check_diagnostic`, `run`, `load_module`.

**`interface_check.py`** is an end to end check on real models. It loads each one, checks the shapes and computes the
legal rate. It saves nothing. Main function: `check`.

**`d1/next_token.py`** scores D1. `score` calls `legal_rate` for the headline number, and `sampled_legal_rate` for a
secondary number (10 sampled moves per position, stored in `details`). It writes `results/D1_next_token/`.

**`d2/state_equiv.py`** scores D2. `build_histories` keeps one row per distinct move prefix, `same_state_pairs` and
`different_state_pairs` build the pairs, `accepted_sets` takes the model's top k moves (k = number of legal moves),
`compression_errors` and `distinction_errors` give the two errors, and `evaluate` combines them. `prepare` builds the
pairs once per distribution, because they depend only on the data. It writes `results/D2_state_equiv/`.

**`d2/toy_check.py`** shows, on two concrete pairs of histories, exactly what D2 computes. It prints boards and writes
no result. Zoo models are not allowed.

**`d3/inductive_bias.py`** scores D3. `examples` picks one random position per game, `random_like` builds the random
baseline, `adapt_and_score` adapts a copy with a new head and returns its balanced accuracy, and `evaluate` runs every
task and seed. `symbolic_arm`, `pysr_fit` and `describe_law` are the PySR part, which is descriptive and does not change
the score. It writes `results/D3_inductive_bias/`.

**`d3/toy_probe.py`** is D3 on a tiny world: a walker on a line with walls. It shows whether a model picks up the true
rule or a shortcut. It writes `outputs/d3_toy/results.json`.

**`d4/representation_probe.py`** scores D4. `load_data` splits the games, `control_table` builds the meaningless control
labels, `fit_probes` trains the linear and MLP probes per layer, and `evaluate_probes` measures them. `causal_check`
runs the causal test using `causal_cases`, `probe_edit`, `boundary_accuracy` and `causal_label`. It writes
`results/D4_probing/`.

**`d4/reproduce_nanda_probe.py`** is the reproduction gate. It checks that our probe gets the published numbers on Othello
GPT (Nanda et al. 2023). Main functions: `resid`, `train_and_test`. It writes to `outputs/d4_repro/`.

**`d4/toy_probe.py`** is D4 on 4x4 Othello with a tiny transformer, and runs on a laptop. It has its own 4x4 rules and
writes to `outputs/d4_toy/`.

Each of `next_token.py`, `state_equiv.py`, `inductive_bias.py` and `representation_probe.py` also has `score`,
`load_any` and `prereg_tag_exists`, and refuses to score zoo models if its preregistration tag is missing.

### `src/fragility/`

**`othello.py`** is fragility v1, frozen under the tag `fragility-v1`. `score_model` runs every variant, using
`symmetry_map`, `transform_games`, `after_pass_positions`, `legal_rate_at` and `detour_rollout`. `suite_hash` is the
hash that the tests pin. It writes `results/fragility/`.

**`othello_v2.py`** is fragility v2, the H3 outcome, frozen under the tag `fragility-v2`. It reuses v1 and replaces two
parts: `detour_rollout` now plays every game to its end, and `after_pass_matched` compares pass positions only with
similar positions. `correct_at` marks which top moves are legal. It writes `results/fragility_v2/`.

### `src/analysis/`

**`hypotheses.py`** is the preregistered analysis. `load_table` builds one row per model, then `h1`, `a3`, `h2`,
`h3_primary`, `h3_secondary`, `causal_and_occupancy` and `fragility_distribution` compute the statistics. `analyse` runs
them all, and `write_outputs` and `_figures` save them. Every interval comes from the configuration bootstrap
(`clusters`, `boot_indices`, `boot`, `interval`).

**`exploratory_othello.py`** holds the checks that are not preregistered: `descriptives`, `fragility_parts`,
`h3_by_distribution`, `h3_sensitivity`, `diagnostic_details`. It reads the same data through `hypotheses.py`.

**`sequence_othello.py`** holds the whole game measures: `teacher_forced`, `free_play`, `measure`, `summarize`, `_figure`,
`smoke`. It refuses to run if anything under `src/` is uncommitted (`uncommitted_code`), so the commit a result records
is the code that produced it.

### `src/shellscripts/`

**`fetch_othello_data.sh`** downloads the synthetic games, the Othello GPT weights and the WTHOR archive into `~/artifacts`.
Files that already exist are skipped.

**`train_slice.sh`** trains one slice (1 to 4) from `zoo_slices.csv`. See Section 1, Step 2.

**`score_slice.sh`** scores one slice (0 to 4). See Section 1, Step 4.

**`first_zoo_models.sh`** trained the first one model per architecture, and writes `src/docs/zoo_first_models.md`.

**`throughput.sh`** does short timed runs per scale, and writes `outputs/zoo/throughput.md`.

### `src/tests/`

Each file tests the matching module. CI runs them all on every pull request (`.github/workflows/tests.yml`).

| Test file | Tests |
|---|---|
| `test_othello_rules.py` | The 4x4 rules in the D4 toy probe |
| `test_othello_domain.py` | The 8x8 rules and arrays in `othello.py` |
| `test_othello_distributions.py` | The three training distributions |
| `test_results.py` | The result format |
| `test_registry.py` | The registry and admission tables |
| `test_interface.py` | `ZooModel` and `run` |
| `test_d1_next_token.py`, `test_d2_state_equiv.py`, `test_d3_inductive_bias.py`, `test_d4_representation_probe.py` | The four diagnostics |
| `test_othello_fragility.py`, `test_othello_fragility_v2.py` | The fragility suites, including their frozen hashes |
| `test_hypotheses.py` | The H1 to H3 analysis |
| `test_sequence_othello.py` | The whole game measures |

### `src/docs/`

| File | What it is |
|---|---|
| `models.csv`, `run_matrix.csv` | The planned models and runs, written by `run_matrix.py` |
| `zoo_slices.csv` | Who trains and scores which of the 81 models |
| `zoo_registry.csv`, `zoo_admission.csv` | Written by `registry.py` |
| `diagnostic_interface.md` | How to write a diagnostic module |
| `results_schema.md` | The result format |
| `d1_next_token_validity.md`, `d2_state_equivalence.md`, `d3_inductive_bias_probe.md`, `d4_representation_probe.md` | One explainer per diagnostic |
| `d4_reproduction.md`, `d4_reproduction_results.json` | The D4 reproduction gate and its results |
| `fragility_suite.md` | The fragility suite |
| `literature_notes.md` | Notes on the papers |
| `throughput.md`, `zoo_first_models.md` | Timing and the first models |


## 6. How to run each step

Run everything from the repo root. Steps that need models or game data run on the EC2. Activate the Python environment
first (see [Section 7](#7-things-that-surprise-people)).

**Setup and data:**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
bash src/shellscripts/fetch_othello_data.sh
```

**Tests, and checks that need no data:**

```bash
python -m pytest src/tests -q
python src/zoo/train.py --synthetic 200 --steps 20 --batch 16 --device cpu
python src/analysis/sequence_othello.py --smoke
```

**Train:**

```bash
bash src/shellscripts/train_slice.sh 2 --dry_run
bash src/shellscripts/train_slice.sh 2
python src/zoo/train.py --arch lstm --scale small --distribution synthetic --seed 0
python src/zoo/registry.py
```

`--dry_run` only prints the plan. Use `tmux` for a real slice, because it runs for hours.

**Score:**

```bash
bash src/shellscripts/score_slice.sh 2 --dry_run
bash src/shellscripts/score_slice.sh 2
bash src/shellscripts/score_slice.sh 0 --only othello_lstm_small_synthetic_s0
```

**Run one diagnostic or suite on one model:**

```bash
python src/diagnostics/d1/next_token.py --models othello_lstm_small_synthetic_s0 --save
python src/diagnostics/d2/state_equiv.py --models othello_lstm_small_synthetic_s0 --save
python src/diagnostics/d3/inductive_bias.py --models othello_lstm_small_synthetic_s0 --save
python src/diagnostics/d4/representation_probe.py --models othello_lstm_small_synthetic_s0 --save
python src/fragility/othello.py --model outputs/zoo/othello_lstm_small_synthetic_s0/model.pt --arch lstm --scale small --train_distribution synthetic --seed 0
python src/fragility/othello_v2.py --model outputs/zoo/othello_lstm_small_synthetic_s0/model.pt --arch lstm --scale small --train_distribution synthetic --seed 0
```

Leave out `--save` to print a score without writing a file. The anchors work the same way, with
`--models othello_gpt_synthetic othello_gpt_championship`.

**Analyse (no GPU needed for the first three):**

```bash
python src/component/results.py --domain othello
python src/analysis/hypotheses.py --domain othello
python src/analysis/exploratory_othello.py
python src/analysis/sequence_othello.py --all --anchors
python src/analysis/sequence_othello.py --summarize
```

**Toys and checks:**

```bash
python src/diagnostics/interface_check.py
python src/diagnostics/d2/toy_check.py --model untrained
python src/diagnostics/d3/toy_probe.py
python src/diagnostics/d4/toy_probe.py
python src/diagnostics/d4/reproduce_nanda_probe.py
```

**Demo:**

```bash
python3 -m streamlit run demo/app.py
```


## 7. Things that surprise people

**Which environment to use.** Script docstrings say `conda env wmconcord`, and the README uses `.venv`. Use whichever has
`requirements.txt` installed on your instance (on Nikhil's instance: `source ~/dl-venv/bin/activate`). PySR installs
Julia the first time it is imported, which takes a few minutes.

**Where checkpoints are.** They never go in git. Each one is on the instance that trained it, in
`outputs/zoo/<model_id>/model.pt`, and on S3 at `s3://dats-capstone/shared/group6/zoo/<model_id>/`.

**The hash is not always checked.** `score_slice.sh` and `load_model` check the checkpoint hash. Running a D1 to D4
script or a fragility script by hand does not, because those load the checkpoint without checking it.

**Git tags block scoring.** D1 and D2 need `prereg-v1.1`, and D3, D4 and `score_slice.sh` need `prereg-v1.2`. If the tag is
missing on your machine, run `git fetch --tags`. The anchors can be scored without a tag.

**Do not casually rerun `run_matrix.py`.** Once frozen, rows are only marked done or skipped, never added.

**Never edit frozen files.** The fragility tests pin a hash of `src/fragility/othello.py` and `othello_v2.py`, so any
change makes them fail.

**Batch size can change a result very slightly on a GPU.** Rounding depends on the batch size, so a near tie in the top
move can break the other way. That is why `sequence_othello.py` uses batch 256, the same as D1. The `free_play` docstring
still says batch size does not change the result, which holds in the CPU test but not exactly on a GPU.

**Move 60 is never trained.** As Section 2 explains, a zoo model reads at most 58 tokens in training, so it never learns
to predict move 60. For the transformer, the position vector for the 59th token (`self.pos` index 58) gets no gradient
at all, so it stays close to its random starting value. What this affects:

| Where | Does it read that last position? | Effect |
|---|---|---|
| Training | No | None |
| D1, competence, fragility teacher forced rates | Reads it but does not score it | None |
| Fragility detours (v1 and v2) | No (they stop at move 59) | None |
| Free play (`free_play`) | Yes, for move 60 | Measured below |
| D2, D3, D4 | Yes, when a game reaches 59 tokens (a small share of positions, roughly 1 in 59) | Not measured |

In free play, the legal rate at moves 59 and 60 is:

| Models | Move 59 | Move 60 |
|---|---|---|
| Zoo transformers | 0.846 | 0.550 |
| Zoo Mamba | 0.816 | 0.801 |
| Zoo LSTMs | 0.315 | 0.389 |
| Othello GPT synthetic | 0.992 | 1.000 |
| Othello GPT championship | 0.956 | 0.967 |

The zoo transformers drop sharply at move 60, while the anchors do not. Including move 60 changes a zoo model's overall
free play rate by 0.003 on average, and by 0.012 at most. The full game rate is probably affected more, because one
illegal move 60 fails the whole game. The stored results do not keep enough per game detail to measure that without a
rerun.
