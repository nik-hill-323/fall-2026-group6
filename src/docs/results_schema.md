# Results schema

Code: `src/component/results.py`. Tests: `src/tests/test_results.py`.

Every diagnostic (D1 to D4) writes **one JSON file per model** in one shared format. The master table
for H1 to H3 is then a single join of these files, and every number in it traces back to the code,
data and machine that produced it.

## The one rule

**`score` is the single number that goes into the master table, and higher always means a more world
model like model.**

| Diagnostic | Natural number | What goes in `score` |
|---|---|---|
| D1 next token validity | legal move rate | the rate itself |
| D2 state equivalence | compression error, distinction error (lower is better) | a higher is better form, for example 1 minus the mean error; both raw errors go in `details` |
| D3 inductive bias | adaptation gain over a random init model | the gain |
| D4 probing | probe accuracy, control accuracy, selectivity per layer | the agreed summary of selectivity; every layer goes in `details` |

The exact choice for each diagnostic is written in its `method` field and in the preregistration.
Without this rule, rank correlations between diagnostics would come out with the wrong sign.

## Where files go

Paths come from `src/component/run_matrix.py`, the same paths listed in `src/docs/run_matrix.csv`:

```
results/D1_next_token/{model_id}.json
results/D2_state_equiv/{model_id}.json
results/D3_inductive_bias/{model_id}.json
results/D4_probing/{model_id}.json
```

Training records of zoo models are not diagnostic results and live in `results/zoo_training/`.

## Fields

| Field | Meaning |
|---|---|
| `model_id` | as in `models.csv`, for example `othello_transformer_small_synthetic_s0` |
| `diagnostic` | `D1`, `D2`, `D3` or `D4` |
| `method` | one line: what the score is and how it is oriented |
| `split` | data split scored on: `train`, `val` or `test` (final scores use `test`) |
| `n_games`, `data_seed` | how much data, and which seed picked it |
| `score` | the master table number, higher is better; empty only when `status` is not `ok` |
| `score_std`, `score_ci` | spread and `[low, high]` interval of the score, when known |
| `details` | anything else, as plain JSON (per layer numbers, raw errors, control accuracy) |
| `status`, `reason` | `ok`, `failed` or `skipped`; a failed or skipped run must say why |
| `domain`, `arch`, `scale`, `distribution`, `seed` | filled from `models.csv` |
| `run_id` | `{model_id}_{diagnostic}` |
| `schema_version` | 1 |
| `provenance` | git commit, time, Python, torch, numpy, GPU, host, runtime |

A result that breaks any of these is rejected **before** it is written (`SchemaError`).

## How a diagnostic uses it

```python
res = make_result(model_id, "D4", method="linear probe, mine/theirs, mean selectivity over layers",
                  split="test", n_games=1000, data_seed=0, score=0.31,
                  details={"selectivity_by_layer": [...]}, runtime_sec=180.0)
save(res)
```

A model that is not in the zoo (for example the reference Othello GPT) passes `domain`, `arch`,
`scale`, `distribution` and `seed` itself.

## Master table

```
python src/component/results.py --domain othello
```

writes `results/master_table.csv`: one row per model, a `score` and `status` column per diagnostic,
and prints how many runs per diagnostic are still missing. Review 3 rule: every planned run has a score
or a written reason (`skipped` or `failed`), with zero silent gaps.
