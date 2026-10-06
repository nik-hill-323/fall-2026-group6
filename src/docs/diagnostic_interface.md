# Diagnostic interface: how to write a diagnostic module

Code: `src/diagnostics/interface.py`. Tests: `src/tests/test_interface.py`.
Results format: `src/docs/results_schema.md`.

Every diagnostic (D1 to D4) is one Python module that follows the same contract, and every model
(transformer, LSTM, Mamba, and the reference Othello GPTs) is reached through the same two calls.
A diagnostic therefore never needs to know which architecture it is scoring.

## The model: `ZooModel`

| Call | Returns | Use |
|---|---|---|
| `model.logits(tokens)` | `(batch, positions, 61)` next move scores | D1, D2, D3 |
| `model.activations(tokens)` | `(layers, batch, positions, width)`: the vector after each layer | D4 |
| `model(tokens)` | same as `logits`, so a ZooModel works with `othello.legal_rate` | |
| `model.model_id`, `model.arch`, `model.n_layers`, `model.d_model`, `model.device` | | |

`tokens` are `(batch, 59)` ints: 1 to 60 are squares, 0 is padding (see `src/domains/othello.py`).
Activations are the residual stream after each block for transformers and Mamba, and the hidden state
after each layer for the LSTM.

Load models with:

```python
itf.othello_gpt("synthetic")                       # reference model, anchor only
itf.from_checkpoint("othello_transformer_large_synthetic_s0")   # a trained zoo model
```

## The contract

A diagnostic module defines two things:

```python
DIAGNOSTIC = "D1"                                   # D1, D2, D3 or D4
METHOD = "legal move rate, argmax, higher is better"   # one line, goes into the result

def score(model, split, n_games, data_seed, **config):
    games = oth.load_distribution(<the model's training distribution>, split, n_games, seed=data_seed)
    ...                                             # compute the diagnostic
    return itf.result_for(model, DIAGNOSTIC, method=METHOD, split=split, n_games=n_games,
                          data_seed=data_seed, score=<higher is better>, details={...})
```

Rules:

* `score` is the master table number and **higher always means a more world model like model**.
* Final scores use the **test** split. Anything the diagnostic fits (a probe, an adaptation) is fitted
  on **val** or train data, never on test.
* Use `src/domains/othello.py` for games, boards and legal moves. Do not copy the rules.
* Raw numbers go in `details` as plain JSON (`float(x)`, `array.tolist()`).
* Do not catch errors to hide them: `run()` turns a crash into a `failed` result with the reason.

## Running it

```python
results = itf.run(my_module, models, split="test", n_games=1000, data_seed=0)
```

`run()` scores each model, saves each result to `results/{diagnostic folder}/{model_id}.json`, and
keeps going if one model crashes. Then `python src/component/results.py --domain othello` builds the
master table.

## Check on the EC2

```
python src/diagnostics/interface_check.py
```

loads the reference Othello GPT and a zoo model through the interface, checks the shapes, and prints
their legal move rate on 1,000 test games (nothing is saved).
