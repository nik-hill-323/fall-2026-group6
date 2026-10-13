## Demo app

A Streamlit app that lets you pick any model in the zoo and watch each diagnostic run on it, with
the metrics next to it. Built for the Preliminary Presentation (October 20) at the instructor's
request: the baseline and each test shown on a chosen model, not described in words.

Run from the repo root, on a machine that has the checkpoints (`outputs/zoo/`) and the Othello data
(`~/artifacts`, see `src/shellscripts/fetch_othello_data.sh`):

```
streamlit run demo/app.py
```

Pages (left sidebar; the model picker is above them):

| Page | Shows | Needs |
|---|---|---|
| Model zoo | the registry, the selected model's training record, which results exist for it | `src/docs/zoo_registry.csv`, `results/zoo_training/` |
| D1 live | a real game replayed move by move: true board, legal moves, the model's pick, running legal move rate | a checkpoint and the Othello data |
| D2, D3, D4 | the saved result for the selected model (score, method, per layer or per task chart); otherwise what the test will show | `results/D2_state_equiv/`, `results/D3_inductive_bias/`, `results/D4_probing/` |
| Fragility | in distribution rate, drop per variant, detour completion rates | `results/fragility/` |
| Concordance | master table of admitted models and Kendall / Spearman agreement between diagnostics | two or more diagnostics scored on three or more models |

The two published Othello GPTs appear in the picker when their weights are under `~/artifacts/othello_gpt_tl`
(they need `transformer-lens` installed). The app only reads files; it never writes results.

Status, October 6: Model zoo, D1 live and the result pages work on the four trained models. Fragility and
D2 to D4 fill in as each owner's module writes results. Concordance fills in once the Othello zoo is scored.

The `fig/` folder is for the one minute project video (`demo.gif`) and demo screenshots.
