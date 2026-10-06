# First zoo models: synthetic Othello, scale small, seed 0

Trained with `src/zoo/train.py` at the full budget in `src/component/run_matrix.py`
on NVIDIA A10G, 2026-10-04, commit eb09a86.
Legal move rate: the share of the model's top next moves that the rules allow, on 1,000 validation games.
This is a training check. The full training records are in `results/zoo_training/`. Diagnostic D1 itself is
scored on the test split and written in the results schema to `results/D1_next_token/` (see `src/docs/results_schema.md`).

| model_id | params | steps | final loss | next token acc | legal move rate (D1) | train minutes |
|---|---|---|---|---|---|---|
| othello_transformer_small_synthetic_s0 | 0.82M | 5000 | 2.641 | 0.131 | 0.8372 | 1.6 |
| othello_lstm_small_synthetic_s0 | 0.54M | 5000 | 3.472 | 0.085 | 0.5660 | 0.7 |
| othello_mamba_small_synthetic_s0 | 0.48M | 5000 | 2.719 | 0.129 | 0.8424 | 17.9 |

Note (2026-10-06): the LSTM row above was trained with the old LSTM class (one multi layer `nn.LSTM`).
The LSTM is now a stack of single layer LSTMs so every layer can be probed, so this checkpoint no longer
loads; its training record was removed and the model is retrained with the Othello zoo.
