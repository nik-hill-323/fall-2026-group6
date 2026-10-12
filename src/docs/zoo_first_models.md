# First zoo models: synthetic Othello, scale small, seed 0

Trained with `src/zoo/train.py` at the full budget in `src/component/run_matrix.py`
on NVIDIA A10G. First trained 2026-10-04 (commit eb09a86); all three retrained on 2026-10-06 so every
record has a checkpoint hash: the LSTM with the stacked LSTM class (PR #61, commit 9ea32b6), the
transformer and Mamba (PR #67, commit 10ecd35). The rows below are the retrained models.
Legal move rate: the share of the model's top next moves that the rules allow, on 1,000 validation games.
This is a training check. The full training records are in `results/zoo_training/`. Diagnostic D1 itself is
scored on the test split and written in the results schema to `results/D1_next_token/` (see `src/docs/results_schema.md`).

| model_id | params | steps | final loss | next token acc | legal move rate (D1) | train minutes |
|---|---|---|---|---|---|---|
| othello_transformer_small_synthetic_s0 | 0.82M | 5000 | 2.641 | 0.131 | 0.8372 | 1.6 |
| othello_lstm_small_synthetic_s0 | 0.54M | 5000 | 3.483 | 0.084 | 0.5610 | 0.7 |
| othello_mamba_small_synthetic_s0 | 0.48M | 5000 | 2.719 | 0.129 | 0.8422 | 17.8 |

Note (2026-10-12, Instructor Review #4): the LSTM row now shows the retrained stacked LSTM
(0.5610). The earlier row (0.5660) came from the old LSTM class (one multi layer `nn.LSTM`), whose
checkpoint no longer loads. All three values match `src/docs/zoo_registry.csv`.
