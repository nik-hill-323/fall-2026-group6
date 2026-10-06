# First zoo models: synthetic Othello, scale small, seed 0

Trained with `src/zoo/train.py` at the full budget in `src/component/run_matrix.py`
on NVIDIA A10G, 2026-10-04, commit eb09a86.
Legal move rate is Diagnostic 1: the share of the model's top next moves that the rules allow, on 1,000 validation games.

| model_id | params | steps | final loss | next token acc | legal move rate (D1) | train minutes |
|---|---|---|---|---|---|---|
| othello_transformer_small_synthetic_s0 | 0.82M | 5000 | 2.641 | 0.131 | 0.8372 | 1.6 |
| othello_lstm_small_synthetic_s0 | 0.54M | 5000 | 3.472 | 0.085 | 0.5660 | 0.7 |
| othello_mamba_small_synthetic_s0 | 0.48M | 5000 | 2.719 | 0.129 | 0.8424 | 17.9 |
