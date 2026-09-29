# D4 reproduction gate: linear probe on Othello GPT

Goal (Instructor Review 2, #22): reproduce one published number with our own code before building on it.

Result: **passed.** Our linear probes match Nanda, Lee and Wattenberg (2023), Table 1, within 1 point on 5 of 6 published values.

## Setup

| Item | Value |
|---|---|
| Model | Othello GPT, synthetic model (Li et al. 2023), TransformerLens weights from `NeelNanda/Othello-GPT-Transformer-Lens` |
| Data | othello_world synthetic games (`alexandretl/othello`), 100,000 train games, 1,000 held out test games from the val split |
| Board labels | our own 8x8 Othello rules; every game replayed, 0 illegal games in 101,000 |
| Probe | one linear layer per layer (512 to 64 squares x 3 classes), residual stream after each block, all positions |
| Training | AdamW, lr 1e-3, weight decay 0.01, batch 256 games, 2 epochs, seed 0 |
| Hardware | 1 x A10G (g5.2xlarge), 206 seconds total |
| Code | `src/diagnostics/d4/reproduce_nanda_probe.py` |

Sanity check: the model's top predicted move is legal 99.93% of the time on test games (Li et al. report about 99.99%).

## Result next to the published numbers

Tolerance chosen before comparing: within 1 percentage point. Nanda et al. trained on 3.5M games; we used 100,000, so slightly lower accuracy is expected.

| Labels | Layer | Ours (%) | Nanda et al. 2023 (%) | Difference | Within 1 point |
|---|---|---|---|---|---|
| mine/theirs/empty | 0 | 90.5 | 90.9 | -0.4 | yes |
| mine/theirs/empty | 4 | 98.7 | 99.0 | -0.3 | yes |
| mine/theirs/empty | 7 | 98.9 | 99.5 | -0.6 | yes |
| black/white/empty | 0 | 75.8 | 62.2 | +13.6 | **no** |
| black/white/empty | 4 | 75.9 | 75.0 | +0.9 | yes |
| black/white/empty | 7 | 75.2 | 74.4 | +0.8 | yes |

All layers:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| mine/theirs/empty | 90.5 | 94.5 | 96.9 | 98.0 | 98.7 | 99.1 | 99.1 | 98.9 |
| black/white/empty | 75.8 | 76.0 | 76.0 | 76.0 | 75.9 | 75.8 | 75.4 | 75.2 |

The key finding of the paper reproduces: a linear probe reads the board almost perfectly when labels are relative to the player (mine/theirs), but only about 75% when labels are absolute colours (black/white).

## Open item

Black/white at layer 0 is 13.6 points above the published value, while every other value matches. Our black/white accuracy is flat at about 76% across all layers, so the most likely cause is a difference in what "layer 0" means or which positions are scored for that row. To check against the authors' code. This does not affect the gate, since D4 uses mine/theirs labels.

## How to rerun

```
python src/diagnostics/d4/reproduce_nanda_probe.py
```

Writes `outputs/d4_repro/results.json` and `outputs/d4_repro/results.md`.

## Reference

Nanda, N., Lee, A., and Wattenberg, M. (2023). Emergent Linear Representations in World Models of Self Supervised Sequence Models. arXiv:2309.00941.
