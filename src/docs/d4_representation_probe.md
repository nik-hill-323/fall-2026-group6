Diagnostic 4: Representation probing. D4 asks whether a model keeps an internal picture of the true state of the world, not just good guesses about the next token. Take a small GPT trained only on Othello move sequences such as "D3 C5 F6". It never sees a board. After each move, we record the model's internal activations, the vector of numbers inside the network at that point, and train a simple classifier called a probe to read the board from that vector: for each of the 64 squares, is it mine, the opponent's, or empty? If the probe reads the board accurately, the board is encoded inside the model. Two checks keep this honest. First, a control probe trained on shuffled labels must score low. Otherwise the probe is memorizing rather than reading the model. The gap between the real probe and the control is called selectivity. Second, a causal check: we edit the activation so one square flips, and see whether the model's predicted legal moves change to match the edited board. If they do, the model actually uses that internal board rather than just storing it.

## Protocol

Code: `src/diagnostics/d4/representation_probe.py`. Tests: `src/tests/test_d4_representation_probe.py`.
Fixed in `reports/preregistration.md` (Section 3 and Amendment 1, A6) before any zoo model is scored.

1. **Data.** Games of the model's own training distribution, data seed 0. The first 10,000 of
   11,000 val games fit the probes, the last 1,000 choose the layer, and 1,000 test games give the
   score, so the test games never influence any choice.
2. **Activations.** The vector after every layer (`ZooModel.activations`): the residual stream for
   transformers and Mamba, the hidden state for the LSTM. Each feature is standardized with its mean
   and spread over the fitting games, so every architecture gets a fair probe.
3. **Probe.** For every layer, one linear classifier reads mine, theirs or empty for all 64 squares.
4. **Control task (Hewitt and Liang 2019).** Each position gets a type: the square just played and
   the position in the game. Every type gets a fixed random label per square, drawn from that
   square's real mix of classes. A control probe is trained the same way on these labels. It can
   only score by memorizing types, never by reading the board.
5. **Selectivity** = probe accuracy minus control accuracy. The best layer is the one with the
   highest selectivity on the layer choice games. **Score** = that layer's selectivity on the test
   games. Higher means more world model like.
6. **Also saved, per layer:** probe accuracy, control accuracy, gain over layer 0, and accuracy on
   occupied squares only (whose disc it is, which the move list alone does not give away). Plus
   the **occupancy only baseline**: the accuracy of knowing only which squares are occupied (empty
   squares called empty, occupied squares given their most common disc class).

Check (from the tests): on a model whose second layer holds the true board, the probe reads it at
0.9999 (1.0 on occupied squares), picks that layer, and the control stays near chance (0.54).

## How to run

```
conda activate wmconcord
cd ~/fall-2026-group6
python -m pytest src/tests/test_d4_representation_probe.py -q
python src/diagnostics/d4/representation_probe.py --models othello_gpt_synthetic othello_gpt_championship
```

Zoo models (`--models <model_id> ... --save`) are refused until the git tag `prereg-v1.1` exists.
The causal check (editing the activation so one square flips) comes in Weeks 9 to 10, with its pass
rule added to the preregistration first.
