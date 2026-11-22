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

Zoo models (`--models <model_id> ... --save`) are refused until the git tag `prereg-v1.2` exists.
The nonlinear probe and the causal check (editing the activation so one square flips) are fixed in
Amendment 2 (B1, B2) and saved in `details["mlp"]` and `details["causal"]`.

## Navigation (domain `maps`, Amendment 5, G7)

The "board" of a route is where the car is. The probe reads the **current intersection**, one of 4,579
classes, from the model's vector at every scored position. Everything else is A6, B1 and B2.

* **Data.** The first 10,000 of 11,000 validation routes fit, the last 1,000 choose the layer, 1,000 test
  routes give the score (data seed 0). Activations standardized per feature over the fitting routes; AdamW
  lr 1e-3, weight decay 0.01, one step per 1,024 positions, seed 0.
* **Epochs, fixed by a rule before any navigation model is scored.** A synthetic layer of width 128 holds a
  fixed random Gaussian vector of the true current intersection (seed 0; no model). The probe is fitted to it on
  each distribution's fitting routes and scored on its test routes. The count is the smallest of 2, 5, 10 and 20
  at which all three distributions reach 0.99 (20 if none). Othello's 2 epochs read 64 x 3 classes; a 4,579 way
  probe may need more, and an under trained probe would compress the differences between models.

  **Epoch count: 2.** The rule ran on 2026-11-21 (ET) from commit 65cd5d8, before any navigation
  model was scored; output `results/prereg/maps_d4_epochs.json`. Probe accuracy on the test routes after 2, 5, 10
  and 20 epochs: shortest paths 0.99772, 0.99954, 0.99954, 0.99954; noisy shortest paths 0.99774, 0.99914, 0.99914, 0.99914;
  random walks 0.99978, 0.99990, 0.99990, 0.99990. All three reach 0.99 at 2 epochs, the same count as Othello.

* **Control task.** The type of a position is (token just read, number of turns so far); each type gets a
  fixed random intersection drawn from the intersection frequencies of the fitting routes (seed 0).
  Selectivity = probe accuracy minus control accuracy; score = selectivity at the layer chosen on the layer
  choice routes, measured on the test routes.
* **Baselines in `details`.** The accuracy of always answering the start, and of always answering the
  destination, with each layer's gain over them.
* **MLP probe** as B1, in `details["mlp"]`.
* **Causal check** as B2, with its settings unchanged (middle to last layer, step 0.05 x sqrt(width),
  confidence 0.99, at most 100 steps, random edit of the same length, 2,000 bootstrap resamples, same pass
  rule and labels). One case per test route: its scored positions in a random order (seed 0), the first with a
  neighbouring intersection, one legal turn away, whose legal next set differs from the current one and is not
  empty (the first such neighbour in turn order). The edit moves the probe's reading to that neighbour; boundary
  accuracy uses k = the size of the neighbour's legal next set. With 4,579 classes a linear probe cannot always
  make a given class its reading at 0.99 within 100 steps, so `edit_success_rate` matters here: it is reported
  for every model.
* **Memory, for the 1.5B anchors only.** `--layers_per_pass` trains the probes of a few layers at a time and
  `--chunk` runs the forward passes in smaller pieces. Neither changes a result (tests check that the probes
  are the same).

Code: `score_maps` in `src/diagnostics/d4/representation_probe.py`. Tests: `src/tests/test_d4_maps.py`.

```bash
python src/diagnostics/d4/representation_probe.py --epoch_rule
python src/diagnostics/d4/representation_probe.py --models maps_transformer_small_shortest_paths_s100 --save
python src/diagnostics/d4/representation_probe.py --models maps_vafa_random_walks --save --layers_per_pass 8 --chunk 32
```
