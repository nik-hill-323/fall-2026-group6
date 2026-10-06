# Diagnostic 3: Inductive Bias Probe

Diagram: `reports/diagrams/d3_inductive_bias.drawio.svg`. Toy code: `src/diagnostics/d3/toy_probe.py`.

## The idea in one example

Suppose a model has been trained on a very large number of Othello games written only as move lists.
It never sees a board and is never told the rules. Its only job is to predict the next move, and it
becomes good at it. The question is whether it learned Othello or only which move tends to follow which.
From the outside the two look the same.

Diagnostic 3 gives the model a job it has never done and almost no examples to learn it from. For
Othello the new job could be "is this square occupied after these moves?" with sixteen labelled games.

* If the model already tracks the board, sixteen examples are enough. It only has to learn what the
  question means.
* If it only memorized move patterns, sixteen examples are useless, because the answer depends on a
  board it does not have. It fails, or it picks up a shortcut that happens to fit the sixteen.

A second model with the same architecture and random weights gets the same sixteen examples. It shows
what sixteen examples are worth on their own. The diagnostic is the difference between the two.
That difference is the model's inductive bias: what it assumes when it has almost no data.

This is the probe of Vafa et al. (2025). Their orbit model predicted trajectories well but, once adapted
to a new physics task, behaved as if it had learned a force law that makes no physical sense.

## Steps

1. Take the pretrained model.
2. Build a small labelled set for a new task whose answer follows from the true rules.
3. Adapt the whole model on that set. The new output head learns at a normal rate and the rest at a
   much smaller rate, so the result shows what the model brought with it and not what a few hundred
   steps can teach.
4. Score it on fresh sequences.
5. Do steps 3 and 4 again with an untrained copy of the model.
6. Score = accuracy of the adapted pretrained model minus accuracy of the adapted untrained model.
7. Compare the adapted model's wrong answers with a known shortcut rule to see which rule it picked up.

## Toy version

Othello is too large for a demo that runs in seconds, so the toy uses the smallest world with the same
shape: a walker on a line with six positions and a wall at each end. Moves are L and R. Walking into a
wall leaves the walker where it is. The model is pretrained on next move prediction over walks that
never hit a wall. The new task is "where is the walker now?" with sixteen labelled walks that do hit
walls. The shortcut is to count R minus L and ignore the walls.

```
python src/diagnostics/d3/toy_probe.py                    # one seed, full walk-through
python src/diagnostics/d3/toy_probe.py --seeds 0 1 2 3 4  # the table below
```

Five seeds, 300 test walks each (Instructor Review 3 asked for more than one seed before this goes on a slide):

| seed | pretrained | scratch | shortcut | gap | on wall cases: true answer | on wall cases: shortcut answer |
|---|---|---|---|---|---|---|
| 0 | 0.64 | 0.39 | 0.53 | +0.25 | 0.51 | 0.19 |
| 1 | 0.78 | 0.45 | 0.49 | +0.33 | 0.72 | 0.03 |
| 2 | 0.70 | 0.37 | 0.49 | +0.33 | 0.64 | 0.07 |
| 3 | 0.64 | 0.52 | 0.57 | +0.12 | 0.55 | 0.07 |
| 4 | 0.83 | 0.52 | 0.52 | +0.31 | 0.78 | 0.03 |
| mean ± sd | 0.72 ± 0.09 | 0.45 ± 0.07 | 0.52 ± 0.03 | 0.27 ± 0.09 | 0.64 ± 0.11 | 0.08 ± 0.07 |

What the table says:

* The gap is positive on all five seeds, 0.27 on average. Pretraining on next move prediction gave the
  model something about position that sixteen examples alone do not.
* On the walks where a wall was hit, the adapted pretrained model gives the true position 64% of the
  time and the shortcut's answer 8% of the time. It did not fall back on counting. Its remaining
  errors are other wrong positions.
* An earlier reading of seed 0 alone, from eight printed rows, was that the model "learned counting,
  not walls". The five seed numbers do not support that, and it should not be said in the presentation.
* The spread across seeds is large (gap from 0.12 to 0.33), so a single seed is not enough to rank
  models. The zoo runs three seeds per model for this reason.

## On the model zoo

For each zoo model the probe will use tasks built from the true board with a small labelled set from
the validation split, an untrained copy of the same architecture as the baseline, and the test split
for scoring. The score written to the master table is the gap. The symbolic regression step of Vafa
et al. is a second, descriptive output and uses several seeds because it is stochastic.

## How it differs from Diagnostic 4

D4 reads the state out of the activations and does not change the model. D3 changes the model and
watches how it learns. A model can pass one and fail the other, which is why WM-Concord runs both on
the same models.
