# Fragility suite (Othello), version 1

Instructor Review 3 (#41): build and freeze the fragility suite before any zoo model is scored. H3 asks
which diagnostic best predicts whether a model breaks on shifted tasks. If "breaks" is defined after
we have seen diagnostic scores, H3 is circular. This file is the definition, written first.

Code: `src/fragility/othello.py`. Tests: `src/tests/test_othello_fragility.py`.

## What it measures

A model is trained to predict the next Othello move. The suite asks: when the situation is related
to training but shifted, how much more often does the model propose an illegal move?

Every variant keeps the Othello rules exactly as they are. Only the states the model is asked about
change, so the right answer always comes from the rules in `src/domains/othello.py`. Nothing in the
suite retrains or adapts the model.

| Variant | What changes | Why it is a shift | Score |
|---|---|---|---|
| in distribution | nothing: test split of the model's own training distribution | the reference | legal move rate |
| cross distribution | test split of each distribution the model was not trained on | random play vs human play reach different boards | legal move rate |
| after pass | only positions where the opponent has no move and the same player moves again | rare, and "players alternate" stops working | legal move rate at those positions |
| symmetry | test games mapped through 180 degree rotation and the two diagonal reflections | same game with squares renamed; human games use one orientation by convention | legal move rate, one per symmetry |
| detour | the model plays itself; with probability p its move is replaced by another legal move, random or the one it rates least likely | pushes the game off the model's preferred line, as in the detour test of Vafa et al. (2024) | share of games finished with no illegal proposal |

Only the three symmetries that keep the start position are used. A 90 degree rotation or a left to
right mirror swaps the colours of the four start squares, which turns black's legal first moves into
white's, so those would not be legal games.

## The number written per model

`results/fragility/{model_id}.json` holds one rate per variant and one **drop** per variant:

* static variants: in distribution legal rate minus the variant's legal rate
* detour: completion rate with no detours (p = 0) minus completion rate with detours, for each mode and p

`fragility` = the mean of all drops. Higher means more fragile. This is the outcome H3 regresses on
each diagnostic score. The per variant drops are kept so H2 can ask which shifts each diagnostic sees.

A drop only means something for a model that works in distribution. An untrained model scores a
fragility near zero (0.02 to 0.03 in a smoke run) because it is equally bad everywhere and has nothing
to lose. So the in distribution rate is always reported next to `fragility`, and H3 uses only models
that pass the zoo's admission threshold on held out next move prediction. That threshold is set from
the first trained models, before any of them is scored on this suite.

## Frozen parameters

| Parameter | Value |
|---|---|
| version | `othello-fragility-v1` |
| hash of the definition | `644671bfb322` |
| games per static variant | 1,000 from the test split, data seed 0 |
| detour games | 500 per setting |
| detour probabilities | 0.0 (reference), 0.1, 0.5 |
| detour modes | random, adversarial |
| detour seed | 0 |

The test split is used only here and for final diagnostic scores, never for training or for fitting
a diagnostic (see the split rules in `src/domains/othello.py`).

## How it is frozen

1. This PR is reviewed and merged before any zoo model is scored by any diagnostic.
2. The merge commit is tagged: `git tag fragility-v1 && git push origin fragility-v1`.
3. `test_suite_is_frozen` fails if the definition changes. A different suite must be a new version
   with a new tag, and the paper reports which version each number came from.

## One difference from the proposal

The proposal lists "altered legal move rules" for the game domains. The model only sees move tokens,
so nothing in its input says the rules changed; it could only follow a new rule after being adapted
to it. Adaptation with little data is exactly what D3 measures. Putting it in the criterion would make
H3 partly "does D3 predict D3". So version 1 keeps the rules fixed and shifts the states. This is a
choice the team should confirm when approving the PR.

## Checks

* A perfect world model (replays the rules, proposes only legal moves) scores fragility 0 on every variant.
* A broken model (always proposes the same square) fails the static score and the detour rollouts.
* Each symmetry keeps the start board, and a mapped game replays to the mapped boards with the same movers.
* The after pass positions are exactly the positions where the opponent has no legal move.
* The same seed gives the same detour rollout.

## Not covered yet

Navigation and the interpreter need their own suites (detour routing, changed instruction semantics).
They will follow the same rule: defined and tagged before any model in that domain is scored.
