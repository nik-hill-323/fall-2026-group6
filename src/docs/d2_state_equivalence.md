# Diagnostic 2: State Equivalence (Myhill Nerode)

Code: `src/diagnostics/d2/state_equiv.py`. Toy check: `src/diagnostics/d2/toy_check.py`.
Tests: `src/tests/test_d2_state_equiv.py`. Diagram: `reports/diagrams/d2_state_equivalence.drawio`.
Protocol: `reports/preregistration.md`, Amendment 1, A4. Paper: Vafa et al. 2024, "Evaluating the
World Model Implicit in a Generative Model".

## The idea with an Othello example

In Othello, the board is the whole truth: once you know the board and whose turn it is, the
move history does not matter for what can happen next. The Myhill Nerode theorem says the same
thing in general: two histories are "the same state" exactly when every possible future is
equally allowed after both. A model with a real world model should therefore treat two
histories that reach the same board the same way, and treat two histories that reach different
boards differently where the boards really differ. D2 checks both.

**Compression (same board, different history).** Play D3, C3, C4. Now play the same three moves
in a different order: C4, C3, D3. Both orders end on the identical board, with white to move,
and in both cases the legal moves are {C5, E3}:

```
board after D3 C3 C4          board after C4 C3 D3
   A B C D E F G H               A B C D E F G H
3  . . W B * . . .            3  . . W B * . . .
4  . . B B B . . .            4  . . B B B . . .
5  . . * B W . . .            5  . . * B W . . .
(* = legal next move, rows 1, 2, 6, 7, 8 are empty)
```

A model that tracks the board must accept the same moves after both histories. If it accepts
{C5, E3} after one order and {C5, F3} after the other, it has not "compressed" the two histories
into one board: it is still reacting to the order of the moves. That is a compression error.

**Distinction (different board, same length).** Play E6, D6, C5. Separately, play F5, D6, C6.
Both are three moves long, but the boards differ, and so do the legal moves:
after the first, {B4, B6, F4, F6}; after the second, {B6, F4, F6}. The one move that is legal
after exactly one of them, B4, is the **boundary** between the two boards. A model that tracks
the board accepts B4 after the first history and rejects it after the second. If it accepts B4
after both, or after neither, it has not "distinguished" the two boards. That is a distinction
error.

Next token validity (D1) can miss both failures: a model can propose a legal move almost every
time while still mixing up boards. D2 looks at pairs of histories instead of single positions,
so it sees what D1 cannot.

## What the test computes

For each model, on the test games of its own training distribution (1,000 games, data seed 0):

1. **Histories.** Every prefix of every game (length 1 to 59) is one history. The rules engine
   (`src/domains/othello.py`) replays it and gives the true board, the player to move, and the
   legal moves L(h). Histories with no legal move are left out. A prefix shared by several games
   is kept once.
2. **Accepted moves.** The model reads the same history and scores all 60 squares. Its accepted
   set A(h) is its k highest scoring squares, where k is the number of truly legal moves. So the
   model gets exactly as many picks as there are legal moves, and no threshold is tuned.
3. **Same state pairs.** Two histories with the identical board and player to move, reached by
   different move sequences. Up to 2,000 pairs, sampled at random with seed 0. These only occur
   where move orders can be swapped, so if fewer than 500 exist, more test games are used, 1,000
   at a time, up to 5,000.
4. **Different state pairs.** Two histories of the same length whose legal move sets differ.
   Up to 2,000 pairs, sampled at random with seed 0.
5. **Compression error** of a same state pair: the share of accepted moves that are accepted
   after only one of the two histories, |A(h) xor A(g)| / |A(h) or A(g)|. 0 is best.
6. **Distinction error** of a different state pair: of the boundary moves (legal after exactly one
   history), the share the model does not separate correctly. 0 is best.
7. **Score** = 1 minus the mean of the mean compression error and the mean distinction error.
   Higher means more world model like, as for every diagnostic.

Both errors, their standard errors, the number of pairs of each kind and the number of games go
into `details` of `results/D2_state_equiv/{model_id}.json` (format: `src/docs/results_schema.md`).

## Sanity checks (from the tests)

| Model | Score | Compression error | Distinction error |
|---|---|---|---|
| Oracle: accepts exactly the legal moves of the true board | 1.000 | 0.000 | 0.000 |
| Constant: always accepts the same squares, ignores the history | 0.516 | 0.000 | 0.969 |
| Noise: a random choice for every exact move sequence | 0.103 | 0.926 | 0.868 |

The constant model shows why both errors are needed: it never confuses two histories of the same
board (it treats everything the same), but it cannot tell any boards apart.

## How to run

```
conda activate wmconcord
cd ~/fall-2026-group6
python -m pytest src/tests/test_d2_state_equiv.py -q
python src/diagnostics/d2/toy_check.py
python src/diagnostics/d2/state_equiv.py --models othello_gpt_synthetic othello_gpt_championship
```

Zoo models (`--models othello_transformer_small_synthetic_s0 ... --save`) are refused until the
git tag `prereg-v1.1` exists, so the protocol is frozen before any zoo score.

## Limits

* Same state pairs come from move orders that can be swapped, which happens mostly in the
  opening (moves 3 to 11 in random games). Compression is therefore measured mostly on early
  boards. The number of pairs at each length is saved in `details`.
* Vafa et al. (2024) also test longer continuations. We use the next move only, which keeps D2
  cheap enough for 81 models per domain and uses the same ground truth as the rest of the study.

## Five minute walkthrough

1. (1 min) The board is the state. Two histories, same board: the future must be the same.
2. (1 min) Compression example: D3 C3 C4 against C4 C3 D3, legal moves {C5, E3} after both.
3. (1 min) Distinction example: E6 D6 C5 against F5 D6 C6, boundary move B4.
4. (1 min) The computation: accepted moves = top k, the two errors, score = 1 minus their mean.
5. (1 min) Run `toy_check.py` live: it prints both pairs, the boards, what the model accepts, and
   each error.
