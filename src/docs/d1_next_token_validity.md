# Diagnostic 1: Next-Token Validity

Suppose the previous moves are D3, C3, and C4. This sequence is the only information given to the sequence model.
Independently, the same move sequence is replayed using an Othello simulator starting from the standard initial board.
The simulator reconstructs the true board state, determines whose turn it is, and calculates all moves that are legally possible from that state.
For example, suppose the legal-move set is {C5, E3}. The same prefix, D3 C3 C4, is then given to the sequence model, which produces probabilities for possible next tokens and proposes its most likely next move.
Suppose the move that actually occurred in the recorded game was C5, but the model proposes E3. The model's prediction is still counted as valid because E3 is also contained in the legal-move set {C5, E3}.
The model does not need to predict the exact move that occurred in the dataset; it only needs to propose a move that is legal from the reconstructed board state.
If instead the model proposes a move such as C3, which is not in the legal set, the prediction is counted as invalid. Diagnostic 1 repeats this comparison across many prefixes and measures how often the model's proposed next token is a legally valid move.

## Protocol

The primary D1 score is the share of evaluated positions where the model's top predicted move is legal (argmax). This is the `legal_rate` computed in `src/domains/othello.py`.

As a secondary score, at each evaluated position, sample 10 moves at temperature 1.0 using sampling seed 0 and report the share of sampled moves that are legal. This sampled score is stored as an extra field and is not the master D1 score.

Evaluation uses 1,000 games from the TEST split of the model's own training distribution, with data seed 0. Every position that has a recorded next move is evaluated.

As sanity checks, an untrained model is expected to score about 0.15, while Othello GPT scores about 0.999. The untrained baseline will be confirmed when the D1 module runs.

Each model produces one JSON result file at:

`results/D1_next_token/{model_id}.json`

The JSON format follows `src/docs/results_schema.md`. The master D1 score is the argmax legal rate.

## Navigation (domain `maps`, Amendment 5, G4)

The same question on the street map: after a route prefix such as `start destination SW SW SE`, is the model's
top token a turn that exists at the car's current intersection (or `end`, if the car is at the destination)?
Replaying the turns on the map gives the legal next set at every step, as the board does for Othello.

* **Score.** Share of scored positions where the top token is legal, computed by `maps.legal_rate`. Scored
  positions are every prediction of a turn or of `end` (positions 1 to length minus 2), the positions the
  training loss covers. The same computation gives Vafa et al.'s next token rate exactly on their models
  (reproduction, `src/docs/navigation_reproduction.md`).
* **Secondary.** At each scored position, 10 tokens sampled at temperature 1.0 from the whole vocabulary
  (sampling seed 0); the share that are legal. An intersection or padding is never legal.
* **Data.** 1,000 test routes of the model's own training distribution, data seed 0.
* **Reference points.** An untrained model scores about 0.0005 (about 2 legal tokens of 4,589); the anchors of
  Vafa et al. score 0.9996 to 1.0000. For an anchor, `details` also counts the routes with an intersection its
  own tokenizer does not have.

Code: `score_maps` in `src/diagnostics/d1/next_token.py`. Tests: `src/tests/test_d1_maps.py`. Loading, gates
and the pilot results folder: `src/docs/navigation_scoring.md`.

```bash
python src/diagnostics/d1/next_token.py --models maps_transformer_small_shortest_paths_s100 --save
```
