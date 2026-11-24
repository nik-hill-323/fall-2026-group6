# Navigation scoring: how every scorer loads, gates and saves (N0)

Every navigation scorer (D1 to D4, the fragility suite, the whole route measure) shares one foundation,
`src/diagnostics/navigation.py`, so the models, the data and the rules are the same for all of them. This page
says what it does and how to run the scorers. The protocol is Amendment 5 of `reports/preregistration.md`
(G4 to G12); the design is `src/docs/navigation_design.md`.

## Which models

| Kind | Ids | Where the checkpoint is | Where results go |
|---|---|---|---|
| zoo | the 81 rows of `src/docs/models.csv` with domain `maps` | `outputs/zoo/<id>/model.pt` | `results/<diagnostic folder>/<id>.json` |
| pilot | the 9 small models with seed 100 (`maps_<arch>_small_<distribution>_s100`) | `outputs/pilot/<id>/model.pt` | `results/pilot_scoring/<diagnostic folder>/<id>.json` |
| anchor | `maps_vafa_shortest_paths`, `maps_vafa_noisy_shortest_paths`, `maps_vafa_random_walks` | `~/artifacts/vafa_weights/<name>/` (from the N2 reproduction) | `results/<diagnostic folder>/<id>.json` |

* **Pilots are never in any analysis** (G2). Their results go to `results/pilot_scoring/`, a folder the analysis
  code never reads. They are committed, because the fragility pilot rule (G8) is computed from them.
* **Anchors** are description only (G11); they never enter an H statistic.

## What is checked before a score

1. **Checkpoint.** A zoo model or pilot is loaded only if the sha256 of its `model.pt` equals
   `checkpoint_sha256` in its training record (`results/zoo_training/` or `results/pilot_training/`). A copied
   or downloaded file that is not the trained model is refused.
2. **Tags.** Navigation zoo models are refused until the git tags `prereg-v1.5` and `maps-fragility-v1` both
   exist (Amendment 5, gates). Pilots and anchors can be scored before.
3. **Committed code.** A result is saved (`--save`) for a zoo model or pilot only from committed code under
   `src/`, so the commit in its provenance holds the code that made it.
4. **Data.** Every data file a scorer reads is compared with its sha256 in `src/domains/maps_data_manifest.json`.
5. **No mixing.** Othello and navigation ids go in separate commands.

A model that cannot be loaded (no checkpoint on this machine, a checksum that does not match) gets no result
file. It is listed as `NOT SCORED` at the end and the command exits with status 1.

## The same positions everywhere

A model reads `tokens[:, :98]`. The scored positions are 1 to length minus 2 of every route: every prediction
of a turn or of `end`, exactly the positions the training loss covers (design 3.6). The helpers
`correct_at`, `top_tokens`, `legal_token_mask` and `current_intersections` all use them.

## Getting the checkpoints onto one machine

Zoo checkpoints are on S3 (81 files):

```bash
aws s3 cp --recursive s3://dats-capstone/shared/group6/navigation/zoo/ outputs/zoo/ \
    --exclude "*" --include "maps_transformer_*"          # or maps_lstm_*, maps_mamba_*
```

The 9 pilot checkpoints were never uploaded: each one is in `outputs/pilot/` on the EC2 that trained it (p1 on
Siddu's, p2 on Nikhil's, p3 on Viharika's, p4 on Siddu's). The owner uploads them once:

```bash
aws s3 cp --recursive outputs/pilot/ s3://dats-capstone/shared/group6/navigation/pilot/
```

and whoever runs the pilot scoring downloads them into `outputs/pilot/`. The checksum check makes sure each
file is the one that was trained.

## Commands (repo root, conda env `wmconcord`)

```bash
python src/diagnostics/d1/next_token.py            --models <ids> --save   # N4, G4
python src/diagnostics/d2/state_equiv.py           --models <ids> --save   # N5, G5
python src/diagnostics/d3/inductive_bias.py        --models <ids> --save   # N6, G6
python src/diagnostics/d4/representation_probe.py  --models <ids> --save   # N7, G7
python src/fragility/maps.py                       --models <ids> --save   # N8, G8
python src/analysis/whole_route_maps.py            --models <ids> --save   # N12, G9
```

Without `--save` a scorer only prints. All use their preregistered defaults (test split, 1,000 routes, data
seed 0).

## Whole route measure (G9)

`src/analysis/whole_route_maps.py`, results in `results/whole_route_maps/` (pilots in
`results/pilot_scoring/whole_route_maps/`), one JSON per model with `status` ok, or failed with the reason. G12 counts
a model as scored only when it has this file too. Preregistered secondary (Amendment 5, G9):

* **Teacher forced**: D1's 1,000 test routes (data seed 0, batch 256, the same scored positions). The overall legal
  rate is computed again with `maps.legal_rate` and must be identical, or the run stops; the saved D1 score is
  compared too when the file is on the machine (`matches_d1_file`). Reported: the share of routes with every top
  token legal, the legal rate by step, errors per route, the step of the first error.
* **Free play**: from the start and destination of the same 1,000 routes, the model drives itself. Its top token is
  scored at every step; the move played is drawn from its own probabilities over the legal next set (softmax
  restricted to the legal tokens, play seed 0, one draw per live drive per step in drive order), so the route it
  reads is always legal. A drive ends at `end` (only legal at the destination), on arriving at a dead end that is
  not the destination (no prediction is made there: every token would be illegal), or at 96 turns (the top token
  read after 96 turns is scored, then the drive stops). Reported: the share of drives with no illegal top token,
  reached the destination, stuck at a dead end and stopped at 96 turns, the legal rate by step and overall, errors
  per drive and the step of the first error.
* Step s is the prediction after reading s + 1 tokens, the same position p = s as D1 (step 1 predicts the first
  turn, at most step 97).
* The runner runs it as job `G9` (`JOBS=G9 bash src/shellscripts/maps_score_slice.sh <slice>`); it is in the default
  job list, after fragility. The two 1.5B anchors run it with `--batch 32` (memory only).
* The secondary analysis of G9 (Spearman of each diagnostic with the share of drives with no illegal top token,
  configuration bootstrap) belongs to the navigation analysis, not to this script.

## Order (Amendment 5)

1. `python src/diagnostics/d4/representation_probe.py --epoch_rule` (G7): fixes the D4 epoch count with a
   synthetic layer, no model involved. The count goes into `MAPS_EPOCHS` and the D4 doc, committed.
2. Every scorer on the 9 pilots and the 3 anchors (`--save`).
3. `python src/fragility/maps.py --pilot_rule` (G8): its outcome goes into `EXCLUDED_BY_PILOT_RULE` and the
   suite doc, committed; then the tag `maps-fragility-v1`.
4. The zoo, by architecture: transformers, then LSTMs, then Mamba (G12).

## Tests

`src/tests/test_navigation_scoring.py`; the whole route measure: `src/tests/test_whole_route_maps.py`. Shared fakes for every navigation scorer test (a small data folder of
random legal routes on the real map, models whose answers are known): `src/tests/maps_scoring_fixtures.py`.
