# Fragility suite for navigation: maps-fragility-v1 (N8)

H3 asks which diagnostic best predicts whether a model breaks on shifted tasks. This suite defines "breaks" for
navigation, and is frozen with the tag `maps-fragility-v1` before any navigation zoo model is scored, so the
outcome cannot be shaped by a diagnostic score (Amendment 5, G8).

Code: `src/fragility/maps.py`. Tests: `src/tests/test_maps_fragility.py`.

## What it measures

How much more often the model proposes a token that is not legal when the routes are related to its training
but shifted. Every right answer comes from the street map (`src/domains/maps.py`); nothing retrains or adapts
the model. It is built on what Othello fragility v2 taught: every comparison is matched, no score can sit at a
floor, and the route the model drives always stays legal, so one wrong proposal never ends the measurement.

| Variant | What changes | Score and drop |
|---|---|---|
| in distribution | nothing: 1,000 test routes of the model's own distribution | legal rate (the reference) |
| cross distribution | 1,000 test routes of each other distribution (start and destination pairs no model trained on) | legal rate; drop = in distribution minus it (2 drops) |
| detour, random | the model drives itself from a test start to its destination (500 routes, seed 0); with probability p its move is replaced by a random other safe turn | legal rate of its top token over the whole drive; drop = rate at p = 0 minus rate at p, for p = 0.1 and 0.5 |
| detour, adversarial | as above, the replacement is the safe turn the model rates lowest | as above |
| least visited intersections | positions at the least visited intersections of the model's distribution, on 5,000 test routes (seed 0) | drop = mean over those positions of the legal rate of the other positions in the same cell minus their own (1 drop) |

**Detour drives.** A turn is **safe** if it is legal and the destination can still be reached from where it
leads within 96 turns in all (so the route fits the 99 token limit); `end` is safe at the destination. At every
step the model's top token is scored against the true legal next set. The move played is the model's highest
rated safe move. With probability p, never at the destination, it is replaced by another safe turn, and kept
when there is no other. The drive ends at `end` or at 96 turns. Because every move is safe, every drive reaches
its destination; "reached the destination with `end`" is saved as a descriptive number and is 1 by
construction. The undisturbed drive (p = 0) is the same for both modes and is run once. The uniform draws of a
rollout come from one generator seeded 0, drawn only when another safe turn exists (as Othello v2).

**Least visited intersections**, fixed from the data before any model existed
(`src/domains/maps_rare_intersections.json`): per distribution, the intersections with the fewest visits in its
training cache, in order of visits, until they hold 10% of the scored positions of its test split (2,749
shortest paths, 2,816 noisy shortest paths, 1,130 random walks). A rare position is compared with the non rare
positions of the same **cell**: same number of legal next tokens, same "at the destination or not", same block
of 10 turns so far, in the same routes. Positions whose cell has no other position are left out and counted.

## The number written per model

`results/fragility_maps/<model_id>.json` (pilots: `results/pilot_scoring/fragility_maps/<model_id>.json`): every
rate, every drop, the detour drives (legal rate, detours made, mean turns, legal rate by step), the least
visited comparison, and

* `fragility` = the mean of the drops the pilot rule keeps (higher = more fragile);
* `relative_fragility` = `fragility` divided by the in distribution legal rate: the **H3 outcome** (as for
  Othello, Amendment 3);
* `suite_hash`, the model's fields and the provenance (commit, machine, time).

There is no symmetry variant: no symmetry maps the street map onto itself.

## Pilot rule (fixed in G8 before any number was seen)

1. Run the suite on the 9 pilots: `python src/fragility/maps.py --models <the 9 pilot ids> --save`.
2. `python src/fragility/maps.py --pilot_rule`: the standard deviation (sample, ddof 1) of every drop across the
   pilots (a cross distribution drop over the 6 pilots that have it). A drop whose standard deviation is below
   0.001 is reported for every model but left out of `fragility`. Written to
   `results/prereg/maps_fragility_pilot_rule.json`.
3. Write the outcome into `EXCLUDED_BY_PILOT_RULE` in `src/fragility/maps.py` and in the line below, set
   `FROZEN_HASH` in `src/tests/test_maps_fragility.py` to `suite_hash()`, commit, merge, and only then tag:
   `git tag maps-fragility-v1 && git push origin maps-fragility-v1`.

**Pilot rule outcome (2026-11-22, Amendment 6):** no drop is left out. Standard deviation of each drop across the 9 pilots (sample, ddof 1), from `results/prereg/maps_fragility_pilot_rule.json` (pilot run at commit 28d3039):

| Drop | Standard deviation | |
|---|---|---|
| `cross_distribution/noisy_shortest_paths` | 0.048915 | kept |
| `cross_distribution/random_walks` | 0.133003 | kept |
| `detour/random/p=0.1` | 0.095053 | kept |
| `detour/random/p=0.5` | 0.193177 | kept |
| `detour/adversarial/p=0.1` | 0.091992 | kept |
| `detour/adversarial/p=0.5` | 0.187290 | kept |
| `least_visited` | 0.017829 | kept |
| `cross_distribution/shortest_paths` | 0.031390 | kept |

`EXCLUDED_BY_PILOT_RULE = []`; suite hash `47068aee3f4b`.

Zoo models are refused while `EXCLUDED_BY_PILOT_RULE` is None, and until both tags `prereg-v1.5` and
`maps-fragility-v1` exist (`src/docs/navigation_scoring.md`).

## How to run

```bash
python -m pytest src/tests/test_maps_fragility.py -q
python src/fragility/maps.py --models maps_transformer_small_shortest_paths_s100 --save
```
