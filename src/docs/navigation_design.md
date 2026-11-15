# Navigation design (domain `maps`)

Status: frozen by Amendment 5 of `reports/preregistration.md` (tag `prereg-v1.5`). Where this file and
Amendment 5 differ, Amendment 5 holds. No navigation zoo model is trained before that tag exists.

This file fixes how navigation plugs into WM-Concord: the data, the map, the vocabulary, the splits,
what the models are trained on, and what D1 to D4, the fragility suite and the whole route measure mean
here. Every navigation ticket (N1 to N11) codes against it. Where Othello already fixed a rule, the
navigation rule is the same unless this file says otherwise and why.

## 1. The domain

Vafa et al. (2024) turned taxi style routes on the Manhattan street map into token sequences:
`start destination turn turn ... end`, for example `42436516 42449938 SW SW SE ... end`. A turn is one
of 8 compass directions (N NE E SE S SW W NW) and moves the car along one street segment to the next
intersection. A model trained on these sequences never sees the map. If it has a world model, it knows
where the car is after any list of turns and which turns exist there. Replaying the turns on the map
gives the exact ground truth at every step.

We use Vafa's data and released checkpoints, and our own D1 to D4 and fragility suite, as for Othello
(team decision, Nov 6). Vafa's compression and distinction tests are not run on our zoo.

## 2. What the data checks found (Nov 7)

Five read only checks on the EC2. Scripts and logs: `~/nav_checks/check_nav_data{,2,3,4,5}.{py,log}`.

### 2.1 The files

| | shortest paths | noisy shortest paths | random walks |
|---|---|---|---|
| Train routes | 2,932,675 | 30,599,313 | 90,646,871 |
| Held out routes | 29,646 | 307,917 | 908,110 |
| Distinct held out start and destination pairs | 29,646 | 30,151 | 48,130 |
| Held out pairs also in the same distribution's train | 0 | 0 | 0 |
| Every held out route replays legally on its own map | yes | yes | yes |
| Longest route in the full train file (tokens) | 99 | 99 | 102 |
| Train routes of 100 to 102 tokens | 0 | 0 | 2,583,557 (2.9%) |
| Held out routes that revisit an intersection | 0 | 0 | 92.7% |
| Held out routes at the destination before their final step | 0 | 0 | 42.5% |
| Held out routes whose destination is a dead end | 0.12% | 0.17% | 0.19% |

### 2.2 The map

* **Two maps.** Noisy shortest paths and random walks share one map (identical turn tables). Shortest
  paths uses a slightly different one.
* **4,579 intersections**: 4,562 with at least one turn, plus **17 dead ends** (a turn leads into them,
  none leads out). Vafa's turn table lists only the 4,562, so the dead ends appear only as the place a
  turn leads to, or as a destination. No route starts at a dead end, and no route reaches a dead end
  unless it is the destination. 9,705 street segments.
* **One intersection renamed**: 42438066 on the shortest paths map is 7732571914 on the other map. Same
  neighbours in and out; in OpenStreetMap the two points are about 2 to 3 m apart.
* **39 streets differ for real**: the shortest paths map has 39 segments the other map does not have.
  6.1% of shortest path routes use one of them. Translating routes by their intersections rescues only
  the 118 routes that pass the renamed intersection.
* **The map is not one connected piece**: 106 strongly connected pieces, the largest with 4,473
  intersections; 505,523 ordered pairs of intersections have no route between them. Between connected
  intersections the turn distance has median 42, 99th percentile 93, maximum 124.
  12 held out pairs have no route on the canonical map; N1 checks that every one of them belongs to a
  dropped shortest path route (every kept route replays, so its pair is connected by construction).
* **Vafa's tokenizers miss some intersections** (at least 80 for shortest paths, 85 for noisy, 2 for
  random walks) and turn any unknown word into padding.

### 2.3 Coordinates

OSMnx 2.1.1 with Vafa's settings (Manhattan, network type drive, OpenStreetMap as of 2024-05-05)
returns coordinates for **all 4,579 intersections**, and **all 9,705 turn labels equal the compass
bearing** between their two intersections. So the coordinates belong to exactly this map.

### 2.4 Splits across distributions

Each held out file is unseen by its own training file, but not by the others: 95% of the noisy held out
pairs are in the shortest paths training data, and about half of the random walk held out pairs are in
the other two training sets. Noisy shortest paths has about 10 held out routes per pair.

## 3. Decisions

### 3.1 One map: the canonical map

All zoo models live on the noisy and random walks map, the **canonical map**: two of the three datasets
and both 1.5B anchors use it, and its coordinates are verified (2.3).

* Shortest path routes are rewritten on the canonical map by following their intersections (the
  renamed intersection gets its canonical id). A route is kept only if every segment it uses exists on
  the canonical map. Kept: train 2,753,554 of 2,932,675 (93.89%), held out 27,881 of 29,646 (94.05%).
* Noisy and random walk routes are kept as they are (all are legal on the canonical map).
* Limitation, stated in the report: the dropped 6.1% of shortest path routes are the ones that used one
  of the 39 streets the canonical map does not have, so they are not a random sample.
* Exception: for the reproduction (N2) the shortest paths anchor is scored on its own map and data.

### 3.2 Map files in the repository

The domain does not read Vafa's pickled tokenizers at run time. N1 writes two small files once, from
the EC2 data, and commits them with their sha256:

* `src/domains/maps_edges.csv`: `node, turn, next_node`, the 9,705 segments of the canonical map.
* `src/domains/maps_coordinates.csv`: `node, lat, lon` for the 4,579 intersections (OpenStreetMap as of
  2024-05-05). The download depends on an outside service, so the file is the record.

Tests and CI then run on the real map without the 40 GB of data.

### 3.3 Vocabulary, shared by all three distributions

| Token ids | Meaning |
|---|---|
| 0 | padding |
| 1 to 4,579 | the intersections of the canonical map, in increasing OpenStreetMap id order |
| 4,580 to 4,587 | the turns N NE E SE S SW W NW |
| 4,588 | end |

`VOCAB = 4,589`. Anchors keep their own tokenizers; `src/diagnostics/interface.py` translates, and a
test checks every token of every anchor tokenizer round trips.

### 3.4 State and legal next set

* **State** after a prefix = (current intersection, destination). Nothing else changes what can
  legally come next, so two prefixes with the same state are Myhill Nerode equivalent.
* **Legal next set** = the turns that exist at the current intersection, plus `end` when the current
  intersection is the destination. At a dead end that is the destination, only `end`. A route may pass
  through its destination and continue. This is the rule of Vafa's next token test.
* At a dead end that is not the destination the legal next set is empty. No data route ever gets
  there; free play can (5.2), and then the route stops.

### 3.5 Splits by pair, shared by all distributions

1. Take the union of the start and destination pairs of the three held out files, after the rename:
   107,331 pairs.
2. Split that union 50/50 into validation pairs and test pairs, seed 0, once, for all distributions.
3. A distribution's validation and test routes are its kept held out routes whose pair is in that half.
4. Remove every route whose pair is in the union from all three training sets: 42,782 shortest path
   routes (1.5%), 411,521 noisy routes (1.3%), 665,057 random walk routes (0.7%), counted before 3.1.

No zoo model ever trains on a validation or test pair, in any distribution, so the test across
distributions is a test on unseen routes. N1 saves both pair lists and Amendment 5 records their sha256.

### 3.6 Training cache and training

* **One maximum length for all three distributions: 99 tokens** (start, destination, at most 96 turns,
  end). Shortest path and noisy routes never exceed 99; random walk routes of 100 to 102 tokens (2.9%)
  are dropped from every split. Why: a transformer only trains the position slots its training routes
  reach. If random walks kept 102, a shortest paths model scored on random walk routes (cross distribution
  fragility) would read position slots it never trained, the Othello move 60 problem across distributions.
  With one limit, every model is trained on every position any scorer reads.
* 2,000,000 routes per distribution, drawn with seed 0 from the training routes left after 3.1, 3.5 and
  the length limit (count the eligible routes in one pass, draw the indices, collect them in a second
  pass, so the draw does not depend on reading order or memory). Stored as `uint16`, width 99, about
  0.4 GB each. Caches and pair files go to `s3://dats-capstone/shared/group6/navigation/`, with sha256 in
  `src/domains/maps_data_manifest.json`.
* Budget, schedule and sizes exactly as Othello: 5,000 steps of 256 routes, `SCALE_CONFIG`, same
  learning rate, warmup and weight decay.
* **Loss only on turns and `end`**: the targets at token index 2 and later. The destination token is a
  random intersection, so predicting it is noise. Padding targets are ignored as before.
* The model reads at most 98 tokens (`N_CTX`) and predicts tokens 2 to 98. **Every position a diagnostic,
  the fragility suite or the whole route measure scores is a position the training loss covers**: input
  positions 1 to (route length minus 2), at most 97. This is the Othello move 60 lesson, enforced by a test
  (`train_targets` gives a target to exactly the scored positions).

### 3.7 Domain id and file names

`run_matrix.csv` and `models.csv` already name the domain `maps` (81 rows, for example
`maps_transformer_small_shortest_paths_s0`, distributions `shortest_paths`, `noisy_shortest_paths`,
`random_walks`). We keep that id, so nothing in the run matrix changes, and name the module
`src/domains/maps.py` so code can load a domain by its id. Reports say "navigation".

### 3.8 Admission

As for Othello: admitted if the validation legal rate (D1 rule, 1,000 validation routes, data seed 0)
is at least 0.50. An untrained model scores about 0.0005 (about 2 legal tokens out of 4,589). The pilots
(6.4) confirm 0.50 separates models that learned from models that did not, before it is frozen.

## 4. The four diagnostics on navigation

Every diagnostic uses the model's own training distribution, data seed 0, test routes for the score and
validation routes for anything fitted or chosen, as for Othello. Scored positions are always those of 3.6.

### D1 next token validity (N4)

Share of scored positions where the model's top token is in the legal next set, on 1,000 test routes.
Secondary: 10 samples per position at temperature 1.0, sampling seed 0. This is Vafa's next token test
on our splits.

### D2 state equivalence (N5)

The Othello computation with the navigation state.

* **Accepted set** of a prefix: the model's k highest scoring tokens, k = size of the legal next set.
* **Same state pairs**: two different prefixes that reach the same (intersection, destination). Test
  routes to the same destination merge before they arrive, and random walks revisit intersections.
  Up to 2,000 pairs, seed 0; if fewer than 500 exist, more test routes are added 1,000 at a time, up to
  5,000.
* **Different state pairs**: two prefixes with the same number of turns whose legal next sets differ.
  Up to 2,000 pairs, seed 0. Boundary = tokens legal after exactly one of them.
* Compression error, distinction error, score = 1 minus the mean of the two, standard errors and pair
  counts, as for Othello.
* Limit, as for Othello: next token only. Two intersections with the same turns are different states
  that D2 cannot tell apart; reported, not fixed.

### D3 inductive bias probe (N6)

Same protocol and constants as Othello (copy of the model plus a new head on the last layer, labelled
examples from validation routes, one random scored position per route, random weight baseline, score
= mean over tasks and seeds of pretrained minus random balanced accuracy). The questions are about the
current state, so only replaying the turns on the map answers them:

| Task | Question | Shortcut it is checked against | Status |
|---|---|---|---|
| T1 | Is the destination at most h = 10 turns away on the canonical map? | number of turns so far | kept |
| T2 | Is the destination at most r = 823.9 m away in a straight line? | number of turns so far | kept |
| T3 | Is the current intersection north of the destination (higher latitude)? | start north of destination | **dropped** |
| T4 | Is the current intersection east of the destination (higher longitude)? | start east of destination | kept |

* h and r are the medians over 30,000 validation positions (10,000 routes per distribution, one random scored
  position each, seed 0), computed from the data alone (`src/domains/maps_tasks.py`).
* Drop rule, fixed before computing: a task is dropped if, in any distribution, its positive rate is outside
  0.10 to 0.90 or its shortcut reaches 0.90 balanced accuracy. T3's shortcut reaches 0.922 (shortest paths) and
  0.908 (noisy), so D3 uses T1, T2 and T4. All numbers: Amendment 5, G6.
* The turn distance is always finite at a scored position: the route itself goes on to the destination.
* **Shortcut check**, as the Othello occupancy shortcut: agreement of the adapted model's answers with the
  shortcut rule, reported next to the score.
* **Symbolic regression arm** (descriptive, as Othello): PySR fits the adapted model's margin from current,
  start and destination coordinates and the number of turns so far. Law recovered = the formula depends
  mainly on current minus destination with the right sign; shortcut = it depends on start minus destination.
* The shortcut agreement of every adapted model is reported next to its score (T4's data shortcut is 0.88).

### D4 representation probe (N7)

The Othello protocol with the board replaced by the current intersection.

* **Probe**: per layer, one linear classifier reads the current intersection (4,579 classes) from the
  standardized vector at each scored position. 10,000 validation routes fit, 1,000 more choose the layer,
  1,000 test routes give the score. MLP probe as in Amendment 2. Epochs: the smallest of 2, 5, 10 and 20 at which
  the probe reads a layer holding the true intersection at 0.99 or more (no model involved; Amendment 5, G7).
* **Control task** (Hewitt and Liang 2019): each position gets a type, (token just read, number of turns
  so far); each type gets a fixed random intersection drawn from the real mix of intersections.
  Selectivity = probe accuracy minus control accuracy; score = selectivity at the best layer, test routes.
* **Causal check**, Amendment 2 B2 with the same settings (middle to last layer, step 0.05 x sqrt(width),
  confidence 0.99, at most 100 steps, random control edit of the same length, 2,000 bootstrap resamples):
  one case per test route, at the first scored position (random order, seed 0) that has a neighbouring
  intersection, one turn away, whose legal next set differs from the current one and is not empty. The
  edit moves the probe's reading to that neighbour. Boundary accuracy with k = size of the edited legal
  next set. Same pass rule and labels (selectivity 0.10; causal score 0.20 with interval above 0). The
  settings are not retuned for navigation; how they behave on Vafa's anchors is reported.

## 5. Fragility and the whole route measure

### 5.1 Fragility suite (N8), frozen with tag `maps-fragility-v1` before any navigation model is scored

Built on what Othello v2 taught: no score that can sit at the floor, every comparison matched, the route
played always stays legal so one error does not end the measurement.

| Variant | What changes | Score and drop |
|---|---|---|
| in distribution | nothing: 1,000 test routes of the model's own distribution | legal rate (reference) |
| cross distribution | 1,000 test routes of each other distribution (unseen pairs, 3.5) | legal rate; drop = in distribution minus it |
| detour, random | the model drives from a test start to its destination (500 routes, seed 0) | legal rate of its top token over the whole drive; drop = rate at p = 0 minus rate at p, for p = 0.1 and 0.5 |
| detour, adversarial | as above, the detour is the safe turn the model rates lowest | as above |
| least visited intersections | positions at the least visited intersections of the model's distribution (5,000 test routes, seed 0) | drop = legal rate at matched positions (same number of legal tokens, same "at destination", same block of 10 turns) minus legal rate at those positions |

**Detour drives** (Vafa's Table 2, made safe and scored per step):

* A turn is **safe** if it is legal and the destination can still be reached from the next intersection
  within the length limit (96 turns in all, so the route fits in 99 tokens). `end` is safe at the destination.
* At each step the model's top token is scored against the true legal next set. The move played is the
  model's highest rated safe move; with probability p (never at the destination, as in Vafa) it is
  replaced by a random safe turn other than the model's choice (random) or by the safe turn the model
  rates lowest (adversarial). If only one safe turn exists, it is played.
* The drive ends at `end` or at the length limit. "Reached the destination with `end`" is saved as a
  descriptive number next to the score.

**Least visited intersections** (`src/domains/maps_rare_intersections.json`): per distribution, the intersections
with the fewest visits in its training cache, in order of visits, until they hold 10% of the scored positions of
its test split (2,749, 2,816 and 1,130 intersections; 18,418, 26,891 and 24,785 positions in 5,000 test routes).
A first definition, the 10% of intersections with the fewest visits, gave only 6 and 38 positions for shortest
and noisy shortest paths, so it was replaced before any model existed.

`fragility` = mean of the drops; H3 uses **relative fragility** = fragility divided by the in
distribution legal rate, as for Othello. No symmetry variant: no symmetry maps the street map onto itself.
A variant whose drop has a standard deviation below 0.001 across the 9 pilots is reported but left out of
`fragility` (decided on the pilots, before any zoo model is scored).

### 5.2 Whole route measure (preregistered as secondary for navigation, as the plan says)

1. Teacher forced: on D1's 1,000 test routes, the share of routes in which every top token is legal, and
   the legal rate by step. The overall rate must equal D1 (checked).
2. Free play: from 1,000 test start and destination pairs (seed 0) the model drives itself. At each step
   its top token is scored; the move played is drawn from its own probabilities over the legal next set
   (seed 0), as in the Othello whole game measure v3. The drive ends at `end`, at the length limit, or at a
   dead end that is not the destination. Reported: share of drives with no illegal top token, share that
   end at the destination, legal rate by step, step of the first error, share stuck at a dead end.

## 6. Before any navigation model is trained

### 6.1 Review

This file reviewed by Nikhil, then shown to Dr. Jafari (Nov 10).

### 6.2 N1: the domain, the data build and their tests

Files: `src/domains/maps.py` (map, vocabulary, state, legal next set, `build`, `train_targets`,
`legal_rate`, distances, coordinates, safe moves, data loading), `src/domains/maps_build.py` (export of the
map files, splits, caches, manifest), `src/domains/maps_edges.csv`, `src/domains/maps_coordinates.csv`,
`src/domains/maps_data_manifest.json`, `src/tests/test_maps_domain.py`, this file.

The Othello lessons, each turned into a test or a check in the build:

* every cached and split route replays legally on the canonical map and ends at its destination;
* no validation or test pair appears in any training cache; validation and test pairs do not overlap;
* vocabulary round trip; padding and intersections never count as a legal next token;
* the training targets and every scorer cover exactly the same positions (3.6), and nothing beyond `N_CTX`;
* the map files match the data checks (4,579 intersections, 9,705 segments, 17 dead ends, the out degree
  counts, 505,523 unreachable ordered pairs, turn distance median 42 and maximum 124) and every turn label
  matches the compass bearing of the coordinates;
* the map files, pair files and caches match the sha256 in the manifest;
* the build refuses to run on uncommitted code and records the commit;
* the same seed gives the same cache; dead ends, passing through the destination, `end` only as the last
  token, and the renamed intersection are covered by fixed cases;
* each test was checked by planting the bug it targets and seeing it fail.

Left to the tickets that use them: anchor token translation (N2), the same batch size as D1 for
evaluation and sampled rollouts (N8 and the whole route measure).

### 6.3 N2 reproduction on Vafa's three checkpoints, on their own maps and tokenizers

* **Token translation** between their tokenizers and ours, with every token of every anchor tokenizer
  checked to round trip.
* **D1** against their next token test (1.00 for all three). Their test reads the held out file with
  `f.read().split("\n")` (a final newline adds one empty entry, which counts in the length) and draws
  1,000 routes with `np.random.RandomState(42).choice(len, 1000, replace=False)`. We copy that exactly,
  including their tokenizer, so unknown intersections become padding as in their run.
* **Probe** against their Table 1 (0.91 shortest paths, 0.92 noisy, 0.99 random walks). Their recipe:
  25,000 training routes drawn with `RandomState(42)` from `train_sequences.txt` read the same way;
  last layer only (after GPT 2's final layer norm); labels = the start at position 0 and the
  intersection after turn j at position j + 1 (the destination position is skipped); one linear
  classifier, Adam lr 0.01, batch 2,048, 100 epochs; a random 80/20 split of positions. Their split is
  not seeded; we seed it (seed 0) and report it. The 14.6 GB random walk train file is streamed, not read
  into memory, with the same indices.
* The 1.5B checkpoints hold optimizer state (17 GB): load on CPU, keep only the weights.
* Our own D1 to D4 on the anchors are then reported as description only.

### 6.4 Pilots

One small model per architecture per distribution (9 models, seed 100), saved under `outputs/pilot/`, never in
any analysis. They are trained before any zoo model is trained (throughput, training and the admission number
checked) and scored before any zoo model is scored (every scorer end to end, the fragility rule of 5.1). The D3
tasks, the thresholds and the least visited intersection sets were fixed from the data alone before any model.

### 6.5 Amendment 5, tag `prereg-v1.5`

Fixes every number in this file, sets the order of training and scoring (transformers, then LSTMs, then Mamba;
primary analysis on all 81) with a cutoff of 2026-11-22 (Instructor Review #8), and adds, for navigation:

* the whole route measure (5.2) as a preregistered secondary measure;
* H3 with the training distribution as a control (the Othello link from D4 to fragility came from the
  championship models), as a preregistered secondary analysis next to the unchanged primary one, so
  Othello and navigation stay comparable.

Then the zoo.

## 7. Also to update

* The report's navigation paragraph says the first step is a reproduction of Vafa's state equivalence
  numbers. Since Nov 6 it is D1 and the probe (6.3).
* Tickets: N1 is `maps.py` (not `navigation.py`).

## 8. Decisions taken in this draft (for the review)

1. Canonical map, shortest path routes rewritten by intersections, 6.1% dropped (3.1).
2. Module `maps.py`, domain id `maps` (3.7).
3. D3 with all four tasks, coordinates from OpenStreetMap as of 2024-05-05 committed with the repo (4).
4. Rare intersections variant kept (5.1).
5. Detours restricted to safe turns, as in Vafa's code (5.1).
