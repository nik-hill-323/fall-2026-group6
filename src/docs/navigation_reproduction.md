# Navigation reproduction gate: Vafa et al. (2024), Table 1

Status: **passed** on Nov 8, 2026, code commit d46aa20 (issue #142). Design: src/docs/navigation_design.md,
section 6.3. Script: src/diagnostics/maps_reproduce_vafa.py. Results: results/reproduction/maps/.

## Why this gate exists

Before any navigation zoo model is scored, our code must reproduce published numbers on the models those
numbers came from, as the Othello gate did (src/docs/d4_reproduction.md). Vafa et al. released the three
navigation models of their Table 1, trained on routes through Manhattan: shortest paths (GPT 2, 12 layers
x 768, 89.3M parameters), noisy shortest paths and random walks (48 layers x 1600, 1.48B parameters each).
If our code gives their numbers on their models, our next token scoring, our token translation and our
probe recipe are right, and the same code can then be trusted on our own zoo.

## Tolerances (fixed in the script before anything was run)

| Check | Rule |
|---|---|
| Next token test | at least 0.99 for every model (the paper rounds to 1.00; its footnote 3 says no model is perfect) |
| Our D1 path against Vafa's computation, same routes | within 0.001 for the two models on the canonical map |
| Current state probe | within 0.02 of the published value |

## Results

| Model | Next token: ours (published) | Our D1 path minus Vafa's | Probe: ours (published) |
|---|---|---|---|
| shortest paths | 0.9996 ± 0.0001 (1.00) | 0.0000 on 946 routes | 0.9089 ± 0.0007 (0.91) |
| noisy shortest paths | 0.9998 ± 0.0001 (1.00) | 0.0000 on 1,000 routes | 0.9142 ± 0.0005 (0.92) |
| random walks | 1.0000 ± 0.0000 (1.00) | 0.0000 on 979 routes | 0.9915 ± 0.0002 (0.99) |

± is the standard error. Every check passes. Our D1 path does not just come close to Vafa's computation; it
gives the same number.

| Model | Next token positions | Probe positions | Intersections in the probe | Best epoch of the probe |
|---|---|---|---|---|
| shortest paths | 39,641 | 981,848 | 4,215 | 0.9112 |
| noisy shortest paths | 52,539 | 1,323,791 | 4,110 | 0.9155 |
| random walks | 50,360 | 1,253,150 | 4,573 | 0.9947 |

The probe score is the accuracy after the last of the 100 epochs, as in his code; the best epoch is shown only
to make clear the result does not depend on that choice.

Context row, the paper's untrained transformer (not part of the gate):

| Model, untrained | Next token: ours (published) | Probe: ours (published) |
|---|---|---|
| shortest paths | 0.0000 (0.03) | 0.0842 (0.10) |
| noisy shortest paths | 0.0006 (0.03) | 0.1074 (0.10) |
| random walks | 0.0000 (0.03) | 0.0927 (0.10) |

The untrained probe matches the paper: a linear probe on a random network still reads about 10% of
intersections, which is why D4 always reports selectivity against a control task. The untrained next token
rate depends on the exact random starting weights; ours come from transformers 5.14.1 with seed 42, his
from transformers 4.41.2, so the two random networks differ. Any rate near 0 makes the same point.

## What was computed, and how it follows Vafa's code

Both tests copy his code step by step (next_token_test.py, probe_test.py, utils.py and model.py of
github.com/keyonvafa/world-model-evaluation) on his data, his tokenizers and his weights.

* **Next token test.** heldout_sequences.txt read with `f.read().split("\n")`; 1,000 routes drawn with
  `np.random.RandomState(42).choice(len, 1000, replace=False)`; batches of 128 padded on the right; at
  every position that predicts a turn or `end`, the top token is legal if it is a turn of the current
  intersection, or `end` at the destination.
* **Our D1 path, same routes.** The same routes written on the canonical map (maps_build.Translator), then
  maps.encode, maps.build, the HFGPT2Model translation of our tokens into his and of his scores back onto
  ours, and maps.legal_rate: the exact code the zoo will be scored with. Routes our domain does not keep are
  left out of both sides: shortest path routes over one of the 39 streets the canonical map lacks (54 of
  1,000), random walk routes longer than 99 tokens (21 of 1,000). No kept route had a word the model's own
  vocabulary lacks.
* **Probe.** 25,000 routes drawn the same way from train_sequences.txt (the 14.6 GB random walks file is
  streamed with the same indices, not read into memory); the last layer (after GPT 2's final layer norm);
  the start at position 0 and the intersection after turn j at position j + 1; labels encoded over all
  intersections seen; a random 80/20 split of positions; one linear layer, Adam lr 0.01, cross entropy,
  batch 2,048 in order, 100 epochs; accuracy on the 20% after the last epoch.

## Where we differ from his code, and why it does not matter

| Difference | Why |
|---|---|
| The probe's 80/20 split is seeded (seed 0) | His is not seeded, so no run can repeat his exactly; seeding makes ours repeatable |
| Weights extracted from his Lightning checkpoints and loaded into transformers 5.14.1 (he used 4.41.2) | Checked on test checkpoints written with 4.41.2: logits and last layer identical to the last bit |
| Attention computed the plain ("eager") way | Same numbers as his version; no faster kernel that could change rounding |
| An empty entry (after a final newline) is skipped and counted | His code would stop with an error on it; none was drawn in any of these runs (all six counts are 0) |
| The shortest paths model is scored on its own map in his test | Its map differs slightly from the canonical one; on the canonical map it is described, not gated |

## Provenance

* Code: commit d46aa20 on branch siddu/vafa-reproduction; every result file records it.
* Weights: ~/artifacts/vafa_weights/{shortest_paths,noisy_shortest_paths,random_walks}/ on Siddu's EC2,
  with checksums.json (sha256 of model.pt, config.json, vocab.json).
* Environment: torch 2.13.0, transformers 5.14.1, NVIDIA A10G.
* Runtime: weights about 90 s per 1.5B checkpoint; next token test about 2 min per model; probe about 15 min
  per 1.5B model.

## What this licenses

* The next token scoring of D1 on the maps domain (maps.build, maps.legal_rate) and the anchor wrapper are
  verified against the published implementation, to the exact number.
* The three models are available to every diagnostic as anchors: `interface.vafa_navigation(which)`, model
  ids `maps_vafa_shortest_paths`, `maps_vafa_noisy_shortest_paths`, `maps_vafa_random_walks`. Our D1 to D4 on
  them are reported as description only, never in the zoo statistics.
* The probe recipe here is Vafa's. Our D4 (design section 4) differs on purpose: every layer, a control task,
  validation routes to fit and test routes to score. The agreement here shows the features and labels line
  up; D4 itself is validated on the zoo through selectivity and the causal check.

## How to rerun

On the EC2, from the repo root, in tmux:

    python src/diagnostics/maps_reproduce_vafa.py prepare
    python src/diagnostics/maps_reproduce_vafa.py next-token --untrained
    python src/diagnostics/maps_reproduce_vafa.py probe --untrained
    python src/diagnostics/maps_reproduce_vafa.py report

The last three refuse to run on uncommitted code. Total time is about one hour.
