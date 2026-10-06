# Zoo throughput and compute projection

Instructor Review 2 and 3: commit the measured throughput and the projected GPU hours next to
`run_matrix.csv`, and decide the scope with numbers. This is that file.

## Measured

NVIDIA A10G (g5.2xlarge), 2026-10-02, `src/shellscripts/throughput.sh`. Each run: 300 timed steps at
batch 256 games, extrapolated to the fixed budget in `src/component/run_matrix.py` (5,000 steps).
Mamba is the plain PyTorch `mambapy` build, because `mamba_ssm` does not compile on the instance.
Mamba large runs in chunks of 64 games to fit GPU memory.

| arch | scale | layers x width | params | s/step | min per model |
|---|---|---|---|---|---|
| transformer | small | 4 x 128 | 0.82M | 0.018 | 1.5 |
| transformer | medium | 6 x 256 | 4.79M | 0.057 | 4.7 |
| transformer | large | 8 x 512 | 25.31M | 0.193 | 16.1 |
| lstm | small | 4 x 128 | 0.54M | 0.007 | 0.6 |
| lstm | medium | 6 x 256 | 3.19M | 0.019 | 1.6 |
| lstm | large | 8 x 512 | 16.87M | 0.071 | 5.9 |
| mamba | small | 4 x 128 | 0.48M | 0.212 | 17.7 |
| mamba | medium | 6 x 256 | 2.66M | 0.636 | 53.0 |
| mamba | large | 8 x 512 | 13.63M | 1.748 | 145.7 |

Legal move rates from these 300 step runs are not final and are not reported here.

## Projection: training only

One domain has 9 models per architecture and scale (3 distributions x 3 seeds).

| Per domain | GPU hours |
|---|---|
| transformer, all scales | 3.3 |
| LSTM, all scales | 1.2 |
| Mamba, all scales | 32.5 |
| **one domain, 81 models** | **37.0** |
| **three domains, 243 models** | **111** |

Mamba is 88% of the training cost. Mamba large alone is 21.9 hours per domain.

This is training time only. Diagnostics come on top. The only measured one so far is the D4
reproduction: 206 seconds for one large model with 100,000 games. D2 and D3 are not costed yet.
Navigation and the interpreter are assumed to cost the same per step as Othello, which is not measured.

## Scope decision (team, 2026-10-05, written before any model is scored)

* Full matrix kept: 3 domains x 3 architectures x 3 scales x 3 distributions x 3 seeds = 243
  models, 111 training GPU hours. Mamba large is kept.
* Order: Othello first. All 81 Othello models are trained, and D1 and D4 are run on them, before
  the Preliminary Presentation (2026-10-20). D2 and D3 on Othello follow. Navigation comes after
  2026-10-20, then the interpreter.
* Compute: training is split across the four g5.2xlarge instances, one per team member: about
  9 GPU hours each for Othello and about 28 each for all three domains. The 9 Mamba large Othello
  models (about 2.4 hours each) are spread so no instance gets more than three.
* H4 held out domain: the interpreter. Its models are trained and scored only after the H1 to H3
  analysis is fixed on Othello and navigation.
* H4 held out architecture: none. With three architecture families, holding one out would remove a
  third of the models from every H1 to H3 analysis and leave only two families for H2.
* If training falls behind: any cut is decided by the team, written here with its date, and made
  before the affected models are scored.

## Budget fixes in this change

* `n_train_games` is now 1,280,000 (5,000 steps x 256 games), so no game repeats within a run. It was
  200,000, which showed each game about six times.
* `models.csv` now points `config` and `checkpoint` at `outputs/zoo/{model_id}/`, which is where
  `src/zoo/train.py` writes them.
