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

## Scope options

| Option | Models | Training GPU hours |
|---|---|---|
| A. Full matrix, three domains | 243 | 111 |
| B. Othello only, full grid | 81 | 37.0 |
| C. Othello only, Mamba at small and medium | 72 | 15.2 |
| D. Othello only, Mamba at small | 63 | 7.2 |

## Proposal, to be confirmed by the team at the October 6 session

Review 3 asks that any cut be written down now, before results exist, so it is not chosen after
seeing them. Six working sessions remain and only one GPU.

* **H1 to H3 on Othello only.** It is the one domain with data, a tested rules module, a reproduced
  diagnostic, and a frozen fragility suite.
* **Option C for the zoo:** transformer and LSTM at all three scales, Mamba at small and medium.
  72 models, about 15 GPU hours of training, which leaves room for four diagnostics per model.
* **Mamba large is dropped**, not held out. It costs 22 hours and adds 9 models.
* **H4 held out domain: navigation**, as Review 3 suggests, run at reduced size only if H1 to H3 are done by November 17.
* **H4 held out architecture: to be decided.** With Mamba in the development zoo at two scales, the
  held out architecture would be a fourth family or would be dropped from H4.

Until the team confirms, `held_out` stays `tbd` in `models.csv` and the matrix stays at 243 rows.
When confirmed, the matrix is regenerated and this section is replaced by the decision and its date.

## Budget fixes in this change

* `n_train_games` is now 1,280,000 (5,000 steps x 256 games), so no game repeats within a run. It was
  200,000, which showed each game about six times.
* `models.csv` now points `config` and `checkpoint` at `outputs/zoo/{model_id}/`, which is where
  `src/zoo/train.py` writes them.
