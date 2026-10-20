## Demo app

A Streamlit app that lets you pick any model in the zoo and watch each diagnostic run on it, with
the metrics next to it. Built for the Preliminary Presentation (October 20) at the instructor's
request: the baseline and each test shown on a chosen model, not described in words.

Run from the repo root. Every page except D1's model picks works with only the repo (results are
committed); D1 needs the checkpoints (`outputs/zoo/`) and the Othello data (`~/artifacts`, see
`src/shellscripts/fetch_othello_data.sh`):

```
streamlit run demo/app.py
```

Seven tabs, in the order of the demo slide (Model zoo, D1, D2, D3, D4, Fragility, Concordance). Each tab shows
one question, one big number with an (i) tooltip that explains it in plain words, a gauge from the zoo's lowest
to its best model with the published Othello-GPT marked, and one animated picture (a game replay, two routes to
the same board, the four "is this square mine?" questions, a layer by layer X-ray, a rotating board). The full
pipeline of each test is in a closed "How it's computed" box. Click a disc on the Model zoo board to select a
model; a tab can be opened from the URL, e.g. `?model=othello_transformer_large_synthetic_s0&tab=D3`.

| Tab | Big number | Picture |
|---|---|---|
| Model zoo | legal move rate of the selected model | 81 models as discs, coloured by legal moves or D1 to D4 |
| D1 | share of legal top moves | a real game, ▶ play; the model's pick when its checkpoint is here |
| D2 | 1 − mean of the two errors | two real move orders reaching the same board, ▶ play |
| D3 | trained minus random copy | a real position with the four questions; trained vs random accuracy |
| D4 | probe selectivity, plus "does it use it?" | probe accuracy per layer; one disc flipped inside the model |
| Fragility | relative fragility v2 | a rotating board; drop per kind of shift |
| Concordance | Kendall W raw → skill removed; H2 and H3 answers | agreement grid, raw or with skill removed |

The two published Othello GPTs appear in the picker when their weights are under `~/artifacts/othello_gpt_tl`
(D1 live needs `transformer-lens` for them). The app only reads files; it never writes results.

Status, October 20: every page works on the full scored zoo (81 of 81) and the analysis of #122.

The `fig/` folder is for the one minute project video (`demo.gif`) and demo screenshots.
