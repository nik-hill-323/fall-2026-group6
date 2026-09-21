# lit notes

## Li et al 2023 - Othello-GPT (ICLR)

- GPT trained only on othello move tokens, never sees a board or the rules
- still predicts legal moves ~99%
- probes on residual stream can recover board state. nonlinear probe works, linear didn't (nanda later showed linear works if you use mine/theirs instead of black/white)
- the important part: they edit the board rep inside the model and the next move prediction changes accordingly -> causal not just correlational
- latent saliency maps, which squares matter for a prediction
- this is our diag 4 + the week 2 reproduction. the intervention exp is basically what we do in week 10

## Vafa et al 2024 - evaluating the world model implicit in a generative model (NeurIPS)

- domain = DFA. two metrics from myhill nerode
  - compression: two seqs land in same state -> model should give same continuations
  - distinction: two seqs in diff states -> model should be able to tell them apart within a short boundary
- experiments: manhattan taxi routes, othello, logic puzzles. training data = shortest path / noisy shortest path / random walk
- models look fine on next token acc and legal move rate but fail the new metrics
- random walk training gives the most coherent world model, interesting
- reconstructed manhattan map has streets that dont exist. ask for a detour and the model breaks
- diag 2 for us + week 2 repro gate. fragility suite = detour idea for every domain

## Vafa et al 2025 - inductive bias probe (ICML)

- take a pretrained model, adapt it on small synthetic datasets made from a "postulated" world model, check if it extrapolates like that world model would
- orbital mechanics, lattice, othello
- orbit model predicts trajectories well but doesnt behave like it knows newton. symbolic regression on the adapted model gives force laws that are nonsense, different law per planet
- point: good at seq prediction != has the world model. learns task heuristics
- diag 3 for us incl the symbolic regression part. SR is stochastic, thats why we need multiple seeds

## Yuan & Sogaard 2025 - revisiting othello WM hypothesis (optional)

- fine tune 7 LLMs on othello sequences
- legal move acc up to 99%, board state probing works on all 7
- representations very similar across models -> they say this means real world model not memorization
- appendix A has the formal def of the hypothesis (env -> internal state mapping)
- opposite conclusion to the vafa papers on the same domain. thats literally our H1. cross-arch similarity is relevant to our architecture axis
