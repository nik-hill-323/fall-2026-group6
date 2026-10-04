#!/usr/bin/env bash
# First zoo models (Instructor Review 3, blocker): one model per architecture on
# synthetic Othello at the full training budget, with its legal move rate (D1).
#
# Run on the EC2 from the repo root with the wmconcord env active:
#
#     bash src/shellscripts/first_zoo_models.sh
#
# About 20 minutes at scale small (transformer 1.5, LSTM 0.6, Mamba 17.7).
# Writes, and these two are meant to be committed:
#     results/D1_next_token/{model_id}.json     full result of each run
#     src/docs/zoo_first_models.md              one table row per model
# Checkpoints stay in outputs/zoo/{model_id}/ (not in git).
#
# This script does NOT run the fragility suite. That is scored only after the
# suite is merged and tagged fragility-v1.

set -euo pipefail
cd "$(dirname "$0")/../.."

SCALE="${SCALE:-small}"
ARCHS="${ARCHS:-transformer lstm mamba}"
SEED="${SEED:-0}"
DOC=src/docs/zoo_first_models.md

# Use whichever interpreter the active environment provides.
PY="$(command -v python || command -v python3)"

# This is a GPU job. Stop early on a machine without one, so a laptop run cannot
# write a table of failures into src/docs. Set ALLOW_CPU=1 to override.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ "${ALLOW_CPU:-0}" != 1 ]; then
    echo "No CUDA GPU found. Run this on the EC2 instance (or set ALLOW_CPU=1)." >&2
    exit 1
  fi
fi

mkdir -p results/D1_next_token

{
  echo "# First zoo models: synthetic Othello, scale ${SCALE}, seed ${SEED}"
  echo
  echo "Trained with \`src/zoo/train.py\` at the full budget in \`src/component/run_matrix.py\`"
  echo "on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo cpu), $(date -u +%F), commit $(git rev-parse --short HEAD)."
  echo "Legal move rate is Diagnostic 1: the share of the model's top next moves that the rules allow, on 1,000 validation games."
  echo
  echo "| model_id | params | steps | final loss | next token acc | legal move rate (D1) | train minutes |"
  echo "|---|---|---|---|---|---|---|"
} > "$DOC"

for arch in $ARCHS; do
  id="othello_${arch}_${SCALE}_synthetic_s${SEED}"
  extra=()
  if [ "$arch" = mamba ] && [ "$SCALE" = large ]; then extra+=(--micro_batch 64); fi
  if [ -n "${N_GAMES:-}" ]; then extra+=(--n_train_games "$N_GAMES"); fi
  echo "=== $id ==="
  if "$PY" src/zoo/train.py --arch "$arch" --scale "$SCALE" --distribution synthetic \
        --seed "$SEED" --log_every 500 ${extra[@]+"${extra[@]}"}; then
    cp "outputs/zoo/$id/results.json" "results/D1_next_token/$id.json"
    "$PY" - "$id" >> "$DOC" <<'PY'
import json, sys
r = json.load(open(f"results/D1_next_token/{sys.argv[1]}.json"))
print(f"| {r['model_id']} | {r['n_params'] / 1e6:.2f}M | {r['train']['steps_run']} | "
      f"{r['train']['final_loss']:.3f} | {r['eval']['next_token_acc']:.3f} | "
      f"{r['eval']['legal_move_rate']:.4f} | {r['train']['wall_sec'] / 60:.1f} |")
PY
  else
    echo "| $id | failed | | | | | |" >> "$DOC"
  fi
done

echo
cat "$DOC"
echo
echo "Next: git add results/D1_next_token src/docs/zoo_first_models.md, commit on a branch, open a PR."
