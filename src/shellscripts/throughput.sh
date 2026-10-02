#!/usr/bin/env bash
# Throughput measurement for PR #19: one short run per scale (and per arch if
# the code exists), extrapolated to the full zoo budget. Run on the EC2 from
# the repo root with the wmconcord env active:
#
#     bash src/shellscripts/throughput.sh
#
# Writes outputs/zoo/throughput.md with one table row per run.

set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-300}"
N_GAMES="${N_GAMES:-50000}"
ARCHS="${ARCHS:-transformer lstm}"
OUT=outputs/zoo/throughput.md
mkdir -p outputs/zoo

{
  echo "Measured on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo cpu), $(date -u +%F)."
  echo "Each run: ${STEPS} timed steps, batch 256 games, ${N_GAMES} training games loaded."
  echo "min/model is extrapolated to the full budget in run_matrix.TRAINING_BUDGET; 27 models per arch and scale (3 domains x 3 distributions x 3 seeds)."
  echo
  echo "| arch | scale | params | steps timed | s/step | min/model (full) | h / 27 models | legal rate | next-tok acc |"
  echo "|---|---|---|---|---|---|---|---|---|"
} > "$OUT"

for arch in $ARCHS; do
  for scale in small medium large; do
    echo "=== $arch / $scale ==="
    python src/zoo/train.py --arch "$arch" --scale "$scale" --steps "$STEPS" \
        --n_train_games "$N_GAMES" --log_every 100 \
      || { echo "| $arch | $scale | failed | | | | | | |" >> "$OUT"; continue; }
    cat "outputs/zoo/othello_${arch}_${scale}_synthetic_s0/row.md" >> "$OUT"
  done
done

echo
cat "$OUT"
