#!/usr/bin/env bash
# Train one slice of the Othello zoo, one model after another.
#
#   bash src/shellscripts/train_slice.sh <slice>            # 1, 2, 3 or 4
#   bash src/shellscripts/train_slice.sh <slice> --dry_run  # only print the plan
#
# The slice list is src/docs/zoo_slices.csv (who trains what, same as the team doc).
# Every model uses the frozen budget: 5,000 steps, batch 256. Mamba large gets
# micro_batch 64 from the CSV. A model that already has a training record in
# results/zoo_training/ is skipped, so after a crash or restart just run the
# same command again. A failed model is logged and the slice goes on.
#
# Run from the repo root, inside tmux, with the wmconcord env active.

set -u

SLICE="${1:-}"
DRY_RUN="${2:-}"
CSV="src/docs/zoo_slices.csv"
RECORDS="results/zoo_training"
LOGS="outputs/zoo_logs"
PY="${PY:-python}"
STEPS=5000
BATCH=256

if [[ ! "$SLICE" =~ ^[1-4]$ ]]; then
  echo "usage: bash src/shellscripts/train_slice.sh <slice 1 to 4> [--dry_run]"
  exit 1
fi
if [[ ! -f "$CSV" || ! -f src/zoo/train.py ]]; then
  echo "run this from the repo root (cannot find $CSV or src/zoo/train.py)"
  exit 1
fi

if [[ "$DRY_RUN" != "--dry_run" ]]; then
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "no CUDA GPU visible to $PY: run 'conda activate wmconcord' and check nvidia-smi"
    exit 1
  fi
fi

mkdir -p "$LOGS"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "slice $SLICE on $(hostname), commit $COMMIT, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$(git status --porcelain -- src 2>/dev/null)" ]]; then
  echo "warning: you have uncommitted changes under src/; the zoo should train from a clean main"
fi

total=0; done_n=0; skipped=0; failed=0; failed_ids=()
while IFS=, read -r slice model_id arch scale dist seed micro est owner; do
  [[ "$slice" == "slice" || "$slice" != "$SLICE" ]] && continue
  total=$((total + 1))
  record="$RECORDS/$model_id.json"
  if [[ -f "$record" ]]; then
    echo "[skip] $model_id (record exists)"
    skipped=$((skipped + 1))
    continue
  fi
  cmd=("$PY" src/zoo/train.py --arch "$arch" --scale "$scale" --distribution "$dist"
       --seed "$seed" --steps "$STEPS" --batch "$BATCH" --micro_batch "$micro")
  if [[ "$DRY_RUN" == "--dry_run" ]]; then
    echo "[plan] $model_id  (about $est min)  ${cmd[*]}"
    continue
  fi
  echo "[run ] $model_id  (about $est min)  started $(date +%H:%M)"
  log="$LOGS/$model_id.log"
  if "${cmd[@]}" < /dev/null > "$log" 2>&1 && [[ -f "$record" ]]; then
    rate="$(grep -o 'legal-move rate [0-9.]*' "$log" | tail -1)"
    echo "[done] $model_id  $rate  finished $(date +%H:%M)"
    done_n=$((done_n + 1))
  else
    echo "[FAIL] $model_id  see $log"
    tail -5 "$log" | sed 's/^/       /'
    failed=$((failed + 1)); failed_ids+=("$model_id")
  fi
done < "$CSV"

echo
echo "slice $SLICE: $total models, $done_n trained now, $skipped skipped, $failed failed"
if (( failed > 0 )); then
  echo "failed: ${failed_ids[*]}"
  echo "fix the cause (see the logs), then run the same command again; finished models are skipped"
  exit 1
fi
if [[ "$DRY_RUN" != "--dry_run" ]]; then
  echo "next: commit only your new records, never src/docs/zoo_registry.csv:"
  echo "  git add $RECORDS/*.json && git status"
fi
