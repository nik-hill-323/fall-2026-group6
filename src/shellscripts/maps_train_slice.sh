#!/usr/bin/env bash
# Train one slice of the navigation (maps) zoo, or one slice of its 9 pilots, one model after another.
#
#   bash src/shellscripts/maps_train_slice.sh p1               # pilots: p1, p2, p3 or p4
#   bash src/shellscripts/maps_train_slice.sh 1                # zoo: 1, 2, 3 or 4 (transformers, then LSTMs, then Mamba)
#   bash src/shellscripts/maps_train_slice.sh 1 transformer    # only this architecture of the slice
#   bash src/shellscripts/maps_train_slice.sh 1 --dry_run      # only print the plan (also: 1 lstm --dry_run)
#
# The slice list is src/docs/maps_zoo_slices.csv. Amendment 5 (reports/preregistration.md):
#   * the 9 pilots (seed 100) are trained before any zoo model; their records go to results/pilot_training/
#     and their checkpoints to outputs/pilot/
#   * a zoo model trains only with the tag prereg-v1.5 present, after all 9 pilot records are on main,
#     from committed code; src/zoo/train.py refuses otherwise, and this script checks the same before starting
#   * order (G12): all transformers, then all LSTMs, then all Mamba models
# Every model uses the frozen budget: 5,000 steps, batch 256, with micro_batch from the CSV. If a model runs
# out of GPU memory, it is retried with half the micro batch (down to 16); the gradient is the same either way
# (micro batching is exact, see src/tests/test_zoo_train_maps.py), and the record keeps the micro batch used.
# A model that already has a training record is skipped, so after a crash or restart run the same command
# again. A failed model is logged and the slice goes on.
#
# Run from the repo root, inside tmux, with the wmconcord env active and the data in ~/artifacts/maps
# (or set MAPS_DATA=/path/to/maps).

set -u

SLICE="${1:-}"
ARG2="${2:-}"
ARG3="${3:-}"
CSV="src/docs/maps_zoo_slices.csv"
LOGS="outputs/zoo_logs"
PY="${PY:-python}"
STEPS=5000
BATCH=256
TAG="prereg-v1.5"

ARCH=""; DRY_RUN=""
for a in "$ARG2" "$ARG3"; do
  case "$a" in
    "") ;;
    --dry_run) DRY_RUN="--dry_run" ;;
    transformer|lstm|mamba) ARCH="$a" ;;
    *) echo "unknown argument: $a"; exit 1 ;;
  esac
done

if [[ "$SLICE" =~ ^[1-4]$ ]]; then
  KIND="zoo"; RECORDS="results/zoo_training"
elif [[ "$SLICE" =~ ^p[1-4]$ ]]; then
  KIND="pilot"; RECORDS="results/pilot_training"
else
  echo "usage: bash src/shellscripts/maps_train_slice.sh <p1 to p4 | 1 to 4> [transformer|lstm|mamba] [--dry_run]"
  exit 1
fi
if [[ ! -f "$CSV" || ! -f src/zoo/train.py ]]; then
  echo "run this from the repo root (cannot find $CSV or src/zoo/train.py)"
  exit 1
fi
DATA_ARGS=()
if [[ -n "${MAPS_DATA:-}" ]]; then DATA_ARGS=(--maps_data "$MAPS_DATA"); fi

# The checks src/zoo/train.py makes for every model, made once up front so a slice never starts only to fail.
if [[ "$DRY_RUN" != "--dry_run" ]]; then
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "no CUDA GPU visible to $PY: run 'conda activate wmconcord' and check nvidia-smi"
    exit 1
  fi
  if [[ -n "$(git status --porcelain -- src 2>/dev/null)" ]]; then
    echo "uncommitted changes under src/ (Amendment 5: train from committed code). See: git status"
    exit 1
  fi
  if [[ "$KIND" == "zoo" ]]; then
    if [[ -z "$(git tag -l "$TAG")" ]]; then
      echo "the tag $TAG is not here; run: git fetch --tags   (no zoo model before it, Amendment 5)"
      exit 1
    fi
    missing=0
    while IFS=, read -r slice model_id rest; do
      [[ "$slice" =~ ^p[1-4]$ && ! -f "results/pilot_training/$model_id.json" ]] && { echo "missing pilot record: $model_id"; missing=1; }
    done < "$CSV"
    if (( missing )); then
      echo "all 9 pilots are trained before any zoo model (Amendment 5, G2); pull main after the pilot records are merged"
      exit 1
    fi
  fi
fi

mkdir -p "$LOGS"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "maps $KIND slice $SLICE${ARCH:+ ($ARCH only)} on $(hostname), commit $COMMIT, $(date -u +%Y-%m-%dT%H:%M:%SZ)"

total=0; done_n=0; skipped=0; failed=0; failed_ids=(); low=()
while IFS=, read -r slice model_id arch scale dist seed micro est owner; do
  [[ "$slice" == "slice" || "$slice" != "$SLICE" ]] && continue
  [[ -n "$ARCH" && "$arch" != "$ARCH" ]] && continue
  total=$((total + 1))
  record="$RECORDS/$model_id.json"
  if [[ -f "$record" ]]; then
    echo "[skip] $model_id (record exists)"
    skipped=$((skipped + 1))
    continue
  fi
  base=("$PY" src/zoo/train.py --domain maps --arch "$arch" --scale "$scale" --distribution "$dist"
        --seed "$seed" --steps "$STEPS" --batch "$BATCH" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"})
  if [[ "$DRY_RUN" == "--dry_run" ]]; then
    echo "[plan] $model_id  (about $est min)  ${base[*]} --micro_batch $micro"
    continue
  fi
  log="$LOGS/$model_id.log"
  mb="$micro"; ok=0
  while true; do
    echo "[run ] $model_id  micro_batch $mb  (about $est min)  started $(date +%H:%M)"
    if "${base[@]}" --micro_batch "$mb" < /dev/null > "$log" 2>&1 && [[ -f "$record" ]]; then
      ok=1; break
    fi
    if grep -qi "out of memory" "$log"; then
      if (( mb == 0 )); then next=128; else next=$((mb / 2)); fi
      if (( next >= 16 )); then
        echo "[oom ] $model_id  retrying with micro_batch $next"
        mb="$next"; continue
      fi
    fi
    break
  done
  if (( ok )); then
    rate="$(grep -o 'legal-move rate [0-9.]*' "$log" | tail -1)"
    echo "[done] $model_id  $rate  finished $(date +%H:%M)"
    value="${rate##* }"
    if [[ -n "$value" ]] && awk -v r="$value" 'BEGIN { exit !(r < 0.50) }'; then
      echo "       below the 0.50 admission rate (Amendment 5, G3): it stays in the records, flag it to the team"
      low+=("$model_id")
    fi
    done_n=$((done_n + 1))
  else
    echo "[FAIL] $model_id  see $log"
    tail -5 "$log" | sed 's/^/       /'
    failed=$((failed + 1)); failed_ids+=("$model_id")
  fi
done < "$CSV"

echo
echo "maps $KIND slice $SLICE: $total models, $done_n trained now, $skipped skipped, $failed failed"
if (( ${#low[@]} > 0 )); then
  echo "below admission: ${low[*]}"
fi
if (( failed > 0 )); then
  echo "failed: ${failed_ids[*]}"
  echo "fix the cause (see the logs), then run the same command again; finished models are skipped"
  exit 1
fi
if [[ "$DRY_RUN" != "--dry_run" ]]; then
  echo "next: commit only your new records:"
  echo "  git add $RECORDS/maps_*.json && git status"
fi
