#!/usr/bin/env bash
# Score one slice of the Othello zoo: D1, D2, D3, D4 and the fragility suite, one model after another.
#
#   bash src/shellscripts/score_slice.sh <slice>                      # 0, 1, 2, 3 or 4
#   bash src/shellscripts/score_slice.sh <slice> --dry_run            # only print the plan
#   bash src/shellscripts/score_slice.sh <slice> --only <model_id>    # one model (the test run)
#   JOBS="D1 D4" bash src/shellscripts/score_slice.sh <slice>         # only some jobs (timing pilot)
#
# The slice list is src/docs/zoo_slices.csv, the same file train_slice.sh uses, so everyone scores
# the models they trained, on the EC2 that holds the checkpoints. Slice 0 is the four first models;
# run it next to your own slice, and models whose checkpoint is not on this machine are skipped.
#
# Before a model is scored, its checkpoint must match checkpoint_sha256 in its training record
# (results/zoo_training/<model_id>.json on main), so every score belongs to the model that was
# trained and uploaded to S3. A job whose result already exists with status ok is skipped, so after
# a crash or restart just run the same command again. A failed job is logged, its result file keeps
# status failed with the error as the reason, and the slice goes on; the next run retries it.
#
# Jobs, in this order (cheapest first, D3 last because PySR makes it the slowest):
#   D1         src/diagnostics/d1/next_token.py            -> results/D1_next_token/<id>.json
#   D2         src/diagnostics/d2/state_equiv.py           -> results/D2_state_equiv/<id>.json
#   D4         src/diagnostics/d4/representation_probe.py  -> results/D4_probing/<id>.json
#   fragility  src/fragility/othello.py (fragility-v1)     -> results/fragility/<id>.json
#   D3         src/diagnostics/d3/inductive_bias.py        -> results/D3_inductive_bias/<id>.json
# Every diagnostic uses its preregistered defaults (test split, 1,000 games, data seed 0).
#
# Run from the repo root of a clean main, inside tmux, with the wmconcord env active.

set -u

SLICE="${1:-}"
shift || true
DRY_RUN=""
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN="--dry_run"; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
CSV="src/docs/zoo_slices.csv"
RECORDS="results/zoo_training"
ZOO="outputs/zoo"
LOGS="outputs/score_logs"
PY="${PY:-python}"
JOBS="${JOBS:-D1 D2 D4 fragility D3}"
PREREG_TAG="prereg-v1.2"

if [[ ! "$SLICE" =~ ^[0-4]$ ]]; then
  echo "usage: bash src/shellscripts/score_slice.sh <slice 0 to 4> [--dry_run] [--only <model_id>]"
  exit 1
fi
if [[ ! -f "$CSV" || ! -f src/fragility/othello.py ]]; then
  echo "run this from the repo root (cannot find $CSV or src/fragility/othello.py)"
  exit 1
fi
for j in $JOBS; do
  [[ "$j" =~ ^(D1|D2|D3|D4|fragility)$ ]] || { echo "unknown job $j (use D1 D2 D3 D4 fragility)"; exit 1; }
done
if [[ "$(git tag -l "$PREREG_TAG")" != "$PREREG_TAG" ]]; then
  echo "the git tag $PREREG_TAG is missing here: run 'git fetch --tags' (no zoo model is scored before it)"
  exit 1
fi
if [[ "$DRY_RUN" != "--dry_run" && -z "${SCORE_SLICE_NO_GPU_CHECK:-}" ]]; then
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "no CUDA GPU visible to $PY: run 'conda activate wmconcord' and check nvidia-smi"
    exit 1
  fi
fi

result_file() {   # job model_id
  case "$1" in
    D1) echo "results/D1_next_token/$2.json" ;;
    D2) echo "results/D2_state_equiv/$2.json" ;;
    D3) echo "results/D3_inductive_bias/$2.json" ;;
    D4) echo "results/D4_probing/$2.json" ;;
    fragility) echo "results/fragility/$2.json" ;;
  esac
}

result_ok() {     # job file: 0 if the file holds a finished result
  [[ -f "$2" ]] || return 1
  if [[ "$1" == "fragility" ]]; then
    "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if 'fragility' in d else 1)" "$2" 2>/dev/null
  else
    "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d.get('status') == 'ok' else 1)" "$2" 2>/dev/null
  fi
}

result_line() {   # job file: the headline numbers, for the log
  "$PY" - "$1" "$2" <<'EOF' 2>/dev/null
import json, sys
job, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
if job == "fragility":
    print(f"fragility {d['fragility']:.4f}  in distribution {d['in_distribution_legal_rate']:.4f}")
elif d.get("status") == "ok":
    extra = ""
    if job == "D4" and "causal" in d["details"]:
        extra = f"  causal {d['details']['causal']['causal_score']:.3f} {d['details']['causal']['causal_label']}"
    print(f"score {d['score']:.4f}{extra}")
else:
    print(f"{d.get('status')}: {d.get('reason', '')[:200]}")
EOF
}

mkdir -p "$LOGS" results/fragility
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "score slice $SLICE on $(hostname), commit $COMMIT, jobs: $JOBS, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$(git status --porcelain -- src 2>/dev/null)" ]]; then
  echo "warning: you have uncommitted changes under src/; scores must come from a clean main"
fi

models=0; ran=0; skipped=0; failed=0; not_here=(); bad_hash=(); not_trained=(); failed_runs=()
start_all=$(date +%s)
while IFS=, read -r slice model_id arch scale dist seed micro est owner; do
  [[ "$slice" == "slice" || "$slice" != "$SLICE" ]] && continue
  [[ -n "$ONLY" && "$model_id" != "$ONLY" ]] && continue
  models=$((models + 1))
  record="$RECORDS/$model_id.json"
  ckpt="$ZOO/$model_id/model.pt"
  if [[ ! -f "$record" ]]; then
    echo "[skip] $model_id: no training record on this branch (pull main)"
    not_trained+=("$model_id"); continue
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $model_id: checkpoint $ckpt is not on this machine"
    not_here+=("$model_id"); continue
  fi
  want="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('checkpoint_sha256', ''))" "$record")"
  if [[ "$DRY_RUN" != "--dry_run" ]]; then
    have="$(sha256sum "$ckpt" | cut -d' ' -f1)"
    if [[ -z "$want" || "$have" != "$want" ]]; then
      echo "[HASH] $model_id: checkpoint sha256 does not match its training record, not scored"
      bad_hash+=("$model_id"); continue
    fi
  fi
  for job in $JOBS; do
    out="$(result_file "$job" "$model_id")"
    if result_ok "$job" "$out"; then
      echo "[skip] $model_id $job (result exists)"
      skipped=$((skipped + 1)); continue
    fi
    case "$job" in
      D1) cmd=("$PY" src/diagnostics/d1/next_token.py --models "$model_id" --save) ;;
      D2) cmd=("$PY" src/diagnostics/d2/state_equiv.py --models "$model_id" --save) ;;
      D3) cmd=("$PY" src/diagnostics/d3/inductive_bias.py --models "$model_id" --save) ;;
      D4) cmd=("$PY" src/diagnostics/d4/representation_probe.py --models "$model_id" --save) ;;
      fragility) cmd=("$PY" src/fragility/othello.py --model "$ckpt" --arch "$arch" --scale "$scale"
                      --train_distribution "$dist" --seed "$seed") ;;
    esac
    if [[ "$DRY_RUN" == "--dry_run" ]]; then
      echo "[plan] $model_id $job  ${cmd[*]}"
      continue
    fi
    log="$LOGS/${model_id}_${job}.log"
    t0=$(date +%s)
    echo "[run ] $model_id $job  started $(date +%H:%M)"
    "${cmd[@]}" < /dev/null > "$log" 2>&1
    secs=$(( $(date +%s) - t0 ))
    if result_ok "$job" "$out"; then
      echo "[done] $model_id $job  $(result_line "$job" "$out")  ${secs} s"
      ran=$((ran + 1))
    else
      echo "[FAIL] $model_id $job  ${secs} s  see $log"
      [[ -f "$out" ]] && echo "       $(result_line "$job" "$out")"
      tail -3 "$log" | sed 's/^/       /'
      failed=$((failed + 1)); failed_runs+=("$model_id/$job")
    fi
  done
done < "$CSV"

mins=$(( ($(date +%s) - start_all) / 60 ))
echo
echo "slice $SLICE: $models models, $ran jobs run now, $skipped skipped (already scored), $failed failed, ${mins} min"
(( ${#not_here[@]} )) && echo "checkpoint not on this machine (scored elsewhere): ${not_here[*]}"
(( ${#not_trained[@]} )) && echo "no training record yet: ${not_trained[*]}"
(( ${#bad_hash[@]} )) && echo "checkpoint does not match its record, NOT scored: ${bad_hash[*]}"
if (( failed > 0 )); then
  echo "failed: ${failed_runs[*]}"
  echo "fix the cause (see the logs), then run the same command again; finished jobs are skipped"
  exit 1
fi
if [[ "$DRY_RUN" != "--dry_run" ]]; then
  echo "next: commit only the result files of your models (never another slice's, never master_table.csv):"
  echo "  git add results/D1_next_token results/D2_state_equiv results/D3_inductive_bias results/D4_probing results/fragility && git status"
fi
