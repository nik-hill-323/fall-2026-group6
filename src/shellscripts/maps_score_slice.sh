#!/usr/bin/env bash
# Score navigation models: D1, D2, D4, the fragility suite, the whole route measure (G9) and D3, one model after
# another (Amendment 5).
#
#   bash src/shellscripts/maps_score_slice.sh pilots                     # the 9 pilots and the 3 anchors
#   bash src/shellscripts/maps_score_slice.sh <slice>                    # a zoo slice of src/docs/maps_score_slices.csv
#   bash src/shellscripts/maps_score_slice.sh <slice> --dry_run          # only print the plan
#   bash src/shellscripts/maps_score_slice.sh <slice> --only <model_id>  # one model
#   JOBS="D1 D4" bash src/shellscripts/maps_score_slice.sh pilots        # only some jobs
#   JOBS=G9 bash src/shellscripts/maps_score_slice.sh <slice>            # only the whole route measure
#   PY=.venv/bin/python bash src/shellscripts/maps_score_slice.sh <slice>  # another Python (default: python)
#
# What it does for every model:
#   1. gets the checkpoint: a zoo model or pilot missing from outputs/ is copied from S3
#      (s3://dats-capstone/shared/group6/navigation/{zoo,pilot}/<id>/model.pt); an anchor needs its weights in
#      ~/artifacts/vafa_weights/<name>/ (from the N2 reproduction), otherwise it is skipped here;
#   2. runs every job with --save. Each scorer checks the checkpoint against checkpoint_sha256 in its training
#      record, the gates (navigation zoo models need the tags prereg-v1.5 and maps-fragility-v1) and that src/ is
#      committed (src/diagnostics/navigation.py);
#   3. skips a job whose result already exists (status ok, or skipped for D3 on the 1.5B anchors; for a zoo
#      model or anchor, a fragility result only if the current, frozen suite made it), so after a crash or restart
#      just run the same command again; a failed job is logged and the slice goes on;
#   4. writes how long every job took to outputs/score_logs/timings.csv (to split the zoo by GPU hours).
#
# Results: zoo models and anchors in results/D1_next_token, D2_state_equiv, D3_inductive_bias, D4_probing and
# fragility_maps, the whole route measure in results/whole_route_maps; pilots in results/pilot_scoring/<same
# folders> (G2: never in any analysis).
# Jobs run cheapest first, D3 last (PySR makes it the slowest). The two 1.5B anchors run D4 with
# --layers_per_pass 8 --chunk 32 (memory only; the probes are the same) and G9 with --batch 32 (memory only).
#
# Run from the repo root of a clean, up to date main, inside tmux, with the wmconcord env active and a valid AWS
# login (aws sso login --use-device-code).

set -u

SLICE="${1:-}"
shift || true
DRY_RUN=""
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN="--dry_run"; shift ;;
    --only) [[ $# -ge 2 ]] || { echo "--only needs a model id"; exit 1; }; ONLY="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
CSV="src/docs/maps_score_slices.csv"
LOGS="outputs/score_logs"
TIMINGS="$LOGS/timings.csv"
S3="${S3_NAV:-s3://dats-capstone/shared/group6/navigation}"
VAFA_WEIGHTS="${WMC_VAFA_WEIGHTS:-$HOME/artifacts/vafa_weights}"
PY="${PY:-python}"
AWS="${AWS:-aws}"
JOBS="${JOBS:-D1 D2 D4 fragility G9 D3}"
DISTS="shortest_paths noisy_shortest_paths random_walks"
BIG_ANCHORS="maps_vafa_noisy_shortest_paths maps_vafa_random_walks"

if [[ -z "$SLICE" ]]; then
  echo "usage: bash src/shellscripts/maps_score_slice.sh <pilots | zoo slice> [--dry_run] [--only <model_id>]"
  exit 1
fi
if [[ ! -f src/diagnostics/navigation.py || ! -f src/fragility/maps.py ]]; then
  echo "run this from the repo root of main with N0 to N8 merged (cannot find navigation.py or fragility/maps.py)"
  exit 1
fi
for j in $JOBS; do
  [[ "$j" =~ ^(D1|D2|D3|D4|fragility|G9)$ ]] || { echo "unknown job $j (use D1 D2 D3 D4 fragility G9)"; exit 1; }
done

# ---- the models of this slice ------------------------------------------------
models=()
if [[ "$SLICE" == "pilots" ]]; then
  for a in transformer lstm mamba; do for d in $DISTS; do models+=("maps_${a}_small_${d}_s100"); done; done
  for d in $DISTS; do models+=("maps_vafa_${d}"); done
else
  if [[ ! -f "$CSV" ]]; then
    echo "no $CSV yet: the zoo slices are written after the pilot run (from its timings)"
    exit 1
  fi
  while IFS=, read -r slice model_id rest || [[ -n "$slice" ]]; do   # also a last line without a newline
    model_id="${model_id%$'\r'}"                                       # and Windows line endings
    [[ "$slice" == "$SLICE" ]] && models+=("$model_id")
  done < "$CSV"
  if (( ${#models[@]} == 0 )); then
    echo "slice $SLICE has no models in $CSV"
    exit 1
  fi
  for t in prereg-v1.5 maps-fragility-v1; do
    if [[ "$(git tag -l "$t")" != "$t" ]]; then
      echo "the git tag $t is missing here: run 'git fetch --tags' (Amendment 5: no navigation zoo model is scored before it)"
      exit 1
    fi
  done
fi

if [[ "$DRY_RUN" != "--dry_run" && -z "${SCORE_SLICE_NO_GPU_CHECK:-}" ]]; then
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "no CUDA GPU visible to $PY: run 'conda activate wmconcord' and check nvidia-smi"
    exit 1
  fi
fi

kind() {          # model_id -> pilot | anchor | zoo
  case "$1" in
    maps_vafa_*) echo anchor ;;
    *_s100) echo pilot ;;
    *) echo zoo ;;
  esac
}

result_file() {   # job model_id
  local base="results"
  [[ "$(kind "$2")" == "pilot" ]] && base="results/pilot_scoring"
  case "$1" in
    D1) echo "$base/D1_next_token/$2.json" ;;
    D2) echo "$base/D2_state_equiv/$2.json" ;;
    D3) echo "$base/D3_inductive_bias/$2.json" ;;
    D4) echo "$base/D4_probing/$2.json" ;;
    fragility) echo "$base/fragility_maps/$2.json" ;;
    G9) echo "$base/whole_route_maps/$2.json" ;;
  esac
}

SUITE_HASH=""
suite_hash() {    # the current fragility suite's hash (frozen at the tag maps-fragility-v1)
  [[ -n "$SUITE_HASH" ]] || SUITE_HASH="$("$PY" -c "import importlib.util as u, sys; s = u.spec_from_file_location('fragility_maps', 'src/fragility/maps.py'); m = u.module_from_spec(s); sys.modules['fragility_maps'] = m; s.loader.exec_module(m); print(m.suite_hash())" 2>/dev/null | tail -1)"
  echo "$SUITE_HASH"
}

result_ok() {     # job file [model_id]: 0 if the file holds a finished result
  [[ -f "$2" ]] || return 1
  if [[ "$1" == "fragility" ]]; then
    if [[ "$(kind "${3:-x_s100}")" == "pilot" ]]; then   # pilots keep their pre rule results (the rule is computed from them)
      "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if 'fragility' in d else 1)" "$2" 2>/dev/null
    else                                                 # zoo models and anchors: only a result of the current suite
      "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if 'fragility' in d and d.get('suite_hash') == sys.argv[2] else 1)" "$2" "$(suite_hash)" 2>/dev/null
    fi
  else
    "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if d.get('status') in ('ok', 'skipped') else 1)" "$2" 2>/dev/null
  fi
}

result_line() {   # job file: the headline numbers, for the log
  "$PY" - "$1" "$2" <<'EOF' 2>/dev/null
import json, sys
job, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
if job == "fragility":
    rel = d.get("relative_fragility")
    print(f"fragility {d['fragility']:+.4f}  relative {rel if rel is None else round(rel, 4)}  "
          f"in distribution {d['in_distribution_legal_rate']:.4f}")
elif job == "G9" and d.get("status") == "ok":
    tf, fp = d["teacher_forced"], d["free_play"]
    print(f"teacher forced all legal {tf['share_routes_all_legal']:.4f}  free play no illegal "
          f"{fp['share_drives_no_illegal']:.4f}  reached {fp['share_reached_destination']:.4f}")
elif d.get("status") == "ok":
    extra = ""
    if job == "D4" and "causal" in d["details"]:
        c = d["details"]["causal"]
        extra = f"  causal {c['causal_score']:.3f} {c['causal_label']}  edit success {c['edit_success_rate']:.3f}"
    print(f"score {d['score']:.4f}{extra}")
else:
    print(f"{d.get('status')}: {d.get('reason', '')[:200]}")
EOF
}

fetch_checkpoint() {   # model_id: 0 if the checkpoint is (now) on this machine
  local k ckpt folder
  k="$(kind "$1")"
  if [[ "$k" == "anchor" ]]; then
    [[ -d "$VAFA_WEIGHTS/${1#maps_vafa_}" ]]
    return
  fi
  folder="zoo"; [[ "$k" == "pilot" ]] && folder="pilot"
  ckpt="outputs/$folder/$1/model.pt"
  [[ -f "$ckpt" ]] && return 0
  [[ "$DRY_RUN" == "--dry_run" ]] && { echo "[plan] $1 checkpoint from S3"; return 0; }
  mkdir -p "outputs/$folder/$1"
  "$AWS" s3 cp "$S3/$folder/$1/model.pt" "$ckpt" --only-show-errors < /dev/null
}

mkdir -p "$LOGS"
[[ -f "$TIMINGS" ]] || echo "model_id,job,seconds,status,host,commit" > "$TIMINGS"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "navigation scoring, slice $SLICE on $(hostname), commit $COMMIT, jobs: $JOBS, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "$(git status --porcelain -- src 2>/dev/null)" ]]; then
  echo "uncommitted changes under src/: the scorers will refuse to save. Commit or reset first."
  [[ "$DRY_RUN" == "--dry_run" ]] || exit 1
fi

n=0; ran=0; skipped=0; failed=0; not_here=(); failed_runs=()
start_all=$(date +%s)
for model_id in "${models[@]}"; do
  [[ -n "$ONLY" && "$model_id" != "$ONLY" ]] && continue
  n=$((n + 1))
  if ! fetch_checkpoint "$model_id"; then
    echo "[skip] $model_id: checkpoint or anchor weights not available on this machine"
    not_here+=("$model_id"); continue
  fi
  for job in $JOBS; do
    out="$(result_file "$job" "$model_id")"
    if result_ok "$job" "$out" "$model_id"; then
      echo "[skip] $model_id $job (result exists)"
      skipped=$((skipped + 1)); continue
    fi
    extra=()
    if [[ "$job" == "D4" && " $BIG_ANCHORS " == *" $model_id "* ]]; then
      extra=(--layers_per_pass 8 --chunk 32)
    fi
    if [[ "$job" == "G9" && " $BIG_ANCHORS " == *" $model_id "* ]]; then
      extra=(--batch 32)
    fi
    case "$job" in
      D1) cmd=("$PY" src/diagnostics/d1/next_token.py --models "$model_id" --save) ;;
      D2) cmd=("$PY" src/diagnostics/d2/state_equiv.py --models "$model_id" --save) ;;
      D3) cmd=("$PY" src/diagnostics/d3/inductive_bias.py --models "$model_id" --save) ;;
      D4) cmd=("$PY" src/diagnostics/d4/representation_probe.py --models "$model_id" --save "${extra[@]}") ;;
      fragility) cmd=("$PY" src/fragility/maps.py --models "$model_id" --save) ;;
      G9) cmd=("$PY" src/analysis/whole_route_maps.py --models "$model_id" --save "${extra[@]}") ;;
    esac
    if [[ "$DRY_RUN" == "--dry_run" ]]; then
      echo "[plan] $model_id $job  ${cmd[*]}  -> $out"
      continue
    fi
    log="$LOGS/${model_id}_${job}.log"
    t0=$(date +%s)
    echo "[run ] $model_id $job  started $(date +%H:%M)"
    "${cmd[@]}" < /dev/null > "$log" 2>&1
    secs=$(( $(date +%s) - t0 ))
    if result_ok "$job" "$out" "$model_id"; then
      echo "[done] $model_id $job  $(result_line "$job" "$out")  ${secs} s"
      echo "$model_id,$job,$secs,ok,$(hostname),$COMMIT" >> "$TIMINGS"
      ran=$((ran + 1))
    else
      echo "[FAIL] $model_id $job  ${secs} s  see $log"
      [[ -f "$out" ]] && echo "       $(result_line "$job" "$out")"
      tail -3 "$log" | sed 's/^/       /'
      echo "$model_id,$job,$secs,failed,$(hostname),$COMMIT" >> "$TIMINGS"
      failed=$((failed + 1)); failed_runs+=("$model_id/$job")
    fi
  done
done

if [[ -n "$ONLY" && $n -eq 0 ]]; then
  echo "$ONLY is not in slice $SLICE"
  exit 1
fi
missing=()            # zoo models and pilots that could not be fetched: the slice is not finished
for m in "${not_here[@]}"; do [[ "$(kind "$m")" != "anchor" ]] && missing+=("$m"); done
mins=$(( ($(date +%s) - start_all) / 60 ))
echo
echo "slice $SLICE: $n models, $ran jobs run now, $skipped skipped (already scored), $failed failed, ${mins} min"
(( ${#not_here[@]} )) && echo "not available on this machine: ${not_here[*]}"
if (( failed > 0 || ${#missing[@]} > 0 )); then
  (( failed > 0 )) && echo "failed: ${failed_runs[*]}"
  (( ${#missing[@]} > 0 )) && echo "checkpoint not copied (S3 login expired?): ${missing[*]}; run 'aws sso login --use-device-code'"
  echo "the slice is not finished: fix the cause, then run the same command again; finished jobs are skipped"
  exit 1
fi
if [[ "$DRY_RUN" != "--dry_run" ]]; then
  echo "next: commit only this slice's result files, on a branch, and open a PR:"
  if [[ "$SLICE" == "pilots" ]]; then
    echo "  git add results/pilot_scoring results/D1_next_token results/D2_state_equiv results/D3_inductive_bias results/D4_probing results/fragility_maps results/whole_route_maps && git status"
  else
    echo "  git add results/D1_next_token results/D2_state_equiv results/D3_inductive_bias results/D4_probing results/fragility_maps results/whole_route_maps && git status"
  fi
fi
