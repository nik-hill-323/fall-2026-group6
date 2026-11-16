#!/usr/bin/env bash
# Train one navigation zoo slice end to end, unattended, in the order Amendment 5 (G12) fixes:
# all 27 transformers, then the 27 LSTMs, then the 27 Mamba models, across the four machines.
#
#   bash src/shellscripts/maps_zoo_overnight.sh 1        # your zoo slice: 1, 2, 3 or 4
#
# For each architecture it:
#   1. waits until all 27 models of the previous architecture have training records on GitHub (pulls every 5 min);
#   2. trains this slice's models of the architecture with src/shellscripts/maps_train_slice.sh;
#   3. shares: commits and pushes this slice's new records, and copies its checkpoints to S3.
# Run it from the repo root, on the branch maps-zoo, inside tmux, with your environment active. Safe to rerun after a
# crash or restart: finished models are skipped and finished phases pass straight through. Log:
# outputs/zoo_logs/overnight_slice_N.log
#
# It stops (and says why) only when a model of its own slice fails twice, because the other machines would otherwise
# wait for a record that never comes. Fix the cause and run the same command again.

set -u

SLICE="${1:-}"
LAUNCHER="${LAUNCHER:-src/shellscripts/maps_train_slice.sh}"
AWS="${AWS:-aws}"
S3_ZOO="${S3_ZOO:-s3://dats-capstone/shared/group6/navigation/zoo/}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
PER_ARCH=27
ARCHS=(transformer lstm mamba)
LABEL=(Transformer LSTM Mamba)

if [[ ! "$SLICE" =~ ^[1-4]$ ]]; then
  echo "usage: bash src/shellscripts/maps_zoo_overnight.sh <1 to 4>"
  exit 1
fi
if [[ ! -f src/docs/maps_zoo_slices.csv ]]; then
  echo "run this from the repo root"
  exit 1
fi
mkdir -p outputs/zoo_logs
LOG="outputs/zoo_logs/overnight_slice_${SLICE}.log"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

# Who commits: the professor counts commits per person, so never commit as the EC2 default user.
NAME="$(git config user.name || true)"
if [[ -z "$NAME" || "$NAME" =~ ^[Uu]buntu$ ]]; then
  echo "git does not know who you are on this machine. Run once, with your own name and GitHub email:"
  echo "  git config --global user.name \"Your Name\""
  echo "  git config --global user.email \"you@example.com\""
  exit 1
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "maps-zoo" ]]; then
  echo "you are on '$BRANCH'; the zoo runs on the branch maps-zoo (see Phase 2 of the guide)"
  exit 1
fi

count() { ls results/zoo_training/maps_"$1"_*.json 2>/dev/null | wc -l | tr -d ' '; }

pull() { git pull --rebase --autostash -q >/dev/null 2>&1 || log "  (git pull failed this time; will try again)"; }

share() {   # $1 = arch, $2 = commit message suffix
  local arch="$1"
  git add results/zoo_training/maps_"$arch"_*.json 2>/dev/null
  if git diff --cached --quiet; then
    log "  nothing new to commit for $arch"
  else
    git commit -q -m "Navigation zoo, slice $SLICE: $arch records$2 (#148)"
    local n
    for n in 1 2 3 4 5; do
      if git pull --rebase --autostash -q >/dev/null 2>&1 && git push -q >/dev/null 2>&1; then
        log "  pushed $arch records to GitHub"
        break
      fi
      log "  push attempt $n failed; retrying in 30 s"
      sleep 30
      [[ $n == 5 ]] && log "  WARNING: could not push; run 'git pull --rebase && git push' by hand"
    done
  fi
  if "$AWS" s3 sync outputs/zoo/ "$S3_ZOO" --exclude "*" --include "maps_*" >/dev/null 2>&1; then
    log "  copied checkpoints to S3"
  else
    log "  WARNING: S3 copy failed (login expired?). Checkpoints are safe on this machine; later run:"
    log "    aws sso login --use-device-code && aws s3 sync outputs/zoo/ $S3_ZOO --exclude \"*\" --include \"maps_*\""
  fi
}

log "overnight run, slice $SLICE, as $NAME, on $(hostname), commit $(git rev-parse --short HEAD)"
pull

for i in 0 1 2; do
  arch="${ARCHS[$i]}"
  if (( i > 0 )); then
    prev="${ARCHS[$((i - 1))]}"
    while true; do
      pull
      have="$(count "$prev")"
      if (( have >= PER_ARCH )); then
        log "all $PER_ARCH ${LABEL[$((i - 1))]} models are on GitHub: starting ${LABEL[$i]}"
        break
      fi
      log "waiting: $have of $PER_ARCH ${LABEL[$((i - 1))]} models on GitHub (G12 order); checking again in $((WAIT_SECONDS / 60)) min"
      sleep "$WAIT_SECONDS"
    done
  fi

  log "training slice $SLICE ${LABEL[$i]} models"
  ok=0
  for attempt in 1 2; do
    bash "$LAUNCHER" "$SLICE" "$arch" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
      ok=1; break
    fi
    log "  some ${LABEL[$i]} models failed (attempt $attempt)"
  done
  share "$arch" ""
  if (( ! ok )); then
    log "STOPPED: a ${LABEL[$i]} model of slice $SLICE failed twice. The other machines wait for its record."
    log "Post the [FAIL] lines above in the group, fix the cause, then run the same command again."
    exit 1
  fi
done

log "slice $SLICE finished: transformers, LSTMs and Mamba trained and shared"
log "zoo records on GitHub now: $(ls results/zoo_training/maps_*.json 2>/dev/null | wc -l | tr -d ' ') of 81"
