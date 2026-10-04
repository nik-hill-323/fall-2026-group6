#!/usr/bin/env bash
# Download every Othello artifact the project uses into ~/artifacts (outside the repo, never in git).
# Safe to rerun: files that already exist are skipped.
#
#     bash src/shellscripts/fetch_othello_data.sh
#
# What lands where:
#   ~/artifacts/othello_data/data/{train,val}/games_*.bin   synthetic games (othello_world, Li et al. 2023),
#                                                          repackaged by alexandretl/othello on Hugging Face
#   ~/artifacts/othello_gpt_tl/synthetic_model.pth          Othello GPT weights in TransformerLens format
#                                                          (NeelNanda/Othello-GPT-Transformer-Lens)
#   ~/artifacts/othello_championship/WTH_YYYY.wtb          WTHOR tournament archive 1977 to 2025
#                                                          (French Othello Federation, ffothello.org)
#
# Expected after a full run (checked on the EC2, Oct 2026):
#   synthetic     190 train files, 48 val files, about 100,000 games each
#   championship  49 .wtb files, 137,548 games before cleaning (src/domains/othello.py removes
#                 duplicates and illegal games, then splits 80 / 10 / 10 into train, val and test)

set -euo pipefail
ART="${ARTIFACTS:-$HOME/artifacts}"
mkdir -p "$ART"

echo "== synthetic games (Hugging Face: alexandretl/othello, about 1 GB download, 2.5 GB unpacked)"
if [ ! -d "$ART/othello_data/data/train" ]; then
  hf download alexandretl/othello --repo-type dataset --local-dir "$ART/othello_data"
  tar -xJf "$ART/othello_data/data.tar.xz" -C "$ART/othello_data"
fi
echo "   train files: $(ls "$ART/othello_data/data/train" | wc -l), val files: $(ls "$ART/othello_data/data/val" | wc -l)"

echo "== Othello GPT weights (Hugging Face: NeelNanda/Othello-GPT-Transformer-Lens)"
if [ ! -f "$ART/othello_gpt_tl/synthetic_model.pth" ]; then
  hf download NeelNanda/Othello-GPT-Transformer-Lens --local-dir "$ART/othello_gpt_tl"
fi
ls -la "$ART/othello_gpt_tl"/*.pth

echo "== championship games (WTHOR, ffothello.org)"
mkdir -p "$ART/othello_championship"
for y in $(seq 1977 2025); do
  wget -q -nc -P "$ART/othello_championship" "https://www.ffothello.org/wthor/base/WTH_$y.wtb" || echo "   missing $y"
done
wget -q -nc -P "$ART/othello_championship" https://www.ffothello.org/wthor/Format_WThor.pdf || true
echo "   .wtb files: $(ls "$ART/othello_championship"/WTH_*.wtb | wc -l)"

echo "done"
