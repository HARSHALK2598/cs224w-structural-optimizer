#!/usr/bin/env bash
# Runs RelBench's official examples/gnn_entity.py, unmodified, for one rel-f1
# task and seeds 0-4 (plans/00-baseline.md, step 3).
#
# Usage (from the project repo root):
#   bash experiments/run_official.sh driver-top3
#   EPOCHS=20 bash experiments/run_official.sh driver-top3   # sensitivity run
#
# Writes, per seed N (OUT = results/official, or results/official_epochs<E>
# when EPOCHS is set):
#   experiments/<OUT>/logs/<task>_seed<N>.log                full output
#   experiments/<OUT>/preds/seed<N>/rel-f1__<task>.csv       test predictions
# Each run is independent: a failed seed is reported and the others continue.
set -uo pipefail

TASK="${1:?usage: run_official.sh <task>}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-}"  # empty: the official script's default (10)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/experiments/results/official${EPOCHS:+_epochs$EPOCHS}"
mkdir -p "$OUT/logs"

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export HF_HOME="$ROOT/data/hf"
cd "$ROOT/reference/relbench/examples"  # the script imports model.py from here

for SEED in $SEEDS; do
  LOG="$OUT/logs/${TASK}_seed${SEED}.log"
  START=$(date +%s)
  python gnn_entity.py --dataset rel-f1 --task "$TASK" --seed "$SEED" \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    --cache_dir "$ROOT/data/relbench_examples" \
    --pred_dir "$OUT/preds/seed${SEED}" > "$LOG" 2>&1
  STATUS=$?
  echo "$TASK seed $SEED: exit $STATUS, $(( $(date +%s) - START ))s, $(grep -a 'Best test' "$LOG" | tail -1)"
done
