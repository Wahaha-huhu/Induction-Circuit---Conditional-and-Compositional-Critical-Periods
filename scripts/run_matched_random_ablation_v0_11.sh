#!/usr/bin/env bash
set -euo pipefail

# Matched random-head controls for the v0.10 key-slot ablation sweep.
# This script does not retrain. It expects successful behavioural runs with model_final.pt.

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCHES=${EVAL_BATCHES:-24}
TOP_KS=${TOP_KS:-"2 4"}
NUM_RANDOM=${NUM_RANDOM:-20}
RANDOM_SEED=${RANDOM_SEED:-12345}
ARMS=${ARMS:-"s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late"}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.95}
MIN_EXCESS=${MIN_EXCESS:-0.50}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/matched_random_ablation}
INCLUDE_SELECTED_IN_RANDOM_POOL=${INCLUDE_SELECTED_IN_RANDOM_POOL:-0}

mkdir -p "$OUT_DIR"

python scripts/select_ablation_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out "${OUT_DIR}/matched_random_selected_runs.txt" \
  --metadata-out "${OUT_DIR}/matched_random_selected_runs.json" \
  --arms "$ARMS" \
  --success-threshold "$SUCCESS_THRESHOLD" \
  --min-excess "$MIN_EXCESS" \
  --require-model

if [[ ! -s "${OUT_DIR}/matched_random_selected_runs.txt" ]]; then
  echo "No successful runs with model_final.pt were selected."
  echo "Check RUNS_DIR=$RUNS_DIR or whether model_final.pt files are still present locally."
  exit 1
fi

EXTRA_RANDOM_FLAG=()
if [[ "$INCLUDE_SELECTED_IN_RANDOM_POOL" == "1" ]]; then
  EXTRA_RANDOM_FLAG=(--include-selected-in-random-pool)
fi

while IFS= read -r RUN_DIR; do
  [[ -z "$RUN_DIR" ]] && continue
  for TOP_K in $TOP_KS; do
    echo "=== matched-random ablation top-${TOP_K}: ${RUN_DIR} ==="
    python scripts/run_matched_random_ablation_eval.py "$RUN_DIR" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --eval-batches "$EVAL_BATCHES" \
      --top-k "$TOP_K" \
      --num-random "$NUM_RANDOM" \
      --random-seed "$RANDOM_SEED" \
      "${EXTRA_RANDOM_FLAG[@]}"
  done
done < "${OUT_DIR}/matched_random_selected_runs.txt"

python scripts/aggregate_matched_random_ablation.py \
  --runs-dir "$RUNS_DIR" \
  --out-dir "$OUT_DIR"

python scripts/pack_results.py \
  --runs-dir "$RUNS_DIR" \
  --extra "$OUT_DIR" \
  --out cp_toy_matched_random_ablation_v0_11.zip

echo "Wrote cp_toy_matched_random_ablation_v0_11.zip"
