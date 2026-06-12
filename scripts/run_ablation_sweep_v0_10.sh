#!/usr/bin/env bash
set -euo pipefail

# Run key-slot mean ablation on already-trained successful HOP_2 behavioural models.
# This script does not retrain. It expects model_final.pt files still present locally.
# Run from repository root after `python -m pip install -e .`.

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCHES=${EVAL_BATCHES:-32}
TOP_KS=${TOP_KS:-"1 2 4"}
ARMS=${ARMS:-"s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late"}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.95}
MIN_EXCESS=${MIN_EXCESS:-0.50}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/ablation}

mkdir -p "$OUT_DIR"

python scripts/select_ablation_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out "${OUT_DIR}/ablation_selected_runs.txt" \
  --metadata-out "${OUT_DIR}/ablation_selected_runs.json" \
  --arms "$ARMS" \
  --success-threshold "$SUCCESS_THRESHOLD" \
  --min-excess "$MIN_EXCESS" \
  --require-model

if [[ ! -s "${OUT_DIR}/ablation_selected_runs.txt" ]]; then
  echo "No successful runs with model_final.pt were selected."
  echo "Check RUNS_DIR=$RUNS_DIR or whether model_final.pt files are still present locally."
  exit 1
fi

while IFS= read -r RUN_DIR; do
  [[ -z "$RUN_DIR" ]] && continue
  for TOP_K in $TOP_KS; do
    echo "=== ablation top-${TOP_K}: ${RUN_DIR} ==="
    python scripts/run_ablation_eval.py "$RUN_DIR" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --eval-batches "$EVAL_BATCHES" \
      --top-k "$TOP_K"
  done
done < "${OUT_DIR}/ablation_selected_runs.txt"

python scripts/aggregate_ablation_results.py \
  --runs-dir "$RUNS_DIR" \
  --out-dir "$OUT_DIR"

python scripts/pack_results.py \
  --runs-dir "$RUNS_DIR" \
  --extra "$OUT_DIR" \
  --out cp_toy_ablation_sweep_v0_10.zip

echo "Wrote cp_toy_ablation_sweep_v0_10.zip"
