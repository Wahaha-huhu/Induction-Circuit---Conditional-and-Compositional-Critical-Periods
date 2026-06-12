#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/twohop_decomposition}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCHES=${EVAL_BATCHES:-16}
CALIB_BATCHES=${CALIB_BATCHES:-8}
TOP_KS=${TOP_KS:-"2 4"}
NUM_RANDOM=${NUM_RANDOM:-20}
ARMS=${ARMS:-s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late}
INCLUDE_CAUSAL_RANKING=${INCLUDE_CAUSAL_RANKING:-0}

mkdir -p "$OUT_DIR"

python scripts/select_ablation_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out "$OUT_DIR/twohop_selected_runs.txt" \
  --metadata-out "$OUT_DIR/twohop_selected_runs.json" \
  --arms "$ARMS" \
  --success-threshold 0.95 \
  --min-excess 0.50 \
  --require-model

EXTRA_FLAGS=()
if [[ "$INCLUDE_CAUSAL_RANKING" == "1" ]]; then
  EXTRA_FLAGS+=(--include-causal-ranking)
fi

while IFS= read -r RUN_DIR; do
  [[ -z "$RUN_DIR" ]] && continue
  for K in $TOP_KS; do
    echo "[twohop] run=$RUN_DIR top_k=$K"
    python scripts/run_twohop_score_ablation_eval.py "$RUN_DIR" \
      --out-dir "$OUT_DIR/evals" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --eval-batches "$EVAL_BATCHES" \
      --calib-batches "$CALIB_BATCHES" \
      --top-k "$K" \
      --num-random "$NUM_RANDOM" \
      "${EXTRA_FLAGS[@]}"
  done
done < "$OUT_DIR/twohop_selected_runs.txt"

python scripts/aggregate_twohop_score_ablation.py --out-dir "$OUT_DIR"

python scripts/pack_results.py \
  --runs-dir "$OUT_DIR" \
  --out cp_toy_twohop_decomposition_v0_13.zip
