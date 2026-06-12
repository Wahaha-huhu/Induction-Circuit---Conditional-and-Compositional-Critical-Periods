#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/head_concentration}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCHES=${EVAL_BATCHES:-8}
MEAN_BATCHES=${MEAN_BATCHES:-4}
ARMS=${ARMS:-s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late}

python scripts/analyze_head_concentration.py \
  --runs-dir "$RUNS_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --eval-batches "$EVAL_BATCHES" \
  --mean-batches "$MEAN_BATCHES" \
  --arms "$ARMS"

python scripts/pack_results.py \
  --runs-dir "$OUT_DIR" \
  --out cp_toy_head_concentration_v0_12.zip
