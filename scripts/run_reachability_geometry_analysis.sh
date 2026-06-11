#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-runs/checkpointed_geometry_followups}
TARGET_SUBSTRING=${TARGET_SUBSTRING:-rewarm_reset}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCHES=${EVAL_BATCHES:-8}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/reachability_geometry}
OUT_ZIP=${OUT_ZIP:-cp_toy_reachability_geometry_analysis.zip}
# Reduce these if running on CPU.
OWN_ALPHAS=${OWN_ALPHAS:-"0,0.25,0.5,0.65,0.75,0.85,0.9,0.95,1.0"}
TARGET_DIRECTION_ALPHAS=${TARGET_DIRECTION_ALPHAS:-"0,0.25,0.5,0.75,1.0,1.25"}

python scripts/analyze_reachability_geometry.py \
  --runs-dir "${RUNS_DIR}" \
  --target-substring "${TARGET_SUBSTRING}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batches "${EVAL_BATCHES}" \
  --own-alphas "${OWN_ALPHAS}" \
  --target-direction-alphas "${TARGET_DIRECTION_ALPHAS}" \
  --out-dir "${OUT_DIR}"

python scripts/pack_results.py \
  --runs-dir "${RUNS_DIR}" \
  --out "${OUT_ZIP}"

echo "wrote ${OUT_ZIP}"
