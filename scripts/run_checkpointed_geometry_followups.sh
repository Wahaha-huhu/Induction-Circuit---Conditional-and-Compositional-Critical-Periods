#!/usr/bin/env bash
set -euo pipefail

# Checkpointed diagnostic reruns for the conditional-compositional reachability hypothesis.
# These are not intended as a new main sweep. They create intro/final checkpoints so
# run_geometry_probe.py can measure distance, gradient alignment, and interpolation.

SEEDS=${SEEDS:-"1 2"}
DEVICE=${DEVICE:-cuda}
LOG_RANK=${LOG_RANK:-1}
INTRO_STEP=${INTRO_STEP:-16000}
MAX_STEPS=${MAX_STEPS:-36000}
BASE_T_SCHEDULE=${BASE_T_SCHEDULE:-30000}
LONG_T_SCHEDULE=${LONG_T_SCHEDULE:-36000}
PEAK_LR=${PEAK_LR:-1e-3}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCHES=${EVAL_BATCHES:-16}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
OUT_ROOT=${OUT_ROOT:-runs/checkpointed_geometry_followups}
# Pre-step checkpoints. The intro checkpoint is also saved as checkpoint_pre_intro.pt.
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"16000,20000,24000,28000,32000"}

RANK_FLAG=""
if [[ "${LOG_RANK}" == "1" ]]; then
  RANK_FLAG="--log-rank-metrics"
fi

COMMON=(
  --condition late_gate_post
  --device "${DEVICE}"
  --intro-step "${INTRO_STEP}"
  --max-steps "${MAX_STEPS}"
  --v-content 64
  --chain-length 8
  --k-max 2
  --p-multi 0.5
  --batch-size "${BATCH_SIZE}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-batches "${EVAL_BATCHES}"
  --peak-lr "${PEAK_LR}"
  --save-intro-checkpoint
  --checkpoint-steps "${CHECKPOINT_STEPS}"
)

for SEED in ${SEEDS}; do
  echo "=== seed ${SEED}: S1 plateau late ==="
  python scripts/run_condition.py \
    "${COMMON[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine \
    --t-schedule "${BASE_T_SCHEDULE}" \
    ${RANK_FLAG} \
    --out-dir "${OUT_ROOT}/seed${SEED}/s1_plateau_late"

  echo "=== seed ${SEED}: S1 longer-cosine late ==="
  python scripts/run_condition.py \
    "${COMMON[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine \
    --t-schedule "${LONG_T_SCHEDULE}" \
    ${RANK_FLAG} \
    --out-dir "${OUT_ROOT}/seed${SEED}/s1_longcos_late"

  echo "=== seed ${SEED}: S2 constant late ==="
  python scripts/run_condition.py \
    "${COMMON[@]}" \
    --seed "${SEED}" \
    --schedule warmup_constant \
    --t-schedule "${BASE_T_SCHEDULE}" \
    ${RANK_FLAG} \
    --out-dir "${OUT_ROOT}/seed${SEED}/s2_constant_late"

  echo "=== seed ${SEED}: rewarm+reset late ==="
  python scripts/run_condition.py \
    "${COMMON[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine_then_rewarm_constant_reset_optim \
    --rewarm-lr "${PEAK_LR}" \
    --t-schedule "${BASE_T_SCHEDULE}" \
    ${RANK_FLAG} \
    --out-dir "${OUT_ROOT}/seed${SEED}/rewarm_reset_late"
done

python scripts/analyze_transition_shape.py \
  --runs-dir "${OUT_ROOT}" \
  --out "${OUT_ROOT}/transition_shape_summary.csv"

python scripts/summarize_runs.py \
  --runs-dir "${OUT_ROOT}" \
  --out "${OUT_ROOT}/run_summaries.csv"

python scripts/pack_results.py \
  --runs-dir "${OUT_ROOT}" \
  --out cp_toy_checkpointed_geometry_followups.zip

echo "wrote cp_toy_checkpointed_geometry_followups.zip"
