#!/usr/bin/env bash
set -euo pipefail

# Small rerun set for E4 if checkpointed models are unavailable.
# It trains the key late arms with checkpoints, then runs transition tracking.
SEEDS=${SEEDS:-"2"}
DEVICE=${DEVICE:-cuda}
OUT_ROOT=${OUT_ROOT:-runs/transition_tracking_reruns_v0_17}
LOG_RANK=${LOG_RANK:-0}

COMMON_ARGS=(
  --condition late_gate_post
  --device "${DEVICE}"
  --intro-step 16000
  --max-steps 36000
  --v-content 64
  --chain-length 8
  --k-max 2
  --p-multi 0.5
  --batch-size 256
  --eval-interval 50
  --eval-batches 16
  --peak-lr 1e-3
  --checkpoint-steps "16000,20000,24000,28000,32000"
  --save-intro-checkpoint
)
if [[ "${LOG_RANK}" == "1" ]]; then
  COMMON_ARGS+=(--log-rank-metrics)
fi

for SEED in ${SEEDS}; do
  echo "=== seed ${SEED}: S1 plateau late ==="
  python scripts/run_condition.py \
    "${COMMON_ARGS[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine \
    --t-schedule 30000 \
    --out-dir "${OUT_ROOT}/seed${SEED}/s1_plateau_late"

  echo "=== seed ${SEED}: S1 longer-cosine late ==="
  python scripts/run_condition.py \
    "${COMMON_ARGS[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine \
    --t-schedule 36000 \
    --out-dir "${OUT_ROOT}/seed${SEED}/s1_longcos_late"

  echo "=== seed ${SEED}: rewarm+reset late ==="
  python scripts/run_condition.py \
    "${COMMON_ARGS[@]}" \
    --seed "${SEED}" \
    --schedule warmup_cosine_then_rewarm_constant_reset_optim \
    --t-schedule 30000 \
    --rewarm-lr 1e-3 \
    --out-dir "${OUT_ROOT}/seed${SEED}/rewarm_reset_late"
done

RUNS_DIR="${OUT_ROOT}" OUT_DIR="${OUT_ROOT}/transition_tracking" \
  BATCH_SIZE=128 EVAL_BATCHES=8 SCORE_BATCHES=4 LENS_BATCHES=4 \
  bash scripts/run_transition_tracking_v0_17.sh

python scripts/pack_results.py \
  --runs-dir "${OUT_ROOT}" \
  --out cp_toy_transition_tracking_reruns_v0_17.zip

echo "wrote cp_toy_transition_tracking_reruns_v0_17.zip"
