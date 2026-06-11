#!/usr/bin/env bash
set -euo pipefail

# Extended late-introduction follow-ups for ambiguous C5/C5b rescue arms.
# Run from the repository root after `python -m pip install -e .`.
#
# Usage examples:
#   bash scripts/run_extended_followups.sh
#   SEEDS="1 2" DEVICE=cuda LOG_RANK=0 bash scripts/run_extended_followups.sh
#   SEEDS="2" MAX_STEPS=36000 EVAL_INTERVAL=100 bash scripts/run_extended_followups.sh
#
# The two S1 arms answer different questions:
#   s1_plateau_late: original S1 horizon; after t_schedule, LR is final_lr plateau.
#   s1_longcos_late: longer cosine horizon; LR decay is stretched to max_steps.

SEEDS=${SEEDS:-"1 2"}
DEVICE=${DEVICE:-cuda}
OUT_ROOT=${OUT_ROOT:-runs/extended_followups}
INTRO_STEP=${INTRO_STEP:-16000}
MAX_STEPS=${MAX_STEPS:-36000}
BASE_T_SCHEDULE=${BASE_T_SCHEDULE:-30000}
LONG_T_SCHEDULE=${LONG_T_SCHEDULE:-36000}
V_CONTENT=${V_CONTENT:-64}
CHAIN_LENGTH=${CHAIN_LENGTH:-8}
K_MAX=${K_MAX:-2}
P_MULTI=${P_MULTI:-0.5}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_BATCHES=${EVAL_BATCHES:-16}
PEAK_LR=${PEAK_LR:-1e-3}
FINAL_LR=${FINAL_LR:-5e-6}
REWARM_LR=${REWARM_LR:-1e-3}
LOG_RANK=${LOG_RANK:-1}

rank_flag=()
if [[ "$LOG_RANK" == "1" ]]; then
  rank_flag=(--log-rank-metrics)
fi

common_args=(
  --condition late_gate_post
  --device "$DEVICE"
  --intro-step "$INTRO_STEP"
  --max-steps "$MAX_STEPS"
  --v-content "$V_CONTENT"
  --chain-length "$CHAIN_LENGTH"
  --k-max "$K_MAX"
  --p-multi "$P_MULTI"
  --batch-size "$BATCH_SIZE"
  --eval-interval "$EVAL_INTERVAL"
  --eval-batches "$EVAL_BATCHES"
  --peak-lr "$PEAK_LR"
  --final-lr "$FINAL_LR"
)

for SEED in $SEEDS; do
  echo "=== seed ${SEED}: S1 original-horizon plateau extension ==="
  python scripts/run_condition.py \
    "${common_args[@]}" \
    --seed "$SEED" \
    --schedule warmup_cosine \
    --t-schedule "$BASE_T_SCHEDULE" \
    --out-dir "${OUT_ROOT}/s${SEED}/s1_plateau_late" \
    "${rank_flag[@]}"

  echo "=== seed ${SEED}: S1 longer-cosine extension ==="
  python scripts/run_condition.py \
    "${common_args[@]}" \
    --seed "$SEED" \
    --schedule warmup_cosine \
    --t-schedule "$LONG_T_SCHEDULE" \
    --out-dir "${OUT_ROOT}/s${SEED}/s1_longcos_late" \
    "${rank_flag[@]}"

  echo "=== seed ${SEED}: S2 constant extended ==="
  python scripts/run_condition.py \
    "${common_args[@]}" \
    --seed "$SEED" \
    --schedule warmup_constant \
    --t-schedule "$BASE_T_SCHEDULE" \
    --out-dir "${OUT_ROOT}/s${SEED}/s2_constant_late" \
    "${rank_flag[@]}"

  echo "=== seed ${SEED}: cosine-history rewarm extended ==="
  python scripts/run_condition.py \
    "${common_args[@]}" \
    --seed "$SEED" \
    --schedule warmup_cosine_then_rewarm_constant \
    --t-schedule "$BASE_T_SCHEDULE" \
    --rewarm-lr "$REWARM_LR" \
    --out-dir "${OUT_ROOT}/s${SEED}/rewarm_late" \
    "${rank_flag[@]}"

  echo "=== seed ${SEED}: cosine-history rewarm+optimizer-reset extended ==="
  python scripts/run_condition.py \
    "${common_args[@]}" \
    --seed "$SEED" \
    --schedule warmup_cosine_then_rewarm_constant_reset_optim \
    --t-schedule "$BASE_T_SCHEDULE" \
    --rewarm-lr "$REWARM_LR" \
    --out-dir "${OUT_ROOT}/s${SEED}/rewarm_reset_late" \
    "${rank_flag[@]}"
done

python scripts/summarize_runs.py \
  --runs-dir "$OUT_ROOT" \
  --out "${OUT_ROOT}/extended_followup_summaries.csv"

python scripts/pack_results.py \
  --runs-dir "$OUT_ROOT" \
  --out cp_toy_extended_followups.zip

echo "Wrote cp_toy_extended_followups.zip"
