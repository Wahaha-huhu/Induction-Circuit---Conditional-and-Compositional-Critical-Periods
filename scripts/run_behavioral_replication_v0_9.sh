#!/usr/bin/env bash
set -euo pipefail

# Multi-seed behavioural replication for the revised v0.9 claim.
# Run from repository root after `python -m pip install -e .`.
#
# Default seeds are 3--7 because seeds 0--2 have already been explored.
# Override with e.g. SEEDS="3" LOG_RANK=0 bash scripts/run_behavioral_replication_v0_9.sh
#
# Arms:
#   s1_late_original: original finite-horizon late S1 run.
#   s1_plateau_late: original cosine horizon, extended into final-LR plateau.
#   s1_longcos_late: extended horizon with stretched cosine decay.
#   s2_constant_late: constant-LR extended late run.
#   rewarm_late: cosine history, high LR restored at intro.
#   rewarm_reset_late: cosine history, high LR restored at intro, AdamW reset.
#   fresh_hop1_s1: selectivity control; fresh independent single-hop under late S1.

SEEDS=${SEEDS:-"3 4 5 6 7"}
DEVICE=${DEVICE:-cuda}
OUT_ROOT=${OUT_ROOT:-runs/behavioral_replication_v0_9}
INTRO_STEP=${INTRO_STEP:-16000}
ORIGINAL_MAX_STEPS=${ORIGINAL_MAX_STEPS:-29500}
EXTENDED_MAX_STEPS=${EXTENDED_MAX_STEPS:-36000}
FRESH_MAX_STEPS=${FRESH_MAX_STEPS:-29500}
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
LOG_RANK=${LOG_RANK:-0}
RUN_FRESH_S2=${RUN_FRESH_S2:-0}

rank_flag=()
if [[ "$LOG_RANK" == "1" ]]; then
  rank_flag=(--log-rank-metrics)
fi

base_args=(
  --device "$DEVICE"
  --intro-step "$INTRO_STEP"
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

run_late_gate() {
  local seed=$1
  local arm=$2
  local schedule=$3
  local max_steps=$4
  local t_schedule=$5
  shift 5
  echo "=== seed ${seed}: ${arm} ==="
  python scripts/run_condition.py \
    --condition late_gate_post \
    "${base_args[@]}" \
    --seed "$seed" \
    --max-steps "$max_steps" \
    --schedule "$schedule" \
    --t-schedule "$t_schedule" \
    --out-dir "${OUT_ROOT}/s${seed}/${arm}" \
    "${rank_flag[@]}" \
    "$@"
}

run_fresh() {
  local seed=$1
  local arm=$2
  local schedule=$3
  local t_schedule=$4
  echo "=== seed ${seed}: ${arm} ==="
  python scripts/run_condition.py \
    --condition fresh_singlehop_late \
    "${base_args[@]}" \
    --seed "$seed" \
    --max-steps "$FRESH_MAX_STEPS" \
    --schedule "$schedule" \
    --t-schedule "$t_schedule" \
    --out-dir "${OUT_ROOT}/s${seed}/${arm}" \
    "${rank_flag[@]}"
}

for SEED in $SEEDS; do
  run_late_gate "$SEED" s1_late_original warmup_cosine "$ORIGINAL_MAX_STEPS" "$BASE_T_SCHEDULE"
  run_late_gate "$SEED" s1_plateau_late warmup_cosine "$EXTENDED_MAX_STEPS" "$BASE_T_SCHEDULE"
  run_late_gate "$SEED" s1_longcos_late warmup_cosine "$EXTENDED_MAX_STEPS" "$LONG_T_SCHEDULE"
  run_late_gate "$SEED" s2_constant_late warmup_constant "$EXTENDED_MAX_STEPS" "$BASE_T_SCHEDULE"
  run_late_gate "$SEED" rewarm_late warmup_cosine_then_rewarm_constant "$EXTENDED_MAX_STEPS" "$BASE_T_SCHEDULE" --rewarm-lr "$REWARM_LR"
  run_late_gate "$SEED" rewarm_reset_late warmup_cosine_then_rewarm_constant_reset_optim "$EXTENDED_MAX_STEPS" "$BASE_T_SCHEDULE" --rewarm-lr "$REWARM_LR"
  run_fresh "$SEED" fresh_hop1_s1 warmup_cosine "$BASE_T_SCHEDULE"
  if [[ "$RUN_FRESH_S2" == "1" ]]; then
    run_fresh "$SEED" fresh_hop1_s2 warmup_constant "$BASE_T_SCHEDULE"
  fi

done

python scripts/summarize_runs.py \
  --runs-dir "$OUT_ROOT" \
  --out "${OUT_ROOT}/all_run_summaries.csv"

python scripts/analyze_transition_shape.py \
  --runs-dir "$OUT_ROOT" \
  --out "${OUT_ROOT}/transition_shape_summary.csv"

python scripts/aggregate_behavioral_results.py \
  --runs-dir "$OUT_ROOT" \
  --out-dir "${OUT_ROOT}/aggregate"

python scripts/pack_results.py \
  --runs-dir "$OUT_ROOT" \
  --extra "${OUT_ROOT}/aggregate" "${OUT_ROOT}/transition_shape_summary.csv" "${OUT_ROOT}/all_run_summaries.csv" \
  --out cp_toy_behavioral_replication_v0_9.zip

echo "Wrote cp_toy_behavioral_replication_v0_9.zip"
