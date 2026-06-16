#!/usr/bin/env bash
set -euo pipefail

# v0.20 mixed-from-start feasibility sweeps.
# This asks whether the model can learn HOP_1 and HOP_2 when both are present
# from step 0, without a staged HOP_1-only pretraining phase.
#
# Example final run:
#   DEVICE=cuda SEEDS="0 1 2 3 4" RUN_PMULTI=1 RUN_LONG=1 bash scripts/run_mixed_from_start_sweep_v0_20.sh

DEVICE=${DEVICE:-cuda}
TORCH_THREADS=${TORCH_THREADS:-}
OUT_ROOT=${OUT_ROOT:-runs/mixed_from_start_sweeps_v0_20}

# Calibrated toy setting used in the main experiments.
V_CONTENT=${V_CONTENT:-64}
CHAIN_LENGTH=${CHAIN_LENGTH:-8}
K_MAX=${K_MAX:-2}
D_MODEL=${D_MODEL:-128}
N_LAYERS=${N_LAYERS:-4}
N_HEADS=${N_HEADS:-4}
D_MLP=${D_MLP:-256}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCHES=${EVAL_BATCHES:-8}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
PEAK_LR=${PEAK_LR:-0.001}
FINAL_LR=${FINAL_LR:-0.00003}
WARMUP_STEPS=${WARMUP_STEPS:-500}
T_SCHEDULE=${T_SCHEDULE:-30000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
P_DYNAMIC=${P_DYNAMIC:-1.0}

# Main mixed-from-start sweep. p=0 is HOP_1-only; p=1 is HOP_2-only.
SEEDS=${SEEDS:-"0 1 2"}
PMULTI_LIST=${PMULTI_LIST:-"0.0 0.1 0.25 0.5 0.75 1.0"}
SCHEDULES=${SCHEDULES:-"warmup_constant"}
MAX_STEPS=${MAX_STEPS:-20000}

# Long p_multi=0.5 run(s). Keep this separate so it can be run even if the full sweep is skipped.
LONG_SEEDS=${LONG_SEEDS:-"0"}
LONG_SCHEDULES=${LONG_SCHEDULES:-"warmup_constant"}
LONG_STEPS=${LONG_STEPS:-80000}
LONG_PMULTI=${LONG_PMULTI:-0.5}

RUN_PMULTI=${RUN_PMULTI:-1}
RUN_LONG=${RUN_LONG:-1}
RUN_AGG=${RUN_AGG:-1}
PACK=${PACK:-1}

COMMON_ARGS=(
  --condition c0
  --device "$DEVICE"
  --v-content "$V_CONTENT"
  --chain-length "$CHAIN_LENGTH"
  --k-max "$K_MAX"
  --d-model "$D_MODEL"
  --n-layers "$N_LAYERS"
  --n-heads "$N_HEADS"
  --d-mlp "$D_MLP"
  --batch-size "$BATCH_SIZE"
  --eval-batches "$EVAL_BATCHES"
  --eval-interval "$EVAL_INTERVAL"
  --peak-lr "$PEAK_LR"
  --final-lr "$FINAL_LR"
  --warmup-steps "$WARMUP_STEPS"
  --weight-decay "$WEIGHT_DECAY"
  --p-dynamic-high "$P_DYNAMIC"
)
if [[ -n "$TORCH_THREADS" ]]; then
  COMMON_ARGS+=(--torch-threads "$TORCH_THREADS")
fi

run_one() {
  local out_dir=$1
  local seed=$2
  local schedule=$3
  local max_steps=$4
  local pmulti=$5
  local args=("${COMMON_ARGS[@]}" --out-dir "$out_dir" --seed "$seed" --schedule "$schedule" --max-steps "$max_steps" --p-multi "$pmulti")
  if [[ "$schedule" == "warmup_cosine" || "$schedule" == warmup_cosine_then_rewarm_constant* ]]; then
    args+=(--t-schedule "$T_SCHEDULE")
  fi
  echo "[v0.20] python scripts/run_condition.py ${args[*]}"
  python scripts/run_condition.py "${args[@]}"
}

if [[ "$RUN_PMULTI" == "1" ]]; then
  echo "[v0.20] A. Mixed-from-start p_multi sweep, max_steps=$MAX_STEPS"
  for schedule in $SCHEDULES; do
    arm="$schedule"
    if [[ "$schedule" == "warmup_constant" ]]; then arm="s2_constant"; fi
    if [[ "$schedule" == "warmup_cosine" ]]; then arm="s1_cosine"; fi
    for pmulti in $PMULTI_LIST; do
      ptag=${pmulti/./p}
      for seed in $SEEDS; do
        run_one "$OUT_ROOT/pmulti_sweep_from_start/$arm/pmulti_${ptag}" "$seed" "$schedule" "$MAX_STEPS" "$pmulti"
      done
    done
  done
fi

if [[ "$RUN_LONG" == "1" ]]; then
  echo "[v0.20] B. Long mixed-from-start p_multi=$LONG_PMULTI, max_steps=$LONG_STEPS"
  for schedule in $LONG_SCHEDULES; do
    arm="$schedule"
    if [[ "$schedule" == "warmup_constant" ]]; then arm="s2_constant"; fi
    if [[ "$schedule" == "warmup_cosine" ]]; then arm="s1_cosine"; fi
    ptag=${LONG_PMULTI/./p}
    for seed in $LONG_SEEDS; do
      run_one "$OUT_ROOT/long_pmulti_0p5/$arm/p0p5_steps_${LONG_STEPS}" "$seed" "$schedule" "$LONG_STEPS" "$LONG_PMULTI"
    done
  done
fi

if [[ "$RUN_AGG" == "1" ]]; then
  python scripts/aggregate_mixed_from_start_sweep_v0_20.py --root "$OUT_ROOT" --out-dir "$OUT_ROOT/summary"
  python scripts/plot_mixed_from_start_sweep_v0_20.py --summary-dir "$OUT_ROOT/summary" --out-dir "$OUT_ROOT/figures"
fi

if [[ "$PACK" == "1" ]]; then
  python scripts/pack_results.py --runs-dir "$OUT_ROOT" --extra "$OUT_ROOT/summary" "$OUT_ROOT/figures" "$OUT_ROOT/tables" RUN_MIXED_FROM_START_SWEEP_V0_20.md --out cp_toy_mixed_from_start_sweep_v0_20.zip
fi
