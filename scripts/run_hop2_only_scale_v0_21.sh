#!/usr/bin/env bash
set -euo pipefail

# v0.21 HOP_2-only-from-scratch step-budget scale.
# This answers: can the model learn HOP_2 directly, without HOP_1-only pretraining
# and without HOP_1 examples mixed in?
#
# Default: one seed, constant LR, p_multi=1.0, multiple max-step budgets.
# Example:
#   DEVICE=cuda SEEDS="0" STEPS_LIST="20000 40000 80000 120000" bash scripts/run_hop2_only_scale_v0_21.sh

DEVICE=${DEVICE:-cuda}
TORCH_THREADS=${TORCH_THREADS:-}
OUT_ROOT=${OUT_ROOT:-runs/hop2_only_scale_v0_21}

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

# HOP_2-only from scratch: p_multi=1.0 for all batches, condition c0, no intro step.
PMULTI=${PMULTI:-1.0}
SEEDS=${SEEDS:-"0"}
SCHEDULES=${SCHEDULES:-"warmup_constant"}
STEPS_LIST=${STEPS_LIST:-"20000 40000 80000 120000"}

RUN_SCALE=${RUN_SCALE:-1}
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
  --p-multi "$PMULTI"
)
if [[ -n "$TORCH_THREADS" ]]; then
  COMMON_ARGS+=(--torch-threads "$TORCH_THREADS")
fi

run_one() {
  local out_dir=$1
  local seed=$2
  local schedule=$3
  local max_steps=$4
  local args=("${COMMON_ARGS[@]}" --out-dir "$out_dir" --seed "$seed" --schedule "$schedule" --max-steps "$max_steps")
  if [[ "$schedule" == "warmup_cosine" || "$schedule" == warmup_cosine_then_rewarm_constant* ]]; then
    args+=(--t-schedule "$T_SCHEDULE")
  fi
  echo "[v0.21] python scripts/run_condition.py ${args[*]}"
  python scripts/run_condition.py "${args[@]}"
}

if [[ "$RUN_SCALE" == "1" ]]; then
  echo "[v0.21] HOP_2-only from scratch scale: p_multi=$PMULTI, steps={$STEPS_LIST}"
  for schedule in $SCHEDULES; do
    arm="$schedule"
    if [[ "$schedule" == "warmup_constant" ]]; then arm="s2_constant"; fi
    if [[ "$schedule" == "warmup_cosine" ]]; then arm="s1_cosine"; fi
    for steps in $STEPS_LIST; do
      for seed in $SEEDS; do
        run_one "$OUT_ROOT/hop2_only_from_start/$arm/steps_${steps}" "$seed" "$schedule" "$steps"
      done
    done
  done
fi

if [[ "$RUN_AGG" == "1" ]]; then
  python scripts/aggregate_hop2_only_scale_v0_21.py --root "$OUT_ROOT" --out-dir "$OUT_ROOT/summary"
  python scripts/plot_hop2_only_scale_v0_21.py --summary-dir "$OUT_ROOT/summary" --out-dir "$OUT_ROOT/figures"
fi

if [[ "$PACK" == "1" ]]; then
  python scripts/pack_results.py --runs-dir "$OUT_ROOT" --extra "$OUT_ROOT/summary" "$OUT_ROOT/figures" "$OUT_ROOT/tables" RUN_HOP2_ONLY_SCALE_V0_21.md --out cp_toy_hop2_only_scale_v0_21.zip
fi
