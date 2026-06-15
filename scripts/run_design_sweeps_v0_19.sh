#!/usr/bin/env bash
set -euo pipefail

# v0.19 design-validation sweeps.
# Goal:
#   (A) calibrate fixed post-introduction HOP_2 budget W_post;
#   (B) sweep HOP_2 introduction step while holding W_post fixed;
#   (C) sweep post-introduction HOP_2 mixture while keeping p_multi_before_intro=0.
#
# Override any variable from the command line, e.g.
#   DEVICE=cuda SEEDS="0 1 2" RUN_WPOST=1 RUN_INTRO=1 RUN_MIX=1 bash scripts/run_design_sweeps_v0_19.sh

DEVICE=${DEVICE:-cuda}
TORCH_THREADS=${TORCH_THREADS:-}
OUT_ROOT=${OUT_ROOT:-runs/design_sweeps_v0_19}

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

# Sweep grids. Keep compact by default; expand SEEDS for final runs.
SEEDS=${SEEDS:-"0 1 2"}
INTRO_EARLY=${INTRO_EARLY:-2500}
INTRO_LATE=${INTRO_LATE:-16000}
WPOST_LIST=${WPOST_LIST:-"2000 4000 6000 8000 10000 12000 13500 16000"}
INTRO_STEPS=${INTRO_STEPS:-"2500 5000 8000 12000 16000 20000"}
PMULTI_LIST=${PMULTI_LIST:-"0.25 0.50 0.75"}

# Choose W_POST after running the W_post calibration. 13500 is the value used in the existing main toy study.
W_POST=${W_POST:-13500}

# Turn sweeps on/off.
RUN_WPOST=${RUN_WPOST:-1}
RUN_INTRO=${RUN_INTRO:-1}
RUN_MIX=${RUN_MIX:-1}
RUN_AGG=${RUN_AGG:-1}
PACK=${PACK:-1}

COMMON_ARGS=(
  --condition late_gate_post
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
  local intro=$4
  local max_steps=$5
  local pmulti=$6
  local rewarm=${7:-0}

  local args=("${COMMON_ARGS[@]}" --out-dir "$out_dir" --seed "$seed" --schedule "$schedule" --intro-step "$intro" --max-steps "$max_steps" --p-multi "$pmulti")
  if [[ "$schedule" == "warmup_cosine" || "$schedule" == warmup_cosine_then_rewarm_constant* ]]; then
    args+=(--t-schedule "$T_SCHEDULE")
  fi
  if [[ "$rewarm" == "1" ]]; then
    args+=(--rewarm-step "$intro" --rewarm-lr "$PEAK_LR")
  fi
  echo "[v0.19] python scripts/run_condition.py ${args[*]}"
  python scripts/run_condition.py "${args[@]}"
}

if [[ "$RUN_WPOST" == "1" ]]; then
  echo "[v0.19] A. W_post calibration: high-update reference schedule, intro=$INTRO_EARLY"
  for wpost in $WPOST_LIST; do
    max_steps=$((INTRO_EARLY + wpost))
    for seed in $SEEDS; do
      run_one "$OUT_ROOT/wpost_calibration/s2_constant/wpost_${wpost}" "$seed" warmup_constant "$INTRO_EARLY" "$max_steps" 0.50 0
      run_one "$OUT_ROOT/wpost_calibration/rewarm_reset/wpost_${wpost}" "$seed" warmup_cosine_then_rewarm_constant_reset_optim "$INTRO_EARLY" "$max_steps" 0.50 1
    done
  done
fi

if [[ "$RUN_INTRO" == "1" ]]; then
  echo "[v0.19] B. Introduction-step sweep with fixed W_POST=$W_POST"
  for intro in $INTRO_STEPS; do
    max_steps=$((intro + W_POST))
    for seed in $SEEDS; do
      run_one "$OUT_ROOT/intro_step_sweep/s1_cosine/intro_${intro}" "$seed" warmup_cosine "$intro" "$max_steps" 0.50 0
      run_one "$OUT_ROOT/intro_step_sweep/s2_constant/intro_${intro}" "$seed" warmup_constant "$intro" "$max_steps" 0.50 0
      run_one "$OUT_ROOT/intro_step_sweep/rewarm_reset/intro_${intro}" "$seed" warmup_cosine_then_rewarm_constant_reset_optim "$intro" "$max_steps" 0.50 1
    done
  done
fi

if [[ "$RUN_MIX" == "1" ]]; then
  echo "[v0.19] C. Mixture-ratio sensitivity at late intro=$INTRO_LATE with fixed W_POST=$W_POST"
  max_steps=$((INTRO_LATE + W_POST))
  for pmulti in $PMULTI_LIST; do
    ptag=${pmulti/./p}
    for seed in $SEEDS; do
      run_one "$OUT_ROOT/mixture_sensitivity/s1_cosine/pmulti_${ptag}" "$seed" warmup_cosine "$INTRO_LATE" "$max_steps" "$pmulti" 0
      run_one "$OUT_ROOT/mixture_sensitivity/s2_constant/pmulti_${ptag}" "$seed" warmup_constant "$INTRO_LATE" "$max_steps" "$pmulti" 0
      run_one "$OUT_ROOT/mixture_sensitivity/rewarm_reset/pmulti_${ptag}" "$seed" warmup_cosine_then_rewarm_constant_reset_optim "$INTRO_LATE" "$max_steps" "$pmulti" 1
    done
  done
fi

if [[ "$RUN_AGG" == "1" ]]; then
  python scripts/aggregate_design_sweeps_v0_19.py --root "$OUT_ROOT" --out-dir "$OUT_ROOT/summary"
  python scripts/plot_design_sweeps_v0_19.py --summary-dir "$OUT_ROOT/summary" --out-dir "$OUT_ROOT/figures"
fi

if [[ "$PACK" == "1" ]]; then
  python scripts/pack_results.py --runs-dir "$OUT_ROOT" --extra "$OUT_ROOT/summary" "$OUT_ROOT/figures" RUN_DESIGN_SWEEPS_V0_19.md --out cp_toy_design_sweeps_v0_19.zip
fi
