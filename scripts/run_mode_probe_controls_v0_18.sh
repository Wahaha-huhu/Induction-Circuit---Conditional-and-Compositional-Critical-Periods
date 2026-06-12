#!/usr/bin/env bash
set -euo pipefail

# v0.18-C: mode-specific probe with shuffled-label controls and failure-mode generality.
RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/mode_probe_controls_v0_18}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-256}
TRAIN_BATCHES=${TRAIN_BATCHES:-16}
EVAL_BATCHES=${EVAL_BATCHES:-8}
PROBE_STEPS=${PROBE_STEPS:-300}
MAX_SUCCESS=${MAX_SUCCESS:-all}
MAX_FAILED=${MAX_FAILED:-all}

# Include S2 constant and S1 longcos as both possible success and possible failure/partial arms.
SUCCESS_ARMS=${SUCCESS_ARMS:-s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late}
FAILED_ARMS=${FAILED_ARMS:-s1_late_original,s1_plateau_late,s1_longcos_late,s2_constant_late}

mkdir -p "${OUT_DIR}"
SELECTED="${OUT_DIR}/probe_control_selected_runs.txt"
python scripts/select_probe_runs.py \
  --runs-dir "${RUNS_DIR}" \
  --out-txt "${SELECTED}" \
  --success-arms "${SUCCESS_ARMS}" \
  --failed-arms "${FAILED_ARMS}" \
  --require-model \
  --max-success "${MAX_SUCCESS}" \
  --max-failed "${MAX_FAILED}"

python scripts/run_mode_probe_controls.py \
  --selected-runs "${SELECTED}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --train-batches "${TRAIN_BATCHES}" \
  --eval-batches "${EVAL_BATCHES}" \
  --probe-steps "${PROBE_STEPS}" \
  --out "${OUT_DIR}/mode_probe_controls_all_layers.csv"

python scripts/aggregate_mode_probe_controls_v0_18.py \
  --controls-csv "${OUT_DIR}/mode_probe_controls_all_layers.csv" \
  --out-layer "${OUT_DIR}/mode_probe_controls_layer_summary.csv" \
  --out-run "${OUT_DIR}/mode_probe_controls_per_run.csv" \
  --out-report "${OUT_DIR}/mode_probe_controls_report.md"

python scripts/pack_results.py \
  --runs-dir "${OUT_DIR}" \
  --out cp_toy_mode_probe_controls_v0_18.zip

echo "wrote cp_toy_mode_probe_controls_v0_18.zip"
