#!/usr/bin/env bash
set -euo pipefail

# E4: analyze checkpointed runs. This does not train; it evaluates saved checkpoints.
RUNS_DIR=${RUNS_DIR:-runs/checkpointed_geometry_followups}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/transition_tracking}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_BATCHES=${EVAL_BATCHES:-8}
SCORE_BATCHES=${SCORE_BATCHES:-4}
LENS_BATCHES=${LENS_BATCHES:-4}
MAX_RUNS=${MAX_RUNS:-all}

mkdir -p "${OUT_DIR}"

mapfile -t RUN_DIRS < <(find "${RUNS_DIR}" -type f \( -name 'checkpoint_pre_intro.pt' -o -name 'checkpoint_pre_step_*.pt' \) -printf '%h\n' | sort -u)
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No checkpointed run dirs found under ${RUNS_DIR}."
  echo "Either set RUNS_DIR to the directory containing checkpoint_pre_step_*.pt files,"
  echo "or run scripts/run_transition_tracking_reruns_v0_17.sh to generate a small checkpointed set."
  exit 1
fi

if [[ "${MAX_RUNS}" != "all" ]]; then
  RUN_DIRS=("${RUN_DIRS[@]:0:${MAX_RUNS}}")
fi

echo "Analyzing ${#RUN_DIRS[@]} checkpointed run dirs from ${RUNS_DIR}"
for RD in "${RUN_DIRS[@]}"; do
  SAFE=$(echo "${RD#${RUNS_DIR}/}" | tr '/ ' '__')
  OUT="${OUT_DIR}/${SAFE}_transition_tracking.csv"
  echo "==> ${RD}"
  python scripts/run_checkpoint_transition_tracking.py "${RD}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batches "${EVAL_BATCHES}" \
    --score-batches "${SCORE_BATCHS:-${SCORE_BATCHES}}" \
    --lens-batches "${LENS_BATCHES}" \
    --out "${OUT}"
done

python scripts/aggregate_transition_tracking_v0_17.py \
  --tracking-dir "${OUT_DIR}" \
  --out-per-checkpoint "${OUT_DIR}/transition_tracking_per_checkpoint.csv" \
  --out-summary "${OUT_DIR}/transition_tracking_summary.csv" \
  --out-report "${OUT_DIR}/transition_tracking_report.md"

python scripts/pack_results.py \
  --runs-dir "${OUT_DIR}" \
  --out cp_toy_transition_tracking_v0_17.zip

echo "wrote cp_toy_transition_tracking_v0_17.zip"
