#!/usr/bin/env bash
set -euo pipefail

# v0.18-A/B: checkpointed reorganisation + candidate-set analysis.
# This does not train. It analyzes saved checkpoints from the checkpointed follow-up runs.
RUNS_DIR=${RUNS_DIR:-runs/checkpointed_geometry_followups}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/reorganisation_candidate_v0_18}
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
  echo "Use v0.17 checkpointed reruns first, or set RUNS_DIR to a directory containing checkpoint_pre_step_*.pt files."
  exit 1
fi
if [[ "${MAX_RUNS}" != "all" ]]; then
  RUN_DIRS=("${RUN_DIRS[@]:0:${MAX_RUNS}}")
fi

echo "Analyzing ${#RUN_DIRS[@]} checkpointed run dirs from ${RUNS_DIR}"
for RD in "${RUN_DIRS[@]}"; do
  SAFE=$(echo "${RD#${RUNS_DIR}/}" | tr '/ ' '__')
  OUT="${OUT_DIR}/${SAFE}_reorganisation_candidate.csv"
  echo "==> ${RD}"
  python scripts/run_reorganisation_candidate_analysis.py "${RD}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batches "${EVAL_BATCHES}" \
    --score-batches "${SCORE_BATCHES}" \
    --lens-batches "${LENS_BATCHES}" \
    --out "${OUT}"
done

python scripts/aggregate_reorganisation_candidate_v0_18.py \
  --tracking-dir "${OUT_DIR}" \
  --out-per-checkpoint "${OUT_DIR}/reorganisation_candidate_per_checkpoint.csv" \
  --out-summary "${OUT_DIR}/reorganisation_candidate_summary.csv" \
  --out-report "${OUT_DIR}/reorganisation_candidate_report.md"

python scripts/pack_results.py \
  --runs-dir "${OUT_DIR}" \
  --out cp_toy_reorganisation_candidate_v0_18.zip

echo "wrote cp_toy_reorganisation_candidate_v0_18.zip"
