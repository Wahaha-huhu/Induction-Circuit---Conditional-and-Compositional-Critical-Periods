#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/first_binding_patching}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_BATCHES=${NUM_BATCHES:-8}
MAX_RUNS=${MAX_RUNS:-6}
HEAD_PATCHING=${HEAD_PATCHING:-0}
ARMS=${ARMS:-rewarm_reset_late,rewarm_late,s2_constant_late,s1_longcos_late}

mkdir -p "$OUT_DIR"
SELECTED="$OUT_DIR/first_binding_patching_selected_runs.txt"
python scripts/select_ablation_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out "$SELECTED" \
  --arms "$ARMS" \
  --require-model

if [[ "$MAX_RUNS" != "all" ]]; then
  head -n "$MAX_RUNS" "$SELECTED" > "$SELECTED.tmp"
  mv "$SELECTED.tmp" "$SELECTED"
fi

n=0
while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue
  n=$((n+1))
  echo "[first-binding patching] $n $run_dir"
  extra=()
  if [[ "$HEAD_PATCHING" == "1" ]]; then
    extra+=(--head-patching)
  fi
  python scripts/run_first_binding_patching.py "$run_dir" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --out "$run_dir/first_binding_patching.csv" \
    "${extra[@]}"
done < "$SELECTED"

python scripts/aggregate_first_binding_patching.py \
  --runs-dir "$RUNS_DIR" \
  --out-dir "$OUT_DIR"

python scripts/pack_results.py \
  --runs-dir "$OUT_DIR" \
  --out cp_toy_first_binding_patching_v0_16.zip
