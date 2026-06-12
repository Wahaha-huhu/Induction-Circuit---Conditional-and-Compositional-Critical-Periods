#!/usr/bin/env bash
set -euo pipefail

RUNS_DIR=${RUNS_DIR:-runs/behavioral_replication_v0_9}
OUT_DIR=${OUT_DIR:-${RUNS_DIR}/mode_specific_probe}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-256}
TRAIN_BATCHES=${TRAIN_BATCHES:-16}
EVAL_BATCHES=${EVAL_BATCHES:-8}
PROBE_STEPS=${PROBE_STEPS:-300}
PROBE_LR=${PROBE_LR:-1e-2}
SUCCESS_ARMS=${SUCCESS_ARMS:-s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late}
FAILED_ARMS=${FAILED_ARMS:-s1_late_original,s1_plateau_late}
MAX_SUCCESS=${MAX_SUCCESS:-all}
MAX_FAILED=${MAX_FAILED:-all}
SKIP_LINEAR_PROBE=${SKIP_LINEAR_PROBE:-0}
MODES=${MODES:-hop1_prompt,hop2_prompt}

mkdir -p "$OUT_DIR/per_run"
SELECTED_TXT="$OUT_DIR/mode_probe_selected_runs.txt"
SELECTED_JSONL="$OUT_DIR/mode_probe_selected_runs.jsonl"

python scripts/select_probe_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out-txt "$SELECTED_TXT" \
  --out-jsonl "$SELECTED_JSONL" \
  --success-arms "$SUCCESS_ARMS" \
  --failed-arms "$FAILED_ARMS" \
  --max-success "$MAX_SUCCESS" \
  --max-failed "$MAX_FAILED" \
  --require-model

n=0
while IFS= read -r rec; do
  [[ -z "$rec" ]] && continue
  run_dir=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_dir"])' "$rec")
  group=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("probe_group","unknown"))' "$rec")
  arm=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("arm","arm"))' "$rec")
  seed=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("seed","seed"))' "$rec")
  n=$((n+1))
  out="$OUT_DIR/per_run/${group}_${arm}_seed${seed}_mode_specific_probe.csv"
  echo "[mode-specific probe] $n group=$group arm=$arm seed=$seed $run_dir"
  extra=()
  if [[ "$SKIP_LINEAR_PROBE" == "1" ]]; then
    extra+=(--skip-linear-probe)
  fi
  python scripts/run_mode_specific_probe.py "$run_dir" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --train-batches "$TRAIN_BATCHES" \
    --eval-batches "$EVAL_BATCHES" \
    --probe-steps "$PROBE_STEPS" \
    --probe-lr "$PROBE_LR" \
    --probe-group "$group" \
    --modes "$MODES" \
    --out "$out" \
    "${extra[@]}"
done < "$SELECTED_JSONL"

python scripts/aggregate_mode_specific_probe.py \
  --runs-dir "$OUT_DIR" \
  --out-dir "$OUT_DIR"

python scripts/pack_results.py \
  --runs-dir "$OUT_DIR" \
  --out cp_toy_mode_specific_probe_v0_15.zip
