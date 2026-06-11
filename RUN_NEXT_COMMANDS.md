# Next Commands: Replication, Specificity Controls, and Mechanism Logging

Use this with `cp_toy_impl_v0_4`.

## Setup

```bash
unzip -o cp_toy_impl_v0_4.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

## Core calibrated values

```bash
V=64
M=8
K=2
BATCH=256
EVAL_BATCHES=16
PEAK_LR=1e-3
T_SCHEDULE=30000
INTRO_EARLY=2500
INTRO_MID=8000
INTRO_LATE=16000
MAX_EARLY=16000
MAX_MID=21500
MAX_LATE=29500
P_MULTI=0.5
```

## Multi-seed replication with rank/consolidation logging

Run at least seeds 1 and 2.

```bash
for SEED in 1 2; do
  # S1 early reference
  python scripts/run_condition.py \
    --condition late_gate_early \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_EARLY} \
    --max-steps ${MAX_EARLY} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_cosine \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/s1_early

  # S1 mid
  python scripts/run_condition.py \
    --condition late_gate_post \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_MID} \
    --max-steps ${MAX_MID} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_cosine \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/s1_mid

  # S1 late failure candidate
  python scripts/run_condition.py \
    --condition late_gate_post \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_LATE} \
    --max-steps ${MAX_LATE} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_cosine \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/s1_late

  # S2 constant late rescue
  python scripts/run_condition.py \
    --condition late_gate_post \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_LATE} \
    --max-steps ${MAX_LATE} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_constant \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/constant_late

  # C5b rewarm late
  python scripts/run_condition.py \
    --condition late_gate_post \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_LATE} \
    --max-steps ${MAX_LATE} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_cosine_then_rewarm_constant \
    --rewarm-lr ${PEAK_LR} \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/rewarm_late

  # C5b rewarm + AdamW state reset late
  python scripts/run_condition.py \
    --condition late_gate_post \
    --seed ${SEED} \
    --device cuda \
    --intro-step ${INTRO_LATE} \
    --max-steps ${MAX_LATE} \
    --t-schedule ${T_SCHEDULE} \
    --schedule warmup_cosine_then_rewarm_constant_reset_optim \
    --rewarm-lr ${PEAK_LR} \
    --v-content ${V} \
    --chain-length ${M} \
    --k-max ${K} \
    --p-multi ${P_MULTI} \
    --batch-size ${BATCH} \
    --eval-interval 50 \
    --eval-batches ${EVAL_BATCHES} \
    --peak-lr ${PEAK_LR} \
    --log-rank-metrics \
    --out-dir runs/replication_s${SEED}/rewarm_reset_late

done
```

## Specificity controls

Run these for seed 0 first, then replicate if useful.

### C7 late QUERY_B routing under S1

```bash
python scripts/run_condition.py \
  --condition c7_query_b_late \
  --seed 0 \
  --device cuda \
  --intro-step ${INTRO_LATE} \
  --max-steps ${MAX_LATE} \
  --t-schedule ${T_SCHEDULE} \
  --schedule warmup_cosine \
  --v-content ${V} \
  --chain-length ${M} \
  --k-max ${K} \
  --batch-size ${BATCH} \
  --eval-interval 50 \
  --eval-batches ${EVAL_BATCHES} \
  --peak-lr ${PEAK_LR} \
  --log-rank-metrics \
  --out-dir runs/specificity/c7_query_b_s1_late
```

### C7 late QUERY_B routing under constant LR

```bash
python scripts/run_condition.py \
  --condition c7_query_b_late \
  --seed 0 \
  --device cuda \
  --intro-step ${INTRO_LATE} \
  --max-steps ${MAX_LATE} \
  --t-schedule ${T_SCHEDULE} \
  --schedule warmup_constant \
  --v-content ${V} \
  --chain-length ${M} \
  --k-max ${K} \
  --batch-size ${BATCH} \
  --eval-interval 50 \
  --eval-batches ${EVAL_BATCHES} \
  --peak-lr ${PEAK_LR} \
  --log-rank-metrics \
  --out-dir runs/specificity/c7_query_b_constant_late
```

### Fresh independent single-hop late under S1

This trains HOP_1 on the base token pool before intro, then HOP_1 on the fresh token pool after intro.

```bash
python scripts/run_condition.py \
  --condition fresh_singlehop_late \
  --seed 0 \
  --device cuda \
  --intro-step ${INTRO_LATE} \
  --max-steps ${MAX_LATE} \
  --t-schedule ${T_SCHEDULE} \
  --schedule warmup_cosine \
  --v-content ${V} \
  --chain-length ${M} \
  --k-max ${K} \
  --batch-size ${BATCH} \
  --eval-interval 50 \
  --eval-batches ${EVAL_BATCHES} \
  --peak-lr ${PEAK_LR} \
  --log-rank-metrics \
  --out-dir runs/specificity/fresh_singlehop_s1_late
```

### Fresh independent single-hop late under constant LR

```bash
python scripts/run_condition.py \
  --condition fresh_singlehop_late \
  --seed 0 \
  --device cuda \
  --intro-step ${INTRO_LATE} \
  --max-steps ${MAX_LATE} \
  --t-schedule ${T_SCHEDULE} \
  --schedule warmup_constant \
  --v-content ${V} \
  --chain-length ${M} \
  --k-max ${K} \
  --batch-size ${BATCH} \
  --eval-interval 50 \
  --eval-batches ${EVAL_BATCHES} \
  --peak-lr ${PEAK_LR} \
  --log-rank-metrics \
  --out-dir runs/specificity/fresh_singlehop_constant_late
```

## Phase-controlled cyclic arm

```bash
python scripts/run_condition.py \
  --condition late_gate_post \
  --seed 0 \
  --device cuda \
  --intro-step ${INTRO_LATE} \
  --phase-align-intro-to-peak \
  --max-steps ${MAX_LATE} \
  --t-schedule ${T_SCHEDULE} \
  --schedule warmup_cyclic \
  --cycle-length 2000 \
  --v-content ${V} \
  --chain-length ${M} \
  --k-max ${K} \
  --p-multi ${P_MULTI} \
  --batch-size ${BATCH} \
  --eval-interval 50 \
  --eval-batches ${EVAL_BATCHES} \
  --peak-lr ${PEAK_LR} \
  --log-rank-metrics \
  --out-dir runs/cyclic_phase_control
```

## Top-2 ablation on successful staged/early models

```bash
python scripts/run_ablation_eval.py <RUN_DIR> --device cuda --top-k 2 --batch-size 256 --eval-batches 32
```

## Pack results

```bash
python scripts/pack_results.py --runs-dir runs --out cp_toy_replication_specificity.zip
```

## v0.5 summary utilities

`run_condition.py` now prints a final summary and writes `summary.json` automatically.

For old runs, run:

```bash
python scripts/analyze_run.py <run_dir>
```

For all runs under a directory, run:

```bash
python scripts/summarize_runs.py --runs-dir runs --out runs/all_run_summaries.csv
```
