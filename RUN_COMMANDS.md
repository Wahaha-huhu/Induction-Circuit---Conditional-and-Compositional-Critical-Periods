# Command Lines for First Cascade-Tier Runs

Run from inside the unzipped `cp_toy_impl` directory.

```bash
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

For 4090/A100 GPU runs, start with the following commands. They are intentionally cascade-first: C0, C1, C3, C7, S1 withholding, late-S1 gate. Do not run the full S2/S3 schedule tier until the late-S1 gate shows impaired late learning.

## 0. Optional CPU smoke test

```bash
python scripts/run_condition.py \
  --condition c0 \
  --device cpu \
  --torch-threads 1 \
  --max-steps 20 \
  --batch-size 8 \
  --eval-batches 1 \
  --eval-interval 10 \
  --v-content 64 \
  --chain-length 8 \
  --k-max 2 \
  --d-model 32 \
  --n-layers 2 \
  --n-heads 2 \
  --d-mlp 64 \
  --out-dir runs_smoke
```

## 1. C0 baseline pilot

```bash
python scripts/run_condition.py \
  --condition c0 \
  --seed 0 \
  --device cuda \
  --max-steps 4000 \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/cascade

python scripts/analyze_run.py runs/cascade/c0_seed0
```

## 2. C3 ablation on C0

```bash
python scripts/run_ablation_eval.py \
  runs/cascade/c0_seed0 \
  --device cuda \
  --top-k 1 \
  --batch-size 128 \
  --eval-batches 16
```

## 3. C1 prerequisite-delay pilot

Use an initial switch step such as 1200. After inspecting C0, replace this with the calibrated rule.

```bash
python scripts/run_condition.py \
  --condition c1 \
  --seed 0 \
  --device cuda \
  --max-steps 4000 \
  --dynamic-switch-step 1200 \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/cascade

python scripts/analyze_run.py runs/cascade/c1_seed0
```

## 4. C7 non-formative query-marker control

```bash
python scripts/run_condition.py \
  --condition c7_query_b \
  --seed 0 \
  --device cuda \
  --max-steps 2000 \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/cascade

python scripts/analyze_run.py runs/cascade/c7_query_b_seed0
```

## 5. S1 withholding calibration

This estimates prerequisite formation and consolidation under `p_multi = 0`.

```bash
python scripts/run_condition.py \
  --condition withhold \
  --seed 0 \
  --device cuda \
  --max-steps 12000 \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/gate

python scripts/analyze_run.py runs/gate/withhold_seed0
```

## 6. Late-S1 gate

You need two introduction steps from the withholding calibration. For an initial pilot, replace these placeholders manually:

- `<T_EARLY>`: shortly after the prerequisite forms.
- `<T_LATE>`: post-consolidation / late probe.
- `W_POST=3000` by design.

Example placeholder values are shown below. Replace them after inspecting the withholding run.

```bash
T_EARLY=1500
T_LATE=9000
W_POST=3000

python scripts/run_condition.py \
  --condition late_gate_early \
  --seed 0 \
  --device cuda \
  --intro-step ${T_EARLY} \
  --max-steps $((T_EARLY + W_POST)) \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/gate

python scripts/run_condition.py \
  --condition late_gate_post \
  --seed 0 \
  --device cuda \
  --intro-step ${T_LATE} \
  --max-steps $((T_LATE + W_POST)) \
  --t-schedule 20000 \
  --schedule warmup_cosine \
  --batch-size 128 \
  --eval-interval 50 \
  --eval-batches 8 \
  --out-dir runs/gate

python scripts/compute_late_gate.py \
  --early-run runs/gate/late_gate_early_seed0 \
  --late-run runs/gate/late_gate_post_seed0 \
  | tee runs/gate/gate_decision_seed0.json
```

## 7. Pack results to send back

Without model checkpoints:

```bash
python scripts/pack_results.py \
  --runs-dir runs \
  --out cp_toy_results_seed0.zip
```

With model checkpoints included:

```bash
python scripts/pack_results.py \
  --runs-dir runs \
  --include-models \
  --out cp_toy_results_seed0_with_models.zip
```

Send back the zip file plus any terminal error logs if a run fails before creating a run directory.


## C5b-1 rewarm control

This schedule follows `warmup_cosine` before `--intro-step`, then switches to a constant `--rewarm-lr` from `--intro-step` onward. Use this to test whether S1 late failure is due to low instantaneous/update LR.

```bash
python scripts/run_condition.py \
  --condition late_gate_post \
  --seed 0 \
  --device cuda \
  --intro-step 16000 \
  --max-steps 29500 \
  --t-schedule 30000 \
  --schedule warmup_cosine_then_rewarm_constant \
  --rewarm-lr 1e-3 \
  --v-content 64 \
  --chain-length 8 \
  --k-max 2 \
  --p-multi 0.5 \
  --batch-size 256 \
  --eval-interval 50 \
  --eval-batches 16 \
  --peak-lr 1e-3 \
  --out-dir runs/c5b_rewarm_late

python scripts/analyze_run.py runs/c5b_rewarm_late/late_gate_post_seed0
```
