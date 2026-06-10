# Conditional / Compositional Critical-Period Toy Implementation

This is the first implementation bundle for the controlled toy tier. It implements the cascade-first build order from the v4.x design patches:

1. generator and target-only loss;
2. key-slot lookup diagnostic, not vanilla repeated-token induction;
3. C0 baseline, C1 prerequisite-delay, C3 ablation evaluation, C7 query-marker control;
4. S1 withholding and late-S1 gate scaffolding;
5. schedule utilities for S1/S2/S3, including cyclic peak alignment helper.

The full multi-schedule C5/C5b horizon loop is intentionally not the first runnable target. Per the final design, build the cascade tier and the late-S1 gate first, then only implement the multi-schedule calibration loop if the gate shows impaired late S1 learning.

## Install

From this directory:

```bash
python -m pip install -e .
```

Requires PyTorch and NumPy. If CUDA is unavailable, pass `--device cpu`.

## Unit tests

```bash
python -m pytest tests
```

## Quick smoke run

For a very small CPU smoke test:

```bash
python scripts/run_condition.py \
  --condition c0 \
  --device cpu \
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

## Main cascade-tier runs

### C0 baseline

```bash
python scripts/run_condition.py \
  --condition c0 \
  --seed 0 \
  --max-steps 4000 \
  --schedule warmup_cosine \
  --out-dir runs/cascade
```

### C1 prerequisite delay

Choose `--dynamic-switch-step` from the calibration rule, for example after an initial C0 pilot estimates `t_induction_ref`.

```bash
python scripts/run_condition.py \
  --condition c1 \
  --seed 0 \
  --max-steps 4000 \
  --dynamic-switch-step 1200 \
  --schedule warmup_cosine \
  --out-dir runs/cascade
```

### C3 ablation evaluation

Run after a trained C0 model exists:

```bash
python scripts/run_ablation_eval.py runs/cascade/c0_seed0 --top-k 1
```

This selects the top key-slot head across all layer-head pairs and applies global-mean ablation.

### C7 non-formative query-marker control

```bash
python scripts/run_condition.py \
  --condition c7_query_b \
  --seed 0 \
  --max-steps 2000 \
  --schedule warmup_cosine \
  --out-dir runs/cascade
```

This uses `QUERY_B` for the same single-hop lookup function.

## Late-S1 gate

The gate has two runs with matched post-introduction dose:

```bash
# Early reference
python scripts/run_condition.py \
  --condition late_gate_early \
  --intro-step <t_intro_early_S1> \
  --max-steps <t_intro_early_S1 + W_post> \
  --schedule warmup_cosine \
  --out-dir runs/gate

# Late post-consolidation probe
python scripts/run_condition.py \
  --condition late_gate_post \
  --intro-step <t_intro_post_cons_S1> \
  --max-steps <t_intro_post_cons_S1 + W_post> \
  --schedule warmup_cosine \
  --out-dir runs/gate
```

Then compute the gate decision:

```bash
python scripts/compute_late_gate.py \
  --early-run runs/gate/late_gate_early_seed0 \
  --late-run runs/gate/late_gate_post_seed0
```

Gate thresholds:

- `late / early <= 0.25`: strong closure candidate; proceed to full C5/C5b.
- `late / early <= 0.50`: impaired late learning; proceed to full C5/C5b.
- `0.50 < late / early <= 0.75`: ambiguous; run one additional gate seed.
- `late / early > 0.75`: no closeable window; run full C4 S1 sweep, then skip/deprioritize C5/C5b.

## Notes

- Loss is masked to the final target prediction only.
- The primary prerequisite metric is `key_slot_lookup_score` on `D_keyslot_singlehop` task-structured sequences.
- Vanilla repeated-token E4 is not implemented as a primary metric in this first bundle.
- Content-shuffled floor is tracked per evaluation checkpoint.
- C5/C5b should be added only after the late-S1 gate warrants the schedule tier.

For CPU smoke tests, small transformer forwards can be slower with many BLAS threads. You can add:

```bash
--torch-threads 1
```

or set:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```
