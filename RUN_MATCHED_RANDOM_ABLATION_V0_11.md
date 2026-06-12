# Matched Random-Head Ablation Controls v0.11

This is a control for the v0.10 key-slot ablation result. It asks whether key-slot-selected heads damage HOP_2 more than matched random heads.

## Purpose

Top-4 key-slot ablation removes 4 of 16 heads. Without a random control, a HOP_2 drop might partly reflect generic damage from removing many heads. This script compares:

- key-slot top-k mean ablation
- random k-head mean ablation, repeated `NUM_RANDOM` times

By default random draws exclude the selected key-slot heads, making the control a non-key-head comparison.

## Required input

Run this on the machine where the v0.9 behavioural runs still exist with `model_final.pt` files:

```bash
runs/behavioral_replication_v0_9/...
```

No retraining is performed.

## Setup

```bash
unzip -o cp_toy_impl_v0_11.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

## Smoke test

```bash
TOP_KS="4" NUM_RANDOM=5 EVAL_BATCHES=8 bash scripts/run_matched_random_ablation_v0_11.sh
```

## Final run

```bash
TOP_KS="2 4" NUM_RANDOM=20 EVAL_BATCHES=24 bash scripts/run_matched_random_ablation_v0_11.sh
```

For higher precision:

```bash
TOP_KS="2 4" NUM_RANDOM=50 EVAL_BATCHES=32 bash scripts/run_matched_random_ablation_v0_11.sh
```

## Output

The script writes:

```text
runs/behavioral_replication_v0_9/matched_random_ablation/matched_random_ablation_per_run.csv
runs/behavioral_replication_v0_9/matched_random_ablation/matched_random_ablation_group_summary.csv
runs/behavioral_replication_v0_9/matched_random_ablation/matched_random_ablation_report.md
cp_toy_matched_random_ablation_v0_11.zip
```

Send back:

```text
cp_toy_matched_random_ablation_v0_11.zip
```

## Interpretation

Use the difference:

```text
key-slot HOP_2 drop - mean random HOP_2 drop
```

as the main specificity score.

Strong compositional-reuse evidence would be:

```text
key-slot drop >> random drop
```

If key-slot and random drops are similar, the safe conclusion is that HOP_2 is distributed/redundant and top-k ablation is not specific enough as circuit evidence.
