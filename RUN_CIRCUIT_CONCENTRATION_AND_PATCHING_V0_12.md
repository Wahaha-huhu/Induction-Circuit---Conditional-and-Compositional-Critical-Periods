# v0.12: Circuit Concentration and Narrow HOP_2 Activation Patching

This patch adds two compact toy analyses and deliberately **does not** implement greedy knockout.

The goal is to enrich the toy section without overclaiming a localized-head mechanism.

## Analysis A: concentration / Gini

This separates two quantities:

1. **Attention concentration**: Gini/entropy over key-slot lookup attention scores across all heads.
2. **Causal-load concentration**: Gini/entropy over single-head HOP_2 ablation drops across all heads.

The second is the more important measure. A sparse key-slot score distribution does not prove causal localization.

### Run

Use the machine where the v0.9 behavioural models still exist with `model_final.pt`.

```bash
unzip -o cp_toy_impl_v0_12.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests

# smoke test / faster
EVAL_BATCHES=4 MEAN_BATCHES=2 bash scripts/run_head_concentration_v0_12.sh

# better precision
EVAL_BATCHES=8 MEAN_BATCHES=4 bash scripts/run_head_concentration_v0_12.sh
```

Outputs:

```text
runs/behavioral_replication_v0_9/head_concentration/head_concentration_per_head.csv
runs/behavioral_replication_v0_9/head_concentration/head_concentration_per_run.csv
runs/behavioral_replication_v0_9/head_concentration/head_concentration_group_summary.csv
cp_toy_head_concentration_v0_12.zip
```

Key readout:

```text
Does schedule change causal-load concentration, or only attention-score concentration?
```

## Analysis B: narrow clean/corrupt activation patching

This is not full causal scrubbing. It asks a simpler question:

```text
Can clean HOP_2 activations at the query position restore the clean answer on a corrupted two-hop input?
```

Clean/corrupt pair:

```text
clean:   A -> B -> C, QUERY HOP_2 A -> C
corrupt: A -> B -> D, QUERY HOP_2 A -> D
```

Metric:

```text
logit(clean_answer) - logit(corrupt_answer)
```

Patching sites:

```text
query-position residual stream after each layer
optionally: each attention-head output at the query position
```

### Run

Start with residual patching only:

```bash
BATCH_SIZE=64 NUM_BATCHES=8 MAX_RUNS=4 HEAD_PATCHING=0 \
  bash scripts/run_activation_patching_v0_12.sh
```

Then, if residual patching shows a strong signal, run head patching:

```bash
BATCH_SIZE=64 NUM_BATCHES=8 MAX_RUNS=4 HEAD_PATCHING=1 \
  bash scripts/run_activation_patching_v0_12.sh
```

Outputs:

```text
runs/behavioral_replication_v0_9/hop2_activation_patching/activation_patching_all_sites.csv
runs/behavioral_replication_v0_9/hop2_activation_patching/activation_patching_best_by_run.csv
cp_toy_activation_patching_v0_12.zip
```

## Interpretation rules

Use these analyses as supporting evidence only.

Safe claim if positive:

```text
Successful HOP_2 models contain patchable query-position representations for the composed answer, and circuit concentration varies across schedules/seeds.
```

Avoid claiming:

```text
HOP_2 is exactly implemented by the top two key-slot heads.
```

The current matched-random ablation results already show that the implementation can be distributed or redundant.
