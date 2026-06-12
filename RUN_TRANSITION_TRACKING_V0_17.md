# v0.17: E4 transition tracking

This package adds the final toy-mechanism diagnostic before moving to the real-model experiment.
It ties the behavioural HOP_2 transition to the formation of HOP_2-mode signals over saved checkpoints.

## What it measures

For each checkpoint in a run, it evaluates:

- HOP_1 and HOP_2 loss/accuracy.
- HOP_2 shuffled-content floor and HOP_2 excess.
- Two-hop attention diagnostics:
  - `first_value`: query attention to B as the value in `A -> B`.
  - `second_key`: query attention to B as the key in `B -> C`.
  - `second_value`: query attention to C as the value/answer in `B -> C`.
- Gini/effective-head concentration for those scores.
- Logit-lens decodability of intermediate `B` and answer `C` at the query position across layers.

This tests the mechanistic prediction:

```text
failed S1 late: HOP_2 accuracy, second-hop routing, and C representation stay low.
successful rewarm/reset: HOP_2 accuracy and late-layer C representation rise sharply; second-hop diagnostics should rise near the transition if the two-hop routing score captures the forming component.
```

## Analyze existing checkpointed runs

If you already have checkpointed runs under `runs/checkpointed_geometry_followups`:

```bash
unzip -o cp_toy_impl_v0_17.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests

bash scripts/run_transition_tracking_v0_17.sh
```

For a fast smoke test:

```bash
MAX_RUNS=2 BATCH_SIZE=64 EVAL_BATCHES=2 SCORE_BATCHES=2 LENS_BATCHES=2 \
  bash scripts/run_transition_tracking_v0_17.sh
```

Outputs:

```text
runs/checkpointed_geometry_followups/transition_tracking/transition_tracking_per_checkpoint.csv
runs/checkpointed_geometry_followups/transition_tracking/transition_tracking_summary.csv
runs/checkpointed_geometry_followups/transition_tracking/transition_tracking_report.md
cp_toy_transition_tracking_v0_17.zip
```

## Generate a small checkpointed rerun if needed

If you do not have checkpoints, run:

```bash
SEEDS="2" LOG_RANK=0 bash scripts/run_transition_tracking_reruns_v0_17.sh
```

This trains:

- `s1_plateau_late`
- `s1_longcos_late`
- `rewarm_reset_late`

with checkpoints at:

```text
16000, 20000, 24000, 28000, 32000, final
```

It then runs the E4 analysis automatically and outputs:

```text
cp_toy_transition_tracking_reruns_v0_17.zip
```

## How to interpret

A strong positive result would show:

1. `s1_plateau_late`: HOP_2 remains near floor; L3 answer logit-lens remains low.
2. `s1_longcos_late`: weak HOP_2 movement and possibly weak second-hop signals.
3. `rewarm_reset_late`: loss decreases first, then HOP_2 accuracy and L3 answer signal jump sharply.
4. If `second_key` attention rises near the behavioural transition, this supports the two-hop-routing interpretation.

If HOP_2 accuracy jumps without second-hop attention score rising, the transition still supports delayed HOP_2-mode formation, but the attention score is only a partial diagnostic rather than the causal component.
