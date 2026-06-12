# v0.10 key-slot ablation sweep

This is the next toy-model evidence layer after the v0.9 behavioural replication. It does **not** retrain models. It runs key-slot mean ablation on already-trained successful HOP_2 runs.

## Purpose

The behavioural sweep showed that late HOP_2 is selectively blocked under original S1 cosine and reliably rescued by rewarm. The ablation sweep asks whether successful HOP_2 models rely on the same key-slot lookup heads used for HOP_1.

This supports the compositional-reuse claim:

```text
HOP_2 is not just a new independent association; it reuses or coordinates the HOP_1 lookup machinery.
```

## Default selected arms

The script selects successful HOP_2 runs from:

```text
s1_longcos_late
s2_constant_late
rewarm_late
rewarm_reset_late
```

A run is selected if:

```text
tail HOP_2 accuracy >= 95%
tail HOP_2 excess >= 50 percentage points
model_final.pt exists locally
```

## Run

From the repo root:

```bash
python -m pip install -e .
python -m pytest tests
bash scripts/run_ablation_sweep_v0_10.sh
```

For a lighter smoke test:

```bash
TOP_KS="2" EVAL_BATCHES=8 bash scripts/run_ablation_sweep_v0_10.sh
```

For final-quality evaluation:

```bash
TOP_KS="1 2 4" EVAL_BATCHES=32 bash scripts/run_ablation_sweep_v0_10.sh
```

## Outputs

The script writes:

```text
runs/behavioral_replication_v0_9/ablation/ablation_selected_runs.txt
runs/behavioral_replication_v0_9/ablation/ablation_selected_runs.json
runs/behavioral_replication_v0_9/ablation/ablation_per_run.csv
runs/behavioral_replication_v0_9/ablation/ablation_group_summary.csv
runs/behavioral_replication_v0_9/ablation/ablation_report.md
cp_toy_ablation_sweep_v0_10.zip
```

Send back:

```text
cp_toy_ablation_sweep_v0_10.zip
```

## Interpretation

The cleanest result would be:

```text
top-1 ablation partially hurts HOP_1/HOP_2
top-2 ablation strongly collapses HOP_1/HOP_2
top-4 ablation collapses further or confirms robustness
```

Use top-2 as the primary ablation if it is stable across seeds and arms.

Do not interpret ablation as proving the schedule mechanism. It supports the circuit-reuse / compositional part of the claim after behaviour has already established the late acquirability barrier.
