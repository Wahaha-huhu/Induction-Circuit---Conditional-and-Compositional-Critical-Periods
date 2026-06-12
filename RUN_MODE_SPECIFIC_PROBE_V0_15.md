# v0.15 mode-specific HOP_1 vs HOP_2 probe

This analysis tests the refined mechanism suggested by the v0.14 probe:
failed late-S1 models may still have the primitive HOP_1 lookup available, but fail to recruit it under a HOP_2 prompt.

For each selected run, the script probes the query-position residual stream under two prompt modes:

```text
HOP_1 prompt: A -> B, QUERY HOP_1 A
  label: B, the ordinary one-hop target

HOP_2 prompt: A -> B -> C, QUERY HOP_2 A
  labels: B, the intermediate; C, the final answer
```

The decisive pattern for a recruitment failure is:

```text
failed S1 late:
  HOP_1 prompt target B is decodable
  HOP_2 prompt intermediate B is not decodable
```

## Run

Use the machine with `runs/behavioral_replication_v0_9` and `model_final.pt` files.

```bash
unzip -o cp_toy_impl_v0_15.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

Smoke test:

```bash
MAX_SUCCESS=2 MAX_FAILED=2 \
BATCH_SIZE=128 TRAIN_BATCHES=8 EVAL_BATCHES=4 PROBE_STEPS=150 \
bash scripts/run_mode_specific_probe_v0_15.sh
```

Final run:

```bash
BATCH_SIZE=256 TRAIN_BATCHES=16 EVAL_BATCHES=8 PROBE_STEPS=300 \
bash scripts/run_mode_specific_probe_v0_15.sh
```

Logit-lens only:

```bash
SKIP_LINEAR_PROBE=1 bash scripts/run_mode_specific_probe_v0_15.sh
```

## Outputs

```text
runs/behavioral_replication_v0_9/mode_specific_probe/mode_specific_probe_all_layers.csv
runs/behavioral_replication_v0_9/mode_specific_probe/mode_specific_probe_per_run.csv
runs/behavioral_replication_v0_9/mode_specific_probe/mode_specific_probe_group_summary.csv
runs/behavioral_replication_v0_9/mode_specific_probe/mode_specific_probe_report.md
cp_toy_mode_specific_probe_v0_15.zip
```

Send back `cp_toy_mode_specific_probe_v0_15.zip`.
