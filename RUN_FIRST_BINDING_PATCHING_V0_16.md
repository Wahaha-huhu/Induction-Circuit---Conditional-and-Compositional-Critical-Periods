# v0.16: First-binding corruption / intermediate-routing patching

This analysis tests whether successful HOP_2 models route through the computed intermediate.

For each diagnostic example, the clean and corrupt inputs differ only in the first binding:

```text
clean:   A -> B
corrupt: A -> B'
```

Both downstream branches are present in both contexts:

```text
B  -> C
B' -> C'
```

The query is always:

```text
QUERY_A HOP_2 A
```

So the clean answer is `C`, and the corrupt answer is `C'`. If the model routes through the computed intermediate, patching clean query-position activations into the corrupt run should flip the answer from `C'` toward `C` at the layer where the intermediate/compositional state is carried.

## Run

Use the machine that still has the v0.9 behavioural runs with `model_final.pt`.

```bash
unzip -o cp_toy_impl_v0_16.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

Smoke test:

```bash
MAX_RUNS=2 BATCH_SIZE=64 NUM_BATCHES=4 HEAD_PATCHING=0 \
  bash scripts/run_first_binding_patching_v0_16.sh
```

Recommended run:

```bash
MAX_RUNS=6 BATCH_SIZE=64 NUM_BATCHES=8 HEAD_PATCHING=0 \
  bash scripts/run_first_binding_patching_v0_16.sh
```

Optional head patching:

```bash
MAX_RUNS=4 BATCH_SIZE=64 NUM_BATCHES=8 HEAD_PATCHING=1 \
  bash scripts/run_first_binding_patching_v0_16.sh
```

## Output

The runner writes:

```text
runs/behavioral_replication_v0_9/first_binding_patching/first_binding_patching_all_sites.csv
runs/behavioral_replication_v0_9/first_binding_patching/first_binding_patching_best_residual_per_run.csv
runs/behavioral_replication_v0_9/first_binding_patching/first_binding_patching_best_group_summary.csv
runs/behavioral_replication_v0_9/first_binding_patching/first_binding_patching_report.md
cp_toy_first_binding_patching_v0_16.zip
```

## Interpretation

Positive evidence for intermediate routing:

```text
corrupt baseline predicts C'
patching clean residual at an intermediate/late query-position layer increases logit(C)-logit(C')
patched clean-answer accuracy increases
```

Layer interpretation:

```text
early patch works: intermediate routing state is early
late patch works: answer/readout state is late
no patch works: model may not route through an explicit patchable intermediate, or the branched diagnostic is too far OOD
```

Important caveat: this diagnostic uses a branched binding set, not a pure chain, so it should be presented as a controlled routing probe rather than a standard in-distribution evaluation.
