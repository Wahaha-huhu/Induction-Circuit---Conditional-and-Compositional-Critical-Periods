# v0.14 Intermediate / Answer Probe

This implements E2 from the two-hop decomposition plan. It asks whether HOP_2-successful models represent the intermediate token `B` before the final answer `C`, and whether failed S1-late models decode `B` but not `C`.

For a forced HOP_2 example:

```text
A -> B -> C
QUERY HOP_2 A -> C
```

we collect the residual stream at the query position after every transformer block and evaluate two diagnostics:

1. **Logit lens:** apply the model final layer norm and unembedding to each layer residual and check whether the intermediate `B` or answer `C` is top-1.
2. **Linear probe:** train a small linear classifier on held-out dynamic HOP_2 examples to decode `B` and `C` from the query residual at each layer.

## Run

Use the machine with the v0.9 behavioural models and `model_final.pt` files.

```bash
unzip -o cp_toy_impl_v0_14.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

Smoke test:

```bash
MAX_SUCCESS=2 MAX_FAILED=2 BATCH_SIZE=128 TRAIN_BATCHES=8 EVAL_BATCHES=4 PROBE_STEPS=150 \
  bash scripts/run_intermediate_answer_probe_v0_14.sh
```

Final run:

```bash
BATCH_SIZE=256 TRAIN_BATCHES=16 EVAL_BATCHES=8 PROBE_STEPS=300 \
  bash scripts/run_intermediate_answer_probe_v0_14.sh
```

For a fast logit-lens-only run:

```bash
SKIP_LINEAR_PROBE=1 bash scripts/run_intermediate_answer_probe_v0_14.sh
```

## Outputs

The runner writes:

```text
runs/behavioral_replication_v0_9/intermediate_answer_probe/intermediate_answer_probe_all_layers.csv
runs/behavioral_replication_v0_9/intermediate_answer_probe/intermediate_answer_probe_per_run.csv
runs/behavioral_replication_v0_9/intermediate_answer_probe/intermediate_answer_probe_group_summary.csv
runs/behavioral_replication_v0_9/intermediate_answer_probe/intermediate_answer_probe_report.md
cp_toy_intermediate_answer_probe_v0_14.zip
```

## Interpretation

Evidence for the two-step mechanism:

```text
successful HOP_2 runs: B decodable earlier, C decodable later
failed S1 late runs:  B decodable but C not decodable
```

This would localise the late acquirability barrier to the second-hop answer-forming step rather than the first-hop lookup.

Informative nulls:

```text
B not decodable: the intermediate may be non-token-like or superposed.
C decodable in failed models: failure may be downstream readout rather than second-hop retrieval.
B and C appear together: the model may use a non-sequential shortcut or highly compressed representation.
```
