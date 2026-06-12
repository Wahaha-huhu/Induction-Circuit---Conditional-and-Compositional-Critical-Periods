# v0.13: Two-hop decomposition score ablation

This analysis tests whether the previous key-slot ablation was ranking the wrong heads.  For a HOP_2 chain `A -> B -> C`, it compares:

- `first_value`: attention from query position to the value slot `B` in the first binding `A -> B`.
- `second_value`: attention from query position to the value slot `C` in the second binding `B -> C`.
- `second_key`: attention from query position to the key slot `B` in the second binding `B -> C`.
- matched random-k head ablation.

The key prediction is that second-hop-ranked heads should disrupt HOP_2 more than first-hop-ranked heads and matched random heads.

## Run

Use the machine where the v0.9 behavioural runs still exist with `model_final.pt`.

```bash
unzip -o cp_toy_impl_v0_13.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests
```

Smoke test:

```bash
TOP_KS="2" NUM_RANDOM=5 EVAL_BATCHES=8 CALIB_BATCHES=4 \
  bash scripts/run_twohop_decomposition_v0_13.sh
```

Final run:

```bash
TOP_KS="2 4" NUM_RANDOM=20 EVAL_BATCHES=16 CALIB_BATCHES=8 \
  bash scripts/run_twohop_decomposition_v0_13.sh
```

Optional, more expensive: include the single-head causal-drop ranking for comparison.

```bash
INCLUDE_CAUSAL_RANKING=1 TOP_KS="2 4" NUM_RANDOM=20 EVAL_BATCHES=16 CALIB_BATCHES=8 \
  bash scripts/run_twohop_decomposition_v0_13.sh
```

## Outputs

The script writes:

- `twohop_score_ablation_per_run.csv`
- `twohop_score_ablation_group_summary.csv`
- `twohop_score_ablation_report.md`
- `cp_toy_twohop_decomposition_v0_13.zip`

## Interpretation

A positive result is:

```text
second_value top-k HOP_2 drop > first_value top-k HOP_2 drop
second_value top-k HOP_2 drop > random-k HOP_2 drop
```

If this fails, the second-hop attention score is not a sufficient localizer, and the circuit should remain described as distributed or not captured by simple attention-position scores.
