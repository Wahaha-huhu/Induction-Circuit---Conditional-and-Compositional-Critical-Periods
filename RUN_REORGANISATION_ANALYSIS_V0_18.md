# v0.18 Reorganisation, Candidate-Set, and Probe-Control Analyses

This version adds a compact analysis package for the refined mechanism:

> HOP_2 acquisition is not just a static missing second hop. The model initially treats a HOP_2 prompt like HOP_1 and can emit the intermediate. After HOP_2 training begins, the naive response is suppressed. Successful high-update runs rebuild a HOP_2 mode; failed low-update runs suppress the naive response but do not complete the rebuild.

The package has two parts.

## A/B. Checkpointed reorganisation + candidate-set analysis

This uses saved checkpoints and evaluates, at every checkpoint:

- HOP_1/HOP_2 loss and accuracy
- HOP_2 excess over shuffled floor
- first-value, second-key, and second-value attention scores
- logit-lens decodability of intermediate `B` and answer `C` under a HOP_2 prompt
- candidate-set/format metrics, including whether the top prediction is an in-context content token or binding value
- probability mass on in-context content tokens, binding values, target, and intermediate

Run on existing checkpointed follow-up runs:

```bash
unzip -o cp_toy_impl_v0_18.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests

bash scripts/run_reorganisation_candidate_v0_18.sh
```

Fast smoke test:

```bash
MAX_RUNS=2 BATCH_SIZE=64 EVAL_BATCHES=2 SCORE_BATCHES=2 LENS_BATCHES=2 \
  bash scripts/run_reorganisation_candidate_v0_18.sh
```

Output:

```text
cp_toy_reorganisation_candidate_v0_18.zip
```

Key expected patterns:

- pre-intro HOP_2 prompt may decode/predict the intermediate `B`, reflecting naive HOP_1-mode behaviour;
- after HOP_2 training begins, failed runs suppress the naive `B` response but never build answer `C`;
- successful runs rebuild a HOP_2 mode, with `C` representation and HOP_2 accuracy emerging sharply;
- candidate-set mass may rise before exact accuracy, explaining the loss-before-accuracy gap as format/candidate-class learning before compositional routing.

## C. Mode-specific probe controls and failure-mode generality

This re-runs the HOP_1-vs-HOP_2 probe with shuffled-label controls.

It selects successful and failed/partial runs from the behavioural sweep and tests:

- under a HOP_1 prompt: can a probe decode `B`?
- under a HOP_2 prompt: can a probe decode intermediate `B` and answer `C`?
- does true-label probe accuracy exceed a shuffled-label control?

Run:

```bash
bash scripts/run_mode_probe_controls_v0_18.sh
```

Fast smoke test:

```bash
MAX_SUCCESS=2 MAX_FAILED=2 BATCH_SIZE=128 TRAIN_BATCHES=8 EVAL_BATCHES=4 PROBE_STEPS=150 \
  bash scripts/run_mode_probe_controls_v0_18.sh
```

Output:

```text
cp_toy_mode_probe_controls_v0_18.zip
```

This is mainly a rigor closer for the v0.15 result. The key readout is `probe_acc_above_shuffle`.

## Suggested order

1. Run `run_reorganisation_candidate_v0_18.sh` on the checkpointed seed-2 runs.
2. Run `run_mode_probe_controls_v0_18.sh` on the behavioural models.
3. If both support the refined story, freeze the toy analysis and move to final toy figures / Pythia acquirability.
