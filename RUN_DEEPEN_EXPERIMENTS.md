# v0.7 Deepening experiments for conditional-compositional reachability

This patch supports the revised mechanism interpretation:

> The apparent window is a conditional compositional reachability effect. After HOP_1 training, the model sits in a HOP_1 low-loss region. HOP_2 requires moving to a smaller HOP_1+HOP_2 region. Success depends on the conditioned starting location, gradient alignment, and effective post-introduction update budget.

## A. Analyze existing logs: transition sharpness and update budget

Use this first on any existing `runs` directory or unpacked result bundle:

```bash
python scripts/analyze_transition_shape.py \
  --runs-dir runs/extended_followups \
  --out runs/extended_followups/transition_shape_summary.csv
```

The CSV includes:

- `tail_hop2_excess`
- `peak_hop2_excess`
- `t_hop2_acc_ge_0.50`, `0.80`, `0.95`, `0.99`
- normalized transition width `norm_width_t90_minus_t10`
- max finite-difference excess slope
- eval-sampled cumulative LR and update-ratio integrals after intro and up to HOP_2 >= 95%

Interpretation:

- S1 plateau failing with small positive excess supports weak movement but no transition.
- Longer-cosine increasing excess but failing supports update-budget sensitivity below threshold.
- Rewarm/reset showing long plateau then narrow transition supports thresholded HOP_2 acquisition.

## B. Checkpointed geometry follow-up

Run a small checkpointed rerun to directly probe distance/alignment.

Default arms:

1. S1 plateau late
2. S1 longer-cosine late
3. S2 constant late
4. rewarm+reset late

Default seeds: `1 2`.

```bash
bash scripts/run_checkpointed_geometry_followups.sh
```

For a lighter smoke run:

```bash
SEEDS="2" LOG_RANK=0 bash scripts/run_checkpointed_geometry_followups.sh
```

Outputs:

```text
runs/checkpointed_geometry_followups/transition_shape_summary.csv
runs/checkpointed_geometry_followups/run_summaries.csv
cp_toy_checkpointed_geometry_followups.zip
```

Each run also saves:

```text
checkpoint_pre_intro.pt
checkpoint_pre_step_16000.pt
checkpoint_pre_step_20000.pt
checkpoint_pre_step_24000.pt
checkpoint_pre_step_28000.pt
checkpoint_pre_step_32000.pt
model_final.pt
```

The intro checkpoint is saved **before** the first HOP_2-introduced training update, so it represents the conditioned HOP_1 starting location.

## C. Parameter-space geometry probe

After a checkpointed run, compare the intro checkpoint with final or an intermediate checkpoint:

```bash
python scripts/run_geometry_probe.py \
  --run-dir runs/checkpointed_geometry_followups/seed2/rewarm_reset_late/late_gate_post_seed2 \
  --checkpoint-a checkpoint_pre_intro.pt \
  --checkpoint-b model_final.pt \
  --device cuda \
  --out runs/checkpointed_geometry_followups/seed2/rewarm_reset_late_geometry.json
```

The JSON includes:

- normalized parameter distance travelled: `relative_update_norm`
- HOP_2 gradient norm at the conditioned starting point
- cosine alignment between the eventual update and the negative HOP_2 gradient at intro
- projection of update onto the intro HOP_2 descent direction
- interpolation HOP_1/HOP_2 loss and accuracy from checkpoint A to B

Interpretation:

- A larger relative update norm for successful rewarm than failed S1 supports reachability radius.
- Positive alignment with the HOP_2 descent direction supports gradient-direction accessibility.
- A sharp HOP_2 jump along interpolation supports a basin/threshold picture.
- If S2 constant fails despite high update but has weaker alignment than rewarm, this supports the claim that total update magnitude is insufficient; path/location and direction matter.

## D. Minimal evidence table to report

For the next update, collect:

| Arm | Seed | HOP_2 tail/excess | transition width | cumulative update proxy | relative distance intro→final | intro-gradient alignment | interpolation sharpness |
|---|---:|---:|---:|---:|---:|---:|---:|

This table connects behavioural results to the geometric mechanism.

## v0.8 path/subspace reachability analysis

After a checkpointed geometry run has been packed with `--include-models`, run:

```bash
bash scripts/run_reachability_geometry_analysis.sh
```

This compares every arm to the successful `rewarm_reset` displacement and asks whether failed arms moved in the successful HOP_2-forming direction. It also tests whether adding the successful displacement to each arm's intro checkpoint is sufficient to induce HOP_2.

For a faster CPU check:

```bash
DEVICE=cpu BATCH_SIZE=64 EVAL_BATCHES=2 \
OWN_ALPHAS="0,0.5,0.75,0.9,1.0" \
TARGET_DIRECTION_ALPHAS="0,0.5,1.0" \
bash scripts/run_reachability_geometry_analysis.sh
```
