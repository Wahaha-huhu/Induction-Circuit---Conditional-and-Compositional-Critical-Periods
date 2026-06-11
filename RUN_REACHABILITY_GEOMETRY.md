# Reachability-geometry analysis

This diagnostic is intended for the checkpointed geometry follow-up runs produced by v0.7 or later.
It tests the parameter-space interpretation of the conditional/compositional window:

> after HOP_1, the model lies on a HOP_1-capable manifold; HOP_2 requires moving along a trajectory that reaches a smaller composed-retrieval region. Total distance or high LR alone may be insufficient if the movement is not in a HOP_2-relevant direction.

## Run the analysis

After packing a checkpointed run with `--include-models`, unzip it or run this inside the experiment repo where the `runs/checkpointed_geometry_followups` directory exists:

```bash
python scripts/analyze_reachability_geometry.py \
  --runs-dir runs/checkpointed_geometry_followups \
  --target-substring rewarm_reset \
  --device cuda \
  --eval-batches 8 \
  --out-dir runs/checkpointed_geometry_followups/reachability_geometry
```

For a CPU-only quick check:

```bash
python scripts/analyze_reachability_geometry.py \
  --runs-dir runs/checkpointed_geometry_followups \
  --target-substring rewarm_reset \
  --device cpu \
  --eval-batches 4 \
  --out-dir runs/checkpointed_geometry_followups/reachability_geometry
```

Then pack the analysis files:

```bash
python scripts/pack_results.py \
  --runs-dir runs/checkpointed_geometry_followups \
  --out cp_toy_reachability_geometry_analysis.zip
```

## Output files

The script writes:

- `directional_summary.csv`: each run's intro-to-final displacement, relative distance, and cosine/projection onto the successful rewarm+reset displacement.
- `own_path_interpolation.csv`: HOP_1/HOP_2 along each run's own straight line from intro to final.
- `target_direction_injection.csv`: HOP_1/HOP_2 after applying the successful displacement vector to each run's own intro checkpoint. This tests whether the successful direction is portable.
- `checkpoint_path_alignment.csv`: saved checkpoint trajectory, segment alignment with the successful displacement, and HOP_2 at each checkpoint.
- `reachability_geometry_manifest.json`: metadata.

## How to interpret

Useful patterns:

1. If a failed arm moves a large distance but has low cosine/projection onto the successful displacement and low HOP_2, then total movement is not enough.
2. If target-direction injection makes a failed intro checkpoint HOP_2-capable, then the successful direction captures a portable HOP_2-forming direction.
3. If target-direction injection fails on other intro checkpoints, the HOP_2 direction is path- or location-dependent, supporting the conditional-starting-location interpretation.
4. If checkpoint-path alignment rises before the HOP_2 jump, then the model may gradually align with the HOP_2-forming direction before crossing the transition threshold.
