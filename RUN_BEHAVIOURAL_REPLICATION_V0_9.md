# v0.9 behavioural replication and safer mechanism analysis

This patch follows the revised experiment priority:

1. Replicate the **behavioural arms** across more seeds before making strong mechanism claims.
2. Treat **thresholded HOP_2 transition** as the main mechanism result.
3. Keep cross-run geometry as exploratory/appendix-only because raw weight-space comparisons across different schedule histories can be frame-confounded.

## Why this patch exists

The current strongest claim is not an irreversible critical period. The safer claim is:

> Late HOP_2 acquisition is a selective, update-budget-gated and thresholded compositional transition. Simple fresh HOP_1 remains learnable late, while HOP_2 can fail under the original decayed S1 regime and can often be rescued by sustained high update budget.

The open empirical question is whether rescue arms such as S2 constant, rewarm, and rewarm+reset succeed reliably across seeds, or whether success is seed/path-sensitive.

## Run the behavioural replication

Default seeds are `3 4 5 6 7`:

```bash
bash scripts/run_behavioral_replication_v0_9.sh
```

For a one-seed smoke test:

```bash
SEEDS="3" LOG_RANK=0 bash scripts/run_behavioral_replication_v0_9.sh
```

For a lighter run, keep rank logging off, which is the default:

```bash
LOG_RANK=0 bash scripts/run_behavioral_replication_v0_9.sh
```

To include the fresh HOP_1 S2 retention comparison as well:

```bash
RUN_FRESH_S2=1 bash scripts/run_behavioral_replication_v0_9.sh
```

## Arms included

| Arm | Purpose |
|---|---|
| `s1_late_original` | Original S1 late finite-horizon failure check. |
| `s1_plateau_late` | Extends original S1 into final-LR plateau. Tests whether simply waiting longer rescues. |
| `s1_longcos_late` | Stretches cosine horizon to 36k. Tests whether reduced decay improves HOP_2 signal. |
| `s2_constant_late` | Constant-LR late run. Tests whether sustained high LR is sufficient across seeds. |
| `rewarm_late` | Cosine history, then high LR restored at intro. |
| `rewarm_reset_late` | Cosine history, high LR restored and optimizer reset at intro. |
| `fresh_hop1_s1` | Selectivity control: simple fresh HOP_1 should learn late under S1. |

## Outputs

The runner writes:

```text
runs/behavioral_replication_v0_9/all_run_summaries.csv
runs/behavioral_replication_v0_9/transition_shape_summary.csv
runs/behavioral_replication_v0_9/aggregate/behavioral_per_run.csv
runs/behavioral_replication_v0_9/aggregate/behavioral_group_summary.csv
runs/behavioral_replication_v0_9/aggregate/behavioral_report.md
cp_toy_behavioral_replication_v0_9.zip
```

The key file is:

```text
runs/behavioral_replication_v0_9/aggregate/behavioral_group_summary.csv
```

It reports success fractions, mean/median excess, and transition timing by arm.

## Aggregate an existing folder manually

```bash
python scripts/aggregate_behavioral_results.py \
  --runs-dir runs/behavioral_replication_v0_9 \
  --out-dir runs/behavioral_replication_v0_9/aggregate
```

## Safer geometry report

If you have reachability-geometry outputs, generate a conservative report:

```bash
python scripts/make_safe_geometry_report.py \
  --analysis-dir runs/reachability_geometry_analysis \
  --out runs/reachability_geometry_analysis/safe_geometry_report.md
```

Use this report to avoid overclaiming cross-run direction or injection results.

## Interpretation rules

- Make behavioural success fractions the headline, not one-seed geometry.
- Use thresholded transition analyses as the main mechanism evidence.
- Use same-run checkpoint/interpolation paths as illustrative mechanism case studies.
- Put cross-run direction/injection in the appendix only, with explicit frame-confounding caveats.
- Do not present injection into a same-pre-intro trajectory as portability evidence; it may simply reconstruct the successful endpoint.
