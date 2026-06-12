# Behavioural replication aggregate report

This report prioritizes behavioural success fractions and transition statistics over single-seed geometry.

## Group summary

| Arm | Runs | Seeds | HOP_2 success | Mean tail HOP_2 | Mean HOP_2 excess | Fresh success | Median HOP_2 t95 |
|---|---:|---|---:|---:|---:|---:|---:|
| fresh_hop1_s1 | 5 | 3 4 5 6 7 | — | 0.0% | -13.9% | 100.0% | — |
| rewarm_late | 5 | 3 4 5 6 7 | 100.0% | 99.8% | 86.2% | — | 2.855e+04 |
| rewarm_reset_late | 5 | 3 4 5 6 7 | 100.0% | 99.9% | 86.4% | — | 2.535e+04 |
| s1_late_original | 5 | 3 4 5 6 7 | 0.0% | 17.0% | 2.7% | — | — |
| s1_longcos_late | 5 | 3 4 5 6 7 | 20.0% | 35.9% | 21.8% | — | 2.84e+04 |
| s1_plateau_late | 5 | 3 4 5 6 7 | 0.0% | 17.1% | 2.8% | — | — |
| s2_constant_late | 5 | 3 4 5 6 7 | 80.0% | 83.1% | 69.5% | — | 2.3e+04 |

## Recommended interpretation rule

- Treat `S1 late`/`S1 plateau` failures as robust only if the HOP_2 success fraction remains low across seeds.
- Treat `S2 constant` or `rewarm` as reliable rescues only if their success fractions are high across seeds.
- Use `fresh_hop1_s1` to test selectivity: it should start near floor and then reach high accuracy late.
- Use transition width and loss/accuracy trajectories to support thresholded delayed generalization.
- Do not infer cross-run weight-space direction from these behavioural aggregates.
