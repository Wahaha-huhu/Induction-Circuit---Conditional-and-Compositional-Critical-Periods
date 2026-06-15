# v0.19 design-validation sweeps

This package adds a compact calibration layer for the toy HOP\_2 introduction design. It is meant to answer three design questions before the final toy section is frozen.

1. **How many post-introduction HOP\_2 updates should be fixed?**  We calibrate `W_post` using high-update reference schedules. The goal is to choose the smallest post-introduction budget where HOP\_2 is reliably learnable in principle.
2. **Does moving the HOP\_2 introduction later reduce final HOP\_2 accuracy even when the post-introduction budget is fixed?**  We sweep the introduction step while holding `W_post` fixed.
3. **Is the result tied to one arbitrary post-introduction mixture ratio?**  We sweep `p_multi_after_intro` while keeping `p_multi_before_intro=0` in every condition.

The key design point is that HOP\_2 is absent before the selected introduction step. The earliest introduction is chosen after HOP\_1 is already learned in the withholding calibration. Thus the experiment tests acquisition of a dependent composition after its prerequisite exists.

## Recommended final command

```bash
DEVICE=cuda \
SEEDS="0 1 2 3 4" \
W_POST=13500 \
RUN_WPOST=1 RUN_INTRO=1 RUN_MIX=1 \
bash scripts/run_design_sweeps_v0_19.sh
```

This writes:

```text
runs/design_sweeps_v0_19/
  wpost_calibration/
  intro_step_sweep/
  mixture_sensitivity/
  summary/
  figures/
cp_toy_design_sweeps_v0_19.zip
```

Send back `cp_toy_design_sweeps_v0_19.zip` for analysis.

## Fast smoke test

Use this to verify the code and plotting without doing the full sweep.

```bash
DEVICE=cpu TORCH_THREADS=4 \
SEEDS="0" \
WPOST_LIST="100" INTRO_STEPS="2500 3000" PMULTI_LIST="0.50" W_POST=100 \
BATCH_SIZE=32 EVAL_BATCHES=1 EVAL_INTERVAL=100 \
D_MODEL=32 N_LAYERS=2 N_HEADS=2 D_MLP=64 \
V_CONTENT=32 CHAIN_LENGTH=4 \
bash scripts/run_design_sweeps_v0_19.sh
```

The smoke test is not scientifically meaningful. It only checks that the scripts run end-to-end.

## Choosing `W_post`

The default `W_POST=13500` matches the existing main toy experiments. For a fully calibrated write-up, run the W\_post sweep first:

```bash
DEVICE=cuda SEEDS="0 1 2" RUN_WPOST=1 RUN_INTRO=0 RUN_MIX=0 bash scripts/run_design_sweeps_v0_19.sh
```

Inspect:

```text
runs/design_sweeps_v0_19/summary/wpost_calibration_summary.csv
runs/design_sweeps_v0_19/figures/fig0b_wpost_calibration.pdf
```

Then choose the smallest `W_post` where the high-update reference arms, especially `s2_constant` and `rewarm_reset`, reliably exceed the HOP\_2 success threshold. Rerun the introduction and mixture sweeps with that value:

```bash
DEVICE=cuda SEEDS="0 1 2 3 4" W_POST=<chosen_value> RUN_WPOST=0 RUN_INTRO=1 RUN_MIX=1 bash scripts/run_design_sweeps_v0_19.sh
```

## Expected figures

The plotting script produces:

```text
fig0b_wpost_calibration.pdf/png
fig0b_wpost_calibration_success_fraction.pdf/png
fig0c_intro_step_vs_final_hop2_accuracy.pdf/png
fig0c_intro_step_success_fraction.pdf/png
fig0d_mixture_sensitivity.pdf/png
fig0d_mixture_success_fraction.pdf/png
v0_19_latex_snippet.tex
```

Recommended thesis placement:

- `fig0b_wpost_calibration`: before the main behavioural sweep, to justify the fixed post-introduction budget.
- `fig0c_intro_step_vs_final_hop2_accuracy`: main design-validation figure.
- `fig0d_mixture_sensitivity`: robustness or appendix figure.

## Notes on interpretation

Use careful wording:

- Do say: **fixed post-introduction budget**, **schedule-dependent acquirability**, **HOP\_2 introduced only after HOP\_1 has been learned**.
- Do not say: **irreversible critical period**. The rewarm arms show that the barrier is recoverable.
- Do not choose `W_post` based on where S1 fails. Choose it based on where HOP\_2 is learnable under high-update reference schedules.
