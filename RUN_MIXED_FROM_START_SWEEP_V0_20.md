# v0.20 mixed-from-start feasibility sweep

This sweep answers a different question from the staged critical-period experiment:

> Can the model learn HOP_1 and HOP_2 simultaneously when both tasks are present from the first step?

The staged experiments keep `p_multi_before_intro=0` and introduce HOP_2 only after HOP_1 has been learned. In contrast, v0.20 uses condition `c0`, so the same fixed `p_multi` is used from step 0 to the end.

## Recommended final run

```bash
unzip -o cp_toy_impl_v0_20.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests

DEVICE=cuda \
SEEDS="0 1 2 3 4" \
PMULTI_LIST="0.0 0.1 0.25 0.5 0.75 1.0" \
SCHEDULES="warmup_constant" \
MAX_STEPS=20000 \
LONG_SEEDS="0" \
LONG_SCHEDULES="warmup_constant" \
LONG_PMULTI=0.5 \
LONG_STEPS=80000 \
RUN_PMULTI=1 \
RUN_LONG=1 \
bash scripts/run_mixed_from_start_sweep_v0_20.sh
```

This creates:

```text
cp_toy_mixed_from_start_sweep_v0_20.zip
```

## Faster smoke run

```bash
DEVICE=cuda \
SEEDS="0" \
PMULTI_LIST="0.0 0.5 1.0" \
MAX_STEPS=4000 \
LONG_SEEDS="0" \
LONG_STEPS=8000 \
BATCH_SIZE=128 \
EVAL_BATCHES=2 \
RUN_PMULTI=1 RUN_LONG=1 \
bash scripts/run_mixed_from_start_sweep_v0_20.sh
```

## Optional S1+S2 comparison

The default uses `warmup_constant` to test feasibility under a high-update schedule. To also include S1 cosine, run:

```bash
DEVICE=cuda \
SEEDS="0 1 2" \
SCHEDULES="warmup_constant warmup_cosine" \
PMULTI_LIST="0.0 0.1 0.25 0.5 0.75 1.0" \
MAX_STEPS=20000 \
LONG_SEEDS="0" \
LONG_SCHEDULES="warmup_constant warmup_cosine" \
LONG_STEPS=80000 \
bash scripts/run_mixed_from_start_sweep_v0_20.sh
```

## Outputs

The scripts generate:

```text
runs/mixed_from_start_sweeps_v0_20/summary/pmulti_from_start_summary.csv
runs/mixed_from_start_sweeps_v0_20/summary/long_pmulti_0p5_summary.csv
runs/mixed_from_start_sweeps_v0_20/figures/fig_mixed_from_start_pmulti_tail_accuracy.pdf
runs/mixed_from_start_sweeps_v0_20/figures/fig_mixed_from_start_both_success_fraction.pdf
runs/mixed_from_start_sweeps_v0_20/figures/fig_long_pmulti_0p5_tail_accuracy.pdf
runs/mixed_from_start_sweeps_v0_20/tables/table_mixed_from_start_pmulti.tex
runs/mixed_from_start_sweeps_v0_20/tables/table_long_mixed_from_start_p05.tex
```

## Intended interpretation

If all p-values fail in the 20k sweep but the long p=0.5 run succeeds, then mixed-from-start learning is possible but much less sample-efficient than staged learning. If the long p=0.5 run also fails, then simultaneous discovery of HOP_1 and HOP_2 is hard in this toy setting, which strengthens the motivation for the staged scaffold.
