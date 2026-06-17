# v0.21: HOP_2-only from-scratch step-budget scale

This control asks whether the model can learn the two-hop task directly when every training example is HOP_2 from the beginning.

It is different from the staged experiment and from the mixed-from-start p_multi=0.5 baseline:

- staged: HOP_1 first, then HOP_2 mixture
- mixed-from-start p=0.5: HOP_1 and HOP_2 both present from step 0
- this control: HOP_2 only from step 0 (`p_multi=1.0`)

## Recommended one-seed run

```bash
unzip -o cp_toy_impl_v0_21.zip
cd cp_toy_impl
python -m pip install -e .
python -m pytest tests

DEVICE=cuda \
SEEDS="0" \
SCHEDULES="warmup_constant" \
STEPS_LIST="20000 40000 80000 120000" \
PMULTI=1.0 \
RUN_SCALE=1 \
RUN_AGG=1 \
PACK=1 \
bash scripts/run_hop2_only_scale_v0_21.sh
```

This writes:

```text
cp_toy_hop2_only_scale_v0_21.zip
```

Send that zip back for analysis.

## If 120k is too expensive

Use:

```bash
STEPS_LIST="20000 40000 80000"
```

## Optional cosine comparison

Only after the constant-LR run is understood:

```bash
SCHEDULES="warmup_cosine"
T_SCHEDULE=30000
```

This asks a different question: whether a decayed cosine schedule prevents HOP_2-only learning from scratch.
