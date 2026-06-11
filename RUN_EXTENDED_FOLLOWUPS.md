# Extended late-introduction follow-ups

These runs separate three possible explanations for the seed-dependent C5/C5b rescue results:

1. **Not enough post-introduction time**.
2. **Current learning rate too low under original S1 cosine**.
3. **Cosine-history / model-age-dependent plasticity change**.

The key late-introduction setting remains:

```text
intro_step = 16000
max_steps  = 36000
```

## Arms

For each seed, the script runs five arms:

| Arm | Schedule | t_schedule | Interpretation |
|---|---|---:|---|
| `s1_plateau_late` | `warmup_cosine` | 30000 | Original S1 schedule extended beyond the cosine horizon. Steps after 30000 are final-LR plateau, not high-LR extra training. |
| `s1_longcos_late` | `warmup_cosine` | 36000 | Longer cosine horizon; asks whether slowing the decay rescues late HOP_2. |
| `s2_constant_late` | `warmup_constant` | 30000 | Constant high-LR late rescue with more post-intro time. |
| `rewarm_late` | `warmup_cosine_then_rewarm_constant` | 30000 | Cosine-history model gets high constant LR at intro. |
| `rewarm_reset_late` | `warmup_cosine_then_rewarm_constant_reset_optim` | 30000 | Same as rewarm, but AdamW optimizer state is reset at intro. |

## Run

From the repository root:

```bash
python -m pip install -e .
python -m pytest tests
bash scripts/run_extended_followups.sh
```

Defaults:

```text
SEEDS="1 2"
DEVICE=cuda
LOG_RANK=1
OUT_ROOT=runs/extended_followups
```

To run only one seed or disable rank logging:

```bash
SEEDS="2" LOG_RANK=0 bash scripts/run_extended_followups.sh
```

## Outputs

The script writes:

```text
runs/extended_followups/extended_followup_summaries.csv
cp_toy_extended_followups.zip
```

Each individual run also writes `summary.json` and prints the final summary at the end.

## Interpretation guide

| Outcome | Interpretation |
|---|---|
| `s1_plateau_late` still fails | Original S1 late failure is not merely a slightly delayed transition under the original schedule. |
| `s1_plateau_late` succeeds very late | Original failure was partly a training-time / final-LR plateau issue. |
| `s1_longcos_late` succeeds | Higher late LR / slower decay rescues HOP_2. |
| `s1_longcos_late` still fails | Stronger evidence for age/history-dependent loss, not only low current LR. |
| constant / rewarm / reset succeed with extension | Previous mixed rescue results were transition-time limited. |
| rewarm+reset still fails while constant succeeds | Stronger evidence for cosine-history-dependent plasticity loss beyond instantaneous LR and AdamW state. |
