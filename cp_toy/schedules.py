from __future__ import annotations

import math
from .config import OptimConfig, ScheduleConfig


def lr_at_step(step: int, optim: OptimConfig, sched: ScheduleConfig) -> float:
    """Return LR for a zero-indexed optimizer step."""
    if step < optim.warmup_steps:
        return optim.peak_lr * float(step + 1) / max(1, optim.warmup_steps)

    if sched.kind == "warmup_constant":
        return optim.peak_lr

    if sched.kind == "warmup_cosine":
        denom = max(1, sched.t_schedule - optim.warmup_steps)
        progress = min(1.0, max(0.0, (step - optim.warmup_steps) / denom))
        return optim.final_lr + 0.5 * (optim.peak_lr - optim.final_lr) * (1.0 + math.cos(math.pi * progress))

    if sched.kind in {"warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"}:
        # C5b-1/C5b-1b rewarm controls. Before rewarm_step, follow the same cosine
        # decay as S1. From rewarm_step onward, restore a constant LR.
        # This tests whether late failure is just low instantaneous/update LR.
        # The *_reset_optim variant uses the same LR function but resets AdamW state
        # at rewarm_step inside the training loop.
        rewarm_step = sched.rewarm_step
        if rewarm_step is None:
            raise ValueError("warmup_cosine_then_rewarm_constant requires sched.rewarm_step")
        if step >= rewarm_step:
            return optim.peak_lr if sched.rewarm_lr is None else sched.rewarm_lr
        denom = max(1, sched.t_schedule - optim.warmup_steps)
        progress = min(1.0, max(0.0, (step - optim.warmup_steps) / denom))
        return optim.final_lr + 0.5 * (optim.peak_lr - optim.final_lr) * (1.0 + math.cos(math.pi * progress))

    if sched.kind == "warmup_cyclic":
        # Cosine cycle from peak to cycle_min and back to peak. The canonical
        # introduction phase is the LR peak, which occurs at cycle boundaries.
        cycle = max(1, sched.cycle_length)
        phase = ((step - optim.warmup_steps) % cycle) / cycle
        min_lr = optim.peak_lr * sched.cycle_min_lr_frac
        return min_lr + 0.5 * (optim.peak_lr - min_lr) * (1.0 + math.cos(2.0 * math.pi * phase))

    raise ValueError(f"unknown schedule kind: {sched.kind}")


def next_cyclic_peak_at_or_after(raw_step: int, optim: OptimConfig, sched: ScheduleConfig) -> int:
    """First S3 LR peak at or after raw_step.

    Peaks occur at warmup_steps + n * cycle_length.
    """
    if sched.kind != "warmup_cyclic":
        return raw_step
    if raw_step <= optim.warmup_steps:
        return optim.warmup_steps
    n = math.ceil((raw_step - optim.warmup_steps) / sched.cycle_length)
    return optim.warmup_steps + n * sched.cycle_length
