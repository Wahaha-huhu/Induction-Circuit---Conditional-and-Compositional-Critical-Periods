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
