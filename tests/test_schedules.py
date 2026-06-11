from cp_toy.config import OptimConfig, ScheduleConfig
from cp_toy.schedules import lr_at_step


def test_rewarm_reset_schedule_lr_matches_rewarm_constant():
    optim = OptimConfig(peak_lr=1e-3, final_lr=1e-5, warmup_steps=10)
    sched = ScheduleConfig(kind="warmup_cosine_then_rewarm_constant_reset_optim", t_schedule=100, rewarm_step=40, rewarm_lr=7e-4)
    assert lr_at_step(39, optim, sched) < optim.peak_lr
    assert lr_at_step(40, optim, sched) == 7e-4
    assert lr_at_step(80, optim, sched) == 7e-4
