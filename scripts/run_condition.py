#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import torch
from dataclasses import replace
from pathlib import Path

from cp_toy.config import DataConfig, ModelConfig, OptimConfig, ScheduleConfig, TrainConfig
from cp_toy.train import train_run


def build_configs(args):
    data = DataConfig(v_content=args.v_content, chain_length=args.chain_length, k_max=args.k_max)
    model = ModelConfig(
        vocab_size=data.vocab_size,
        seq_len=data.input_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    optim = OptimConfig(peak_lr=args.peak_lr, final_lr=args.final_lr, warmup_steps=args.warmup_steps)
    sched = ScheduleConfig(kind=args.schedule, t_schedule=args.t_schedule, cycle_length=args.cycle_length, rewarm_step=args.rewarm_step, rewarm_lr=args.rewarm_lr)

    intro_step = None
    dynamic_switch_step = None
    p_multi_before_intro = 0.0
    p_multi_frozen = args.p_multi
    p_dynamic_low = args.p_dynamic_low
    p_dynamic_high = args.p_dynamic_high
    query_marker = "A"

    condition = args.condition.lower()
    if condition == "c0":
        # Baseline: high dynamic pressure and both query types from the start.
        pass
    elif condition == "c1":
        # Delay prerequisite pressure: low beta_dynamic early, high later.
        dynamic_switch_step = args.dynamic_switch_step
        if dynamic_switch_step is None:
            raise ValueError("C1 requires --dynamic-switch-step")
    elif condition == "withhold":
        # S1 withholding calibration or equivalent: no multi-hop signal.
        intro_step = args.max_steps + 1
        p_multi_before_intro = 0.0
        p_multi_frozen = args.p_multi
    elif condition == "late_gate_early":
        # Early-introduction reference for the late-S1 gate.
        intro_step = args.intro_step
        if intro_step is None:
            raise ValueError("late_gate_early requires --intro-step")
    elif condition == "late_gate_post":
        # Post-consolidation late probe for the late-S1 gate.
        intro_step = args.intro_step
        if intro_step is None:
            raise ValueError("late_gate_post requires --intro-step")
    elif condition == "c7_query_b":
        # Non-formative routing control: same single-hop lookup under QUERY_B.
        query_marker = "B"
        p_multi_frozen = 0.0
    else:
        raise ValueError(f"unknown condition {args.condition}")

    train = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        out_dir=str(Path(args.out_dir) / f"{condition}_seed{args.seed}"),
        p_dynamic_high=p_dynamic_high,
        p_dynamic_low=p_dynamic_low,
        p_multi_frozen=p_multi_frozen,
        p_multi_before_intro=p_multi_before_intro,
        intro_step=intro_step,
        dynamic_switch_step=dynamic_switch_step,
        query_marker=query_marker,
    )
    return data, model, optim, sched, train


def main():
    p = argparse.ArgumentParser(description="Run toy conditional/compositional critical-period condition")
    p.add_argument("--condition", required=True, choices=["c0", "c1", "withhold", "late_gate_early", "late_gate_post", "c7_query_b"])
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-threads", type=int, default=None, help="Set torch CPU threads for small CPU smoke tests")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=8)

    p.add_argument("--v-content", type=int, default=256)
    p.add_argument("--chain-length", type=int, default=16)
    p.add_argument("--k-max", type=int, default=2)
    p.add_argument("--p-dynamic-high", type=float, default=1.0)
    p.add_argument("--p-dynamic-low", type=float, default=0.05)
    p.add_argument("--p-multi", type=float, default=0.5)
    p.add_argument("--dynamic-switch-step", type=int, default=None)
    p.add_argument("--intro-step", type=int, default=None)

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-mlp", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--schedule", choices=["warmup_cosine", "warmup_constant", "warmup_cyclic", "warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"], default="warmup_cosine")
    p.add_argument("--t-schedule", type=int, default=20000)
    p.add_argument("--cycle-length", type=int, default=2000)
    p.add_argument("--rewarm-step", type=int, default=None, help="Step where rewarm schedule switches to constant LR; defaults to --intro-step")
    p.add_argument("--rewarm-lr", type=float, default=None, help="Constant LR after rewarm; defaults to --peak-lr")
    p.add_argument("--peak-lr", type=float, default=5e-4)
    p.add_argument("--final-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    args = p.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    if args.schedule in {"warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"}:
        if args.rewarm_step is None:
            if args.intro_step is None:
                raise ValueError("rewarm schedules require --rewarm-step or --intro-step")
            args.rewarm_step = args.intro_step

    cfgs = build_configs(args)
    out = train_run(*cfgs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
