#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import torch
from pathlib import Path

from cp_toy.config import DataConfig, ModelConfig, OptimConfig, ScheduleConfig, TrainConfig
from cp_toy.train import train_run
from cp_toy.schedules import next_cyclic_peak_at_or_after
from cp_toy.summary import print_summary, write_summary


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
    optim = OptimConfig(
        peak_lr=args.peak_lr,
        final_lr=args.final_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    # Optional S3 phase control: align intro/rewarm to first LR peak at or after intro.
    actual_intro_step = args.intro_step
    if args.phase_align_intro_to_peak and args.schedule == "warmup_cyclic":
        if actual_intro_step is None:
            raise ValueError("--phase-align-intro-to-peak requires --intro-step")
        tmp_sched = ScheduleConfig(kind=args.schedule, t_schedule=args.t_schedule, cycle_length=args.cycle_length)
        actual_intro_step = next_cyclic_peak_at_or_after(actual_intro_step, optim, tmp_sched)
        print(f"phase aligned intro_step: raw={args.intro_step} actual={actual_intro_step}")

    rewarm_step = args.rewarm_step
    if args.schedule in {"warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"}:
        if rewarm_step is None:
            if actual_intro_step is None:
                raise ValueError("rewarm schedules require --rewarm-step or --intro-step")
            rewarm_step = actual_intro_step

    sched = ScheduleConfig(
        kind=args.schedule,
        t_schedule=args.t_schedule,
        cycle_length=args.cycle_length,
        cycle_min_lr_frac=args.cycle_min_lr_frac,
        rewarm_step=rewarm_step,
        rewarm_lr=args.rewarm_lr,
    )

    intro_step = None
    dynamic_switch_step = None
    p_multi_before_intro = 0.0
    p_multi_frozen = args.p_multi
    p_dynamic_low = args.p_dynamic_low
    p_dynamic_high = args.p_dynamic_high
    query_marker = "A"
    query_marker_after_intro = None
    eval_query_b = False
    token_pool = args.token_pool
    token_pool_after_intro = None
    eval_base_fresh = False

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
        # Withholding calibration: no multi-hop signal.
        intro_step = args.max_steps + 1
        p_multi_before_intro = 0.0
        p_multi_frozen = args.p_multi
    elif condition in {"late_gate_early", "late_gate_post"}:
        # Early/mid/late dependent introduction.
        intro_step = actual_intro_step
        if intro_step is None:
            raise ValueError(f"{condition} requires --intro-step")
    elif condition == "c7_query_b":
        # Original simple non-formative routing control: train QUERY_B from the start.
        query_marker = "B"
        p_multi_frozen = 0.0
    elif condition == "c7_query_b_late":
        # Non-formative routing control: train QUERY_A single-hop before intro, then
        # QUERY_B single-hop after intro. This asks whether a new surface query marker
        # can route into the already-learned lookup circuit late.
        intro_step = actual_intro_step
        if intro_step is None:
            raise ValueError("c7_query_b_late requires --intro-step")
        query_marker = "A"
        query_marker_after_intro = "B"
        eval_query_b = True
        p_multi_before_intro = 0.0
        p_multi_frozen = 0.0
    elif condition == "fresh_singlehop_late":
        # Specificity control: train single-hop on base token pool before intro,
        # then single-hop on a held-out fresh token pool after intro.
        intro_step = actual_intro_step
        if intro_step is None:
            raise ValueError("fresh_singlehop_late requires --intro-step")
        token_pool = "base"
        token_pool_after_intro = "fresh"
        eval_base_fresh = True
        p_multi_before_intro = 0.0
        p_multi_frozen = 0.0
    else:
        raise ValueError(f"unknown condition {args.condition}")

    suffix = condition
    if args.phase_align_intro_to_peak and args.schedule == "warmup_cyclic" and args.intro_step is not None:
        suffix += f"_peak{actual_intro_step}"

    train = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        out_dir=str(Path(args.out_dir) / f"{suffix}_seed{args.seed}"),
        p_dynamic_high=p_dynamic_high,
        p_dynamic_low=p_dynamic_low,
        p_multi_frozen=p_multi_frozen,
        p_multi_before_intro=p_multi_before_intro,
        intro_step=intro_step,
        dynamic_switch_step=dynamic_switch_step,
        query_marker=query_marker,
        query_marker_after_intro=query_marker_after_intro,
        eval_query_b=eval_query_b,
        token_pool=token_pool,
        token_pool_after_intro=token_pool_after_intro,
        eval_base_fresh=eval_base_fresh,
        log_rank_metrics=args.log_rank_metrics,
    )
    return data, model, optim, sched, train


def main():
    p = argparse.ArgumentParser(description="Run toy conditional/compositional critical-period condition")
    p.add_argument("--condition", required=True, choices=[
        "c0", "c1", "withhold", "late_gate_early", "late_gate_post", "c7_query_b", "c7_query_b_late", "fresh_singlehop_late"
    ])
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
    p.add_argument("--token-pool", choices=["all", "base", "fresh"], default="all")

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-mlp", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--schedule", choices=[
        "warmup_cosine", "warmup_constant", "warmup_cyclic",
        "warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"
    ], default="warmup_cosine")
    p.add_argument("--t-schedule", type=int, default=20000)
    p.add_argument("--cycle-length", type=int, default=2000)
    p.add_argument("--cycle-min-lr-frac", type=float, default=0.1)
    p.add_argument("--phase-align-intro-to-peak", action="store_true", help="For S3, shift intro to first LR peak at or after --intro-step")
    p.add_argument("--rewarm-step", type=int, default=None, help="Step where rewarm schedule switches to constant LR; defaults to --intro-step")
    p.add_argument("--rewarm-lr", type=float, default=None, help="Constant LR after rewarm; defaults to --peak-lr")
    p.add_argument("--peak-lr", type=float, default=5e-4)
    p.add_argument("--final-lr", type=float, default=5e-6)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-rank-metrics", action="store_true", help="Log weight/rank/update markers; slower but useful for replications")
    p.add_argument("--no-final-summary", action="store_true", help="Do not print/save summary.json at end of run")
    args = p.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    cfgs = build_configs(args)
    out = train_run(*cfgs)
    print(f"wrote {out}")
    if not args.no_final_summary:
        try:
            summary = write_summary(out)
            print_summary(summary)
            print(f"wrote {out / 'summary.json'}")
        except Exception as exc:
            print(f"WARNING: failed to summarize run {out}: {exc}")


if __name__ == "__main__":
    main()
