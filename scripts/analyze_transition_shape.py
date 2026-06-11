#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cp_toy.config import OptimConfig, ScheduleConfig
from cp_toy.schedules import lr_at_step


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("metrics.jsonl")):
        yield p.parent


def numeric(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def nearest_after(rows: List[Dict[str, Any]], step: Optional[int]) -> List[Dict[str, Any]]:
    if step is None:
        return rows
    return [r for r in rows if int(r.get("step", -1)) >= int(step)]


def first_ge(rows: List[Dict[str, Any]], key: str, threshold: float, consecutive: int = 1) -> Optional[int]:
    c = 0
    for r in rows:
        v = numeric(r.get(key))
        if v is not None and v >= threshold:
            c += 1
            if c >= consecutive:
                return int(r["step"])
        else:
            c = 0
    return None


def tail_mean(vals: List[float], frac: float = 0.10) -> Optional[float]:
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    return float(sum(vals[-n:]) / n)


def cumulative_integral(rows: List[Dict[str, Any]], key: str, stop_step: Optional[int] = None) -> Optional[float]:
    """Approximate integral of an eval-logged scalar over training steps.

    For lr, this estimates sum_t lr_t from sparse logs. For update_to_weight_ratio,
    it is an eval-sampled proxy, not the exact per-step cumulative update.
    """
    filt = []
    for r in rows:
        step = int(r.get("step", -1))
        if stop_step is not None and step > int(stop_step):
            continue
        v = numeric(r.get(key))
        if v is not None:
            filt.append((step, v))
    if len(filt) < 2:
        return None
    total = 0.0
    for (s0, v0), (s1, v1) in zip(filt[:-1], filt[1:]):
        if s1 <= s0:
            continue
        total += 0.5 * (v0 + v1) * (s1 - s0)
    return float(total)


def make_excess_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        h2 = numeric(r.get("hop2_acc"))
        floor = numeric(r.get("floor_hop2_acc"))
        if h2 is not None and floor is not None:
            rr["hop2_excess"] = h2 - floor
        out.append(rr)
    return out


def summarize(run_dir: Path) -> Dict[str, Any]:
    rows = make_excess_rows(read_jsonl(run_dir / "metrics.jsonl"))
    cfg = load_config(run_dir)
    train = cfg.get("train", {}) if cfg else {}
    sched = cfg.get("schedule", {}) if cfg else {}
    optim = cfg.get("optim", {}) if cfg else {}
    intro = train.get("intro_step")
    post = nearest_after(rows, intro)
    if not post:
        post = rows
    last = rows[-1]

    vals_acc = [numeric(r.get("hop2_acc")) for r in post]
    vals_acc = [v for v in vals_acc if v is not None]
    vals_loss = [numeric(r.get("hop2_loss")) for r in post]
    vals_loss = [v for v in vals_loss if v is not None]
    vals_excess = [numeric(r.get("hop2_excess")) for r in post]
    vals_excess = [v for v in vals_excess if v is not None]
    vals_floor = [numeric(r.get("floor_hop2_acc")) for r in post]
    vals_floor = [v for v in vals_floor if v is not None]

    peak_excess = max(vals_excess) if vals_excess else None
    tail_excess = tail_mean(vals_excess)
    tail_acc = tail_mean(vals_acc)
    tail_loss = tail_mean(vals_loss)
    tail_floor = tail_mean(vals_floor)
    loss_start = vals_loss[0] if vals_loss else None
    loss_min = min(vals_loss) if vals_loss else None
    loss_tail_drop = (loss_start - tail_loss) if (loss_start is not None and tail_loss is not None) else None
    loss_min_drop = (loss_start - loss_min) if (loss_start is not None and loss_min is not None) else None

    # Absolute behavioural thresholds.
    t_acc_50 = first_ge(post, "hop2_acc", 0.50, consecutive=1)
    t_acc_80 = first_ge(post, "hop2_acc", 0.80, consecutive=1)
    t_acc_95 = first_ge(post, "hop2_acc", 0.95, consecutive=1)
    t_acc_99 = first_ge(post, "hop2_acc", 0.99, consecutive=1)
    t_ex_03 = first_ge(post, "hop2_excess", 0.03, consecutive=1)
    t_ex_10 = first_ge(post, "hop2_excess", 0.10, consecutive=1)
    t_ex_50 = first_ge(post, "hop2_excess", 0.50, consecutive=1)
    t_ex_80 = first_ge(post, "hop2_excess", 0.80, consecutive=1)

    # Normalized transition thresholds relative to observed peak excess.
    norm_t10 = norm_t50 = norm_t90 = None
    width_90_10 = None
    if peak_excess is not None and peak_excess >= 0.10:
        norm_t10 = first_ge(post, "hop2_excess", 0.10 * peak_excess, consecutive=1)
        norm_t50 = first_ge(post, "hop2_excess", 0.50 * peak_excess, consecutive=1)
        norm_t90 = first_ge(post, "hop2_excess", 0.90 * peak_excess, consecutive=1)
        if norm_t10 is not None and norm_t90 is not None:
            width_90_10 = int(norm_t90) - int(norm_t10)

    # Max finite-difference slope of excess per 1000 steps.
    max_slope = None
    slope_step = None
    pairs = []
    for r0, r1 in zip(post[:-1], post[1:]):
        e0 = numeric(r0.get("hop2_excess")); e1 = numeric(r1.get("hop2_excess"))
        s0 = int(r0.get("step", 0)); s1 = int(r1.get("step", 0))
        if e0 is None or e1 is None or s1 <= s0:
            continue
        pairs.append(((e1 - e0) / (s1 - s0) * 1000.0, s1))
    if pairs:
        max_slope, slope_step = max(pairs, key=lambda x: x[0])

    # Integrals after intro and to transition.
    cum_lr = cumulative_integral(post, "lr")
    cum_ur = cumulative_integral(post, "update_to_weight_ratio")
    cum_lr_to95 = cumulative_integral(post, "lr", t_acc_95) if t_acc_95 is not None else None
    cum_ur_to95 = cumulative_integral(post, "update_to_weight_ratio", t_acc_95) if t_acc_95 is not None else None

    try:
        optim_cfg = OptimConfig(**optim)
        sched_cfg = ScheduleConfig(**sched)
        lr_intro_formula = lr_at_step(intro, optim_cfg, sched_cfg) if intro is not None else None
    except Exception:
        lr_intro_formula = None

    return {
        "run_dir": str(run_dir),
        "condition": run_dir.name,
        "seed": train.get("seed"),
        "schedule": sched.get("kind"),
        "intro_step": intro,
        "max_steps": train.get("max_steps"),
        "t_schedule": sched.get("t_schedule"),
        "rewarm_step": sched.get("rewarm_step"),
        "lr_intro_formula": lr_intro_formula,
        "lr_tail_mean": tail_mean([numeric(r.get("lr")) for r in post if numeric(r.get("lr")) is not None]),
        "tail_hop2_acc": tail_acc,
        "tail_floor_hop2_acc": tail_floor,
        "tail_hop2_excess": tail_excess,
        "peak_hop2_excess": peak_excess,
        "tail_hop2_loss": tail_loss,
        "post_intro_loss_start": loss_start,
        "post_intro_loss_min": loss_min,
        "loss_drop_start_to_tail": loss_tail_drop,
        "loss_drop_start_to_min": loss_min_drop,
        "t_hop2_acc_ge_0.50": t_acc_50,
        "t_hop2_acc_ge_0.80": t_acc_80,
        "t_hop2_acc_ge_0.95": t_acc_95,
        "t_hop2_acc_ge_0.99": t_acc_99,
        "t_hop2_excess_ge_0.03": t_ex_03,
        "t_hop2_excess_ge_0.10": t_ex_10,
        "t_hop2_excess_ge_0.50": t_ex_50,
        "t_hop2_excess_ge_0.80": t_ex_80,
        "norm_t10_peak_excess": norm_t10,
        "norm_t50_peak_excess": norm_t50,
        "norm_t90_peak_excess": norm_t90,
        "norm_width_t90_minus_t10": width_90_10,
        "max_excess_slope_per_1k_steps": max_slope,
        "max_slope_end_step": slope_step,
        "cum_lr_post_intro_eval_integral": cum_lr,
        "cum_update_ratio_post_intro_eval_integral": cum_ur,
        "cum_lr_to_hop2_95_eval_integral": cum_lr_to95,
        "cum_update_ratio_to_hop2_95_eval_integral": cum_ur_to95,
        "final_hop1_acc": numeric(last.get("hop1_acc")),
        "final_hop2_acc": numeric(last.get("hop2_acc")),
        "final_hop2_loss": numeric(last.get("hop2_loss")),
        "final_floor_hop2_acc": numeric(last.get("floor_hop2_acc")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze HOP_2 transition sharpness and update-budget proxies from metrics.jsonl logs")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.runs_dir)
    rows = []
    for rd in run_dirs(root):
        try:
            rows.append(summarize(rd))
        except Exception as exc:
            print(f"WARNING: skipping {rd}: {exc}", file=sys.stderr)
    if not rows:
        raise SystemExit(f"no run metrics found under {root}")
    keys = sorted({k for r in rows for k in r.keys()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
