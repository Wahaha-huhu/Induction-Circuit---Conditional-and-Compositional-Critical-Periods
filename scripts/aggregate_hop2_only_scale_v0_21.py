#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_p_multi(obj):
    if isinstance(obj, dict):
        vals = []
        for k, v in obj.items():
            if k == "p_multi" or k == "p_multi_frozen":
                vals.append(v)
            vals.extend(find_p_multi(v))
        return vals
    if isinstance(obj, list):
        vals = []
        for x in obj:
            vals.extend(find_p_multi(x))
        return vals
    return []


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tail-n", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    traj_rows = []
    per_rows = []
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        metrics = list(read_jsonl(metrics_path))
        if not metrics:
            continue

        seed = cfg.get("train", {}).get("seed")
        schedule = cfg.get("schedule", {}).get("kind", "unknown")
        max_steps = cfg.get("train", {}).get("max_steps") or max(m["step"] for m in metrics)
        pvals = find_p_multi(cfg)
        p_multi = None
        for v in pvals:
            try:
                fv = float(v)
            except Exception:
                continue
            if abs(fv - 1.0) < 1e-9:
                p_multi = fv
                break
        if p_multi is None:
            p_multi = float("nan")
        m = re.search(r"steps_(\d+)", str(run_dir))
        step_budget = int(m.group(1)) if m else int(max_steps)

        for mrow in metrics:
            traj_rows.append({
                "run_dir": str(run_dir),
                "seed": seed,
                "schedule": schedule,
                "p_multi": p_multi,
                "step_budget": step_budget,
                "step": mrow.get("step"),
                "hop1_acc": mrow.get("hop1_acc"),
                "hop2_acc": mrow.get("hop2_acc"),
                "floor_hop2_acc": mrow.get("floor_hop2_acc"),
                "hop1_loss": mrow.get("hop1_loss"),
                "hop2_loss": mrow.get("hop2_loss"),
                "train_loss": mrow.get("train_loss"),
                "lr": mrow.get("lr"),
            })
        tail = metrics[-args.tail_n:]
        def mean_key(key):
            vals = [m.get(key) for m in tail if m.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        last = metrics[-1]
        t95 = next((m["step"] for m in metrics if m.get("hop2_acc", 0.0) >= 0.95), None)
        t50 = next((m["step"] for m in metrics if m.get("hop2_acc", 0.0) >= 0.50), None)
        per_rows.append({
            "run_dir": str(run_dir),
            "seed": seed,
            "schedule": schedule,
            "p_multi": p_multi,
            "step_budget": step_budget,
            "final_step": last.get("step"),
            "final_hop1_acc": last.get("hop1_acc"),
            "final_hop2_acc": last.get("hop2_acc"),
            "tail_hop1_acc": mean_key("hop1_acc"),
            "tail_hop2_acc": mean_key("hop2_acc"),
            "tail_floor_hop2_acc": mean_key("floor_hop2_acc"),
            "tail_hop2_excess": mean_key("hop2_acc") - mean_key("floor_hop2_acc"),
            "final_hop1_loss": last.get("hop1_loss"),
            "final_hop2_loss": last.get("hop2_loss"),
            "hop2_success": bool(mean_key("hop2_acc") >= 0.95),
            "t_hop2_acc_ge_0p5": t50,
            "t_hop2_acc_ge_0p95": t95,
        })

    traj = pd.DataFrame(traj_rows)
    per = pd.DataFrame(per_rows)
    if not traj.empty:
        traj.sort_values(["schedule", "step_budget", "seed", "step"]).to_csv(out / "hop2_only_scale_trajectories.csv", index=False)
    if not per.empty:
        per.sort_values(["schedule", "step_budget", "seed"]).to_csv(out / "hop2_only_scale_per_run.csv", index=False)
        group = per.groupby(["schedule", "step_budget", "p_multi"], dropna=False).agg(
            n_runs=("seed", "count"),
            success_count=("hop2_success", "sum"),
            mean_tail_hop2_acc=("tail_hop2_acc", "mean"),
            sem_tail_hop2_acc=("tail_hop2_acc", sem),
            mean_tail_hop2_excess=("tail_hop2_excess", "mean"),
            sem_tail_hop2_excess=("tail_hop2_excess", sem),
            mean_final_hop2_loss=("final_hop2_loss", "mean"),
            median_t95=("t_hop2_acc_ge_0p95", "median"),
        ).reset_index()
        group["success_fraction"] = group["success_count"] / group["n_runs"]
        group.sort_values(["schedule", "step_budget"]).to_csv(out / "hop2_only_scale_group_summary.csv", index=False)
        print(group.to_string(index=False))
    else:
        print(f"No runs found under {root}")


if __name__ == "__main__":
    main()
