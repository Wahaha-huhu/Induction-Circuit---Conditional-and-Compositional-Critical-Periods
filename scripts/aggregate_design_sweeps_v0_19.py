#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.10) -> Optional[float]:
    vals = [num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    return float(sum(vals[-n:]) / n)


def first_ge(rows: List[Dict[str, Any]], key: str, thr: float, consecutive: int = 1) -> Optional[int]:
    c = 0
    for r in rows:
        v = num(r.get(key))
        if v is not None and v >= thr:
            c += 1
            if c >= consecutive:
                return int(r.get("step", 0))
        else:
            c = 0
    return None


def nearest_at_or_after(rows: List[Dict[str, Any]], step: Optional[int]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if step is None:
        return rows[0]
    for r in rows:
        if int(r.get("step", -1)) >= int(step):
            return r
    return rows[-1]


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else None


def sd(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def infer_sweep_arm_param(root: Path, run_dir: Path) -> Tuple[str, str, str]:
    """Expected path: root/sweep_type/arm/param/late_gate_post_seedX."""
    try:
        rel = run_dir.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = run_dir.parts
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    return "unknown", "unknown", "unknown"


def parse_param(sweep_type: str, param: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if sweep_type == "wpost_calibration" and param.startswith("wpost_"):
        out["w_post"] = int(param.split("_", 1)[1])
    elif sweep_type == "intro_step_sweep" and param.startswith("intro_"):
        out["intro_step_grid"] = int(param.split("_", 1)[1])
    elif sweep_type == "mixture_sensitivity" and param.startswith("pmulti_"):
        out["p_multi_grid"] = float(param.split("_", 1)[1].replace("p", "."))
    return out


def add_excess(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        acc = num(r.get("hop2_acc"))
        floor = num(r.get("floor_hop2_acc"))
        if acc is not None and floor is not None:
            r["hop2_excess"] = acc - floor


def summarize_run(root: Path, run_dir: Path, success_thr: float) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    add_excess(rows)
    cfg = read_config(run_dir)
    train = cfg.get("train", {}) if cfg else {}
    sched = cfg.get("schedule", {}) if cfg else {}
    optim = cfg.get("optim", {}) if cfg else {}
    sweep_type, arm, param = infer_sweep_arm_param(root, run_dir)
    meta = parse_param(sweep_type, param)
    intro = train.get("intro_step")
    max_steps = train.get("max_steps")
    p_multi = train.get("p_multi_frozen")
    post_rows = [r for r in rows if intro is None or int(r.get("step", -1)) >= int(intro)]
    if not post_rows:
        post_rows = rows
    intro_row = nearest_at_or_after(rows, intro)
    last = rows[-1] if rows else {}
    tail_h2 = tail_mean(post_rows, "hop2_acc")
    tail_excess = tail_mean(post_rows, "hop2_excess")
    t95 = first_ge(post_rows, "hop2_acc", success_thr)
    t50 = first_ge(post_rows, "hop2_acc", 0.50)
    t99 = first_ge(post_rows, "hop2_acc", 0.99)
    hop2_success = bool((t95 is not None) or (tail_h2 is not None and tail_h2 >= success_thr))
    post_steps = None
    if intro is not None and max_steps is not None:
        post_steps = int(max_steps) - int(intro)
    row = {
        "run_dir": str(run_dir),
        "sweep_type": sweep_type,
        "arm": arm,
        "param": param,
        "seed": train.get("seed"),
        "schedule": sched.get("kind"),
        "intro_step": intro,
        "max_steps": max_steps,
        "post_steps": post_steps,
        "p_multi": p_multi,
        "t_schedule": sched.get("t_schedule"),
        "peak_lr": optim.get("peak_lr"),
        "final_lr": optim.get("final_lr"),
        "rewarm_step": sched.get("rewarm_step"),
        "final_step": int(last.get("step", -1)) if last else None,
        "intro_hop1_acc": num(intro_row.get("hop1_acc")) if intro_row else None,
        "intro_hop2_acc": num(intro_row.get("hop2_acc")) if intro_row else None,
        "tail_hop1_acc": tail_mean(post_rows, "hop1_acc"),
        "tail_hop2_acc": tail_h2,
        "tail_floor_hop2_acc": tail_mean(post_rows, "floor_hop2_acc"),
        "tail_hop2_excess": tail_excess,
        "tail_hop2_loss": tail_mean(post_rows, "hop2_loss"),
        "t_hop2_acc_ge_0.50": t50,
        f"t_hop2_acc_ge_{success_thr:.2f}": t95,
        "t_hop2_acc_ge_0.99": t99,
        "hop2_success": hop2_success,
    }
    row.update(meta)
    return row


def group_rows(per_run: List[Dict[str, Any]], success_thr: float) -> Dict[str, List[Dict[str, Any]]]:
    group_fields = {
        "wpost_calibration": ["sweep_type", "arm", "w_post"],
        "intro_step_sweep": ["sweep_type", "arm", "intro_step_grid"],
        "mixture_sensitivity": ["sweep_type", "arm", "p_multi_grid"],
    }
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in group_fields}
    for sweep_type, fields in group_fields.items():
        rows = [r for r in per_run if r.get("sweep_type") == sweep_type]
        keys = sorted({tuple(r.get(f) for f in fields) for r in rows})
        for key in keys:
            rs = [r for r in rows if tuple(r.get(f) for f in fields) == key]
            if not rs:
                continue
            g = {f: v for f, v in zip(fields, key)}
            g.update({
                "n_runs": len(rs),
                "n_success": sum(1 for r in rs if str(r.get("hop2_success")).lower() == "true" or r.get("hop2_success") is True),
                "success_fraction": sum(1 for r in rs if r.get("hop2_success") is True) / len(rs),
                "mean_tail_hop2_acc": mean(r.get("tail_hop2_acc") for r in rs),
                "sd_tail_hop2_acc": sd(r.get("tail_hop2_acc") for r in rs),
                "mean_tail_hop2_excess": mean(r.get("tail_hop2_excess") for r in rs),
                "sd_tail_hop2_excess": sd(r.get("tail_hop2_excess") for r in rs),
                "mean_tail_hop1_acc": mean(r.get("tail_hop1_acc") for r in rs),
                "mean_intro_hop1_acc": mean(r.get("intro_hop1_acc") for r in rs),
                "mean_post_steps": mean(r.get("post_steps") for r in rs),
                "mean_p_multi": mean(r.get("p_multi") for r in rs),
                "success_threshold": success_thr,
            })
            out[sweep_type].append(g)
    return out


def write_report(path: Path, grouped: Dict[str, List[Dict[str, Any]]], success_thr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v0.19 design sweep report",
        "",
        f"Success threshold: HOP_2 tail accuracy >= {success_thr:.2f} or first crossing of that threshold.",
        "",
        "## Intended use",
        "",
        "These sweeps validate the design choices used by the toy critical-period experiment: the fixed post-introduction budget, the introduction-step axis, and the post-introduction HOP_2 mixture ratio.",
        "",
    ]
    for name, rows in grouped.items():
        lines.append(f"## {name}")
        lines.append("")
        if not rows:
            lines.append("No completed runs found.")
            lines.append("")
            continue
        # Compact markdown table with core columns.
        if name == "wpost_calibration":
            cols = ["arm", "w_post", "n_runs", "success_fraction", "mean_tail_hop2_acc", "mean_tail_hop2_excess"]
        elif name == "intro_step_sweep":
            cols = ["arm", "intro_step_grid", "n_runs", "success_fraction", "mean_tail_hop2_acc", "mean_tail_hop2_excess"]
        else:
            cols = ["arm", "p_multi_grid", "n_runs", "success_fraction", "mean_tail_hop2_acc", "mean_tail_hop2_excess"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/design_sweeps_v0_19")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--success-threshold", type=float, default=0.95)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for p in sorted(root.rglob("metrics.jsonl")):
        # Ignore packed or copied summaries.
        if "summary" in p.parts or "figures" in p.parts:
            continue
        try:
            per_run.append(summarize_run(root, p.parent, args.success_threshold))
        except Exception as exc:  # noqa: BLE001
            per_run.append({"run_dir": str(p.parent), "error": repr(exc)})

    grouped = group_rows([r for r in per_run if "error" not in r], args.success_threshold)

    write_csv(out_dir / "design_sweeps_v0_19_per_run.csv", per_run)
    for name, rows in grouped.items():
        write_csv(out_dir / f"{name}_summary.csv", rows)
    all_group_rows = []
    for name, rows in grouped.items():
        all_group_rows.extend(rows)
    write_csv(out_dir / "design_sweeps_v0_19_group_summary.csv", all_group_rows)
    write_report(out_dir / "design_sweeps_v0_19_report.md", grouped, args.success_threshold)
    print(f"wrote summaries to {out_dir}")
    print(f"completed runs: {sum(1 for r in per_run if 'error' not in r)}; errors: {sum(1 for r in per_run if 'error' in r)}")


if __name__ == "__main__":
    main()
