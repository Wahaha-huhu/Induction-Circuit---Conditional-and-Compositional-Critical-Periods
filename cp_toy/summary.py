from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import OptimConfig, ScheduleConfig
from .schedules import lr_at_step


def load_rows(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_config(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir) / "config.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_crossing(rows: List[Dict[str, Any]], key: str, threshold: float, consecutive: int = 3) -> Optional[int]:
    count = 0
    for r in rows:
        val = r.get(key, None)
        ok = False
        if isinstance(val, (int, float)):
            ok = val >= threshold
        if ok:
            count += 1
            if count >= consecutive:
                return int(r["step"])
        else:
            count = 0
    return None


def tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.10) -> Optional[float]:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    n = max(1, int(round(len(vals) * frac)))
    return float(sum(vals[-n:]) / n)


def max_numeric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(max(vals))


def nearest_row(rows: List[Dict[str, Any]], step: Optional[int]) -> Optional[Dict[str, Any]]:
    if step is None or not rows:
        return None
    return min(rows, key=lambda r: abs(int(r["step"]) - int(step)))


def _metric_keys(last: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for k, v in last.items():
        if not isinstance(v, (int, float)):
            continue
        if k == "step":
            continue
        if (
            k.endswith("_acc")
            or k.endswith("_loss")
            or k.startswith("floor_")
            or "floor_" in k
            or "keyslot" in k
            or k in {
                "train_loss",
                "train_acc",
                "lr",
                "p_dynamic",
                "p_multi",
                "weight_norm",
                "stable_rank_mean",
                "effective_rank_mean",
                "top_singular_value_mean",
                "top_singular_value_max",
                "update_to_weight_ratio",
            }
        ):
            keys.append(k)
    return sorted(keys)


def _safe_lr_at_step(step: Optional[int], cfg: Dict[str, Any]) -> Optional[float]:
    if step is None or not cfg:
        return None
    try:
        optim = OptimConfig(**cfg.get("optim", {}))
        sched = ScheduleConfig(**cfg.get("schedule", {}))
        return float(lr_at_step(int(step), optim, sched))
    except Exception:
        return None


def _schedule_metadata(run_dir: Path, rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    train = cfg.get("train", {}) if cfg else {}
    optim = cfg.get("optim", {}) if cfg else {}
    sched = cfg.get("schedule", {}) if cfg else {}
    last = rows[-1]
    final_step = int(last["step"])
    intro_step = train.get("intro_step")
    max_steps = train.get("max_steps")
    t_schedule = sched.get("t_schedule")
    schedule_kind = sched.get("kind")
    rewarm_step = sched.get("rewarm_step")

    intro_eval = nearest_row(rows, intro_step)
    t_schedule_eval = nearest_row(rows, t_schedule)

    if intro_step is None:
        post_intro_steps = None
    else:
        post_intro_steps = max(0, int(max_steps if max_steps is not None else final_step + 1) - int(intro_step))

    if t_schedule is None:
        steps_after_t_schedule = None
    else:
        steps_after_t_schedule = max(0, int(max_steps if max_steps is not None else final_step + 1) - int(t_schedule))

    lr_tail = tail_mean(rows, "lr")
    lr_intro_eval = intro_eval.get("lr") if intro_eval else None
    lr_t_schedule_eval = t_schedule_eval.get("lr") if t_schedule_eval else None

    # This label is descriptive only; it prevents accidentally interpreting an S1
    # extension beyond t_schedule as ordinary high-LR extra training.
    extension_label = None
    if schedule_kind == "warmup_cosine" and t_schedule is not None and max_steps is not None:
        if int(max_steps) > int(t_schedule):
            extension_label = "cosine_final_lr_plateau_extension"
        elif int(max_steps) == int(t_schedule):
            extension_label = "cosine_full_horizon"
        else:
            extension_label = "cosine_pre_horizon"
    elif schedule_kind in {"warmup_cosine_then_rewarm_constant", "warmup_cosine_then_rewarm_constant_reset_optim"}:
        extension_label = "cosine_history_then_high_lr_rewarm"
    elif schedule_kind == "warmup_constant":
        extension_label = "constant_high_lr"
    elif schedule_kind == "warmup_cyclic":
        extension_label = "cyclic_lr"

    return {
        "condition": run_dir.name,
        "schedule_kind": schedule_kind,
        "extension_label": extension_label,
        "seed": train.get("seed"),
        "intro_step": intro_step,
        "max_steps": max_steps,
        "final_step": final_step,
        "t_schedule": t_schedule,
        "rewarm_step": rewarm_step,
        "peak_lr": optim.get("peak_lr"),
        "final_lr": optim.get("final_lr"),
        "rewarm_lr": sched.get("rewarm_lr"),
        "post_intro_steps": post_intro_steps,
        "steps_after_t_schedule": steps_after_t_schedule,
        "lr_at_intro_formula": _safe_lr_at_step(intro_step, cfg),
        "lr_at_intro_eval": lr_intro_eval,
        "intro_eval_step": int(intro_eval["step"]) if intro_eval else None,
        "lr_at_t_schedule_formula": _safe_lr_at_step(t_schedule, cfg),
        "lr_at_t_schedule_eval": lr_t_schedule_eval,
        "t_schedule_eval_step": int(t_schedule_eval["step"]) if t_schedule_eval else None,
        "lr_final_eval": last.get("lr"),
        "lr_tail_mean": lr_tail,
    }


def _snapshot_metrics(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {}
    out: Dict[str, Any] = {"step": int(row["step"])}
    for k in _metric_keys(row):
        if k in row:
            out[k] = row[k]
    return out


def summarize_run(
    run_dir: str | Path,
    single_hop_threshold: float = 0.95,
    multi_hop_threshold: float = 0.80,
    keyslot_frac: float = 0.5,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    rows = load_rows(run_dir)
    if not rows:
        raise ValueError(f"no rows in {run_dir / 'metrics.jsonl'}")
    cfg = load_config(run_dir)
    last = rows[-1]
    final_metrics = {k: last[k] for k in _metric_keys(last)}

    metadata = _schedule_metadata(run_dir, rows, cfg)
    intro_row = nearest_row(rows, metadata.get("intro_step"))
    t_schedule_row = nearest_row(rows, metadata.get("t_schedule"))

    # Tail means for core accuracies and specificity-control accuracies.
    tail_metrics: Dict[str, float] = {}
    for k in _metric_keys(last):
        if k.endswith("_acc") or k in {
            "lr",
            "weight_norm",
            "stable_rank_mean",
            "effective_rank_mean",
            "update_to_weight_ratio",
            "top_singular_value_mean",
            "top_singular_value_max",
        }:
            tm = tail_mean(rows, k)
            if tm is not None:
                tail_metrics[k] = tm

    transitions: Dict[str, Optional[int]] = {}
    # Generic transitions for all hop accuracies, including prefixed panels.
    for k in sorted(last.keys()):
        if not k.endswith("_acc"):
            continue
        if "hop1_acc" in k:
            transitions[f"t_{k}_ge_{single_hop_threshold}"] = first_crossing(rows, k, single_hop_threshold)
            transitions[f"t_{k}_ge_0.99"] = first_crossing(rows, k, 0.99)
        elif "hop" in k:
            transitions[f"t_{k}_ge_0.50"] = first_crossing(rows, k, 0.50)
            transitions[f"t_{k}_ge_{multi_hop_threshold}"] = first_crossing(rows, k, multi_hop_threshold)
            transitions[f"t_{k}_ge_0.95"] = first_crossing(rows, k, 0.95)
            transitions[f"t_{k}_ge_0.99"] = first_crossing(rows, k, 0.99)

    # Keyslot half-max transitions for every keyslot_top1-like key.
    for k in sorted(last.keys()):
        if k.endswith("keyslot_top1") or k == "keyslot_top1":
            mx = max_numeric(rows, k)
            if mx is not None:
                transitions[f"t_{k}_ge_half_max"] = first_crossing(rows, k, keyslot_frac * mx)

    # Excess-over-floor summaries for hop1/hop2 panels where floor exists.
    excess: Dict[str, Optional[float]] = {}
    prefixes = ["", "queryA_", "queryB_", "base_", "fresh_"]
    for prefix in prefixes:
        for h in (1, 2, 3, 4):
            acc_key = f"{prefix}hop{h}_acc"
            floor_key = f"{prefix}floor_hop{h}_acc"
            if acc_key in last and floor_key in last:
                acc_tail = tail_mean(rows, acc_key)
                floor_tail = tail_mean(rows, floor_key)
                if acc_tail is not None and floor_tail is not None:
                    excess[f"{prefix}hop{h}_excess_tail"] = max(0.0, acc_tail - floor_tail)

    out = {
        "run_dir": str(run_dir),
        "n_eval_rows": len(rows),
        "final_step": int(last["step"]),
        "metadata": metadata,
        "intro_snapshot": _snapshot_metrics(intro_row),
        "t_schedule_snapshot": _snapshot_metrics(t_schedule_row),
        "final_metrics": final_metrics,
        "tail_metrics": tail_metrics,
        "excess_over_floor": excess,
        "transitions": transitions,
    }
    return out


def write_summary(run_dir: str | Path, **kwargs: Any) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    summary = summarize_run(run_dir, **kwargs)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def print_summary(summary: Dict[str, Any], max_metric_lines: int = 80) -> None:
    print("\n=== RUN SUMMARY ===")
    print(f"run_dir: {summary['run_dir']}")
    print(f"final_step: {summary['final_step']}  eval_rows: {summary['n_eval_rows']}")

    meta = summary.get("metadata", {})
    if meta:
        print("\nschedule/window metadata:")
        for k in [
            "schedule_kind",
            "extension_label",
            "seed",
            "intro_step",
            "max_steps",
            "t_schedule",
            "post_intro_steps",
            "steps_after_t_schedule",
            "peak_lr",
            "final_lr",
            "rewarm_step",
            "rewarm_lr",
            "lr_at_intro_formula",
            "lr_final_eval",
            "lr_tail_mean",
        ]:
            print(f"  {k}: {meta.get(k)}")

    if summary.get("intro_snapshot"):
        print("\nintro snapshot, nearest eval row:")
        for k, v in sorted(summary["intro_snapshot"].items()):
            print(f"  {k}: {v}")

    print("\nfinal metrics:")
    items = sorted(summary["final_metrics"].items())
    for i, (k, v) in enumerate(items):
        if i >= max_metric_lines:
            print(f"  ... ({len(items) - max_metric_lines} more metrics omitted; see summary.json)")
            break
        print(f"  {k}: {v}")

    if summary.get("excess_over_floor"):
        print("\nexcess over floor, tail mean:")
        for k, v in sorted(summary["excess_over_floor"].items()):
            print(f"  {k}: {v}")

    print("\ntransitions:")
    for k, v in sorted(summary["transitions"].items()):
        print(f"  {k}: {v}")
    print("=== END SUMMARY ===\n")
