from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_rows(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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


def _metric_keys(last: Dict[str, Any]) -> List[str]:
    priority_prefixes = ("", "queryA_", "queryB_", "base_", "fresh_")
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
    last = rows[-1]
    final_metrics = {k: last[k] for k in _metric_keys(last)}

    # Tail means for core accuracies and specificity-control accuracies.
    tail_metrics: Dict[str, float] = {}
    for k in _metric_keys(last):
        if k.endswith("_acc") or k in {
            "lr",
            "weight_norm",
            "stable_rank_mean",
            "effective_rank_mean",
            "update_to_weight_ratio",
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
