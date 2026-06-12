#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import (
    compute_global_head_means,
    evaluate_by_hop,
    key_slot_lookup_scores,
)
from cp_toy.model import HeadSelection, TinyTransformer
from cp_toy.train import resolve_device

ARM_NAMES = {
    "s1_late_original", "s1_plateau_late", "s1_longcos_late", "s2_constant_late",
    "rewarm_late", "rewarm_reset_late", "fresh_hop1_s1", "fresh_hop1_s2",
}
DEFAULT_ARMS = "s1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.10) -> Optional[float]:
    vals = [numeric(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    return float(sum(vals[-n:]) / n)


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    if run_dir.parent.name in ARM_NAMES:
        return run_dir.parent.name
    return run_dir.name


def load_seed(run_dir: Path) -> Optional[int]:
    cfgp = run_dir / "config.json"
    if not cfgp.exists():
        return None
    try:
        cfg = json.loads(cfgp.read_text())
        seed = cfg.get("train", {}).get("seed")
        return int(seed) if seed is not None else None
    except Exception:
        return None


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("metrics.jsonl")):
        yield p.parent


def gini(values: List[float]) -> float:
    xs = [max(0.0, float(v)) for v in values if math.isfinite(float(v))]
    n = len(xs)
    if n == 0:
        return float("nan")
    s = sum(xs)
    if s <= 1e-12:
        return 0.0
    xs.sort()
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return float((2.0 * weighted) / (n * s) - (n + 1.0) / n)


def entropy_stats(values: List[float]) -> Dict[str, float]:
    xs = torch.tensor([max(0.0, float(v)) for v in values], dtype=torch.float64)
    n = xs.numel()
    if n == 0 or float(xs.sum().item()) <= 1e-12:
        return {"entropy": 0.0, "entropy_norm": 0.0, "effective_n": 0.0}
    p = xs / xs.sum()
    ent = float(-(p * (p + 1e-12).log()).sum().item())
    return {
        "entropy": ent,
        "entropy_norm": float(ent / math.log(n)) if n > 1 else 0.0,
        "effective_n": float(math.exp(ent)),
    }


def top_share(values: List[float], k: int) -> float:
    xs = sorted([max(0.0, float(v)) for v in values], reverse=True)
    total = sum(xs)
    if total <= 1e-12:
        return 0.0
    return float(sum(xs[:k]) / total)


def flat_head_selection(layer: int, head: int) -> HeadSelection:
    return {int(layer): [int(head)]}


def load_model(run_dir: Path, device: torch.device) -> Tuple[TinyTransformer, DataConfig, ModelConfig, Dict[str, Any]]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg["data"])
    model_cfg = ModelConfig(**cfg["model"])
    model = TinyTransformer(model_cfg).to(device)
    state = torch.load(run_dir / "model_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg, model_cfg, cfg


def select_runs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.runs_dir)
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    selected: List[Dict[str, Any]] = []
    for rd in iter_run_dirs(root):
        arm = derive_arm(rd)
        if arms and arm not in arms:
            continue
        if args.require_model and not (rd / "model_final.pt").exists():
            continue
        rows = read_jsonl(rd / "metrics.jsonl")
        for r in rows:
            a = numeric(r.get("hop2_acc"))
            f = numeric(r.get("floor_hop2_acc"))
            if a is not None and f is not None:
                r["hop2_excess"] = a - f
        tail_h2 = tail_mean(rows, "hop2_acc")
        tail_excess = tail_mean(rows, "hop2_excess")
        if args.success_only:
            if tail_h2 is None or tail_h2 < args.success_threshold:
                continue
            if tail_excess is None or tail_excess < args.min_excess:
                continue
        selected.append({
            "run_dir": str(rd),
            "arm": arm,
            "seed": load_seed(rd),
            "tail_hop2_acc": tail_h2,
            "tail_hop2_excess": tail_excess,
        })
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description="Head-score and causal-load concentration analysis.")
    p.add_argument("--runs-dir", default="runs/behavioral_replication_v0_9")
    p.add_argument("--out-dir", default="runs/behavioral_replication_v0_9/head_concentration")
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--mean-batches", type=int, default=4)
    p.add_argument("--success-only", action="store_true", default=True)
    p.add_argument("--include-failures", action="store_false", dest="success_only")
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--min-excess", type=float, default=0.50)
    p.add_argument("--require-model", action="store_true", default=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    runs = select_runs(args)
    per_head_rows: List[Dict[str, Any]] = []
    per_run_rows: List[Dict[str, Any]] = []

    print(f"selected {len(runs)} runs for concentration analysis")
    for rec in runs:
        rd = Path(rec["run_dir"])
        print(f"analyzing seed={rec.get('seed')} arm={rec['arm']} {rd}")
        model, data_cfg, model_cfg, _cfg = load_model(rd, device)
        gen = ChainBatchGenerator(data_cfg, seed=20260 + int(rec.get("seed") or 0))

        scores = key_slot_lookup_scores(model, gen, args.batch_size, args.eval_batches, device)
        score_vals = scores.flatten().tolist()
        means = compute_global_head_means(model, gen, args.batch_size, args.mean_batches, device)
        base = evaluate_by_hop(model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max)
        base_h1 = float(base.get("hop1_acc", float("nan")))
        base_h2 = float(base.get("hop2_acc", float("nan")))
        base_h2_loss = float(base.get("hop2_loss", float("nan")))

        drop_h1_vals: List[float] = []
        drop_h2_vals: List[float] = []
        loss_inc_vals: List[float] = []
        for layer in range(model_cfg.n_layers):
            for head in range(model_cfg.n_heads):
                ablated = evaluate_by_hop(
                    model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max,
                    ablate_heads=flat_head_selection(layer, head), ablation_means=means,
                )
                h1_acc = float(ablated.get("hop1_acc", float("nan")))
                h2_acc = float(ablated.get("hop2_acc", float("nan")))
                h2_loss = float(ablated.get("hop2_loss", float("nan")))
                d1 = base_h1 - h1_acc
                d2 = base_h2 - h2_acc
                dloss = h2_loss - base_h2_loss
                drop_h1_vals.append(max(0.0, d1))
                drop_h2_vals.append(max(0.0, d2))
                loss_inc_vals.append(max(0.0, dloss))
                per_head_rows.append({
                    **rec,
                    "layer": layer,
                    "head": head,
                    "keyslot_score": float(scores[layer, head].item()),
                    "base_hop1_acc": base_h1,
                    "base_hop2_acc": base_h2,
                    "ablated_hop1_acc": h1_acc,
                    "ablated_hop2_acc": h2_acc,
                    "hop1_drop": d1,
                    "hop2_drop": d2,
                    "hop2_loss_increase": dloss,
                })

        score_ent = entropy_stats(score_vals)
        drop_ent = entropy_stats(drop_h2_vals)
        loss_ent = entropy_stats(loss_inc_vals)
        per_run_rows.append({
            **rec,
            "n_heads": model_cfg.n_layers * model_cfg.n_heads,
            "base_hop1_acc": base_h1,
            "base_hop2_acc": base_h2,
            "keyslot_score_gini": gini(score_vals),
            "keyslot_score_entropy_norm": score_ent["entropy_norm"],
            "keyslot_score_effective_n": score_ent["effective_n"],
            "keyslot_score_top1_share": top_share(score_vals, 1),
            "keyslot_score_top2_share": top_share(score_vals, 2),
            "keyslot_score_top4_share": top_share(score_vals, 4),
            "hop2_drop_gini": gini(drop_h2_vals),
            "hop2_drop_entropy_norm": drop_ent["entropy_norm"],
            "hop2_drop_effective_n": drop_ent["effective_n"],
            "hop2_drop_top1_share": top_share(drop_h2_vals, 1),
            "hop2_drop_top2_share": top_share(drop_h2_vals, 2),
            "hop2_drop_top4_share": top_share(drop_h2_vals, 4),
            "hop2_lossinc_gini": gini(loss_inc_vals),
            "hop2_lossinc_entropy_norm": loss_ent["entropy_norm"],
            "hop2_lossinc_effective_n": loss_ent["effective_n"],
            "hop2_lossinc_top4_share": top_share(loss_inc_vals, 4),
            "max_single_hop2_drop": max(drop_h2_vals) if drop_h2_vals else float("nan"),
            "mean_single_hop2_drop": sum(drop_h2_vals) / len(drop_h2_vals) if drop_h2_vals else float("nan"),
        })

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "head_concentration_per_head.csv", per_head_rows)
    write_csv(out_dir / "head_concentration_per_run.csv", per_run_rows)

    # group summary by arm
    group_rows: List[Dict[str, Any]] = []
    arms = sorted({r["arm"] for r in per_run_rows})
    metrics = [
        "keyslot_score_gini", "keyslot_score_effective_n", "keyslot_score_top4_share",
        "hop2_drop_gini", "hop2_drop_effective_n", "hop2_drop_top4_share",
        "max_single_hop2_drop", "mean_single_hop2_drop",
    ]
    for arm in arms:
        rows = [r for r in per_run_rows if r["arm"] == arm]
        out = {"arm": arm, "n": len(rows)}
        for m in metrics:
            vals = [numeric(r.get(m)) for r in rows]
            vals = [v for v in vals if v is not None]
            out[f"{m}_mean"] = sum(vals) / len(vals) if vals else None
            out[f"{m}_median"] = sorted(vals)[len(vals)//2] if vals else None
        group_rows.append(out)
    write_csv(out_dir / "head_concentration_group_summary.csv", group_rows)

    report = out_dir / "head_concentration_report.md"
    report.write_text(
        "# Head concentration analysis\n\n"
        "This analysis separates two notions that should not be conflated:\n\n"
        "1. **Attention concentration**: Gini/entropy over key-slot lookup attention scores.\n"
        "2. **Causal-load concentration**: Gini/entropy over single-head HOP_2 ablation drops.\n\n"
        "A high key-slot-score Gini is descriptive; it does not by itself prove causal localization. "
        "The causal-drop Gini is the stronger readout.\n\n"
        f"Selected runs: {len(per_run_rows)}\n\n"
        "Outputs:\n"
        "- `head_concentration_per_head.csv`\n"
        "- `head_concentration_per_run.csv`\n"
        "- `head_concentration_group_summary.csv`\n",
        encoding="utf-8",
    )
    manifest = {
        "args": vars(args),
        "n_runs": len(per_run_rows),
        "outputs": [
            "head_concentration_per_head.csv",
            "head_concentration_per_run.csv",
            "head_concentration_group_summary.csv",
            "head_concentration_report.md",
        ],
    }
    (out_dir / "head_concentration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
