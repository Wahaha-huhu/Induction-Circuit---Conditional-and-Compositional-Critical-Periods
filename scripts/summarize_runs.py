#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List
from cp_toy.summary import write_summary


def find_run_dirs(root: Path) -> List[Path]:
    return sorted(p.parent for p in root.rglob("metrics.jsonl"))


def add_flat(flat: Dict[str, Any], prefix: str, d: Dict[str, Any]) -> None:
    for k, v in d.items():
        flat[f"{prefix}.{k}"] = v


def main():
    p = argparse.ArgumentParser(description="Summarize every run under a runs directory")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default=None, help="CSV output path; defaults to <runs-dir>/all_run_summaries.csv")
    p.add_argument("--single-hop-threshold", type=float, default=0.95)
    p.add_argument("--multi-hop-threshold", type=float, default=0.80)
    args = p.parse_args()

    root = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else root / "all_run_summaries.csv"
    run_dirs = find_run_dirs(root)
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        s = write_summary(rd, single_hop_threshold=args.single_hop_threshold, multi_hop_threshold=args.multi_hop_threshold)
        flat: Dict[str, Any] = {
            "run_dir": s["run_dir"],
            "final_step": s["final_step"],
            "n_eval_rows": s["n_eval_rows"],
        }
        add_flat(flat, "meta", s.get("metadata", {}))
        add_flat(flat, "intro", s.get("intro_snapshot", {}))
        add_flat(flat, "tsched", s.get("t_schedule_snapshot", {}))
        for prefix, d in [
            ("final", s["final_metrics"]),
            ("tail", s["tail_metrics"]),
            ("excess", s["excess_over_floor"]),
            ("trans", s["transitions"]),
        ]:
            add_flat(flat, prefix, d)
        rows.append(flat)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"summarized {len(rows)} runs")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
