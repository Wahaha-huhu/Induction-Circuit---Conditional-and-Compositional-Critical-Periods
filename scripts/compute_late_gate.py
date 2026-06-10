#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path


def load_last_mean(run_dir: Path, hop_key: str = "hop2_acc", floor_key: str = "floor_hop2_acc", frac: float = 0.10):
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl", "r", encoding="utf-8")]
    n = max(1, int(len(rows) * frac))
    tail = rows[-n:]
    excess = [(r[hop_key] - r.get(floor_key, 0.0)) for r in tail]
    return sum(excess) / len(excess)


def main():
    p = argparse.ArgumentParser(description="Apply v4.4 late-S1 gate threshold")
    p.add_argument("--early-run", required=True)
    p.add_argument("--late-run", required=True)
    p.add_argument("--early-min", type=float, default=0.50)
    args = p.parse_args()

    early = load_last_mean(Path(args.early_run))
    late = load_last_mean(Path(args.late_run))
    ratio = late / early if early > 1e-9 else float("inf")
    if early < args.early_min:
        decision = "invalid_early_reference"
    elif ratio <= 0.25:
        decision = "strong_closure__proceed_C5_C5b"
    elif ratio <= 0.50:
        decision = "impaired__proceed_C5_C5b"
    elif ratio <= 0.75:
        decision = "ambiguous__run_one_more_gate_seed"
    else:
        decision = "no_window__run_full_C4_S1_then_skip_C5_C5b"
    print(json.dumps({"early_excess": early, "late_excess": late, "late_to_early_ratio": ratio, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
