#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def first_crossing(rows, key, threshold, consecutive=3):
    count = 0
    for r in rows:
        if r.get(key, float("-inf")) >= threshold:
            count += 1
            if count >= consecutive:
                return r["step"]
        else:
            count = 0
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--single-hop-threshold", type=float, default=0.95)
    p.add_argument("--multi-hop-threshold", type=float, default=0.80)
    args = p.parse_args()

    rows = load_rows(Path(args.run_dir) / "metrics.jsonl")
    last = rows[-1]
    print("final metrics:")
    for k in sorted(last):
        if k != "step":
            print(f"  {k}: {last[k]}")
    print("transitions:")
    print("  t_component_hop1:", first_crossing(rows, "hop1_acc", args.single_hop_threshold))
    # For K=2 the dependent is hop2. For larger K, check every hop manually.
    if "hop2_acc" in last:
        print("  t_dependent_hop2:", first_crossing(rows, "hop2_acc", args.multi_hop_threshold))
    print("  t_keyslot_top1_0p5:", first_crossing(rows, "keyslot_top1", 0.5 * max(r["keyslot_top1"] for r in rows)))


if __name__ == "__main__":
    main()
