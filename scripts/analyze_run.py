#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from cp_toy.summary import print_summary, write_summary


def main():
    p = argparse.ArgumentParser(description="Print and save final metrics/transitions for a run directory")
    p.add_argument("run_dir")
    p.add_argument("--single-hop-threshold", type=float, default=0.95)
    p.add_argument("--multi-hop-threshold", type=float, default=0.80)
    p.add_argument("--json", action="store_true", help="Print raw summary JSON instead of human-readable text")
    args = p.parse_args()

    summary = write_summary(
        args.run_dir,
        single_hop_threshold=args.single_hop_threshold,
        multi_hop_threshold=args.multi_hop_threshold,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)
        print(f"wrote {args.run_dir.rstrip('/')}/summary.json")


if __name__ == "__main__":
    main()
