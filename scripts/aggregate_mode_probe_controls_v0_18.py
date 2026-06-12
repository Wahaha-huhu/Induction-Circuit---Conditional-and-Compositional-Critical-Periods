#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(x: Any) -> float | None:
    try:
        if x is None or x == '':
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def mean(vals: List[float]) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate v0.18 mode-specific probe controls.')
    p.add_argument('--controls-csv', required=True)
    p.add_argument('--out-layer', default=None)
    p.add_argument('--out-run', default=None)
    p.add_argument('--out-report', default=None)
    args = p.parse_args()
    rows = read_csv(Path(args.controls_csv))
    out_layer = Path(args.out_layer) if args.out_layer else Path(args.controls_csv).with_name('mode_probe_controls_layer_summary.csv')
    out_run = Path(args.out_run) if args.out_run else Path(args.controls_csv).with_name('mode_probe_controls_per_run.csv')
    out_report = Path(args.out_report) if args.out_report else Path(args.controls_csv).with_name('mode_probe_controls_report.md')

    # Layer-level group summary.
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get('arm',''), r.get('mode',''), r.get('label_name',''), r.get('layer',''))
        groups[key].append(r)
    layer_summary: List[Dict[str, Any]] = []
    for (arm, mode, label, layer), rs in sorted(groups.items()):
        layer_summary.append({
            'arm': arm,
            'mode': mode,
            'label_name': label,
            'layer': layer,
            'n': len(rs),
            'true_probe_eval_acc_mean': mean([num(r.get('true_probe_eval_acc')) for r in rs]),
            'shuffled_label_probe_eval_acc_mean': mean([num(r.get('shuffled_label_probe_eval_acc')) for r in rs]),
            'probe_acc_above_shuffle_mean': mean([num(r.get('probe_acc_above_shuffle')) for r in rs]),
            'majority_baseline_acc_mean': mean([num(r.get('majority_baseline_acc')) for r in rs]),
        })
    with out_layer.open('w', newline='', encoding='utf-8') as f:
        fields = list(layer_summary[0].keys()) if layer_summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(layer_summary)

    # Run-level maxima.
    run_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        run_groups[(r.get('run_dir',''), r.get('label_name',''))].append(r)
    run_summary: List[Dict[str, Any]] = []
    for (rd, label), rs in sorted(run_groups.items()):
        first = rs[0]
        true_vals = [num(r.get('true_probe_eval_acc')) for r in rs]
        shuf_vals = [num(r.get('shuffled_label_probe_eval_acc')) for r in rs]
        above_vals = [num(r.get('probe_acc_above_shuffle')) for r in rs]
        max_idx = max(range(len(rs)), key=lambda i: true_vals[i] if true_vals[i] is not None else -1)
        run_summary.append({
            'run_dir': rd,
            'arm': first.get('arm',''),
            'seed': first.get('seed',''),
            'mode': first.get('mode',''),
            'label_name': label,
            'best_layer': rs[max_idx].get('layer',''),
            'max_true_probe_eval_acc': true_vals[max_idx],
            'matched_shuffled_acc_at_best_layer': shuf_vals[max_idx],
            'max_probe_acc_above_shuffle': max([v for v in above_vals if v is not None], default=None),
            'max_lens_acc': max([v for v in [num(v) for r in rs for k,v in r.items() if k.endswith('_lens_acc') and label.split('_')[-1] in k] if v is not None], default=None),
        })
    with out_run.open('w', newline='', encoding='utf-8') as f:
        fields = list(run_summary[0].keys()) if run_summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(run_summary)

    report = ['# v0.18 mode-specific probe controls report\n']
    report.append(f'Loaded {len(rows)} layer-label rows.\n')
    report.append('## Run-level maxima\n')
    report.append('| arm | seed | mode | label | best layer | true acc | shuffled acc | true - shuffled |')
    report.append('|---|---:|---|---|---:|---:|---:|---:|')
    for r in run_summary:
        def fmt(x):
            v = num(x)
            return 'NA' if v is None else f'{v:.3f}'
        report.append(f"| {r['arm']} | {r['seed']} | {r['mode']} | {r['label_name']} | {r['best_layer']} | {fmt(r['max_true_probe_eval_acc'])} | {fmt(r['matched_shuffled_acc_at_best_layer'])} | {fmt(r['max_probe_acc_above_shuffle'])} |")
    report.append('\nInterpretation: true-probe accuracy should exceed shuffled-label controls by a large margin for genuine decodability.')
    out_report.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'wrote {out_layer}')
    print(f'wrote {out_run}')
    print(f'wrote {out_report}')


if __name__ == '__main__':
    main()
