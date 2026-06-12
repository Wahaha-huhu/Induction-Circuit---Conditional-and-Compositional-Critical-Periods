#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(x: Any) -> float | None:
    try:
        if x is None or x == '':
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def first_cross(rows: List[Dict[str, Any]], key: str, threshold: float) -> float | None:
    rows = sorted(rows, key=lambda r: num(r.get('step')) or -1)
    for r in rows:
        v = num(r.get(key))
        if v is not None and v >= threshold:
            return num(r.get('step'))
    return None


def delta(rows: List[Dict[str, Any]], key: str) -> float | None:
    rows = sorted(rows, key=lambda r: num(r.get('step')) or -1)
    vals = [(num(r.get('step')), num(r.get(key))) for r in rows]
    vals = [(s, v) for s, v in vals if s is not None and v is not None]
    if len(vals) < 2:
        return None
    return vals[-1][1] - vals[0][1]


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate E4 transition-tracking CSVs.')
    p.add_argument('--tracking-dir', default='runs/checkpointed_geometry_followups/transition_tracking')
    p.add_argument('--out-per-checkpoint', default=None)
    p.add_argument('--out-summary', default=None)
    p.add_argument('--out-report', default=None)
    args = p.parse_args()

    root = Path(args.tracking_dir)
    files = sorted(root.rglob('transition_tracking.csv')) + sorted(root.rglob('*_transition_tracking.csv'))
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    files = unique
    all_rows: List[Dict[str, Any]] = []
    for f in files:
        all_rows.extend(read_csv(f))
    if not all_rows:
        raise FileNotFoundError(f'No transition_tracking.csv files found under {root}')

    out_per = Path(args.out_per_checkpoint) if args.out_per_checkpoint else root / 'transition_tracking_per_checkpoint.csv'
    out_sum = Path(args.out_summary) if args.out_summary else root / 'transition_tracking_summary.csv'
    out_rep = Path(args.out_report) if args.out_report else root / 'transition_tracking_report.md'
    out_per.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for r in all_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out_per.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in all_rows:
        groups.setdefault((r.get('arm', ''), r.get('seed', ''), r.get('run_dir', '')), []).append(r)

    summary: List[Dict[str, Any]] = []
    for (arm, seed, run_dir), rows in sorted(groups.items()):
        rows_s = sorted(rows, key=lambda r: num(r.get('step')) or -1)
        first = rows_s[0]
        last = rows_s[-1]
        rec = {
            'arm': arm,
            'seed': seed,
            'run_dir': run_dir,
            'n_checkpoints': len(rows_s),
            'first_step': first.get('step'),
            'last_step': last.get('step'),
            'initial_hop2_acc': first.get('hop2_acc'),
            'final_hop2_acc': last.get('hop2_acc'),
            'initial_hop2_excess': first.get('hop2_excess'),
            'final_hop2_excess': last.get('hop2_excess'),
            't_hop2_ge_0p5': first_cross(rows_s, 'hop2_acc', 0.50),
            't_hop2_ge_0p95': first_cross(rows_s, 'hop2_acc', 0.95),
            'delta_second_key_max': delta(rows_s, 'second_key_max'),
            'delta_second_value_max': delta(rows_s, 'second_value_max'),
            'delta_first_value_max': delta(rows_s, 'first_value_max'),
            'delta_lens_L3_C_acc': delta(rows_s, 'lens_L3_C_acc'),
            'delta_lens_L1_B_acc': delta(rows_s, 'lens_L1_B_acc'),
            'final_second_key_max': last.get('second_key_max'),
            'final_lens_L3_C_acc': last.get('lens_L3_C_acc'),
            'final_lens_L1_B_acc': last.get('lens_L1_B_acc'),
        }
        summary.append(rec)

    sum_fields: List[str] = []
    for r in summary:
        for k in r.keys():
            if k not in sum_fields:
                sum_fields.append(k)
    with out_sum.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    lines = [
        '# E4 transition tracking report',
        '',
        f'Found {len(summary)} runs and {len(all_rows)} checkpoint evaluations.',
        '',
        '| arm | seed | checkpoints | final HOP_2 | t HOP_2>=95% | Δ second-key max | Δ L3 C lens |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary:
        def fmt(v):
            vv = num(v)
            if vv is None:
                return '—' if v in (None, '') else str(v)
            if abs(vv) >= 100:
                return f'{vv:.0f}'
            return f'{vv:.3f}'
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['n_checkpoints']} | {fmt(r['final_hop2_acc'])} | {fmt(r['t_hop2_ge_0p95'])} | {fmt(r['delta_second_key_max'])} | {fmt(r['delta_lens_L3_C_acc'])} |"
        )
    lines.extend([
        '',
        'Interpretation guide:',
        '- Success should show HOP_2 accuracy/excess and late-layer answer representation rising together.',
        '- If second-key attention rises near the behavioural transition, it supports the two-hop-routing diagnostic.',
        '- If failed runs keep HOP_2 and L3 answer lens near floor, this supports failure to instantiate HOP_2 mode.',
    ])
    out_rep.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {out_per}')
    print(f'wrote {out_sum}')
    print(f'wrote {out_rep}')


if __name__ == '__main__':
    main()
