#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return [dict(row) for row in csv.DictReader(f)]


def num(v: Any) -> Optional[float]:
    if v is None or v == '':
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def maxnum(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return max(xs)


def first_ge(rows: List[Dict[str, Any]], key: str, threshold: float) -> Optional[int]:
    for r in sorted(rows, key=lambda x: int(float(x['layer']))):
        v = num(r.get(key))
        if v is not None and v >= threshold:
            return int(float(r['layer']))
    return None


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate intermediate/answer probe CSVs.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'intermediate_answer_probe'
    files = sorted(root.rglob('*intermediate_answer_probe.csv'))
    # Avoid recursively re-reading aggregate files in the out directory.
    files = [f for f in files if f.name not in {'intermediate_answer_probe_all_layers.csv', 'intermediate_answer_probe_per_run.csv', 'intermediate_answer_probe_group_summary.csv'}]

    all_rows: List[Dict[str, Any]] = []
    for f in files:
        for r in read_csv(f):
            r['source_file'] = str(f)
            all_rows.append(r)
    if not all_rows:
        raise SystemExit(f'no intermediate_answer_probe.csv files found under {root}')

    per_key = defaultdict(list)
    for r in all_rows:
        key = (r.get('run_dir'), r.get('arm'), r.get('seed'), r.get('probe_group'))
        per_key[key].append(r)

    per_run: List[Dict[str, Any]] = []
    for (run_dir, arm, seed, group), rows in sorted(per_key.items(), key=lambda kv: (str(kv[0][3]), str(kv[0][1]), str(kv[0][2]))):
        rec: Dict[str, Any] = {
            'run_dir': run_dir,
            'arm': arm,
            'seed': seed,
            'probe_group': group,
            'n_layers': len(rows),
            'max_lens_intermediate_acc': maxnum(num(r.get('lens_intermediate_acc')) for r in rows),
            'max_lens_answer_acc': maxnum(num(r.get('lens_answer_acc')) for r in rows),
            'max_probe_intermediate_acc': maxnum(num(r.get('intermediate_probe_eval_acc')) for r in rows),
            'max_probe_answer_acc': maxnum(num(r.get('answer_probe_eval_acc')) for r in rows),
            'first_layer_intermediate_probe_ge_0p5': first_ge(rows, 'intermediate_probe_eval_acc', 0.5),
            'first_layer_answer_probe_ge_0p5': first_ge(rows, 'answer_probe_eval_acc', 0.5),
            'first_layer_intermediate_probe_ge_0p8': first_ge(rows, 'intermediate_probe_eval_acc', 0.8),
            'first_layer_answer_probe_ge_0p8': first_ge(rows, 'answer_probe_eval_acc', 0.8),
            'first_layer_intermediate_lens_ge_0p5': first_ge(rows, 'lens_intermediate_acc', 0.5),
            'first_layer_answer_lens_ge_0p5': first_ge(rows, 'lens_answer_acc', 0.5),
        }
        bi = rec['first_layer_intermediate_probe_ge_0p8']
        ca = rec['first_layer_answer_probe_ge_0p8']
        rec['answer_minus_intermediate_first_0p8_layer'] = (ca - bi) if bi is not None and ca is not None else None
        per_run.append(rec)

    group_map = defaultdict(list)
    for r in per_run:
        group_map[(r.get('probe_group'), r.get('arm'))].append(r)
    group_rows: List[Dict[str, Any]] = []
    for (group, arm), rows in sorted(group_map.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        group_rows.append({
            'probe_group': group,
            'arm': arm,
            'n_runs': len(rows),
            'mean_max_probe_intermediate_acc': mean(num(r.get('max_probe_intermediate_acc')) for r in rows),
            'mean_max_probe_answer_acc': mean(num(r.get('max_probe_answer_acc')) for r in rows),
            'mean_first_layer_intermediate_probe_ge_0p8': mean(num(r.get('first_layer_intermediate_probe_ge_0p8')) for r in rows),
            'mean_first_layer_answer_probe_ge_0p8': mean(num(r.get('first_layer_answer_probe_ge_0p8')) for r in rows),
            'mean_answer_minus_intermediate_first_0p8_layer': mean(num(r.get('answer_minus_intermediate_first_0p8_layer')) for r in rows),
            'mean_max_lens_intermediate_acc': mean(num(r.get('max_lens_intermediate_acc')) for r in rows),
            'mean_max_lens_answer_acc': mean(num(r.get('max_lens_answer_acc')) for r in rows),
        })

    all_out = out_dir / 'intermediate_answer_probe_all_layers.csv'
    per_out = out_dir / 'intermediate_answer_probe_per_run.csv'
    group_out = out_dir / 'intermediate_answer_probe_group_summary.csv'
    write_csv(all_out, all_rows)
    write_csv(per_out, per_run)
    write_csv(group_out, group_rows)

    report = out_dir / 'intermediate_answer_probe_report.md'
    lines = [
        '# Intermediate/answer probe report',
        '',
        'This aggregates query-position decodability for the intermediate token B and final answer C on forced HOP_2 examples.',
        '',
        '## Group summary',
        '',
        '| group | arm | n | max probe B | max probe C | first B>=0.8 | first C>=0.8 | C-B layer |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in group_rows:
        lines.append(
            f"| {r.get('probe_group')} | {r.get('arm')} | {r.get('n_runs')} | "
            f"{float(r['mean_max_probe_intermediate_acc'] or 0):.3f} | {float(r['mean_max_probe_answer_acc'] or 0):.3f} | "
            f"{r.get('mean_first_layer_intermediate_probe_ge_0p8')} | {r.get('mean_first_layer_answer_probe_ge_0p8')} | "
            f"{r.get('mean_answer_minus_intermediate_first_0p8_layer')} |"
        )
    lines += ['', '## Interpretation guide', '',
              '- If successful runs decode B earlier than C, this supports a sequential two-hop computation.',
              '- If failed S1 runs decode B but not C, the barrier localises to second-hop answer formation rather than first-hop retrieval.',
              '- If neither B nor C is decodable, the model may use a non-token-like or superposed mechanism.']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'wrote {all_out}')
    print(f'wrote {per_out}')
    print(f'wrote {group_out}')
    print(f'wrote {report}')


if __name__ == '__main__':
    main()
