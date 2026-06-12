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
    return None if not xs else sum(xs) / len(xs)


def maxnum(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return None if not xs else max(xs)


def first_layer_at(rows: List[Dict[str, Any]], metric: str, threshold: float) -> Optional[int]:
    for r in sorted(rows, key=lambda x: int(float(x.get('layer', 0)))):
        v = num(r.get(metric))
        if v is not None and v >= threshold:
            return int(float(r.get('layer', 0)))
    return None


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate v0.15 mode-specific HOP_1/HOP_2 probe outputs.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9/mode_specific_probe')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root

    files = sorted((root / 'per_run').glob('*mode_specific_probe.csv')) if (root / 'per_run').exists() else sorted(root.rglob('*mode_specific_probe.csv'))
    all_rows: List[Dict[str, Any]] = []
    for f in files:
        all_rows.extend(read_csv(f))
    if not all_rows:
        raise SystemExit(f'no mode_specific_probe csv files found under {root}')

    per_run: List[Dict[str, Any]] = []
    by_run: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_run[(r.get('run_dir'), r.get('arm'), r.get('seed'), r.get('probe_group'))].append(r)

    metrics = {
        'hop1_prompt': ['hop1_target'],
        'hop2_prompt': ['hop2_intermediate', 'hop2_answer'],
    }
    for (run_dir, arm, seed, group), rows in by_run.items():
        rec: Dict[str, Any] = {'run_dir': run_dir, 'arm': arm, 'seed': seed, 'probe_group': group}
        for mode, labels in metrics.items():
            mrows = [r for r in rows if r.get('mode') == mode]
            for label in labels:
                for kind in ['probe_eval_acc', 'lens_acc']:
                    metric = f'{label}_{kind}'
                    rec[f'max_{mode}_{label}_{kind}'] = maxnum(num(r.get(metric)) for r in mrows)
                    rec[f'first_layer_{mode}_{label}_{kind}_ge_0p5'] = first_layer_at(mrows, metric, 0.5)
                    rec[f'first_layer_{mode}_{label}_{kind}_ge_0p8'] = first_layer_at(mrows, metric, 0.8)
        h1 = num(rec.get('max_hop1_prompt_hop1_target_probe_eval_acc'))
        h2b = num(rec.get('max_hop2_prompt_hop2_intermediate_probe_eval_acc'))
        h2c = num(rec.get('max_hop2_prompt_hop2_answer_probe_eval_acc'))
        if h1 is not None and h2b is not None:
            rec['mode_gap_hop1_target_minus_hop2_intermediate_probe_acc'] = h1 - h2b
        if h2b is not None and h2c is not None:
            rec['hop2_answer_minus_intermediate_probe_acc'] = h2c - h2b
        rec['supports_recruitment_failure'] = bool(h1 is not None and h2b is not None and h1 >= 0.8 and h2b < 0.3)
        per_run.append(rec)

    group_rows: List[Dict[str, Any]] = []
    by_group: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_run:
        by_group[(r.get('probe_group'), r.get('arm'))].append(r)
    for (group, arm), rows in sorted(by_group.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        rec: Dict[str, Any] = {'probe_group': group, 'arm': arm, 'n_runs': len(rows)}
        for key in [
            'max_hop1_prompt_hop1_target_probe_eval_acc',
            'max_hop2_prompt_hop2_intermediate_probe_eval_acc',
            'max_hop2_prompt_hop2_answer_probe_eval_acc',
            'mode_gap_hop1_target_minus_hop2_intermediate_probe_acc',
            'hop2_answer_minus_intermediate_probe_acc',
        ]:
            rec[f'mean_{key}'] = mean(num(r.get(key)) for r in rows)
        rec['n_supports_recruitment_failure'] = sum(1 for r in rows if str(r.get('supports_recruitment_failure')).lower() == 'true')
        group_rows.append(rec)

    all_out = out_dir / 'mode_specific_probe_all_layers.csv'
    per_out = out_dir / 'mode_specific_probe_per_run.csv'
    group_out = out_dir / 'mode_specific_probe_group_summary.csv'
    report = out_dir / 'mode_specific_probe_report.md'
    write_csv(all_out, all_rows)
    write_csv(per_out, per_run)
    write_csv(group_out, group_rows)

    lines = [
        '# Mode-specific HOP_1 vs HOP_2 intermediate probe',
        '',
        'This analysis tests whether failed late models still represent the lookup target under a HOP_1 prompt, but fail to recruit that lookup under a HOP_2 prompt.',
        '',
        'Key prediction for a recruitment failure:',
        '',
        '```text',
        'failed S1 late:',
        '  HOP_1 prompt target B decodable',
        '  HOP_2 prompt intermediate B not decodable',
        '```',
        '',
        '## Group means',
        '',
        '| group | arm | n | HOP1 target probe | HOP2 intermediate probe | HOP2 answer probe | HOP1-HOP2inter gap |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in group_rows:
        def fmt(x):
            v = num(x)
            return '' if v is None else f'{v:.3f}'
        lines.append(
            f"| {r.get('probe_group')} | {r.get('arm')} | {r.get('n_runs')} | "
            f"{fmt(r.get('mean_max_hop1_prompt_hop1_target_probe_eval_acc'))} | "
            f"{fmt(r.get('mean_max_hop2_prompt_hop2_intermediate_probe_eval_acc'))} | "
            f"{fmt(r.get('mean_max_hop2_prompt_hop2_answer_probe_eval_acc'))} | "
            f"{fmt(r.get('mean_mode_gap_hop1_target_minus_hop2_intermediate_probe_acc'))} |"
        )
    lines += [
        '',
        'Interpretation guide:',
        '',
        '- High HOP1 target decoding but low HOP2 intermediate decoding supports failure to recruit lookup machinery in HOP_2 mode.',
        '- High HOP2 intermediate but low HOP2 answer would localise the bottleneck to second-hop answer formation.',
        '- High HOP2 answer in successful runs confirms the late-layer answer representation observed in v0.14.',
    ]
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {all_out}')
    print(f'wrote {per_out}')
    print(f'wrote {group_out}')
    print(f'wrote {report}')


if __name__ == '__main__':
    main()
