#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ARM_NAMES = {
    's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
    'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2',
}


def numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def mean(vals: List[Any]) -> Optional[float]:
    xs = [numeric(v) for v in vals]
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(vals: List[Any]) -> Optional[float]:
    xs = [numeric(v) for v in vals]
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def derive_arm_from_run_dir(run_dir: str) -> str:
    p = Path(run_dir)
    for part in reversed(p.parts):
        if part in ARM_NAMES:
            return part
    return p.parent.name


def iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('twohop_score_ablation_*_top*.json')):
        yield p


def topk_from_name(path: Path) -> Optional[int]:
    m = re.search(r'_top(\d+)\.json$', path.name)
    return int(m.group(1)) if m else None


def seed_from_run_dir(run_dir: str) -> Optional[int]:
    m = re.search(r'seed(\d+)', run_dir)
    return int(m.group(1)) if m else None


def selection_str(sel: Any) -> str:
    if isinstance(sel, dict):
        pairs = []
        for layer in sorted(sel, key=lambda x: int(x)):
            for h in sel[layer]:
                pairs.append(f'L{layer}H{h}')
        return ','.join(pairs)
    if isinstance(sel, list):
        return ','.join(f'L{l}H{h}' for l, h in sel)
    return str(sel)


def summarize_one(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text())
    run_dir = str(d.get('run_dir', ''))
    s = d.get('summary', {})
    base = d.get('base', {})
    selected = d.get('selected_heads', {})
    row: Dict[str, Any] = {
        'json_path': str(path),
        'run_dir': run_dir,
        'arm': derive_arm_from_run_dir(run_dir),
        'seed': seed_from_run_dir(run_dir),
        'top_k': d.get('top_k') if d.get('top_k') is not None else topk_from_name(path),
        'num_random': d.get('num_random'),
        'base_hop1_acc': numeric(base.get('hop1_acc')),
        'base_hop2_acc': numeric(base.get('hop2_acc')),
    }
    for name, sel in selected.items():
        row[f'{name}_selected_heads'] = selection_str(sel)
    for k, v in s.items():
        row[k] = numeric(v) if numeric(v) is not None else v
    # Relative comparisons of score rankings.
    for score_name in ['first_value', 'second_value', 'second_key', 'causal_drop']:
        d2 = numeric(row.get(f'{score_name}_hop2_drop'))
        r2 = numeric(row.get('random_hop2_drop_mean'))
        if d2 is not None and r2 is not None:
            row[f'{score_name}_minus_random_hop2_drop'] = d2 - r2
        if score_name != 'second_value':
            sv = numeric(row.get('second_value_hop2_drop'))
            if d2 is not None and sv is not None:
                row[f'second_value_minus_{score_name}_hop2_drop'] = sv - d2
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys = sorted({k for r in rows for k in r})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get('arm')), int(r.get('top_k') or -1))].append(r)
    out: List[Dict[str, Any]] = []
    score_names = ['first_value', 'second_value', 'second_key', 'causal_drop']
    for (arm, top_k), rs in sorted(groups.items()):
        g: Dict[str, Any] = {
            'arm': arm,
            'top_k': top_k,
            'n_runs': len(rs),
            'seeds': ' '.join(str(r.get('seed')) for r in rs),
            'mean_base_hop2_acc': mean([r.get('base_hop2_acc') for r in rs]),
            'mean_random_hop2_drop': mean([r.get('random_hop2_drop_mean') for r in rs]),
        }
        for name in score_names:
            g[f'mean_{name}_hop2_drop'] = mean([r.get(f'{name}_hop2_drop') for r in rs])
            g[f'median_{name}_hop2_drop'] = median([r.get(f'{name}_hop2_drop') for r in rs])
            g[f'mean_{name}_ablated_hop2_acc'] = mean([r.get(f'{name}_hop2_ablated_acc') for r in rs])
            g[f'mean_{name}_minus_random_hop2_drop'] = mean([r.get(f'{name}_minus_random_hop2_drop') for r in rs])
            g[f'n_{name}_drop_gt_random_mean'] = sum(1 for r in rs if (numeric(r.get(f'{name}_minus_random_hop2_drop')) or -999) > 0)
        g['mean_second_value_minus_first_value_hop2_drop'] = mean([r.get('second_value_minus_first_value_hop2_drop') for r in rs])
        g['n_second_value_gt_first_value'] = sum(1 for r in rs if (numeric(r.get('second_value_minus_first_value_hop2_drop')) or -999) > 0)
        g['n_second_value_gt_random'] = sum(1 for r in rs if (numeric(r.get('second_value_minus_random_hop2_drop')) or -999) > 0)
        out.append(g)
    return out


def fmt(v: Any, pct: bool = False) -> str:
    x = numeric(v)
    if x is None:
        return '—'
    return f'{100*x:.1f}%' if pct else f'{x:.4g}'


def write_report(path: Path, per_rows: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append('# Two-hop decomposition score ablation\n')
    lines.append('This report compares head rankings for HOP_2 examples: first-hop value score (query -> B), second-hop value score (query -> C), second-hop key score (query -> B key in the B->C binding), and matched random-k head ablation. The key test is whether second-hop-ranked heads disrupt HOP_2 more than first-hop-ranked heads and random matched heads.\n')
    lines.append('## Group summary\n')
    lines.append('| arm | top-k | n | base HOP_2 | first-value drop | second-value drop | second-key drop | random drop | second - first | second > first | second > random |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for g in groups:
        lines.append(
            f"| {g.get('arm')} | {g.get('top_k')} | {g.get('n_runs')} | {fmt(g.get('mean_base_hop2_acc'), pct=True)} | "
            f"{fmt(g.get('mean_first_value_hop2_drop'), pct=True)} | {fmt(g.get('mean_second_value_hop2_drop'), pct=True)} | "
            f"{fmt(g.get('mean_second_key_hop2_drop'), pct=True)} | {fmt(g.get('mean_random_hop2_drop'), pct=True)} | "
            f"{fmt(g.get('mean_second_value_minus_first_value_hop2_drop'), pct=True)} | "
            f"{g.get('n_second_value_gt_first_value')}/{g.get('n_runs')} | {g.get('n_second_value_gt_random')}/{g.get('n_runs')} |"
        )
    lines.append('\n## Interpretation guardrails\n')
    lines.append('- A second-hop score is still an attention diagnostic. Treat it as mechanistic evidence only if second-hop-ranked ablation beats first-hop-ranked and matched random ablation.\n')
    lines.append('- If neither first-hop nor second-hop rankings beat random, the implementation is distributed or not captured by these scores.\n')
    lines.append('- If second-hop ranking works only in some arms, report schedule/seed-dependent circuit realisation rather than a universal two-head mechanism.\n')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate E1 two-hop score ablation outputs.')
    p.add_argument('--out-dir', default='runs/behavioral_replication_v0_9/twohop_decomposition')
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    rows = [summarize_one(p) for p in iter_files(out_dir)]
    rows.sort(key=lambda r: (str(r.get('arm')), int(r.get('seed') or -1), int(r.get('top_k') or -1)))
    groups = group_summary(rows)
    write_csv(out_dir / 'twohop_score_ablation_per_run.csv', rows)
    write_csv(out_dir / 'twohop_score_ablation_group_summary.csv', groups)
    write_report(out_dir / 'twohop_score_ablation_report.md', rows, groups)
    print(f'aggregated {len(rows)} two-hop score ablation files')
    print(f'wrote {out_dir / "twohop_score_ablation_per_run.csv"}')
    print(f'wrote {out_dir / "twohop_score_ablation_group_summary.csv"}')
    print(f'wrote {out_dir / "twohop_score_ablation_report.md"}')


if __name__ == '__main__':
    main()
