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


def stdev(vals: List[Any]) -> Optional[float]:
    xs = [numeric(v) for v in vals]
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    if run_dir.parent.name in ARM_NAMES:
        return run_dir.parent.name
    return run_dir.name


def load_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / 'config.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('matched_random_ablation_top*.json')):
        yield p


def topk_from_name(path: Path) -> Optional[int]:
    m = re.search(r'matched_random_ablation_top(\d+)\.json$', path.name)
    return int(m.group(1)) if m else None


def selection_str(sel: Any) -> str:
    if isinstance(sel, dict):
        parts = []
        for layer in sorted(sel, key=lambda x: int(x)):
            for h in sel[layer]:
                parts.append(f'L{layer}H{h}')
        return ','.join(parts)
    if isinstance(sel, list):
        return ','.join(f'L{l}H{h}' for l, h in sel)
    return str(sel)


def summarize_one(path: Path) -> Dict[str, Any]:
    run_dir = path.parent
    cfg = load_config(run_dir)
    train = cfg.get('train', {}) if cfg else {}
    sched = cfg.get('schedule', {}) if cfg else {}
    d = json.loads(path.read_text())
    s = d.get('summary', {})
    base = d.get('base', {})
    key = d.get('keyslot_ablated', {})
    draws = d.get('random_draws', [])
    row: Dict[str, Any] = {
        'run_dir': str(run_dir),
        'arm': derive_arm(run_dir),
        'condition': run_dir.name,
        'seed': train.get('seed'),
        'schedule': sched.get('kind'),
        'top_k': topk_from_name(path),
        'num_random': d.get('num_random'),
        'random_excludes_selected': d.get('random_excludes_selected'),
        'selected_heads': selection_str(d.get('selected_heads')),
        'base_hop1_acc': numeric(base.get('hop1_acc')),
        'base_hop2_acc': numeric(base.get('hop2_acc')),
        'keyslot_hop1_acc': numeric(key.get('hop1_acc')),
        'keyslot_hop2_acc': numeric(key.get('hop2_acc')),
    }
    # Flatten summary values.
    for k, v in s.items():
        row[k] = numeric(v) if numeric(v) is not None else v
    # Empirical p-value: probability a random draw drops at least as much as key-slot.
    for hop in [1, 2]:
        key_drop = numeric(row.get(f'keyslot_hop{hop}_drop'))
        rand_drops = []
        for dr in draws:
            ab = dr.get('ablated', {}) if isinstance(dr, dict) else {}
            ba = numeric(base.get(f'hop{hop}_acc'))
            aa = numeric(ab.get(f'hop{hop}_acc'))
            if ba is not None and aa is not None:
                rand_drops.append(ba - aa)
        if key_drop is not None and rand_drops:
            row[f'random_hop{hop}_drop_std'] = stdev(rand_drops)
            row[f'random_hop{hop}_drop_median'] = median(rand_drops)
            row[f'p_random_ge_keyslot_hop{hop}_drop'] = (sum(1 for x in rand_drops if x >= key_drop) + 1) / (len(rand_drops) + 1)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = sorted({k for r in rows for k in r})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get('arm')), int(r.get('top_k') or -1))].append(r)
    out: List[Dict[str, Any]] = []
    for (arm, top_k), rs in sorted(groups.items()):
        out.append({
            'arm': arm,
            'top_k': top_k,
            'n_runs': len(rs),
            'seeds': ' '.join(str(r.get('seed')) for r in rs),
            'mean_base_hop2_acc': mean([r.get('base_hop2_acc') for r in rs]),
            'mean_keyslot_hop2_acc': mean([r.get('keyslot_hop2_acc') for r in rs]),
            'mean_random_hop2_acc': mean([r.get('random_hop2_acc_mean') for r in rs]),
            'mean_keyslot_hop2_drop': mean([r.get('keyslot_hop2_drop') for r in rs]),
            'median_keyslot_hop2_drop': median([r.get('keyslot_hop2_drop') for r in rs]),
            'mean_random_hop2_drop': mean([r.get('random_hop2_drop_mean') for r in rs]),
            'median_random_hop2_drop': median([r.get('random_hop2_drop_mean') for r in rs]),
            'mean_keyslot_minus_random_hop2_drop': mean([r.get('keyslot_minus_random_hop2_drop') for r in rs]),
            'median_keyslot_minus_random_hop2_drop': median([r.get('keyslot_minus_random_hop2_drop') for r in rs]),
            'mean_p_random_ge_keyslot_hop2_drop': mean([r.get('p_random_ge_keyslot_hop2_drop') for r in rs]),
            'n_keyslot_drop_gt_random_mean': sum(1 for r in rs if (numeric(r.get('keyslot_minus_random_hop2_drop')) or -999) > 0),
            'n_keyslot_drop_gt_20pp': sum(1 for r in rs if (numeric(r.get('keyslot_hop2_drop')) or 0) >= 0.20),
            'n_keyslot_minus_random_gt_20pp': sum(1 for r in rs if (numeric(r.get('keyslot_minus_random_hop2_drop')) or 0) >= 0.20),
            'mean_keyslot_hop1_drop': mean([r.get('keyslot_hop1_drop') for r in rs]),
            'mean_random_hop1_drop': mean([r.get('random_hop1_drop_mean') for r in rs]),
        })
    return out


def fmt(v: Any, pct: bool = False) -> str:
    x = numeric(v)
    if x is None:
        return '—'
    return f'{100*x:.1f}%' if pct else f'{x:.4g}'


def write_report(path: Path, per_rows: List[Dict[str, Any]], group_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append('# Matched random-head ablation controls\n')
    lines.append('This report compares key-slot-selected mean ablation against repeated random matched-head mean ablations. By default, random draws exclude the selected key-slot heads. The primary estimand is `keyslot HOP_2 drop - mean random HOP_2 drop`.\n')
    lines.append('## Group summary\n')
    lines.append('| arm | top-k | n | base HOP_2 | key-slot HOP_2 | random HOP_2 | key-slot drop | random drop | excess key-slot drop | runs key > random |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for g in group_rows:
        lines.append(
            f"| {g.get('arm')} | {g.get('top_k')} | {g.get('n_runs')} | "
            f"{fmt(g.get('mean_base_hop2_acc'), pct=True)} | {fmt(g.get('mean_keyslot_hop2_acc'), pct=True)} | "
            f"{fmt(g.get('mean_random_hop2_acc'), pct=True)} | {fmt(g.get('mean_keyslot_hop2_drop'), pct=True)} | "
            f"{fmt(g.get('mean_random_hop2_drop'), pct=True)} | {fmt(g.get('mean_keyslot_minus_random_hop2_drop'), pct=True)} | "
            f"{g.get('n_keyslot_drop_gt_random_mean')}/{g.get('n_runs')} |"
        )
    lines.append('\n## Interpretation guardrails\n')
    lines.append('- Treat key-slot ablation as evidence for lookup-head involvement only if its drop exceeds matched random-head ablation.\n')
    lines.append('- Top-4 removes 25% of all heads in this model; matched random controls are required before using top-4 as circuit evidence.\n')
    lines.append('- A weak top-2 but stronger top-4 result suggests distributed/redundant lookup implementations, not a single clean head-level circuit.\n')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate matched random-head ablation control outputs.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out-dir', default='runs/behavioral_replication_v0_9/matched_random_ablation')
    args = p.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    rows = [summarize_one(p) for p in iter_files(root)]
    rows.sort(key=lambda r: (str(r.get('arm')), int(r.get('seed') or -1), int(r.get('top_k') or -1)))
    groups = group_summary(rows)
    write_csv(out_dir / 'matched_random_ablation_per_run.csv', rows)
    write_csv(out_dir / 'matched_random_ablation_group_summary.csv', groups)
    write_report(out_dir / 'matched_random_ablation_report.md', rows, groups)
    print(f'aggregated {len(rows)} matched-random ablation files')
    print(f'wrote {out_dir / "matched_random_ablation_per_run.csv"}')
    print(f'wrote {out_dir / "matched_random_ablation_group_summary.csv"}')
    print(f'wrote {out_dir / "matched_random_ablation_report.md"}')


if __name__ == '__main__':
    main()
