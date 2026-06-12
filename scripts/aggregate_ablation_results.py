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


def mean(vals: List[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def stdev(vals: List[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.stdev(vals) if len(vals) >= 2 else None


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


def iter_ablation_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('ablation_top*.json')):
        yield p


def topk_from_name(path: Path) -> Optional[int]:
    m = re.search(r'ablation_top(\d+)\.json$', path.name)
    return int(m.group(1)) if m else None


def selected_heads_string(obj: Any) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    parts = []
    for layer in sorted(obj, key=lambda x: int(x)):
        heads = obj[layer]
        if isinstance(heads, list):
            for h in heads:
                parts.append(f'L{layer}H{h}')
        else:
            parts.append(f'L{layer}H{heads}')
    return ','.join(parts)


def summarize_one(path: Path) -> Dict[str, Any]:
    run_dir = path.parent
    cfg = load_config(run_dir)
    train = cfg.get('train', {}) if cfg else {}
    sched = cfg.get('schedule', {}) if cfg else {}
    d = json.loads(path.read_text())
    base = d.get('base', {})
    ab = d.get('ablated', {})
    top_k = topk_from_name(path)
    row: Dict[str, Any] = {
        'run_dir': str(run_dir),
        'arm': derive_arm(run_dir),
        'condition': run_dir.name,
        'seed': train.get('seed'),
        'schedule': sched.get('kind'),
        'top_k': top_k,
        'selected_heads': selected_heads_string(d.get('selected_heads')),
        'base_hop1_acc': numeric(base.get('hop1_acc')),
        'base_hop2_acc': numeric(base.get('hop2_acc')),
        'base_hop1_loss': numeric(base.get('hop1_loss')),
        'base_hop2_loss': numeric(base.get('hop2_loss')),
        'ablated_hop1_acc': numeric(ab.get('hop1_acc')),
        'ablated_hop2_acc': numeric(ab.get('hop2_acc')),
        'ablated_hop1_loss': numeric(ab.get('hop1_loss')),
        'ablated_hop2_loss': numeric(ab.get('hop2_loss')),
    }
    for hop in [1, 2]:
        b = row.get(f'base_hop{hop}_acc')
        a = row.get(f'ablated_hop{hop}_acc')
        if b is not None and a is not None:
            row[f'hop{hop}_acc_drop'] = float(b) - float(a)
            row[f'hop{hop}_relative_remaining'] = float(a) / max(float(b), 1e-12)
            row[f'hop{hop}_collapse_fraction'] = 1.0 - row[f'hop{hop}_relative_remaining']
        bl = row.get(f'base_hop{hop}_loss')
        al = row.get(f'ablated_hop{hop}_loss')
        if bl is not None and al is not None:
            row[f'hop{hop}_loss_increase'] = float(al) - float(bl)
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get('arm')), int(r.get('top_k') or -1))].append(r)
    out = []
    for (arm, top_k), rs in sorted(groups.items()):
        out.append({
            'arm': arm,
            'top_k': top_k,
            'n_runs': len(rs),
            'seeds': ' '.join(str(r.get('seed')) for r in rs),
            'mean_base_hop1_acc': mean([r.get('base_hop1_acc') for r in rs]),
            'mean_base_hop2_acc': mean([r.get('base_hop2_acc') for r in rs]),
            'mean_ablated_hop1_acc': mean([r.get('ablated_hop1_acc') for r in rs]),
            'mean_ablated_hop2_acc': mean([r.get('ablated_hop2_acc') for r in rs]),
            'mean_hop1_acc_drop': mean([r.get('hop1_acc_drop') for r in rs]),
            'mean_hop2_acc_drop': mean([r.get('hop2_acc_drop') for r in rs]),
            'std_hop2_acc_drop': stdev([r.get('hop2_acc_drop') for r in rs]),
            'mean_hop1_collapse_fraction': mean([r.get('hop1_collapse_fraction') for r in rs]),
            'mean_hop2_collapse_fraction': mean([r.get('hop2_collapse_fraction') for r in rs]),
            'mean_hop2_loss_increase': mean([r.get('hop2_loss_increase') for r in rs]),
        })
    return out


def fmt(v: Any, pct: bool = False) -> str:
    x = numeric(v)
    if x is None:
        return '—'
    return f'{100*x:.1f}%' if pct else f'{x:.4g}'


def write_report(path: Path, per_rows: List[Dict[str, Any]], group_rows: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append('# Key-slot head ablation sweep\n')
    lines.append('This report aggregates mean-ablation evaluations on successful HOP_2 models. Heads are selected by the held-out key-slot lookup score, then each selected head output is replaced by its global same-head mean during evaluation.\n')
    lines.append('## Group summary\n')
    lines.append('| arm | top-k | n | base HOP_2 | ablated HOP_2 | HOP_2 drop | HOP_2 collapse | HOP_1 drop |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for g in group_rows:
        lines.append(
            f"| {g.get('arm')} | {g.get('top_k')} | {g.get('n_runs')} | "
            f"{fmt(g.get('mean_base_hop2_acc'), pct=True)} | {fmt(g.get('mean_ablated_hop2_acc'), pct=True)} | "
            f"{fmt(g.get('mean_hop2_acc_drop'), pct=True)} | {fmt(g.get('mean_hop2_collapse_fraction'), pct=True)} | "
            f"{fmt(g.get('mean_hop1_acc_drop'), pct=True)} |"
        )
    lines.append('\n## Interpretation guidance\n')
    lines.append('- Strong evidence for lookup-head reuse: top-2 or top-4 ablation sharply reduces both HOP_1 and HOP_2 accuracy in successful HOP_2 models.\n')
    lines.append('- If top-1 is partial but top-2/top-4 collapses, present top-2 as the primary intervention and top-1/top-4 as robustness.\n')
    lines.append('- This ablation supports the compositional-reuse claim only for runs where HOP_2 was behaviourally acquired. It should not be interpreted as a schedule mechanism by itself.\n')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate ablation_topK.json files into per-run and group summaries.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out-dir', default='runs/behavioral_replication_v0_9/ablation')
    args = p.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    rows = [summarize_one(p) for p in iter_ablation_files(root)]
    rows.sort(key=lambda r: (str(r.get('arm')), int(r.get('seed') or -1), int(r.get('top_k') or -1)))
    groups = group_summary(rows)
    write_csv(out_dir / 'ablation_per_run.csv', rows)
    write_csv(out_dir / 'ablation_group_summary.csv', groups)
    write_report(out_dir / 'ablation_report.md', rows, groups)
    print(f'aggregated {len(rows)} ablation result files')
    print(f'wrote {out_dir / "ablation_per_run.csv"}')
    print(f'wrote {out_dir / "ablation_group_summary.csv"}')
    print(f'wrote {out_dir / "ablation_report.md"}')


if __name__ == '__main__':
    main()
