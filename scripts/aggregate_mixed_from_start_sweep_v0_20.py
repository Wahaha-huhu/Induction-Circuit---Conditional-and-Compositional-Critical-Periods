#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / 'config.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.10) -> Optional[float]:
    vals = [num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    return float(sum(vals[-n:]) / n)


def first_ge(rows: List[Dict[str, Any]], key: str, thr: float, consecutive: int = 1) -> Optional[int]:
    c = 0
    for r in rows:
        v = num(r.get(key))
        if v is not None and v >= thr:
            c += 1
            if c >= consecutive:
                return int(r.get('step', 0))
        else:
            c = 0
    return None


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else None


def sd(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def infer_sweep_arm_param(root: Path, run_dir: Path) -> Tuple[str, str, str]:
    # Expected path: root/sweep_type/arm/param/c0_seedX
    try:
        rel = run_dir.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = run_dir.parts
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    return 'unknown', 'unknown', 'unknown'


def parse_param(param: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if param.startswith('pmulti_'):
        out['p_multi_grid'] = float(param.split('_', 1)[1].replace('p', '.'))
    elif param.startswith('p0p5_steps_'):
        out['p_multi_grid'] = 0.5
        out['long_steps_grid'] = int(param.rsplit('_', 1)[1])
    return out


def add_excess(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        h1 = num(r.get('hop1_acc'))
        f1 = num(r.get('floor_hop1_acc'))
        h2 = num(r.get('hop2_acc'))
        f2 = num(r.get('floor_hop2_acc'))
        if h1 is not None and f1 is not None:
            r['hop1_excess'] = h1 - f1
        if h2 is not None and f2 is not None:
            r['hop2_excess'] = h2 - f2


def summarize_run(root: Path, run_dir: Path, success_thr: float) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / 'metrics.jsonl')
    add_excess(rows)
    cfg = read_config(run_dir)
    train = cfg.get('train', {}) if cfg else {}
    sched = cfg.get('schedule', {}) if cfg else {}
    optim = cfg.get('optim', {}) if cfg else {}
    data = cfg.get('data', {}) if cfg else {}
    model = cfg.get('model', {}) if cfg else {}
    sweep_type, arm, param = infer_sweep_arm_param(root, run_dir)
    meta = parse_param(param)
    last = rows[-1] if rows else {}

    tail_h1 = tail_mean(rows, 'hop1_acc')
    tail_h2 = tail_mean(rows, 'hop2_acc')
    row = {
        'run_dir': str(run_dir),
        'sweep_type': sweep_type,
        'arm': arm,
        'param': param,
        'seed': train.get('seed'),
        'schedule': sched.get('kind'),
        'max_steps': train.get('max_steps'),
        'final_step': int(last.get('step', -1)) if last else None,
        'p_multi': train.get('p_multi_frozen'),
        'p_multi_before_intro': train.get('p_multi_before_intro'),
        'intro_step': train.get('intro_step'),
        't_schedule': sched.get('t_schedule'),
        'peak_lr': optim.get('peak_lr'),
        'final_lr': optim.get('final_lr'),
        'v_content': data.get('v_content'),
        'chain_length': data.get('chain_length'),
        'd_model': model.get('d_model'),
        'tail_hop1_acc': tail_h1,
        'tail_hop2_acc': tail_h2,
        'tail_hop1_excess': tail_mean(rows, 'hop1_excess'),
        'tail_hop2_excess': tail_mean(rows, 'hop2_excess'),
        'tail_hop1_loss': tail_mean(rows, 'hop1_loss'),
        'tail_hop2_loss': tail_mean(rows, 'hop2_loss'),
        'final_hop1_acc': num(last.get('hop1_acc')) if last else None,
        'final_hop2_acc': num(last.get('hop2_acc')) if last else None,
        'final_hop1_loss': num(last.get('hop1_loss')) if last else None,
        'final_hop2_loss': num(last.get('hop2_loss')) if last else None,
        't_hop1_acc_ge_0.50': first_ge(rows, 'hop1_acc', 0.50),
        f't_hop1_acc_ge_{success_thr:.2f}': first_ge(rows, 'hop1_acc', success_thr),
        't_hop2_acc_ge_0.50': first_ge(rows, 'hop2_acc', 0.50),
        f't_hop2_acc_ge_{success_thr:.2f}': first_ge(rows, 'hop2_acc', success_thr),
        'hop1_success': bool(tail_h1 is not None and tail_h1 >= success_thr),
        'hop2_success': bool(tail_h2 is not None and tail_h2 >= success_thr),
        'both_success': bool(tail_h1 is not None and tail_h2 is not None and tail_h1 >= success_thr and tail_h2 >= success_thr),
        'success_threshold': success_thr,
    }
    row.update(meta)
    return row


def group_rows(per_run: List[Dict[str, Any]], success_thr: float) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    specs = {
        'pmulti_sweep_from_start': ['sweep_type', 'arm', 'p_multi_grid'],
        'long_pmulti_0p5': ['sweep_type', 'arm', 'long_steps_grid'],
    }
    for sweep_type, fields in specs.items():
        rows = [r for r in per_run if r.get('sweep_type') == sweep_type]
        out: List[Dict[str, Any]] = []
        keys = sorted({tuple(r.get(f) for f in fields) for r in rows})
        for key in keys:
            rs = [r for r in rows if tuple(r.get(f) for f in fields) == key]
            if not rs:
                continue
            g = {f: v for f, v in zip(fields, key)}
            g.update({
                'n_runs': len(rs),
                'n_hop1_success': sum(1 for r in rs if r.get('hop1_success') is True),
                'n_hop2_success': sum(1 for r in rs if r.get('hop2_success') is True),
                'n_both_success': sum(1 for r in rs if r.get('both_success') is True),
                'hop1_success_fraction': sum(1 for r in rs if r.get('hop1_success') is True) / len(rs),
                'hop2_success_fraction': sum(1 for r in rs if r.get('hop2_success') is True) / len(rs),
                'both_success_fraction': sum(1 for r in rs if r.get('both_success') is True) / len(rs),
                'mean_tail_hop1_acc': mean(r.get('tail_hop1_acc') for r in rs),
                'sd_tail_hop1_acc': sd(r.get('tail_hop1_acc') for r in rs),
                'mean_tail_hop2_acc': mean(r.get('tail_hop2_acc') for r in rs),
                'sd_tail_hop2_acc': sd(r.get('tail_hop2_acc') for r in rs),
                'mean_tail_hop1_excess': mean(r.get('tail_hop1_excess') for r in rs),
                'mean_tail_hop2_excess': mean(r.get('tail_hop2_excess') for r in rs),
                'mean_tail_hop1_loss': mean(r.get('tail_hop1_loss') for r in rs),
                'mean_tail_hop2_loss': mean(r.get('tail_hop2_loss') for r in rs),
                'mean_max_steps': mean(r.get('max_steps') for r in rs),
                'mean_p_multi': mean(r.get('p_multi') for r in rs),
                'success_threshold': success_thr,
            })
            out.append(g)
        grouped[sweep_type] = out
    return grouped


def write_report(path: Path, grouped: Dict[str, List[Dict[str, Any]]], success_thr: float) -> None:
    lines = [
        '# v0.20 mixed-from-start feasibility sweep report',
        '',
        f'Success threshold: tail accuracy >= {success_thr:.2f}.',
        '',
        'This sweep tests whether the model can discover HOP_1 and HOP_2 simultaneously when the HOP_1/HOP_2 mixture is present from the first training step. This is distinct from the staged critical-period setting, where HOP_1 is first learned before HOP_2 is introduced.',
        '',
    ]
    for name, rows in grouped.items():
        lines.append(f'## {name}')
        lines.append('')
        if not rows:
            lines.append('No completed runs found.')
            lines.append('')
            continue
        if name == 'pmulti_sweep_from_start':
            cols = ['arm', 'p_multi_grid', 'n_runs', 'n_both_success', 'both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']
        else:
            cols = ['arm', 'long_steps_grid', 'n_runs', 'n_both_success', 'both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    vals.append(f'{v:.4f}')
                else:
                    vals.append(str(v))
            lines.append('| ' + ' | '.join(vals) + ' |')
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs/mixed_from_start_sweeps_v0_20')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--success-thr', type=float, default=0.95)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'summary'
    run_dirs = sorted(p for p in root.glob('*/*/*/c0_seed*') if (p / 'metrics.jsonl').exists())
    per = [summarize_run(root, p, args.success_thr) for p in run_dirs]
    grouped = group_rows(per, args.success_thr)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'mixed_from_start_v0_20_per_run.csv', per)
    write_csv(out_dir / 'pmulti_from_start_summary.csv', grouped.get('pmulti_sweep_from_start', []))
    write_csv(out_dir / 'long_pmulti_0p5_summary.csv', grouped.get('long_pmulti_0p5', []))
    write_report(out_dir / 'mixed_from_start_v0_20_report.md', grouped, args.success_thr)
    print(f'wrote summary to {out_dir}')


if __name__ == '__main__':
    main()
