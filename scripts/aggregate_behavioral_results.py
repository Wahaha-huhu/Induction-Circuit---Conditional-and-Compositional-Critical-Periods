#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / 'config.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def tail_mean(rows: List[Dict[str, Any]], key: str, frac: float = 0.10) -> Optional[float]:
    vals = [numeric(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = max(1, round(len(vals) * frac))
    return float(sum(vals[-n:]) / n)


def first_ge(rows: List[Dict[str, Any]], key: str, threshold: float, consecutive: int = 1) -> Optional[int]:
    c = 0
    for r in rows:
        v = numeric(r.get(key))
        if v is not None and v >= threshold:
            c += 1
            if c >= consecutive:
                return int(r.get('step', 0))
        else:
            c = 0
    return None


def nearest_at_or_after(rows: List[Dict[str, Any]], step: Optional[int]) -> Optional[Dict[str, Any]]:
    if step is None:
        return rows[0] if rows else None
    for r in rows:
        if int(r.get('step', -1)) >= int(step):
            return r
    return rows[-1] if rows else None


def derive_arm(run_dir: Path) -> str:
    # Expected layout: runs/behavioral_replication/s3/<arm>/<condition_seed>/metrics.jsonl
    parts = run_dir.parts
    arm_names = {
        's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
        'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2',
        'query_b_s1', 'query_b_s2',
    }
    for p in reversed(parts):
        if p in arm_names:
            return p
    # Fallback: use parent dir if run leaf is condition_seed.
    if run_dir.parent.name.startswith(('s1_', 's2_', 'rewarm', 'fresh_', 'query_')):
        return run_dir.parent.name
    return run_dir.name


def run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('metrics.jsonl')):
        yield p.parent


def add_excess(rows: List[Dict[str, Any]], acc_key: str, floor_key: str, out_key: str) -> None:
    for r in rows:
        acc = numeric(r.get(acc_key))
        floor = numeric(r.get(floor_key))
        if acc is not None and floor is not None:
            r[out_key] = acc - floor


def summarize_run(run_dir: Path, hop2_success_threshold: float, simple_success_threshold: float) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / 'metrics.jsonl')
    add_excess(rows, 'hop2_acc', 'floor_hop2_acc', 'hop2_excess')
    add_excess(rows, 'fresh_hop1_acc', 'fresh_floor_hop1_acc', 'fresh_hop1_excess')
    add_excess(rows, 'queryB_hop1_acc', 'floor_queryB_hop1_acc', 'queryB_hop1_excess')
    cfg = load_config(run_dir)
    train = cfg.get('train', {}) if cfg else {}
    sched = cfg.get('schedule', {}) if cfg else {}
    intro_step = train.get('intro_step')
    post = [r for r in rows if intro_step is None or int(r.get('step', -1)) >= int(intro_step)]
    if not post:
        post = rows
    intro_row = nearest_at_or_after(rows, intro_step)
    last = rows[-1]
    arm = derive_arm(run_dir)
    is_fresh = 'fresh' in arm or run_dir.name.startswith('fresh_singlehop')

    t_hop2_95 = first_ge(post, 'hop2_acc', hop2_success_threshold)
    t_fresh_95 = first_ge(post, 'fresh_hop1_acc', simple_success_threshold)
    t_hop2_50 = first_ge(post, 'hop2_acc', 0.50)
    t_hop2_99 = first_ge(post, 'hop2_acc', 0.99)

    # Normalized width relative to observed peak excess, only meaningful when there is substantial excess.
    post_excess = [numeric(r.get('hop2_excess')) for r in post]
    post_excess = [v for v in post_excess if v is not None]
    peak_excess = max(post_excess) if post_excess else None
    norm_t10 = norm_t90 = None
    width_t90_t10 = None
    if peak_excess is not None and peak_excess >= 0.10:
        norm_t10 = first_ge(post, 'hop2_excess', 0.10 * peak_excess)
        norm_t90 = first_ge(post, 'hop2_excess', 0.90 * peak_excess)
        if norm_t10 is not None and norm_t90 is not None:
            width_t90_t10 = int(norm_t90) - int(norm_t10)

    tail_h2 = tail_mean(post, 'hop2_acc')
    tail_floor_h2 = tail_mean(post, 'floor_hop2_acc')
    tail_excess_h2 = tail_mean(post, 'hop2_excess')
    tail_fresh = tail_mean(post, 'fresh_hop1_acc')
    tail_fresh_excess = tail_mean(post, 'fresh_hop1_excess')
    intro_h2 = numeric(intro_row.get('hop2_acc')) if intro_row else None
    intro_fresh = numeric(intro_row.get('fresh_hop1_acc')) if intro_row else None

    hop2_success = bool(t_hop2_95 is not None or (tail_h2 is not None and tail_h2 >= hop2_success_threshold))
    fresh_success = bool(t_fresh_95 is not None or (tail_fresh is not None and tail_fresh >= simple_success_threshold))
    fresh_not_zeroshot = None
    if is_fresh and intro_fresh is not None:
        fresh_not_zeroshot = bool(intro_fresh <= 0.10)

    return {
        'run_dir': str(run_dir),
        'arm': arm,
        'condition': run_dir.name,
        'seed': train.get('seed'),
        'schedule': sched.get('kind'),
        'intro_step': intro_step,
        'max_steps': train.get('max_steps'),
        't_schedule': sched.get('t_schedule'),
        'rewarm_lr': sched.get('rewarm_lr'),
        'final_step': int(last.get('step', -1)),
        'is_fresh_control': is_fresh,
        'intro_hop1_acc': numeric(intro_row.get('hop1_acc')) if intro_row else None,
        'intro_hop2_acc': intro_h2,
        'intro_fresh_hop1_acc': intro_fresh,
        'tail_hop1_acc': tail_mean(post, 'hop1_acc'),
        'tail_hop2_acc': tail_h2,
        'tail_floor_hop2_acc': tail_floor_h2,
        'tail_hop2_excess': tail_excess_h2,
        'peak_hop2_excess': peak_excess,
        'tail_hop2_loss': tail_mean(post, 'hop2_loss'),
        'tail_fresh_hop1_acc': tail_fresh,
        'tail_fresh_hop1_excess': tail_fresh_excess,
        'tail_base_hop1_acc': tail_mean(post, 'base_hop1_acc'),
        'tail_update_to_weight_ratio': tail_mean(post, 'update_to_weight_ratio'),
        't_hop2_acc_ge_0.50': t_hop2_50,
        f't_hop2_acc_ge_{hop2_success_threshold:.2f}': t_hop2_95,
        't_hop2_acc_ge_0.99': t_hop2_99,
        f't_fresh_hop1_acc_ge_{simple_success_threshold:.2f}': t_fresh_95,
        'norm_t10_peak_excess': norm_t10,
        'norm_t90_peak_excess': norm_t90,
        'norm_width_t90_minus_t10': width_t90_t10,
        'hop2_success': hop2_success,
        'fresh_success': fresh_success,
        'fresh_not_zeroshot': fresh_not_zeroshot,
    }


def mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def stdev(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def quantile(vals: List[float], q: float) -> Optional[float]:
    vals = sorted([v for v in vals if v is not None and math.isfinite(float(v))])
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(str(r.get('arm')), []).append(r)
    out: List[Dict[str, Any]] = []
    for arm, rs in sorted(by_arm.items()):
        seeds = sorted({str(r.get('seed')) for r in rs})
        hop2_successes = [bool(r.get('hop2_success')) for r in rs if not r.get('is_fresh_control')]
        fresh_successes = [bool(r.get('fresh_success')) for r in rs if r.get('is_fresh_control')]
        fresh_not_zeroshot = [bool(r.get('fresh_not_zeroshot')) for r in rs if r.get('fresh_not_zeroshot') is not None]
        h2_excess = [numeric(r.get('tail_hop2_excess')) for r in rs]
        h2_tail = [numeric(r.get('tail_hop2_acc')) for r in rs]
        fresh_tail = [numeric(r.get('tail_fresh_hop1_acc')) for r in rs]
        widths = [numeric(r.get('norm_width_t90_minus_t10')) for r in rs]
        trans95_keys = [k for k in rs[0].keys() if k.startswith('t_hop2_acc_ge_') and k.endswith('0.95')]
        t95_key = trans95_keys[0] if trans95_keys else 't_hop2_acc_ge_0.95'
        t95s = [numeric(r.get(t95_key)) for r in rs]
        out.append({
            'arm': arm,
            'n_runs': len(rs),
            'seeds': ' '.join(seeds),
            'hop2_success_count': sum(hop2_successes),
            'hop2_success_fraction': (sum(hop2_successes) / len(hop2_successes)) if hop2_successes else None,
            'fresh_success_count': sum(fresh_successes),
            'fresh_success_fraction': (sum(fresh_successes) / len(fresh_successes)) if fresh_successes else None,
            'fresh_not_zeroshot_count': sum(fresh_not_zeroshot),
            'fresh_not_zeroshot_fraction': (sum(fresh_not_zeroshot) / len(fresh_not_zeroshot)) if fresh_not_zeroshot else None,
            'mean_tail_hop2_acc': mean([v for v in h2_tail if v is not None]),
            'std_tail_hop2_acc': stdev([v for v in h2_tail if v is not None]),
            'mean_tail_hop2_excess': mean([v for v in h2_excess if v is not None]),
            'std_tail_hop2_excess': stdev([v for v in h2_excess if v is not None]),
            'p25_tail_hop2_excess': quantile([v for v in h2_excess if v is not None], 0.25),
            'median_tail_hop2_excess': quantile([v for v in h2_excess if v is not None], 0.50),
            'p75_tail_hop2_excess': quantile([v for v in h2_excess if v is not None], 0.75),
            'mean_tail_fresh_hop1_acc': mean([v for v in fresh_tail if v is not None]),
            'median_transition_step_hop2_95': quantile([v for v in t95s if v is not None], 0.50),
            'median_transition_width_t90_t10': quantile([v for v in widths if v is not None], 0.50),
        })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, groups: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Behavioural replication aggregate report',
        '',
        'This report prioritizes behavioural success fractions and transition statistics over single-seed geometry.',
        '',
        '## Group summary',
        '',
        '| Arm | Runs | Seeds | HOP_2 success | Mean tail HOP_2 | Mean HOP_2 excess | Fresh success | Median HOP_2 t95 |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for g in groups:
        def fmt(v: Any, pct: bool = False) -> str:
            if v is None or v == '':
                return '—'
            try:
                x = float(v)
                return f'{100*x:.1f}%' if pct else f'{x:.4g}'
            except Exception:
                return str(v)
        lines.append(
            f"| {g.get('arm')} | {g.get('n_runs')} | {g.get('seeds')} | "
            f"{fmt(g.get('hop2_success_fraction'), pct=True)} | "
            f"{fmt(g.get('mean_tail_hop2_acc'), pct=True)} | "
            f"{fmt(g.get('mean_tail_hop2_excess'), pct=True)} | "
            f"{fmt(g.get('fresh_success_fraction'), pct=True)} | "
            f"{fmt(g.get('median_transition_step_hop2_95'))} |"
        )
    lines += [
        '',
        '## Recommended interpretation rule',
        '',
        '- Treat `S1 late`/`S1 plateau` failures as robust only if the HOP_2 success fraction remains low across seeds.',
        '- Treat `S2 constant` or `rewarm` as reliable rescues only if their success fractions are high across seeds.',
        '- Use `fresh_hop1_s1` to test selectivity: it should start near floor and then reach high accuracy late.',
        '- Use transition width and loss/accuracy trajectories to support thresholded delayed generalization.',
        '- Do not infer cross-run weight-space direction from these behavioural aggregates.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Aggregate behavioural replication runs into per-run and group success summaries')
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out-dir', default=None, help='Output directory; defaults to <runs-dir>/aggregate')
    ap.add_argument('--hop2-success-threshold', type=float, default=0.95)
    ap.add_argument('--simple-success-threshold', type=float, default=0.95)
    args = ap.parse_args()
    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'aggregate'
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs(root):
        try:
            rows.append(summarize_run(rd, args.hop2_success_threshold, args.simple_success_threshold))
        except Exception as exc:  # noqa: BLE001
            print(f'WARNING: skipping {rd}: {exc}', file=sys.stderr)
    if not rows:
        raise SystemExit(f'no metrics.jsonl found under {root}')
    groups = group_summary(rows)
    write_csv(out_dir / 'behavioral_per_run.csv', rows)
    write_csv(out_dir / 'behavioral_group_summary.csv', groups)
    write_markdown(out_dir / 'behavioral_report.md', groups)
    print(f'wrote {out_dir / "behavioral_per_run.csv"}')
    print(f'wrote {out_dir / "behavioral_group_summary.csv"}')
    print(f'wrote {out_dir / "behavioral_report.md"}')


if __name__ == '__main__':
    main()
