#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def fnum(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None


def fmt(v: Any, pct: bool = False) -> str:
    x = fnum(v)
    if x is None:
        return '—'
    return f'{100*x:.1f}%' if pct else f'{x:.4g}'


def find_col(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Write a conservative/safe interpretation report for reachability geometry outputs')
    ap.add_argument('--analysis-dir', required=True, help='Directory containing reachability geometry CSV outputs')
    ap.add_argument('--out', required=True, help='Markdown report path')
    args = ap.parse_args()
    root = Path(args.analysis_dir)
    out = Path(args.out)

    directional = read_csv(root / 'directional_summary.csv')
    own_path = read_csv(root / 'own_path_interpolation.csv')
    checkpoint = read_csv(root / 'checkpoint_path_alignment.csv')
    injection = read_csv(root / 'target_direction_injection.csv')

    lines: List[str] = []
    lines += [
        '# Conservative reachability-geometry report',
        '',
        'This report intentionally separates robust same-run/path evidence from exploratory cross-run geometry.',
        '',
        '## What is safe to claim',
        '',
        '1. **Same-run checkpoint trajectories and interpolation are useful.** They show whether a successful run crosses into HOP_2 behaviour sharply along its own path.',
        '2. **Transition-sharpness evidence is central.** The main mechanism claim should be delayed/thresholded compositional acquisition, not a universal parameter-space direction.',
        '3. **Cross-run Euclidean directions are exploratory.** Runs trained under different schedules can occupy different symmetry frames or basins, so raw weight-space cosines/injections across such runs should not be headline evidence.',
        '',
        '## What should not be claimed',
        '',
        '- Do not present injection into an arm with the same pre-intro trajectory as evidence of portability; it can nearly reconstruct the successful endpoint by arithmetic.',
        '- Do not claim that S2 moved in the “wrong” direction solely from a low cosine with a cosine-history successful direction; S2 may be in a different coordinate frame.',
        '- Do not infer a general geometry mechanism from a single seed. Use behavioural success fractions first.',
        '',
    ]

    if directional:
        lines += [
            '## Directional summary, exploratory only',
            '',
            '| Arm/run | Relative distance | Final HOP_2 acc | Cosine with target direction | Conservative reading |',
            '|---|---:|---:|---:|---|',
        ]
        for r in directional:
            name_col = find_col(r, ['run_label', 'arm', 'run_name', 'run_dir', 'source_run'])
            dist_col = find_col(r, ['relative_distance_intro_to_final', 'relative_distance_from_intro', 'relative_distance', 'rel_distance', 'distance_rel'])
            acc_col = find_col(r, ['final_hop2_acc', 'hop2_acc_final', 'final.acc.hop2', 'hop2_acc'])
            cos_col = find_col(r, ['cosine_delta_with_target_success_delta', 'cosine_with_target_direction', 'cosine_to_target', 'cosine', 'cos_with_target'])
            name = r.get(name_col, 'run') if name_col else 'run'
            reading = 'same-frame/path evidence if shared pre-intro; otherwise exploratory'
            if 's2' in str(name).lower() or 'constant' in str(name).lower():
                reading = 'cross-schedule comparison; frame-confounded, behavioural only'
            if 'rewarm' in str(name).lower() and 'reset' in str(name).lower():
                reading = 'successful reference path'
            lines.append(f"| {name} | {fmt(r.get(dist_col)) if dist_col else '—'} | {fmt(r.get(acc_col), pct=True) if acc_col else '—'} | {fmt(r.get(cos_col)) if cos_col else '—'} | {reading} |")
        lines.append('')

    if own_path:
        lines += [
            '## Same-run interpolation evidence',
            '',
            'Use this section as the main geometry visualization: if HOP_2 accuracy appears sharply along a successful run\'s own interpolation path, it supports a thresholded transition.',
            '',
        ]
        # Show a compact sample for any run with high final acc.
        by_run: Dict[str, List[Dict[str, Any]]] = {}
        for r in own_path:
            name = r.get('run_label') or r.get('run_dir') or r.get('run_name') or r.get('arm') or 'run'
            by_run.setdefault(str(name), []).append(r)
        shown = 0
        for name, rs in by_run.items():
            # prefer runs whose final/high alpha has high hop2
            acc_col = find_col(rs[0], ['hop2_acc', 'eval_hop2_acc', 'acc_hop2'])
            alpha_col = find_col(rs[0], ['alpha', 'interp_alpha'])
            if acc_col is None or alpha_col is None:
                continue
            max_acc = max([fnum(r.get(acc_col)) or 0.0 for r in rs])
            if max_acc < 0.8 and shown > 0:
                continue
            lines += [f'### {name}', '', '| alpha | HOP_2 acc | HOP_2 loss |', '|---:|---:|---:|']
            loss_col = find_col(rs[0], ['hop2_loss', 'eval_hop2_loss', 'loss_hop2'])
            for r in sorted(rs, key=lambda rr: fnum(rr.get(alpha_col)) or 0.0):
                lines.append(f"| {fmt(r.get(alpha_col))} | {fmt(r.get(acc_col), pct=True)} | {fmt(r.get(loss_col)) if loss_col else '—'} |")
            lines.append('')
            shown += 1
            if shown >= 2:
                break

    if checkpoint:
        lines += [
            '## Checkpoint-path evidence',
            '',
            'Checkpoint paths are safe when interpreted within the same run. They can show partial loss reduction followed by a sharp behavioural jump.',
            '',
        ]
        by_run: Dict[str, List[Dict[str, Any]]] = {}
        for r in checkpoint:
            name = r.get('run_label') or r.get('run_dir') or r.get('run_name') or r.get('arm') or 'run'
            by_run.setdefault(str(name), []).append(r)
        shown = 0
        for name, rs in by_run.items():
            acc_col = find_col(rs[0], ['hop2_acc', 'eval_hop2_acc', 'acc_hop2'])
            step_col = find_col(rs[0], ['step', 'checkpoint_step', 'checkpoint'])
            dist_col = find_col(rs[0], ['distance_from_intro_relative', 'relative_distance_from_intro', 'relative_distance', 'rel_distance'])
            if acc_col is None or step_col is None:
                continue
            max_acc = max([fnum(r.get(acc_col)) or 0.0 for r in rs])
            if max_acc < 0.8 and shown > 0:
                continue
            lines += [f'### {name}', '', '| step | rel. distance | HOP_2 acc | HOP_2 loss |', '|---:|---:|---:|---:|']
            loss_col = find_col(rs[0], ['hop2_loss', 'eval_hop2_loss', 'loss_hop2'])
            
            def _ck_sort(rr):
                raw = rr.get(step_col)
                low = str(raw).lower()
                if low == 'intro':
                    return -1.0
                if low == 'final':
                    return 1e18
                return fnum(raw) if fnum(raw) is not None else 0.0
            for r in sorted(rs, key=_ck_sort):
                raw_step = r.get(step_col)
                step_display = str(raw_step) if raw_step is not None else '—'
                lines.append(f"| {step_display} | {fmt(r.get(dist_col)) if dist_col else '—'} | {fmt(r.get(acc_col), pct=True)} | {fmt(r.get(loss_col)) if loss_col else '—'} |")
            lines.append('')
            shown += 1
            if shown >= 2:
                break

    if injection:
        lines += [
            '## Injection analysis status',
            '',
            'The injection file was detected, but this report does not treat cross-run injection as mechanistic evidence. It can be used as a debugging/sanity check only unless the compared checkpoints share the same pre-introduction trajectory and the interpretation is explicitly limited.',
            '',
        ]

    lines += [
        '## Recommended thesis placement',
        '',
        '- Main text: behavioural success fractions, specificity controls, thresholded transition, and top-2 ablation.',
        '- Mechanism section: same-run checkpoint/interpolation path as a case-study visualization of threshold crossing.',
        '- Appendix only: cross-run direction cosines/injections, with the frame-confounding caveat.',
        '',
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
