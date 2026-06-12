#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None


def iter_rows(root: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(root.rglob('first_binding_patching.csv')):
        for r in read_csv(p):
            r['csv_path'] = str(p)
            yield r


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')


def groupby(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[tuple, List[Dict[str, Any]]]:
    out: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k, '') for k in keys)].append(r)
    return out


def summarize_group(rows: List[Dict[str, Any]], group_fields: Dict[str, Any]) -> Dict[str, Any]:
    metrics = [
        'clean_acc_mean', 'corrupt_acc_mean', 'patched_clean_answer_acc_mean', 'patched_corrupt_answer_acc_mean',
        'clean_logit_diff_mean', 'corrupt_logit_diff_mean', 'patched_logit_diff_mean', 'restoration_mean',
    ]
    out = dict(group_fields)
    out['n'] = len(rows)
    for m in metrics:
        vals = [to_float(r.get(m)) for r in rows]
        vals = [v for v in vals if v is not None]
        out[m] = mean(vals) if vals else float('nan')
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate first-binding corruption/intermediate-routing patching results.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'first_binding_patching'
    rows = list(iter_rows(root))
    if not rows:
        raise SystemExit(f'No first_binding_patching.csv files found under {root}')

    write_csv(out_dir / 'first_binding_patching_all_sites.csv', rows)

    # Best residual site per run.
    best_rows: List[Dict[str, Any]] = []
    for key, rs in groupby([r for r in rows if r.get('patch_kind') == 'residual'], ['run_dir']).items():
        best = max(rs, key=lambda r: to_float(r.get('restoration_mean')) if to_float(r.get('restoration_mean')) is not None else -999)
        best_rows.append(best)
    write_csv(out_dir / 'first_binding_patching_best_residual_per_run.csv', best_rows)

    # Best head per run, if head patching was run.
    best_head_rows: List[Dict[str, Any]] = []
    head_rows = [r for r in rows if r.get('patch_kind') == 'head']
    for key, rs in groupby(head_rows, ['run_dir']).items():
        best = max(rs, key=lambda r: to_float(r.get('restoration_mean')) if to_float(r.get('restoration_mean')) is not None else -999)
        best_head_rows.append(best)
    if best_head_rows:
        write_csv(out_dir / 'first_binding_patching_best_head_per_run.csv', best_head_rows)

    group_summary: List[Dict[str, Any]] = []
    for key, rs in groupby(rows, ['arm', 'patch_kind', 'layer']).items():
        arm, patch_kind, layer = key
        group_summary.append(summarize_group(rs, {'arm': arm, 'patch_kind': patch_kind, 'layer': layer}))
    write_csv(out_dir / 'first_binding_patching_group_by_layer.csv', group_summary)

    best_group: List[Dict[str, Any]] = []
    for key, rs in groupby(best_rows, ['arm']).items():
        best_group.append(summarize_group(rs, {'arm': key[0], 'site': 'best_residual_per_run'}))
    if best_head_rows:
        for key, rs in groupby(best_head_rows, ['arm']).items():
            best_group.append(summarize_group(rs, {'arm': key[0], 'site': 'best_head_per_run'}))
    write_csv(out_dir / 'first_binding_patching_best_group_summary.csv', best_group)

    report = out_dir / 'first_binding_patching_report.md'
    lines = [
        '# First-binding corruption / intermediate-routing patching report',
        '',
        'Clean and corrupt inputs differ only in the first binding value A->B versus A->B\'. Both downstream bindings B->C and B\'->C\' are present in both contexts. Patching clean query-position activations into the corrupt run tests whether the model can route through the clean intermediate to recover the clean answer.',
        '',
        '## Best residual patch per arm',
        '',
        '| Arm | n | mean restoration | patched clean-answer acc | corrupt-answer acc |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in sorted(best_group, key=lambda x: (x.get('site',''), x.get('arm',''))):
        if r.get('site') != 'best_residual_per_run':
            continue
        lines.append(f"| {r.get('arm')} | {r.get('n')} | {float(r.get('restoration_mean')):.3f} | {float(r.get('patched_clean_answer_acc_mean')):.3f} | {float(r.get('patched_corrupt_answer_acc_mean')):.3f} |")
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    manifest = {
        'runs_dir': str(root),
        'n_rows': len(rows),
        'n_best_residual': len(best_rows),
        'n_best_head': len(best_head_rows),
        'outputs': [
            'first_binding_patching_all_sites.csv',
            'first_binding_patching_best_residual_per_run.csv',
            'first_binding_patching_best_group_summary.csv',
            'first_binding_patching_group_by_layer.csv',
            'first_binding_patching_report.md',
        ],
    }
    (out_dir / 'first_binding_patching_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'wrote outputs to {out_dir}')


if __name__ == '__main__':
    main()
