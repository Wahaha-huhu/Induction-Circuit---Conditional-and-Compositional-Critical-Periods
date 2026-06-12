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
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def mean(vals: List[float]) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate v0.18 reorganisation/candidate tracking CSVs.')
    p.add_argument('--tracking-dir', default='runs/checkpointed_geometry_followups/reorganisation_candidate_v0_18')
    p.add_argument('--out-per-checkpoint', default=None)
    p.add_argument('--out-summary', default=None)
    p.add_argument('--out-report', default=None)
    args = p.parse_args()
    root = Path(args.tracking_dir)
    files = sorted(root.rglob('*_reorganisation_candidate.csv')) + sorted(root.rglob('reorganisation_candidate_tracking.csv'))
    # de-duplicate
    seen = set()
    files2 = []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            files2.append(f)
    rows: List[Dict[str, Any]] = []
    for f in files2:
        for r in read_csv(f):
            r['source_csv'] = str(f)
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f'No v0.18 tracking CSVs found under {root}')
    out_per = Path(args.out_per_checkpoint) if args.out_per_checkpoint else root / 'reorganisation_candidate_per_checkpoint.csv'
    out_sum = Path(args.out_summary) if args.out_summary else root / 'reorganisation_candidate_summary.csv'
    out_report = Path(args.out_report) if args.out_report else root / 'reorganisation_candidate_report.md'
    out_per.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out_per.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary by arm/seed: intro, minimum, final, and crossing status.
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r.get('arm',''), r.get('seed','')), []).append(r)
    summary: List[Dict[str, Any]] = []
    for (arm, seed), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: num(r.get('step')) if num(r.get('step')) is not None else -1)
        first = rs[0]
        final = rs[-1]
        hop2s = [(num(r.get('step')), num(r.get('hop2_acc'))) for r in rs]
        crossing = next((s for s, a in hop2s if s is not None and a is not None and a >= 0.95), None)
        row = {
            'arm': arm,
            'seed': seed,
            'n_checkpoints': len(rs),
            'intro_step': first.get('step'),
            'final_step': final.get('step'),
            'hop2_ge_95_step': crossing,
            'intro_hop2_acc': first.get('hop2_acc'),
            'final_hop2_acc': final.get('hop2_acc'),
            'intro_L3_B_acc': first.get('lens_L3_B_acc'),
            'intro_L3_C_acc': first.get('lens_L3_C_acc'),
            'final_L3_B_acc': final.get('lens_L3_B_acc'),
            'final_L3_C_acc': final.get('lens_L3_C_acc'),
            'intro_candidate_mass': first.get('prob_in_context_content_mass'),
            'final_candidate_mass': final.get('prob_in_context_content_mass'),
            'intro_top1_in_context': first.get('top1_is_in_context_content'),
            'final_top1_in_context': final.get('top1_is_in_context_content'),
            'intro_second_key_max': first.get('second_key_max'),
            'final_second_key_max': final.get('second_key_max'),
            'intro_second_value_max': first.get('second_value_max'),
            'final_second_value_max': final.get('second_value_max'),
        }
        # Stage-one diagnostic: max candidate mass before HOP2 crosses 50%.
        pre = []
        for r in rs:
            a = num(r.get('hop2_acc'))
            if a is None or a < 0.50:
                pre.append(r)
        row['max_pretransition_candidate_mass'] = max([num(r.get('prob_in_context_content_mass')) or 0 for r in pre], default=None)
        row['max_pretransition_top1_in_context'] = max([num(r.get('top1_is_in_context_content')) or 0 for r in pre], default=None)
        row['min_pretransition_hop2_loss'] = min([num(r.get('hop2_loss')) for r in pre if num(r.get('hop2_loss')) is not None], default=None)
        summary.append(row)
    sum_fields: List[str] = []
    for r in summary:
        for k in r:
            if k not in sum_fields:
                sum_fields.append(k)
    with out_sum.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary)

    report = []
    report.append('# v0.18 reorganisation and candidate-set tracking report\n')
    report.append(f'Loaded {len(rows)} checkpoints from {len(files2)} CSV files.\n')
    report.append('## Per-run summary\n')
    report.append('| arm | seed | final HOP2 | HOP2 >=95 step | intro L3 B | final L3 B | final L3 C | max pretransition candidate mass |')
    report.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in summary:
        def fmt(x):
            v = num(x)
            return 'NA' if v is None else f'{v:.3f}'
        report.append(f"| {r['arm']} | {r['seed']} | {fmt(r.get('final_hop2_acc'))} | {r.get('hop2_ge_95_step') or 'never'} | {fmt(r.get('intro_L3_B_acc'))} | {fmt(r.get('final_L3_B_acc'))} | {fmt(r.get('final_L3_C_acc'))} | {fmt(r.get('max_pretransition_candidate_mass'))} |")
    report.append('\n## Interpretation guide\n')
    report.append('- If intro B is high under a HOP_2 prompt, the model initially treats HOP_2 like HOP_1.')
    report.append('- If B/C disappear after HOP_2 training starts but only successful runs rebuild C, the barrier is incomplete reorganisation into HOP_2 mode.')
    report.append('- If candidate mass rises before exact accuracy, the loss-before-accuracy gap is consistent with learning output format/candidate class before learning the composed route.')
    out_report.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'wrote {out_per}')
    print(f'wrote {out_sum}')
    print(f'wrote {out_report}')


if __name__ == '__main__':
    main()
