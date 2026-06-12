#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ARM_NAMES = {
    's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
    'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2',
    'query_b_s1', 'query_b_s2',
}
DEFAULT_SUCCESS_ARMS = 's1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late'
DEFAULT_FAILED_ARMS = 's1_late_original,s1_plateau_late'


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


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


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    return run_dir.parent.name if run_dir.parent.name in ARM_NAMES else run_dir.name


def load_seed(run_dir: Path) -> Optional[int]:
    cfgp = run_dir / 'config.json'
    if cfgp.exists():
        try:
            cfg = json.loads(cfgp.read_text())
            seed = cfg.get('train', {}).get('seed')
            return int(seed) if seed is not None else None
        except Exception:
            pass
    m = re.search(r'seed(\d+)', run_dir.name)
    return int(m.group(1)) if m else None


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('metrics.jsonl')):
        yield p.parent


def main() -> None:
    p = argparse.ArgumentParser(description='Select successful and failed HOP_2 runs for intermediate/answer probe analysis.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out-txt', default='runs/behavioral_replication_v0_9/intermediate_answer_probe/probe_selected_runs.txt')
    p.add_argument('--out-jsonl', default=None)
    p.add_argument('--success-arms', default=DEFAULT_SUCCESS_ARMS)
    p.add_argument('--failed-arms', default=DEFAULT_FAILED_ARMS)
    p.add_argument('--success-threshold', type=float, default=0.95)
    p.add_argument('--success-min-excess', type=float, default=0.50)
    p.add_argument('--failed-max-acc', type=float, default=0.25)
    p.add_argument('--failed-max-excess', type=float, default=0.12)
    p.add_argument('--require-model', action='store_true')
    p.add_argument('--max-success', default='all', help='Maximum selected successful runs, or all')
    p.add_argument('--max-failed', default='all', help='Maximum selected failed runs, or all')
    args = p.parse_args()

    root = Path(args.runs_dir)
    success_arms = {a.strip() for a in args.success_arms.split(',') if a.strip()}
    failed_arms = {a.strip() for a in args.failed_arms.split(',') if a.strip()}
    max_success = None if args.max_success == 'all' else int(args.max_success)
    max_failed = None if args.max_failed == 'all' else int(args.max_failed)

    selected_success: List[Dict[str, Any]] = []
    selected_failed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for rd in iter_run_dirs(root):
        arm = derive_arm(rd)
        if arm not in success_arms and arm not in failed_arms:
            continue
        if args.require_model and not (rd / 'model_final.pt').exists():
            skipped.append({'run_dir': str(rd), 'arm': arm, 'reason': 'missing_model_final'})
            continue
        try:
            rows = read_jsonl(rd / 'metrics.jsonl')
        except Exception as exc:  # noqa: BLE001
            skipped.append({'run_dir': str(rd), 'arm': arm, 'reason': f'read_error:{exc}'})
            continue
        for r in rows:
            a = numeric(r.get('hop2_acc'))
            f = numeric(r.get('floor_hop2_acc'))
            if a is not None and f is not None:
                r['hop2_excess'] = a - f
        tail_h2 = tail_mean(rows, 'hop2_acc')
        tail_excess = tail_mean(rows, 'hop2_excess')
        rec = {
            'run_dir': str(rd),
            'arm': arm,
            'seed': load_seed(rd),
            'tail_hop2_acc': tail_h2,
            'tail_hop2_excess': tail_excess,
        }
        if arm in success_arms and tail_h2 is not None and tail_excess is not None and tail_h2 >= args.success_threshold and tail_excess >= args.success_min_excess:
            rec['probe_group'] = 'success'
            selected_success.append(rec)
        elif arm in failed_arms and tail_h2 is not None and tail_excess is not None and tail_h2 <= args.failed_max_acc and tail_excess <= args.failed_max_excess:
            rec['probe_group'] = 'failed'
            selected_failed.append(rec)
        else:
            rec['reason'] = 'threshold_not_met'
            skipped.append(rec)

    selected_success = sorted(selected_success, key=lambda x: (str(x.get('arm')), int(x.get('seed') or -1)))
    selected_failed = sorted(selected_failed, key=lambda x: (str(x.get('arm')), int(x.get('seed') or -1)))
    if max_success is not None:
        selected_success = selected_success[:max_success]
    if max_failed is not None:
        selected_failed = selected_failed[:max_failed]
    selected = selected_success + selected_failed

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text('\n'.join(r['run_dir'] for r in selected) + ('\n' if selected else ''), encoding='utf-8')
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_txt.with_suffix('.jsonl')
    with out_jsonl.open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    meta = out_txt.with_name(out_txt.stem + '_manifest.json')
    meta.write_text(json.dumps({'selected': selected, 'skipped': skipped}, indent=2, sort_keys=True), encoding='utf-8')

    print(f'selected {len(selected_success)} successful and {len(selected_failed)} failed runs')
    for r in selected:
        print(f"  {r['probe_group']:7s} seed={r.get('seed')} arm={r['arm']} tail_hop2={r.get('tail_hop2_acc')} excess={r.get('tail_hop2_excess')} {r['run_dir']}")
    print(f'wrote {out_txt}')
    print(f'wrote {out_jsonl}')
    print(f'wrote {meta}')


if __name__ == '__main__':
    main()
