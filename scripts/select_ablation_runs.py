#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ARM_NAMES = {
    's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
    'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2',
    'query_b_s1', 'query_b_s2',
}
DEFAULT_ARMS = 's1_longcos_late,s2_constant_late,rewarm_late,rewarm_reset_late'


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
    if run_dir.parent.name in ARM_NAMES:
        return run_dir.parent.name
    return run_dir.name


def load_seed(run_dir: Path) -> Optional[int]:
    cfgp = run_dir / 'config.json'
    if not cfgp.exists():
        return None
    try:
        cfg = json.loads(cfgp.read_text())
        seed = cfg.get('train', {}).get('seed')
        return int(seed) if seed is not None else None
    except Exception:
        return None


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob('metrics.jsonl')):
        yield p.parent


def main() -> None:
    p = argparse.ArgumentParser(description='Select successful HOP_2 runs for key-slot ablation evaluation.')
    p.add_argument('--runs-dir', default='runs/behavioral_replication_v0_9')
    p.add_argument('--out', default='runs/behavioral_replication_v0_9/ablation_selected_runs.txt')
    p.add_argument('--metadata-out', default=None)
    p.add_argument('--arms', default=DEFAULT_ARMS, help='Comma-separated arm names to consider')
    p.add_argument('--success-threshold', type=float, default=0.95)
    p.add_argument('--min-excess', type=float, default=0.50, help='Require tail HOP_2 excess >= this value')
    p.add_argument('--require-model', action='store_true', help='Skip runs without model_final.pt')
    args = p.parse_args()

    root = Path(args.runs_dir)
    arms = {a.strip() for a in args.arms.split(',') if a.strip()}
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for rd in iter_run_dirs(root):
        arm = derive_arm(rd)
        if arms and arm not in arms:
            continue
        if args.require_model and not (rd / 'model_final.pt').exists():
            skipped.append({'run_dir': str(rd), 'arm': arm, 'reason': 'missing_model_final'})
            continue
        try:
            rows = read_jsonl(rd / 'metrics.jsonl')
        except Exception as exc:
            skipped.append({'run_dir': str(rd), 'arm': arm, 'reason': f'read_error:{exc}'})
            continue
        for r in rows:
            a = numeric(r.get('hop2_acc'))
            f = numeric(r.get('floor_hop2_acc'))
            if a is not None and f is not None:
                r['hop2_excess'] = a - f
        tail_h2 = tail_mean(rows, 'hop2_acc')
        tail_excess = tail_mean(rows, 'hop2_excess')
        ok = (tail_h2 is not None and tail_h2 >= args.success_threshold and
              tail_excess is not None and tail_excess >= args.min_excess)
        rec = {
            'run_dir': str(rd),
            'arm': arm,
            'seed': load_seed(rd),
            'tail_hop2_acc': tail_h2,
            'tail_hop2_excess': tail_excess,
            'selected': ok,
        }
        if ok:
            selected.append(rec)
        else:
            rec['reason'] = 'below_success_threshold'
            skipped.append(rec)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('\n'.join(r['run_dir'] for r in selected) + ('\n' if selected else ''), encoding='utf-8')

    meta_out = Path(args.metadata_out) if args.metadata_out else outp.with_suffix('.json')
    meta_out.write_text(json.dumps({'selected': selected, 'skipped': skipped}, indent=2, sort_keys=True), encoding='utf-8')

    print(f'selected {len(selected)} runs for ablation')
    for r in selected:
        print(f"  seed={r.get('seed')} arm={r['arm']} tail_hop2={r['tail_hop2_acc']:.4f} excess={r['tail_hop2_excess']:.4f} {r['run_dir']}")
    print(f'wrote {outp}')
    print(f'wrote {meta_out}')


if __name__ == '__main__':
    main()
